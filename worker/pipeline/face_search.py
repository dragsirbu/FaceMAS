import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # FaceMAS/
sys.path.insert(0, str(PROJECT_ROOT / "AdaFace"))
import net

from worker.pipeline.detect_align import detect_and_align
from worker.pipeline.io import bgr_to_jpeg_b64

AGEDB_DIR = PROJECT_ROOT / "data" / "AgeDB"
ADAFACE_CKPT = PROJECT_ROOT / "AdaFace" / "pretrained" / "adaface_ir50_ms1mv2.ckpt"
INDEX_PATH = PROJECT_ROOT / "data" / "agedb_index.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy singletons ──────────────────────────────────────────────
_model = None
_index_filenames = None
_index_embeddings = None


def _get_model():
    global _model
    if _model is None:
        m = net.build_model("ir_50")
        sd = torch.load(str(ADAFACE_CKPT), map_location="cpu", weights_only=False)["state_dict"]
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
        m.eval().to(DEVICE)
        _model = m
    return _model


def _get_index():
    global _index_filenames, _index_embeddings
    if _index_filenames is None:
        data = np.load(str(INDEX_PATH))
        _index_filenames = data["filenames"]
        _index_embeddings = data["embeddings"]  # (N, 512), L2-normalised
    return _index_filenames, _index_embeddings


def _parse_agedb_filename(fname: str) -> tuple:
    """Parse '10000_GoldieHawn_62_f.jpg' → ('Goldie Hawn', 62)"""
    stem = Path(fname).stem                     # '10000_GoldieHawn_62_f'
    parts = stem.split("_")                     # ['10000', 'GoldieHawn', '62', 'f']
    if len(parts) < 3:
        return (stem, None)
    # name is parts[1], age is parts[2] - name uses CamelCase, split on uppercase
    raw_name = parts[1]
    # Insert spaces before uppercase letters: 'GoldieHawn' → 'Goldie Hawn'
    import re
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw_name)
    try:
        age = int(parts[2])
    except ValueError:
        age = None
    return (name, age)


def _bgr112_to_tensor(bgr_112: np.ndarray) -> torch.Tensor:
    t = ((bgr_112[:, :, ::-1].astype(np.float32) / 255.0) - 0.5) / 0.5
    return torch.from_numpy(t.transpose(2, 0, 1)).float()


def embed_aligned_112(bgr_112: np.ndarray) -> np.ndarray:
    """Return L2-normalised 512-d embedding for an already-aligned 112×112 BGR image."""
    model = _get_model()
    t = _bgr112_to_tensor(bgr_112).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat, _ = model(t)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy().flatten()


def search(bgr: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Detect, align, embed a BGR query image, then return top-k AgeDB matches.
    Each result: {filename, name, age, similarity, base64}.
    """
    det = detect_and_align(bgr)
    if not det.get("face_found"):
        return []

    query_emb = embed_aligned_112(det["aligned_112"])
    filenames, embeddings = _get_index()

    # cosine similarity (both are L2-normalised → dot product)
    sims = embeddings @ query_emb  # (N,)
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        fname = str(filenames[idx])
        name, age = _parse_agedb_filename(fname)

        # Load the AgeDB thumbnail
        thumb_path = AGEDB_DIR / fname
        b64 = ""
        if thumb_path.exists():
            thumb_bgr = cv2.imread(str(thumb_path))
            if thumb_bgr is not None:
                b64 = bgr_to_jpeg_b64(thumb_bgr)

        results.append({
            "filename": fname,
            "name": name,
            "age": age,
            "similarity": round(float(sims[idx]), 4),
            "mime": "image/jpeg",
            "base64": b64,
        })

    return results
