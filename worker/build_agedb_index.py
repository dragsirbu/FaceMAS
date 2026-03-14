"""
Pre-compute AdaFace IR-50 embeddings for every image in AgeDB.
Outputs: /workspace/data/agedb_index.npz
  - filenames: (N,) array of filename strings
  - embeddings: (N, 512) float32 L2-normalised embeddings

Usage:
    python -m worker.build_agedb_index
"""

import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Make AdaFace importable
sys.path.insert(0, "/workspace/AdaFace")
sys.path.insert(0, "/workspace/AdaFace/face_alignment")
import net
from mtcnn import MTCNN

AGEDB_DIR = Path("/workspace/data/AgeDB")
ADAFACE_CKPT = Path("/workspace/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt")
OUT_PATH = Path("/workspace/data/agedb_index.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# ArcFace 112 template
T112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def load_adaface():
    model = net.build_model("ir_50")
    sd = torch.load(str(ADAFACE_CKPT), map_location="cpu")["state_dict"]
    model_sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(model_sd)
    model.eval().to(DEVICE)
    return model

def bgr_to_tensor(bgr_112: np.ndarray) -> torch.Tensor:
    """Convert aligned 512x512 BGR to AdaFace input tensor."""
    t = ((bgr_112[:, :, ::-1].astype(np.float32) / 255.0) - 0.5) / 0.5
    return torch.from_numpy(t.transpose(2, 0, 1)).float()

def align_face(bgr: np.ndarray, mtcnn_model: MTCNN) -> np.ndarray | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    try:
        boxes, landmarks = mtcnn_model.detect_faces(
            pil_img, mtcnn_model.min_face_size, mtcnn_model.thresholds,
            mtcnn_model.nms_thresholds, mtcnn_model.factor,
        )
    except Exception:
        return None
    if len(boxes) == 0 or len(landmarks) == 0:
        return None
    # pick largest face
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    lm = landmarks[idx]
    kps = np.array([[lm[j], lm[j + 5]] for j in range(5)], dtype=np.float32)
    # Align to 512x512 instead of 112x112
    T512 = T112 * (512.0 / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps, T512, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, (512, 512), flags=cv2.INTER_LINEAR, borderValue=0)

def main():
    print("Loading MTCNN face detector...")
    mtcnn_model = MTCNN(device="cuda:0" if torch.cuda.is_available() else "cpu", crop_size=(512, 512))

    print("Loading AdaFace IR-50...")
    model = load_adaface()

    image_files = sorted([f for f in AGEDB_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Found {len(image_files)} images in AgeDB")

    filenames = []
    tensors = []
    skipped = 0

    for img_path in tqdm(image_files, desc="Aligning"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue
        aligned = align_face(bgr, mtcnn_model)
        if aligned is None:
            skipped += 1
            continue
        filenames.append(img_path.name)
        tensors.append(bgr_to_tensor(aligned))

    print(f"Aligned: {len(tensors)}, Skipped: {skipped}")

    # Batch embed
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tensors), BATCH_SIZE), desc="Embedding"):
            batch = torch.stack(tensors[i:i + BATCH_SIZE]).to(DEVICE)
            features, _ = model(batch)
            # L2 normalise
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    filenames_arr = np.array(filenames)

    np.savez(str(OUT_PATH), filenames=filenames_arr, embeddings=embeddings)
    print(f"Saved {OUT_PATH} - {embeddings.shape[0]} embeddings × {embeddings.shape[1]}d")

if __name__ == "__main__":
    main()
