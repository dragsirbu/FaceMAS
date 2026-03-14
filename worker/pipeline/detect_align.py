import sys
import os
import numpy as np
import cv2
from PIL import Image

# Make AdaFace MTCNN importable
sys.path.insert(0, "/workspace/AdaFace/face_alignment")
from mtcnn import MTCNN

T112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def _tpl(sz: int) -> np.ndarray:
    return T112 * (sz / 112.0)

_mtcnn = MTCNN(device="cuda:0", crop_size=(112, 112))

def _pick_largest(boxes):
    """Return (box, index) for the largest bounding box, or (None, -1)."""
    if len(boxes) == 0:
        return None, -1
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    return boxes[idx], idx

def align(bgr: np.ndarray, kps5: np.ndarray, out_size: int) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), _tpl(out_size), method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("align: estimateAffinePartial2D failed")
    return cv2.warpAffine(bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderValue=0)

def detect_and_align(bgr: np.ndarray) -> dict:
    # MTCNN expects a PIL RGB image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    try:
        boxes, landmarks = _mtcnn.detect_faces(
            pil_img, _mtcnn.min_face_size, _mtcnn.thresholds,
            _mtcnn.nms_thresholds, _mtcnn.factor,
        )
    except Exception:
        return {"face_found": False, "multiple_faces": False}

    if len(boxes) == 0 or len(landmarks) == 0:
        return {"face_found": False, "multiple_faces": False}

    box, idx = _pick_largest(boxes)
    if box is None:
        return {"face_found": False, "multiple_faces": False}

    # MTCNN landmark format: [x1..x5, y1..y5] → [[x1,y1], ..., [x5,y5]]
    lm = landmarks[idx]
    kps = np.array([[lm[j], lm[j + 5]] for j in range(5)], dtype=np.float64)

    return {
        "face_found": True,
        "multiple_faces": len(boxes) > 1,
        "bbox": box[:4].astype(float).tolist(),
        "landmarks_5": kps.tolist(),
        "aligned_512": align(bgr, kps, 512),
        "aligned_256": align(bgr, kps, 256),
        "aligned_112": align(bgr, kps, 112),
    }