import io
import uuid
import asyncio
import threading
import shutil
import cv2

from pathlib import Path
from typing import List, Dict, Any


from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from worker.config import MAX_IMAGES
from worker.pipeline.io import pil_to_bgr, bgr_to_jpeg_b64
from worker.pipeline.detect_align import detect_and_align
from worker.pipeline.ofiq import score_dir
from worker.pipeline.selfage import train, age_edit, collect_new_images
from worker.pipeline.face_search import search as face_search

app = FastAPI()

JOBS: Dict[str, Dict[str, Any]] = {}
SELFAGE_LOCK = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_facepipe_dirs() -> tuple[str, Path, Path, Path]:
    req_id = uuid.uuid4().hex[:12]
    workdir = Path(f"/tmp/facepipe_{req_id}")
    self_ref = workdir / "self_ref"
    selfage_exp = workdir / "selfage_exp"
    test_dir = workdir / "test"

    self_ref.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    selfage_exp.mkdir(parents=True, exist_ok=True)
    return req_id, self_ref, test_dir, selfage_exp

def _run_selfage_job(job_id: str, self_ref: Path, test_dir: Path, exp_dir: Path, results: list, target_age: int = 50, gender: str = "", attributes: str = ""):
    try:
        with SELFAGE_LOCK:  # one SelfAge at a time
            JOBS[job_id]["status"] = "scoring"

            # OFIQ quality scoring (moved here to avoid blocking the HTTP response)
            try:
                scores = score_dir(self_ref)
                test_scores = score_dir(test_dir)
                scores.update(test_scores)
                for r in results:
                    d = r.pop("_disk", None)
                    if d and d in scores:
                        r["ofiq_uqs"] = float(scores[d])
            except Exception as e:
                print(f"OFIQ scoring failed (non-fatal): {e}")
                for r in results:
                    r.pop("_disk", None)

            # Publish results (with OFIQ scores) before training starts
            JOBS[job_id]["results"] = results

            JOBS[job_id]["status"] = "training"
            train(self_ref, exp_dir)

            JOBS[job_id]["status"] = "editing"
            t0, _ = age_edit(data_dir=test_dir, exp_dir=exp_dir, target_ages=[target_age], gender=gender, attributes=attributes)

            aged = []
            aged_bgrs = {}  # filename -> bgr for face search
            for p in collect_new_images(exp_dir, since_ts=t0, limit=30):
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                aged.append({"filename": p.name, "mime": "image/jpeg", "base64": bgr_to_jpeg_b64(bgr)})
                aged_bgrs[p.name] = bgr

            # Face search: top 5 AgeDB matches for original test + each aged output
            JOBS[job_id]["status"] = "searching"
            test_matches = []
            aged_matches = {}
            search_errors = []
            try:
                test_imgs = list(test_dir.glob("*.jpg"))
                if test_imgs:
                    test_bgr = cv2.imread(str(test_imgs[0]))
                    if test_bgr is not None:
                        test_matches = face_search(test_bgr, top_k=5)
            except Exception as e:
                search_errors.append(f"test search: {e}")
                print(f"Face search (test) failed: {e}")
            for fname, abgr in aged_bgrs.items():
                try:
                    aged_matches[fname] = face_search(abgr, top_k=5)
                except Exception as e:
                    search_errors.append(f"{fname}: {e}")
                    print(f"Face search ({fname}) failed: {e}")

            update = {
                "status": "done", "aged": aged, "results": results,
                "test_matches": test_matches, "aged_matches": aged_matches,
                "_test_dir": str(test_dir), "_exp_dir": str(exp_dir),
            }
            if search_errors:
                update["search_errors"] = search_errors
            JOBS[job_id].update(update)
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})


def _run_reedit_job(job_id: str, test_dir: Path, exp_dir: Path, target_age: int, gender: str, attributes: str):
    try:
        with SELFAGE_LOCK:
            JOBS[job_id]["status"] = "editing"
            JOBS[job_id].pop("aged", None)

            t0, _ = age_edit(data_dir=test_dir, exp_dir=exp_dir, target_ages=[target_age], gender=gender, attributes=attributes)

            aged = []
            aged_bgrs = {}
            for p in collect_new_images(exp_dir, since_ts=t0, limit=30):
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                aged.append({"filename": p.name, "mime": "image/jpeg", "base64": bgr_to_jpeg_b64(bgr)})
                aged_bgrs[p.name] = bgr

            # Face search: top 5 AgeDB matches for original test + each aged output
            JOBS[job_id]["status"] = "searching"
            test_matches = []
            aged_matches = {}
            search_errors = []
            try:
                test_imgs = list(test_dir.glob("*.jpg"))
                if test_imgs:
                    test_bgr = cv2.imread(str(test_imgs[0]))
                    if test_bgr is not None:
                        test_matches = face_search(test_bgr, top_k=5)
            except Exception as e:
                search_errors.append(f"test search: {e}")
                print(f"Face search (test) failed: {e}")
            for fname, abgr in aged_bgrs.items():
                try:
                    aged_matches[fname] = face_search(abgr, top_k=5)
                except Exception as e:
                    search_errors.append(f"{fname}: {e}")
                    print(f"Face search ({fname}) failed: {e}")

            update = {"status": "done", "aged": aged, "test_matches": test_matches, "aged_matches": aged_matches}
            if search_errors:
                update["search_errors"] = search_errors
            JOBS[job_id].update(update)
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})



@app.post("/v1/quality-assess")
async def quality_assess(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(400, "No images uploaded")
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"Too many images (max {MAX_IMAGES})")

    req_id, self_ref, test_dir, _ = make_facepipe_dirs()

    results = []
    for i, uf in enumerate(images):
        raw = await uf.read()
        bgr = pil_to_bgr(Image.open(io.BytesIO(raw)))

        det = detect_and_align(bgr)
        item = {
            "filename": uf.filename,
            "face_found": det.get("face_found", False),
            "multiple_faces": det.get("multiple_faces", False),
            "bbox": det.get("bbox"),
            "landmarks_5": det.get("landmarks_5"),
            "aligned_256": None,
            "aligned_112": None,
            "ofiq_uqs": None,
        }

        if not item["face_found"]:
            results.append(item)
            continue

        a256 = det["aligned_256"]
        a112 = det["aligned_112"]

        disk_name = f"img_{i:02d}.jpg"
        cv2.imwrite(str(self_ref / disk_name), a256)
        if i == len(images) - 1:
            cv2.imwrite(str(test_dir / disk_name), a256)

        item["aligned_256"] = {"mime": "image/jpeg", "base64": bgr_to_jpeg_b64(a256)}
        item["aligned_112"] = {"mime": "image/jpeg", "base64": bgr_to_jpeg_b64(a112)}
        item["_disk"] = disk_name
        results.append(item)

    scores = score_dir(self_ref)
    print("OFIQ scores keys:", list(scores.keys())[:10])
    for r in results:
        d = r.pop("_disk", None)
        if d and d in scores:
            r["ofiq_uqs"] = float(scores[d])
            
    return JSONResponse({"ok": True, "req_id": req_id, "results": results})


@app.post("/v1/pipeline")
async def pipeline(images: List[UploadFile] = File(...), target_age: int = Form(50), gender: str = Form(""), attributes: str = Form("")):
    if not images:
        raise HTTPException(400, "No images uploaded")
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"Too many images (max {MAX_IMAGES})")
    if len(images) < 3:
        raise HTTPException(400, "Need at least 3 images for SelfAge training")

    req_id, self_ref, test_dir, selfage_exp = make_facepipe_dirs()

    results = []
    for i, uf in enumerate(images):
        raw = await uf.read()
        bgr = pil_to_bgr(Image.open(io.BytesIO(raw)))

        det = detect_and_align(bgr)
        item = {
            "filename": uf.filename,
            "face_found": det.get("face_found", False),
            "multiple_faces": det.get("multiple_faces", False),
            "bbox": det.get("bbox"),
            "landmarks_5": det.get("landmarks_5"),
            "aligned_256": None,
            "aligned_112": None,
            "ofiq_uqs": None,
        }

        if not item["face_found"]:
            results.append(item)
            continue

        a256 = det["aligned_256"]
        a112 = det["aligned_112"]

        disk_name = f"img_{i:02d}.jpg"
        if i == len(images) - 1:
            cv2.imwrite(str(test_dir / disk_name), a256)
        else:
            cv2.imwrite(str(self_ref / disk_name), a256)

        item["aligned_256"] = {"mime": "image/jpeg", "base64": bgr_to_jpeg_b64(a256)}
        item["aligned_112"] = {"mime": "image/jpeg", "base64": bgr_to_jpeg_b64(a112)}
        item["_disk"] = disk_name
        results.append(item)

    if not any(self_ref.glob("*.jpg")):
        raise HTTPException(400, "No faces detected in any uploaded image")

    # Start async job (avoid 524 timeout)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "req_id": req_id}

    t = threading.Thread(
        target=_run_selfage_job,
        args=(job_id, self_ref, test_dir, selfage_exp, results, target_age, gender, attributes),
        daemon=True,
    )
    t.start()

    return JSONResponse({"ok": True, "req_id": req_id, "job_id": job_id, "status": "queued"}, status_code=202)

@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job_id")
    # Don't expose internal paths to the client
    out = {k: v for k, v in j.items() if not k.startswith("_")}
    return out


@app.post("/v1/jobs/{job_id}/re-edit")
async def re_edit(job_id: str, target_age: int = Form(50), gender: str = Form(""), attributes: str = Form("")):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job_id")
    if j.get("status") not in ("done", "error"):
        raise HTTPException(409, "Job is still running")
    test_dir = j.get("_test_dir")
    exp_dir = j.get("_exp_dir")
    if not test_dir or not exp_dir:
        raise HTTPException(400, "No trained weights available for this job")

    JOBS[job_id]["status"] = "queued"

    t = threading.Thread(
        target=_run_reedit_job,
        args=(job_id, Path(test_dir), Path(exp_dir), target_age, gender, attributes),
        daemon=True,
    )
    t.start()

    return JSONResponse({"ok": True, "job_id": job_id, "status": "queued"}, status_code=202)
