# FaceMAS - Facial Manipulation as a Service

## Architecture Overview

```
┌─────────────────────┐        HTTP / JSON        ┌────────────────────────┐
│   Next.js Frontend  │ ◄──────────────────────►   │   FastAPI Backend       │
│   (face-mas/)       │                            │   (worker/)             │
│                     │                            │                        │
│  page.tsx           │   POST /v1/pipeline        │  main.py  (endpoints)  │
│  lib/client.ts      │   POST /v1/quality-assess  │  config.py             │
│                     │   GET  /v1/jobs/{id}       │  pipeline/             │
│                     │   POST /v1/jobs/{id}/re-edit│   detect_align.py     │
└─────────────────────┘                            │   ofiq.py              │
                                                   │   selfage.py           │
                                                   │   io.py                │
                                                   │   face_search.py       │
                                                   └───────┬────────────────┘
                                                           │
                                       ┌───────────────────┼──────────────────┐
                                       ▼                   ▼                  ▼
                                    MTCNN            OFIQ Binary        SelfAge
                                  (detection +       (C++ / UQS)       (conda env)
                                   alignment)                          SD 1.5 + LoRA
                                                                       + P2P editing
```

---

## Pipeline Flow

### Full Pipeline (`POST /v1/pipeline`)

```
User uploads 3-5 reference photos + 1 test photo
                      │
                      ▼
             ┌────────────────┐
             │  Detect & Align │   MTCNN (3-stage cascade)
             │  (per image)    │   → 5-point landmarks
             │                 │   → ArcFace 512×512 aligned crops (previously 256×256/112×112)
             └───────┬────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Images 0..n-2         Image n-1 (last)
    → self_ref/             → test/
    (reference dir)         (test dir)
          │                     │
          ▼                     ▼
     ┌──────────────────────────────┐
     │   HTTP 202 returned          │   Job ID assigned, async processing begins
     │   status: "queued"           │
     └──────────┬───────────────────┘
                │  (background thread)
                ▼
     ┌──────────────────┐
     │  OFIQ Scoring     │   status: "scoring"
     │  self_ref/ + test/ │   Unified Quality Score per image
     │                    │   Results published to job state → pollable immediately
     └────────┬───────────┘
              ▼
     ┌──────────────────┐
     │  SelfAge Train    │   status: "training"
     │  LoRA fine-tune   │   800 steps, rank 16, resolution 512
     │  on self_ref/     │   Output: exp_dir/pytorch_lora_weights.safetensors
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │  SelfAge Edit     │   status: "editing"
     │  P2P age editing  │   DDIM null-text inversion + Prompt-to-Prompt
     │  on test/         │   Supports target_age, gender, attributes
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │  AgeDB Search     │   Embed test + aged images via AdaFace
     │  Top-5 cosine sim │   Return 5 nearest AgeDB matches for each
     └────────┬──────────┘
              ▼
         status: "done"
         Response includes: results (QA), aged (base64 images), matches
```

### Re-Edit (`POST /v1/jobs/{id}/re-edit`)

Skips OFIQ scoring and LoRA training. Reuses the previously trained weights to re-run only the editing step with new parameters (target age, gender, attributes).

```
Re-edit request (target_age, gender, attributes)
              │
              ▼
     ┌──────────────────┐
     │  SelfAge Edit     │   status: "editing"
     │  (reuses LoRA)    │   Same test_dir and exp_dir from original job
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │  AgeDB Search     │
     └────────┬──────────┘
              ▼
         status: "done"
```

---

## API Reference

### `POST /v1/quality-assess`

Synchronous quality assessment. Detects faces, aligns, and runs OFIQ scoring.

| Parameter | Type         | Description                    |
|-----------|--------------|--------------------------------|
| `images`  | File[] (multipart) | 1-10 image files          |

**Response** `200 OK`:
```json
{
  "ok": true,
  "req_id": "abc123...",
  "results": [
    {
      "filename": "photo1.jpg",
      "face_found": true,
      "multiple_faces": false,
      "bbox": [x1, y1, x2, y2],
      "landmarks_5": [[x,y], ...],
      "aligned_256": { "mime": "image/jpeg", "base64": "..." },
      "aligned_112": { "mime": "image/jpeg", "base64": "..." },
      "ofiq_uqs": 72.3
    }
  ]
}
```

---

### `POST /v1/pipeline`

Starts the full async pipeline (detect → OFIQ → train → edit). Returns immediately with a job ID.

| Parameter    | Type         | Default | Description                              |
|-------------|--------------|---------|------------------------------------------|
| `images`    | File[] (multipart) | -   | 3-10 images; last = test, rest = reference |
| `target_age` | int (form)  | 50      | Target age for age editing (1-100)       |
| `gender`    | string (form) | ""     | `"male"` or `"female"` (empty = auto)   |
| `attributes` | string (form) | ""    | Extra P2P prompt tokens, e.g. `"smiling, wearing glasses"` |

**Response** `202 Accepted`:
```json
{
  "ok": true,
  "req_id": "abc123...",
  "job_id": "def456...",
  "status": "queued"
}
```

---

### `GET /v1/jobs/{job_id}`

Poll job status. Internal fields (prefixed `_`) are filtered out.

**Response** (during processing):
```json
{
  "status": "scoring | training | editing | searching",
  "req_id": "...",
  "results": [ ... ]       // populated after scoring completes
}
```

**Response** (on completion):
```json
{
  "status": "done",
  "results": [ ... ],
  "aged": [
    { "filename": "img_00.jpg", "mime": "image/jpeg", "base64": "..." }
  ],
  "test_matches": [
    { "filename": "10000_GoldieHawn_62_f.jpg", "name": "Goldie Hawn", "age": 62, "similarity": 0.87, "base64": "..." }
  ],
  "aged_matches": {
    "img_00.jpg": [
      { "filename": "1000_StephenHawking_1_m.jpg", "name": "Stephen Hawking", "age": 1, "similarity": 0.83, "base64": "..." }
    ]
  }
}
```

**Response** (on error):
```json
{
  "status": "error",
  "error": "description of what failed"
}
```

---

### `POST /v1/jobs/{job_id}/re-edit`

Re-run editing with new parameters. Only available when the job status is `"done"` or `"error"`.

| Parameter    | Type         | Default | Description                     |
|-------------|--------------|---------|--------------------------------------|
| `target_age` | int (form)  | 50      | New target age                       |
| `gender`    | string (form) | ""     | New gender override                  |
| `attributes` | string (form) | ""    | New attribute prompt tokens          |

**Response** `202 Accepted`:
```json
{ "ok": true, "job_id": "def456...", "status": "queued" }
```

Then poll via `GET /v1/jobs/{job_id}` as usual.

---

## Backend Modules

### `worker/config.py`

All tuneable parameters, overridable via environment variables:

| Variable               | Default                | Description                  |
|------------------------|------------------------|------------------------------|
| `OFIQ_BIN`             | OFIQ install path      | Path to OFIQ binary          |
| `OFIQ_CONFIG`          | `ofiq_config_uqs.jaxn` | OFIQ config file             |
| `SELFAGE_REPO`         | `/workspace/SelfAge`   | SelfAge repository root      |
| `SELFAGE_REG_DIR`      | `.../CelebA_regularization_dex` | Regularisation images |
| `SELFAGE_INSTANCE_PROMPT` | `"photo of sks person"` | DreamBooth instance prompt |
| `SELFAGE_RESOLUTION`   | 512                    | Training resolution          |
| `SELFAGE_MAX_TRAIN_STEPS` | 800                 | LoRA fine-tuning steps       |
| `SELFAGE_RANK`         | 16                     | LoRA rank                    |
| `UQS_GOOD`             | 60                     | OFIQ score ≥ this → "OK"    |
| `UQS_WARN`             | 45                     | OFIQ score ≥ this → "Low"   |
| `MAX_IMAGES`           | 10                     | Max upload count             |

### `worker/pipeline/detect_align.py`

- **MTCNN** (PNet → RNet → ONet cascade) with CUDA, from AdaFace's `face_alignment/mtcnn.py`
- Uses an image pyramid with `min_face_size=20`, making it robust to small faces
- **`_pick_largest()`** - selects the face with the largest bounding-box area
- **`align()`** - ArcFace-template similarity transform via `cv2.estimateAffinePartial2D`
- **`detect_and_align()`** - returns `face_found`, `bbox`, `landmarks_5`, `aligned_256`, `aligned_112`

### `worker/pipeline/ofiq.py`

- **`score_dir(img_dir)`** - runs the OFIQ C++ binary on a directory of aligned images
- Parses the semicolon-delimited CSV output for `UnifiedQualityScore.scalar`
- Returns `Dict[filename, float]`

### `worker/pipeline/selfage.py`

- **`train(self_ref_dir, exp_dir)`** - launches `accelerate launch scripts/train.py` inside the `selfage` conda env
- **`age_edit(data_dir, exp_dir, target_ages, gender, attributes)`** - launches `python scripts/age_editing.py`
  - Uses `AttentionReplace` (same-length prompts) or `AttentionRefine` (different-length, e.g. when attributes are added)
- **`find_personalized_path(exp_dir)`** - searches for LoRA weight files (`.safetensors` / `.bin`)
- **`collect_new_images(root, since_ts)`** - finds images modified after a timestamp

### `worker/pipeline/face_search.py`

- **`_get_model()`** - lazy-loads AdaFace IR-50 (singleton, loaded once on first search)
- **`_get_index()`** - lazy-loads `agedb_index.npz` with 16,190 pre-computed 512-d embeddings
- **`_parse_agedb_filename()`** - parses `{id}_{Name}_{age}_{gender}.jpg` → `(name, age)`, splitting CamelCase names into words
- **`embed_aligned_112(bgr_112)`** - embeds an already-aligned 112×112 BGR image → L2-normalised 512-d vector
- **`search(bgr, top_k=5)`** - detects + aligns face, embeds it, computes cosine similarity against the full AgeDB index, returns top-k matches with name, age, similarity, and base64 thumbnail

### `worker/pipeline/io.py`

- **`pil_to_bgr()`** - PIL Image → BGR numpy array
- **`bgr_to_jpeg_b64()`** - BGR numpy → base64-encoded JPEG string

### `worker/main.py`

- **JOBS dict** - in-memory job store (not persistent across restarts)
- **SELFAGE_LOCK** - `threading.Lock` ensuring only one SelfAge job runs at a time
- **`_run_selfage_job()`** - full background worker: OFIQ → train → edit → face search → collect results
- **`_run_reedit_job()`** - lightweight background worker: edit → face search → collect results
- Internal keys (`_test_dir`, `_exp_dir`) stored in JOBS but filtered from API responses

---

## Frontend

### `face-mas/lib/client.ts`

TypeScript API client. All calls go to `NEXT_PUBLIC_WORKER_URL`.

| Function         | Method | Endpoint                        | Description              |
|------------------|--------|----------------------------------|-----------------------------|
| `qualityAssess`  | POST   | `/v1/quality-assess`             | Sync QA (no training)       |
| `startPipeline`  | POST   | `/v1/pipeline`                   | Start full async pipeline   |
| `reEdit`         | POST   | `/v1/jobs/{id}/re-edit`          | Re-edit with new params     |
| `getJob`         | GET    | `/v1/jobs/{id}`                  | Poll job status             |
| `b64ToDataUrl`   | -      | -                                | Convert base64 → data URI   |

`toFormData()` appends reference files first, test file last (backend relies on ordering).

### `face-mas/app/page.tsx`

Single-page React app with two-column layout:

**Left column - Input**:
- Drag-and-drop zones for reference (3-5) and test (1) photos
- Thumbnail previews with remove buttons
- Controls: target age (1-100), gender (Auto/Male/Female), attributes (free text)
- "Quality Assess" button (sync), "Clear All"

**Right column - Results**:
- Quality check grid: aligned thumbnails with OFIQ badge (OK ≥ 60, Low ≥ 40, Reject < 40)
- Aged outputs grid (populated when job completes)
- "Re-Edit" button to re-run editing with different parameters

**Bottom bar**:
- Readiness indicator, job ID + status display, "Add More" and "Run Pipeline" buttons

**Polling**: `pollJob()` polls every 1.5s, updates `qaResults` on each tick (so OFIQ cards appear before training finishes), and sets `aged` on status `"done"`.

---

## AgeDB Face Search

### Overview

After age editing completes, the system embeds both the **original test image** and each **aged output** using AdaFace IR-50, then searches a pre-computed AgeDB index to find the **top 5 most similar faces** by cosine similarity. For each match, the **identity name**, **age**, and **cosine similarity score** are displayed. This provides a visual identity-preservation check: if the aged output still matches the same people as the original, identity has been preserved.

### Index Pre-computation (`worker/build_agedb_index.py`)

Already implemented. Processes every image in `/workspace/data/AgeDB/`:

1. Detect + align each face to 112×112 using MTCNN
2. Embed with AdaFace IR-50 (batch size 64)
3. L2-normalise embeddings
4. Save to `/workspace/data/agedb_index.npz`:
   - `filenames`: `(N,)` string array
   - `embeddings`: `(N, 512)` float32 matrix

Run once: `python -m worker.build_agedb_index`

### Search Module (`worker/pipeline/face_search.py`)

```
face_search.py
├── load_index()        Load agedb_index.npz (cached singleton)
├── embed_image()       Detect, align 112×112, run AdaFace → 512-d vector
└── search(query_bgr)   Embed query → cosine sim against index → top 5
```

**Input**: BGR image (aligned or unaligned - module handles detection)

AgeDB filenames follow the pattern `name_age_0001.jpg`. The search module parses each filename to extract the identity name and age.

**Output**:
```python
[
    {"filename": "John_Smith_72_0001.jpg", "name": "John Smith", "age": 72, "similarity": 0.87, "base64": "..."},
    {"filename": "Jane_Doe_65_0042.jpg", "name": "Jane Doe", "age": 65, "similarity": 0.83, "base64": "..."},
    ...
]
```

### Integration Points

**Backend** - in `_run_selfage_job()` and `_run_reedit_job()`, after editing:

```
editing → collect aged images → face_search on test image → face_search on each aged image → done
```

Job result includes:
```json
{
  "status": "done",
  "aged": [ ... ],
  "results": [ ... ],
  "test_matches": [
    { "filename": "John_Smith_72_0001.jpg", "name": "John Smith", "age": 72, "similarity": 0.87, "base64": "..." }
  ],
  "aged_matches": {
    "img_00.jpg": [
      { "filename": "Jane_Doe_65_0042.jpg", "name": "Jane Doe", "age": 65, "similarity": 0.83, "base64": "..." }
    ]
  }
}
```

**Frontend** - section below aged outputs:

- "AgeDB Matches - Original" card row: top 5 thumbnails, each labelled with **name**, **age**, and **cosine similarity score**
- "AgeDB Matches - Aged" card row per aged output: top 5 thumbnails with name, age, and similarity
- Colour coding: high similarity (≥ 0.6) green, medium (≥ 0.4) amber, low (< 0.4) red

### AgeDB Filename Format

AgeDB filenames follow the pattern `{id}_{CamelCaseName}_{age}_{gender}.jpg`, e.g. `10000_GoldieHawn_62_f.jpg`. The search module splits CamelCase into display names (`Goldie Hawn`) and extracts the integer age.

### Purpose

The face search serves the thesis evaluation:

1. **Identity preservation metric**: If the original test and aged output match the same AgeDB identities, the aging transformation preserved identity.
2. **Visual ground truth**: Shows real people at various ages who look most similar, giving a qualitative sanity check.
3. **Cross-age retrieval**: Since AgeDB contains identities at multiple ages, the search naturally reveals whether the system's output looks like the same person aged versus a different person entirely.

---

## SelfAge Modifications for Attribute Editing

The upstream SelfAge repository only supports age-to-age editing via Prompt-to-Prompt (P2P). We modified three files to enable arbitrary attribute manipulation (e.g. "smiling", "wearing glasses", "receding hairline") alongside age editing.

### 1. `SelfAge/scripts/age_editing.py`

- **Added `--attributes` argument** to `parse_args()` (default: empty string)
- **Modified target prompt construction**: when attributes are provided, the target prompt becomes `"{age}-year-old {attributes}"` instead of just `"{age}-year-old"`. For example, with `--attributes="smiling, receding hairline"` and target age 65:
  - Source prompt: `"photo of sks man as 25-year-old"`
  - Target prompt: `"photo of sks man as 65-year-old smiling, receding hairline"`

### 2. `SelfAge/utils/inference_utils.py`

- **Dynamic controller selection**: changed `is_replace_controller` from hardcoded `True` to:
  ```python
  is_replace_controller = len(inversion_prompt.split()) == len(new_prompt.split())
  ```
  When the target prompt is longer (due to appended attributes), the system automatically uses `AttentionRefine` instead of `AttentionReplace`. `AttentionReplace` requires both prompts to have the same number of tokens and would crash with extended prompts. `AttentionRefine` handles different-length prompts by aligning shared tokens and refining attention on the new ones.

### 3. `SelfAge/utils/seq_aligner.py`

- **Fixed `mis_match_char` bug**: the `ScoreParams.mis_match_char()` method returned `self` (the ScoreParams object) instead of `self.match` (an integer score) on character match. This bug was latent because `AttentionReplace` (same-length prompts) never triggered the sequence alignment code path. Once `AttentionRefine` was enabled for different-length prompts, the alignment would crash with a type error. Fixed by returning `self.match` instead of `self`.

### How It Works End-to-End

1. The frontend sends an `attributes` string via the pipeline or re-edit endpoint
2. The backend passes `--attributes="..."` to `age_editing.py` via the CLI
3. `age_editing.py` appends the attributes to the P2P target prompt
4. `inference_utils.py` detects the prompt length mismatch and selects `AttentionRefine`
5. `AttentionRefine` uses `seq_aligner` to align shared tokens between source and target, then refines attention maps for the newly added attribute tokens
6. The diffusion model generates the aged + attribute-modified output

---

## Concurrency & Constraints

- **SELFAGE_LOCK**: Only one SelfAge job (train or edit) can run at a time. Subsequent requests queue behind the lock.
- **In-memory JOBS**: Job state is not persisted - a server restart loses all job history.
- **GPU requirement**: MTCNN, OFIQ, and SelfAge all require a CUDA-capable GPU.
- **Conda isolation**: SelfAge runs in its own conda environment (`selfage`) via `conda run -n selfage`, separate from the FastAPI process.

---

## Environment Setup

```bash
# Worker
cd /workspace/worker
pip install -r requirements.txt
uvicorn worker.main:app --host 0.0.0.0 --port 8000

# Frontend
cd /workspace/face-mas
echo "NEXT_PUBLIC_WORKER_URL=http://localhost:8000" > .env.local
npm install
npm run dev

# AgeDB index (one-time)
cd /workspace
python -m worker.build_agedb_index
```
