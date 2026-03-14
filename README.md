# FaceMAS — Facial Manipulation as a Service

A face aging pipeline that takes reference photos of a person, trains a personalized LoRA model, and generates aged versions of a test photo using diffusion-based editing. Includes quality assessment (OFIQ), face search against AgeDB, and a web UI.

## Pipeline Flow

1. **Upload**: User uploads 3–5 reference photos + 1 test photo via the web UI
2. **Detect & Align**: MTCNN detects faces, aligns to ArcFace template (512×512)
3. **Quality Score**: OFIQ binary scores each aligned face (Unified Quality Score)
4. **Train**: SelfAge fine-tunes a LoRA adapter on the reference photos (~200 steps)
5. **Age Edit**: Prompt-to-Prompt diffusion edits the test photo to a target age
6. **Face Search**: AdaFace embeds original + aged photos, finds top-5 AgeDB matches
7. **Results**: Web UI displays quality scores, aged outputs, and AgeDB matches

---

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on A40/A100, ~16 GB VRAM minimum)
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ (for the frontend)
- **Conda/Miniconda**: Required for the SelfAge environment
- **Disk space**: ~30 GB for all dependencies and model weights

---

## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/dragsirbu/FaceMAS.git
cd FaceMAS
```

### 2. Install external dependencies

The pipeline depends on three external projects that are **not** included in this repo due to their size. Clone them into the workspace root:

#### 2a. OFIQ (Open Face Image Quality)

OFIQ provides the C++ binary for face image quality assessment.

```bash
# Clone and build OFIQ
git clone https://github.com/BSI-OFIQ/OFIQ-Project.git
cd OFIQ-Project

# Build (requires CMake 3.16+, GCC 9+, and OpenCV dev libraries)
# See OFIQ-Project/README.md for full build instructions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
cd ../..

# Verify the binary exists:
ls OFIQ-Project/install_x86_64_linux/Release/bin/OFIQSampleApp
```

#### 2b. SelfAge (Personalized Age Editing)

SelfAge provides the LoRA training and Prompt-to-Prompt age editing.

```bash
# Clone the SelfAge repository
git clone https://github.com/jonyzhang2023/SelfAge.git
cd SelfAge

# Create the conda environment
conda create -n selfage python=3.10 -y
conda activate selfage

# Install SelfAge dependencies
pip install -r requirements.txt

# Download the regularization dataset (CelebA)
# Place it at: SelfAge/data/CelebA_regularization_dex/
# See SelfAge/README.md for download links

conda deactivate
cd ..
```

**Important SelfAge modifications**: This pipeline requires three modifications to the upstream SelfAge code for attribute editing support. Apply them manually:

1. **`SelfAge/scripts/age_editing.py`**: Add `--attributes` argument to `parse_args()`. When provided, append attributes to the target prompt (e.g., `"65-year-old smiling, receding hairline"`).

2. **`SelfAge/utils/inference_utils.py`**: Change `is_replace_controller` from hardcoded `True` to:
   ```python
   is_replace_controller = len(inversion_prompt.split()) == len(new_prompt.split())
   ```
   This enables `AttentionRefine` for different-length prompts when attributes are added.

3. **`SelfAge/utils/seq_aligner.py`**: In `ScoreParams.mis_match_char()`, return `self.match` instead of `self` to fix a type error triggered by `AttentionRefine`.

#### 2c. AdaFace (Face Recognition)

AdaFace provides face embeddings for the AgeDB face search feature.

```bash
# Clone AdaFace
git clone https://github.com/mk-minchul/AdaFace.git
cd AdaFace

# Download the pretrained checkpoint
mkdir -p pretrained
# Download adaface_ir50_ms1mv2.ckpt from the AdaFace releases
# Place it at: AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
# See AdaFace/README.md for download links

cd ..
```

#### 2d. AgeDB Dataset

The face search feature uses AgeDB as the reference database.

```bash
# Create data directory
mkdir -p data/AgeDB

# Download AgeDB dataset images and place them in data/AgeDB/
# Filenames follow the pattern: {id}_{CamelCaseName}_{age}_{gender}.jpg
# e.g., 10000_GoldieHawn_62_f.jpg
```

### 3. Install backend (worker) dependencies

```bash
# Install system-level Python dependencies
pip install -r requirements.txt

# Install worker-specific dependencies
pip install -r worker/requirements.txt
```

### 4. Build the AgeDB face search index (one-time)

```bash
python -m worker.build_agedb_index
# Creates: data/agedb_index.npz (~50 MB, 16K embeddings × 512 dimensions)
```

### 5. Install frontend dependencies

```bash
cd face-mas
npm install

# Create your environment file
cp .env.example .env.local
# Edit .env.local if the backend runs on a different host/port

cd ..
```

---

## Running the Pipeline

### Start the backend

```bash
# From the workspace root
uvicorn worker.main:app --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`.

### Start the frontend

```bash
cd face-mas
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### Using the pipeline

1. Open `http://localhost:3000` in your browser
2. Upload 3–5 **reference photos** (clear, front-facing photos of the same person)
3. Upload 1 **test photo** (the photo to be aged)
4. Set the **target age** (1–100), optionally set gender and attributes
5. Click **"Run Pipeline"** and wait for results (~2–5 minutes depending on GPU)
6. View quality scores, aged outputs, and AgeDB similarity matches
7. Use **"Re-Edit"** to try different ages/attributes without retraining

---

## Configuration

All backend settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OFIQ_BIN` | `OFIQ-Project/install_.../OFIQSampleApp` | Path to OFIQ binary |
| `OFIQ_CONFIG` | `OFIQ-Project/data/ofiq_config_uqs.jaxn` | OFIQ config file |
| `SELFAGE_REPO` | `./SelfAge` | SelfAge repository root |
| `SELFAGE_REG_DIR` | `SelfAge/data/CelebA_regularization_dex` | Regularization images directory |
| `SELFAGE_INSTANCE_PROMPT` | `"photo of sks person"` | DreamBooth instance prompt |
| `SELFAGE_RESOLUTION` | `512` | Training resolution |
| `SELFAGE_MAX_TRAIN_STEPS` | `200` | LoRA fine-tuning steps |
| `SELFAGE_RANK` | `16` | LoRA rank |
| `UQS_GOOD` | `60` | OFIQ score threshold for "OK" |
| `UQS_WARN` | `45` | OFIQ score threshold for "Low" |
| `MAX_IMAGES` | `10` | Maximum images per request |

Frontend configuration (in `face-mas/.env.local`):

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_WORKER_URL` | `http://localhost:8000` | Backend worker URL |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/quality-assess` | POST | Synchronous quality assessment (detect + align + OFIQ) |
| `/v1/pipeline` | POST | Start full async pipeline (returns job ID) |
| `/v1/jobs/{job_id}` | GET | Poll job status and results |
| `/v1/jobs/{job_id}/re-edit` | POST | Re-run age editing with new parameters |

See [`docs/FaceMAS_Pipeline.md`](docs/FaceMAS_Pipeline.md) for full API documentation including request/response schemas.

---

## Project Structure

```
FaceMAS/
├── worker/                    # FastAPI backend
│   ├── main.py                # API endpoints & job management
│   ├── config.py              # Environment-based configuration
│   ├── requirements.txt       # Python dependencies
│   ├── build_agedb_index.py   # One-time AgeDB index builder
│   └── pipeline/
│       ├── detect_align.py    # MTCNN face detection & alignment
│       ├── ofiq.py            # OFIQ quality scoring (subprocess)
│       ├── selfage.py         # SelfAge training & editing (conda)
│       ├── face_search.py     # AdaFace embeddings & AgeDB search
│       └── io.py              # Image I/O utilities
├── face-mas/                  # Next.js frontend
│   ├── app/
│   │   ├── page.tsx           # Main UI (upload, results, re-edit)
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Tailwind styles
│   ├── lib/
│   │   └── client.ts          # Worker API client
│   ├── package.json
│   └── .env.example           # Template for environment config
├── docs/
│   └── FaceMAS_Pipeline.md    # Detailed pipeline documentation
├── requirements.txt           # Root-level Python dependencies (PyTorch, etc.)
└── README.md
```

External dependencies (clone into root, not tracked in git):
```
├── OFIQ-Project/              # Face image quality assessment (C++)
├── SelfAge/                   # Age editing via diffusion models
├── AdaFace/                   # Face recognition embeddings
├── data/
│   ├── AgeDB/                 # Reference face database
│   └── agedb_index.npz        # Pre-computed embeddings index
└── conda/                     # Miniconda installation
```

---

## RunPod / Cloud GPU Setup

If running on RunPod or similar GPU cloud:

1. Choose a template with **CUDA 11.8+** and **Python 3.10/3.11** (e.g., `runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04`)
2. Ensure at least **16 GB VRAM** (A40, A100, or RTX 4090)
3. Expose **port 8000** (backend) and **port 3000** (frontend)
4. Follow the setup instructions above inside the pod
5. For the frontend, set `NEXT_PUBLIC_WORKER_URL` to the RunPod proxy URL for port 8000

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `OFIQSampleApp: not found` | Build OFIQ or set `OFIQ_BIN` to the correct path |
| `conda: command not found` | Install Miniconda and add to PATH |
| `selfage` conda env not found | Run `conda create -n selfage python=3.10` and install deps |
| CUDA out of memory | Reduce `SELFAGE_RESOLUTION` to 256 or `SELFAGE_MAX_TRAIN_STEPS` |
| No faces detected | Ensure photos are well-lit, front-facing, with visible faces |
| Frontend can't reach backend | Check `NEXT_PUBLIC_WORKER_URL` in `.env.local` matches backend address |
| `agedb_index.npz` not found | Run `python -m worker.build_agedb_index` first |
