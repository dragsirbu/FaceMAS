import os
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

from worker.config import (
    SELFAGE_REPO,
    SELFAGE_REG_DIR,
    SELFAGE_INSTANCE_PROMPT,
    SELFAGE_RESOLUTION,
    SELFAGE_MAX_TRAIN_STEPS,
    SELFAGE_RANK,
)

# If conda isn't on PATH in your service process, set this once
CONDA_PREFIX_BIN = "/workspace/conda/condabin"
SELFAGE_CONDA_ENV = "selfage"

def _env_with_pythonpath() -> dict:
    env = os.environ.copy()

    # make conda visible (same as your notebook cell)
    if CONDA_PREFIX_BIN and CONDA_PREFIX_BIN not in env.get("PATH", ""):
        env["PATH"] = CONDA_PREFIX_BIN + ":" + env.get("PATH", "")

    # make SelfAge importable
    root = str(Path(SELFAGE_REPO))
    env["PYTHONPATH"] = root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    return env

def _req_dir(p: str, name: str):
    if not p or not Path(p).exists():
        raise RuntimeError(f"{name} missing: {p}")

def _run(cmd: List[str], cwd: Path):
    res = subprocess.run(cmd, cwd=str(cwd), env=_env_with_pythonpath(), text=True)
    if res.returncode != 0:
        raise RuntimeError(f"SelfAge failed: {' '.join(map(str, cmd))}\nReturn code: {res.returncode}")

def train(self_ref_dir: Path, exp_dir: Path):
    _req_dir(SELFAGE_REPO, "SELFAGE_REPO")
    _req_dir(SELFAGE_REG_DIR, "SELFAGE_REG_DIR")
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", SELFAGE_CONDA_ENV,
        "accelerate", "launch", "scripts/train.py",
        f"--self_ref_data_dir={self_ref_dir}",
        f"--output_dir={exp_dir}",
        f"--regularization_dir={Path(SELFAGE_REG_DIR)}",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--contrast_weight=0.1",
        f"--instance_prompt={SELFAGE_INSTANCE_PROMPT}",
        f"--resolution={int(SELFAGE_RESOLUTION)}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={int(SELFAGE_MAX_TRAIN_STEPS)}",
        "--checkpointing_steps=200",
        f"--validation_prompt={SELFAGE_INSTANCE_PROMPT}",
        "--validation_epochs=1",
        "--train_text_encoder",
        f"--rank={int(SELFAGE_RANK)}",
        "--num_instance_images=3",
    ]
    _run(cmd, cwd=Path(SELFAGE_REPO))

def find_personalized_path(exp_dir: Path) -> Path:
    candidates = [
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    for fn in candidates:
        hits = list(exp_dir.rglob(fn))
        if hits:
            return hits[0].parent
    raise RuntimeError(f"Could not find LoRA weights under: {exp_dir}")

def age_edit(
    data_dir: Path,
    exp_dir: Path,
    target_ages: List[int],
    gender: str = "",
    attributes: str = "",
    test_batch_size: int = 4,
    test_workers: int = 4,
) -> Tuple[float, Path]:
    _req_dir(SELFAGE_REPO, "SELFAGE_REPO")

    personalized_path = find_personalized_path(exp_dir)
    ages = ",".join(str(a) for a in target_ages)

    cmd = [
        "conda", "run", "-n", SELFAGE_CONDA_ENV,
        "python", "scripts/age_editing.py",
        f"--data_path={data_dir}",
        f"--exp_dir={exp_dir}",
        f"--personalized_path={personalized_path}",
        "--target_age", ages,
        f"--test_batch_size={test_batch_size}",
        f"--test_workers={test_workers}",
    ]
    if gender:
        cmd.append(f"--gender={gender}")
    if attributes:
        cmd.append(f"--attributes={attributes}")

    t0 = time.time()
    _run(cmd, cwd=Path(SELFAGE_REPO))
    return t0, exp_dir

def collect_new_images(root: Path, since_ts: float, limit: int = 50) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_mtime >= since_ts:
            imgs.append(p)
    imgs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return imgs[:limit]
