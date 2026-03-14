import json
import csv
import subprocess
from pathlib import Path
from typing import Dict

from worker.config import OFIQ_BIN, OFIQ_CONFIG

def _run(cmd: str):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"OFIQ failed\nCMD: {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

def _parse(out_dir: Path) -> Dict[str, float]:
    fp = out_dir / "results.csv"
    if not fp.exists():
        return {}

    scores: Dict[str, float] = {}
    with fp.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")
        if not reader.fieldnames:
            return {}

        # exact headers (strip just in case)
        fn_col = None
        uqs_col = None
        for h in reader.fieldnames:
            hs = h.strip().lstrip("\ufeff")
            if hs == "Filename":
                fn_col = h
            if hs == "UnifiedQualityScore.scalar":
                uqs_col = h

        if fn_col is None or uqs_col is None:
            return {}

        for row in reader:
            fn = Path((row.get(fn_col) or "").strip()).name
            val = (row.get(uqs_col) or "").strip()
            if not fn or not val:
                continue
            try:
                scores[fn] = float(val)
            except Exception:
                pass
                
    return scores

def score_dir(img_dir: Path) -> Dict[str, float]:
    if not OFIQ_BIN or not Path(OFIQ_BIN).exists():
        raise RuntimeError("OFIQ_BIN not set / missing")
    if not OFIQ_CONFIG or not Path(OFIQ_CONFIG).exists():
        raise RuntimeError("OFIQ_CONFIG not set / missing")

    out_dir = img_dir.parent / "ofiq_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "results.csv"   # <- output FILE
    cmd = f'"{OFIQ_BIN}" -i "{img_dir}" -o "{out_csv}" -c "{OFIQ_CONFIG}"'
    _run(cmd)

    # parse either the explicit CSV or anything else OFIQ produced
    scores = _parse(out_dir)
    return scores