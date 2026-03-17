import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # FaceMAS/

# OFIQ
OFIQ = PROJECT_ROOT / "OFIQ-Project"
OFIQ_BIN = os.getenv("OFIQ_BIN") or str(OFIQ / "install_x86_64_linux" / "Release" / "bin" / "OFIQSampleApp")
OFIQ_CONFIG = os.getenv("OFIQ_CONFIG") or str(OFIQ / "data" / "ofiq_config.jaxn")

# SelfAge
SELFAGE = PROJECT_ROOT / "SelfAge"
SELFAGE_REPO = os.getenv("SELFAGE_REPO") or str(SELFAGE)
SELFAGE_REG_DIR = os.getenv("SELFAGE_REG_DIR") or str(SELFAGE / "data" / "CelebA_regularization_dex")
SELFAGE_INSTANCE_PROMPT = os.getenv("SELFAGE_INSTANCE_PROMPT") or "photo of sks person"

SELFAGE_RESOLUTION = int(os.getenv("SELFAGE_RESOLUTION", "512"))
SELFAGE_MAX_TRAIN_STEPS = int(os.getenv("SELFAGE_MAX_TRAIN_STEPS", "200"))
SELFAGE_RANK = int(os.getenv("SELFAGE_RANK", "16"))

# Service settings
UQS_GOOD = float(os.getenv("UQS_GOOD", "60"))
UQS_WARN = float(os.getenv("UQS_WARN", "45"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "10"))

# Temp working area (optional)
FACEPIPE_TMP = Path(os.getenv("FACEPIPE_TMP", "/tmp"))