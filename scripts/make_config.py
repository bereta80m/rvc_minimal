import json
import os
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RVC_MINIMAL_ROOT = SCRIPT_DIR.parent
MODELS_DIR = RVC_MINIMAL_ROOT / "models"
REPO_ROOT = RVC_MINIMAL_ROOT


def ensure_repo_cwd() -> None:
    os.chdir(REPO_ROOT)


def setup_env(model_dir: Path | None = None, index_dir: Path | None = None) -> None:
    model_dir = (model_dir or MODELS_DIR).resolve()
    index_dir = (index_dir or model_dir).resolve()
    os.environ["weight_root"] = str(model_dir)
    os.environ["index_root"] = str(index_dir)
    os.environ.setdefault("outside_index_root", str(index_dir))
    os.environ.setdefault("rmvpe_root", str((REPO_ROOT / "assets" / "rmvpe").resolve()))
    os.environ.setdefault("weight_uvr5_root", str((REPO_ROOT / "assets" / "uvr5_weights").resolve()))


class SimpleConfig:
    def __init__(self, device: str | None = None, is_half: bool | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if is_half is None:
            is_half = "cuda" in str(device)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_half = bool(is_half)

        # Defaults used by RVC WebUI for inference pipeline
        self.x_pad = 3 if self.is_half else 1
        self.x_query = 10 if self.is_half else 6
        self.x_center = 60 if self.is_half else 38
        self.x_max = 65 if self.is_half else 41


def load_train_config_template(sr: str, version: str) -> dict:
    if version == "v1" or sr == "40k":
        rel = Path("configs") / "v1" / f"{sr}.json"
    else:
        rel = Path("configs") / "v2" / f"{sr}.json"
    path = REPO_ROOT / rel
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_pretrained(sr: str, version: str, if_f0: bool) -> tuple[str, str]:
    suffix = "" if version == "v1" else "_v2"
    f0_prefix = "f0" if if_f0 else ""
    g = REPO_ROOT / "assets" / f"pretrained{suffix}" / f"{f0_prefix}G{sr}.pth"
    d = REPO_ROOT / "assets" / f"pretrained{suffix}" / f"{f0_prefix}D{sr}.pth"
    return (str(g) if g.exists() else "", str(d) if d.exists() else "")
