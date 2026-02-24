import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from make_config import REPO_ROOT, SimpleConfig, ensure_repo_cwd, setup_env

ensure_repo_cwd()
setup_env()
sys.path.insert(0, str(REPO_ROOT))

from infer.modules.vc.modules import VC


def _resolve_from_cwd(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def main():
    p = argparse.ArgumentParser(description="RVC minimal convert (CLI)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output.wav")
    p.add_argument("--model", required=True, help="Path to .pth (recommended: rvc_minimal/models/*.pth)")
    p.add_argument("--index", default="", help="Optional path to .index")
    p.add_argument("--transpose", type=int, default=0)
    p.add_argument("--f0_method", default="rmvpe")
    p.add_argument("--index_rate", type=float, default=0.75)
    p.add_argument("--filter_radius", type=int, default=3)
    p.add_argument("--resample_sr", type=int, default=0)
    p.add_argument("--rms_mix_rate", type=float, default=0.25)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--no_half", action="store_true", help="Force float32")
    args = p.parse_args()

    input_path = _resolve_from_cwd(args.input)
    output_path = _resolve_from_cwd(args.output)
    model_path = _resolve_from_cwd(args.model)
    index_path = _resolve_from_cwd(args.index) if args.index else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if index_path and not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    # VC.get_vc() loads model by basename from weight_root; point it to the model folder.
    setup_env(
        model_dir=model_path.parent,
        index_dir=(index_path.parent if index_path else model_path.parent),
    )

    cfg = SimpleConfig(device=args.device, is_half=(not args.no_half))
    vc = VC(cfg)
    vc.get_vc(model_path.name)

    info, (sr, audio) = vc.vc_single(
        sid=0,
        input_audio_path=str(input_path),
        f0_up_key=args.transpose,
        f0_file=None,
        f0_method=args.f0_method,
        file_index=(str(index_path) if index_path else ""),
        file_index2="",
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )

    print(info)
    if sr is None or audio is None:
        raise RuntimeError("Conversion failed. Check the error output above.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    print(f"OK -> {output_path} (sr={sr})")


if __name__ == "__main__":
    main()
