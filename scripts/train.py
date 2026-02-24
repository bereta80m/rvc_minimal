import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from random import shuffle

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_config import (  # noqa: E402
    MODELS_DIR,
    REPO_ROOT,
    default_pretrained,
    ensure_repo_cwd,
    load_train_config_template,
    setup_env,
)


SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def exp_logs_dir(exp: str) -> Path:
    return REPO_ROOT / "logs" / exp


def reset_experiment_logs(exp: str) -> Path:
    log_dir = exp_logs_dir(exp)
    if log_dir.exists():
        print(f"[reset] removing previous experiment logs: {log_dir}")
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def cleanup_experiment_logs(exp: str) -> None:
    log_dir = exp_logs_dir(exp)
    if log_dir.exists():
        print(f"[cleanup] removing failed experiment logs: {log_dir}")
        shutil.rmtree(log_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVC minimal training CLI (no infer-web)")
    p.add_argument("--dataset", required=True, help="Dataset folder with wavs")
    p.add_argument("--exp", required=True, help="Experiment name (logs/<exp>)")
    p.add_argument("--sr", choices=["32k", "40k", "48k"], default="40k")
    p.add_argument("--version", choices=["v1", "v2"], default="v2")
    p.add_argument("--if_f0", type=int, choices=[0, 1], default=1)
    p.add_argument("--spk_id", type=int, default=0)
    p.add_argument("--np", type=int, default=4, help="CPU processes for preprocess/extract")
    p.add_argument("--gpus", default="0", help='GPU ids split by "-", e.g. 0 or 0-1')
    p.add_argument("--gpus_rmvpe", default="", help='GPU ids for rmvpe_gpu split by "-", default uses --gpus')
    p.add_argument("--f0_method", default="rmvpe", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--total_epoch", type=int, default=20)
    p.add_argument("--save_every_epoch", type=int, default=5)
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop early if training metric stops improving for N epochs (0 disables)",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early-stop patience",
    )
    p.add_argument(
        "--early_stop_metric",
        choices=["loss_mel", "loss_gen_all"],
        default="loss_mel",
        help="Training metric used for early stopping (validation loss is not available in this repo)",
    )
    p.add_argument("--save_latest", action="store_true")
    p.add_argument("--cache_gpu", action="store_true")
    p.add_argument("--save_every_weights", action="store_true")
    p.add_argument("--pretrained_g", default="")
    p.add_argument("--pretrained_d", default="")
    p.add_argument("--device", default=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") or _cuda_available() else "cpu"))
    p.add_argument("--is_half", type=int, choices=[0, 1], default=None)
    p.add_argument(
        "--feature_retry_no_half",
        action="store_true",
        default=True,
        help="Retry feature extraction with full precision if the first attempt fails",
    )
    p.add_argument(
        "--feature_fallback_cpu",
        action="store_true",
        default=True,
        help="Fallback feature extraction to CPU if CUDA extraction fails",
    )
    p.add_argument(
        "--keep_failed_logs",
        action="store_true",
        help="Preserve logs/<exp> when the pipeline fails",
    )
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_extract", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_index", action="store_true")
    p.add_argument("--copy_to_models", action="store_true", help="Copy final .pth/.index to rvc_minimal/models")
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def preprocess(dataset_dir: Path, exp: str, sr: str, n_p: int) -> None:
    log_dir = exp_logs_dir(exp)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "infer/modules/train/preprocess.py",
        str(dataset_dir),
        str(SR_DICT[sr]),
        str(n_p),
        str(log_dir),
        "False",  # noparallel
        "3.7",    # preprocess_per
    ]
    run_cmd(cmd)


def extract_f0_and_features(
    exp: str,
    version: str,
    if_f0: bool,
    f0_method: str,
    n_p: int,
    gpus: str,
    gpus_rmvpe: str,
    device: str,
    is_half: bool,
    feature_retry_no_half: bool,
    feature_fallback_cpu: bool,
) -> None:
    exp_dir = exp_logs_dir(exp)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if if_f0:
        if f0_method != "rmvpe_gpu":
            run_cmd(
                [
                    sys.executable,
                    "infer/modules/train/extract/extract_f0_print.py",
                    str(exp_dir),
                    str(n_p),
                    f0_method,
                ]
            )
        else:
            ids = (gpus_rmvpe or gpus).split("-")
            total = str(len(ids))
            for idx, gpu in enumerate(ids):
                run_cmd(
                    [
                        sys.executable,
                        "infer/modules/train/extract/extract_f0_rmvpe.py",
                        total,
                        str(idx),
                        gpu,
                        str(exp_dir),
                        str(bool(is_half)),
                    ]
                )

    gpu_ids = gpus.split("-") if gpus else ["0"]
    total_parts = str(len(gpu_ids))
    for idx, gpu in enumerate(gpu_ids):
        base_cmd = [
            sys.executable,
            "infer/modules/train/extract_feature_print.py",
            device,
            total_parts,
            str(idx),
            gpu,
            str(exp_dir),
            version,
            str(bool(is_half)),
        ]
        try:
            run_cmd(base_cmd)
        except subprocess.CalledProcessError as err:
            attempts = [(device, bool(is_half))]
            recovered = False

            if feature_retry_no_half and bool(is_half):
                retry_cmd = base_cmd.copy()
                retry_cmd[2] = device
                retry_cmd[-1] = "False"
                print(
                    f"[retry] extract_feature_print failed (rc={err.returncode}) on {device}, retrying with is_half=False"
                )
                try:
                    run_cmd(retry_cmd)
                    recovered = True
                except subprocess.CalledProcessError as err2:
                    err = err2
                    attempts.append((device, False))

            if (not recovered) and feature_fallback_cpu and device != "cpu":
                cpu_retry = ("cpu", False)
                if cpu_retry not in attempts:
                    retry_cmd = base_cmd.copy()
                    retry_cmd[2] = "cpu"
                    retry_cmd[-1] = "False"
                    print("[fallback] extract_feature_print retrying on cpu with is_half=False")
                    run_cmd(retry_cmd)
                    recovered = True

            if not recovered:
                raise err


def build_filelist_and_config(
    exp: str,
    sr: str,
    version: str,
    if_f0: bool,
    spk_id: int,
) -> tuple[Path, Path]:
    exp_dir = exp_logs_dir(exp)
    exp_dir.mkdir(parents=True, exist_ok=True)

    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

    def base_id(p: Path) -> str:
        # Match WebUI behavior: supports files like "xxx.wav.npy" by taking prefix before first dot.
        return p.name.split(".")[0]

    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            {base_id(p) for p in gt_wavs_dir.iterdir() if p.is_file()}
            & {base_id(p) for p in feature_dir.iterdir() if p.is_file()}
            & {base_id(p) for p in f0_dir.iterdir() if p.is_file()}
            & {base_id(p) for p in f0nsf_dir.iterdir() if p.is_file()}
        )
    else:
        names = {base_id(p) for p in gt_wavs_dir.iterdir() if p.is_file()} & {
            base_id(p) for p in feature_dir.iterdir() if p.is_file()
        }

    rows: list[str] = []

    def esc(path: Path) -> str:
        return str(path).replace("\\", "\\\\")

    for name in names:
        if if_f0:
            gt_esc = esc(gt_wavs_dir)
            feat_esc = esc(feature_dir)
            f0_esc = esc(exp_dir / "2a_f0")
            f0nsf_esc = esc(exp_dir / "2b-f0nsf")
            rows.append(
                f"{gt_esc}\\\\{name}.wav|"
                f"{feat_esc}\\\\{name}.npy|"
                f"{f0_esc}\\\\{name}.wav.npy|"
                f"{f0nsf_esc}\\\\{name}.wav.npy|"
                f"{spk_id}"
            )
        else:
            gt_esc = esc(gt_wavs_dir)
            feat_esc = esc(feature_dir)
            rows.append(
                f"{gt_esc}\\\\{name}.wav|"
                f"{feat_esc}\\\\{name}.npy|"
                f"{spk_id}"
            )

    # Optional mute samples if present
    fea_dim = 256 if version == "v1" else 768
    mute_root = REPO_ROOT / "logs" / "mute"
    if mute_root.exists():
        if if_f0:
            mute_gt = esc(mute_root / "0_gt_wavs" / f"mute{sr}.wav")
            mute_feat = esc(mute_root / f"3_feature{fea_dim}" / "mute.npy")
            mute_f0 = esc(mute_root / "2a_f0" / "mute.wav.npy")
            mute_f0nsf = esc(mute_root / "2b-f0nsf" / "mute.wav.npy")
            mute_line = (
                f"{mute_gt}|"
                f"{mute_feat}|"
                f"{mute_f0}|"
                f"{mute_f0nsf}|"
                f"{spk_id}"
            )
        else:
            mute_gt = esc(mute_root / "0_gt_wavs" / f"mute{sr}.wav")
            mute_feat = esc(mute_root / f"3_feature{fea_dim}" / "mute.npy")
            mute_line = (
                f"{mute_gt}|"
                f"{mute_feat}|"
                f"{spk_id}"
            )
        rows.extend([mute_line, mute_line])

    if not rows:
        raise RuntimeError("No training rows generated. Check preprocess/extract outputs in logs/<exp>.")

    shuffle(rows)
    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(rows), encoding="utf-8")

    config_json = load_train_config_template(sr=sr, version=version)
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config_json, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    return filelist_path, config_path


def train_model(
    exp: str,
    sr: str,
    version: str,
    if_f0: bool,
    gpus: str,
    batch_size: int,
    total_epoch: int,
    save_every_epoch: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_metric: str,
    pretrained_g: str,
    pretrained_d: str,
    save_latest: bool,
    cache_gpu: bool,
    save_every_weights: bool,
) -> None:
    cmd = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e",
        exp,
        "-sr",
        sr,
        "-f0",
        "1" if if_f0 else "0",
        "-bs",
        str(batch_size),
        "-te",
        str(total_epoch),
        "-se",
        str(save_every_epoch),
        "--early_stop_patience",
        str(early_stop_patience),
        "--early_stop_min_delta",
        str(early_stop_min_delta),
        "--early_stop_metric",
        early_stop_metric,
        "-l",
        "1" if save_latest else "0",
        "-c",
        "1" if cache_gpu else "0",
        "-sw",
        "1" if save_every_weights else "0",
        "-v",
        version,
    ]
    if gpus:
        cmd.extend(["-g", gpus])
    if pretrained_g:
        cmd.extend(["-pg", pretrained_g])
    if pretrained_d:
        cmd.extend(["-pd", pretrained_d])
    run_cmd(cmd)


def train_index(exp: str, version: str) -> Path:
    import faiss
    from sklearn.cluster import MiniBatchKMeans

    exp_dir = exp_logs_dir(exp)
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature dir missing: {feature_dir}")

    files = sorted([p for p in feature_dir.iterdir() if p.suffix == ".npy"])
    if not files:
        raise RuntimeError(f"No feature .npy files in {feature_dir}")

    npys = [np.load(str(p)) for p in files]
    big_npy = np.concatenate(npys, 0)
    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]

    if big_npy.shape[0] > 2e5:
        print(f"[index] kmeans compressing {big_npy.shape[0]} frames to 10k centers")
        try:
            n_cpu = max(1, (os.cpu_count() or 1))
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception as e:
            print(f"[index] kmeans skipped due to error: {e}")

    np.save(str(exp_dir / "total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    dim = 256 if version == "v1" else 768
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    print(f"[index] training dim={dim} n_ivf={n_ivf}")
    index.train(big_npy)

    trained_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp}_{version}.index"
    faiss.write_index(index, str(trained_path))
    print("[index] adding vectors")
    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i : i + 8192])

    added_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp}_{version}.index"
    faiss.write_index(index, str(added_path))
    print(f"[index] OK -> {added_path}")
    return added_path


def maybe_copy_outputs(exp: str, index_path: Path | None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    weight_candidates = sorted((REPO_ROOT / "assets" / "weights").glob(f"{exp}*.pth"))
    if weight_candidates:
        for src in weight_candidates:
            dst = MODELS_DIR / src.name
            shutil.copy2(src, dst)
            print(f"[copy] model -> {dst}")
    else:
        print("[copy] no weights found in assets/weights (did you enable --save_every_weights?)")

    if index_path and index_path.exists():
        dst = MODELS_DIR / f"{exp}.index"
        shutil.copy2(index_path, dst)
        print(f"[copy] index -> {dst}")


def main() -> int:
    ensure_repo_cwd()
    setup_env()
    args = parse_args()

    if args.skip_preprocess or args.skip_extract or args.skip_train or args.skip_index:
        raise ValueError(
            "This train.py is configured for full training only. Do not use --skip_* flags."
        )

    dataset_dir = resolve_path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if args.version == "v1" and args.sr == "32k":
        raise ValueError("v1 does not support 32k in this WebUI config. Use 40k/48k or version v2.")

    if args.is_half is None:
        is_half = _cuda_available()
    else:
        is_half = bool(args.is_half)
    device = args.device

    pretrained_g = args.pretrained_g
    pretrained_d = args.pretrained_d
    if not pretrained_g and not pretrained_d:
        pretrained_g, pretrained_d = default_pretrained(args.sr, args.version, bool(args.if_f0))
        print(f"[pretrained] G={pretrained_g or '(none)'}")
        print(f"[pretrained] D={pretrained_d or '(none)'}")

    reset_experiment_logs(args.exp)
    try:
        preprocess(dataset_dir, args.exp, args.sr, args.np)

        extract_f0_and_features(
            exp=args.exp,
            version=args.version,
            if_f0=bool(args.if_f0),
            f0_method=args.f0_method,
            n_p=args.np,
            gpus=args.gpus,
            gpus_rmvpe=args.gpus_rmvpe,
            device=device,
            is_half=is_half,
            feature_retry_no_half=args.feature_retry_no_half,
            feature_fallback_cpu=args.feature_fallback_cpu,
        )

        filelist_path, config_path = build_filelist_and_config(
            exp=args.exp,
            sr=args.sr,
            version=args.version,
            if_f0=bool(args.if_f0),
            spk_id=args.spk_id,
        )
        print(f"[ok] filelist -> {filelist_path}")
        print(f"[ok] config   -> {config_path}")

        train_model(
            exp=args.exp,
            sr=args.sr,
            version=args.version,
            if_f0=bool(args.if_f0),
            gpus=args.gpus,
            batch_size=args.batch_size,
            total_epoch=args.total_epoch,
            save_every_epoch=args.save_every_epoch,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_metric=args.early_stop_metric,
            pretrained_g=pretrained_g,
            pretrained_d=pretrained_d,
            save_latest=args.save_latest,
            cache_gpu=args.cache_gpu,
            save_every_weights=args.save_every_weights,
        )

        # If requested, require an exported small model to exist before continuing.
        if args.save_every_weights:
            weight_candidates = sorted((REPO_ROOT / "assets" / "weights").glob(f"{args.exp}*.pth"))
            if not weight_candidates:
                raise RuntimeError(
                    "Training subprocess ended but no .pth was found in assets/weights."
                )

        built_index = train_index(args.exp, args.version)

        if args.copy_to_models:
            maybe_copy_outputs(args.exp, built_index)

        print("[done] training pipeline finished")
        return 0
    except Exception:
        if args.keep_failed_logs:
            print(f"[cleanup] skipped; preserving logs at {exp_logs_dir(args.exp)}")
        else:
            cleanup_experiment_logs(args.exp)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
