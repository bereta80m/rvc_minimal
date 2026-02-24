#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-vast}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[error] venv not found at $VENV_DIR. Run: bash ops/vast/setup_train_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

usage() {
  cat <<'EOF'
Usage:
  bash ops/vast/train_vast.sh --dataset <path> --exp <name> [options]

Required:
  --dataset PATH            Dataset folder (e.g. dataset_raw/my_voice)
  --exp NAME                Experiment name (logs/<exp>)

Options:
  --sr {32k|40k|48k}        Default: 40k
  --version {v1|v2}         Default: v2
  --if-f0 {0|1}             Default: 1
  --f0-method METHOD        Default: rmvpe_gpu
  --gpus IDS                Default: 0
  --gpus-rmvpe IDS          Default: same as --gpus
  --batch-size N            Default: 4
  --epochs N                Default: 100
  --save-every N            Default: 5
  --np N                    Default: min(8, nproc)
  --is-half {auto|0|1}      Default: auto
  --device {cuda|cpu}       Default: cuda
  --copy-to-models          Enable copy to models/ (default)
  --no-copy-to-models       Disable copy to models/
  --save-every-weights      Enable save exported weights (default)
  --no-save-every-weights   Disable save exported weights
  --save-latest             Enable save_latest
  --cache-gpu               Enable cache_gpu
  --help                    Show this help
EOF
}

pick_np_default() {
  local n=4
  if command -v nproc >/dev/null 2>&1; then
    n="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    n="$(getconf _NPROCESSORS_ONLN || echo 4)"
  fi
  if [[ "$n" -gt 8 ]]; then
    n=8
  fi
  if [[ "$n" -lt 1 ]]; then
    n=1
  fi
  echo "$n"
}

DATASET=""
EXP=""
SR="40k"
VERSION="v2"
IF_F0="1"
F0_METHOD="rmvpe_gpu"
GPUS="0"
GPUS_RMVPE=""
BATCH_SIZE="4"
EPOCHS="100"
SAVE_EVERY="5"
NP="$(pick_np_default)"
IS_HALF="auto"
DEVICE="cuda"
COPY_TO_MODELS="1"
SAVE_EVERY_WEIGHTS="1"
SAVE_LATEST="0"
CACHE_GPU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --exp) EXP="${2:-}"; shift 2 ;;
    --sr) SR="${2:-}"; shift 2 ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --if-f0) IF_F0="${2:-}"; shift 2 ;;
    --f0-method) F0_METHOD="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --gpus-rmvpe) GPUS_RMVPE="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --save-every) SAVE_EVERY="${2:-}"; shift 2 ;;
    --np) NP="${2:-}"; shift 2 ;;
    --is-half) IS_HALF="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --copy-to-models) COPY_TO_MODELS="1"; shift ;;
    --no-copy-to-models) COPY_TO_MODELS="0"; shift ;;
    --save-every-weights) SAVE_EVERY_WEIGHTS="1"; shift ;;
    --no-save-every-weights) SAVE_EVERY_WEIGHTS="0"; shift ;;
    --save-latest) SAVE_LATEST="1"; shift ;;
    --cache-gpu) CACHE_GPU="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$DATASET" || -z "$EXP" ]]; then
  echo "[error] --dataset and --exp are required" >&2
  usage
  exit 2
fi

if [[ ! -d "$DATASET" ]]; then
  echo "[error] dataset folder not found: $DATASET" >&2
  exit 3
fi

if ! find "$DATASET" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.m4a' -o -iname '*.aac' \) | grep -q .; then
  echo "[error] dataset has no supported audio files: $DATASET" >&2
  exit 4
fi

case "$SR" in
  32k|40k|48k) ;;
  *) echo "[error] invalid --sr: $SR" >&2; exit 2 ;;
esac
case "$VERSION" in
  v1|v2) ;;
  *) echo "[error] invalid --version: $VERSION" >&2; exit 2 ;;
esac
case "$IF_F0" in
  0|1) ;;
  *) echo "[error] invalid --if-f0: $IF_F0" >&2; exit 2 ;;
esac
case "$IS_HALF" in
  auto|0|1) ;;
  *) echo "[error] invalid --is-half: $IS_HALF" >&2; exit 2 ;;
esac

if [[ -z "$GPUS_RMVPE" ]]; then
  GPUS_RMVPE="$GPUS"
fi

mkdir -p logs models
timestamp="$(date -u +%Y%m%d_%H%M%S)"
tmp_console_dir="logs/_vast_console"
mkdir -p "$tmp_console_dir"
tmp_console_log="$tmp_console_dir/${EXP}_${timestamp}.log"

cmd=(python scripts/train.py
  --dataset "$DATASET"
  --exp "$EXP"
  --sr "$SR"
  --version "$VERSION"
  --if_f0 "$IF_F0"
  --np "$NP"
  --gpus "$GPUS"
  --gpus_rmvpe "$GPUS_RMVPE"
  --f0_method "$F0_METHOD"
  --batch_size "$BATCH_SIZE"
  --total_epoch "$EPOCHS"
  --save_every_epoch "$SAVE_EVERY"
  --device "$DEVICE"
  --keep_failed_logs
)

if [[ "$IS_HALF" != "auto" ]]; then
  cmd+=(--is_half "$IS_HALF")
fi
if [[ "$COPY_TO_MODELS" == "1" ]]; then
  cmd+=(--copy_to_models)
fi
if [[ "$SAVE_EVERY_WEIGHTS" == "1" ]]; then
  cmd+=(--save_every_weights)
fi
if [[ "$SAVE_LATEST" == "1" ]]; then
  cmd+=(--save_latest)
fi
if [[ "$CACHE_GPU" == "1" ]]; then
  cmd+=(--cache_gpu)
fi

train_cmd_pretty="$(printf '%q ' "${cmd[@]}")"
echo "[info] training command:"
echo "  $train_cmd_pretty"

echo "$train_cmd_pretty" > "$tmp_console_dir/${EXP}_${timestamp}.command.txt"

set +e
"${cmd[@]}" 2>&1 | tee "$tmp_console_log"
rc=${PIPESTATUS[0]}
set -e

mkdir -p "logs/$EXP"
cp -f "$tmp_console_log" "logs/$EXP/vast_train_console.log" || true
cp -f "$tmp_console_dir/${EXP}_${timestamp}.command.txt" "logs/$EXP/train_command.txt" || true

if [[ $rc -ne 0 ]]; then
  echo "[error] training failed with exit code $rc" >&2
  echo "[hint] console log: logs/$EXP/vast_train_console.log" >&2
  exit "$rc"
fi

echo
echo "[ok] training completed"
echo "[info] expected outputs:"
echo "  models/${EXP}*.pth"
echo "  models/${EXP}.index"
echo "  logs/${EXP}/vast_train_console.log"
echo "[next] package artifacts:"
echo "  bash ops/vast/package_artifacts.sh --exp $EXP --include-logs 1"
