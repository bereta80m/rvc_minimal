#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-vast}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[error] missing required command: $cmd" >&2
    exit 1
  fi
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[error] python/python3 not found" >&2
  exit 1
fi

require_cmd git
require_cmd "$PYTHON_BIN"

echo "[info] repo: $ROOT_DIR"
echo "[info] python: $PYTHON_BIN"
"$PYTHON_BIN" --version

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] nvidia-smi"
  nvidia-smi || true
else
  echo "[warn] nvidia-smi not found in PATH"
fi

echo "[info] host torch diagnostics (pre-venv)"
"$PYTHON_BIN" - <<'PY' || true
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"gpu[{i}]={p.name} vram_gb={p.total_memory/1024**3:.2f}")
except Exception as e:
    print(f"torch_import_error={type(e).__name__}: {e}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[info] creating venv at $VENV_DIR (with --system-site-packages to reuse template torch)"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
else
  echo "[info] reusing existing venv: $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-train.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[warn] ffmpeg binary not found. Install it if preprocessing fails (python package ffmpeg-python is not enough)."
fi

mkdir -p dataset_raw logs models api_data exports

echo "[info] final environment check"
python ops/vast/check_gpu_env.py

echo
echo "[ok] Vast training environment ready."
echo "[next] Activate env: source $VENV_DIR/bin/activate"
echo "[next] Train example:"
echo "       bash ops/vast/train_vast.sh --dataset dataset_raw/<exp> --exp <exp> --gpus 0"
