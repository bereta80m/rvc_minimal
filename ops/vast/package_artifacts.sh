#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash ops/vast/package_artifacts.sh --exp <name> [options]

Required:
  --exp NAME

Options:
  --include-logs {0|1}   Default: 1
  --output-dir PATH      Default: exports
  --strict {0|1}         Default: 1
  --help                 Show this help
EOF
}

EXP=""
INCLUDE_LOGS="1"
OUTPUT_DIR="exports"
STRICT="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="${2:-}"; shift 2 ;;
    --include-logs) INCLUDE_LOGS="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --strict) STRICT="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EXP" ]]; then
  echo "[error] --exp is required" >&2
  usage
  exit 2
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
staging="$OUTPUT_DIR/_staging_${EXP}_${timestamp}"
archive="$OUTPUT_DIR/${EXP}_${timestamp}.tar.gz"
mkdir -p "$staging"

declare -a weight_files=()
while IFS= read -r -d '' f; do weight_files+=("$f"); done < <(find "models" -maxdepth 1 -type f -name "${EXP}*.pth" -print0 2>/dev/null || true)
if [[ ${#weight_files[@]} -eq 0 ]]; then
  while IFS= read -r -d '' f; do weight_files+=("$f"); done < <(find "assets/weights" -maxdepth 1 -type f -name "${EXP}*.pth" -print0 2>/dev/null || true)
fi

index_file=""
if [[ -f "models/${EXP}.index" ]]; then
  index_file="models/${EXP}.index"
else
  first_index="$(find "logs/$EXP" -maxdepth 1 -type f -name "added_*_${EXP}_*.index" 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "$first_index" ]]; then
    index_file="$first_index"
  fi
fi

if [[ "$STRICT" == "1" && ${#weight_files[@]} -eq 0 ]]; then
  echo "[error] no .pth found for exp '$EXP'" >&2
  echo "[info] searched models/${EXP}*.pth then assets/weights/${EXP}*.pth" >&2
  rm -rf "$staging"
  exit 3
fi

if [[ "$STRICT" == "1" && -z "$index_file" ]]; then
  echo "[error] no .index found for exp '$EXP'" >&2
  echo "[info] searched models/${EXP}.index then logs/$EXP/added_*_${EXP}_*.index" >&2
  rm -rf "$staging"
  exit 4
fi

mkdir -p "$staging/models" "$staging/meta"

copied_files=()
for src in "${weight_files[@]}"; do
  if [[ -f "$src" ]]; then
    cp -f "$src" "$staging/models/"
    copied_files+=("models/$(basename "$src")")
  fi
done

if [[ -n "$index_file" && -f "$index_file" ]]; then
  cp -f "$index_file" "$staging/models/${EXP}.index"
  copied_files+=("models/${EXP}.index")
fi

if [[ "$INCLUDE_LOGS" == "1" && -d "logs/$EXP" ]]; then
  mkdir -p "$staging/logs"
  cp -a "logs/$EXP" "$staging/logs/"
fi

if [[ -f "logs/$EXP/train_command.txt" ]]; then
  cp -f "logs/$EXP/train_command.txt" "$staging/meta/train_command.txt"
elif [[ -d "logs/$EXP" ]]; then
  candidate_cmd="$(find "logs/$EXP" -maxdepth 1 -type f -name "*command*.txt" | head -n 1 || true)"
  if [[ -n "$candidate_cmd" ]]; then
    cp -f "$candidate_cmd" "$staging/meta/train_command.txt"
  fi
fi

export VAST_EXPORT_EXP="$EXP"
export VAST_EXPORT_STAGING="$staging"
export VAST_EXPORT_INCLUDE_LOGS="$INCLUDE_LOGS"
export VAST_EXPORT_INDEX_FILE="$index_file"
export VAST_EXPORT_FILES="$(printf '%s\n' "${copied_files[@]}")"
python - <<'PY'
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

exp = os.environ["VAST_EXPORT_EXP"]
staging = Path(os.environ["VAST_EXPORT_STAGING"])
include_logs = os.environ.get("VAST_EXPORT_INCLUDE_LOGS", "1") == "1"
index_file = os.environ.get("VAST_EXPORT_INDEX_FILE", "")
files_env = os.environ.get("VAST_EXPORT_FILES", "")
files = [line for line in files_env.splitlines() if line.strip()]

gpu_info = ""
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_info = out.stdout.strip() if out.returncode == 0 else ""
except Exception:
    gpu_info = ""

repo_commit = ""
try:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    repo_commit = out.stdout.strip() if out.returncode == 0 else ""
except Exception:
    repo_commit = ""

manifest = {
    "exp": exp,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "gpu_info": gpu_info,
    "files": files,
    "repo_commit": repo_commit,
    "train_script": "ops/vast/train_vast.sh",
    "index_source": index_file,
    "include_logs": include_logs,
    "notes": "Packaged for ephemeral Vast.ai training workflow",
}
(staging / "meta" / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

tar -czf "$archive" -C "$staging" .
rm -rf "$staging"

echo "[ok] archive created: $archive"
echo "[hint] download with scp (replace host/port/path):"
echo "  scp -P <SSH_PORT> <USER>@<HOST>:$(pwd)/$archive ."
