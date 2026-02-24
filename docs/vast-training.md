# Vast.ai Training Runbook (RVC Minimal, GPU training only)

This guide prepares `rvc_minimal` for an ephemeral Vast.ai workflow:

1. Rent a GPU instance
2. SSH in
3. Clone the repo
4. Run one setup command
5. Run training
6. Package `.pth` / `.index` (+ optional logs)
7. Download the tarball
8. Destroy the instance

This workflow does **not** run the API on Vast. Inference/conversion remains local.

## 1) What To Rent In Vast.ai

Use template:

- `PyTorch (Vast)`

Recommended for RVC training:

- `RTX 3060 12GB` or better

Minimum practical specs:

- `12GB VRAM` recommended
- `>=16GB RAM`
- `>=60GB disk` (better `100GB`)
- `Verified` host
- Enough `Max Duration` for your training run

Notes:

- 8GB GPUs can work but may require lower `--batch-size` and more tuning.
- Bandwidth may be billed separately on some hosts.

## 2) SSH + Clone Repo

From the Vast.ai instance page, copy the SSH command (host + port vary). Example:

```bash
ssh -p <SSH_PORT> root@<HOST>
```

Then clone:

```bash
git clone <YOUR_REPO_URL> rvc_minimal
cd rvc_minimal
```

## 3) One-Time Setup (per instance)

Run:

```bash
bash ops/vast/setup_train_env.sh
```

What it does:

- Creates `.venv-vast` (with `--system-site-packages` to reuse template PyTorch/CUDA)
- Installs `requirements-train.txt`
- Runs GPU/import diagnostics
- Verifies write access for training output folders

If setup fails on `ffmpeg` binary:

- Install system `ffmpeg` on the instance (depends on host image)
- Re-run setup

## 4) Upload Dataset To The Instance

Put your dataset under `dataset_raw/<exp>/` with supported audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`).

### Option A: `scp`

```bash
scp -P <SSH_PORT> -r ./my_dataset root@<HOST>:/workspace/rvc_minimal/dataset_raw/my_voice
```

### Option B: `rsync` (recommended for large datasets / resume)

```bash
rsync -avP -e "ssh -p <SSH_PORT>" ./my_dataset/ root@<HOST>:/workspace/rvc_minimal/dataset_raw/my_voice/
```

### Option C: Download directly on the instance

```bash
mkdir -p dataset_raw/my_voice
wget -O dataset_raw/my_voice/data.zip <URL>
# unzip / extract as needed
```

## 5) Run Training (GPU)

Example:

```bash
bash ops/vast/train_vast.sh \
  --dataset dataset_raw/mi_voice \
  --exp mi_voice_v1 \
  --sr 40k \
  --version v2 \
  --if-f0 1 \
  --f0-method rmvpe_gpu \
  --gpus 0 \
  --gpus-rmvpe 0 \
  --batch-size 4 \
  --epochs 100 \
  --save-every 5
```

Defaults the wrapper applies:

- `--copy_to_models` enabled
- `--save_every_weights` enabled
- `--keep_failed_logs` enabled

Logs:

- Training console output ends up at `logs/<exp>/vast_train_console.log`
- The exact command is saved to `logs/<exp>/train_command.txt`

## 6) Package Artifacts

Create a downloadable tarball:

```bash
bash ops/vast/package_artifacts.sh --exp mi_voice_v1 --include-logs 1
```

Output:

- `exports/mi_voice_v1_<timestamp>.tar.gz`

The archive includes:

- `.pth` weight(s)
- `.index`
- logs (if enabled)
- `meta/manifest.json`
- `meta/train_command.txt` (if present)

## 7) Download The Tarball

Use `scp` from your local machine:

```bash
scp -P <SSH_PORT> root@<HOST>:/workspace/rvc_minimal/exports/mi_voice_v1_<timestamp>.tar.gz .
```

Validate locally:

```bash
tar -tzf mi_voice_v1_<timestamp>.tar.gz | head
```

## 8) Destroy The Vast Instance

After the tarball is downloaded and verified:

- Stop / destroy the instance in Vast.ai UI

This workflow assumes no persistent volume and no need to keep the environment alive.

## Troubleshooting

### `torch.cuda.is_available() == False`

- Check `nvidia-smi`
- Confirm you used `PyTorch (Vast)` template
- Re-run `bash ops/vast/setup_train_env.sh`
- Verify the instance actually has a GPU attached

### Out of memory (OOM)

- Lower `--batch-size` (first thing to try)
- Try `--is-half 1` (or leave `auto` if CUDA works)
- Use a GPU with more VRAM (12GB+ recommended)

### `rmvpe_gpu` fails

Try CPU-based fallback method:

```bash
bash ops/vast/train_vast.sh ... --f0-method rmvpe
```

### Missing `faiss` / `sklearn` / audio libs

Reinstall inside the venv:

```bash
source .venv-vast/bin/activate
python -m pip install -r requirements-train.txt
python ops/vast/check_gpu_env.py
```

### `ffmpeg` binary missing

`ffmpeg-python` is only the Python wrapper. You also need the system `ffmpeg` executable in `PATH`.

Install it with your host image package manager, then rerun setup.

## End-to-End Example (copy/paste)

```bash
git clone <YOUR_REPO_URL> rvc_minimal
cd rvc_minimal
bash ops/vast/setup_train_env.sh
# upload dataset to dataset_raw/<exp>/
bash ops/vast/train_vast.sh --dataset dataset_raw/<exp> --exp <exp> --gpus 0
bash ops/vast/package_artifacts.sh --exp <exp>
# scp exports/<exp>_<timestamp>.tar.gz
```
