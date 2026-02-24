import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset_raw"
MODELS_ROOT = REPO_ROOT / "models"
INPUT_ROOT = REPO_ROOT / "input"
API_DATA_ROOT = REPO_ROOT / "api_data"
JOBS_ROOT = API_DATA_ROOT / "jobs"
UPLOADS_ROOT = API_DATA_ROOT / "uploads"
OUTPUTS_ROOT = API_DATA_ROOT / "outputs"

for p in [DATASET_ROOT, MODELS_ROOT, INPUT_ROOT, JOBS_ROOT, UPLOADS_ROOT, OUTPUTS_ROOT]:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_audio_filename(name: str) -> bool:
    return Path(name).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def sanitize_name(value: str, label: str = "name") -> str:
    clean = value.strip()
    if not clean:
        raise HTTPException(status_code=400, detail=f"{label} is required")
    if clean in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", clean):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}. Use letters, numbers, dot, underscore, hyphen.",
        )
    return clean


def _detect_median_pitch_hz(audio_path: Path) -> float:
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Auto transpose requires librosa (install in API env). Error: {exc}",
        ) from exc

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if y is None or len(y) == 0:
        raise HTTPException(status_code=400, detail=f"Empty audio: {audio_path.name}")

    # YIN returns one f0 estimate per frame; keep only valid positive values.
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0 = np.asarray(f0, dtype=float)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect pitch on audio: {audio_path.name}",
        )
    return float(np.median(voiced))


def compute_auto_transpose_semitones(input_audio: Path, reference_audio: Path) -> int:
    import math

    input_hz = _detect_median_pitch_hz(input_audio)
    ref_hz = _detect_median_pitch_hz(reference_audio)
    semitones = 12 * math.log2(ref_hz / input_hz)
    return int(round(semitones))


def resolve_existing_path(raw: str, *, default_root: Path | None = None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        resolved = p
    else:
        candidate = (default_root / p).resolve() if default_root else None
        if candidate and candidate.exists():
            resolved = candidate
        else:
            resolved = (REPO_ROOT / p).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {resolved}")
    return resolved


def _subprocess_stream_to_log(*, command: list[str], log_file, job_id: str, env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        JOBS.append_log(job_id, line)
    return proc.wait()


def _command_arg_value(cmd: list[str], arg_name: str) -> str | None:
    try:
        idx = cmd.index(arg_name)
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def _build_convert_no_half_retry_cmd(cmd: list[str]) -> list[str] | None:
    if "scripts/convert.py" not in cmd:
        return None
    if "--no_half" in cmd:
        return None
    if (_command_arg_value(cmd, "--device") or "").lower() != "cuda":
        return None
    retry = cmd.copy()
    retry.append("--no_half")
    return retry


def build_train_cmd(req: "TrainRequest") -> list[str]:
    dataset_dir = (
        resolve_existing_path(req.dataset_dir)
        if req.dataset_dir
        else (DATASET_ROOT / sanitize_name(req.exp, "exp")).resolve()
    )
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_dir}")
    if not any(p.is_file() and _is_audio_filename(p.name) for p in dataset_dir.iterdir()):
        raise HTTPException(status_code=400, detail=f"Dataset has no supported audio files: {dataset_dir}")

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--dataset",
        str(dataset_dir),
        "--exp",
        req.exp,
        "--sr",
        req.sr,
        "--version",
        req.version,
        "--if_f0",
        str(req.if_f0),
        "--spk_id",
        str(req.spk_id),
        "--np",
        str(req.np),
        "--gpus",
        req.gpus,
        "--gpus_rmvpe",
        req.gpus_rmvpe,
        "--f0_method",
        req.f0_method,
        "--batch_size",
        str(req.batch_size),
        "--total_epoch",
        str(req.total_epoch),
        "--save_every_epoch",
        str(req.save_every_epoch),
        "--early_stop_patience",
        str(req.early_stop_patience),
        "--early_stop_min_delta",
        str(req.early_stop_min_delta),
        "--early_stop_metric",
        req.early_stop_metric,
        "--device",
        req.device,
    ]
    if req.pretrained_g:
        cmd.extend(["--pretrained_g", req.pretrained_g])
    if req.pretrained_d:
        cmd.extend(["--pretrained_d", req.pretrained_d])
    if req.is_half is not None:
        cmd.extend(["--is_half", "1" if req.is_half else "0"])
    if req.save_latest:
        cmd.append("--save_latest")
    if req.cache_gpu:
        cmd.append("--cache_gpu")
    if req.save_every_weights:
        cmd.append("--save_every_weights")
    if req.copy_to_models:
        cmd.append("--copy_to_models")
    if req.feature_retry_no_half:
        cmd.append("--feature_retry_no_half")
    if req.feature_fallback_cpu:
        cmd.append("--feature_fallback_cpu")
    if req.keep_failed_logs:
        cmd.append("--keep_failed_logs")
    return cmd


def build_convert_cmd(
    *,
    input_path: Path,
    output_path: Path,
    req: "ConvertParams",
) -> list[str]:
    model_path = resolve_existing_path(req.model, default_root=MODELS_ROOT)
    index_path = (
        resolve_existing_path(req.index, default_root=MODELS_ROOT)
        if req.index
        else None
    )

    cmd = [
        sys.executable,
        "scripts/convert.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(model_path),
        "--transpose",
        str(req.transpose),
        "--f0_method",
        req.f0_method,
        "--index_rate",
        str(req.index_rate),
        "--filter_radius",
        str(req.filter_radius),
        "--resample_sr",
        str(req.resample_sr),
        "--rms_mix_rate",
        str(req.rms_mix_rate),
        "--protect",
        str(req.protect),
        "--device",
        req.device,
    ]
    if index_path:
        cmd.extend(["--index", str(index_path)])
    if req.no_half:
        cmd.append("--no_half")
    return cmd


@dataclass
class JobRecord:
    job_id: str
    kind: Literal["train", "convert"]
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    command: list[str] = field(default_factory=list)
    return_code: int | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": self.command,
            "return_code": self.return_code,
            "error": self.error,
            "result": self.result,
            "log_path": self.log_path,
            "log_tail": list(self.log_tail),
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, kind: Literal["train", "convert"], command: list[str], result: dict[str, Any] | None = None) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        log_file = JOBS_ROOT / f"{job_id}.log"
        job = JobRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            created_at=utc_now_iso(),
            command=command,
            result=result or {},
            log_path=str(log_file),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.log_tail.append(line.rstrip("\n"))

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


JOBS = JobStore()
RUN_LOCK = threading.Lock()


def _run_job(job: JobRecord) -> None:
    JOBS.update(job.job_id, status="queued")
    log_file = Path(job.log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with RUN_LOCK:
        JOBS.update(job.job_id, status="running", started_at=utc_now_iso())
        env = os.environ.copy()
        env.setdefault("USE_LIBUV", "0")

        try:
            with log_file.open("w", encoding="utf-8", errors="replace") as lf:
                lf.write(f"[start] {utc_now_iso()}\n")
                lf.write(f"[cmd] {' '.join(job.command)}\n")
                lf.flush()

                rc = _subprocess_stream_to_log(
                    command=job.command,
                    log_file=lf,
                    job_id=job.job_id,
                    env=env,
                )
                retry_cmd = _build_convert_no_half_retry_cmd(job.command) if rc != 0 else None
                if rc != 0 and retry_cmd is not None:
                    lf.write(
                        f"[retry] convert failed rc={rc}; retrying once with --no_half\n"
                    )
                    lf.write(f"[cmd-retry] {' '.join(retry_cmd)}\n")
                    lf.flush()
                    JOBS.append_log(job.job_id, f"[retry] convert failed rc={rc}; retrying once with --no_half")
                    job.command = retry_cmd
                    result = dict(job.result)
                    result["convert_retry_no_half"] = True
                    JOBS.update(job.job_id, command=retry_cmd, result=result)
                    rc = _subprocess_stream_to_log(
                        command=retry_cmd,
                        log_file=lf,
                        job_id=job.job_id,
                        env=env,
                    )

                if rc != 0:
                    JOBS.update(
                        job.job_id,
                        status="failed",
                        return_code=rc,
                        finished_at=utc_now_iso(),
                        error=f"Process exited with code {rc}",
                    )
                    return

                JOBS.update(
                    job.job_id,
                    status="completed",
                    return_code=rc,
                    finished_at=utc_now_iso(),
                )
        except Exception as exc:
            JOBS.update(
                job.job_id,
                status="failed",
                finished_at=utc_now_iso(),
                error=str(exc),
            )
            try:
                with log_file.open("a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"\n[api-error] {exc}\n")
            except Exception:
                pass


def start_background_job(job: JobRecord) -> None:
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()


class TrainRequest(BaseModel):
    dataset_dir: str | None = None
    exp: str
    sr: Literal["32k", "40k", "48k"] = "40k"
    version: Literal["v1", "v2"] = "v2"
    if_f0: int = Field(default=1, ge=0, le=1)
    spk_id: int = 0
    np: int = 4
    gpus: str = "0"
    gpus_rmvpe: str = "0"
    f0_method: Literal["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"] = "rmvpe_gpu"
    batch_size: int = 4
    total_epoch: int = 100
    save_every_epoch: int = 5
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_metric: Literal["loss_mel", "loss_gen_all"] = "loss_mel"
    feature_retry_no_half: bool = True
    feature_fallback_cpu: bool = True
    keep_failed_logs: bool = True
    save_latest: bool = False
    cache_gpu: bool = False
    save_every_weights: bool = True
    copy_to_models: bool = True
    pretrained_g: str = ""
    pretrained_d: str = ""
    device: str = "cuda"
    is_half: bool | None = True


class ConvertParams(BaseModel):
    model: str
    index: str = ""
    transpose: int = 0
    auto_transpose: bool = False
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    device: str = "cuda"
    no_half: bool = False





app = FastAPI(title="RVC Minimal API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "repo_root": str(REPO_ROOT)}


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    return {
        "pth": sorted([p.name for p in MODELS_ROOT.glob("*.pth")]),
        "index": sorted([p.name for p in MODELS_ROOT.glob("*.index")]),
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, list[str]]:
    dirs = [p.name for p in DATASET_ROOT.iterdir() if p.is_dir()]
    return {"datasets": sorted(dirs)}


@app.post("/api/uploads/dataset")
async def upload_dataset(
    exp: str = Form(...),
    files: list[UploadFile] = File(...),
    clear_existing: bool = Form(True),
) -> dict[str, Any]:
    exp_name = sanitize_name(exp, "exp")
    target_dir = DATASET_ROOT / exp_name

    if clear_existing and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        filename = sanitize_name(Path(f.filename or "file.wav").name, "filename")
        dst = target_dir / filename
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No files uploaded")

    return {
        "message": "Dataset uploaded",
        "exp": exp_name,
        "dataset_dir": str(target_dir.resolve()),
        "files_saved": saved,
        "count": len(saved),
    }


@app.post("/api/train")
def start_train(req: TrainRequest) -> dict[str, Any]:
    req.exp = sanitize_name(req.exp, "exp")
    cmd = build_train_cmd(req)
    job = JOBS.create(kind="train", command=cmd, result={"exp": req.exp})
    start_background_job(job)
    return {"job_id": job.job_id, "status": "queued", "kind": "train"}


@app.post("/api/convert")
async def start_convert(
    file: UploadFile = File(...),
    reference_file: UploadFile | None = File(None),
    model: str = Form(...),
    index: str = Form(""),
    transpose: int = Form(0),
    auto_transpose: bool = Form(False),
    auto_transpose_fallback_to_zero: bool = Form(True),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    filter_radius: int = Form(3),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
    device: str = Form("cuda"),
    no_half: bool = Form(False),
) -> dict[str, Any]:
    ext = Path(file.filename or "input.wav").suffix or ".wav"
    upload_id = uuid.uuid4().hex[:12]
    input_path = UPLOADS_ROOT / f"{upload_id}{ext}"
    output_path = OUTPUTS_ROOT / f"{upload_id}_converted.wav"
    reference_path: Path | None = None

    with input_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    if not _is_audio_filename(file.filename or ""):
        # Keep permissive behavior (do not block), but record it for debugging.
        input_name_warning = f"input extension may be unsupported: {file.filename}"
    else:
        input_name_warning = ""

    if reference_file is not None:
        ref_ext = Path(reference_file.filename or "reference.wav").suffix or ".wav"
        reference_path = UPLOADS_ROOT / f"{upload_id}_reference{ref_ext}"
        with reference_path.open("wb") as out:
            shutil.copyfileobj(reference_file.file, out)

    resolved_transpose = transpose
    auto_transpose_info: dict[str, Any] = {"enabled": bool(auto_transpose), "applied": False}
    if auto_transpose:
        if reference_path is None:
            # Requested behavior for now: if auto flag is enabled but no reference provided, fallback to 0.
            resolved_transpose = 0
            auto_transpose_info["reason"] = "reference_file missing; fallback transpose=0"
        else:
            try:
                resolved_transpose = compute_auto_transpose_semitones(input_path, reference_path)
                auto_transpose_info.update(
                    {
                        "applied": True,
                        "reference_file": reference_file.filename,
                        "computed_transpose": resolved_transpose,
                    }
                )
            except Exception as exc:
                if auto_transpose_fallback_to_zero:
                    resolved_transpose = 0
                    auto_transpose_info.update(
                        {
                            "applied": False,
                            "fallback_to_zero": True,
                            "error": str(exc),
                        }
                    )
                else:
                    raise

    params = ConvertParams(
        model=model,
        index=index,
        transpose=resolved_transpose,
        auto_transpose=auto_transpose,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        device=device,
        no_half=no_half,
    )
    cmd = build_convert_cmd(input_path=input_path, output_path=output_path, req=params)
    job = JOBS.create(
        kind="convert",
        command=cmd,
        result={
            "input_path": str(input_path),
            "output_path": str(output_path),
            "original_filename": file.filename,
            "reference_path": (str(reference_path) if reference_path else ""),
            "transpose_requested": transpose,
            "transpose_applied": resolved_transpose,
            "auto_transpose": auto_transpose_info,
            "warnings": ([input_name_warning] if input_name_warning else []),
        },
    )
    start_background_job(job)
    return {
        "job_id": job.job_id,
        "status": "queued",
        "kind": "convert",
        "transpose_applied": resolved_transpose,
        "download_url": f"/api/jobs/{job.job_id}/download",
    }


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": JOBS.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return JOBS.get(job_id).to_dict()


@app.get("/api/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_job_logs(job_id: str) -> str:
    job = JOBS.get(job_id)
    log_path = Path(job.log_path)
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.get("/api/jobs/{job_id}/download")
def download_convert_output(job_id: str):
    job = JOBS.get(job_id)
    if job.kind != "convert":
        raise HTTPException(status_code=400, detail="Download only available for convert jobs")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job status is {job.status}")

    output_path = Path(job.result.get("output_path", ""))
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(str(output_path), filename=output_path.name, media_type="audio/wav")
