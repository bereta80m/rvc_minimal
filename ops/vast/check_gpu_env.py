import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _import_status(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        return True, "ok"
    except Exception as exc:  # pragma: no cover - diagnostic script
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    checks = {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "commands": {
            "ffmpeg": shutil.which("ffmpeg") or "",
            "nvidia-smi": shutil.which("nvidia-smi") or "",
            "git": shutil.which("git") or "",
        },
        "imports": {},
        "write_access": {},
    }

    for mod in ["numpy", "sklearn", "faiss", "librosa", "soundfile", "ffmpeg", "av"]:
        ok, msg = _import_status(mod)
        checks["imports"][mod] = {"ok": ok, "detail": msg}

    torch_info: dict[str, object] = {}
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        torch_info["version"] = getattr(torch, "__version__", "")
        torch_info["cuda_available"] = cuda_ok
        torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
        torch_info["device_count"] = int(torch.cuda.device_count()) if cuda_ok else 0
        if cuda_ok:
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
            torch_info["devices"] = devices
    except Exception as exc:  # pragma: no cover - diagnostic script
        torch_info["error"] = f"{type(exc).__name__}: {exc}"
    checks["torch"] = torch_info

    for rel in ["dataset_raw", "logs", "models", "api_data", "exports"]:
        p = repo_root / rel
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test_tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            checks["write_access"][rel] = {"ok": True, "path": str(p)}
        except Exception as exc:  # pragma: no cover - diagnostic script
            checks["write_access"][rel] = {
                "ok": False,
                "path": str(p),
                "detail": f"{type(exc).__name__}: {exc}",
            }

    print(json.dumps(checks, indent=2))

    imports_ok = all(v["ok"] for v in checks["imports"].values())
    cuda_ok = bool(checks.get("torch", {}).get("cuda_available"))
    if not imports_ok:
        return 2
    if not cuda_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
