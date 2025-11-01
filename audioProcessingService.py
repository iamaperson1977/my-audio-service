# audioProcessingService.py
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS (stacked at top)
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import uuid
import time
import base64
import atexit
import shutil
import tempfile
import logging
import threading
import subprocess
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS / ENV (stacked at top)
# ──────────────────────────────────────────────────────────────────────────────
PYTHON_SERVICE_KEY = os.getenv("PYTHON_SERVICE_KEY")  # REQUIRED
if not PYTHON_SERVICE_KEY:
    raise RuntimeError("PYTHON_SERVICE_KEY env var is required")

FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")       # optional, default 'ffmpeg'

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING (stacked at top)
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger("audio-service")

# ──────────────────────────────────────────────────────────────────────────────
# APP / PATHS (top)
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

UPLOAD_DIR = tempfile.mkdtemp(prefix="upload_")
OUTPUT_DIR = tempfile.mkdtemp(prefix="output_")

# Torch info for /health (optional)
try:
    import torch
    TORCH_VER = torch.__version__
    CUDA_OK = bool(torch.cuda.is_available())
except Exception:
    TORCH_VER = "unavailable"
    CUDA_OK = False

logger.info("🚀 Python audio service starting")
logger.info(f"📁 Upload: {UPLOAD_DIR}")
logger.info(f"📁 Output: {OUTPUT_DIR}")
logger.info(f"🐍 Python: {sys.version.split()[0]}")
logger.info(f"🔥 Torch: {TORCH_VER} | CUDA: {CUDA_OK}")
logger.info(f"🎬 FFmpeg: {FFMPEG_BIN}")

# ──────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────────────────────
def _cleanup():
    for p in (UPLOAD_DIR, OUTPUT_DIR):
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.warning("Cleanup failed for %s: %s", p, e)

atexit.register(_cleanup)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _run(cmd: list[str], timeout: int) -> None:
    """Run a command and raise with useful context on failure."""
    logger.info("⚙️  Exec: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.stdout:
        logger.debug("stdout: %s", res.stdout[:2000])
    if res.stderr:
        logger.debug("stderr: %s", res.stderr[:2000])
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {res.stderr}")

def _send_callback(
    url: str,
    base44_app_id: str,
    payload: Dict,
    timeout_s: int = 90,
    max_attempts: int = 4,
    backoff_base: float = 0.75
) -> requests.Response:
    """POST JSON with required headers; retry on transient failures."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
        "Base44-App-Id": base44_app_id,   # REQUIRED by Base44
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            if resp.status_code < 500:
                return resp
            logger.warning("🌐 Callback %s/%s → %s (retrying)", attempt, max_attempts, resp.status_code)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("🌐 Callback attempt %s/%s failed: %s", attempt, max_attempts, exc)
        time.sleep(backoff_base * (2 ** (attempt - 1)))
    if last_exc:
        raise last_exc
    raise RuntimeError("Callback failed without exception")

# ──────────────────────────────────────────────────────────────────────────────
# WORKER: Demucs → stream partials → final "done"
# ──────────────────────────────────────────────────────────────────────────────
def _worker(job_id: str, src: str, wav: str, cb_url: str, project_id: str, app_id: str):
    try:
        logger.info("🎸 [%s] Starting Demucs", job_id)
        demucs = [
            sys.executable, "-m", "demucs",
            "-o", OUTPUT_DIR,
            "-n", "mdx_extra",
            "--segment", "10",
            "--jobs", "2",
            wav
        ]
        _run(demucs, timeout=1800)

        base_name = os.path.splitext(os.path.basename(wav))[0]
        out_dir = os.path.join(OUTPUT_DIR, "mdx_extra", base_name)
        if not os.path.isdir(out_dir):
            raise RuntimeError(f"Demucs output dir not found: {out_dir}")
        logger.info("📂 [%s] Output: %s | %s", job_id, out_dir, os.listdir(out_dir))

        stems = ["vocals", "drums", "bass", "other"]
        sent = 0
        for part in stems:
            path = os.path.join(out_dir, f"{part}.wav")
            if not os.path.exists(path):
                logger.warning("⚠️  [%s] Missing stem: %s", job_id, part)
                continue
            with open(path, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload = {
                "project_id": project_id,
                "mode": "partial",
                "part": part,
                "data_base64": data_b64
            }
            r = _send_callback(cb_url, app_id, payload)
            logger.info("📤 [%s] Sent partial '%s' → %s", job_id, part, r.status_code)
            if r.status_code != 200:
                raise RuntimeError(f"Partial '{part}' failed: {r.status_code} {r.text[:200]}")
            sent += 1

        if sent == 0:
            raise RuntimeError("No stems generated")

        # Finalize
        done_payload = {"project_id": project_id, "mode": "done"}
        r = _send_callback(cb_url, app_id, done_payload)
        logger.info("🏁 [%s] Done → %s", job_id, r.status_code)
        if r.status_code != 200:
            raise RuntimeError(f"Finalize failed: {r.status_code} {r.text[:200]}")

    except Exception as e:
        logger.error("❌ [%s] Job failed: %s", job_id, e)
        try:
            err = {"project_id": project_id, "success": False, "error": str(e)}
            _send_callback(cb_url, app_id, err, timeout_s=30)
        except Exception as ce:
            logger.error("Error callback failed: %s", ce)
    finally:
        # Best-effort cleanup
        for p in (src, wav, os.path.join(OUTPUT_DIR, "mdx_extra", os.path.splitext(os.path.basename(wav))[0])):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/separate_stems", methods=["POST"])
def separate_stems():
    job_id = str(uuid.uuid4())[:8]
    try:
        f = request.files.get("file")
        cb_url = request.form.get("callback_url")
        project_id = request.form.get("project_id")
        app_id = request.form.get("base44_app_id")

        if not all([f, cb_url, project_id, app_id]):
            return jsonify({"error": "Missing file, callback_url, project_id, or base44_app_id"}), 400

        u = urlparse(cb_url)
        if u.scheme not in ("http", "https") or not u.netloc:
            return jsonify({"error": "Invalid callback_url"}), 400

        src = os.path.join(UPLOAD_DIR, f"{job_id}_{f.filename}")
        f.save(src)

        wav = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
        ff = [FFMPEG_BIN, "-threads", "0", "-i", src, "-ar", "44100", "-ac", "2", "-loglevel", "error", "-y", wav]
        _run(ff, timeout=600)

        threading.Thread(
            target=_worker,
            args=(job_id, src, wav, cb_url, project_id, app_id),
            name=f"job-{job_id}",
            daemon=True
        ).start()

        return jsonify({"success": True, "job_id": job_id, "project_id": project_id}), 202

    except Exception as e:
        # Best-effort immediate error callback
        try:
            if "cb_url" in locals() and "project_id" in locals() and "app_id" in locals():
                _send_callback(cb_url, app_id, {"project_id": project_id, "success": False, "error": str(e)}, timeout_s=30)
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "python_version": sys.version.split()[0],
        "torch_version": TORCH_VER,
        "cuda_available": CUDA_OK
    }), 200

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("🌐 Serving on :%s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)









