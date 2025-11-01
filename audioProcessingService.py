# audioProcessingService.py
import os
import sys
import uuid
import base64
import atexit
import shutil
import tempfile
import logging
import threading
import subprocess
from typing import Dict

import requests
from flask import Flask, request, jsonify

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
log = logging.getLogger("audio-service")

# ────────────────────────────────────────────────────────────────────────────
# Env
# ────────────────────────────────────────────────────────────────────────────
PYTHON_SERVICE_KEY = os.environ.get("PYTHON_SERVICE_KEY")
BASE44_APP_ID = os.environ.get("BASE44_APP_ID", "")
FFMPEG_BIN = os.environ.get("FFMPEG_PATH", "ffmpeg")

if not PYTHON_SERVICE_KEY:
    raise RuntimeError("PYTHON_SERVICE_KEY env var is required")
if not BASE44_APP_ID:
    log.warning("BASE44_APP_ID not set (callbacks will be unauthorized on Base44)")

# ────────────────────────────────────────────────────────────────────────────
# Flask
# ────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_DIR = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_DIR = tempfile.mkdtemp(prefix="audio_output_")

def _cleanup():
    for p in (UPLOAD_DIR, OUTPUT_DIR):
        try:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            log.warning("Cleanup warning for %s: %s", p, e)

atexit.register(_cleanup)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def run(cmd: list[str]) -> None:
    """Run a shell command, raising on non-zero exit with useful context."""
    log.info("Exec: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        log.info("stdout: %s", res.stdout.strip()[:2000])
    if res.stderr:
        log.info("stderr: %s", res.stderr.strip()[:2000])
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {res.stderr}")

def send_callback(callback_url: str, payload: dict) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
        "Base44-App-Id": BASE44_APP_ID,
    }
    log.info("POST -> %s  keys=%s", callback_url, list(payload.keys()))
    resp = requests.post(callback_url, json=payload, headers=headers, timeout=120)
    log.info("Callback status: %s", resp.status_code)
    if resp.status_code != 200:
        log.error("Callback body: %s", resp.text[:500])
    return resp

def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ────────────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────────────
def worker(job_id: str, src_path: str, wav_path: str, callback_url: str, project_id: str):
    try:
        log.info("===== JOB %s START =====", job_id)
        # Demucs → mdx_extra (balanced speed/quality) into OUTPUT_DIR
        demucs_cmd = [
            sys.executable, "-m", "demucs",
            "-o", OUTPUT_DIR,
            "-n", "mdx_extra",
            "--segment", "10",
            "--jobs", "2",
            wav_path,
        ]
        run(demucs_cmd)

        base = os.path.splitext(os.path.basename(wav_path))[0]
        demucs_out = os.path.join(OUTPUT_DIR, "mdx_extra", base)
        if not os.path.isdir(demucs_out):
            raise RuntimeError(f"Demucs output missing: {demucs_out}")

        # Expected stems
        stems: Dict[str, str] = {
            "vocals": os.path.join(demucs_out, "vocals.wav"),
            "drums":  os.path.join(demucs_out, "drums.wav"),
            "bass":   os.path.join(demucs_out, "bass.wav"),
            "other":  os.path.join(demucs_out, "other.wav"),
        }

        # Send each stem as a separate (small) callback
        for name, path in stems.items():
            if not os.path.exists(path):
                log.warning("Stem missing: %s", name)
                continue
            b64 = encode_b64(path)
            resp = send_callback(callback_url, {
                "project_id": project_id,
                "success": True,
                "mode": "partial",
                "part": name,
                "data_base64": b64,
            })
            if resp.status_code != 200:
                raise RuntimeError(f"Partial callback failed for {name}: {resp.status_code}")

        # Final “done” signal (lets Base44 mark completed)
        done_resp = send_callback(callback_url, {
            "project_id": project_id,
            "success": True,
            "mode": "done"
        })
        if done_resp.status_code != 200:
            raise RuntimeError(f"Done callback failed: {done_resp.status_code}")

        log.info("===== JOB %s OK =====", job_id)

    except Exception as e:
        log.exception("JOB %s FAILED: %s", job_id, e)
        # Best-effort error callback
        try:
            send_callback(callback_url, {
                "project_id": project_id,
                "success": False,
                "error": str(e),
            })
        except Exception as e2:
            log.error("Failed to send error callback: %s", e2)
    finally:
        # local cleanup
        for p in (src_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────
@app.route("/separate_stems", methods=["POST"])
def separate_stems():
    job_id = str(uuid.uuid4())[:8]
    try:
        f = request.files.get("file")
        callback_url = request.form.get("callback_url")
        project_id = request.form.get("project_id")
        app_id_seen = request.form.get("base44_app_id")  # for logging only

        if not f or not callback_url or not project_id:
            return jsonify({"error": "Missing file, callback_url, or project_id"}), 400

        log.info("JOB %s new request", job_id)
        log.info("Base44 app id (form): %s", app_id_seen)
        log.info("Upload dir: %s", UPLOAD_DIR)
        log.info("Output dir: %s", OUTPUT_DIR)
        log.info("FFmpeg: %s", FFMPEG_BIN)

        # Save original
        src_path = os.path.join(UPLOAD_DIR, f"{job_id}_{f.filename}")
        f.save(src_path)

        # Convert to wav (44.1kHz, stereo)
        wav_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
        ffmpeg_cmd = [
            FFMPEG_BIN, "-y",
            "-i", src_path,
            "-ar", "44100",
            "-ac", "2",
            "-loglevel", "error",
            wav_path,
        ]
        run(ffmpeg_cmd)

        # Background process
        th = threading.Thread(
            target=worker,
            args=(job_id, src_path, wav_path, callback_url, project_id),
            name=f"job-{job_id}",
            daemon=True,
        )
        th.start()

        return jsonify({"success": True, "job_id": job_id, "project_id": project_id}), 202

    except Exception as e:
        log.exception("REQUEST FAILED: %s", e)
        # best effort immediate error callback if we can
        try:
            if request.form.get("callback_url") and request.form.get("project_id"):
                send_callback(request.form["callback_url"], {
                    "project_id": request.form["project_id"],
                    "success": False,
                    "error": str(e),
                })
        except Exception:
            pass
        return jsonify({"error": str(e), "error_type": type(e).__name__}), 500

@app.route("/health", methods=["GET"])
def health():
    import torch  # lazy import; we only need the version
    return jsonify({
        "status": "healthy",
        "service": "audio-processing-service",
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": getattr(torch.cuda, "is_available", lambda: False)(),
    }), 200

# ────────────────────────────────────────────────────────────────────────────
# Local run
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting on 0.0.0.0:%s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)








