
import os
import sys
import uuid
import logging
import tempfile
import shutil
import subprocess
import atexit
import requests
import threading
import base64
from urllib.parse import urlparse

from flask import Flask, request, jsonify

# ── Optional torch (for health/logging only) ───────────────────────────────────
try:
    import torch  # noqa: F401
    TORCH_VER = torch.__version__
    CUDA_OK = bool(torch.cuda.is_available())
except Exception:
    torch = None
    TORCH_VER = "unavailable"
    CUDA_OK = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB cap

# ── Env & paths ────────────────────────────────────────────────────────────────
PYTHON_SERVICE_KEY = os.environ.get("PYTHON_SERVICE_KEY")
if not PYTHON_SERVICE_KEY:
    raise Exception("PYTHON_SERVICE_KEY environment variable required")

# If you set FFMPEG_PATH in Railway, it will use that; otherwise 'ffmpeg' on PATH
FFMPEG_BIN = os.environ.get("FFMPEG_PATH", "ffmpeg")

UPLOAD_FOLDER = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_output_")

logger.info("=" * 80)
logger.info("🚀 PYTHON SERVICE STARTING (STREAMING MODE)")
logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
logger.info(f"📁 Output folder: {OUTPUT_FOLDER}")
logger.info(f"🐍 Python: {sys.version.split()[0]}  exec: {sys.executable}")
logger.info(f"🎬 FFmpeg: {FFMPEG_BIN}")
logger.info(f"🔥 Torch: {TORCH_VER}  CUDA: {CUDA_OK}")
logger.info("=" * 80)

# ── Cleanup on exit ────────────────────────────────────────────────────────────
def cleanup_temp_dirs():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        logger.info("🧹 Temp directories cleaned up")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

atexit.register(cleanup_temp_dirs)

# ── Helper: Send partial stem ──────────────────────────────────────────────────
def send_partial_stem(job_id, callback_url, project_id, base44_app_id, part_name, wav_path):
    """
    Sends a single stem to Base44 as soon as it's ready.
    Uses mode="partial" for streaming.
    """
    try:
        logger.info(f"📤 [{job_id}] Sending {part_name} to callback...")
        
        # Read and encode stem
        with open(wav_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Send immediately
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
            "Base44-App-Id": base44_app_id,
        }
        payload = {
            "project_id": project_id,
            "mode": "partial",
            "part": part_name,
            "data_base64": b64_data,
        }
        
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=120)
        logger.info(f"✅ [{job_id}] {part_name} uploaded! Status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.warning(f"⚠️ [{job_id}] Callback returned {resp.status_code}: {resp.text[:200]}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [{job_id}] Failed to send {part_name}: {e}")
        return False

# ── Helper: Send finalization ──────────────────────────────────────────────────
def send_done_signal(job_id, callback_url, project_id, base44_app_id):
    """
    Sends final 'done' signal to mark processing as complete.
    """
    try:
        logger.info(f"🏁 [{job_id}] Sending completion signal...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
            "Base44-App-Id": base44_app_id,
        }
        payload = {
            "project_id": project_id,
            "mode": "done",
        }
        
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=60)
        logger.info(f"✅ [{job_id}] Completion signal sent! Status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.warning(f"⚠️ [{job_id}] Done callback returned {resp.status_code}")
        
    except Exception as e:
        logger.error(f"❌ [{job_id}] Failed to send done signal: {e}")

# ── Background worker (STREAMING MODE) ─────────────────────────────────────────
def process_job_async(job_id, input_path, wav_path, callback_url, project_id, base44_app_id):
    """
    Runs Demucs and sends each stem to Base44 AS SOON AS IT'S READY.
    This provides real-time progress feedback to the user!
    """
    try:
        logger.info(f"🎸 [{job_id}] Worker start (STREAMING)")
        logger.info(f"🔗 [{job_id}] Callback URL: {callback_url}")
        logger.info(f"🆔 [{job_id}] Project ID: {project_id}")
        logger.info(f"🆔 [{job_id}] Base44 App ID: {base44_app_id}")

        # 1) Separate stems with Demucs
        demucs_cmd = [
            sys.executable, "-m", "demucs",
            "-o", OUTPUT_FOLDER,
            "-n", "mdx_extra",
            "--segment", "10",
            "--jobs", "2",
            wav_path,
        ]
        logger.info(f"🔧 [{job_id}] Demucs: {' '.join(demucs_cmd)}")
        demucs_res = subprocess.run(
            demucs_cmd, capture_output=True, text=True, timeout=1800
        )
        logger.debug(f"📤 [{job_id}] Demucs stdout:\n{demucs_res.stdout}")
        logger.debug(f"📤 [{job_id}] Demucs stderr:\n{demucs_res.stderr}")
        if demucs_res.returncode != 0:
            raise Exception(f"Demucs failed: {demucs_res.stderr}")

        # 2) Demucs output dir: OUTPUT_FOLDER/mdx_extra/<basename>
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        out_dir = os.path.join(OUTPUT_FOLDER, "mdx_extra", base_name)
        logger.info(f"📂 [{job_id}] Checking output dir: {out_dir}")
        if not os.path.exists(out_dir):
            contents = "; ".join(os.listdir(OUTPUT_FOLDER))
            logger.error(f"OUTPUT_FOLDER contents: {contents}")
            raise Exception(f"Output dir not found: {out_dir}")

        # 3) STREAMING: Send each stem as soon as it's ready!
        stem_files = {
            "vocals": os.path.join(out_dir, "vocals.wav"),
            "drums":  os.path.join(out_dir, "drums.wav"),
            "bass":   os.path.join(out_dir, "bass.wav"),
            "other":  os.path.join(out_dir, "other.wav"),
        }
        
        uploaded_count = 0
        for stem_name, stem_path in stem_files.items():
            if os.path.exists(stem_path):
                success = send_partial_stem(
                    job_id, callback_url, project_id, base44_app_id, 
                    stem_name, stem_path
                )
                if success:
                    uploaded_count += 1
                    # Delete stem after successful upload to save space
                    try:
                        os.remove(stem_path)
                    except Exception as e:
                        logger.warning(f"⚠️ [{job_id}] Could not delete {stem_name}: {e}")
            else:
                logger.warning(f"⚠️ [{job_id}] Missing stem: {stem_name}")

        if uploaded_count == 0:
            raise Exception("No stems generated or uploaded successfully")

        # 4) Send completion signal
        send_done_signal(job_id, callback_url, project_id, base44_app_id)

        # 5) Cleanup
        try:
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(wav_path): os.remove(wav_path)
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
        except Exception as e:
            logger.warning(f"Cleanup warn: {e}")

        logger.info(f"🎉 [{job_id}] COMPLETE! Uploaded {uploaded_count} stems")

    except Exception as e:
        logger.error(f"❌ [{job_id}] FAIL: {e}")
        # Best-effort error callback
        try:
            err_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
                "Base44-App-Id": base44_app_id,
            }
            requests.post(
                callback_url,
                json={"project_id": project_id, "success": False, "error": str(e)},
                headers=err_headers,
                timeout=30,
            )
        except Exception as e2:
            logger.error(f"Callback error: {e2}")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/separate_stems", methods=["POST"])
def separate_stems():
    """
    Accepts:
      file: audio file (multipart)
      callback_url: Base44 function URL to POST results to
      project_id: your Base44 Project id
      base44_app_id: YOUR Base44 App Id (used for 'Base44-App-Id' header)

    Starts a background thread to process and callback.
    """
    job_id = str(uuid.uuid4())[:8]
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        callback_url = request.form.get("callback_url")
        project_id = request.form.get("project_id")
        base44_app_id = request.form.get("base44_app_id")

        # Validation
        if not callback_url or not project_id or not base44_app_id:
            return jsonify({"error": "Missing callback_url, project_id, or base44_app_id"}), 400
        u = urlparse(callback_url)
        if u.scheme not in ("http", "https") or not u.netloc:
            return jsonify({"error": "Invalid callback_url"}), 400

        # Save original upload
        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
        file.save(input_path)

        # Convert to WAV (44.1kHz, stereo)
        wav_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")
        ffmpeg_cmd = [
            FFMPEG_BIN, "-threads", "0",
            "-i", input_path,
            "-ar", "44100",
            "-ac", "2",
            "-loglevel", "error",
            "-y", wav_path,
        ]
        logger.info(f"🔧 [{job_id}] FFmpeg: {' '.join(ffmpeg_cmd)}")
        ffm_res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
        if ffm_res.returncode != 0:
            return jsonify({"error": f"FFmpeg failed: {ffm_res.stderr}"}), 500

        # Start background worker
        thread = threading.Thread(
            target=process_job_async,
            args=(job_id, input_path, wav_path, callback_url, project_id, base44_app_id),
            name=f"job-{job_id}",
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "message": "Job started (streaming mode)",
            "job_id": job_id,
            "project_id": project_id,
        }), 202

    except Exception as e:
        # Immediate error path: best-effort notify Base44
        try:
            if "callback_url" in locals() and "project_id" in locals() and "base44_app_id" in locals():
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {PYTHON_SERVICE_KEY}",
                    "Base44-App-Id": base44_app_id,
                }
                requests.post(
                    callback_url,
                    json={"project_id": project_id, "success": False, "error": str(e)},
                    headers=headers,
                    timeout=30,
                )
        except Exception:
            pass
        return jsonify({"error": str(e), "error_type": type(e).__name__}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "audio-processing-service (streaming)",
        "python_version": sys.version.split()[0],
        "torch_version": TORCH_VER,
        "cuda_available": CUDA_OK,
    }), 200

# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🌐 Starting Flask server on port {port} (STREAMING MODE)")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)











