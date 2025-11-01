import os, uuid, logging, tempfile, shutil, subprocess, atexit, requests, threading, base64, sys
from flask import Flask, request, jsonify
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

PYTHON_SERVICE_KEY = os.environ.get('PYTHON_SERVICE_KEY')
if not PYTHON_SERVICE_KEY:
    raise Exception('PYTHON_SERVICE_KEY environment variable required')

FFMPEG_BIN = os.environ.get('FFMPEG_PATH', 'ffmpeg')
UPLOAD_FOLDER = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_output_")

logger.info('=' * 80)
logger.info('🚀 PYTHON SERVICE STARTING')
logger.info(f"📁 Upload: {UPLOAD_FOLDER}")
logger.info(f"📁 Output: {OUTPUT_FOLDER}")
logger.info(f"🐍 sys.executable: {sys.executable}")
logger.info(f"🎬 FFMPEG: {FFMPEG_BIN}")
logger.info(f"🔥 Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
logger.info('=' * 80)

def cleanup_temp_dirs():
    try:
        if os.path.exists(UPLOAD_FOLDER): shutil.rmtree(UPLOAD_FOLDER)
        if os.path.exists(OUTPUT_FOLDER): shutil.rmtree(OUTPUT_FOLDER)
        logger.info("🧹 Temp directories cleaned up")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
atexit.register(cleanup_temp_dirs)

def process_job_async(job_id, input_path, wav_path, callback_url, project_id, base44_app_id):
    try:
        logger.info(f"🎸 [{job_id}] Worker start")

        # Demucs separation (uses current venv)
        demucs_cmd = [sys.executable, '-m', 'demucs', '-o', OUTPUT_FOLDER, '-n', 'mdx_extra', '--segment', '10', '--jobs', '2', wav_path]
        logger.info(f"🔧 [{job_id}] Demucs: {' '.join(demucs_cmd)}")
        res = subprocess.run(demucs_cmd, capture_output=True, text=True)
        logger.debug(f"STDOUT: {res.stdout}"); logger.debug(f"STDERR: {res.stderr}")
        if res.returncode != 0: raise Exception(f"Demucs failed: {res.stderr}")

        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        out_dir = os.path.join(OUTPUT_FOLDER, 'mdx_extra', base_name)
        if not os.path.exists(out_dir): raise Exception(f"Output dir not found: {out_dir}")

        stem_files = {
            'vocals': os.path.join(out_dir, 'vocals.wav'),
            'drums':  os.path.join(out_dir, 'drums.wav'),
            'bass':   os.path.join(out_dir, 'bass.wav'),
            'other':  os.path.join(out_dir, 'other.wav'),
        }
        stems_b64 = {}
        for name, path in stem_files.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    stems_b64[name] = base64.b64encode(f.read()).decode('utf-8')
                    logger.info(f"✅ [{job_id}] Encoded {name}")
            else:
                logger.warning(f"⚠️ [{job_id}] Missing stem: {name}")

        if not stems_b64: raise Exception("No stems generated")

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'}
        payload = {'project_id': project_id, 'success': True, 'stems_base64': stems_b64}
        logger.info(f"📞 [{job_id}] Callback → {callback_url}")
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=120)
        logger.info(f"📥 [{job_id}] Callback status: {resp.status_code}")
        if resp.status_code != 200: raise Exception(f"Callback failed: {resp.status_code} {resp.text[:200]}")

        # Cleanup
        try:
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(wav_path): os.remove(wav_path)
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
        except Exception as e:
            logger.warning(f"Cleanup warn: {e}")

        logger.info(f"🎉 [{job_id}] Done")
    except Exception as e:
        logger.error(f"❌ [{job_id}] FAIL: {e}")
        try:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'}
            requests.post(callback_url, json={'project_id': project_id, 'success': False, 'error': str(e)}, headers=headers, timeout=30)
        except Exception as e2:
            logger.error(f"Callback error: {e2}")

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    job_id = str(uuid.uuid4())[:8]
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_app_id = request.form.get('base44_app_id')
        if not callback_url or not project_id or not base44_app_id:
            return jsonify({'error': 'Missing callback_url, project_id, or base44_app_id'}), 400

        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
        file.save(input_path)
        wav_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")

        ffmpeg_cmd = [FFMPEG_BIN, '-threads', '0', '-i', input_path, '-ar', '44100', '-ac', '2', '-loglevel', 'error', '-y', wav_path]
        logger.info(f"🔧 [{job_id}] FFmpeg: {' '.join(ffmpeg_cmd)}")
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return jsonify({'error': f'FFmpeg failed: {res.stderr}'}), 500

        thread = threading.Thread(target=process_job_async, args=(job_id, input_path, wav_path, callback_url, project_id, base44_app_id), name=f"job-{job_id}", daemon=True)
        thread.start()
        return jsonify({'success': True, 'message': 'Job started', 'job_id': job_id, 'project_id': project_id}), 202

    except Exception as e:
        try:
            if 'callback_url' in locals() and 'project_id' in locals():
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'}
                requests.post(callback_url, json={'project_id': project_id, 'success': False, 'error': str(e)}, headers=headers, timeout=30)
        except Exception:
            pass
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy','service':'audio-processing-service','python_version':sys.version,'pytorch_version':torch.__version__,'cuda_available':torch.cuda.is_available()}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)





