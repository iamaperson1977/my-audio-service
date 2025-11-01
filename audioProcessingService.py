import os
import uuid
import logging
import tempfile
import shutil
import subprocess
import atexit
import requests
import threading
import base64
import sys
from flask import Flask, request, jsonify
import torch

# Configure logging with MAXIMUM detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the service key for authenticating callbacks
PYTHON_SERVICE_KEY = os.environ.get('PYTHON_SERVICE_KEY')
if not PYTHON_SERVICE_KEY:
    logger.error('❌ PYTHON_SERVICE_KEY not set!')
    raise Exception('PYTHON_SERVICE_KEY environment variable required')

# Get FFmpeg path (use env var if set, otherwise default to 'ffmpeg')
FFMPEG_BIN = os.environ.get('FFMPEG_PATH', 'ffmpeg')

# Create temp directories
UPLOAD_FOLDER = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_output_")

logger.info('=' * 80)
logger.info('🚀 PYTHON SERVICE STARTING')
logger.info('=' * 80)
logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
logger.info(f"📁 Output folder: {OUTPUT_FOLDER}")
logger.info(f"🐍 Python version: {sys.version}")
logger.info(f"🐍 sys.executable: {sys.executable}")
logger.info(f"📂 System PATH: {os.environ.get('PATH', '(not set)')}")
logger.info(f"🎬 FFmpeg binary: {FFMPEG_BIN}")
logger.info(f"🔥 PyTorch version: {torch.__version__}")
logger.info(f"💻 CUDA available: {torch.cuda.is_available()}")
logger.info(f"🔑 PYTHON_SERVICE_KEY is set: {bool(PYTHON_SERVICE_KEY)}")
logger.info('=' * 80)

# Cleanup on exit
def cleanup_temp_dirs():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        logger.info("🧹 Temp directories cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup: {e}")

atexit.register(cleanup_temp_dirs)

def process_job_async(job_id, input_path, wav_path, callback_url, project_id, base44_app_id):
    """Background worker function - returns stems as base64 to callback"""
    try:
        logger.info(f"🎸 [{job_id}] Background worker started")
        logger.info(f"🆔 [{job_id}] Received base44_app_id: {base44_app_id}")
        
        # Run Demucs stem separation with OPTIMIZED SETTINGS
        logger.info('=' * 80)
        logger.info(f"🎸 [{job_id}] STARTING DEMUCS STEM SEPARATION (OPTIMIZED)")
        logger.info('=' * 80)
        
        # Use sys.executable to guarantee correct venv
        demucs_cmd = [
            sys.executable, '-m', 'demucs',
            '-o', OUTPUT_FOLDER,
            '-n', 'mdx_extra',
            '--segment', '10',
            '--jobs', '2',
            wav_path
        ]
        logger.info(f"🔧 [{job_id}] Demucs command: {' '.join(demucs_cmd)}")
        logger.info(f"⚡ [{job_id}] Using mdx_extra model for faster processing")
        
        logger.info(f"⏳ [{job_id}] Running Demucs (estimated: 30-90 seconds)...")
        result = subprocess.run(demucs_cmd, capture_output=True, text=True)
        
        logger.debug(f"📤 [{job_id}] Demucs stdout: {result.stdout}")
        logger.debug(f"📤 [{job_id}] Demucs stderr: {result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"❌ [{job_id}] Demucs failed")
            logger.error(f"   Return code: {result.returncode}")
            logger.error(f"   Stderr: {result.stderr}")
            raise Exception(f"Demucs failed: {result.stderr}")
        
        logger.info(f"✅ [{job_id}] Demucs separation complete!")
        
        # Find output directory (Demucs creates mdx_extra/audio_name/)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_dir = os.path.join(OUTPUT_FOLDER, 'mdx_extra', base_name)
        
        logger.info(f"📂 [{job_id}] Looking for output in: {output_dir}")
        
        if not os.path.exists(output_dir):
            logger.error(f"❌ [{job_id}] Output directory not found: {output_dir}")
            logger.error(f"   Contents of OUTPUT_FOLDER: {os.listdir(OUTPUT_FOLDER)}")
            raise Exception(f"Output directory not found: {output_dir}")
        
        logger.info(f"✅ [{job_id}] Output directory found")
        logger.info(f"📂 [{job_id}] Contents: {os.listdir(output_dir)}")
        
        # Expected stems from Demucs mdx_extra model
        stem_files = {
            'vocals': os.path.join(output_dir, 'vocals.wav'),
            'drums': os.path.join(output_dir, 'drums.wav'),
            'bass': os.path.join(output_dir, 'bass.wav'),
            'other': os.path.join(output_dir, 'other.wav')
        }
        
        # Check which stems exist and encode to base64
        logger.info('=' * 80)
        logger.info(f"🎼 [{job_id}] ENCODING STEMS TO BASE64")
        logger.info('=' * 80)
        
        stems_base64 = {}
        
        for stem_name, stem_path in stem_files.items():
            if os.path.exists(stem_path):
                size = os.path.getsize(stem_path)
                logger.info(f"✅ [{job_id}] {stem_name}: {size} bytes ({size / 1024 / 1024:.2f} MB)")
                
                # Read file and encode to base64
                with open(stem_path, 'rb') as f:
                    file_data = f.read()
                    encoded = base64.b64encode(file_data).decode('utf-8')
                    stems_base64[stem_name] = encoded
                    logger.info(f"   Encoded {stem_name} to base64 ({len(encoded)} chars)")
            else:
                logger.warning(f"⚠️ [{job_id}] {stem_name} not found at {stem_path}")
        
        if not stems_base64:
            logger.error(f"❌ [{job_id}] No stems were created!")
            raise Exception("No stems were generated by Demucs")
        
        logger.info(f"✅ [{job_id}] Encoded {len(stems_base64)} stems to base64")
        
        # Send callback with stems
        logger.info('=' * 80)
        logger.info(f"📞 [{job_id}] SENDING CALLBACK TO BASE44")
        logger.info('=' * 80)
        logger.info(f"🔗 [{job_id}] Callback URL: {callback_url}")
        
        callback_payload = {
            'project_id': project_id,
            'success': True,
            'stems_base64': stems_base64
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'
        }
        
        logger.info(f"🔑 [{job_id}] Using PYTHON_SERVICE_KEY for authentication")
        logger.info(f"📤 [{job_id}] Callback headers: {list(headers.keys())}")
        logger.info(f"📦 [{job_id}] Sending {len(stems_base64)} stems in callback")
        
        callback_response = requests.post(
            callback_url,
            json=callback_payload,
            headers=headers,
            timeout=120
        )
        
        logger.info(f"📥 [{job_id}] Callback response status: {callback_response.status_code}")
        logger.info(f"📥 [{job_id}] Callback response text: {callback_response.text[:500]}")
        
        if callback_response.status_code != 200:
            logger.error(f"❌ [{job_id}] Callback failed!")
            logger.error(f"   Status: {callback_response.status_code}")
            logger.error(f"   Response: {callback_response.text}")
            raise Exception(f"Callback failed: {callback_response.status_code}")
        
        logger.info(f"✅ [{job_id}] Callback successful!")
        
        # Cleanup files
        logger.info(f"🧹 [{job_id}] Cleaning up temporary files...")
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            logger.info(f"✅ [{job_id}] Cleanup complete")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ [{job_id}] Cleanup warning: {cleanup_error}")
        
        logger.info('=' * 80)
        logger.info(f"🎉 [{job_id}] JOB COMPLETED SUCCESSFULLY")
        logger.info('=' * 80)
        
    except Exception as error:
        logger.error('=' * 80)
        logger.error(f"❌ [{job_id}] JOB FAILED")
        logger.error('=' * 80)
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        
        # Send error callback
        try:
            logger.info(f"📞 [{job_id}] Sending error callback...")
            
            error_payload = {
                'project_id': project_id,
                'success': False,
                'error': str(error)
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'
            }
            
            callback_response = requests.post(
                callback_url,
                json=error_payload,
                headers=headers,
                timeout=30
            )
            
            logger.info(f"✅ [{job_id}] Error callback sent (status: {callback_response.status_code})")
            
        except Exception as callback_error:
            logger.error(f"❌ [{job_id}] Failed to send error callback: {callback_error}")

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """Endpoint to start stem separation job"""
    job_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info('=' * 80)
        logger.info(f'🎵 [{job_id}] NEW SEPARATION REQUEST RECEIVED')
        logger.info('=' * 80)
        
        # Get uploaded file
        if 'file' not in request.files:
            logger.error(f"❌ [{job_id}] No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_app_id = request.form.get('base44_app_id')
        
        logger.info(f"📂 [{job_id}] File: {file.filename}")
        logger.info(f"🔗 [{job_id}] Callback URL: {callback_url}")
        logger.info(f"🆔 [{job_id}] Project ID: {project_id}")
        logger.info(f"🆔 [{job_id}] Base44 App ID: {base44_app_id}")
        
        if not callback_url or not project_id or not base44_app_id:
            logger.error(f"❌ [{job_id}] Missing required parameters")
            return jsonify({'error': 'Missing callback_url, project_id, or base44_app_id'}), 400
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"💾 [{job_id}] File saved: {input_path} ({file_size} bytes, {file_size / 1024 / 1024:.2f} MB)")
        
        # Convert to WAV using FFmpeg
        wav_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")
        
        logger.info(f"🎬 [{job_id}] Converting to WAV with FFmpeg...")
        
        ffmpeg_cmd = [
            FFMPEG_BIN,
            '-threads', '0',
            '-i', input_path,
            '-ar', '44100',
            '-ac', '2',
            '-loglevel', 'error',
            '-y',
            wav_path
        ]
        
        logger.info(f"🔧 [{job_id}] FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"❌ [{job_id}] FFmpeg conversion failed")
            logger.error(f"   Return code: {result.returncode}")
            logger.error(f"   Stderr: {result.stderr}")
            return jsonify({'error': f'FFmpeg failed: {result.stderr}'}), 500
        
        wav_size = os.path.getsize(wav_path)
        logger.info(f"✅ [{job_id}] WAV created: {wav_path} ({wav_size} bytes, {wav_size / 1024 / 1024:.2f} MB)")
        
        # Start background processing thread
        logger.info(f"🚀 [{job_id}] Starting background processing thread...")
        
        thread = threading.Thread(
            target=process_job_async,
            args=(job_id, input_path, wav_path, callback_url, project_id, base44_app_id)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"✅ [{job_id}] Background thread started")
        logger.info('=' * 80)
        
        return jsonify({
            'success': True,
            'message': 'Job started',
            'job_id': job_id,
            'project_id': project_id
        }), 202
        
    except Exception as error:
        logger.error('=' * 80)
        logger.error(f"❌ [{job_id}] REQUEST HANDLER ERROR")
        logger.error('=' * 80)
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        
        # Try to send error callback if we have the info
        try:
            if 'callback_url' in locals() and 'project_id' in locals():
                logger.info(f"📞 [{job_id}] Sending immediate error callback...")
                
                error_payload = {
                    'project_id': project_id,
                    'success': False,
                    'error': str(error)
                }
                
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {PYTHON_SERVICE_KEY}'
                }
                
                requests.post(callback_url, json=error_payload, headers=headers, timeout=30)
                logger.info(f"✅ [{job_id}] Error callback sent")
        except Exception as callback_error:
            logger.error(f"❌ [{job_id}] Failed to send error callback: {callback_error}")
        
        return jsonify({
            'error': str(error),
            'error_type': type(error).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-service',
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"🌐 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)




