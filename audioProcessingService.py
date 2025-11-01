import os
import uuid
import logging
import tempfile
import shutil
import subprocess
import atexit
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
import torch

# Configure logging with MAXIMUM detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create temp directories
UPLOAD_FOLDER = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_output_")

logger.info('=' * 80)
logger.info('🚀 PYTHON SERVICE STARTING')
logger.info('=' * 80)
logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
logger.info(f"📁 Output folder: {OUTPUT_FOLDER}")
logger.info(f"🐍 Python version: {os.sys.version}")
logger.info(f"🔥 PyTorch version: {torch.__version__}")
logger.info(f"💻 CUDA available: {torch.cuda.is_available()}")
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

def upload_stem_to_base44(stem_name, stem_path, base44_service_key, job_id):
    """
    OPTIMIZATION 3: Parallel upload function for concurrent stem uploads
    """
    try:
        logger.info(f"📤 [{job_id}] Uploading {stem_name}...")
        
        with open(stem_path, 'rb') as f:
            upload_response = requests.post(
                'https://app.base44.com/api/integrations/Core.UploadFile',
                headers={
                    'Authorization': f'Bearer {base44_service_key}'
                },
                files={'file': (f'{stem_name}.wav', f, 'audio/wav')},
                timeout=60  # 60 second timeout per upload
            )
        
        logger.debug(f"📥 [{job_id}] Upload response status for {stem_name}: {upload_response.status_code}")
        
        if upload_response.status_code != 200:
            logger.error(f"❌ [{job_id}] Upload failed for {stem_name}")
            logger.error(f"   Status: {upload_response.status_code}")
            logger.error(f"   Response: {upload_response.text}")
            raise Exception(f"Failed to upload {stem_name}: {upload_response.text}")
        
        file_url = upload_response.json().get('file_url')
        logger.info(f"✅ [{job_id}] {stem_name} uploaded: {file_url[:80]}...")
        
        return stem_name, file_url
    
    except Exception as e:
        logger.error(f"❌ [{job_id}] Error uploading {stem_name}: {str(e)}")
        raise

def process_job_async(job_id, input_path, wav_path, callback_url, project_id, base44_service_key, base44_app_id):
    """Background worker function with OPTIMIZATIONS 3 & 4"""
    try:
        logger.info(f"🎸 [{job_id}] Background worker started")
        
        # Run Demucs stem separation with OPTIMIZED SETTINGS
        logger.info('=' * 80)
        logger.info(f"🎸 [{job_id}] STARTING DEMUCS STEM SEPARATION (OPTIMIZED)")
        logger.info('=' * 80)
        
        # OPTIMIZATION 2: Use mdx_extra model (faster than htdemucs)
        # Add segment processing for better memory efficiency
        # Keep 4-stem separation for full instrument breakdown
        demucs_cmd = [
            'python3', '-m', 'demucs',
            '-o', OUTPUT_FOLDER,
            '-n', 'mdx_extra',  # CHANGED: Faster model
            '--segment', '10',   # ADDED: Process in 10-second segments (faster, less memory)
            '--jobs', '2',       # ADDED: Use 2 parallel jobs if CPU allows
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
        
        # Check which stems exist
        logger.info('=' * 80)
        logger.info(f"🎼 [{job_id}] VERIFYING STEMS")
        logger.info('=' * 80)
        for stem_name, stem_path in stem_files.items():
            exists = os.path.exists(stem_path)
            logger.info(f"{'✅' if exists else '❌'} [{job_id}] {stem_name}: {exists}")
            if exists:
                size = os.path.getsize(stem_path)
                logger.info(f"   Size: {size} bytes ({size / 1024 / 1024:.2f} MB)")
        
        # OPTIMIZATION 3: PARALLEL STEM UPLOADS
        logger.info('=' * 80)
        logger.info(f"☁️ [{job_id}] UPLOADING STEMS TO BASE44 (PARALLEL)")
        logger.info('=' * 80)
        
        stems_urls = {}
        upload_tasks = []
        
        # Create list of stems to upload (only existing files)
        for stem_name, stem_path in stem_files.items():
            if os.path.exists(stem_path):
                upload_tasks.append((stem_name, stem_path))
            else:
                logger.warning(f"⚠️ [{job_id}] Missing stem: {stem_name}, skipping")
        
        # Upload stems in parallel using ThreadPoolExecutor
        logger.info(f"🚀 [{job_id}] Starting {len(upload_tasks)} parallel uploads...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all upload tasks
            future_to_stem = {
                executor.submit(upload_stem_to_base44, stem_name, stem_path, base44_service_key, job_id): stem_name
                for stem_name, stem_path in upload_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_stem):
                stem_name = future_to_stem[future]
                try:
                    returned_stem_name, file_url = future.result()
                    stems_urls[returned_stem_name] = file_url
                except Exception as e:
                    logger.error(f"❌ [{job_id}] Upload failed for {stem_name}: {str(e)}")
                    raise
        
        logger.info(f"✅ [{job_id}] All stems uploaded successfully!")
        logger.info(f"📊 [{job_id}] Total stems: {len(stems_urls)}")
        
        # Cleanup
        logger.info(f"🧹 [{job_id}] Cleaning up temp files...")
        try:
            os.remove(input_path)
            os.remove(wav_path)
            shutil.rmtree(output_dir)
            logger.info(f"✅ [{job_id}] Cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ [{job_id}] Cleanup warning: {e}")
        
        # Call callback URL
        logger.info('=' * 80)
        logger.info(f"📞 [{job_id}] CALLING CALLBACK URL")
        logger.info('=' * 80)
        logger.info(f"🔗 [{job_id}] URL: {callback_url}")
        
        callback_payload = {
            'project_id': project_id,
            'stems_urls': stems_urls,
            'success': True
        }
        logger.debug(f"📦 [{job_id}] Callback payload: {callback_payload}")
        
        callback_response = requests.post(
            callback_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {base44_service_key}',
                'Base44-App-Id': base44_app_id
            },
            json=callback_payload,
            timeout=30
        )
        
        logger.info(f"📥 [{job_id}] Callback response status: {callback_response.status_code}")
        logger.debug(f"📥 [{job_id}] Callback response: {callback_response.text}")
        
        if callback_response.status_code != 200:
            logger.error(f"❌ [{job_id}] Callback failed!")
            logger.error(f"   Status: {callback_response.status_code}")
            logger.error(f"   Response: {callback_response.text}")
        else:
            logger.info(f"✅ [{job_id}] Callback successful!")
        
        logger.info('=' * 80)
        logger.info(f"🎉 [{job_id}] JOB COMPLETED SUCCESSFULLY")
        logger.info('=' * 80)
    
    except Exception as e:
        logger.error('=' * 80)
        logger.error(f"❌ [{job_id}] FATAL ERROR IN BACKGROUND JOB")
        logger.error('=' * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception(f"Full traceback:")
        
        # Try to notify callback of failure
        try:
            if callback_url and project_id and base44_service_key and base44_app_id:
                logger.info(f"📞 [{job_id}] Notifying callback of failure...")
                requests.post(
                    callback_url,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {base44_service_key}',
                        'Base44-App-Id': base44_app_id
                    },
                    json={
                        'project_id': project_id,
                        'success': False,
                        'error': str(e)
                    },
                    timeout=10
                )
                logger.info(f"✅ [{job_id}] Failure notification sent")
        except Exception as callback_error:
            logger.error(f"❌ [{job_id}] Failed to notify callback: {callback_error}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info('❤️ Health check requested')
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-stem-separation-optimized',
        'model': 'mdx_extra',
        'optimizations': ['parallel_uploads', 'ffmpeg_optimized', 'segment_processing'],
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER,
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    OPTIMIZED: ASYNC STEM SEPARATION WITH FASTER MODEL + PARALLEL UPLOADS
    """
    job_id = str(uuid.uuid4())[:8]
    
    logger.info('=' * 80)
    logger.info(f"🎵 [{job_id}] NEW STEM SEPARATION JOB STARTED")
    logger.info('=' * 80)
    
    try:
        # Log all form data
        logger.debug(f"📦 [{job_id}] Form data keys: {list(request.form.keys())}")
        logger.debug(f"📦 [{job_id}] Files: {list(request.files.keys())}")
        
        # Get form data
        if 'file' not in request.files:
            logger.error(f"❌ [{job_id}] No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        logger.info(f"📄 [{job_id}] File received: {file.filename}")
        logger.info(f"📊 [{job_id}] File content type: {file.content_type}")
        
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_service_key = request.form.get('base44_service_key')
        base44_app_id = request.form.get('base44_app_id')
        
        logger.info(f"🔗 [{job_id}] Callback URL: {callback_url}")
        logger.info(f"🆔 [{job_id}] Project ID: {project_id}")
        logger.info(f"🔑 [{job_id}] Service key received: {bool(base44_service_key)}")
        logger.info(f"🆔 [{job_id}] App ID received: {bool(base44_app_id)}")
        
        if not callback_url or not project_id or not base44_app_id:
            logger.error(f"❌ [{job_id}] Missing required parameters")
            logger.error(f"   callback_url: {callback_url}")
            logger.error(f"   project_id: {project_id}")
            logger.error(f"   base44_app_id: {base44_app_id}")
            return jsonify({'error': 'Missing callback_url, project_id, or base44_app_id'}), 400
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input{os.path.splitext(file.filename)[1]}")
        logger.info(f"💾 [{job_id}] Saving file to: {input_path}")
        file.save(input_path)
        
        file_size = os.path.getsize(input_path)
        logger.info(f"✅ [{job_id}] File saved successfully")
        logger.info(f"📊 [{job_id}] File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # OPTIMIZATION 4: FFmpeg with optimized settings
        wav_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input.wav")
        if not input_path.endswith('.wav'):
            logger.info(f"🔄 [{job_id}] Converting to WAV with optimized FFmpeg...")
            
            # OPTIMIZATION 4: Optimized FFmpeg command
            # -threads 0: Use all available CPU threads
            # -loglevel error: Reduce verbose output
            # -preset ultrafast: Fast encoding (for conversion, not compression)
            ffmpeg_cmd = [
                'ffmpeg',
                '-threads', '0',        # Use all CPU threads
                '-i', input_path,       # Input file
                '-ar', '44100',         # Sample rate 44.1kHz
                '-ac', '2',             # Stereo
                '-loglevel', 'error',   # Less verbose
                '-y',                   # Overwrite output
                wav_path
            ]
            
            logger.debug(f"🔄 [{job_id}] FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.stderr:
                logger.debug(f"📤 [{job_id}] FFmpeg stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"❌ [{job_id}] FFmpeg conversion failed")
                logger.error(f"   Return code: {result.returncode}")
                logger.error(f"   Stderr: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
            
            logger.info(f"✅ [{job_id}] Conversion complete")
        else:
            logger.info(f"📋 [{job_id}] File already WAV, copying...")
            shutil.copy(input_path, wav_path)
            logger.info(f"✅ [{job_id}] File copied")
        
        wav_size = os.path.getsize(wav_path)
        logger.info(f"📊 [{job_id}] WAV file size: {wav_size} bytes ({wav_size / 1024 / 1024:.2f} MB)")
        
        # Start background processing thread
        logger.info(f"🚀 [{job_id}] Starting background processing thread...")
        thread = threading.Thread(
            target=process_job_async,
            args=(job_id, input_path, wav_path, callback_url, project_id, base44_service_key, base44_app_id)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"✅ [{job_id}] Background job started, returning immediately")
        
        return jsonify({
            'success': True,
            'project_id': project_id,
            'message': 'Stem separation started in background (fully optimized)'
        })
    
    except Exception as e:
        logger.error('=' * 80)
        logger.error(f"❌ [{job_id}] FATAL ERROR")
        logger.error('=' * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception(f"Full traceback:")
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info('=' * 80)
    logger.info(f"🚀 Starting Flask server on port {port}")
    logger.info('=' * 80)
    app.run(host='0.0.0.0', port=port, debug=False)



