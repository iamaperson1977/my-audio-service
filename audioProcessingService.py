import os
import uuid
import logging
import tempfile
import shutil
import subprocess
import atexit
import requests
from flask import Flask, request, jsonify
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create temp directories
UPLOAD_FOLDER = tempfile.mkdtemp(prefix="audio_upload_")
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_output_")

logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
logger.info(f"📁 Output folder: {OUTPUT_FOLDER}")

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-stem-separation-only',
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER
    })

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    SIMPLIFIED: ONLY SEPARATES STEMS, NOTHING ELSE
    """
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"🎵 [{job_id}] Starting stem separation job")
    
    try:
        # Get form data
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_service_key = request.form.get('base44_service_key')
        
        if not callback_url or not project_id:
            return jsonify({'error': 'Missing callback_url or project_id'}), 400
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input{os.path.splitext(file.filename)[1]}")
        file.save(input_path)
        logger.info(f"💾 [{job_id}] Saved input: {input_path}")
        
        # Convert to WAV if needed
        wav_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input.wav")
        if not input_path.endswith('.wav'):
            logger.info(f"🔄 [{job_id}] Converting to WAV...")
            result = subprocess.run([
                'ffmpeg', '-i', input_path, '-ar', '44100', '-ac', '2', '-y', wav_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        else:
            shutil.copy(input_path, wav_path)
        
        # Run Demucs stem separation
        logger.info(f"🎸 [{job_id}] Running Demucs stem separation...")
        
        result = subprocess.run([
            'python3', '-m', 'demucs',
            '--two-stems=vocals',
            '-o', OUTPUT_FOLDER,
            '-n', 'htdemucs',
            wav_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Demucs failed: {result.stderr}")
        
        logger.info(f"✅ [{job_id}] Demucs separation complete")
        
        # Find output directory
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_dir = os.path.join(OUTPUT_FOLDER, 'htdemucs', base_name)
        
        if not os.path.exists(output_dir):
            raise Exception(f"Output directory not found: {output_dir}")
        
        # Expected stems
        stem_files = {
            'vocals': os.path.join(output_dir, 'vocals.wav'),
            'drums': os.path.join(output_dir, 'drums.wav'),
            'bass': os.path.join(output_dir, 'bass.wav'),
            'other': os.path.join(output_dir, 'other.wav')
        }
        
        # Upload stems to Base44
        stems_urls = {}
        logger.info(f"☁️ [{job_id}] Uploading stems to Base44...")
        
        for stem_name, stem_path in stem_files.items():
            if not os.path.exists(stem_path):
                logger.warning(f"⚠️ [{job_id}] Missing stem: {stem_name}")
                continue
            
            with open(stem_path, 'rb') as f:
                upload_response = requests.post(
                    'https://app.base44.com/api/integrations/Core.UploadFile',
                    headers={'Authorization': f'Bearer {base44_service_key}'},
                    files={'file': (f'{stem_name}.wav', f, 'audio/wav')}
                )
            
            if upload_response.status_code != 200:
                raise Exception(f"Failed to upload {stem_name}: {upload_response.text}")
            
            file_url = upload_response.json().get('file_url')
            stems_urls[stem_name] = file_url
            logger.info(f"✅ [{job_id}] Uploaded {stem_name}")
        
        # Cleanup
        try:
            os.remove(input_path)
            os.remove(wav_path)
            shutil.rmtree(output_dir)
        except:
            pass
        
        # Call callback
        logger.info(f"📞 [{job_id}] Calling callback URL...")
        callback_response = requests.post(
            callback_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {base44_service_key}'
            },
            json={
                'project_id': project_id,
                'stems_urls': stems_urls,
                'success': True
            },
            timeout=30
        )
        
        if callback_response.status_code != 200:
            logger.error(f"❌ [{job_id}] Callback failed: {callback_response.text}")
        else:
            logger.info(f"✅ [{job_id}] Callback successful")
        
        return jsonify({
            'success': True,
            'project_id': project_id,
            'message': 'Stem separation complete'
        })
    
    except Exception as e:
        logger.error(f"❌ [{job_id}] Error: {str(e)}")
        
        try:
            if callback_url and project_id and base44_service_key:
                requests.post(
                    callback_url,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {base44_service_key}'
                    },
                    json={
                        'project_id': project_id,
                        'success': False,
                        'error': str(e)
                    },
                    timeout=10
                )
        except:
            pass
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)







