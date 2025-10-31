import os
import uuid
import logging
import tempfile
import shutil
import subprocess
import atexit
import threading
import requests
import base64
from pydub import AudioSegment
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_uploads_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix="audio_outputs_")
logging.info(f"Using UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logging.info(f"Using OUTPUT_FOLDER: {OUTPUT_FOLDER}")

# ==================== FILE FORMAT HANDLING ====================

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub/ffmpeg."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(2).set_frame_rate(44100) # Demucs needs 44.1kHz
        audio.export(output_path, format='wav')
        logging.info(f"✅ Converted {input_path} to WAV")
        return True
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        return False

# ==================== BASE44 FILE UPLOAD HELPER ====================

def upload_file_to_base44(file_path, filename, base44_service_key, base44_app_id):
    """Upload a file directly to Base44 storage using their API."""
    try:
        logging.info(f"☁️ Uploading {filename} to Base44 storage...")
        
        base44_api_url = f"https://api.base44.com/v1/apps/{base44_app_id}/integrations/Core/UploadFile"
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'audio/mpeg')} 
            headers = {
                'Authorization': f'Bearer {base44_service_key}'
            }
            
            logging.info(f"📤 Sending POST request to {base44_api_url}...")
            response = requests.post(base44_api_url, files=files, headers=headers, timeout=120)
        
        if response.status_code >= 200 and response.status_code < 300:
            result = response.json()
            file_url = result.get('file_url')
            logging.info(f"✅ Uploaded {filename}: {file_url[:50]}...")
            return file_url
        else:
            logging.error(f"❌ Base44 upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Error uploading {filename} to Base44: {e}", exc_info=True)
        return None

# ==================== STEM SEPARATION ====================

def process_stems_async(wav_path, output_dir, callback_url, project_id, base44_api_key, base44_app_id, base44_service_key, input_path_to_clean, wav_path_to_clean, output_dir_to_clean):
    """Background thread: Run Demucs, upload stems, callback."""
    try:
        logging.info(f"🎵 Running demucs with HIGH QUALITY settings (htdemucs_ft) for {wav_path}...")
        
        # Use subprocess to call demucs
        subprocess.run([
            'python', '-m', 'demucs.separate', # Use 'python -m' for reliability
            '-n', 'htdemucs_ft', # High-quality model
            '-o', output_dir,
            '--mp3',
            '--mp3-bitrate', '192',
            '--jobs', '2',
            wav_path
        ], check=True, capture_output=True, text=True, timeout=1740) # 29 minute timeout
        
        logging.info(f"✅ Demucs complete for {wav_path}")
        
        stem_track_dir = os.path.join(output_dir, 'htdemucs_ft', os.path.splitext(os.path.basename(wav_path))[0])
        
        if not os.path.exists(stem_track_dir):
            logging.error(f"Stem track directory not found: {stem_track_dir}. Searching...")
            found_stem_dir = False
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_track_dir = root
                    found_stem_dir = True
                    break
            if not found_stem_dir:
                raise ValueError(f"Could not find any stem output directory inside {output_dir}")
        
        logging.info(f"Stems located in: {stem_track_dir}")

        # Upload stems to Base44
        stems_urls = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_file_path = os.path.join(stem_track_dir, f"{stem_name}.mp3")
            if os.path.exists(stem_file_path):
                logging.info(f"Uploading {stem_name}...")
                url = upload_file_to_base44(stem_file_path, f"{stem_name}.mp3", base44_service_key, base44_app_id)
                if url:
                    stems_urls[stem_name] = url
                    logging.info(f"✅ {stem_name} uploaded: {url}")
                else:
                    logging.warning(f"⚠️ Failed to upload {stem_name}.mp3")
            else:
                logging.warning(f"⚠️ Missing {stem_name}.mp3 at {stem_file_path}")
        
        if not stems_urls:
            raise ValueError("No stems were successfully separated or uploaded.")
        
        logging.info(f"All stems processed. URLs: {stems_urls}")
        
        # Callback to Deno
        logging.info(f"📞 Calling back to {callback_url}")
        callback_headers = {
            "Authorization": f"Bearer {base44_api_key}",
            "Base44-App-Id": base44_app_id,
            "Content-Type": "application/json"
        }
        callback_payload = {
            'success': True,
            'project_id': project_id,
            'stems_urls': stems_urls
        }
        
        callback_response = requests.post(callback_url, json=callback_payload, headers=callback_headers, timeout=30)
        
        logging.info(f"Callback response: {callback_response.status_code}")
        
    except Exception as e:
        logging.error(f"❌ Async separation failed: {e}", exc_info=True)
        try:
            callback_headers = {
                "Authorization": f"Bearer {base44_api_key}",
                "Base44-App-Id": base44_app_id,
                "Content-Type": "application/json"
            }
            requests.post(callback_url, json={
                'success': False,
                'project_id': project_id,
                'error': str(e)
            }, headers=callback_headers, timeout=30)
        except Exception as callback_error:
            logging.error(f"Callback (error) also failed: {callback_error}")
    
    finally:
        # Cleanup temp files
        try:
            if input_path_to_clean and os.path.exists(input_path_to_clean):
                os.remove(input_path_to_clean)
            if wav_path_to_clean and os.path.exists(wav_path_to_clean) and wav_path_to_clean != input_path_to_clean:
                os.remove(wav_path_to_clean)
            if output_dir_to_clean and os.path.exists(output_dir_to_clean):
                shutil.rmtree(output_dir_to_clean)
            logging.info(f"Stem separation cleanup complete for {project_id}")
        except Exception as cleanup_e:
            logging.error(f"Error during cleanup: {cleanup_e}")

# ==================== FLASK ROUTES ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-stem-separation-only',
    })

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """ASYNC endpoint: Separate stems, upload to Base44, callback when done."""
    input_path = None
    wav_path = None
    output_dir = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_api_key = request.form.get('base44_api_key')
        base44_app_id = request.form.get('base44_app_id')
        base44_service_key = request.form.get('base44_service_key')
        
        logging.info(f"📥 Received separation request for project: {project_id}")
        
        if not all([callback_url, project_id, base44_api_key, base44_app_id, base44_service_key]):
            logging.error("Missing required form fields")
            return jsonify({'error': 'Missing required fields'}), 400
        
        unique_id = str(uuid.uuid4())
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'

        input_path = os.path.join(UPLOAD_FOLDER, f"sep_{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"sep_{unique_id}_input.wav")
        output_dir = os.path.join(OUTPUT_FOLDER, f"sep_{unique_id}_stems")

        file.save(input_path)
        logging.info(f"File saved to: {input_path}")

        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert to WAV")
        else:
            shutil.copy(input_path, wav_path) # It's already a wav, just copy
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Start background thread
        thread = threading.Thread(
            target=process_stems_async,
            args=(
                wav_path, output_dir, callback_url, project_id, 
                base44_api_key, base44_app_id, base44_service_key,
                input_path, wav_path, output_dir # Pass paths for cleanup
            ),
            daemon=True
        )
        thread.start()
        
        logging.info(f"✅ Background processing started for {project_id}")
        
        return jsonify({'success': True, 'project_id': project_id, 'message': 'Processing started'}), 202
        
    except Exception as e:
        logging.error(f"❌ Error starting separation: {e}", exc_info=True)
        # Cleanup on immediate failure
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception as cleanup_e:
            logging.error(f"Cleanup error: {cleanup_e}")
        return jsonify({'error': str(e)}), 500

@atexit.register
def cleanup():
    """Cleanup temp directories on exit."""
    try:
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        logging.info("Cleaned up temporary directories")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)





