import os
import uuid
import subprocess
import sys
import json
import base64
import tempfile
import shutil
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """Separates audio into stems using demucs. Returns base64 encoded MP3s."""
    input_path = None
    output_dir = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        unique_id = str(uuid.uuid4())
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
        
        file.save(input_path)
        logging.info(f"📁 Saved: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"🎵 Running Demucs FAST...")
        
        subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs',
            '-o', output_dir,
            '--mp3',
            '--mp3-bitrate', '192',
            '--jobs', '2',
            input_path
        ], check=True, capture_output=True, text=True, timeout=600)
        
        logging.info(f"✅ Demucs complete")
        
        # Find stems directory
        stem_track_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_path))[0])
        
        if not os.path.exists(stem_track_dir):
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_track_dir = root
                    break
        
        logging.info(f"Stems: {stem_track_dir}")

        stems_data = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_file = os.path.join(stem_track_dir, f"{stem_name}.mp3")
            if os.path.exists(stem_file):
                with open(stem_file, 'rb') as f:
                    stems_data[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                logging.info(f"✅ {stem_name}.mp3")
        
        if not stems_data:
            raise ValueError("No stems created")

        logging.info("🎉 Complete!")
        return jsonify(stems_data), 200
    
    except subprocess.TimeoutExpired:
        logging.error("❌ Timeout")
        return jsonify({"error": "Timeout - file too long"}), 500
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Demucs failed: {e.stderr}")
        return jsonify({"error": "Separation failed", "details": e.stderr}), 500
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    import atexit
    
    def cleanup():
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    logging.error(f"Cleanup error: {e}")
    
    atexit.register(cleanup)
    app.run(host='0.0.0.0', port=5000, debug=False)
🚀 DEPLOYMENT STEPS:

Create requirements.txt:
flask
demucs
torch
torchaudio
Deploy to Railway:

Connect your GitHub repo
Railway auto-detects Flask
Get your URL: https://your-app.railway.app
Update Frontend:

In pages/Processing.js line 48, change:
const BACKEND_URL = "https://your-app.railway.app";
✨ What this does:

Python backend: ONLY Demucs separation → returns base64 MP3s
Base44 function: Receives base64 → uploads to Base44 storage → saves to database
Frontend: Orchestrates everything, shows players with volume controls
Ready to deploy! 🎵

4 minutes ago
OKAY CAN I HAVE THE CODE TO GIVE TO THE GITHUB PLEASE. MAKE SURE ITS PERFECT THINK ABOUT EVERYTHING THAT CAN GO WRONG AND AVOID IT WHIKE WRITING THIS .PY CODE

2 minutes ago
Revert this
Base44 AI
Base44

Thought for 5s
Here's your PRODUCTION-READY code for GitHub/Railway deployment:

📁 app.py (Main Flask Backend)
import os
import uuid
import subprocess
import sys
import json
import base64
import tempfile
import shutil
import logging
import signal
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time

# ==================== CONFIGURATION ====================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'opus'}
DEMUCS_TIMEOUT = 600  # 10 minutes max processing time
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='stems_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='stems_output_')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logger.info(f"Upload folder: {UPLOAD_FOLDER}")
logger.info(f"Output folder: {OUTPUT_FOLDER}")

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(filepath):
    """Safely delete a file."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up: {filepath}")
    except Exception as e:
        logger.error(f"Failed to cleanup {filepath}: {e}")

def cleanup_directory(directory):
    """Safely delete a directory and all contents."""
    try:
        if directory and os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"Cleaned up directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to cleanup directory {directory}: {e}")

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except:
        pass

def validate_audio_file(filepath):
    """Validate that the file is actually an audio file."""
    try:
        # Try to get file info using ffprobe
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_format', '-show_streams',
            '-print_format', 'json', filepath
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return False, "Invalid audio file"
        
        data = json.loads(result.stdout)
        
        # Check if it has audio streams
        has_audio = any(stream.get('codec_type') == 'audio' 
                       for stream in data.get('streams', []))
        
        if not has_audio:
            return False, "No audio stream found"
        
        # Check duration (optional - reject files longer than 15 minutes)
        duration = float(data.get('format', {}).get('duration', 0))
        if duration > 900:  # 15 minutes
            return False, "Audio file too long (max 15 minutes)"
        
        if duration < 1:
            return False, "Audio file too short"
        
        return True, "Valid audio file"
    
    except subprocess.TimeoutExpired:
        return False, "File validation timeout"
    except json.JSONDecodeError:
        return False, "Could not parse audio metadata"
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, f"Validation failed: {str(e)}"

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        logger.info(f"Converting {input_path} to WAV...")
        
        result = subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ar', '44100',  # Sample rate
            '-ac', '2',      # Stereo
            '-y',            # Overwrite output
            output_path
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        if not os.path.exists(output_path):
            logger.error(f"Conversion failed: output file not created")
            return False
        
        logger.info(f"✅ Converted to WAV: {output_path}")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timeout")
        return False
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return False

# ==================== MAIN ENDPOINT ====================

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    Separate audio into stems using Demucs.
    Returns base64-encoded MP3 files for each stem.
    """
    input_path = None
    wav_path = None
    output_dir = None
    unique_id = str(uuid.uuid4())
    
    try:
        # ===== VALIDATION =====
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Secure the filename
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        
        logger.info(f"📁 Processing: {original_filename} (ID: {unique_id})")
        
        # ===== SAVE UPLOADED FILE =====
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        
        try:
            file.save(input_path)
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return jsonify({"error": "Failed to save uploaded file"}), 500
        
        # Check file size
        file_size = os.path.getsize(input_path)
        if file_size > MAX_FILE_SIZE:
            cleanup_file(input_path)
            return jsonify({
                "error": f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
            }), 400
        
        if file_size < 1024:  # Less than 1KB
            cleanup_file(input_path)
            return jsonify({"error": "File too small or corrupted"}), 400
        
        logger.info(f"File saved: {input_path} ({file_size / (1024*1024):.2f}MB)")
        
        # ===== VALIDATE AUDIO =====
        is_valid, validation_msg = validate_audio_file(input_path)
        if not is_valid:
            cleanup_file(input_path)
            return jsonify({"error": f"Invalid audio file: {validation_msg}"}), 400
        
        logger.info(f"✅ Validation passed: {validation_msg}")
        
        # ===== CONVERT TO WAV =====
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_converted.wav")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                cleanup_file(input_path)
                return jsonify({"error": "Failed to convert audio file"}), 500
            cleanup_file(input_path)  # Remove original after conversion
            input_path = None
        else:
            # If already WAV, just rename
            shutil.move(input_path, wav_path)
            input_path = None
        
        # ===== RUN DEMUCS =====
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🎵 Running Demucs (htdemucs fast model)...")
        start_time = time.time()
        
        demucs_cmd = [
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs',           # Fast hybrid transformer model
            '-o', output_dir,            # Output directory
            '--mp3',                     # Output as MP3
            '--mp3-bitrate', '192',      # Good quality
            '--jobs', '2',               # Parallel processing
            '--float32',                 # Use float32 for better quality
            wav_path
        ]
        
        try:
            result = subprocess.run(
                demucs_cmd,
                capture_output=True,
                text=True,
                timeout=DEMUCS_TIMEOUT,
                check=True
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Demucs completed in {processing_time:.1f}s")
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Demucs timeout after {DEMUCS_TIMEOUT}s")
            cleanup_file(wav_path)
            cleanup_directory(output_dir)
            return jsonify({
                "error": f"Processing timeout. File may be too long (max {DEMUCS_TIMEOUT//60} min)"
            }), 500
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Demucs failed: {e.stderr}")
            cleanup_file(wav_path)
            cleanup_directory(output_dir)
            return jsonify({
                "error": "Stem separation failed",
                "details": e.stderr[:500] if e.stderr else "Unknown error"
            }), 500
        
        # ===== FIND STEM FILES =====
        # Demucs creates: output_dir/htdemucs/filename/vocals.mp3, drums.mp3, etc.
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        stem_dir = os.path.join(output_dir, 'htdemucs', base_name)
        
        if not os.path.exists(stem_dir):
            logger.warning(f"Expected stem directory not found: {stem_dir}")
            # Try to find stems anywhere in output
            stem_dir = None
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_dir = root
                    logger.info(f"Found stems at: {stem_dir}")
                    break
        
        if not stem_dir or not os.path.exists(stem_dir):
            cleanup_file(wav_path)
            cleanup_directory(output_dir)
            return jsonify({"error": "No stems were generated"}), 500
        
        # ===== ENCODE STEMS TO BASE64 =====
        stems_data = {}
        expected_stems = ['vocals', 'drums', 'bass', 'other']
        
        for stem_name in expected_stems:
            stem_file = os.path.join(stem_dir, f"{stem_name}.mp3")
            
            if os.path.exists(stem_file):
                try:
                    file_size = os.path.getsize(stem_file)
                    
                    if file_size < 100:  # Less than 100 bytes - likely empty
                        logger.warning(f"⚠️ {stem_name}.mp3 is suspiciously small ({file_size}B)")
                        continue
                    
                    with open(stem_file, 'rb') as f:
                        stem_bytes = f.read()
                        stems_data[stem_name] = base64.b64encode(stem_bytes).decode('utf-8')
                    
                    logger.info(f"✅ Encoded {stem_name}.mp3 ({file_size / 1024:.1f}KB)")
                
                except Exception as e:
                    logger.error(f"Failed to encode {stem_name}: {e}")
            else:
                logger.warning(f"⚠️ Missing {stem_name}.mp3")
        
        # ===== VALIDATE RESULTS =====
        if not stems_data:
            cleanup_file(wav_path)
            cleanup_directory(output_dir)
            return jsonify({"error": "No stems were successfully created"}), 500
        
        if len(stems_data) < 4:
            logger.warning(f"Only {len(stems_data)}/4 stems created: {list(stems_data.keys())}")
        
        # ===== CLEANUP =====
        cleanup_file(wav_path)
        cleanup_directory(output_dir)
        
        logger.info(f"🎉 Success! Returning {len(stems_data)} stems")
        
        return jsonify(stems_data), 200
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        
        # Emergency cleanup
        cleanup_file(input_path)
        cleanup_file(wav_path)
        cleanup_directory(output_dir)
        
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Railway."""
    try:
        # Check if demucs is available
        result = subprocess.run(
            [sys.executable, '-m', 'demucs', '--help'],
            capture_output=True,
            timeout=5
        )
        demucs_available = result.returncode == 0
        
        # Check ffmpeg
        ffmpeg_result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        ffmpeg_available = ffmpeg_result.returncode == 0
        
        return jsonify({
            "status": "healthy",
            "service": "Stem Separation API",
            "demucs": "available" if demucs_available else "unavailable",
            "ffmpeg": "available" if ffmpeg_available else "unavailable",
            "upload_folder": UPLOAD_FOLDER,
            "output_folder": OUTPUT_FOLDER
        }), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API info."""
    return jsonify({
        "service": "Stem Separation API",
        "version": "1.0.0",
        "model": "Demucs htdemucs (fast)",
        "endpoints": {
            "POST /separate_stems": "Separate audio into stems",
            "GET /health": "Health check",
            "GET /": "This page"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_duration_minutes": DEMUCS_TIMEOUT // 60
    }), 200

# ==================== CLEANUP ON EXIT ====================

def cleanup_on_exit():
    """Clean up temporary directories on shutdown."""
    logger.info("Cleaning up temporary directories...")
    cleanup_directory(UPLOAD_FOLDER)
    cleanup_directory(OUTPUT_FOLDER)

import atexit
atexit.register(cleanup_on_exit)

# ==================== RUN SERVER ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)











