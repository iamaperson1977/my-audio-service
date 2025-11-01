import os
import uuid
import subprocess
import sys
import json
import base64
import tempfile
import shutil
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CRITICAL: Allow frontend to call backend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"📁 Saved: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🎵 Running Demucs...")
        start_time = time.time()
        
        subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs',
            '-o', output_dir,
            '--mp3',
            '--mp3-bitrate', '192',
            '--jobs', '2',
            input_path
        ], check=True, capture_output=True, text=True, timeout=600)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Demucs complete in {processing_time:.1f}s")
        
        # Find stems directory
        stem_track_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(input_path))[0])
        
        if not os.path.exists(stem_track_dir):
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_track_dir = root
                    break
        
        logger.info(f"Stems directory: {stem_track_dir}")

        stems_data = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_file = os.path.join(stem_track_dir, f"{stem_name}.mp3")
            if os.path.exists(stem_file):
                with open(stem_file, 'rb') as f:
                    stems_data[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                logger.info(f"✅ Encoded {stem_name}.mp3")
        
        if not stems_data:
            raise ValueError("No stems were created")

        logger.info(f"🎉 Success! Returning {len(stems_data)} stems")
        return jsonify(stems_data), 200
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Processing timeout")
        return jsonify({"error": "Processing timeout - file too long (max 10 minutes)"}), 500
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Demucs failed: {e.stderr}")
        return jsonify({"error": "Stem separation failed", "details": str(e.stderr)[:200]}), 500
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # CRITICAL: Always cleanup, even if error
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Cleaned up input: {input_path}")
        except Exception as e:
            logger.error(f"Failed to delete input: {e}")
        
        try:
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logger.info(f"Cleaned up output: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to delete output: {e}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Check demucs is available
        result = subprocess.run(
            [sys.executable, '-m', 'demucs', '--help'],
            capture_output=True,
            timeout=5
        )
        demucs_ok = result.returncode == 0
        
        # Check ffmpeg
        ffmpeg_result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        ffmpeg_ok = ffmpeg_result.returncode == 0
        
        return jsonify({
            "status": "healthy",
            "demucs": "available" if demucs_ok else "unavailable",
            "ffmpeg": "available" if ffmpeg_ok else "unavailable"
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """API info."""
    return jsonify({
        "service": "Stem Separation API",
        "version": "1.0",
        "endpoints": {
            "POST /separate_stems": "Separate audio into 4 stems",
            "GET /health": "Health check"
        }
    }), 200

if __name__ == '__main__':
    import atexit
    
    def cleanup():
        """Cleanup on shutdown."""
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    logger.info(f"Shutdown cleanup: {folder}")
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
    
    atexit.register(cleanup)
    
    # CRITICAL: Get port from Railway environment
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)












