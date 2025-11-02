import os
import uuid
import subprocess
import base64
import tempfile
import json
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/separated'
DEMUCS_MODEL = 'htdemucs_6s'
SHIFTS = 5
TIMEOUT = 1200  # 20 minutes

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': DEMUCS_MODEL,
        'shifts': SHIFTS,
        'demucs': 'available',
        'ffmpeg': 'available'
    })


@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    Separate audio into stems using Demucs AI model.
    Returns base64-encoded audio for each stem.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # Generate unique ID for this job
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] Starting separation for: {file.filename}")
    
    start_time = time.time()
    
    try:
        # Save uploaded file with unique name
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{job_id}{file_ext}"
        input_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(input_path)
        
        logger.info(f"[{job_id}] File saved: {input_path}")

        # Prepare output directory
        job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        # Run Demucs separation
        logger.info(f"[{job_id}] Running Demucs with model={DEMUCS_MODEL}, shifts={SHIFTS}")
        
        subprocess.run([
            'python', '-m', 'demucs',
            '-n', DEMUCS_MODEL,
            '--shifts', str(SHIFTS),
            '-o', job_output_dir,
            input_path
        ], check=True, timeout=TIMEOUT)

        # Locate separated stems
        base_name = os.path.splitext(temp_filename)[0]
        stems_dir = os.path.join(job_output_dir, DEMUCS_MODEL, base_name)

        if not os.path.exists(stems_dir):
            raise FileNotFoundError(f"Output directory not found: {stems_dir}")

        logger.info(f"[{job_id}] Stems directory: {stems_dir}")

        # Encode stems to base64
        stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        stems = {}
        
        for stem_name in stem_names:
            stem_path = os.path.join(stems_dir, f'{stem_name}.wav')
            
            if os.path.exists(stem_path):
                file_size = os.path.getsize(stem_path)
                logger.info(f"[{job_id}] Encoding {stem_name}: {file_size} bytes")
                
                with open(stem_path, 'rb') as f:
                    stems[stem_name] = base64.b64encode(f.read()).decode('utf-8')
            else:
                logger.warning(f"[{job_id}] Missing stem: {stem_name}")
                stems[stem_name] = None

        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(job_output_dir):
            import shutil
            shutil.rmtree(job_output_dir)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[{job_id}] Completed in {elapsed}s")

        return jsonify(stems)

    except subprocess.TimeoutExpired:
        logger.error(f"[{job_id}] Timeout after {TIMEOUT}s")
        return jsonify({'error': 'Processing timeout (20 minutes exceeded)'}), 504
    
    except subprocess.CalledProcessError as e:
        logger.error(f"[{job_id}] Demucs error: {e}")
        return jsonify({'error': f'Separation failed: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)













