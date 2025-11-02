import os
import uuid
import subprocess
import base64
import json
import logging
import time
import threading
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
JOBS_FOLDER = '/tmp/jobs'
DEMUCS_MODEL = 'htdemucs_6s'
SHIFTS = 2

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

# In-memory job storage (for Railway, use Redis in production)
jobs = {}


def get_job_file(job_id):
    """Get path to job status file"""
    return os.path.join(JOBS_FOLDER, f'{job_id}.json')


def save_job(job_id, data):
    """Save job status to file and memory"""
    jobs[job_id] = data
    with open(get_job_file(job_id), 'w') as f:
        json.dump(data, f)


def load_job(job_id):
    """Load job status from file or memory"""
    if job_id in jobs:
        return jobs[job_id]
    
    job_file = get_job_file(job_id)
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            data = json.load(f)
            jobs[job_id] = data
            return data
    
    return None


def process_separation_task(job_id, input_path, original_filename):
    """Background task to process audio separation"""
    try:
        logger.info(f"[{job_id}] Starting background processing")
        
        # Update status
        save_job(job_id, {
            'status': 'processing',
            'progress': 10,
            'message': 'Initializing Demucs AI...',
            'started_at': time.time()
        })

        # Prepare output directory
        job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        # Update progress
        save_job(job_id, {
            'status': 'processing',
            'progress': 20,
            'message': 'Running AI separation (this takes 10-15 minutes)...',
            'started_at': jobs[job_id]['started_at']
        })

        # Run Demucs separation
        logger.info(f"[{job_id}] Running Demucs with model={DEMUCS_MODEL}, shifts={SHIFTS}")
        
        result = subprocess.run([
            'python', '-m', 'demucs',
            '-n', DEMUCS_MODEL,
            '--shifts', str(SHIFTS),
            '-o', job_output_dir,
            input_path
        ], check=True, capture_output=True, text=True)

        # Update progress
        save_job(job_id, {
            'status': 'processing',
            'progress': 80,
            'message': 'Encoding stems...',
            'started_at': jobs[job_id]['started_at']
        })

        # Locate separated stems
        file_ext = os.path.splitext(original_filename)[1]
        base_name = job_id + file_ext.replace('.', '_') if file_ext else job_id
        
        # Try different possible output paths
        possible_paths = [
            os.path.join(job_output_dir, DEMUCS_MODEL, base_name),
            os.path.join(job_output_dir, DEMUCS_MODEL, job_id),
            os.path.join(job_output_dir, DEMUCS_MODEL, os.path.splitext(original_filename)[0])
        ]
        
        stems_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                stems_dir = path
                break
        
        if not stems_dir:
            # List what's actually in the output directory
            logger.error(f"[{job_id}] Tried paths: {possible_paths}")
            logger.error(f"[{job_id}] Output dir contents: {os.listdir(job_output_dir)}")
            if os.path.exists(os.path.join(job_output_dir, DEMUCS_MODEL)):
                logger.error(f"[{job_id}] Model dir contents: {os.listdir(os.path.join(job_output_dir, DEMUCS_MODEL))}")
            raise FileNotFoundError(f"Output directory not found. Tried: {possible_paths}")

        logger.info(f"[{job_id}] Stems directory: {stems_dir}")

        # Encode stems to base64
        stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        stems_data = {}
        
        for stem_name in stem_names:
            stem_path = os.path.join(stems_dir, f'{stem_name}.wav')
            
            if os.path.exists(stem_path):
                file_size = os.path.getsize(stem_path)
                logger.info(f"[{job_id}] Encoding {stem_name}: {file_size} bytes")
                
                with open(stem_path, 'rb') as f:
                    stems_data[stem_name] = base64.b64encode(f.read()).decode('utf-8')
            else:
                logger.warning(f"[{job_id}] Missing stem: {stem_name}")
                stems_data[stem_name] = None

        # Calculate processing time
        elapsed = round(time.time() - jobs[job_id]['started_at'], 2)

        # Save completed job
        save_job(job_id, {
            'status': 'completed',
            'progress': 100,
            'message': 'Separation complete!',
            'stems': stems_data,
            'processing_time': elapsed,
            'completed_at': time.time()
        })

        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(job_output_dir):
            import shutil
            shutil.rmtree(job_output_dir)

        logger.info(f"[{job_id}] Completed successfully in {elapsed}s")

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        save_job(job_id, {
            'status': 'failed',
            'progress': 0,
            'message': str(e),
            'error': str(e)
        })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': DEMUCS_MODEL,
        'shifts': SHIFTS,
        'active_jobs': len(jobs)
    })


@app.route('/start_separation', methods=['POST'])
def start_separation():
    """Start separation job and return job_id immediately"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] Starting new job for: {file.filename}")
    
    try:
        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{job_id}{file_ext}"
        input_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(input_path)
        
        logger.info(f"[{job_id}] File saved: {input_path}")

        # Initialize job status
        save_job(job_id, {
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued...',
            'filename': file.filename
        })

        # Start background processing thread
        thread = threading.Thread(
            target=process_separation_task,
            args=(job_id, input_path, file.filename)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Processing started'
        })

    except Exception as e:
        logger.error(f"[{job_id}] Startup error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get status of a specific job"""
    job_data = load_job(job_id)
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_data)


@app.route('/list_jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': list(jobs.keys()),
        'count': len(jobs)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)















