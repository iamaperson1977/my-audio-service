from flask import Flask, request, jsonify
import os
import tempfile
import logging
import soundfile as sf
import numpy as np
from scipy import signal
import uuid
import subprocess
import base64
import threading
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()

# Store job results
jobs = {}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/start_separation', methods=['POST'])
def start_separation():
    """Start async stem separation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        job_id = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
        file.save(input_path)
        
        jobs[job_id] = {'status': 'processing', 'result': None, 'error': None}
        
        # Start background thread
        thread = threading.Thread(target=run_demucs, args=(job_id, input_path))
        thread.start()
        
        return jsonify({'job_id': job_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_demucs(job_id, input_path):
    """Background Demucs processing"""
    try:
        logging.info(f"Running Demucs for job {job_id}")
        
        result = subprocess.run([
            'demucs',
            '--two-stems=vocals',
            '-o', OUTPUT_FOLDER,
            input_path
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            jobs[job_id] = {'status': 'failed', 'error': 'Demucs failed'}
            return
        
        demucs_output = os.path.join(OUTPUT_FOLDER, 'htdemucs')
        if not os.path.exists(demucs_output):
            demucs_output = os.path.join(OUTPUT_FOLDER, 'mdx_extra')
        
        output_dirs = [d for d in os.listdir(demucs_output) if os.path.isdir(os.path.join(demucs_output, d))]
        stem_folder = os.path.join(demucs_output, output_dirs[0])
        
        stems_base64 = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_path = os.path.join(stem_folder, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                mp3_path = stem_path.replace('.wav', '.mp3')
                subprocess.run(['ffmpeg', '-i', stem_path, '-b:a', '192k', mp3_path, '-y'], capture_output=True)
                
                with open(mp3_path, 'rb') as f:
                    stems_base64[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                
                os.remove(stem_path)
                os.remove(mp3_path)
        
        os.remove(input_path)
        
        jobs[job_id] = {'status': 'completed', 'result': {'stems': stems_base64}}
        logging.info(f"Job {job_id} completed")
        
    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'error': str(e)}

@app.route('/check_status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with AI decisions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        ai_decisions = eval(request.form['ai_decisions'])
        output_format = request.form.get('output_format', 'wav')
        
        input_path = os.path.join(UPLOAD_FOLDER, f"process_{uuid.uuid4().hex}.wav")
        file.save(input_path)
        
        audio, sr = sf.read(input_path)
        
        if 'restoration' in ai_decisions:
            restoration = ai_decisions['restoration']
            if 'noise_reduction' in restoration:
                nr = restoration['noise_reduction']
                audio = apply_noise_gate(audio, nr.get('threshold_db', -60), nr.get('reduction_db', 12), sr)
        
        if 'mixing' in ai_decisions:
            mixing = ai_decisions['mixing']
            if 'equalizer' in mixing and 'bands' in mixing['equalizer']:
                for band in mixing['equalizer']['bands']:
                    audio = apply_eq_band(audio, sr, band['frequency'], band['gain_db'], band['q_factor'])
            
            if 'compression' in mixing:
                comp = mixing['compression']
                audio = apply_compression(audio, sr, comp.get('threshold_db', -20), comp.get('ratio', '4:1'))
        
        if 'mastering' in ai_decisions:
            mastering = ai_decisions['mastering']
            if 'limiting' in mastering:
                lim = mastering['limiting']
                audio = apply_limiter(audio, lim.get('target_lufs', -14), lim.get('ceiling_db', -0.5))
        
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{uuid.uuid4().hex}.{output_format}")
        sf.write(output_path, audio, sr)
        
        with open(output_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(input_path)
        os.remove(output_path)
        
        return jsonify({'success': True, 'processed_audio_base64': audio_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def apply_noise_gate(audio, threshold_db, reduction_db, sr):
    threshold_linear = 10 ** (threshold_db / 20)
    reduction_linear = 10 ** (reduction_db / 20)
    
    if len(audio.shape) == 1:
        envelope = np.abs(audio)
    else:
        envelope = np.max(np.abs(audio), axis=1)
    
    gate = np.where(envelope > threshold_linear, 1.0, 1.0 / reduction_linear)
    
    if len(audio.shape) == 2:
        gate = gate[:, np.newaxis]
    
    return audio * gate

def apply_eq_band(audio, sr, frequency, gain_db, q_factor):
    gain_linear = 10 ** (gain_db / 20)
    
    w0 = 2 * np.pi * frequency / sr
    alpha = np.sin(w0) / (2 * q_factor)
    
    if gain_db > 0:
        A = np.sqrt(gain_linear)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
    else:
        A = np.sqrt(1 / gain_linear)
        b0 = 1 + alpha / A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha / A
        a0 = 1 + alpha * A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha * A
    
    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]
    
    if len(audio.shape) == 1:
        return signal.lfilter(b, a, audio)
    else:
        return np.array([signal.lfilter(b, a, audio[:, i]) for i in range(audio.shape[1])]).T

def apply_compression(audio, sr, threshold_db, ratio_str):
    ratio = float(ratio_str.split(':')[0])
    threshold = 10 ** (threshold_db / 20)
    
    if len(audio.shape) == 1:
        envelope = np.abs(audio)
    else:
        envelope = np.max(np.abs(audio), axis=1)
    
    compressed_envelope = np.where(
        envelope > threshold,
        threshold + (envelope - threshold) / ratio,
        envelope
    )
    
    gain = np.where(envelope > 0, compressed_envelope / envelope, 1.0)
    
    if len(audio.shape) == 2:
        gain = gain[:, np.newaxis]
    
    return audio * gain

def apply_limiter(audio, target_lufs, ceiling_db):
    ceiling = 10 ** (ceiling_db / 20)
    
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_lufs / 20) * 0.1
    
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    
    audio = np.clip(audio, -ceiling, ceiling)
    
    return audio

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)







