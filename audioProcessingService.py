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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-pro',
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER
    })

@app.route('/separate_stems_sync', methods=['POST'])
def separate_stems_sync():
    """Separate audio into stems and return as base64"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        input_path = os.path.join(UPLOAD_FOLDER, f"input_{uuid.uuid4().hex}.mp3")
        file.save(input_path)
        
        logging.info(f"Running Demucs on {input_path}")
        
        # Run Demucs
        result = subprocess.run([
            'demucs',
            '--two-stems=vocals',
            '-o', OUTPUT_FOLDER,
            input_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Demucs error: {result.stderr}")
            return jsonify({'error': 'Stem separation failed'}), 500
        
        # Find output directory
        demucs_output = os.path.join(OUTPUT_FOLDER, 'htdemucs')
        if not os.path.exists(demucs_output):
            demucs_output = os.path.join(OUTPUT_FOLDER, 'mdx_extra')
        
        # Get the actual folder name (Demucs creates a folder named after input file)
        output_dirs = [d for d in os.listdir(demucs_output) if os.path.isdir(os.path.join(demucs_output, d))]
        if not output_dirs:
            return jsonify({'error': 'No stems generated'}), 500
        
        stem_folder = os.path.join(demucs_output, output_dirs[0])
        
        # Read stems and encode to base64
        stems_base64 = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_path = os.path.join(stem_folder, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                # Convert to MP3 first to reduce size
                mp3_path = stem_path.replace('.wav', '.mp3')
                subprocess.run(['ffmpeg', '-i', stem_path, '-b:a', '192k', mp3_path, '-y'], 
                             capture_output=True)
                
                with open(mp3_path, 'rb') as f:
                    stems_base64[stem_name] = base64.b64encode(f.read()).decode('utf-8')
                
                os.remove(stem_path)
                os.remove(mp3_path)
        
        # Cleanup
        os.remove(input_path)
        
        logging.info(f"Stems separated: {list(stems_base64.keys())}")
        return jsonify({'success': True, 'stems': stems_base64})
        
    except Exception as e:
        logging.error(f"Error in separate_stems_sync: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with AI decisions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        if 'ai_decisions' not in request.form:
            return jsonify({'error': 'No AI decisions provided'}), 400
        
        file = request.files['file']
        ai_decisions = eval(request.form['ai_decisions'])
        output_format = request.form.get('output_format', 'wav')
        
        # Save input
        input_path = os.path.join(UPLOAD_FOLDER, f"process_{uuid.uuid4().hex}.wav")
        file.save(input_path)
        
        # Load audio
        audio, sr = sf.read(input_path)
        
        # Apply restoration
        if 'restoration' in ai_decisions:
            restoration = ai_decisions['restoration']
            if 'noise_reduction' in restoration:
                nr = restoration['noise_reduction']
                threshold = nr.get('threshold_db', -60)
                reduction = nr.get('reduction_db', 12)
                audio = apply_noise_gate(audio, threshold, reduction, sr)
        
        # Apply mixing
        if 'mixing' in ai_decisions:
            mixing = ai_decisions['mixing']
            if 'equalizer' in mixing and 'bands' in mixing['equalizer']:
                for band in mixing['equalizer']['bands']:
                    audio = apply_eq_band(audio, sr, band['frequency'], band['gain_db'], band['q_factor'])
            
            if 'compression' in mixing:
                comp = mixing['compression']
                audio = apply_compression(audio, sr, comp.get('threshold_db', -20), comp.get('ratio', '4:1'))
        
        # Apply mastering
        if 'mastering' in ai_decisions:
            mastering = ai_decisions['mastering']
            if 'limiting' in mastering:
                lim = mastering['limiting']
                audio = apply_limiter(audio, lim.get('target_lufs', -14), lim.get('ceiling_db', -0.5))
        
        # Save output
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{uuid.uuid4().hex}.{output_format}")
        sf.write(output_path, audio, sr)
        
        # Read and encode as base64
        with open(output_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)
        
        return jsonify({'success': True, 'processed_audio_base64': audio_base64})
        
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

def apply_noise_gate(audio, threshold_db, reduction_db, sr):
    """Simple noise gate"""
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
    """Apply parametric EQ band"""
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
    """Simple compression"""
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
    """Simple limiter with loudness normalization"""
    ceiling = 10 ** (ceiling_db / 20)
    
    # Measure current loudness (simplified RMS)
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_lufs / 20) * 0.1
    
    # Apply gain to reach target
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    
    # Hard limit
    audio = np.clip(audio, -ceiling, ceiling)
    
    return audio

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)






