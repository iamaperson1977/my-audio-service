import os
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt, tf2sos, iirpeak, iirnotch, lfilter
from scipy import signal
import logging
from pydub import AudioSegment
import librosa
import tempfile
import shutil
import numba
import subprocess
import sys
import json
import atexit
import base64
import threading
import requests

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
        audio = audio.set_channels(2).set_frame_rate(44100)
        audio.export(output_path, format='wav')
        logging.info(f"✅ Converted {input_path} to WAV")
        return True
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        return False

def read_wav(filepath):
    """Reads WAV file into NumPy array."""
    try:
        audio_data, sample_rate = sf.read(filepath, dtype='float32')
        logging.info(f"Read WAV: SR={sample_rate}, Shape={audio_data.shape}")
        return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Error reading WAV: {e}")
        return None, None

def write_wav(filepath, audio_data, sample_rate):
    """Writes NumPy array to WAV file."""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        audio_data = np.clip(audio_data, -1.0, 1.0)
        sf.write(filepath, audio_data, sample_rate)
        logging.info(f"Wrote WAV: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error writing WAV: {e}")
        return False

# ==================== BASE44 FILE UPLOAD ====================

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

# ==================== DSP FUNCTIONS ====================

@numba.jit(nopython=True)
def _gate_channel(audio_channel, threshold_linear, attack_coeff, release_coeff, ratio):
    """Optimized core logic for noise gate on a single channel."""
    envelope = 0.0
    gain_smooth = 1.0
    output = np.zeros_like(audio_channel)
    inv_ratio = 1.0 / ratio if ratio > 0 else 1.0

    for i, sample in enumerate(audio_channel):
        detection_signal = abs(sample)
        if detection_signal > envelope:
            envelope = attack_coeff * envelope + (1.0 - attack_coeff) * detection_signal
        else:
            envelope = release_coeff * envelope + (1.0 - release_coeff) * detection_signal

        target_gain = 1.0
        if envelope < threshold_linear:
            under_amount = threshold_linear - envelope
            target_gain = 1.0 - (under_amount / threshold_linear) * (ratio - 1.0) * inv_ratio
            target_gain = max(0.0, target_gain)

        if target_gain < gain_smooth:
            gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
        else:
            gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain

        output[i] = sample * gain_smooth
    return output

def apply_noise_gate(audio_data, threshold_db=-60, ratio=10, attack_ms=5, release_ms=50, sample_rate=44100):
    """Professional noise gate with envelope follower."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -80, 0)
        ratio = np.clip(ratio, 1, 100)
        attack_ms = np.clip(attack_ms, 0.1, 100)
        release_ms = np.clip(release_ms, 1, 1000)
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        
        if audio_data.ndim == 1:
            return _gate_channel(audio_data, threshold_linear, attack_coeff, release_coeff, ratio).astype(np.float32)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = _gate_channel(audio_data[:, ch], threshold_linear, attack_coeff, release_coeff, ratio)
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"Noise gate error: {e}")
        return audio_data

def apply_deesser(audio_data, sample_rate, frequency=6000, threshold_db=-15, ratio=3, attack_ms=1, release_ms=10):
    """De-esser for controlling harsh sibilance."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        frequency = np.clip(frequency, 2000, 12000)
        threshold_db = np.clip(threshold_db, -40, 0)
        ratio = np.clip(ratio, 1, 10)
        
        threshold_linear = 10 ** (threshold_db / 20)
        reduction_linear = 1.0 / ratio
        
        nyquist = sample_rate / 2
        if frequency >= nyquist:
            frequency = nyquist * 0.9
        
        Q = 2.0
        w0 = 2 * np.pi * frequency / sample_rate
        alpha = np.sin(w0) / (2 * Q)
        
        b0 = 1 + alpha
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        
        if audio_data.ndim == 1:
            filtered = lfilter(b, a, audio_data)
            envelope = 0.0
            gain_smooth = 1.0
            output = np.zeros_like(audio_data)
            
            for i in range(len(audio_data)):
                detection = abs(filtered[i])
                if detection > envelope:
                    envelope = attack_coeff * envelope + (1.0 - attack_coeff) * detection
                else:
                    envelope = release_coeff * envelope + (1.0 - release_coeff) * detection
                
                if envelope > threshold_linear:
                    target_gain = threshold_linear + (envelope - threshold_linear) * reduction_linear
                    target_gain = target_gain / envelope if envelope > 0 else 1.0
                else:
                    target_gain = 1.0
                
                if target_gain < gain_smooth:
                    gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
                else:
                    gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain
                
                output[i] = audio_data[i] * gain_smooth
            
            return output.astype(np.float32)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                filtered = lfilter(b, a, audio_data[:, ch])
                envelope = 0.0
                gain_smooth = 1.0
                
                for i in range(len(audio_data)):
                    detection = abs(filtered[i])
                    if detection > envelope:
                        envelope = attack_coeff * envelope + (1.0 - attack_coeff) * detection
                    else:
                        envelope = release_coeff * envelope + (1.0 - release_coeff) * detection
                    
                    if envelope > threshold_linear:
                        target_gain = threshold_linear + (envelope - threshold_linear) * reduction_linear
                        target_gain = target_gain / envelope if envelope > 0 else 1.0
                    else:
                        target_gain = 1.0
                    
                    if target_gain < gain_smooth:
                        gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
                    else:
                        gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain
                    
                    output[i, ch] = audio_data[i, ch] * gain_smooth
            
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"De-esser error: {e}")
        return audio_data

def apply_highpass_filter(audio_data, sample_rate, cutoff_freq=80):
    """Apply high-pass filter."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        cutoff_freq = np.clip(cutoff_freq, 20, 500)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
        
        sos = butter(4, normalized_cutoff, btype='high', output='sos')
        
        if audio_data.ndim == 1:
            return sosfilt(sos, audio_data).astype(np.float32)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = sosfilt(sos, audio_data[:, ch])
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"Highpass filter error: {e}")
        return audio_data

def apply_parametric_eq(audio_data, sample_rate, frequency, gain_db, q_factor=1.0):
    """Apply parametric EQ band."""
    try:
        if audio_data.size == 0 or abs(gain_db) < 0.1:
            return audio_data
        
        frequency = np.clip(frequency, 20, sample_rate/2 * 0.95)
        gain_db = np.clip(gain_db, -24, 24)
        q_factor = np.clip(q_factor, 0.1, 10)
        
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * frequency / sample_rate
        alpha = np.sin(w0) / (2 * q_factor)
        
        cos_w0 = np.cos(w0)
        
        if gain_db >= 0:
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        else:
            b0 = 1 + alpha / A
            b1 = -2 * cos_w0
            b2 = 1 - alpha / A
            a0 = 1 + alpha * A
            a1 = -2 * cos_w0
            a2 = 1 - alpha * A
        
        sos = np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])
        
        if audio_data.ndim == 1:
            return sosfilt(sos, audio_data).astype(np.float32)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = sosfilt(sos, audio_data[:, ch])
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"Parametric EQ error: {e}")
        return audio_data

@numba.jit(nopython=True)
def _compress_channel(audio_channel, threshold_linear, ratio, attack_coeff, release_coeff, makeup_gain_linear):
    """Optimized compression on single channel."""
    envelope = 0.0
    gain_smooth = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        detection = abs(sample)
        if detection > envelope:
            envelope = attack_coeff * envelope + (1.0 - attack_coeff) * detection
        else:
            envelope = release_coeff * envelope + (1.0 - release_coeff) * detection
        
        if envelope > threshold_linear:
            over_amount = envelope - threshold_linear
            compressed_over = over_amount / ratio
            target_envelope = threshold_linear + compressed_over
            target_gain = target_envelope / envelope if envelope > 0 else 1.0
        else:
            target_gain = 1.0
        
        if target_gain < gain_smooth:
            gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
        else:
            gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain
        
        output[i] = sample * gain_smooth * makeup_gain_linear
    
    return output

def apply_compression(audio_data, sample_rate, threshold_db=-20, ratio=4.0, attack_ms=10, release_ms=100, makeup_gain_db=0):
    """Professional compressor."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -60, 0)
        ratio = np.clip(ratio, 1.0, 20.0)
        attack_ms = np.clip(attack_ms, 0.1, 100)
        release_ms = np.clip(release_ms, 1, 1000)
        makeup_gain_db = np.clip(makeup_gain_db, 0, 24)
        
        threshold_linear = 10 ** (threshold_db / 20)
        makeup_gain_linear = 10 ** (makeup_gain_db / 20)
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        
        if audio_data.ndim == 1:
            return _compress_channel(audio_data, threshold_linear, ratio, attack_coeff, release_coeff, makeup_gain_linear).astype(np.float32)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = _compress_channel(audio_data[:, ch], threshold_linear, ratio, attack_coeff, release_coeff, makeup_gain_linear)
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"Compression error: {e}")
        return audio_data

def apply_limiter(audio_data, target_lufs=-14, ceiling_db=-0.5):
    """Limiter with loudness normalization."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        target_lufs = np.clip(target_lufs, -30, 0)
        ceiling_db = np.clip(ceiling_db, -3, 0)
        ceiling_linear = 10 ** (ceiling_db / 20)
        
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-6:
            return audio_data
        
        target_rms = 10 ** ((target_lufs + 3) / 20)
        gain = target_rms / rms
        
        audio_data = audio_data * gain
        audio_data = np.clip(audio_data, -ceiling_linear, ceiling_linear)
        
        return audio_data.astype(np.float32)
    except Exception as e:
        logging.error(f"Limiter error: {e}")
        return audio_data

# ==================== MAIN PROCESSING FUNCTIONS ====================

def apply_ai_decisions(audio_data, sample_rate, ai_decisions):
    """Apply all AI decisions to audio."""
    try:
        logging.info("🎛️ Applying AI decisions to audio...")
        
        # RESTORATION
        if 'restoration' in ai_decisions:
            restoration = ai_decisions['restoration']
            
            if 'noise_reduction' in restoration and restoration['noise_reduction'].get('apply', True):
                nr = restoration['noise_reduction']
                threshold = nr.get('threshold_db', -60)
                ratio = nr.get('ratio', 10)
                logging.info(f"  Noise Gate: {threshold}dB, ratio {ratio}:1")
                audio_data = apply_noise_gate(audio_data, threshold, ratio, sample_rate=sample_rate)
            
            if 'deessing' in restoration and restoration['deessing'].get('apply', False):
                de = restoration['deessing']
                freq = de.get('frequency', 6000)
                thresh = de.get('threshold_db', -15)
                ratio = de.get('ratio', 3)
                logging.info(f"  De-esser: {freq}Hz, {thresh}dB, ratio {ratio}:1")
                audio_data = apply_deesser(audio_data, sample_rate, freq, thresh, ratio)
        
        # MIXING
        if 'mixing' in ai_decisions:
            mixing = ai_decisions['mixing']
            
            if 'highpass' in mixing and mixing['highpass'].get('apply', True):
                hp_freq = mixing['highpass'].get('frequency', 80)
                logging.info(f"  Highpass: {hp_freq}Hz")
                audio_data = apply_highpass_filter(audio_data, sample_rate, hp_freq)
            
            if 'equalizer' in mixing and 'bands' in mixing['equalizer']:
                for band in mixing['equalizer']['bands']:
                    if band.get('apply', True):
                        freq = band.get('frequency', 1000)
                        gain = band.get('gain_db', 0)
                        q = band.get('q_factor', 1.0)
                        logging.info(f"  EQ: {freq}Hz {gain:+.1f}dB Q={q}")
                        audio_data = apply_parametric_eq(audio_data, sample_rate, freq, gain, q)
            
            if 'compression' in mixing and mixing['compression'].get('apply', True):
                comp = mixing['compression']
                thresh = comp.get('threshold_db', -20)
                ratio_str = comp.get('ratio', '4:1')
                ratio = float(ratio_str.split(':')[0]) if ':' in ratio_str else 4.0
                attack = comp.get('attack_ms', 10)
                release = comp.get('release_ms', 100)
                makeup = comp.get('makeup_gain_db', 2)
                logging.info(f"  Compression: {thresh}dB, {ratio}:1, A={attack}ms R={release}ms")
                audio_data = apply_compression(audio_data, sample_rate, thresh, ratio, attack, release, makeup)
        
        # MASTERING
        if 'mastering' in ai_decisions:
            mastering = ai_decisions['mastering']
            
            if 'limiting' in mastering and mastering['limiting'].get('apply', True):
                lim = mastering['limiting']
                target = lim.get('target_lufs', -14)
                ceiling = lim.get('ceiling_db', -0.5)
                logging.info(f"  Limiter: {target} LUFS, ceiling {ceiling}dB")
                audio_data = apply_limiter(audio_data, target, ceiling)
        
        logging.info("✅ AI decisions applied successfully")
        return audio_data
        
    except Exception as e:
        logging.error(f"Error applying AI decisions: {e}", exc_info=True)
        return audio_data

# ==================== ASYNC STEM SEPARATION ====================

def process_stems_async(input_path, callback_url, project_id, service_key, app_id):
    """Background thread: Run Demucs, upload stems, callback."""
    try:
        logging.info(f"🎸 Starting Demucs separation for {project_id}")
        
        # Run Demucs
        result = subprocess.run([
            'demucs', '--two-stems=vocals', '-o', OUTPUT_FOLDER, input_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Demucs failed: {result.stderr}")
        
        # Find output
        demucs_output = os.path.join(OUTPUT_FOLDER, 'htdemucs')
        if not os.path.exists(demucs_output):
            demucs_output = os.path.join(OUTPUT_FOLDER, 'mdx_extra')
        
        output_dirs = [d for d in os.listdir(demucs_output) if os.path.isdir(os.path.join(demucs_output, d))]
        if not output_dirs:
            raise Exception('No Demucs output')
        
        stem_folder = os.path.join(demucs_output, output_dirs[0])
        
        # Upload stems to Base44
        stems_urls = {}
        for stem in ['vocals', 'drums', 'bass', 'other']:
            wav_path = os.path.join(stem_folder, f"{stem}.wav")
            if not os.path.exists(wav_path):
                continue
            
            # Convert to MP3
            mp3_path = wav_path.replace('.wav', '.mp3')
            subprocess.run(['ffmpeg', '-i', wav_path, '-b:a', '192k', mp3_path, '-y'], capture_output=True)
            
            # Upload
            url = upload_file_to_base44(mp3_path, f"{stem}.mp3", service_key, app_id)
            if url:
                stems_urls[stem] = url
            
            os.remove(wav_path)
            os.remove(mp3_path)
        
        os.remove(input_path)
        
        # Callback to Deno
        logging.info(f"📞 Calling back to {callback_url}")
        requests.post(callback_url, json={
            'success': True,
            'project_id': project_id,
            'stems_urls': stems_urls
        }, timeout=30)
        
        logging.info(f"✅ Separation complete for {project_id}")
        
    except Exception as e:
        logging.error(f"❌ Async separation failed: {e}")
        try:
            requests.post(callback_url, json={
                'success': False,
                'project_id': project_id,
                'error': str(e)
            }, timeout=30)
        except:
            pass

# ==================== FLASK ROUTES ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'audio-processing-pro',
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER
    })

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """ASYNC endpoint: Separate stems, upload to Base44, callback when done."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_api_key = request.form.get('base44_api_key')
        base44_app_id = request.form.get('base44_app_id')
        base44_service_key = request.form.get('base44_service_key')
        
        logging.info(f"Received separation request:")
        logging.info(f"  project_id: {project_id}")
        logging.info(f"  callback_url: {callback_url}")
        logging.info(f"  has api_key: {bool(base44_api_key)}")
        logging.info(f"  app_id: {base44_app_id}")
        
        if not all([callback_url, project_id, base44_api_key, base44_app_id, base44_service_key]):
            return jsonify({
                'error': 'Missing required fields',
                'received': {
                    'callback_url': bool(callback_url),
                    'project_id': bool(project_id),
                    'base44_api_key': bool(base44_api_key),
                    'base44_app_id': bool(base44_app_id),
                    'base44_service_key': bool(base44_service_key)
                }
            }), 400
        
        input_path = os.path.join(UPLOAD_FOLDER, f"sep_{uuid.uuid4().hex}.mp3")
        file.save(input_path)
        
        thread = threading.Thread(
            target=process_stems_async,
            args=(input_path, callback_url, project_id, base44_service_key, base44_app_id)
        )
        thread.start()
        
        return jsonify({'success': True, 'project_id': project_id}), 200
        
    except Exception as e:
        logging.error(f"Error starting separation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with AI decisions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        if 'ai_decisions' not in request.form:
            return jsonify({'error': 'No AI decisions provided'}), 400
        
        file = request.files['file']
        ai_decisions = json.loads(request.form['ai_decisions'])
        
        input_path = os.path.join(UPLOAD_FOLDER, f"process_{uuid.uuid4().hex}.wav")
        file.save(input_path)
        
        wav_path = input_path
        if not input_path.endswith('.wav'):
            wav_path = input_path.replace(os.path.splitext(input_path)[1], '.wav')
            convert_to_wav(input_path, wav_path)
        
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None:
            return jsonify({'error': 'Failed to read audio'}), 500
        
        audio_data = apply_ai_decisions(audio_data, sample_rate, ai_decisions)
        
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{uuid.uuid4().hex}.wav")
        write_wav(output_path, audio_data, sample_rate)
        
        with open(output_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(input_path)
        if wav_path != input_path:
            os.remove(wav_path)
        os.remove(output_path)
        
        return jsonify({'success': True, 'processed_audio_base64': audio_base64})
        
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
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
    app.run(host='0.0.0.0', port=5000, debug=False)




