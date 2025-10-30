import os
import uuid
import numpy as np
import soundfile as sf
# Use specific imports from scipy.signal for clarity
from scipy.signal import butter, sosfilt, sosfiltfilt, tf2sos, iirpeak, iirnotch
from scipy import signal # Keep for potential other uses
import logging
from pydub import AudioSegment
import librosa # Keep, may be needed by demucs dependencies
import tempfile
import shutil
import numba # <-- IMPORT IS HERE
import subprocess
import sys
import json # For parsing instructions
import atexit
import base64 # <-- NEW IMPORT, GOOD
import threading # <-- NEW IMPORT for Async
import requests  # <-- NEW IMPORT for Callback

from flask import Flask, request, jsonify

app = Flask(__name__)
# Enhanced logging format for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_uploads_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_outputs_')
logging.info(f"Using UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logging.info(f"Using OUTPUT_FOLDER: {OUTPUT_FOLDER}")

# ==================== HELPER FUNCTIONS ====================

def check_audio_level(audio_data, label="Audio"):
    """Log audio level information for debugging."""
    if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
        logging.warning(f"{label}: EMPTY OR INVALID ARRAY")
        return 0.0, 0.0 # Return zeros for safety
    
    try:
        rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2)) # Use float64 for mean calculation stability
        peak = np.max(np.abs(audio_data))
        
        logging.info(f"{label} - RMS: {rms:.6f}, Peak: {peak:.6f}, Shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        
        if peak < 1e-5: # Use a small threshold for near silence
            logging.warning(f"{label}: AUDIO IS SILENT OR NEARLY SILENT!")
        
        return rms, peak
    except Exception as e:
        logging.error(f"Error in check_audio_level for {label}: {e}")
        return 0.0, 0.0

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

# ==================== DSP FUNCTIONS (High Quality) ====================

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
            return _gate_channel(audio_data, threshold_linear, attack_coeff, release_coeff, ratio)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = _gate_channel(audio_data[:, ch], threshold_linear, attack_coeff, release_coeff, ratio)
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"Noise gate error: {e}")
        return audio_data


def apply_parametric_eq(audio_data, sample_rate, frequency, gain_db, q_factor=1.0, filter_type='peak'):
    """Professional parametric EQ with multiple filter types."""
    try:
        if audio_data.size == 0 or abs(gain_db) < 0.01:
            return audio_data
        
        frequency = np.clip(frequency, 20, sample_rate / 2 - 1)
        gain_db = np.clip(gain_db, -24, 24)
        q_factor = np.clip(q_factor, 0.1, 10.0)
        
        w0 = 2 * np.pi * frequency / sample_rate
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * q_factor)
        
        if filter_type == 'peak':
            b0 = 1 + alpha * A; b1 = -2 * np.cos(w0); b2 = 1 - alpha * A
            a0 = 1 + alpha / A; a1 = -2 * np.cos(w0); a2 = 1 - alpha / A
        
        elif filter_type == 'lowshelf':
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        
        elif filter_type == 'highshelf':
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        
        elif filter_type == 'lowpass':
            b0 = (1 - np.cos(w0)) / 2; b1 = 1 - np.cos(w0); b2 = (1 - np.cos(w0)) / 2
            a0 = 1 + alpha; a1 = -2 * np.cos(w0); a2 = 1 - alpha
        
        elif filter_type == 'highpass':
            b0 = (1 + np.cos(w0)) / 2; b1 = -(1 + np.cos(w0)); b2 = (1 + np.cos(w0)) / 2
            a0 = 1 + alpha; a1 = -2 * np.cos(w0); a2 = 1 - alpha
        
        elif filter_type == 'notch':
            b0 = 1; b1 = -2 * np.cos(w0); b2 = 1
            a0 = 1 + alpha; a1 = -2 * np.cos(w0); a2 = 1 - alpha
        
        else:
            logging.warning(f"Unknown EQ filter type: {filter_type}, skipping.")
            return audio_data
        
        sos = np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])
        
        if audio_data.ndim == 1:
            return sosfiltfilt(sos, audio_data).astype(np.float32)
        else:
            result = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                result[:, ch] = sosfiltfilt(sos, audio_data[:, ch])
            return result.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Parametric EQ error: {e}")
        return audio_data


@numba.jit(nopython=True)
def compress_channel(audio_channel, threshold_linear, ratio, attack_coeff, release_coeff, knee_db, makeup_gain_linear):
    """Optimized compression with soft knee (Corrected dB logic)."""
    envelope_db = -100.0 # Track envelope in dB
    gain_smooth = 1.0 # Track gain smoothing factor
    output = np.zeros_like(audio_channel)
    inv_ratio_minus_one = (1.0 / ratio) - 1.0
    threshold_db = 20.0 * np.log10(threshold_linear)
    knee_width_db_half = knee_db / 2.0

    for i, sample in enumerate(audio_channel):
        input_level_db = 20.0 * np.log10(max(abs(sample), 1e-15))
        if input_level_db > envelope_db:
            envelope_db = attack_coeff * envelope_db + (1.0 - attack_coeff) * input_level_db
        else:
            envelope_db = release_coeff * envelope_db + (1.0 - release_coeff) * input_level_db

        gain_reduction_db = 0.0
        overshoot_db = envelope_db - threshold_db

        if knee_db > 0 and abs(overshoot_db) <= knee_width_db_half:
            # Inside soft knee
            gain_reduction_db = inv_ratio_minus_one * ((overshoot_db + knee_width_db_half)**2) / (2.0 * knee_db)
        elif overshoot_db > knee_width_db_half:
            # Above hard knee
            gain_reduction_db = inv_ratio_minus_one * overshoot_db

        target_gain = 10.0**(gain_reduction_db / 20.0)
        
        if target_gain < gain_smooth:
            gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
        else:
            gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain
        
        gain_to_apply = max(0.0, min(gain_smooth, 1.0))
        output[i] = sample * gain_to_apply * makeup_gain_linear
    
    return output


def apply_compressor(audio_data, sample_rate, threshold_db=-20, ratio=4.0, attack_ms=5, release_ms=50, knee_db=6, makeup_gain_db=0):
    """Professional compressor with soft knee."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -60, 0)
        ratio = np.clip(ratio, 1, 20)
        attack_ms = np.clip(attack_ms, 0.1, 100)
        release_ms = np.clip(release_ms, 1, 1000)
        knee_db = np.clip(knee_db, 0, 12)
        makeup_gain_db = np.clip(makeup_gain_db, 0, 24)
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        makeup_gain_linear = 10 ** (makeup_gain_db / 20)
        
        if audio_data.ndim == 1:
            return compress_channel(audio_data, threshold_linear, ratio, attack_coeff, release_coeff, knee_db, makeup_gain_linear)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = compress_channel(audio_data[:, ch], threshold_linear, ratio, attack_coeff, release_coeff, knee_db, makeup_gain_linear)
            return output.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Compressor error: {e}")
        return audio_data


@numba.jit(nopython=True)
def limit_channel(audio_channel, threshold_linear, release_coeff):
    """Fast brick-wall limiter."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        target_gain = 1.0
        abs_sample = abs(sample)
        if abs_sample * gain > threshold_linear:
            target_gain = threshold_linear / max(abs_sample, 1e-10)
        
        if target_gain < gain:
            gain = target_gain # Instant attack
        else:
            gain = release_coeff * gain + (1.0 - release_coeff) * target_gain # Release
        
        output[i] = sample * gain
    
    return output


def apply_limiter(audio_data, sample_rate, threshold_db=-1.0, release_ms=50):
    """Brick-wall limiter."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -12, 0)
        release_ms = np.clip(release_ms, 1, 500)
        
        threshold_linear = 10 ** (threshold_db / 20)
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        
        if audio_data.ndim == 1:
            return limit_channel(audio_data, threshold_linear, release_coeff)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = limit_channel(audio_data[:, ch], threshold_linear, release_coeff)
            return output.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Limiter error: {e}")
        return audio_data


@numba.jit(nopython=True)
def _deesser_channel(audio_channel, hf_env, threshold_linear, ratio, attack_coeff, release_coeff):
    """De-esser core logic."""
    gain_reduction = 1.0
    output = np.zeros_like(audio_channel)
    for i in range(len(audio_channel)):
        detection = abs(hf_env[i])
        target_gain = 1.0
        if detection > threshold_linear:
            overshoot = detection - threshold_linear
            compressed_overshoot = overshoot / ratio
            target_gain = (threshold_linear + compressed_overshoot) / detection
        if target_gain < gain_reduction:
            gain_reduction = attack_coeff * gain_reduction + (1.0 - attack_coeff) * target_gain
        else:
            gain_reduction = release_coeff * gain_reduction + (1.0 - release_coeff) * target_gain
        output[i] = audio_channel[i] * gain_reduction
    return output

def apply_deesser(audio_data, sample_rate, frequency=6000, threshold_db=-15, ratio=3, attack_ms=1, release_ms=10):
    """De-esser for controlling harsh sibilance."""
    try:
        if audio_data.size == 0: return audio_data
        frequency = np.clip(frequency, 2000, 12000)
        threshold_db = np.clip(threshold_db, -40, 0)
        ratio = np.clip(ratio, 1, 10)
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        nyquist = sample_rate / 2
        normalized_freq = frequency / nyquist
        sos = butter(4, normalized_freq, btype='high', output='sos')
        if audio_data.ndim == 1:
            hf_signal = sosfilt(sos, audio_data)
            return _deesser_channel(audio_data, hf_signal, threshold_linear, ratio, attack_coeff, release_coeff)
        else:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                hf_signal = sosfilt(sos, audio_data[:, ch])
                output[:, ch] = _deesser_channel(audio_data[:, ch], hf_signal, threshold_linear, ratio, attack_coeff, release_coeff)
            return output.astype(np.float32)
    except Exception as e:
        logging.error(f"De-esser error: {e}")
        return audio_data

def normalize_loudness(audio_data, sample_rate, target_lufs=-14.0):
    """Normalize audio to target LUFS (approximate)."""
    try:
        if audio_data.size == 0: return audio_data
        target_lufs = np.clip(target_lufs, -30, -5)
        
        # Use librosa for a much more accurate RMS (closer to perceptual loudness)
        # Note: This is still a very rough approximation of LUFS.
        rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
        avg_rms = np.mean(rms)
        
        if avg_rms < 1e-10:
             logging.warning("Audio is silent, skipping loudness normalization")
             return audio_data

        current_lufs_approx = 20 * np.log10(avg_rms) + 10.0 # Heuristic offset
        
        gain_db = target_lufs - current_lufs_approx
        gain_db = np.clip(gain_db, -24, 24)
        gain_linear = 10 ** (gain_db / 20)
        
        normalized = audio_data * gain_linear
        # Apply a final limiter, not just a clip
        normalized = apply_limiter(normalized, sample_rate, threshold_db=-1.0)
        
        logging.info(f"Normalized: {current_lufs_approx:.1f} LUFS (approx) -> {target_lufs:.1f} LUFS (gain: {gain_db:.1f} dB)")
        return normalized.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Loudness normalization error: {e}")
        return audio_data


# ==================== STEM SEPARATION (ASYNC + AUTH FIX) ====================

def process_stems_async(wav_path, output_dir, callback_url, project_id, base44_api_key, base44_app_id, input_path_to_clean, wav_path_to_clean, output_dir_to_clean):
    """Background processing function that calls back when done."""
    try:
        logging.info(f"🎵 Running demucs with HIGH QUALITY settings (htdemucs_ft) for {wav_path}...")
        
        subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs_ft', # <-- Using high-quality model
            '-o', output_dir,
            '--mp3',
            '--mp3-bitrate', '192',
            '--jobs', '2',
            wav_path
        ], check=True, capture_output=True, text=True, timeout=840) # 14 minute timeout
        
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

        stems_data = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_file_path = os.path.join(stem_track_dir, f"{stem_name}.mp3")
            if os.path.exists(stem_file_path):
                with open(stem_file_path, 'rb') as f:
                    stem_bytes = f.read()
                    stems_data[stem_name] = base64.b64encode(stem_bytes).decode('utf-8')
                logging.info(f"✅ Encoded {stem_name}.mp3 to base64")
            else:
                logging.warning(f"⚠️ Missing {stem_name}.mp3 at {stem_file_path}")
        
        if not stems_data:
            raise ValueError("No stems were successfully separated and encoded.")

        logging.info("🎉 Stem separation complete! Calling back to Deno...")
        
        # --- Use passed-in API Key for callback ---
        callback_headers = {
            "Authorization": f"Bearer {base44_api_key}",
            "Base44-App-Id": base44_app_id,
            "Content-Type": "application/json"
        }
        callback_data = {
            "success": True,
            "project_id": project_id,
            "stems_base64": stems_data
        }

        callback_response = requests.post(
            callback_url, 
            json=callback_data, 
            headers=callback_headers, 
            timeout=60
        )
        
        if callback_response.status_code >= 200 and callback_response.status_code < 300:
            logging.info(f"✅ Callback successful: {callback_response.text}")
        else:
            logging.error(f"❌ Callback failed: {callback_response.status_code} - {callback_response.text}")
    
    except Exception as e:
        logging.error(f"❌ Async processing error: {e}", exc_info=True)
        # --- Use passed-in API Key for ERROR callback ---
        try:
            callback_headers = {
                "Authorization": f"Bearer {base44_api_key}",
                "Base44-App-Id": base44_app_id,
                "Content-Type": "application/json"
            }
            requests.post(callback_url, json={
                "success": False,
                "project_id": project_id,
                "error": str(e)
            }, headers=callback_headers, timeout=60)
        except Exception as callback_e:
            logging.error(f"❌ Failed to send error callback: {callback_e}")
    
    finally:
        # Cleanup
        try:
            if input_path_to_clean and os.path.exists(input_path_to_clean):
                os.remove(input_path_to_clean)
            if wav_path_to_clean and os.path.exists(wav_path_to_clean) and wav_path_to_clean != input_path_to_clean:
                os.remove(wav_path_to_clean)
            if output_dir_to_clean and os.path.exists(output_dir_to_clean):
                shutil.rmtree(output_dir_to_clean)
            logging.info(f"Stem separation cleanup complete.")
        except Exception as cleanup_e:
            logging.error(f"Error during async cleanup: {cleanup_e}", exc_info=True)

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    Separates audio into stems using demucs.
    NOW ASYNC: Accepts job, responds immediately, then processes and calls back.
    """
    input_path = None
    wav_path = None
    output_dir = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        
        # --- FIX: Get auth details from Base44 request FORM DATA ---
        callback_url = request.form.get('callback_url')
        project_id = request.form.get('project_id')
        base44_api_key = request.form.get('base44_api_key') # Base44 Deno must send this
        base44_app_id = request.form.get('base44_app_id')     # Base44 Deno must send this
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not all([callback_url, project_id, base44_api_key, base44_app_id]):
            logging.error(f"Missing required form fields. Got project_id: {project_id}, callback_url: {callback_url}, api_key: {bool(base44_api_key)}, app_id: {bool(base44_app_id)}")
            return jsonify({"error": "Missing required fields: callback_url, project_id, base44_api_key, or base44_app_id"}), 400
        
        unique_id = str(uuid.uuid4())
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.wav")
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
        
        file.save(input_path)
        logging.info(f"📁 Saved input file: {input_path}")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert input to WAV")
        else:
            shutil.copy(input_path, wav_path) # Use copy to be safe
        
        os.makedirs(output_dir, exist_ok=True)
        
        # RESPOND IMMEDIATELY - Job accepted
        logging.info(f"✅ Job accepted for project {project_id}. Starting background processing...")
        
        # Start background processing in a thread
        thread = threading.Thread(
            target=process_stems_async,
            args=(
                wav_path, output_dir, callback_url, project_id, 
                base44_api_key, base44_app_id, # Pass auth details
                input_path, wav_path, output_dir # Pass paths for cleanup
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Stem separation job started",
            "project_id": project_id
        }), 202  # 202 Accepted
    
    except Exception as e:
        logging.error(f"❌ Job acceptance error: {e}", exc_info=True)
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
        
        return jsonify({"error": str(e)}), 500


# ==================== MAIN PROCESSING ====================
@app.route('/process', methods=['POST'])
def process_audio():
    """Process a single audio stem with AI decisions."""
    input_path = None
    wav_path = None
    processed_wav_path = None
    unique_id = str(uuid.uuid4()) # Added for logging
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        if 'ai_decisions' not in request.form:
            return jsonify({"error": "No AI decisions"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        ai_decisions = json.loads(request.form['ai_decisions'])
        
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_converted.wav")
        processed_wav_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_processed.wav")
        
        file.save(input_path)
        logging.info(f"Saved input file: {input_path}")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert to WAV")
        else:
            wav_path = input_path # Use original file if already WAV
        
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None:
            raise ValueError("Failed to read WAV audio")
        
        if audio_data.size == 0:
            raise ValueError("Empty audio file after conversion")
        
        logging.info(f"🎵 Starting DSP processing: SR={sample_rate}, Shape={audio_data.shape}")
        current_audio = audio_data.copy()
        
        # --- Helper to get params ---
        def get_params(phase_name, effect_name):
             return ai_decisions.get(phase_name, {}).get(effect_name, {})

        # RESTORATION
        if ai_decisions.get('restoration'):
            logging.info("🧹 Applying RESTORATION effects...")
            params = get_params('restoration', 'noise_reduction')
            if params and params.get('apply', False):
                 current_audio = apply_noise_gate(current_audio,
                     threshold_db=params.get('threshold_db', -60),
                     ratio=params.get('ratio', 10),
                     sample_rate=sample_rate)

            params = get_params('restoration', 'deessing')
            if params and params.get('apply', False):
                current_audio = apply_deesser(current_audio,
                    sample_rate=sample_rate,
                    frequency=params.get('frequency', 6000),
                    threshold_db=params.get('threshold_db', -15),
                    ratio=params.get('ratio', 3))
        
        # MIXING
        if ai_decisions.get('mixing'):
            logging.info("🎛️ Applying MIXING effects...")
            
            hp_params = get_params('mixing', 'highpass')
            if hp_params is None or hp_params.get('apply', True):
                hp_freq = hp_params.get('frequency', 80) if hp_params else 80
                current_audio = apply_parametric_eq(current_audio, sample_rate, frequency=hp_freq, gain_db=0, q_factor=1.0, filter_type='highpass')

            eq_params = get_params('mixing', 'equalizer')
            if eq_params and 'bands' in eq_params and isinstance(eq_params['bands'], list):
                 logging.info(f"Applying {len(eq_params['bands'])} EQ bands...")
                 for band in eq_params['bands'][:8]: # Limit bands
                     if isinstance(band, dict) and band.get('apply', True):
                         current_audio = apply_parametric_eq(current_audio,
                             sample_rate=sample_rate,
                             frequency=band.get('frequency', 1000),
                             gain_db=band.get('gain_db', 0),
                             q_factor=band.get('q_factor', 1.0),
                             filter_type=band.get('type', 'peak'))
            
            comp_params = get_params('mixing', 'compression')
            if comp_params and comp_params.get('apply', False):
                ratio_val = 4.0
                ratio_str = comp_params.get('ratio', '4:1')
                try: ratio_val = float(ratio_str.split(':')[0])
                except: pass
                current_audio = apply_compressor(current_audio,
                    sample_rate=sample_rate,
                    threshold_db=comp_params.get('threshold_db', -18),
                    ratio=ratio_val,
                    attack_ms=comp_params.get('attack_ms', 10),
                    release_ms=comp_params.get('release_ms', 100),
                    makeup_gain_db=comp_params.get('makeup_gain_db', 0),
                    knee_db=comp_params.get('knee_db', 3))

        # MASTERING
        if ai_decisions.get('mastering'):
            logging.info("✨ Applying MASTERING effects...")
            
            limit_params = get_params('mastering', 'limiting')
            if limit_params and limit_params.get('apply', True):
                current_audio = normalize_loudness(current_audio,
                    sample_rate=sample_rate,
                    target_lufs=limit_params.get('target_lufs', -14))
                current_audio = apply_limiter(current_audio,
                    sample_rate=sample_rate,
                    threshold_db=limit_params.get('ceiling_db', -0.5),
                    release_ms=limit_params.get('release_ms', 50))
            else:
                 current_audio = apply_limiter(current_audio, sample_rate=sample_rate, threshold_db=-0.5)

        
        current_audio = np.clip(current_audio, -1.0, 1.0)

        if not write_wav(processed_wav_path, current_audio, sample_rate):
            raise ValueError("Failed to write processed WAV")
        
        with open(processed_wav_path, 'rb') as f:
            processed_bytes = f.read()
            processed_base64 = base64.b64encode(processed_bytes).decode('utf-8')

        logging.info("🎉 DSP processing complete and audio encoded!")
        return jsonify({"processed_audio_base64": processed_base64, "format": "wav"}), 200
    
    except Exception as e:
        logging.error(f"❌ Processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files and directories
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and os.path.exists(wav_path) and wav_path != input_path:
                os.remove(wav_path)
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
            logging.info(f"Processing cleanup complete for {unique_id}")
        except Exception as cleanup_e:
            logging.error(f"Error during cleanup: {cleanup_e}", exc_info=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check."""
    return jsonify({"status": "healthy", "service": "AI Studio Pro"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root."""
    return jsonify({
        "service": "AI Studio Pro Audio Processing",
        "version": "4.0.0", # <-- Version bump for new auth
        "optimizations": "Async demucs, base64 data transfer, callback auth",
        "endpoints": {
            "POST /separate_stems": "Separate audio into stems, returns 202, calls back with base64 data",
            "POST /process": "Process audio with AI DSP, returns base64 encoded WAV",
            "GET /health": "Health check"
        }
    }), 200

# Helper function for cleanup on error
def cleanup_temp_dirs_on_error(unique_id, *paths):
    logging.warning(f"Request {unique_id}: Cleaning up temp files due to error...")
    cleaned_count = 0
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
                logging.info(f"Request {unique_id}: Cleaned up errored file: {os.path.basename(p)}")
                cleaned_count += 1
            except Exception as e:
                logging.error(f"Request {unique_id}: Error cleaning up errored file {os.path.basename(p)}: {e}")
    logging.warning(f"Request {unique_id}: Error cleanup finished. Removed {cleaned_count} files.")

if __name__ == '__main__':
    # Register cleanup for when the Flask app exits
    def cleanup_temp_dirs():
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    logging.info(f"Cleaned up temporary directory: {folder}")
                except Exception as e:
                    logging.error(f"Failed to clean up {folder}: {e}")

    atexit.register(cleanup_temp_dirs)
    
    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask server on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)







