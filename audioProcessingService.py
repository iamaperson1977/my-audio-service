import os
import uuid
import numpy as np
import soundfile as sf
# Use specific imports from scipy.signal for clarity
from scipy.signal import butter, lfilter, iirpeak, iirnotch
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

from flask import Flask, request, jsonify

app = Flask(__name__)
# Enhanced logging format for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')
logging.info(f"Using UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logging.info(f"Using OUTPUT_FOLDER: {OUTPUT_FOLDER}")

# ==================== FILE FORMAT HANDLING ====================

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(44100) # Ensure consistent sample rate
        audio.export(output_path, format='wav')
        logging.info(f"Converted {input_path} to WAV: channels={audio.channels}, sr=44100")
        return True
    except Exception as e:
        logging.error(f"Error converting to WAV: {e}")
        return False

def convert_from_wav(wav_path, output_path, output_format='mp3'):
    """Convert WAV to other formats."""
    try:
        audio = AudioSegment.from_wav(wav_path)
        
        if output_format == 'mp3':
            audio.export(output_path, format='mp3', bitrate='192k') # Optimized bitrate
        elif output_format == 'flac':
            audio.export(output_path, format='flac')
        elif output_format in ['aac', 'm4a']:
            audio.export(output_path, format='aac', codec='aac', bitrate='256k')
        elif output_format == 'ogg':
            audio.export(output_path, format='ogg', codec='libvorbis')
        else:
            audio.export(output_path, format='wav') # Default to wav if format unknown
        
        logging.info(f"Converted WAV to {output_format}")
        return True
    except Exception as e:
        logging.error(f"Error converting from WAV: {e}")
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
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1.0 / (max(attack_ms, 0.1) * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (max(release_ms, 1.0) * sample_rate / 1000.0))
        ratio = max(ratio, 1.0)
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                 output[:, ch] = _gate_channel(audio_data[:, ch], threshold_linear, attack_coeff, release_coeff, ratio)
        else:
            output = _gate_channel(audio_data, threshold_linear, attack_coeff, release_coeff, ratio)
        
        logging.info(f"Applied noise gate: {threshold_db}dB, ratio={ratio}:1")
        return output
    except Exception as e:
        logging.error(f"Noise gate error: {e}")
        return audio_data

def apply_parametric_eq(audio_data, frequency, gain_db, q_factor, filter_type='peak', sample_rate=44100):
    """Professional parametric EQ (FIXED LOGIC)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        frequency = np.clip(frequency, 20, nyquist - 10) 
        q_factor = np.clip(q_factor, 0.1, 20.0) 
        gain_db = np.clip(gain_db, -24, 24)
        
        # --- CORRECTED Biquad Filter Design ---
        A = 10**(gain_db / 40.0)     # Linear gain
        w0 = 2 * np.pi * frequency / sample_rate
        alpha = np.sin(w0) / (2 * q_factor)

        if filter_type == 'peak':
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
        elif filter_type == 'lowshelf':
            beta = np.sqrt(A**2 + A**2) / q_factor # approx
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0)
        elif filter_type == 'highshelf':
            beta = np.sqrt(A**2 + A**2) / q_factor # approx
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0)
        elif filter_type == 'lowpass':
            b0 = (1 - np.cos(w0)) / 2
            b1 = 1 - np.cos(w0)
            b2 = (1 - np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        elif filter_type == 'highpass':
            b0 = (1 + np.cos(w0)) / 2
            b1 = -(1 + np.cos(w0))
            b2 = (1 + np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        elif filter_type == 'notch':
            b0 = 1
            b1 = -2 * np.cos(w0)
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        else:
            logging.warning(f"Unknown filter type: {filter_type}, skipping EQ.")
            return audio_data
            
        # Normalize coefficients by a0
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        
        # Apply filter
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = lfilter(b, a, audio_data[:, ch])
        else:
            output = lfilter(b, a, audio_data)
        
        logging.info(f"Applied EQ: {filter_type} @ {frequency}Hz, {gain_db:+.1f}dB")
        return output
    except Exception as e:
        logging.error(f"EQ error: {e}")
        return audio_data

@numba.jit(nopython=True)
def compress_channel(audio_channel, threshold_linear, ratio, attack_coeff, release_coeff, makeup_gain, knee_db):
    """Compress single channel with soft knee (Corrected dB logic)."""
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

        # --- 💥 CRITICAL FIX HERE 💥 ---
        # Replace np.clip(gain, 0.0, 1.0) with standard min/max
        gain_to_apply = max(0.0, min(gain_smooth, 1.0))
        # --- 💥 END OF FIX 💥 ---

        output[i] = sample * gain_to_apply * makeup_gain
    return output

def apply_compressor(audio_data, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50, 
                    makeup_gain_db=0, knee_db=0, sample_rate=44100):
    """Professional compressor (uses corrected logic)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -60, 0)
        ratio = np.clip(ratio, 1.0, 20.0)
        attack_ms = max(0.1, attack_ms)
        release_ms = max(1.0, release_ms)
        makeup_gain_db = np.clip(makeup_gain_db, -12, 24)
        knee_db = np.clip(knee_db, 0, 12) 
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
        makeup_gain = 10 ** (makeup_gain_db / 20)
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = compress_channel(
                    audio_data[:, ch], threshold_linear, ratio,
                    attack_coeff, release_coeff, makeup_gain, knee_db
                )
        else:
            output = compress_channel(
                audio_data, threshold_linear, ratio,
                attack_coeff, release_coeff, makeup_gain, knee_db
            )
        
        logging.info(f"Applied compressor: {ratio}:1 @ {threshold_db}dB, makeup={makeup_gain_db}dB")
        return output
    except Exception as e:
        logging.error(f"Compressor error: {e}")
        return audio_data

@numba.jit(nopython=True)
def limit_channel(audio_channel, ceiling, release_coeff):
    """Limit single channel."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        abs_sample = abs(sample)
        if abs_sample * gain > ceiling:
            gain = ceiling / max(abs_sample, 1e-10)
        else:
            gain = release_coeff * gain + (1 - release_coeff) * 1.0
            gain = min(gain, 1.0)
        output[i] = sample * gain
    return output

def apply_brickwall_limiter(audio_data, ceiling_db=-0.3, release_ms=50, sample_rate=44100):
    """Brickwall limiter."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        ceiling_linear = 10 ** (ceiling_db / 20)
        release_coeff = np.exp(-1000 / (max(release_ms, 1.0) * sample_rate))
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = limit_channel(audio_data[:, ch], ceiling_linear, release_coeff)
        else:
            output = limit_channel(audio_data, ceiling_linear, release_coeff)
        
        logging.info(f"Applied limiter: ceiling={ceiling_db}dB")
        return output
    except Exception as e:
        logging.error(f"Limiter error: {e}")
        return audio_data

def apply_loudness_normalization(audio_data, target_lufs=-14, sample_rate=44100):
    """Loudness normalization (simplified RMS, should be replaced with pyloudnorm)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        current_rms = np.sqrt(np.mean(audio_data**2))
        if current_rms < 1e-6:
            logging.warning("Silent audio, skipping normalization.")
            return audio_data

        target_rms = 10.0**(-18.0 / 20.0) # Target RMS of -18dBFS
        
        gain_factor = target_rms / current_rms
        gain_db = 20 * np.log10(gain_factor)
        gain_db = np.clip(gain_db, -24, 24)
        gain_linear = 10 ** (gain_db / 20)
        
        output = audio_data * gain_linear
        output = apply_brickwall_limiter(output, ceiling_db=-1.0, sample_rate=sample_rate)
        
        logging.info(f"Applied simplified loudness normalization (Target RMS -18dBFS).")
        return output
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        return audio_data

def apply_stereo_widener(audio_data, width_percent=150):
    """Stereo widening."""
    try:
        if audio_data.ndim < 2 or audio_data.shape[1] < 2:
            logging.warning("Cannot apply stereo widener to mono audio.")
            return audio_data 
        
        width_percent = np.clip(width_percent, 0, 200)
        width_factor = width_percent / 100.0
        
        mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
        side = (audio_data[:, 0] - audio_data[:, 1]) / 2
        side_processed = side * width_factor
        
        output = np.zeros_like(audio_data)
        output[:, 0] = mid + side_processed 
        output[:, 1] = mid - side_processed
        
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        logging.info(f"Stereo widener: {width_percent}%")
        return output
    except Exception as e:
        logging.error(f"Widener error: {e}")
        return audio_data

def apply_saturation(audio_data, drive_db=6, mix=0.5):
    """Harmonic saturation (using tanh as a simple non-linear function)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        drive_db = np.clip(drive_db, 0, 24)
        mix = np.clip(mix, 0, 1)
        
        drive_linear = 10 ** (drive_db / 20)
        saturated_signal = np.tanh(audio_data * drive_linear)
        # Rescale
        saturated_signal /= max(np.tanh(drive_linear * 0.8), 1e-6)
        
        output = audio_data * (1 - mix) + saturated_signal * mix
        
        logging.info(f"Saturation: drive={drive_db}dB, mix={mix*100:.0f}%")
        return output
    except Exception as e:
        logging.error(f"Saturation error: {e}")
        return audio_data

@numba.jit(nopython=True)
def _deesser_channel(audio_channel, b_detect, a_detect, b_process, a_process, threshold_linear, reduction_linear, attack_coeff, release_coeff):
    """Optimized core logic for de-esser on a single channel."""
    detect_signal = lfilter(b_detect, a_detect, audio_channel)
    envelope = np.abs(detect_signal)
    
    # Simple RMS-like smoothing (moving average)
    window_size = 100
    if envelope.size > window_size:
        padded_env = np.pad(envelope, (window_size//2, window_size//2), mode='edge')
        smooth_envelope = np.convolve(padded_env, np.ones(window_size)/window_size, mode='valid')
    else:
        smooth_envelope = envelope

    gain = np.where(smooth_envelope > threshold_linear, reduction_linear, 1.0)
    
    # Smooth the gain
    gain_smooth = np.ones_like(gain)
    for i in range(1, len(gain)):
         if gain[i] < gain_smooth[i-1]: # Attacking
              gain_smooth[i] = attack_coeff * gain_smooth[i-1] + (1.0-attack_coeff) * gain[i]
         else: # Releasing
              gain_smooth[i] = release_coeff * gain_smooth[i-1] + (1.0-release_coeff) * gain[i]
    
    high_freq_signal = lfilter(b_process, a_process, audio_channel)
    low_freq_signal = audio_channel - high_freq_signal
    output = low_freq_signal + high_freq_signal * gain_smooth
    return output

def apply_deesser(audio_data, freq_hz=6000, threshold_db=-15, reduction_db=6, sample_rate=44100):
    """De-esser (simplified dynamic EQ using a bandpass filter for detection)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        freq_hz = np.clip(freq_hz, 2000, nyquist - 1000)
        threshold_db = np.clip(threshold_db, -60, 0)
        reduction_db = np.clip(reduction_db, 0, 24)
        
        lower_sibilance_freq = (freq_hz * 0.7) / nyquist
        upper_sibilance_freq = (freq_hz * 1.3) / nyquist
        lower_sibilance_freq = np.clip(lower_sibilance_freq, 0.01, 0.99)
        upper_sibilance_freq = np.clip(upper_sibilance_freq, 0.01, 0.99)
        if lower_sibilance_freq >= upper_sibilance_freq:
            lower_sibilance_freq = upper_sibilance_freq - 0.01
        
        b_detect, a_detect = signal.butter(4, [lower_sibilance_freq, upper_sibilance_freq], btype='bandpass')
        process_freq = (freq_hz * 0.8) / nyquist
        b_process, a_process = signal.butter(4, process_freq, btype='highpass')
        
        threshold_linear = 10 ** (threshold_db / 20)
        reduction_linear = 10 ** (-reduction_db / 20)
        
        # Envelope coefficients
        attack_ms = 1 # Fast attack
        release_ms = 50 # Moderate release
        attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
        
        # Note: This is still slow because lfilter/convolve are called outside Numba
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                detect_signal = lfilter(b_detect, a_detect, audio_data[:, ch])
                envelope = np.abs(detect_signal)
                envelope = np.convolve(envelope, np.ones(100)/100, mode='same')
                gain = np.where(envelope > threshold_linear, reduction_linear, 1.0)
                
                high_freq_signal = lfilter(b_process, a_process, audio_data[:, ch])
                low_freq_signal = audio_data[:, ch] - high_freq_signal
                output[:, ch] = low_freq_signal + high_freq_signal * gain
        else:
            detect_signal = lfilter(b_detect, a_detect, audio_data)
            envelope = np.abs(detect_signal)
            envelope = np.convolve(envelope, np.ones(100)/100, mode='same')
            gain = np.where(envelope > threshold_linear, reduction_linear, 1.0)
            
            high_freq_signal = lfilter(b_process, a_process, audio_data)
            low_freq_signal = audio_data - high_freq_signal
            output = low_freq_signal + high_freq_signal * gain
        
        logging.info(f"De-esser: {freq_hz}Hz, {threshold_db}dB, {reduction_db}dB reduction")
        return output
    except Exception as e:
        logging.error(f"De-esser error: {e}")
        return audio_data


# ==================== STEM SEPARATION ====================

@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    Separates audio into stems using demucs.
    Returns individual stems as base64 encoded data to be uploaded by Deno.
    """
    input_path = None
    wav_path = None
    output_dir = None
    unique_id = str(uuid.uuid4()) # Added for logging
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.wav")
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
        
        file.save(input_path)
        logging.info(f"📁 Saved: {input_path}")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert input to WAV")
        else:
            wav_path = input_path # It was already a WAV
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"🎵 Running demucs with HIGH QUALITY settings (htdemucs_ft) for {wav_path}...")
        
        subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs_ft',           # <-- CORRECTED: Use High-Quality model
            '-o', output_dir,
            '--mp3',                     # Output as MP3
            '--mp3-bitrate', '192',      # Good quality for streaming
            '--jobs', '2',               # Use 2 CPU cores
            wav_path
        ], check=True, capture_output=True, text=True, timeout=600) # 10 minute timeout
        
        logging.info(f"✅ Demucs complete for {wav_path}")
        
        # --- CORRECTED: Look for the 'htdemucs_ft' subfolder ---
        stem_track_dir = os.path.join(output_dir, 'htdemucs_ft', os.path.splitext(os.path.basename(wav_path))[0])
        
        if not os.path.exists(stem_track_dir):
            logging.error(f"Stem track directory not found: {stem_track_dir}. Trying to find...")
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

        logging.info("🎉 Stem separation and encoding complete!")
        return jsonify(stems_data), 200
    
    except subprocess.TimeoutExpired:
        logging.error("❌ Demucs timeout (>10 minutes)", exc_info=True)
        return jsonify({"error": "Processing timeout. Audio file may be too long or server is overloaded.", "details": "Demucs subprocess timed out."}), 500
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Demucs failed with error code {e.returncode}: {e.stderr}", exc_info=True)
        return jsonify({"error": "Stem separation failed. Please check audio file validity.", "details": e.stderr}), 500
    except Exception as e:
        logging.error(f"❌ Stem separation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files and directories
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and os.path.exists(wav_path) and wav_path != input_path:
                os.remove(wav_path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            logging.info(f"Stem separation cleanup complete for {unique_id}")
        except Exception as cleanup_e:
            logging.error(f"Error during stem separation cleanup: {cleanup_e}", exc_info=True)

# ==================== MAIN PROCESSING ====================

@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with AI decisions."""
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
        output_format = request.form.get('output_format', 'wav').lower() # Default to wav
        
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
            wav_path = input_path # <-- FIX: Use original file if already WAV
        
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None:
            raise ValueError("Failed to read WAV audio")
        
        if audio_data.size == 0:
            raise ValueError("Empty audio file after conversion")
        
        logging.info(f"🎵 Starting DSP processing: SR={sample_rate}, Shape={audio_data.shape}")
        current_audio = audio_data.copy()
        
        # RESTORATION
        if 'restoration' in ai_decisions:
            logging.info("🧹 Applying RESTORATION effects...")
            restoration = ai_decisions['restoration']
            
            if 'noise_reduction' in restoration and restoration['noise_reduction']:
                params = restoration['noise_reduction']
                current_audio = apply_noise_gate(
                    current_audio,
                    threshold_db=params.get('threshold_db', -60),
                    ratio=params.get('ratio', 10), # <-- Restored original param
                    sample_rate=sample_rate
                )
            
            if 'deessing' in restoration and restoration['deessing']:
                params = restoration['deessing']
                freq_range_str = params.get('frequency_range', '6000') # default to center 6kHz
                try:
                    if '-' in freq_range_str:
                        freq = int(freq_range_str.split('-')[0])
                    else:
                        freq = int(freq_range_str)
                except ValueError:
                    freq = 6000 # Fallback
                
                current_audio = apply_deesser(
                    current_audio, 
                    freq_hz=freq, 
                    threshold_db=params.get('threshold_db', -15),
                    reduction_db=params.get('reduction_db', 6),
                    sample_rate=sample_rate
                )
        
        # MIXING
        if 'mixing' in ai_decisions:
            logging.info("🎛️ Applying MIXING effects...")
            mixing = ai_decisions['mixing']
            
            current_audio = apply_parametric_eq(
                current_audio, 80, 0, 0.7, 'highpass', sample_rate 
            )
            
            if 'equalizer' in mixing and 'bands' in mixing['equalizer'] and mixing['equalizer']['bands']:
                for band in mixing['equalizer']['bands']:
                    filter_type = band.get('type', 'peak')
                    
                    if filter_type not in ['peak', 'notch', 'lowshelf', 'highshelf', 'lowpass', 'highpass']:
                        if band.get('gain_db', 0) < -6:
                            filter_type = 'notch'
                        elif band.get('frequency', 1000) < 150:
                            filter_type = 'lowshelf'
                        elif band.get('frequency', 1000) > 8000:
                            filter_type = 'highshelf'
                        else:
                            filter_type = 'peak'
                    
                    current_audio = apply_parametric_eq(
                        current_audio,
                        frequency=band.get('frequency', 1000),
                        gain_db=band.get('gain_db', 0),
                        q_factor=band.get('q_factor', 1.0),
                        filter_type=filter_type,
                        sample_rate=sample_rate
                    )
            
            if 'compression' in mixing and mixing['compression']:
                comp = mixing['compression']
                ratio_str = comp.get('ratio', '4:1')
                try:
                    ratio = float(ratio_str.split(':')[0])
                except ValueError:
                    ratio = 4.0 # Fallback
                
                # Pop 'ratio' if it exists before passing **comp
                comp.pop('ratio', None)
                
                current_audio = apply_compressor(
                    current_audio,
                    threshold_db=comp.get('threshold_db', -20),
                    ratio=ratio,
                    attack_ms=comp.get('attack_ms', 5),
                    release_ms=comp.get('release_ms', 50),
                    makeup_gain_db=comp.get('makeup_gain_db', 3),
                    knee_db=comp.get('knee_db', 0), 
                    sample_rate=sample_rate
                )
            
            if 'panning' in mixing and mixing['panning']:
                pass 
            
            if 'saturation' in mixing and mixing['saturation']:
                params = mixing['saturation']
                current_audio = apply_saturation(
                    current_audio,
                    drive_db=params.get('drive_db', 3),
                    mix=params.get('mix_percent', 20) / 100.0
                )


        # MASTERING
        if 'mastering' in ai_decisions:
            logging.info("✨ Applying MASTERING effects...")
            mastering = ai_decisions['mastering']
            
            if current_audio.ndim > 1 and current_audio.shape[1] >= 2:
                current_audio = apply_stereo_widener(current_audio, 120) 
            
            if 'limiting' in mastering and mastering['limiting']:
                target_lufs = mastering['limiting'].get('target_lufs', -14)
                current_audio = apply_loudness_normalization(
                    current_audio, target_lufs, sample_rate
                )
            else: 
                 current_audio = apply_brickwall_limiter(current_audio, ceiling_db=-1.0, sample_rate=sample_rate)
        
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
        "version": "3.0.0", 
        "optimizations": "Fast htdemucs, base64 data transfer, local cleanup",
        "endpoints": {
            "POST /separate_stems": "Separate audio into stems, returns base64 encoded MP3s",
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




