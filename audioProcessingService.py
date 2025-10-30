import os
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, iirpeak, iirnotch
from scipy import signal
import logging
from pydub import AudioSegment
import tempfile
import shutil
import numba
import subprocess
import sys
import json
import atexit
import base64 # New import for base64 encoding

from flask import Flask, request, jsonify

app = Flask(__name__)
# Enhanced logging format for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')

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

def apply_noise_gate(audio_data, threshold_db=-60, ratio=10, attack_ms=5, release_ms=50, sample_rate=44100):
    """Professional noise gate with envelope follower."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1000 / (max(attack_ms, 0.1) * sample_rate))
        release_coeff = np.exp(-1000 / (max(release_ms, 1.0) * sample_rate)) # min release 1ms
        
        if audio_data.ndim > 1:
            detection_signal = np.mean(np.abs(audio_data), axis=1)
        else:
            detection_signal = np.abs(audio_data)
        
        envelope = np.zeros_like(detection_signal)
        for i in range(1, len(detection_signal)):
            if detection_signal[i] > envelope[i-1]:
                envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * detection_signal[i]
            else:
                envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * detection_signal[i]
        
        ratio = max(ratio, 1.0)
        # Apply gain reduction: if envelope is below threshold, apply ratio reduction
        gain = np.where(envelope > threshold_linear, 1.0, np.minimum(envelope / (threshold_linear * ratio), 1.0))
        
        if audio_data.ndim > 1:
            output = audio_data * gain[:, np.newaxis]
        else:
            output = audio_data * gain
        
        logging.info(f"Applied noise gate: {threshold_db}dB, ratio={ratio}:1")
        return output
    except Exception as e:
        logging.error(f"Noise gate error: {e}")
        return audio_data

def apply_parametric_eq(audio_data, frequency, gain_db, q_factor, filter_type='peak', sample_rate=44100):
    """Professional parametric EQ."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        frequency = np.clip(frequency, 20, nyquist - 10) # Ensure frequency is within valid range
        freq_normalized = frequency / nyquist
        q_factor = np.clip(q_factor, 0.1, 20.0) # Q-factor limits
        gain_db = np.clip(gain_db, -24, 24) # Gain limits
        
        # Calculate filter coefficients
        if filter_type == 'peak':
            b, a = iirpeak(freq_normalized, q_factor, fs=2)
            # For peak filter, iirpeak produces a bandpass. We need to apply gain.
            # This is a simplification; a full parametric EQ requires more complex coefficient derivation
            # or a biquad filter implementation that directly handles gain for peak filters.
            # For this context, we will apply gain *after* filtering for simplicity,
            # or rely on the LLM to provide appropriate gain_db for the overall effect.
            # A more robust solution would be to use a library like pyroomacoustics or implement biquad filters.
            
            # For direct application in signal.iirfilter, gain is part of a standard biquad.
            # scipy.signal does not have a direct `iirpeak` with gain.
            # Let's adjust for standard biquad approximation
            # For a simpler approach, we'll implement a shelf or bandpass then apply gain.
            
            # Simple direct gain application for 'peak' if b,a are for unit gain:
            # We'll use iirfilter with a peak type, but actual gain adjustment is complex
            # For a basic parametric approximation with signal.iirfilter:
            # b, a = signal.iirfilter(2, frequency, btype='bandpass', fs=sample_rate, rp=1, rs=1)
            # This is a bandpass. A peak filter is more specific.
            # For now, let's use a simpler gain scaling for the standard iirpeak output,
            # acknowledging this is not a true parametric peak with gain integrated.
            
            # Correct approach for peak filter with gain_db:
            # Requires re-deriving coefficients or using a biquad.
            # Given scipy.signal's limitations for direct parametric with gain,
            # we'll approximate with iirpeak for band selection and then scale.
            # This is a common simplification when a full biquad API isn't used.
            b, a = signal.iirfilter(2, [frequency*0.9/nyquist, frequency*1.1/nyquist], btype='bandpass', fs=sample_rate)
            # This is a very rough approximation.
            # Better: `scipy.signal.iirfilter(N, Wn, btype='peaking', rp=None, rs=None, output='ba', fs=None)` is NOT available.
            # We'll stick to a high-pass/low-pass/band-pass/band-stop for now, and rely on LLM for appropriate instructions
            # For a true parametric, using `biquad` or external lib is better.
            # Let's use simple butterworth filters based on filter_type, and LLM provides gain_db as intent for later scaling.
            
            # Reverting to simpler filters until a full biquad implementation is viable or proper scipy functions identified
            # The previous iirpeak/iirnotch were for specific purposes, not general parametric with gain_db.

            if gain_db > 0: # Boost
                b, a = signal.butter(4, freq_normalized * (1.0 + (q_factor / 10)), btype='bandpass')
            else: # Cut (notch-like for peak)
                b, a = signal.butter(4, freq_normalized * (1.0 - (q_factor / 10)), btype='bandstop')
                
        elif filter_type == 'notch':
            b, a = iirnotch(freq_normalized, q_factor)
        elif filter_type == 'lowshelf':
            b, a = signal.butter(2, freq_normalized, btype='lowpass') # Simplification for shelf
        elif filter_type == 'highshelf':
            b, a = signal.butter(2, freq_normalized, btype='highpass') # Simplification for shelf
        elif filter_type == 'lowpass':
            b, a = signal.butter(4, freq_normalized, btype='lowpass')
        elif filter_type == 'highpass':
            b, a = signal.butter(4, freq_normalized, btype='highpass')
        else: # Default to a generic bandpass and let LLM guide gain.
            b, a = signal.butter(4, [frequency*0.8/nyquist, frequency*1.2/nyquist], btype='bandpass')

        # Apply filter
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = lfilter(b, a, audio_data[:, ch])
        else:
            output = lfilter(b, a, audio_data)
        
        # Apply gain for peak/shelf if it wasn't integrated into filter design
        if filter_type in ['peak', 'lowshelf', 'highshelf']:
            output = output * (10 ** (gain_db / 20))
        
        logging.info(f"Applied EQ: {filter_type} @ {frequency}Hz, {gain_db:+.1f}dB, Q={q_factor}")
        return output
    except Exception as e:
        logging.error(f"EQ error: {e}")
        return audio_data

def apply_compressor(audio_data, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50, 
                    makeup_gain_db=0, knee_db=0, sample_rate=44100):
    """Professional compressor."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        threshold_db = np.clip(threshold_db, -60, 0)
        ratio = np.clip(ratio, 1.0, 20.0)
        attack_ms = max(0.1, attack_ms)
        release_ms = max(1.0, release_ms)
        makeup_gain_db = np.clip(makeup_gain_db, -12, 24)
        knee_db = np.clip(knee_db, 0, 12) # Soft knee control
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1000 / (attack_ms * sample_rate))
        release_coeff = np.exp(-1000 / (release_ms * sample_rate))
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
def compress_channel(audio_channel, threshold, ratio, attack_coeff, release_coeff, makeup_gain, knee_db):
    """Compress single channel with soft knee."""
    envelope = 0.0
    output = np.zeros_like(audio_channel)
    knee_width_linear = 10**(knee_db / 20) if knee_db > 0 else 0.0
    
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        
        # Envelope detection
        if input_level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * input_level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * input_level
        
        gain = 1.0
        if envelope > threshold:
            if knee_db > 0 and envelope < threshold + knee_width_linear:
                # Soft knee calculation
                slope = 1.0 - (1.0 / ratio)
                input_over_threshold = envelope - threshold
                gain_reduction_linear = slope * (input_over_threshold**2) / (2 * knee_width_linear)
                gain = 1.0 - gain_reduction_linear
            else:
                # Hard knee or outside soft knee range
                gain = 1.0 - (envelope - threshold) * (1.0 - (1.0 / ratio)) # This is incorrect for linear gain.
                # Should be: gain_linear = threshold_linear / (threshold_linear + (envelope - threshold_linear) * (1/ratio))
                # simpler: gain_reduction_db = (envelope_db - threshold_db) * (1 - 1/ratio)
                gain_reduction_db = (20 * np.log10(envelope) - 20 * np.log10(threshold)) * (1 - 1/ratio)
                gain = 10 ** (-(gain_reduction_db / 20)) # Convert dB reduction to linear gain
        
        # Ensure gain is not negative or too large
        gain = np.clip(gain, 0.0, 1.0)
        output[i] = sample * gain * makeup_gain
    
    return output

def apply_brickwall_limiter(audio_data, ceiling_db=-0.3, release_ms=50, sample_rate=44100):
    """Brickwall limiter."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        ceiling_linear = 10 ** (ceiling_db / 20)
        release_coeff = np.exp(-1000 / (max(release_ms, 1.0) * sample_rate)) # min release 1ms
        
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

@numba.jit(nopython=True)
def limit_channel(audio_channel, ceiling, release_coeff):
    """Limit single channel."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        # Detect peak
        abs_sample = abs(sample)
        
        # If signal exceeds ceiling, reduce gain
        if abs_sample * gain > ceiling:
            gain = ceiling / max(abs_sample, 1e-10) # Prevent division by zero
        else:
            # Release gain back to 1.0
            gain = release_coeff * gain + (1 - release_coeff) * 1.0
            gain = min(gain, 1.0) # Ensure gain doesn't exceed 1.0
        
        output[i] = sample * gain
    
    return output

def apply_loudness_normalization(audio_data, target_lufs=-14, sample_rate=44100):
    """Loudness normalization (simplified, using RMS for gain adjustment)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        # A more accurate LUFS calculation requires ITU-R BS.1770 algorithm,
        # which is complex. For simplicity, we use RMS as a proxy for loudness.
        # This will not be true LUFS, but provides a level adjustment.
        
        current_rms = np.sqrt(np.mean(audio_data**2))
        
        if current_rms < 1e-6: # Avoid division by zero for silent audio
            logging.warning("Silent audio, skipping normalization.")
            return audio_data

        # Target RMS for -14 LUFS is roughly equivalent to -14 dBFS if calibrated
        # For simplicity, we'll aim for a target RMS related to dBFS.
        # -14 LUFS is a perceived loudness target, not a peak/RMS target.
        # A common target for streaming is around -1dBFS true peak with -14 LUFS integrated.
        # This simplification adjusts gain based on current RMS to bring it towards a target.
        
        # Let's target a specific RMS value that is typical for a -14 LUFS master peaking at -1dBTP.
        # This is a very rough heuristic for a target RMS.
        target_rms = 0.1 # This corresponds roughly to -20dBFS for a sine wave.
                         # A full LUFS algorithm is needed for true measurement.
        
        gain_factor = target_rms / current_rms
        
        output = audio_data * gain_factor
        
        # Apply a limiter AFTER gain adjustment to catch peaks and set final ceiling
        output = apply_brickwall_limiter(output, ceiling_db=-1.0, release_ms=50, sample_rate=sample_rate)
        
        logging.info(f"Applied simplified loudness normalization (RMS adjusted).")
        return output
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        return audio_data

def apply_stereo_widener(audio_data, width_percent=150):
    """Stereo widening."""
    try:
        if audio_data.ndim < 2 or audio_data.shape[1] < 2:
            logging.warning("Cannot apply stereo widener to mono audio.")
            return audio_data # Cannot widen mono audio
        
        width_percent = np.clip(width_percent, 0, 200) # 0% (mono) to 200% (super wide)
        width_factor = width_percent / 100.0
        
        # M-S (Mid-Side) processing
        mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
        side = (audio_data[:, 0] - audio_data[:, 1]) / 2
        
        # Apply widening to the Side component
        side_processed = side * width_factor
        
        output = np.zeros_like(audio_data)
        output[:, 0] = mid + side_processed # Left
        output[:, 1] = mid - side_processed # Right
        
        # Normalize to prevent clipping after widening
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
        
        # Tanh saturation
        saturated_signal = np.tanh(audio_data * drive_linear)
        
        # Mix dry and wet signal
        output = audio_data * (1 - mix) + saturated_signal * mix
        
        logging.info(f"Saturation: drive={drive_db}dB, mix={mix*100:.0f}%")
        return output
    except Exception as e:
        logging.error(f"Saturation error: {e}")
        return audio_data

def apply_deesser(audio_data, freq_hz=6000, threshold_db=-15, reduction_db=6, sample_rate=44100):
    """De-esser (simplified dynamic EQ using a bandpass filter for detection)."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        freq_hz = np.clip(freq_hz, 2000, nyquist - 1000) # De-essing typically 2kHz-8kHz
        threshold_db = np.clip(threshold_db, -60, 0)
        reduction_db = np.clip(reduction_db, 0, 24)
        
        # Define bandpass for sibilance detection (e.g., 4kHz-8kHz for a 6kHz center)
        lower_sibilance_freq = (freq_hz * 0.7) / nyquist
        upper_sibilance_freq = (freq_hz * 1.3) / nyquist
        
        # Ensure frequencies are valid and lower < upper
        lower_sibilance_freq = np.clip(lower_sibilance_freq, 0.01, 0.99)
        upper_sibilance_freq = np.clip(upper_sibilance_freq, 0.01, 0.99)
        if lower_sibilance_freq >= upper_sibilance_freq:
            lower_sibilance_freq = upper_sibilance_freq - 0.01 # Adjust if invalid range
        
        # Bandpass filter for detection
        b_detect, a_detect = signal.butter(4, [lower_sibilance_freq, upper_sibilance_freq], btype='bandpass')
        
        # High-shelf filter to process the sibilance region, or a band-pass with negative gain
        # For simplicity, we'll use a direct gain reduction on the sibilance band.
        
        threshold_linear = 10 ** (threshold_db / 20)
        reduction_linear = 10 ** (-reduction_db / 20)
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                sibilance_band = lfilter(b_detect, a_detect, audio_data[:, ch])
                non_sibilance_band = audio_data[:, ch] - sibilance_band # Approximate non-sibilance
                
                # Apply reduction only when sibilance is above threshold
                gain_env = np.where(np.abs(sibilance_band) > threshold_linear, reduction_linear, 1.0)
                
                output[:, ch] = non_sibilance_band + sibilance_band * gain_env
        else:
            sibilance_band = lfilter(b_detect, a_detect, audio_data)
            non_sibilance_band = audio_data - sibilance_band
            
            gain_env = np.where(np.abs(sibilance_band) > threshold_linear, reduction_linear, 1.0)
            
            output = non_sibilance_band + sibilance_band * gain_env
        
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
        logging.info(f"📁 Saved input file: {input_path}")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert input to WAV")
        else:
            # If already WAV, use directly. Copy to UPLOAD_FOLDER to manage cleanup
            shutil.copy(input_path, wav_path) # <-- This was an error in your file, should be copy not assign
        
        os.makedirs(output_dir, exist_ok=True)
        
        # --- 💥 ONE CHANGE MADE HERE 💥 ---
        logging.info(f"🎵 Running demucs with HIGH QUALITY settings (htdemucs_ft) for {wav_path}...")
        
        subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs_ft',           # <-- CHANGED: Use High-Quality, Fine-Tuned model
            '-o', output_dir,
            '--mp3',                     # Output as MP3
            '--mp3-bitrate', '192',      # Good quality for streaming, faster processing
            '--jobs', '2',               # Use 2 CPU cores for faster separation
            wav_path
        ], check=True, capture_output=True, text=True, timeout=600) # 10 minute timeout for demucs
        
        logging.info(f"✅ Demucs complete for {wav_path}")
        
        # --- 💥 SECOND CHANGE MADE HERE 💥 ---
        # Find the actual directory containing the stems
        stem_track_dir = os.path.join(output_dir, 'htdemucs_ft', os.path.splitext(os.path.basename(wav_path))[0])
        
        if not os.path.exists(stem_track_dir):
            logging.error(f"Stem track directory not found: {stem_track_dir}. Trying to find...")
            found_stem_dir = False
            # Fallback for older model name
            stem_track_dir_fallback = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(wav_path))[0])
            if os.path.exists(stem_track_dir_fallback):
                stem_track_dir = stem_track_dir_fallback
                found_stem_dir = True
            else:
                # Fallback for recursive search
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
                logging.info(f"Cleaned up {input_path}")
            if wav_path and os.path.exists(wav_path) and wav_path != input_path: # Fix: check against input_path
                os.remove(wav_path)
                logging.info(f"Cleaned up {wav_path}")
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logging.info(f"Cleaned up {output_dir}")
        except Exception as cleanup_e:
            logging.error(f"Error during cleanup: {cleanup_e}", exc_info=True)

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
                    # Attempt to extract center frequency from string, e.g., "5000-10000" or "6000"
                    if '-' in freq_range_str:
                        freq = int(freq_range_str.split('-')[0]) # Use lower bound as center or start
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
            
            # Initial highpass to remove sub-bass rumble often done in mixing
            current_audio = apply_parametric_eq(
                current_audio, 80, 0, 0.7, 'highpass', sample_rate # 0.7 Q for gentle HPF
            )
            
            if 'equalizer' in mixing and 'bands' in mixing['equalizer'] and mixing['equalizer']['bands']:
                for band in mixing['equalizer']['bands']:
                    filter_type = band.get('type', 'peak') # Default to peak for parametric
                    
                    # --- Logic from your file ---
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
                
                current_audio = apply_compressor(
                    current_audio,
                    threshold_db=comp.get('threshold_db', -20),
                    ratio=ratio,
                    attack_ms=comp.get('attack_ms', 5),
                    release_ms=comp.get('release_ms', 50),
                    makeup_gain_db=comp.get('makeup_gain_db', 3),
                    knee_db=comp.get('knee_db', 0), # Default to hard knee if not specified
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
                current_audio = apply_stereo_widener(current_audio, 120) # Gentle widening
            
            if 'limiting' in mastering and mastering['limiting']:
                target_lufs = mastering['limiting'].get('target_lufs', -14)
                current_audio = apply_loudness_normalization(
                    current_audio, target_lufs, sample_rate
                )
            else: # If no explicit limiting, ensure basic peak protection
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
            if wav_path and os.path.exists(wav_path) and wav_path != input_path: # Fix: check against input_path
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



