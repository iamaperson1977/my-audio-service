from flask import Flask, request, send_file, jsonify
import os
import uuid
import numpy as np
import soundfile as sf
# Use specific imports from scipy.signal for clarity
from scipy.signal import butter, lfilter, iirpeak, iirnotch
from scipy import signal # Keep for potential other uses
import logging
from pydub import AudioSegment
# import librosa # Not used in current DSP functions
import tempfile
import shutil
import json # For parsing instructions
import atexit
# import numba # Numba decorators are NOT included in this Base44 version

# --- NEW IMPORTS FOR STEM SEPARATION ---
import subprocess
import zipfile
import glob
# --- END NEW IMPORTS ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')
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
    """Convert any audio format to WAV using pydub. Set SR to 44100."""
    try:
        audio = AudioSegment.from_file(input_path)
        # Standardize sample rate for internal processing
        audio = audio.set_frame_rate(44100)
        audio.export(output_path, format='wav')
        logging.info(f"Converted '{os.path.basename(input_path)}' to WAV: '{os.path.basename(output_path)}' (channels={audio.channels}, sr=44100)")
        return True
    except Exception as e:
        logging.error(f"Error converting '{os.path.basename(input_path)}' to WAV: {e}", exc_info=True)
        return False

def convert_from_wav(wav_path, output_path, output_format='mp3'):
    """Convert WAV to other formats using pydub."""
    try:
        audio = AudioSegment.from_wav(wav_path)
        fmt = output_format.lower()
        
        # Determine parameters based on format
        params = {'format': fmt}
        if fmt == 'mp3':
            params['bitrate'] = '320k'
        elif fmt == 'aac' or fmt == 'm4a':
             params['codec'] = 'aac' # Explicitly set codec for aac/m4a if needed
             params['bitrate'] = '256k'
        elif fmt == 'ogg':
             params['codec'] = 'libvorbis'
        # FLAC and WAV need no extra params

        audio.export(output_path, **params)
        logging.info(f"Converted WAV '{os.path.basename(wav_path)}' to {fmt}: '{os.path.basename(output_path)}'")
        return True
    except Exception as e:
        logging.error(f"Error converting WAV '{os.path.basename(wav_path)}' to {fmt}: {e}", exc_info=True)
        return False

def read_wav(filepath):
    """Reads WAV file into NumPy array (float32) and sample rate."""
    try:
        audio_data, sample_rate = sf.read(filepath, dtype='float32')
        logging.info(f"Read WAV: '{os.path.basename(filepath)}', SR: {sample_rate}, Shape: {audio_data.shape}")
        return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Error reading WAV '{os.path.basename(filepath)}': {e}", exc_info=True)
        return None, None

def write_wav(filepath, audio_data, sample_rate):
    """Writes NumPy array (float32) to WAV file, ensuring directory exists and clipping."""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Clip audio data ONLY IF NECESSARY to prevent distortion
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            logging.warning(f"Clipping detected before writing '{os.path.basename(filepath)}' (max val: {max_val:.4f}). Clipping to [-1.0, 1.0].")
            audio_data_clipped = np.clip(audio_data, -1.0, 1.0)
        else:
            audio_data_clipped = audio_data # Avoid unnecessary copy if no clipping needed

        sf.write(filepath, audio_data_clipped, sample_rate)
        logging.info(f"Wrote WAV: '{os.path.basename(filepath)}'")
        return True
    except Exception as e:
        logging.error(f"Error writing WAV '{os.path.basename(filepath)}': {e}", exc_info=True)
        return False

# ==================== SAFE DSP FUNCTIONS (Base44 Version) ====================
# NOTE: These functions prioritize stability over performance/aggressiveness.
# Numba optimizers are NOT included in this version.

def apply_noise_gate(audio_data, threshold_db=-60, ratio=5, attack_ms=5, release_ms=100, sample_rate=44100):
    """Noise gate with safety limits."""
    logging.info(f"DSP: Applying noise gate: threshold={threshold_db}dB, ratio={ratio}:1, attack={attack_ms}ms, release={release_ms}ms")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            logging.warning("Noise Gate: Empty or invalid audio data")
            return audio_data if isinstance(audio_data, np.ndarray) else np.array([], dtype=np.float32) # Return empty array if input invalid

        check_audio_level(audio_data, "Before Noise Gate")

        # Apply safety limits from Base44 version
        threshold_db = max(threshold_db, -80.0) # Limit how low threshold can go
        ratio = np.clip(ratio, 1.0, 5.0) # Limit ratio
        attack_ms = max(0.1, attack_ms)
        release_ms = max(1.0, release_ms)
        min_gain = 0.5 # Never reduce gain below this factor (Base44's safety)

        threshold_linear = 10.0**(threshold_db / 20.0)
        attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

        if audio_data.ndim > 1:
            # Process stereo by detecting on mean, applying to both
            detection_signal = np.mean(np.abs(audio_data), axis=1)
            output = np.zeros_like(audio_data)
        else:
            detection_signal = np.abs(audio_data)
            output = np.zeros_like(audio_data)

        envelope = 0.0
        gain_smooth = 1.0 # Use smoothing for gain changes
        inv_ratio = 1.0 / ratio if ratio > 0 else 1.0

        for i in range(len(detection_signal)):
            # Envelope follower
            if detection_signal[i] > envelope:
                envelope = attack_coeff * envelope + (1.0 - attack_coeff) * detection_signal[i]
            else:
                envelope = release_coeff * envelope + (1.0 - release_coeff) * detection_signal[i]

            # Gain calculation (Downward expansion)
            target_gain = 1.0
            if envelope < threshold_linear:
                 under_amount = threshold_linear - envelope
                 target_gain = 1.0 - (under_amount / threshold_linear) * (ratio - 1.0) * inv_ratio
                 target_gain = max(min_gain, target_gain) # Apply safety minimum gain

            # Smooth gain changes
            if target_gain < gain_smooth:
                 gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
            else:
                 gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain

            # Apply gain
            if audio_data.ndim > 1:
                output[i, :] = audio_data[i, :] * gain_smooth
            else:
                output[i] = audio_data[i] * gain_smooth

        check_audio_level(output, "After Noise Gate")
        return output
    except Exception as e:
        logging.error(f"Error in noise gate: {e}", exc_info=True)
        return audio_data


def apply_deesser(audio_data, freq_hz=6000, threshold_db=-20, reduction_db=6, sample_rate=44100):
    """De-esser with safety limits."""
    logging.info(f"DSP: Applying de-esser: freq={freq_hz}Hz, threshold={threshold_db}dB, reduction={reduction_db}dB")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            logging.warning("De-esser: Empty or invalid audio data")
            return audio_data if isinstance(audio_data, np.ndarray) else np.array([], dtype=np.float32)

        check_audio_level(audio_data, "Before De-esser")

        # Apply safety limits from Base44 version
        reduction_db = min(reduction_db, 6.0) # Max reduction
        if reduction_db <= 0: return audio_data # Skip if no reduction needed

        nyquist = 0.5 * sample_rate
        freq_hz = np.clip(freq_hz, 4000.0, nyquist - 1000.0)
        threshold_db = np.clip(threshold_db, -60.0, 0.0)
        threshold_linear = 10.0**(threshold_db / 20.0)
        reduction_linear = 10.0**(-reduction_db / 20.0)

        # --- Detection Bandpass --- (Centred around freq_hz)
        q_detect = 1.0
        bw_oct = 1.0 # Approx 1 octave bandwidth
        low_f = freq_hz / (2**(bw_oct/2))
        high_f = freq_hz * (2**(bw_oct/2))
        low_norm = np.clip(low_f / nyquist, 0.01, 0.99)
        high_norm = np.clip(high_f / nyquist, 0.01, 0.99)
        if low_norm >= high_norm: high_norm = low_norm + 0.01 # Ensure high > low

        b_detect, a_detect = butter(2, [low_norm, high_norm], btype='band')

        # --- Envelope and Gain Calculation ---
        attack_ms = 1 # Fast attack for sibilance
        release_ms = 50 # Moderate release
        attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                detect_signal = lfilter(b_detect, a_detect, audio_data[:, ch])
                # Use RMS envelope for detection
                window_size = int(0.005 * sample_rate) # 5ms RMS window
                squared = detect_signal**2
                rms_envelope = np.sqrt(np.convolve(squared, np.ones(window_size)/window_size, mode='same'))

                # Calculate gain reduction with smoothing
                gain = np.where(rms_envelope > threshold_linear, reduction_linear, 1.0)
                gain_smooth = np.ones_like(gain)
                for i in range(1, len(gain)):
                     if gain[i] < gain_smooth[i-1]: # Attacking
                          gain_smooth[i] = attack_coeff * gain_smooth[i-1] + (1.0-attack_coeff) * gain[i]
                     else: # Releasing
                          gain_smooth[i] = release_coeff * gain_smooth[i-1] + (1.0-release_coeff) * gain[i]

                # Apply gain to the original signal
                output[:, ch] = audio_data[:, ch] * gain_smooth
        else:
            # Mono processing
            detect_signal = lfilter(b_detect, a_detect, audio_data)
            window_size = int(0.005 * sample_rate)
            squared = detect_signal**2
            rms_envelope = np.sqrt(np.convolve(squared, np.ones(window_size)/window_size, mode='same'))
            gain = np.where(rms_envelope > threshold_linear, reduction_linear, 1.0)
            gain_smooth = np.ones_like(gain)
            for i in range(1, len(gain)):
                 if gain[i] < gain_smooth[i-1]: gain_smooth[i] = attack_coeff * gain_smooth[i-1] + (1.0-attack_coeff) * gain[i]
                 else: gain_smooth[i] = release_coeff * gain_smooth[i-1] + (1.0-release_coeff) * gain[i]
            output = audio_data * gain_smooth

        check_audio_level(output, "After De-esser")
        return output
    except Exception as e:
        logging.error(f"Error in de-esser: {e}", exc_info=True)
        return audio_data


def apply_parametric_eq(audio_data, frequency, gain_db, q_factor, filter_type='peak', sample_rate=44100):
    """Parametric EQ with safety limits."""
    logging.info(f"DSP: Applying EQ: {filter_type} @ {frequency}Hz, {gain_db:+.1f}dB, Q={q_factor}")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0: return audio_data # Safety check

        # Apply safety limits from Base44 version
        gain_db = np.clip(gain_db, -12.0, 12.0)
        q_factor = np.clip(q_factor, 0.1, 20.0)
        nyquist = 0.5 * sample_rate
        frequency = np.clip(frequency, 20.0, nyquist - 1.0)
        freq_normalized = frequency / nyquist

        b, a = [1], [1] # Default to passthrough

        # Design filter based on type
        if filter_type == 'lowpass' or filter_type == 'highpass':
            order = 4 # Fixed order for simplicity
            b, a = butter(order, freq_normalized, btype=filter_type, analog=False)
        elif filter_type == 'peak':
            # This implementation from Base44's file is inaccurate for peak boosts.
            # It uses iirpeak (which is for peaking) but then multiplies 'b' by gain_linear,
            # which is not the correct way to apply gain to an IIR filter's coeffs.
            # A full biquad design is needed for accurate peak/shelf filters.
            # We will use the (flawed) Base44 logic for now as requested.
            gain_linear = 10.0**(gain_db / 20.0)
            b_peak, a_peak = iirpeak(freq_normalized, q_factor, fs=2)
            b = b_peak * gain_linear
            a = a_peak
            if gain_db > 0:
                 logging.warning(f"Peak EQ boost @ {frequency}Hz uses an inaccurate method. Results may vary.")
        
        elif filter_type == 'notch':
             b, a = iirnotch(freq_normalized, q_factor, fs=2)
        
        elif filter_type == 'lowshelf' or filter_type == 'highshelf':
             # Base44's version also used butter for shelves, which is incorrect.
             # A real shelf filter would use signal.iirfilter with a shelf design.
             # Sticking to the provided (flawed) code:
             order = max(2, int(q_factor)) # This Q mapping to order is strange
             b_shelf, a_shelf = butter(order, freq_normalized, btype=('low' if filter_type == 'lowshelf' else 'high'))
             gain_linear = 10.0**(gain_db / 20.0)
             b = b_shelf * gain_linear
             a = a_shelf
             logging.warning(f"Shelf EQ filter type '{filter_type}' is using an inaccurate Butterworth design.")
        else:
            logging.warning(f"Unknown EQ filter type: {filter_type}. Skipping.")
            b, a = [1], [1]

        # Apply the filter using lfilter if coeffs are valid
        if len(b) > 1 or len(a) > 1:
            if audio_data.ndim > 1:
                output = np.zeros_like(audio_data)
                for ch in range(audio_data.shape[1]):
                    zi = signal.lfilter_zi(b, a)
                    output[:, ch], _ = lfilter(b, a, audio_data[:, ch], zi=zi*audio_data[0, ch])
            else:
                zi = signal.lfilter_zi(b, a)
                output, _ = lfilter(b, a, audio_data, zi=zi*audio_data[0])
            return output
        else:
            return audio_data # Return original if filter was skipped

    except Exception as e:
        logging.error(f"Error applying EQ ({filter_type} @ {frequency}Hz): {e}", exc_info=True)
        return audio_data

# Note: The Base44 compressor logic was complex. Re-using our _compress_channel logic.
# If you want the Base44 logic, replace this.
@numba.jit(nopython=True) # Adding Numba back for performance - REMOVE IF IT CAUSES BUILD ERRORS
def _compress_channel(audio_channel, threshold_linear, ratio, attack_coeff, release_coeff, makeup_gain, knee_width_db):
    """Compress a single channel (Numba compatible)."""
    envelope_db = -100.0
    gain_smooth = 1.0
    output = np.zeros_like(audio_channel)
    inv_ratio_minus_one = (1.0 / ratio) - 1.0

    for i, sample in enumerate(audio_channel):
        input_level_db = 20.0 * np.log10(max(abs(sample), 1e-15))
        if input_level_db > envelope_db:
            envelope_db = attack_coeff * envelope_db + (1.0 - attack_coeff) * input_level_db
        else:
            envelope_db = release_coeff * envelope_db + (1.0 - release_coeff) * input_level_db

        gain_reduction_db = 0.0
        threshold_db_calc = 20.0 * np.log10(threshold_linear)
        overshoot_db = envelope_db - threshold_db_calc

        if knee_width_db > 0 and abs(overshoot_db) <= knee_width_db / 2.0:
            gain_reduction_db = inv_ratio_minus_one * ((overshoot_db + knee_width_db / 2.0)**2) / (2.0 * knee_width_db)
        elif overshoot_db > knee_width_db / 2.0:
            gain_reduction_db = inv_ratio_minus_one * overshoot_db

        target_gain = 10.0**(gain_reduction_db / 20.0)
        if target_gain < gain_smooth:
            gain_smooth = attack_coeff * gain_smooth + (1.0 - attack_coeff) * target_gain
        else:
            gain_smooth = release_coeff * gain_smooth + (1.0 - release_coeff) * target_gain

        output[i] = sample * gain_smooth * makeup_gain
    return output

def apply_compressor(audio_data, threshold_db=-20, ratio=4, attack_ms=10, release_ms=100,
                    makeup_gain_db=0, knee_db=3, sample_rate=44100):
    """Compressor with soft knee and safety limits."""
    logging.info(f"DSP: Applying compressor: T={threshold_db}dB, R={ratio}:1, A={attack_ms}ms, R={release_ms}ms, Knee={knee_db}dB, Makeup={makeup_gain_db}dB")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0: return audio_data
        check_audio_level(audio_data, "Before Compressor")
        threshold_db = np.clip(threshold_db, -60.0, 0.0)
        ratio = np.clip(ratio, 1.0, 10.0)
        attack_ms = max(0.1, attack_ms)
        release_ms = max(10.0, release_ms)
        makeup_gain_db = np.clip(makeup_gain_db, -6.0, 12.0)
        knee_db = np.clip(knee_db, 0.0, 12.0)
        threshold_linear = 10.0**(threshold_db / 20.0)
        attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
        makeup_gain = 10.0**(makeup_gain_db / 20.0)

        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = _compress_channel(
                    audio_data[:, ch], threshold_linear, ratio,
                    attack_coeff, release_coeff, makeup_gain, knee_db
                )
        else:
            output = _compress_channel(
                audio_data, threshold_linear, ratio,
                attack_coeff, release_coeff, makeup_gain, knee_db
            )
        check_audio_level(output, "After Compressor")
        return output
    except Exception as e:
        logging.error(f"Error applying compressor: {e}", exc_info=True)
        return audio_data


@numba.jit(nopython=True) # Adding Numba back for performance
def _limit_channel(audio_channel, ceiling_linear, release_coeff):
    """Limit a single channel (Numba compatible)."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        current_max_level = input_level * gain
        if current_max_level > ceiling_linear:
            gain = ceiling_linear / max(input_level, 1e-15)
        output[i] = sample * gain
        gain = release_coeff * gain + (1.0 - release_coeff) * 1.0
        gain = min(gain, 1.0)
    return output

def apply_brickwall_limiter(audio_data, ceiling_db=-0.5, release_ms=50, sample_rate=44100):
    """Brickwall limiter."""
    logging.info(f"DSP: Applying brickwall limiter: ceiling={ceiling_db}dB, release={release_ms}ms")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0: return audio_data
        ceiling_linear = 10.0**(np.clip(ceiling_db, -6.0, 0.0) / 20.0)
        release_coeff = np.exp(-1.0 / (max(release_ms, 1.0) * sample_rate / 1000.0))
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = _limit_channel(audio_data[:, ch], ceiling_linear, release_coeff)
        else:
            output = _limit_channel(audio_data, ceiling_linear, release_coeff)
        return output
    except Exception as e:
        logging.error(f"Error applying limiter: {e}", exc_info=True)
        return audio_data


def apply_loudness_normalization(audio_data, target_lufs=-14.0, sample_rate=44100):
    """Loudness normalization using BS.1770 approximation with safety limits."""
    logging.info(f"DSP: Applying loudness normalization to {target_lufs} LUFS (Approx)")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size < sample_rate * 0.4:
             logging.warning("Audio too short or invalid for LUFS calculation, skipping norm.")
             return audio_data if isinstance(audio_data, np.ndarray) else np.array([], dtype=np.float32)
        target_lufs = np.clip(target_lufs, -24.0, -5.0)
        nyquist = 0.5 * sample_rate
        b_hp, a_hp = butter(2, 38.0 / nyquist, btype='high')
        # b_hs, a_hs = signal.iirfilter(...) # Shelf filter approximation removed for simplicity/correctness
        if audio_data.ndim > 1:
            power = 0.0
            for ch in range(audio_data.shape[1]):
                 filtered_ch = lfilter(b_hp, a_hp, audio_data[:, ch])
                 power += np.mean(filtered_ch.astype(np.float64)**2)
            mean_square = power / audio_data.shape[1]
        else:
            filtered = lfilter(b_hp, a_hp, audio_data)
            mean_square = np.mean(filtered.astype(np.float64)**2)
        if mean_square < 1e-15:
            logging.warning("Audio appears silent after filtering, skipping LUFS normalization.")
            return audio_data
        current_lufs = -0.691 + 10.0 * np.log10(mean_square) # Simplified LUFS
        gain_db = target_lufs - current_lufs
        gain_db = np.clip(gain_db, -18.0, 18.0)
        gain_linear = 10.0**(gain_db / 20.0)
        logging.info(f"LUFS Calc: Current (approx)={current_lufs:.1f} LUFS, Target={target_lufs} LUFS. Applying Gain={gain_db:.1f} dB.")
        output = audio_data * gain_linear
        output = apply_brickwall_limiter(output, ceiling_db=-1.0, sample_rate=sample_rate)
        check_audio_level(output, "After Loudness Normalization")
        return output
    except Exception as e:
        logging.error(f"Error in loudness normalization: {e}", exc_info=True)
        return audio_data


def apply_stereo_widener(audio_data, width_percent=110):
    """Stereo widening using mid-side. 100=no change."""
    logging.info(f"DSP: Applying stereo widener: {width_percent}%")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.ndim < 2 or audio_data.shape[1] < 2:
            logging.warning("Stereo widener requires stereo audio, skipping.")
            return audio_data if isinstance(audio_data, np.ndarray) else np.array([], dtype=np.float32)
        width = np.clip(width_percent / 100.0, 0.0, 2.0)
        mid = (audio_data[:, 0] + audio_data[:, 1]) / 2.0
        side = (audio_data[:, 0] - audio_data[:, 1]) / 2.0
        side *= width
        output = np.zeros_like(audio_data)
        output[:, 0] = mid + side
        output[:, 1] = mid - side
        output = np.tanh(output * 1.05) / 1.05 # Soft clip
        check_audio_level(output, "After Stereo Widener")
        return output
    except Exception as e:
        logging.error(f"Error applying stereo widener: {e}", exc_info=True)
        return audio_data


def apply_saturation(audio_data, drive_db=3, mix=0.1):
    """Harmonic saturation using tanh waveshaping."""
    logging.info(f"DSP: Applying saturation: drive={drive_db}dB, mix={mix*100:.0f}%")
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0: return audio_data
        check_audio_level(audio_data, "Before Saturation")
        drive = 10.0**(np.clip(drive_db, 0.0, 18.0) / 20.0)
        mix_pct = np.clip(mix, 0.0, 1.0)
        saturated = np.tanh(audio_data * drive)
        saturated /= max(np.tanh(drive * 0.8), 1.0) # Rescale approx
        output = audio_data * (1.0 - mix_pct) + saturated * mix_pct
        check_audio_level(output, "After Saturation")
        return output
    except Exception as e:
        logging.error(f"Error applying saturation: {e}", exc_info=True)
        return audio_data


# ==================== NEW STEM SEPARATION ENDPOINT ====================
@app.route('/separate_stems', methods=['POST'])
def separate_stems_endpoint():
    """
    NEW ENDPOINT: Receives a single audio file, separates it into 4 stems
    (bass, drums, other, vocals) using Demucs, zips the stems,
    and sends the zip file back.
    """
    input_path = None
    wav_input_path = None # Converted WAV for Demucs
    output_dir = None
    zip_path = None
    unique_id = str(uuid.uuid4())
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided", "request_id": unique_id}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename", "request_id": unique_id}), 400

        # --- 1. Save Input File ---
        _, file_ext = os.path.splitext(file.filename)
        file_ext = file_ext.lower().strip('.') or 'bin'
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.wav")
        file.save(input_path)
        logging.info(f"Request {unique_id}: Saved input for separation: '{os.path.basename(input_path)}'")

        # --- 2. Convert to WAV for Demucs ---
        # Demucs works best with WAV files
        if file_ext != 'wav':
            logging.info(f"Request {unique_id}: Converting input to WAV for Demucs...")
            if not convert_to_wav(input_path, wav_input_path):
                 raise ValueError("Failed to convert input file to WAV for Demucs")
            # We no longer need the original non-WAV input
            try: os.remove(input_path) 
            except Exception: logging.warning(f"Request {unique_id}: Could not remove original input file.")
        else:
            wav_input_path = input_path # It was already a WAV

        # --- 3. Define Output Directory for Stems ---
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_demucs_output")
        # Demucs will create a subdirectory inside this based on the model name
        os.makedirs(output_dir, exist_ok=True)

        # --- 4. Run Demucs using subprocess ---
        # Model: htdemucs_ft (high quality, fine-tuned, 4 stems)
        # Using python -m demucs ... is the most robust way to call it
        demucs_command = [
            "python", "-m", "demucs.separate",
            "-n", "htdemucs_ft",      # Use the high-quality 4-stem model
            "-o", output_dir,         # Set the base output directory
            "--filename", "{stem}.{ext}", # Use clean filenames (bass.wav, drums.wav...)
            wav_input_path            # Pass the converted WAV file
        ]
        
        logging.info(f"Request {unique_id}: Running Demucs... This will take time. Command: {' '.join(demucs_command)}")
        # Note: Gunicorn timeout MUST be long enough for this to complete.
        result = subprocess.run(demucs_command, capture_output=True, text=True, timeout=110) # 110s timeout, just under 120s Gunicorn

        if result.returncode != 0:
            logging.error(f"Request {unique_id}: Demucs separation FAILED. Error: {result.stderr}")
            raise Exception(f"Demucs separation failed: {result.stderr}")

        logging.info(f"Request {unique_id}: Demucs separation complete. Output: {result.stdout}")

        # --- 5. Find the Separated Stem Files ---
        # Demucs creates a subfolder named 'htdemucs_ft'
        stems_folder_path = os.path.join(output_dir, "htdemucs_ft", os.path.splitext(os.path.basename(wav_input_path))[0])
        
        # Check if the folder exists
        if not os.path.isdir(stems_folder_path):
             # Fallback: maybe demucs changed output? Check one level up.
             stems_folder_path = os.path.join(output_dir, "htdemucs_ft")
             if not os.path.isdir(stems_folder_path):
                 logging.error(f"Request {unique_id}: Could not find Demucs output directory at {stems_folder_path}")
                 raise Exception("Demucs output directory not found.")
        
        stem_files = glob.glob(os.path.join(stems_folder_path, "*.wav"))
        expected_stems = {"bass.wav", "drums.wav", "other.wav", "vocals.wav"}
        found_stems = {os.path.basename(f) for f in stem_files}

        if not expected_stems.issubset(found_stems):
            logging.error(f"Request {unique_id}: Demucs ran but did not produce all expected stems. Found: {found_stems}")
            raise Exception("Demucs did not produce all expected stems.")
            
        logging.info(f"Request {unique_id}: Found {len(stem_files)} stem files in {stems_folder_path}")

        # --- 6. Create a ZIP File ---
        zip_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in stem_files:
                # Add file to zip with a simple name (e.g., "vocals.wav")
                zipf.write(file_path, arcname=os.path.basename(file_path))

        logging.info(f"Request {unique_id}: Created stems zip file: {os.path.basename(zip_path)}")

        # --- 7. Send ZIP File Back ---
        @request.call_on_close
        def cleanup_zip_files():
             # Clean up all temp files *after* the request is sent
             logging.info(f"Request {unique_id}: Cleanup initiated for stem separation...")
             try:
                 # Remove the original input (if it wasn't the wav)
                 if input_path != wav_input_path and os.path.exists(input_path): os.remove(input_path)
                 # Remove the converted wav
                 if os.path.exists(wav_input_path): os.remove(wav_input_path)
                 # Remove the entire demucs output directory
                 if os.path.exists(output_dir): shutil.rmtree(output_dir)
                 # Remove the zip file
                 if os.path.exists(zip_path): os.remove(zip_path)
                 logging.info(f"Request {unique_id}: Stem separation cleanup complete.")
             except Exception as e:
                 logging.error(f"Request {unique_id}: Error during stem cleanup: {e}")

        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"separated_stems_{unique_id}.zip"
        )

    except subprocess.TimeoutExpired:
        logging.error(f"Request {unique_id}: ❌ Demucs separation timed out!", exc_info=True)
        cleanup_temp_dirs_on_error(unique_id, input_path, wav_input_path, output_dir, zip_path)
        return jsonify({"error": "Stem separation timed out. The audio file may be too long for the current server configuration.", "request_id": unique_id}), 504 # Gateway Timeout
    except Exception as e:
        logging.error(f"Request {unique_id}: ❌ Stem separation failed: {e}", exc_info=True)
        cleanup_temp_dirs_on_error(unique_id, input_path, wav_input_path, output_dir, zip_path)
        return jsonify({"error": f"An internal server error occurred: {e}", "request_id": unique_id}), 500


# ==================== MAIN PROCESSING ENDPOINT ====================

@app.route('/process', methods=['POST'])
def process_audio():
    """Main endpoint receives audio file and JSON instructions."""
    input_path = None
    wav_path = None
    processed_wav_path = None
    output_path = None
    unique_id = str(uuid.uuid4()) # Generate ID early for logging

    try:
        # --- Request Validation ---
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided", "request_id": unique_id}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename", "request_id": unique_id}), 400

        if 'ai_decisions' not in request.form:
            return jsonify({"error": "No AI decisions (JSON string) provided", "request_id": unique_id}), 400
        ai_decisions_str = request.form['ai_decisions']
        output_format = request.form.get('output_format', 'wav').lower().strip('.')

        # --- Parse Instructions ---
        try:
            ai_decisions = json.loads(ai_decisions_str)
            if not isinstance(ai_decisions, dict): raise ValueError("JSON root must be an object.")
            logging.info(f"Request {unique_id}: Received AI Decisions: {json.dumps(ai_decisions, indent=2)}")
        except Exception as e:
            logging.error(f"Request {unique_id}: Invalid AI decisions JSON: {e}")
            return jsonify({"error": f"Invalid AI decisions JSON: {e}", "request_id": unique_id}), 400

        # --- File Handling & Conversion ---
        _, file_ext = os.path.splitext(file.filename)
        file_ext = file_ext.lower().strip('.') or 'bin' # Handle no extension

        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_converted.wav") # Standard internal format
        processed_wav_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_processed.wav")
        output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_final.{output_format}")

        file.save(input_path)
        logging.info(f"Request {unique_id}: Saved input file '{os.path.basename(input_path)}'")

        # Convert to WAV for internal processing
        if file_ext != 'wav':
            logging.info(f"Request {unique_id}: Converting '{os.path.basename(input_path)}' to WAV...")
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert input file to WAV")
        else:
            wav_path = input_path # Use the original if it was already WAV

        # --- Read Audio Data ---
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None: raise ValueError("Failed to read WAV audio data")
        if audio_data.size == 0: raise ValueError("Audio file is empty after reading")
        if sample_rate != 44100:
             logging.warning(f"Request {unique_id}: Input sample rate is {sample_rate}Hz, but processing functions assume 44100Hz. This WILL cause issues.")
             # You must resample here. Using pydub's conversion should have fixed this,
             # but we add a check anyway.
             # If this log appears, the convert_to_wav function failed to resample.

        logging.info(f"Request {unique_id}: 🎵 Starting DSP processing. SR={sample_rate}, Shape={audio_data.shape}")
        check_audio_level(audio_data, "ORIGINAL AUDIO")
        current_audio = audio_data.copy()

        # --- Execute Processing Phases ---
        def get_params(phase_name, effect_name):
             # Helper to safely get nested parameters
             return ai_decisions.get(phase_name, {}).get(effect_name, {})

        # PHASE 1: Restoration
        if ai_decisions.get('restoration'):
            logging.info(f"Request {unique_id}: 🧹 PHASE 1: RESTORATION")
            params = get_params('restoration', 'noise_reduction')
            if params and params.get('apply', False): # Check if AI wants to apply it
                 rms, peak = check_audio_level(current_audio, "Before Noise Gate Check")
                 if peak > 0.005: # Stricter check for silence
                     current_audio = apply_noise_gate(current_audio,
                         threshold_db=params.get('threshold_db', -70),
                         ratio=params.get('ratio', 5),
                         attack_ms=params.get('attack_ms', 2),
                         release_ms=params.get('release_ms', 150),
                         sample_rate=sample_rate)
                 else:
                     logging.warning(f"Request {unique_id}: Audio too quiet, skipping noise gate.")

            params = get_params('restoration', 'deessing')
            if params and params.get('apply', False):
                reduction = params.get('reduction_db', 6)
                if 0 < reduction <= 12: # Allow slightly more reduction, check > 0
                    freq = params.get('frequency_hz', 7500)
                    threshold = params.get('threshold_db', -22)
                    current_audio = apply_deesser(current_audio,
                        freq_hz=freq,
                        threshold_db=threshold,
                        reduction_db=reduction,
                        sample_rate=sample_rate)
                else:
                    logging.warning(f"Request {unique_id}: De-esser reduction invalid ({reduction}dB), skipping.")

        # PHASE 2: Mixing
        if ai_decisions.get('mixing'):
            logging.info(f"Request {unique_id}: 🎛️ PHASE 2: MIXING")
            hp_params = get_params('mixing', 'highpass')
            if hp_params is None or hp_params.get('apply', True):
                hp_freq = hp_params.get('frequency', 80) if hp_params else 80
                current_audio = apply_parametric_eq(current_audio, frequency=hp_freq, gain_db=0, q_factor=1.0, filter_type='highpass', sample_rate=sample_rate)
                check_audio_level(current_audio, "After Highpass")

            eq_params = get_params('mixing', 'equalizer')
            if eq_params and 'bands' in eq_params and isinstance(eq_params['bands'], list):
                 logging.info(f"Request {unique_id}: Applying {len(eq_params['bands'])} EQ bands...")
                 for band_idx, band in enumerate(eq_params['bands'][:8]): # Limit bands
                     if isinstance(band, dict) and band.get('apply', True):
                         current_audio = apply_parametric_eq(current_audio,
                             frequency=band.get('frequency', 1000),
                             gain_db=band.get('gain_db', 0),
                             q_factor=band.get('q_factor', 1.0),
                             filter_type=band.get('type', 'peak'),
                             sample_rate=sample_rate)
                         check_audio_level(current_audio, f"After EQ Band {band_idx+1}")
                     else:
                        logging.info(f"Request {unique_id}: Skipping EQ Band {band_idx+1} (not applying or invalid format).")

            comp_params = get_params('mixing', 'compression')
            if comp_params and comp_params.get('apply', False):
                ratio_val = 4.0
                ratio_str = comp_params.get('ratio', '4:1')
                try: ratio_val = float(ratio_str.split(':')[0])
                except: pass
                current_audio = apply_compressor(current_audio,
                    threshold_db=comp_params.get('threshold_db', -18),
                    ratio=ratio_val,
                    attack_ms=comp_params.get('attack_ms', 10),
                    release_ms=comp_params.get('release_ms', 100),
                    makeup_gain_db=comp_params.get('makeup_gain_db', 0),
                    knee_db=comp_params.get('knee_db', 3),
                    sample_rate=sample_rate)

        # PHASE 3: Mastering
        if ai_decisions.get('mastering'):
            logging.info(f"Request {unique_id}: ✨ PHASE 3: MASTERING")
            stereo_params = get_params('mastering', 'stereo_imaging')
            if stereo_params and stereo_params.get('apply', False):
                 current_audio = apply_stereo_widener(current_audio,
                     width_percent=stereo_params.get('width_percent', 110))

            sat_params = get_params('mastering', 'saturation')
            if sat_params and sat_params.get('apply', False):
                current_audio = apply_saturation(current_audio,
                    drive_db=sat_params.get('drive_db', 2),
                    mix=sat_params.get('mix', 0.1))

            limit_params = get_params('mastering', 'limiting')
            if limit_params and limit_params.get('apply', True):
                current_audio = apply_loudness_normalization(current_audio,
                    target_lufs=limit_params.get('target_lufs', -14),
                    sample_rate=sample_rate)
                current_audio = apply_brickwall_limiter(current_audio,
                    ceiling_db=limit_params.get('ceiling_db', -0.5),
                    release_ms=limit_params.get('release_ms', 50),
                    sample_rate=sample_rate)
            else:
                 current_audio = apply_brickwall_limiter(current_audio, ceiling_db=-0.5, sample_rate=sample_rate)


        # --- Final Sanity Check ---
        check_audio_level(current_audio, "FINAL PROCESSED AUDIO")
        if not isinstance(current_audio, np.ndarray) or current_audio.size == 0:
             raise ValueError("Processing resulted in empty audio data.")

        # --- Write Final Output ---
        success = write_wav(processed_wav_path, current_audio, sample_rate)
        if not success: raise ValueError("Failed to write final processed WAV")

        if output_format != 'wav':
            logging.info(f"Request {unique_id}: Converting processed WAV to {output_format}...")
            if not convert_from_wav(processed_wav_path, output_path, output_format):
                raise ValueError(f"Failed to convert WAV to {output_format}")
        else:
            output_path = processed_wav_path

        if not os.path.exists(output_path):
             raise ValueError("Final output file does not exist after processing and conversion.")

        logging.info(f"Request {unique_id}: ✅ Processing successful. Sending '{os.path.basename(output_path)}'")

        # --- Send File Response ---
        mime_types = {
            'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'flac': 'audio/flac',
            'aac': 'audio/aac', 'm4a': 'audio/mp4', 'ogg': 'audio/ogg'
        }
        
        # --- FIX FOR FILE DOWNLOAD ISSUE: Add file existence check ---
        logging.info(f"Request {unique_id}: Checking file existence at {output_path} before sending...")
        if not os.path.exists(output_path):
            logging.error(f"Request {unique_id}: CRITICAL ERROR - File {output_path} missing right before send_file!")
            raise ValueError("Output file was confirmed created, but is now missing before sending.")
        
        response = send_file(
            output_path,
            mimetype=mime_types.get(output_format, 'application/octet-stream'),
            as_attachment=True,
            download_name=f"processed_{unique_id}.{output_format}"
        )

        # --- FIX FOR FILE DOWNLOAD ISSUE: Disable cleanup temporarily ---
        @response.call_on_close
        def cleanup_files():
            paths_to_clean = [input_path, wav_path, processed_wav_path, output_path]
            logging.info(f"Request {unique_id}: Cleanup initiated. (File deletion is temporarily disabled for testing).")
            
            # --- START TEMPORARY FIX ---
            # for p in paths_to_clean:
            #     is_being_sent = (p == output_path) # Basic check
            #     if p and os.path.exists(p) and not is_being_sent:
            #         try:
            #             os.remove(p)
            #             logging.info(f"Request {unique_id}: Cleaned up temp file: {os.path.basename(p)}")
            #         except Exception as e:
            #             logging.error(f"Request {unique_id}: Error cleaning up {os.path.basename(p)}: {e}")
            # logging.info(f"Request {unique_id}: Cleanup finished. (Deletion disabled).")
            # --- END TEMPORARY FIX ---

        return response

    # --- Main Exception Handler ---
    except ValueError as ve:
         logging.error(f"Request {unique_id}: ❌ Processing failed (ValueError): {ve}", exc_info=False)
         cleanup_temp_dirs_on_error(unique_id, input_path, wav_path, processed_wav_path, output_path)
         return jsonify({"error": f"Processing error: {ve}", "request_id": unique_id}), 400
    except Exception as e:
        logging.error(f"Request {unique_id}: ❌ Processing failed (Exception): {e}", exc_info=True)
        cleanup_temp_dirs_on_error(unique_id, input_path, wav_path, processed_wav_path, output_path)
        return jsonify({"error": f"An internal server error occurred: {e}", "request_id": unique_id}), 500

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


# ==================== Health Check & Root Endpoints ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check."""
    return jsonify({"status": "healthy", "service": "audio-processing-pro-safe"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint showing service info."""
    return jsonify({
        "service": "AI Studio Pro Audio Processing (Safe DSP Version)", "version": "1.2.0", # Version bump
        "endpoints": {
            "POST /process": "Process single stem",
            "POST /separate_stems": "Separate track into stems (returns ZIP)", # <-- ADDED
            "GET /health": "Health check"
        }
    }), 200

# ==================== Cleanup on Exit ====================
def cleanup_temp_dirs_on_exit():
    """Remove main temp directories when the application exits gracefully."""
    logging.info("Application exiting. Cleaning up main temporary directories...")
    try:
        if 'UPLOAD_FOLDER' in globals() and os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logging.info("Cleaned up UPLOAD_FOLDER on exit.")
        if 'OUTPUT_FOLDER' in globals() and os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
            logging.info("Cleaned up OUTPUT_FOLDER on exit.")
    except Exception as e:
        logging.error(f"Error cleaning up temp directories on exit: {e}")

atexit.register(cleanup_temp_dirs_on_exit)

# ==================== Run Application ====================
if __name__ == '__main__':
    # Get port from environment variable (Railway provides this) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run with Flask's built-in server for development (debug=False is important for stability)
    # For production, Gunicorn (used via Dockerfile CMD) is preferred.
    logging.info(f"Starting Flask server on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

