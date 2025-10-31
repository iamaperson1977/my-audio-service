from flask import Flask, request, send_file, jsonify
import os
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, firwin, convolve
from scipy import signal
import logging
from pydub import AudioSegment
import librosa
import tempfile
import shutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')

# ==================== FILE FORMAT HANDLING ====================

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub. Preserves original channel count."""
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Preserve original channel count (don't force stereo)
        # Only convert sample rate to 44100 for consistency
        audio = audio.set_frame_rate(44100)
        
        audio.export(output_path, format='wav')
        logging.info(f"Converted {input_path} to WAV: {output_path} (channels={audio.channels}, sr=44100)")
        return True
    except Exception as e:
        logging.error(f"Error converting to WAV: {e}")
        return False

def convert_from_wav(wav_path, output_path, output_format='mp3'):
    """Convert WAV to other formats."""
    try:
        audio = AudioSegment.from_wav(wav_path)
        
        if output_format == 'mp3':
            audio.export(output_path, format='mp3', bitrate='320k')
        elif output_format == 'flac':
            audio.export(output_path, format='flac')
        elif output_format == 'aac' or output_format == 'm4a':
            audio.export(output_path, format='aac', codec='aac', bitrate='256k')
        elif output_format == 'ogg':
            audio.export(output_path, format='ogg', codec='libvorbis')
        else:
            # Default to WAV
            audio.export(output_path, format='wav')
        
        logging.info(f"Converted WAV to {output_format}: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting from WAV: {e}")
        return False

def read_wav(filepath):
    """Reads WAV file into NumPy array."""
    try:
        audio_data, sample_rate = sf.read(filepath, dtype='float32')
        logging.info(f"Read WAV: {filepath}, SR: {sample_rate}, Shape: {audio_data.shape}")
        return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Error reading WAV: {e}")
        return None, None

def write_wav(filepath, audio_data, sample_rate):
    """Writes NumPy array to WAV file."""
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Clip audio to prevent clipping artifacts in file
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        sf.write(filepath, audio_data, sample_rate)
        logging.info(f"Wrote WAV: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error writing WAV: {e}")
        return False

# ==================== ADVANCED DSP FUNCTIONS ====================

def apply_noise_gate(audio_data, threshold_db=-60, ratio=10, attack_ms=5, release_ms=50, sample_rate=44100):
    """Professional noise gate with envelope follower."""
    try:
        if audio_data.size == 0:
            logging.warning("Empty audio data in noise gate")
            return audio_data
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1000 / (max(attack_ms, 0.1) * sample_rate))
        release_coeff = np.exp(-1000 / (max(release_ms, 0.1) * sample_rate))
        
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
        
        # Prevent division by zero
        ratio = max(ratio, 1.0)
        gain = np.where(envelope > threshold_linear, 1.0, np.minimum(envelope / (threshold_linear * ratio), 1.0))
        
        if audio_data.ndim > 1:
            output = audio_data * gain[:, np.newaxis]
        else:
            output = audio_data * gain
        
        logging.info(f"Applied noise gate: threshold={threshold_db}dB, ratio={ratio}:1")
        return output
    except Exception as e:
        logging.error(f"Error in noise gate: {e}")
        return audio_data

def apply_parametric_eq(audio_data, frequency, gain_db, q_factor, filter_type='peak', sample_rate=44100):
    """
    Professional parametric EQ with multiple filter types.
    filter_type: 'peak', 'notch', 'lowshelf', 'highshelf', 'lowpass', 'highpass'
    """
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        
        # Clamp frequency to valid range
        frequency = np.clip(frequency, 20, nyquist - 10)
        freq_normalized = frequency / nyquist
        
        # Clamp Q factor to valid range
        q_factor = np.clip(q_factor, 0.1, 20.0)
        
        # Clamp gain to reasonable range
        gain_db = np.clip(gain_db, -24, 24)
        
        if filter_type == 'peak':
            b, a = signal.iirpeak(freq_normalized, q_factor, fs=2)
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        elif filter_type == 'notch':
            b, a = signal.iirnotch(freq_normalized, q_factor, fs=2)
        elif filter_type == 'lowshelf':
            order = max(2, int(q_factor))
            b, a = signal.butter(order, freq_normalized, btype='low')
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        elif filter_type == 'highshelf':
            order = max(2, int(q_factor))
            b, a = signal.butter(order, freq_normalized, btype='high')
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        elif filter_type == 'lowpass':
            order = max(2, int(q_factor))
            b, a = signal.butter(order, freq_normalized, btype='low')
        elif filter_type == 'highpass':
            order = max(2, int(q_factor))
            b, a = signal.butter(order, freq_normalized, btype='high')
        else:
            logging.warning(f"Unknown filter type: {filter_type}, using peak")
            b, a = signal.iirpeak(freq_normalized, q_factor, fs=2)
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = lfilter(b, a, audio_data[:, ch])
        else:
            output = lfilter(b, a, audio_data)
        
        logging.info(f"Applied EQ: {filter_type} @ {frequency}Hz, {gain_db:+.1f}dB, Q={q_factor}")
        return output
    except Exception as e:
        logging.error(f"Error in parametric EQ: {e}")
        return audio_data

def apply_compressor(audio_data, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50, 
                    makeup_gain_db=0, knee_db=0, sample_rate=44100):
    """
    Professional compressor with soft knee support.
    """
    try:
        if audio_data.size == 0:
            return audio_data
        
        # Clamp parameters to valid ranges
        threshold_db = np.clip(threshold_db, -60, 0)
        ratio = np.clip(ratio, 1.0, 20.0)
        attack_ms = max(0.1, attack_ms)
        release_ms = max(1.0, release_ms)
        makeup_gain_db = np.clip(makeup_gain_db, -12, 24)
        knee_db = np.clip(knee_db, 0, 12)
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1000 / (attack_ms * sample_rate))
        release_coeff = np.exp(-1000 / (release_ms * sample_rate))
        makeup_gain = 10 ** (makeup_gain_db / 20)
        knee_width = knee_db / 2
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                output[:, ch] = compress_channel(
                    audio_data[:, ch], threshold_linear, ratio,
                    attack_coeff, release_coeff, makeup_gain, knee_width
                )
        else:
            output = compress_channel(
                audio_data, threshold_linear, ratio,
                attack_coeff, release_coeff, makeup_gain, knee_width
            )
        
        logging.info(f"Applied compressor: {ratio}:1 @ {threshold_db}dB, knee={knee_db}dB")
        return output
    except Exception as e:
        logging.error(f"Error in compressor: {e}")
        return audio_data

def compress_channel(audio_channel, threshold, ratio, attack_coeff, release_coeff, makeup_gain, knee_width):
    """Compress a single audio channel with envelope follower."""
    envelope = 0.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        
        # Envelope follower
        if input_level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * input_level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * input_level
        
        # Gain computer with soft knee
        if envelope > threshold:
            if knee_width > 0.001:
                # Soft knee
                overshoot = envelope - threshold
                if overshoot < knee_width:
                    gain_reduction = (overshoot ** 2) / (4 * knee_width * ratio) if knee_width > 0 else 0
                else:
                    gain_reduction = overshoot / ratio + knee_width / (4 * ratio)
                gain = (threshold + gain_reduction) / max(envelope, 1e-10)
            else:
                # Hard knee
                gain_reduction = threshold + (envelope - threshold) / ratio
                gain = gain_reduction / max(envelope, 1e-10)
        else:
            gain = 1.0
        
        output[i] = sample * gain * makeup_gain
    
    return output

def apply_brickwall_limiter(audio_data, ceiling_db=-0.3, release_ms=50, sample_rate=44100):
    """
    Brickwall limiter to prevent clipping.
    """
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
        
        logging.info(f"Applied brickwall limiter: ceiling={ceiling_db}dB")
        return output
    except Exception as e:
        logging.error(f"Error in limiter: {e}")
        return audio_data

def limit_channel(audio_channel, ceiling, release_coeff):
    """Limit a single channel with lookahead."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        
        # Instant attack if over ceiling
        if input_level * gain > ceiling:
            gain = ceiling / max(input_level, 1e-10)
        else:
            # Slow release back to unity
            gain = release_coeff * gain + (1 - release_coeff) * 1.0
            gain = min(gain, 1.0)  # Clamp to unity
        
        output[i] = sample * gain
    
    return output

def apply_loudness_normalization(audio_data, target_lufs=-14, sample_rate=44100):
    """
    Loudness normalization using ITU-R BS.1770 approximation.
    """
    try:
        if audio_data.size == 0:
            return audio_data
        
        # Clamp target LUFS to reasonable range
        target_lufs = np.clip(target_lufs, -24, -6)
        
        # K-weighting filter approximation (simplified)
        # High-pass at 38 Hz
        nyquist = sample_rate / 2
        hp_freq = min(38, nyquist - 10) / nyquist
        b_hp, a_hp = butter(2, hp_freq, btype='high')
        
        # High-shelf at 1.5 kHz
        hs_freq = min(1500, nyquist - 10) / nyquist
        b_hs, a_hs = butter(2, hs_freq, btype='high')
        
        if audio_data.ndim > 1:
            filtered = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                filtered[:, ch] = lfilter(b_hp, a_hp, audio_data[:, ch])
                filtered[:, ch] = lfilter(b_hs, a_hs, filtered[:, ch])
        else:
            filtered = lfilter(b_hp, a_hp, audio_data)
            filtered = lfilter(b_hs, a_hs, filtered)
        
        # Calculate mean square
        mean_square = np.mean(filtered ** 2)
        
        # Prevent log of zero
        if mean_square < 1e-10:
            logging.warning("Audio is silent, skipping loudness normalization")
            return audio_data
        
        current_lufs = -0.691 + 10 * np.log10(mean_square)
        
        # Calculate required gain
        gain_db = target_lufs - current_lufs
        
        # Clamp gain to reasonable range
        gain_db = np.clip(gain_db, -24, 24)
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        output = audio_data * gain_linear
        
        # Apply brickwall limiter to prevent clipping
        output = apply_brickwall_limiter(output, ceiling_db=-1.0, sample_rate=sample_rate)
        
        logging.info(f"Loudness normalized: {current_lufs:.1f} → {target_lufs} LUFS ({gain_db:+.1f}dB)")
        return output
    except Exception as e:
        logging.error(f"Error in loudness normalization: {e}")
        return audio_data

def apply_stereo_widener(audio_data, width_percent=150):
    """
    Stereo widening using mid-side processing.
    width_percent: 100 = no change, >100 = wider, <100 = narrower
    """
    try:
        if audio_data.ndim < 2:
            logging.warning("Stereo widener requires stereo audio")
            return audio_data
        
        if audio_data.shape[1] < 2:
            logging.warning("Stereo widener requires at least 2 channels")
            return audio_data
        
        # Clamp width to reasonable range
        width_percent = np.clip(width_percent, 0, 200)
        width = width_percent / 100.0
        
        # Convert to mid-side
        mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
        side = (audio_data[:, 0] - audio_data[:, 1]) / 2
        
        # Apply width
        side = side * width
        
        # Convert back to left-right
        output = np.zeros_like(audio_data)
        output[:, 0] = mid + side
        output[:, 1] = mid - side
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        logging.info(f"Applied stereo widener: {width_percent}% width")
        return output
    except Exception as e:
        logging.error(f"Error in stereo widener: {e}")
        return audio_data

def apply_saturation(audio_data, drive_db=6, mix=0.5):
    """
    Harmonic saturation/distortion using tanh waveshaping.
    """
    try:
        if audio_data.size == 0:
            return audio_data
        
        # Clamp parameters
        drive_db = np.clip(drive_db, 0, 24)
        mix = np.clip(mix, 0, 1)
        
        drive = 10 ** (drive_db / 20)
        
        # Apply tanh saturation
        saturated = np.tanh(audio_data * drive) / np.tanh(drive)
        
        # Mix with dry signal
        output = audio_data * (1 - mix) + saturated * mix
        
        logging.info(f"Applied saturation: drive={drive_db}dB, mix={mix*100:.0f}%")
        return output
    except Exception as e:
        logging.error(f"Error in saturation: {e}")
        return audio_data

def apply_deesser(audio_data, freq_hz=6000, threshold_db=-15, reduction_db=6, sample_rate=44100):
    """
    Advanced de-esser with sidechain detection.
    """
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        
        # Clamp parameters to valid ranges
        freq_hz = np.clip(freq_hz, 2000, nyquist - 1000)
        threshold_db = np.clip(threshold_db, -60, 0)
        reduction_db = np.clip(reduction_db, 0, 24)
        
        # Bandpass filter for sibilance detection
        low_freq = max(freq_hz - 2000, 100) / nyquist
        high_freq = min(freq_hz + 4000, nyquist - 100) / nyquist
        
        # Ensure low < high
        if low_freq >= high_freq:
            low_freq = high_freq - 0.1
        
        b_detect, a_detect = butter(2, [low_freq, high_freq], btype='band')
        
        # High-pass for processing
        process_freq = freq_hz / nyquist
        b_process, a_process = butter(2, process_freq, btype='high')
        
        threshold_linear = 10 ** (threshold_db / 20)
        reduction_linear = 10 ** (-reduction_db / 20)
        
        if audio_data.ndim > 1:
            output = np.zeros_like(audio_data)
            for ch in range(audio_data.shape[1]):
                # Detect sibilance
                detect_signal = lfilter(b_detect, a_detect, audio_data[:, ch])
                envelope = np.abs(detect_signal)
                
                # Smooth envelope
                envelope = np.convolve(envelope, np.ones(100)/100, mode='same')
                
                # Create gain reduction
                gain = np.where(envelope > threshold_linear, reduction_linear, 1.0)
                
                # Apply only to high frequencies
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
        
        logging.info(f"Applied de-esser: {freq_hz}Hz, threshold={threshold_db}dB, reduction={reduction_db}dB")
        return output
    except Exception as e:
        logging.error(f"Error in de-esser: {e}")
        return audio_data

# ==================== MAIN PROCESSING ENDPOINT ====================

@app.route('/process', methods=['POST'])
def process_audio():
    """
    Main endpoint that receives audio and AI instructions.
    Supports multiple input/output formats.
    """
    
    input_path = None
    wav_path = None
    processed_wav_path = None
    output_path = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        if 'ai_decisions' not in request.form:
            return jsonify({"error": "No AI decisions provided"}), 400
        
        file = request.files['file']
        output_format = request.form.get('output_format', 'wav').lower()
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Parse AI decisions
        import json
        try:
            ai_decisions = json.loads(request.form['ai_decisions'])
        except Exception as e:
            return jsonify({"error": f"Invalid AI decisions JSON: {e}"}), 400
        
        # Generate unique IDs
        unique_id = str(uuid.uuid4())
        
        # Get file extension safely
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'unknown'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_converted.wav")
        processed_wav_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_processed.wav")
        output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_final.{output_format}")
        
        # Save uploaded file
        file.save(input_path)
        logging.info(f"Saved input file: {input_path}")
        
        # Convert to WAV if not already
        if file_ext != 'wav':
            logging.info(f"Converting {input_path} to WAV...")
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert input file to WAV")
        else:
            wav_path = input_path
        
        # Read audio
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None:
            raise ValueError("Failed to read audio file")
        
        # Validate audio data
        if audio_data.size == 0:
            raise ValueError("Audio file is empty")
        
        logging.info(f"🎵 Processing audio: SR={sample_rate}, Shape={audio_data.shape}")
        current_audio = audio_data.copy()
        
        # === PHASE 1: RESTORATION ===
        if 'restoration' in ai_decisions:
            logging.info("🧹 PHASE 1: RESTORATION")
            restoration = ai_decisions['restoration']
            
            # Noise reduction (noise gate)
            if 'noise_reduction' in restoration:
                params = restoration['noise_reduction']
                current_audio = apply_noise_gate(
                    current_audio,
                    threshold_db=params.get('threshold_db', -60),
                    ratio=10,
                    sample_rate=sample_rate
                )
            
            # De-essing
            if 'deessing' in restoration:
                params = restoration['deessing']
                freq_range = params.get('frequency_range', '5000-10000')
                try:
                    freq = int(freq_range.split('-')[0])
                except (ValueError, AttributeError):
                    freq = 6000
                threshold = params.get('threshold_db', -15)
                reduction = params.get('reduction_db', 6)
                current_audio = apply_deesser(
                    current_audio, 
                    freq_hz=freq, 
                    threshold_db=threshold,
                    reduction_db=reduction,
                    sample_rate=sample_rate
                )
        
        # === PHASE 2: MIXING ===
        if 'mixing' in ai_decisions:
            logging.info("🎛️ PHASE 2: MIXING")
            mixing = ai_decisions['mixing']
            
            # Highpass filter (remove rumble)
            current_audio = apply_parametric_eq(
                current_audio, 
                frequency=80, 
                gain_db=0, 
                q_factor=2, 
                filter_type='highpass',
                sample_rate=sample_rate
            )
            
            # Apply EQ bands
            if 'equalizer' in mixing and 'bands' in mixing['equalizer']:
                for band in mixing['equalizer']['bands']:
                    filter_type = 'peak'  # Default
                    
                    # Determine filter type from band properties
                    band_type = band.get('type', 'peak')
                    if band_type == 'highpass' or band_type == 'lowpass':
                        filter_type = band_type
                    elif band.get('type') == 'cut' and band.get('gain_db', 0) < -6:
                        filter_type = 'notch'
                    elif band.get('frequency', 0) < 150 and band.get('gain_db', 0) != 0:
                        filter_type = 'lowshelf'
                    elif band.get('frequency', 0) > 8000 and band.get('gain_db', 0) != 0:
                        filter_type = 'highshelf'
                    
                    current_audio = apply_parametric_eq(
                        current_audio,
                        frequency=band.get('frequency', 1000),
                        gain_db=band.get('gain_db', 0),
                        q_factor=band.get('q_factor', 1.0),
                        filter_type=filter_type,
                        sample_rate=sample_rate
                    )
            
            # Apply compression
            if 'compression' in mixing:
                comp = mixing['compression']
                ratio_str = comp.get('ratio', '4:1')
                try:
                    ratio = float(ratio_str.split(':')[0])
                except (ValueError, AttributeError):
                    ratio = 4.0
                
                current_audio = apply_compressor(
                    current_audio,
                    threshold_db=comp.get('threshold_db', -20),
                    ratio=ratio,
                    attack_ms=comp.get('attack_ms', 5),
                    release_ms=comp.get('release_ms', 50),
                    makeup_gain_db=comp.get('makeup_gain_db', 3),
                    knee_db=comp.get('knee_db', 2),
                    sample_rate=sample_rate
                )
        
        # === PHASE 3: MASTERING ===
        if 'mastering' in ai_decisions:
            logging.info("✨ PHASE 3: MASTERING")
            mastering = ai_decisions['mastering']
            
            # Stereo widening (if stereo)
            if current_audio.ndim > 1 and current_audio.shape[1] >= 2:
                current_audio = apply_stereo_widener(current_audio, width_percent=120)
            
            # Subtle saturation
            current_audio = apply_saturation(current_audio, drive_db=3, mix=0.2)
            
            # Loudness normalization
            if 'limiting' in mastering:
                target_lufs = mastering['limiting'].get('target_lufs', -14)
                current_audio = apply_loudness_normalization(
                    current_audio,
                    target_lufs=target_lufs,
                    sample_rate=sample_rate
                )
        
        # Write processed WAV
        success = write_wav(processed_wav_path, current_audio, sample_rate)
        if not success:
            raise ValueError("Failed to write processed WAV file")
        
        # Convert to requested output format
        if output_format != 'wav':
            logging.info(f"Converting to {output_format}...")
            if not convert_from_wav(processed_wav_path, output_path, output_format):
                raise ValueError(f"Failed to convert to {output_format}")
        else:
            output_path = processed_wav_path
        
        # Verify output file exists
        if not os.path.exists(output_path):
            raise ValueError("Output file was not created")
        
        # Return processed audio
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'flac': 'audio/flac',
            'aac': 'audio/aac',
            'm4a': 'audio/aac',
            'ogg': 'audio/ogg'
        }
        
        logging.info(f"✅ Processing complete! Returning {output_format}")
        
        # Use as_attachment=False to prevent immediate deletion
        response = send_file(
            output_path,
            mimetype=mime_types.get(output_format, 'audio/wav'),
            as_attachment=True,
            download_name=f'processed_{unique_id}.{output_format}'
        )
        
        # Cleanup temp files after sending (will happen after response is sent)
        @response.call_on_close
        def cleanup():
            try:
                if input_path and os.path.exists(input_path) and input_path != wav_path:
                    os.remove(input_path)
                if wav_path and os.path.exists(wav_path) and wav_path != input_path and wav_path != processed_wav_path:
                    os.remove(wav_path)
                if processed_wav_path and os.path.exists(processed_wav_path) and processed_wav_path != output_path:
                    os.remove(processed_wav_path)
                # Note: Don't delete output_path as it's still being sent
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
        
        return response
    
    except Exception as e:
        # Cleanup on error
        try:
            for path in [input_path, wav_path, processed_wav_path, output_path]:
                if path and os.path.exists(path):
                    os.remove(path)
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")
        
        logging.error(f"❌ Processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "service": "audio-processing-pro",
        "upload_folder": UPLOAD_FOLDER,
        "output_folder": OUTPUT_FOLDER
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "service": "AI Studio Pro Audio Processing",
        "version": "1.0.0",
        "endpoints": {
            "POST /process": "Process audio with AI decisions",
            "GET /health": "Health check"
        }
    }), 200

if __name__ == '__main__':
    import atexit
    
    # Cleanup temp directories on exit
    def cleanup_temp_dirs():
        try:
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
            if os.path.exists(OUTPUT_FOLDER):
                shutil.rmtree(OUTPUT_FOLDER)
            logging.info("Cleaned up temporary directories")
        except Exception as e:
            logging.error(f"Error cleaning up temp directories: {e}")
    
    atexit.register(cleanup_temp_dirs)
    
    app.run(host='0.0.0.0', port=5000, debug=False)







