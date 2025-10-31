from flask import Flask, request, send_file, jsonify
import os
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from scipy import signal
import logging
from pydub import AudioSegment
import tempfile
import shutil
import numba
import zipfile
import subprocess
import sys

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')

# ==================== FILE FORMAT HANDLING ====================

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(44100)
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
            audio.export(output_path, format='mp3', bitrate='320k')
        elif output_format == 'flac':
            audio.export(output_path, format='flac')
        elif output_format in ['aac', 'm4a']:
            audio.export(output_path, format='aac', codec='aac', bitrate='256k')
        elif output_format == 'ogg':
            audio.export(output_path, format='ogg', codec='libvorbis')
        else:
            audio.export(output_path, format='wav')
        
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
        
        ratio = max(ratio, 1.0)
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
        frequency = np.clip(frequency, 20, nyquist - 10)
        freq_normalized = frequency / nyquist
        q_factor = np.clip(q_factor, 0.1, 20.0)
        gain_db = np.clip(gain_db, -24, 24)
        
        if filter_type == 'peak':
            b, a = signal.iirpeak(freq_normalized, q_factor, fs=2)
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        elif filter_type == 'notch':
            b, a = signal.iirnotch(freq_normalized, q_factor, fs=2)
        elif filter_type in ['lowshelf', 'highshelf']:
            order = max(2, int(q_factor))
            btype = 'low' if filter_type == 'lowshelf' else 'high'
            b, a = signal.butter(order, freq_normalized, btype=btype)
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        elif filter_type in ['lowpass', 'highpass']:
            order = max(2, int(q_factor))
            btype = 'low' if filter_type == 'lowpass' else 'high'
            b, a = signal.butter(order, freq_normalized, btype=btype)
        else:
            b, a = signal.iirpeak(freq_normalized, q_factor, fs=2)
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
        
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
        
        logging.info(f"Applied compressor: {ratio}:1 @ {threshold_db}dB")
        return output
    except Exception as e:
        logging.error(f"Compressor error: {e}")
        return audio_data

@numba.jit(nopython=True)
def compress_channel(audio_channel, threshold, ratio, attack_coeff, release_coeff, makeup_gain, knee_width):
    """Compress single channel."""
    envelope = 0.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        
        if input_level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * input_level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * input_level
        
        if envelope > threshold:
            if knee_width > 0.001:
                overshoot = envelope - threshold
                if overshoot < knee_width:
                    gain_reduction = (overshoot ** 2) / (4 * knee_width * ratio) if knee_width > 0 else 0
                else:
                    gain_reduction = overshoot / ratio + knee_width / (4 * ratio)
                gain = (threshold + gain_reduction) / max(envelope, 1e-10)
            else:
                gain_reduction = threshold + (envelope - threshold) / ratio
                gain = gain_reduction / max(envelope, 1e-10)
        else:
            gain = 1.0
        
        output[i] = sample * gain * makeup_gain
    
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

@numba.jit(nopython=True)
def limit_channel(audio_channel, ceiling, release_coeff):
    """Limit single channel."""
    gain = 1.0
    output = np.zeros_like(audio_channel)
    
    for i, sample in enumerate(audio_channel):
        input_level = abs(sample)
        
        if input_level * gain > ceiling:
            gain = ceiling / max(input_level, 1e-10)
        else:
            gain = release_coeff * gain + (1 - release_coeff) * 1.0
            gain = min(gain, 1.0)
        
        output[i] = sample * gain
    
    return output

def apply_loudness_normalization(audio_data, target_lufs=-14, sample_rate=44100):
    """Loudness normalization."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        target_lufs = np.clip(target_lufs, -24, -6)
        
        nyquist = sample_rate / 2
        hp_freq = min(38, nyquist - 10) / nyquist
        b_hp, a_hp = butter(2, hp_freq, btype='high')
        
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
        
        mean_square = np.mean(filtered ** 2)
        
        if mean_square < 1e-10:
            logging.warning("Silent audio, skipping normalization")
            return audio_data
        
        current_lufs = -0.691 + 10 * np.log10(mean_square)
        gain_db = target_lufs - current_lufs
        gain_db = np.clip(gain_db, -24, 24)
        gain_linear = 10 ** (gain_db / 20)
        
        output = audio_data * gain_linear
        output = apply_brickwall_limiter(output, ceiling_db=-1.0, sample_rate=sample_rate)
        
        logging.info(f"Normalized: {current_lufs:.1f} → {target_lufs} LUFS ({gain_db:+.1f}dB)")
        return output
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        return audio_data

def apply_stereo_widener(audio_data, width_percent=150):
    """Stereo widening."""
    try:
        if audio_data.ndim < 2 or audio_data.shape[1] < 2:
            return audio_data
        
        width_percent = np.clip(width_percent, 0, 200)
        width = width_percent / 100.0
        
        mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
        side = (audio_data[:, 0] - audio_data[:, 1]) / 2
        side = side * width
        
        output = np.zeros_like(audio_data)
        output[:, 0] = mid + side
        output[:, 1] = mid - side
        
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        logging.info(f"Stereo widener: {width_percent}%")
        return output
    except Exception as e:
        logging.error(f"Widener error: {e}")
        return audio_data

def apply_saturation(audio_data, drive_db=6, mix=0.5):
    """Harmonic saturation."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        drive_db = np.clip(drive_db, 0, 24)
        mix = np.clip(mix, 0, 1)
        
        drive = 10 ** (drive_db / 20)
        saturated = np.tanh(audio_data * drive) / np.tanh(drive)
        output = audio_data * (1 - mix) + saturated * mix
        
        logging.info(f"Saturation: drive={drive_db}dB, mix={mix*100:.0f}%")
        return output
    except Exception as e:
        logging.error(f"Saturation error: {e}")
        return audio_data

def apply_deesser(audio_data, freq_hz=6000, threshold_db=-15, reduction_db=6, sample_rate=44100):
    """De-esser."""
    try:
        if audio_data.size == 0:
            return audio_data
        
        nyquist = sample_rate / 2
        freq_hz = np.clip(freq_hz, 2000, nyquist - 1000)
        threshold_db = np.clip(threshold_db, -60, 0)
        reduction_db = np.clip(reduction_db, 0, 24)
        
        low_freq = max(freq_hz - 2000, 100) / nyquist
        high_freq = min(freq_hz + 4000, nyquist - 100) / nyquist
        
        if low_freq >= high_freq:
            low_freq = high_freq - 0.1
        
        b_detect, a_detect = butter(2, [low_freq, high_freq], btype='band')
        process_freq = freq_hz / nyquist
        b_process, a_process = butter(2, process_freq, btype='high')
        
        threshold_linear = 10 ** (threshold_db / 20)
        reduction_linear = 10 ** (-reduction_db / 20)
        
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
    """Separates audio into stems using demucs."""
    input_path = None
    wav_path = None
    output_dir = None
    zip_path = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        unique_id = str(uuid.uuid4())
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.wav")
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
        zip_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems.zip")
        
        file.save(input_path)
        logging.info(f"📁 Saved: {input_path}")
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert to WAV")
        else:
            wav_path = input_path
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"🎵 Running demucs...")
        
        result = subprocess.run([
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs',
            '-o', output_dir,
            '--mp3',
            '--mp3-bitrate', '320',
            wav_path
        ], check=True, capture_output=True, text=True, timeout=600)
        
        logging.info(f"✅ Demucs complete")
        
        # Find stem directory
        stem_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(wav_path))[0])
        
        if not os.path.exists(stem_dir):
            # Try alternate paths
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_dir = root
                    break
            
            if not os.path.exists(stem_dir):
                raise ValueError(f"Stem directory not found: {stem_dir}")
        
        logging.info(f"Found stems: {stem_dir}")
        
        # Create ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for stem_name in ['vocals', 'drums', 'bass', 'other']:
                stem_file = os.path.join(stem_dir, f"{stem_name}.mp3")
                if os.path.exists(stem_file):
                    zipf.write(stem_file, f"{stem_name}.mp3")
                    logging.info(f"✅ Added {stem_name}.mp3")
                else:
                    logging.warning(f"⚠️ Missing {stem_name}.mp3")
        
        if not os.path.exists(zip_path):
            raise ValueError("Failed to create ZIP")
        
        response = send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'stems_{unique_id}.zip'
        )
        
        @response.call_on_close
        def cleanup():
            try:
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)
                if wav_path and os.path.exists(wav_path) and wav_path != input_path:
                    os.remove(wav_path)
                if output_dir and os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
        
        return response
    
    except subprocess.TimeoutExpired:
        logging.error("❌ Demucs timeout (>10 minutes)")
        return jsonify({"error": "Processing timeout. Audio file may be too long."}), 500
    
    except Exception as e:
        try:
            for path in [input_path, wav_path, zip_path]:
                if path and os.path.exists(path):
                    os.remove(path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except:
            pass
        
        logging.error(f"❌ Stem separation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==================== MAIN PROCESSING ====================

@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with AI decisions."""
    input_path = None
    wav_path = None
    processed_wav_path = None
    output_path = None
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        if 'ai_decisions' not in request.form:
            return jsonify({"error": "No AI decisions"}), 400
        
        file = request.files['file']
        output_format = request.form.get('output_format', 'wav').lower()
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        import json
        ai_decisions = json.loads(request.form['ai_decisions'])
        
        unique_id = str(uuid.uuid4())
        filename_parts = file.filename.rsplit('.', 1)
        file_ext = filename_parts[1].lower() if len(filename_parts) > 1 else 'mp3'
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_converted.wav")
        processed_wav_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_processed.wav")
        output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_final.{output_format}")
        
        file.save(input_path)
        
        if file_ext != 'wav':
            if not convert_to_wav(input_path, wav_path):
                raise ValueError("Failed to convert to WAV")
        else:
            wav_path = input_path
        
        audio_data, sample_rate = read_wav(wav_path)
        if audio_data is None:
            raise ValueError("Failed to read audio")
        
        if audio_data.size == 0:
            raise ValueError("Empty audio file")
        
        logging.info(f"🎵 Processing: SR={sample_rate}, Shape={audio_data.shape}")
        current_audio = audio_data.copy()
        
        # RESTORATION
        if 'restoration' in ai_decisions:
            logging.info("🧹 RESTORATION")
            restoration = ai_decisions['restoration']
            
            if 'noise_reduction' in restoration:
                params = restoration['noise_reduction']
                current_audio = apply_noise_gate(
                    current_audio,
                    threshold_db=params.get('threshold_db', -60),
                    ratio=10,
                    sample_rate=sample_rate
                )
            
            if 'deessing' in restoration:
                params = restoration['deessing']
                freq_range = params.get('frequency_range', '5000-10000')
                try:
                    freq = int(freq_range.split('-')[0])
                except:
                    freq = 6000
                current_audio = apply_deesser(
                    current_audio, 
                    freq_hz=freq, 
                    threshold_db=params.get('threshold_db', -15),
                    reduction_db=params.get('reduction_db', 6),
                    sample_rate=sample_rate
                )
        
        # MIXING
        if 'mixing' in ai_decisions:
            logging.info("🎛️ MIXING")
            mixing = ai_decisions['mixing']
            
            current_audio = apply_parametric_eq(
                current_audio, 80, 0, 2, 'highpass', sample_rate
            )
            
            if 'equalizer' in mixing and 'bands' in mixing['equalizer']:
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
            
            if 'compression' in mixing:
                comp = mixing['compression']
                ratio_str = comp.get('ratio', '4:1')
                try:
                    ratio = float(ratio_str.split(':')[0])
                except:
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
        
        # MASTERING
        if 'mastering' in ai_decisions:
            logging.info("✨ MASTERING")
            mastering = ai_decisions['mastering']
            
            if current_audio.ndim > 1 and current_audio.shape[1] >= 2:
                current_audio = apply_stereo_widener(current_audio, 120)
            
            current_audio = apply_saturation(current_audio, 3, 0.2)
            
            if 'limiting' in mastering:
                target_lufs = mastering['limiting'].get('target_lufs', -14)
                current_audio = apply_loudness_normalization(
                    current_audio, target_lufs, sample_rate
                )
        
        if not write_wav(processed_wav_path, current_audio, sample_rate):
            raise ValueError("Failed to write WAV")
        
        if output_format != 'wav':
            if not convert_from_wav(processed_wav_path, output_path, output_format):
                raise ValueError(f"Failed to convert to {output_format}")
        else:
            output_path = processed_wav_path
        
        if not os.path.exists(output_path):
            raise ValueError("Output file missing")
        
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'flac': 'audio/flac',
            'aac': 'audio/aac',
            'm4a': 'audio/aac',
            'ogg': 'audio/ogg'
        }
        
        logging.info(f"✅ Complete! Returning {output_format}")
        
        response = send_file(
            output_path,
            mimetype=mime_types.get(output_format, 'audio/wav'),
            as_attachment=True,
            download_name=f'processed_{unique_id}.{output_format}'
        )
        
        @response.call_on_close
        def cleanup():
            try:
                for path in [input_path, wav_path, processed_wav_path]:
                    if path and os.path.exists(path) and path != output_path:
                        os.remove(path)
            except:
                pass
        
        return response
    
    except Exception as e:
        try:
            for path in [input_path, wav_path, processed_wav_path, output_path]:
                if path and os.path.exists(path):
                    os.remove(path)
        except:
            pass
        
        logging.error(f"❌ Processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check."""
    return jsonify({"status": "healthy", "service": "AI Studio Pro"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root."""
    return jsonify({
        "service": "AI Studio Pro Audio Processing",
        "version": "2.0.0",
        "endpoints": {
            "POST /separate_stems": "Separate audio into stems",
            "POST /process": "Process audio with AI",
            "GET /health": "Health check"
        }
    }), 200

if __name__ == '__main__':
    import atexit
    
    def cleanup_temp_dirs():
        try:
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
            if os.path.exists(OUTPUT_FOLDER):
                shutil.rmtree(OUTPUT_FOLDER)
            logging.info("Cleaned up temp dirs")
        except:
            pass
    
    atexit.register(cleanup_temp_dirs)
    
    app.run(host='0.0.0.0', port=5000, debug=False)







