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
import threading
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Use temp directory for better cleanup
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='audio_upload_')
OUTPUT_FOLDER = tempfile.mkdtemp(prefix='audio_output_')

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

# DSP FUNCTIONS
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    """Design a lowpass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_highpass(cutoff, fs, order=5):
    """Design a highpass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

@numba.jit(nopython=True)
def simple_compressor_jit(audio, threshold, ratio, attack_samples, release_samples):
    """Simple compressor using Numba for speed."""
    output = np.zeros_like(audio)
    envelope = 0.0
    
    for i in range(len(audio)):
        input_level = abs(audio[i])
        
        # Envelope follower
        if input_level > envelope:
            envelope += (input_level - envelope) / attack_samples
        else:
            envelope += (input_level - envelope) / release_samples
        
        # Apply compression
        if envelope > threshold:
            gain_reduction = 1.0 - ((envelope - threshold) * (1.0 - 1.0/ratio))
            output[i] = audio[i] * max(0.0, min(1.0, gain_reduction))
        else:
            output[i] = audio[i]
    
    return output

def apply_compression(audio, sr, threshold_db=-20, ratio=4.0, attack_ms=10, release_ms=100):
    """Apply dynamic range compression."""
    try:
        threshold = 10 ** (threshold_db / 20.0)
        attack_samples = max(1, int(sr * attack_ms / 1000.0))
        release_samples = max(1, int(sr * release_ms / 1000.0))
        
        if audio.ndim == 1:
            compressed = simple_compressor_jit(audio, threshold, ratio, attack_samples, release_samples)
        else:
            compressed = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                compressed[:, ch] = simple_compressor_jit(audio[:, ch], threshold, ratio, attack_samples, release_samples)
        
        return compressed
    except Exception as e:
        logging.error(f"Compression error: {e}")
        return audio

def apply_eq_band(audio, sr, frequency, gain_db, q_factor=1.0):
    """Apply EQ to a specific frequency band using peaking filter."""
    try:
        if abs(gain_db) < 0.1:
            return audio
        
        gain = 10 ** (gain_db / 20.0)
        w0 = 2 * np.pi * frequency / sr
        alpha = np.sin(w0) / (2 * q_factor)
        
        # Peaking EQ coefficients
        A = np.sqrt(gain)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        
        if audio.ndim == 1:
            filtered = lfilter(b, a, audio)
        else:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = lfilter(b, a, audio[:, ch])
        
        return filtered
    except Exception as e:
        logging.error(f"EQ error: {e}")
        return audio

def spectral_gate(audio, sr, threshold_db=-60, reduction_db=15):
    """Apply spectral gating for noise reduction."""
    try:
        from scipy import fft
        
        # STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2
        
        if audio.ndim == 1:
            channels = [audio]
        else:
            channels = [audio[:, i] for i in range(audio.shape[1])]
        
        output_channels = []
        
        for channel in channels:
            f, t, Zxx = signal.stft(channel, sr, nperseg=nperseg, noverlap=noverlap)
            
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            threshold = 10 ** (threshold_db / 20.0)
            reduction = 10 ** (-reduction_db / 20.0)
            
            mask = np.where(magnitude > threshold, 1.0, reduction)
            Zxx_filtered = magnitude * mask * np.exp(1j * phase)
            
            _, audio_filtered = signal.istft(Zxx_filtered, sr, nperseg=nperseg, noverlap=noverlap)
            
            audio_filtered = audio_filtered[:len(channel)]
            output_channels.append(audio_filtered)
        
        if audio.ndim == 1:
            return output_channels[0]
        else:
            return np.column_stack(output_channels)
    except Exception as e:
        logging.error(f"Spectral gate error: {e}")
        return audio

def normalize_audio(audio, target_lufs=-14.0):
    """Normalize audio to target LUFS."""
    try:
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-6:
            return audio
        
        current_db = 20 * np.log10(rms)
        target_db = target_lufs
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20.0)
        
        normalized = audio * gain
        
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
        
        return normalized
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        return audio

# NEW: Endpoint to download temporary stems ZIP
@app.route('/download_temp_stems/<filename>', methods=['GET'])
def download_temp_stems(filename):
    """Serve temporary stems ZIP file for download."""
    try:
        # Security: only allow ZIP files from OUTPUT_FOLDER
        if not filename.endswith('.zip'):
            return jsonify({"error": "Invalid file type"}), 400
        
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
        
        logging.info(f"Serving temp file: {filename}")
        
        # Send file and schedule cleanup after download
        response = send_file(
            file_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
        
        # Clean up the file after sending
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Cleaned up temp file: {filename}")
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
        
        return response
        
    except Exception as e:
        logging.error(f"Error serving temp file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ASYNC STEM SEPARATION ENDPOINT
@app.route('/separate_stems_async', methods=['POST'])
def separate_stems_async():
    """Start async stem separation and call webhook when done."""
    try:
        data = request.json
        file_url = data.get('file_url')
        project_id = data.get('project_id')
        webhook_url = data.get('webhook_url')
        service_role_key = data.get('service_role_key')
        
        if not all([file_url, project_id, webhook_url, service_role_key]):
            return jsonify({"error": "Missing parameters"}), 400
        
        logging.info(f"🎵 Starting async stem separation for project {project_id}")
        
        # Get the public URL of this service (for download endpoint)
        # Use environment variable or construct from request
        service_url = os.getenv('PYTHON_AUDIO_SERVICE_URL', request.url_root.rstrip('/'))
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_stems_async,
            args=(file_url, project_id, webhook_url, service_role_key, service_url)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Stem separation started",
            "project_id": project_id
        }), 200
        
    except Exception as e:
        logging.error(f"❌ Error starting async separation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def process_stems_async(file_url, project_id, webhook_url, service_role_key, service_url):
    """Background thread to process stems and call webhook when done."""
    input_path = None
    wav_path = None
    output_dir = None
    zip_path = None
    
    try:
        logging.info(f"📥 Downloading audio from {file_url[:50]}...")
        
        # Download file
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.mp3")
        
        audio_response = requests.get(file_url, timeout=60)
        audio_response.raise_for_status()
        
        with open(input_path, 'wb') as f:
            f.write(audio_response.content)
        
        logging.info(f"✅ Downloaded {len(audio_response.content)} bytes")
        
        # Convert to WAV
        wav_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.wav")
        if not convert_to_wav(input_path, wav_path):
            raise ValueError("Failed to convert to WAV")
        
        output_dir = os.path.join(OUTPUT_FOLDER, f"{unique_id}_stems")
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
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_dir = root
                    break
            
            if not os.path.exists(stem_dir):
                raise ValueError(f"Stem directory not found: {stem_dir}")
        
        logging.info(f"Found stems: {stem_dir}")
        
        # Create ZIP
        zip_filename = f"{unique_id}_stems.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
        
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
        
        logging.info(f"📦 ZIP created: {zip_path}")
        
        # Create publicly accessible URL for the ZIP file
        public_zip_url = f"{service_url}/download_temp_stems/{zip_filename}"
        logging.info(f"📦 Public ZIP URL: {public_zip_url}")
        
        # Call webhook with the public download URL
        logging.info(f"📞 Calling webhook: {webhook_url}")
        webhook_response = requests.post(
            webhook_url,
            json={
                "project_id": project_id,
                "stems_zip_url": public_zip_url,  # FIXED: Now a public HTTP URL
                "service_role_key": service_role_key
            },
            timeout=60
        )
        
        if webhook_response.status_code == 200:
            logging.info(f"✅ Webhook called successfully")
        else:
            logging.error(f"❌ Webhook failed: {webhook_response.status_code} - {webhook_response.text}")
        
        # Cleanup (except ZIP - it will be cleaned up after download)
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if wav_path and os.path.exists(wav_path) and wav_path != input_path:
                os.remove(wav_path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
        
    except Exception as e:
        logging.error(f"❌ Async processing error: {e}", exc_info=True)
        
        # Try to notify webhook of failure
        try:
            requests.post(
                webhook_url,
                json={
                    "project_id": project_id,
                    "error": str(e),
                    "service_role_key": service_role_key
                },
                timeout=10
            )
        except:
            pass

# ORIGINAL SYNCHRONOUS ENDPOINT (keep for backup)
@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """Separates audio into stems using demucs (SYNCHRONOUS)."""
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
        
        stem_dir = os.path.join(output_dir, 'htdemucs', os.path.splitext(os.path.basename(wav_path))[0])
        
        if not os.path.exists(stem_dir):
            for root, dirs, files in os.walk(output_dir):
                if any(f.endswith('.mp3') for f in files):
                    stem_dir = root
                    break
            
            if not os.path.exists(stem_dir):
                raise ValueError(f"Stem directory not found: {stem_dir}")
        
        logging.info(f"Found stems: {stem_dir}")
        
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
        return jsonify({"error": "Processing timeout"}), 500
    
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
            "POST /separate_stems_async": "Separate audio into stems (async with webhook)",
            "POST /separate_stems": "Separate audio into stems (sync)",
            "GET /download_temp_stems/<filename>": "Download temporary stems ZIP",
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







