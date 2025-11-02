# Standard library imports
import os
import re
import base64
import tempfile
import subprocess

# Third-party imports
import torch
import torchaudio
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from demucs.apply import apply_model
from demucs.pretrained import get_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load Demucs model at startup
print("Loading Demucs model...")
model = get_model('htdemucs_ft')
print("Demucs model loaded successfully!")


@app.route('/separate_stems', methods=['POST'])
def separate_stems():
    """
    Existing endpoint for Demucs stem separation
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, 'input_audio.mp3')
            file.save(input_path)
            
            print(f"Processing file: {file.filename}")
            
            # Load audio
            wav, sr = torchaudio.load(input_path)
            
            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            
            # Resample if needed
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                wav = resampler(wav)
            
            # Apply model
            print("Separating stems with Demucs...")
            with torch.no_grad():
                sources = apply_model(model, wav[None], device='cpu')[0]
            
            # Get stem names
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            # Convert each stem to base64
            stems = {}
            for i, stem_name in enumerate(stem_names):
                stem_audio = sources[i]
                
                # Save to temporary file
                stem_path = os.path.join(temp_dir, f'{stem_name}.mp3')
                torchaudio.save(
                    stem_path,
                    stem_audio.cpu(),
                    model.samplerate,
                    format='mp3'
                )
                
                # Read and encode to base64
                with open(stem_path, 'rb') as f:
                    stem_base64 = base64.b64encode(f.read()).decode('utf-8')
                    stems[stem_name] = stem_base64
            
            print("Stem separation complete!")
            return jsonify(stems), 200
            
    except Exception as e:
        print(f"Error during separation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/apply_ffmpeg_recommendations', methods=['POST'])
def apply_ffmpeg_recommendations():
    """
    New endpoint for applying FFmpeg-based audio enhancements
    based on AI recommendations
    """
    try:
        data = request.get_json()
        
        stem_url = data.get('stem_url')
        stem_name = data.get('stem_name')
        recommendations = data.get('recommendations')
        
        if not stem_url or not stem_name or not recommendations:
            return jsonify({'error': 'Missing required fields: stem_url, stem_name, recommendations'}), 400
        
        print(f"Processing FFmpeg enhancement for {stem_name}")
        print(f"AI Recommendations: {recommendations}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the original stem
            input_path = os.path.join(temp_dir, f'input_{stem_name}.mp3')
            output_path = os.path.join(temp_dir, f'enhanced_{stem_name}.mp3')
            
            print(f"Downloading stem from {stem_url}")
            response = requests.get(stem_url)
            if response.status_code != 200:
                return jsonify({'error': f'Failed to download stem: {response.status_code}'}), 400
            
            with open(input_path, 'wb') as f:
                f.write(response.content)
            
            # Parse AI recommendations and build FFmpeg filters
            filters = parse_ai_recommendations(recommendations)
            
            if not filters:
                # If no filters were generated, apply a basic pass-through
                filters = ['anull']  # No-op filter
            
            print(f"FFmpeg filters to apply: {filters}")
            
            # Build FFmpeg command
            filter_chain = ','.join(filters)
            
            ffmpeg_command = [
                'ffmpeg',
                '-i', input_path,
                '-af', filter_chain,
                '-y',  # Overwrite output
                output_path
            ]
            
            print(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")
            
            # Execute FFmpeg
            result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return jsonify({'error': f'FFmpeg processing failed: {result.stderr[:200]}'}), 500
            
            print("FFmpeg processing complete!")
            
            # Read the enhanced file and encode to base64
            with open(output_path, 'rb') as f:
                enhanced_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'enhanced_audio': enhanced_base64,
                'filters_applied': filters
            }), 200
            
    except Exception as e:
        print(f"Error during FFmpeg enhancement: {str(e)}")
        return jsonify({'error': str(e)}), 500


def parse_ai_recommendations(recommendations):
    """
    Parse AI recommendations JSON and convert to FFmpeg filter strings
    """
    filters = []
    
    recs = recommendations.get('recommendations', [])
    
    for rec in recs:
        action = rec.get('action', '').lower()
        details = rec.get('details', '').lower()
        
        # EQ Processing
        if 'eq' in action:
            filters.extend(parse_eq_recommendation(details))
        
        # Compression
        if 'compression' in action or 'compress' in action:
            filters.append(parse_compression_recommendation(details))
        
        # Reverb
        if 'reverb' in action:
            filters.append(parse_reverb_recommendation(details))
        
        # Delay
        if 'delay' in action:
            filters.append(parse_delay_recommendation(details))
    
    return filters


def parse_eq_recommendation(details):
    """
    Parse EQ recommendations and return FFmpeg equalizer filters
    """
    filters = []
    
    # High-pass filter
    if 'high-pass' in details or 'highpass' in details or 'rumble' in details:
        hpf_match = re.search(r'(\d+)\s*hz', details, re.IGNORECASE)
        if hpf_match:
            freq = hpf_match.group(1)
            filters.append(f'highpass=f={freq}')
    
    # Low-pass filter
    if 'low-pass' in details or 'lowpass' in details or 'fizz' in details or 'harsh' in details:
        lpf_match = re.search(r'(\d+)\s*hz', details, re.IGNORECASE)
        if lpf_match:
            freq = lpf_match.group(1)
            filters.append(f'lowpass=f={freq}')
    
    # Parametric EQ (boost/cut at specific frequencies)
    freq_pattern = r'(\d+(?:\.\d+)?)\s*(hz|khz)'
    db_pattern = r'([+-]?\d+(?:\.\d+)?)\s*db'
    
    freq_matches = list(re.finditer(freq_pattern, details, re.IGNORECASE))
    db_match = re.search(db_pattern, details, re.IGNORECASE)
    
    for freq_match in freq_matches:
        freq = float(freq_match.group(1))
        unit = freq_match.group(2).lower()
        
        if unit == 'khz':
            freq *= 1000
        
        # Determine gain
        gain = 0
        if db_match:
            gain = float(db_match.group(1))
        elif 'cut' in details or 'reduce' in details:
            gain = -3
        elif 'boost' in details or 'add' in details or 'enhance' in details:
            gain = 3
        
        if gain != 0:
            filters.append(f'equalizer=f={int(freq)}:t=q:w=1:g={gain}')
    
    return filters


def parse_compression_recommendation(details):
    """
    Parse compression recommendations and return FFmpeg compressor filter
    """
    threshold = -20
    ratio = 3
    attack = 5
    release = 100
    
    # Parse ratio
    ratio_match = re.search(r'ratio.*?(\d+(?:\.\d+)?)\s*:\s*1', details, re.IGNORECASE)
    if ratio_match:
        ratio = float(ratio_match.group(1))
    
    # Parse threshold
    threshold_match = re.search(r'threshold.*?(-?\d+(?:\.\d+)?)\s*db', details, re.IGNORECASE)
    if threshold_match:
        threshold = float(threshold_match.group(1))
    
    # Parse attack
    attack_match = re.search(r'attack.*?(\d+(?:\.\d+)?)\s*ms', details, re.IGNORECASE)
    if attack_match:
        attack = float(attack_match.group(1))
    
    # Parse release
    release_match = re.search(r'release.*?(\d+(?:\.\d+)?)\s*ms', details, re.IGNORECASE)
    if release_match:
        release = float(release_match.group(1))
    
    return f'acompressor=threshold={threshold}dB:ratio={ratio}:attack={attack}:release={release}'


def parse_reverb_recommendation(details):
    """
    Parse reverb recommendations and return FFmpeg reverb filter
    """
    room_size = 50
    
    if 'short' in details or 'small' in details or 'tight' in details:
        room_size = 30
    elif 'large' in details or 'hall' in details or 'cathedral' in details:
        room_size = 70
    elif 'medium' in details:
        room_size = 50
    
    # Using aecho as a simple reverb simulation
    return f'aecho=0.8:0.9:{room_size}:0.3'


def parse_delay_recommendation(details):
    """
    Parse delay recommendations and return FFmpeg delay filter
    """
    delay_time = 50
    
    if 'slapback' in details:
        delay_time = 75
    else:
        delay_match = re.search(r'(\d+)\s*ms', details, re.IGNORECASE)
        if delay_match:
            delay_time = int(delay_match.group(1))
    
    return f'aecho=0.8:0.88:{delay_time}:0.4'


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'ffmpeg_available': check_ffmpeg()}), 200


def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return result.returncode == 0
    except:
        return False


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    print(f"FFmpeg available: {check_ffmpeg()}")
    app.run(host='0.0.0.0', port=port, debug=False)


















