import re
import subprocess
import tempfile
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import save_audio
import torchaudio
import io
import base64

app = Flask(__name__)
CORS(app)

def parse_ai_recommendations(recommendations):
    """Parse AI recommendations and build FFmpeg filter chain with AGGRESSIVE, AUDIBLE processing"""
    filters = []
    
    for rec in recommendations.get('recommendations', []):
        action = rec.get('action', '').lower()
        details = rec.get('details', '').lower()
        
        # EQ PROCESSING - Make it VERY audible
        if 'eq' in action:
            # High-pass filter to remove rumble
            if 'high-pass' in details or 'rumble' in details or 'low-end' in details:
                freq_match = re.search(r'(\d+)\s*hz', details, re.IGNORECASE)
                if freq_match:
                    freq = int(freq_match.group(1))
                    filters.append(f'highpass=f={freq}:poles=2')
            
            # Find all frequency boosts/cuts
            freq_patterns = re.finditer(r'([+-]?\d+(?:\.\d+)?)\s*db.*?(\d+(?:\.\d+)?)\s*(hz|khz)', details, re.IGNORECASE)
            for match in freq_patterns:
                gain = float(match.group(1))
                freq = float(match.group(2))
                unit = match.group(3).lower()
                
                if unit == 'khz':
                    freq *= 1000
                
                # Make the EQ changes MORE dramatic (multiply by 1.5)
                gain = gain * 1.5
                filters.append(f'equalizer=f={freq}:t=q:w=2:g={gain}')
            
            # Also look for frequency ranges mentioned without explicit dB values
            if 'boost' in details or 'add' in details:
                freq_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(hz|khz)', details, re.IGNORECASE)
                for match in freq_matches:
                    freq = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit == 'khz':
                        freq *= 1000
                    # Apply a noticeable +6dB boost
                    filters.append(f'equalizer=f={freq}:t=q:w=2:g=6')
            
            if 'cut' in details or 'reduce' in details or 'muddy' in details or 'mud' in details:
                freq_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(hz|khz)', details, re.IGNORECASE)
                for match in freq_matches:
                    freq = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit == 'khz':
                        freq *= 1000
                    # Apply a noticeable -6dB cut
                    filters.append(f'equalizer=f={freq}:t=q:w=2:g=-6')
        
        # COMPRESSION - Make it punch harder
        if 'compress' in action or 'compression' in action:
            # Extract compression parameters
            ratio = 4.0  # Default aggressive ratio
            threshold = -18  # Default
            attack = 5
            release = 50
            makeup = 8  # Add makeup gain for punch
            
            ratio_match = re.search(r'(\d+(?:\.\d+)?)\s*:\s*1', details)
            if ratio_match:
                ratio = float(ratio_match.group(1))
            
            threshold_match = re.search(r'threshold.*?(-?\d+(?:\.\d+)?)\s*db', details, re.IGNORECASE)
            if threshold_match:
                threshold = float(threshold_match.group(1))
            
            attack_match = re.search(r'attack.*?(\d+(?:\.\d+)?)\s*ms', details, re.IGNORECASE)
            if attack_match:
                attack = float(attack_match.group(1))
            
            release_match = re.search(r'release.*?(\d+(?:\.\d+)?)\s*ms', details, re.IGNORECASE)
            if release_match:
                release = float(release_match.group(1))
            
            filters.append(f'acompressor=threshold={threshold}dB:ratio={ratio}:attack={attack}:release={release}:makeup={makeup}dB')
        
        # REVERB - Add noticeable space
        if 'reverb' in action or 'reverb' in details:
            # Stronger reverb effect
            if 'large' in details or 'hall' in details:
                filters.append('aecho=0.8:0.9:1000|1800:0.4|0.3')
            elif 'small' in details or 'short' in details:
                filters.append('aecho=0.8:0.88:400|600:0.3|0.25')
            else:
                filters.append('aecho=0.8:0.9:800|1300:0.35|0.28')
        
        # DELAY - Add depth
        if 'delay' in action or 'delay' in details:
            delay_time = 100
            delay_match = re.search(r'(\d+)\s*ms', details)
            if delay_match:
                delay_time = int(delay_match.group(1))
            
            filters.append(f'aecho=0.8:0.9:{delay_time}|{delay_time*1.5}:0.4|0.3')
        
        # SATURATION / WARMTH
        if 'saturat' in details or 'warm' in details or 'harmonic' in details:
            # Add subtle saturation for warmth
            filters.append('volume=1.5,acompressor=threshold=-12dB:ratio=3:attack=1:release=50,volume=0.8')
    
    # If no filters were created, apply a DEFAULT DRAMATIC ENHANCEMENT
    if not filters:
        print("⚠️ No specific filters created, applying default dramatic enhancement")
        filters = [
            'equalizer=f=100:t=q:w=2:g=-4',  # Cut mud
            'equalizer=f=3000:t=q:w=2:g=5',  # Boost presence
            'equalizer=f=10000:t=q:w=2:g=4',  # Add air
            'acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=6dB',  # Punch
        ]
    
    return filters

@app.route('/apply_ffmpeg_recommendations', methods=['POST'])
def apply_ffmpeg_recommendations():
    try:
        data = request.json
        stem_url = data.get('stem_url')
        stem_name = data.get('stem_name')
        recommendations = data.get('recommendations')
        
        if not stem_url or not stem_name or not recommendations:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        print(f"\n{'='*60}")
        print(f"🎛️ APPLYING AI RECOMMENDATIONS TO {stem_name.upper()}")
        print(f"{'='*60}")
        print(f"Recommendations received: {recommendations}")
        
        # Download the stem
        print(f"📥 Downloading stem from: {stem_url[:80]}...")
        response = requests.get(stem_url, timeout=60)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download stem: {response.status_code}'}), 400
        
        # Create temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as input_file:
            input_path = input_file.name
            input_file.write(response.content)
        
        output_path = tempfile.mktemp(suffix='.mp3')
        
        # Parse recommendations and build filter chain
        filters = parse_ai_recommendations(recommendations)
        filter_chain = ','.join(filters)
        
        print(f"\n🎚️ FFMPEG FILTER CHAIN:")
        print(f"{filter_chain}")
        print(f"\n{'='*60}\n")
        
        # Apply FFmpeg processing
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-af', filter_chain,
            '-b:a', '320k',  # High quality output
            '-y',
            output_path
        ]
        
        print(f"🎵 Executing FFmpeg command...")
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error:\n{result.stderr}")
            return jsonify({'error': f'FFmpeg processing failed: {result.stderr[:500]}'}), 500
        
        print(f"✅ FFmpeg processing complete!")
        
        # Read the enhanced file
        with open(output_path, 'rb') as f:
            enhanced_audio = f.read()
        
        # Convert to base64 for upload
        enhanced_base64 = base64.b64encode(enhanced_audio).decode('utf-8')
        
        # Clean up temp files
        os.unlink(input_path)
        os.unlink(output_path)
        
        print(f"🎉 ENHANCEMENT COMPLETE - Sending back enhanced audio\n{'='*60}\n")
        
        # Return as base64 so frontend can create object URL
        return jsonify({
            'success': True,
            'enhanced_audio_base64': enhanced_base64,
            'filters_applied': filters
        })
        
    except Exception as e:
        print(f"❌ Error in apply_ffmpeg_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ... keep existing /separate_stems endpoint ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

















