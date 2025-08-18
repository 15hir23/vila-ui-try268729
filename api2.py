import cv2
from collections import defaultdict
import numpy as np
import time
import base64
import requests
import json
from io import BytesIO
from PIL import Image
import ssl
import urllib3
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

# Disable SSL warnings and configure SSL context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"
VILA_API_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"

print("VILA model will be used via NVIDIA API for video analysis and summarization")

# ---- Helper Functions ----
def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for API"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize for API efficiency
        pil_image = pil_image.resize((512, 384))
        
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

def make_vila_request(payload):
    """Make request to VILA API with error handling"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        session = requests.Session()
        session.verify = False
        
        response = session.post(VILA_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"VILA API Error: {response.status_code} - {response.text}")
            return f"API Error ({response.status_code}): Could not analyze video with VILA"
            
    except requests.exceptions.SSLError as ssl_err:
        print(f"SSL Error with VILA API: {ssl_err}")
        return "SSL Error: Could not connect to VILA API"
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error with VILA API: {req_err}")
        return "Network Error: Could not reach VILA API"
    except Exception as e:
        print(f"Error in VILA request: {e}")
        return f"Request Error: {str(e)}"

def analyze_video_with_vila(key_frames, video_duration):
    """Use VILA to analyze and summarize the entire video"""
    try:
        if len(key_frames) < 3:
            return "Insufficient frames for analysis"
        
        # Encode key frames to base64
        encoded_frames = []
        for frame in key_frames:
            encoded_frame = encode_frame_to_base64(frame)
            if encoded_frame:
                encoded_frames.append(encoded_frame)
        
        if not encoded_frames:
            return "Error: Could not encode frames for analysis"
        
        # Simple prompt without pre-entered content
        # Example of improved prompt
        prompt = """Analyze these video frames sequentially and:
        1. Identify any abnormal events or objects
        2. Note any sudden changes in the scene
        3. Report potential safety or security issues
        4. For each anomaly, estimate its severity (low/medium/high)

        Provide your findings in this format:
        - Time estimate: [rough timestamp]
        - Anomaly type: [description]
        - Severity: [level]
        - Details: [explanation]"""

        # Prepare the request payload
        payload = {
            "model": "nvidia/vila",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {"url": frame}
                        } for frame in encoded_frames
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": False
        }
        
        print("Sending frames to VILA for comprehensive video analysis...")
        return make_vila_request(payload)
        
    except Exception as e:
        print(f"Error in VILA video analysis: {e}")
        return f"Analysis Error: {str(e)}"

def detect_anomalies_with_vila(key_frames, video_duration):
    """Use VILA to detect anomalies and unusual events in the video"""
    try:
        if len(key_frames) < 3:
            return "Insufficient frames for anomaly detection"
        
        # Encode key frames to base64
        encoded_frames = []
        for frame in key_frames:
            encoded_frame = encode_frame_to_base64(frame)
            if encoded_frame:
                encoded_frames.append(encoded_frame)
        
        if not encoded_frames:
            return "Error: Could not encode frames for anomaly detection"
        
        # Simple prompt for anomaly detection
        prompt = f"Look at this video sequence and identify any unusual events, accidents, or anomalies."

        # Prepare the request payload
        payload = {
            "model": "nvidia/vila",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {"url": frame}
                        } for frame in encoded_frames
                    ]
                }
            ],
            "max_tokens": 600,
            "temperature": 0.2,
            "stream": False
        }
        
        print("Analyzing frames for anomalies with VILA...")
        return make_vila_request(payload)
        
    except Exception as e:
        print(f"Error in VILA anomaly detection: {e}")
        return f"Anomaly Detection Error: {str(e)}"

def extract_key_frames(cap, total_frames, num_frames=12):
    """Extract key frames evenly distributed throughout the video"""
    key_frames = []
    
    try:
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                key_frames.append(frame)
    except Exception as e:
        print(f"Error extracting frames: {e}")
    
    return key_frames

def process_video_file(video_file, analysis_type="general"):
    """Process uploaded video file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            temp_path = temp_file.name
        
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            os.unlink(temp_path)
            return {"error": "Could not open video file"}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(3)) or 640
        h = int(cap.get(4)) or 480
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

        print(f"Video info: {duration:.1f}s, {total_frames} frames, {fps} FPS, {w}x{h}")

        # Extract key frames
        num_frames = 20 if analysis_type == "anomaly" else 15
        key_frames = extract_key_frames(cap, total_frames, num_frames=num_frames)
        print(f"Extracted {len(key_frames)} key frames")

        cap.release()
        os.unlink(temp_path)  # Clean up temp file

        if not key_frames:
            return {"error": "Could not extract frames from video"}

        # Analyze based on type
        if analysis_type == "anomaly":
            vila_result = detect_anomalies_with_vila(key_frames, duration)
            
            report = f"🚨 ANOMALY DETECTION SUMMARY\n"
            report += "=" * 50 + "\n\n"
            report += f"📊 Video Details:\n"
            report += f"• Duration: {duration:.2f} seconds ({total_frames} frames)\n"
            report += f"• Resolution: {w}x{h} @ {fps} FPS\n"
            report += f"• Frames Analyzed: {len(key_frames)} key frames\n\n"
            report += f"🤖 VILA Analysis:\n"
            report += "-" * 30 + "\n"
            report += vila_result
        else:
            vila_result = analyze_video_with_vila(key_frames, duration)
            
            report = f"🎥 VIDEO ANALYSIS REPORT\n"
            report += "=" * 50 + "\n\n"
            report += f"📊 Technical Details:\n"
            report += f"• Duration: {duration:.2f} seconds\n"
            report += f"• Total Frames: {total_frames}\n"
            report += f"• Frame Rate: {fps} FPS\n"
            report += f"• Resolution: {w}x{h}\n\n"
            report += f"🤖 VILA AI Analysis:\n"
            report += "-" * 30 + "\n"
            report += vila_result

        return {"success": True, "report": report, "duration": duration}
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return {"error": f"Error processing video: {str(e)}"}

# ---- Flask Routes ----
@app.route('/')
def home():
    return "VILA Video Analyzer API is running!"

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """General video analysis endpoint"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    result = process_video_file(video_file, "general")
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/detect-anomalies', methods=['POST'])
def detect_anomalies():
    """Anomaly detection endpoint"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    result = process_video_file(video_file, "anomaly")
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "VILA Video Analyzer API is running"})

if __name__ == "__main__":
    print("Starting VILA Video Analyzer API...")
    print("Access the API at: http://localhost:5000")
    print("Available endpoints:")
    print("  POST /analyze - General video analysis")
    print("  POST /detect-anomalies - Anomaly detection")
    print("  GET /health - Health check")
    
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )