import cv2
import gradio as gr
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
import threading
from datetime import datetime
import queue

# Disable SSL warnings and configure SSL context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"
VILA_API_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"

print("VILA model will be used via NVIDIA API for video analysis and summarization")

# Global variables for live tracking
live_tracking_active = False
live_cap = None
live_frame_buffer = []
live_analysis_queue = queue.Queue()
live_anomaly_queue = queue.Queue()
last_analysis_time = 0
frame_accumulator = []
current_live_frame = None
live_reports_content = ""

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
        
        # Create comprehensive prompt for video analysis
        prompt = f"""Analyze this sequence of {len(encoded_frames)} frames from a {video_duration:.1f}-second video. Provide a detailed summary of:

1. What activities and actions are happening in the video?
2. How many people are visible and what are they doing?
3. What is the setting/environment?
4. Are there any notable movements, interactions, or changes over time?
5. Overall description of the scene and story.

Please write a natural, flowing description as if you're describing the video to someone who can't see it."""

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
        
        # Create specific prompt for anomaly detection - focused on summary
        prompt = f"""Analyze this sequence of {len(encoded_frames)} frames from a {video_duration:.1f}-second video and provide a SUMMARY of anomalies detected.

🚨 DETECT THESE ANOMALIES:
- Objects falling (boxes, items, equipment)
- People falling, tripping, or stumbling  
- Equipment malfunctions or failures
- Spills, breaks, or structural damage
- Collisions or impacts
- Unusual behavior or safety incidents
- Loitering in sensitive zones
- Theft/shoplifting or concealment behavior
- Unattended objects or packages
- Crowd formations in restricted areas
- Violence, fights, or physical altercations
- Intrusion during non-operational hours
- Suspicious or erratic movement patterns
- Vandalism or property damage
- Camera blocking or tampering
- Vehicle moving in wrong direction
- Abandoned vehicles in unusual locations
- Unusual speed (too fast movement)
- Queue jumping or overcrowding
- Missing protective gear (helmets, vests)
- Unauthorized carrying of weapons/packages
- Trespassing or fence climbing
- Any disruption to normal operations

Provide a CONCISE SUMMARY (not timestamps) describing:
1. What types of anomalies were observed
2. Brief description of each incident
3. Overall assessment of the scene

If NO anomalies detected, state: "No significant anomalies detected - normal activity observed."

Keep response brief and focused on WHAT happened, not WHEN it happened."""

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

def create_output_video(cap, output_path, fps, w, h, video_type="analysis"):
    """Create output video with overlays"""
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps if fps > 0 else 0

            if video_type == "anomaly":
                cv2.putText(frame, f"ANOMALY SCAN - Time: {timestamp:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "🚨 Scanning for anomalies...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"Time: {timestamp:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            
            # Progress indicator
            if frame_count % (fps * 2) == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Video processing progress: {progress:.1f}%")

        out.release()
        return True
        
    except Exception as e:
        print(f"Error creating output video: {e}")
        return False

# ---- Live Video Functions (FIXED) ----
def live_frame_capture_worker():
    """Background worker to continuously capture frames and update display"""
    global current_live_frame, frame_accumulator, live_cap
    
    while live_tracking_active:
        try:
            if live_cap is not None:
                ret, frame = live_cap.read()
                if ret and frame is not None:
                    # Add frame to accumulator for analysis
                    frame_accumulator.append(frame.copy())
                    
                    # Keep accumulator manageable (last 20 seconds at 30fps = 600 frames)
                    if len(frame_accumulator) > 600:
                        frame_accumulator = frame_accumulator[-600:]
                    
                    # Add overlay to current frame
                    display_frame = frame.copy()
                    elapsed_time = time.time() - last_analysis_time if last_analysis_time > 0 else 0
                    
                    cv2.putText(display_frame, f"🔴 LIVE - {datetime.now().strftime('%H:%M:%S')}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Frames: {len(frame_accumulator)} | Next: {20 - (elapsed_time % 20):.0f}s", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Analysis every 20s | Anomaly check every 5s", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Convert BGR to RGB for display
                    current_live_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in frame capture worker: {e}")
            time.sleep(1)

def live_analysis_worker():
    """Background worker for periodic live video analysis (every 20 seconds)"""
    global frame_accumulator, last_analysis_time, live_analysis_queue, live_reports_content
    
    while live_tracking_active:
        try:
            current_time = time.time()
            
            # Check if 20 seconds have passed since last analysis
            if current_time - last_analysis_time >= 20 and len(frame_accumulator) >= 5:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting 20-second analysis...")
                
                # Select frames from accumulator (take every nth frame to get ~10 frames)
                step = max(1, len(frame_accumulator) // 10)
                analysis_frames = frame_accumulator[::step][:10]
                
                if analysis_frames:
                    # Perform analysis
                    analysis_result = analyze_video_with_vila(analysis_frames, 20.0)
                    
                    # Create timestamped report
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    report = f"📹 LIVE ANALYSIS [{timestamp}]\n"
                    report += "=" * 40 + "\n"
                    report += f"Frames analyzed: {len(analysis_frames)}\n"
                    report += f"Time period: 20 seconds\n\n"
                    report += analysis_result + "\n\n"
                    
                    # Update global reports content
                    live_reports_content = report + live_reports_content
                    
                    # Send to queue for UI update
                    live_analysis_queue.put(("analysis", report))
                
                # Reset timer (keep accumulator for next analysis)
                last_analysis_time = current_time
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Error in live analysis worker: {e}")
            time.sleep(5)

def live_anomaly_worker():
    """Background worker for real-time anomaly detection"""
    global frame_accumulator, live_anomaly_queue, live_reports_content
    
    last_anomaly_check = time.time()
    
    while live_tracking_active:
        try:
            current_time = time.time()
            
            # Check for anomalies every 5 seconds with recent frames
            if current_time - last_anomaly_check >= 5 and len(frame_accumulator) >= 3:
                # Take last 5 frames for anomaly detection
                recent_frames = frame_accumulator[-5:] if len(frame_accumulator) >= 5 else frame_accumulator
                
                if len(recent_frames) >= 3:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking for anomalies...")
                    
                    # Detect anomalies
                    anomaly_result = detect_anomalies_with_vila(recent_frames, 5.0)
                    
                    # Only report if anomalies detected (not "normal activity")
                    if "No significant anomalies detected" not in anomaly_result and "normal activity observed" not in anomaly_result.lower():
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        alert = f"🚨 ANOMALY ALERT [{timestamp}]\n"
                        alert += "=" * 40 + "\n"
                        alert += anomaly_result + "\n\n"
                        
                        # Update global reports content
                        live_reports_content = alert + live_reports_content
                        
                        # Send to queue for UI update
                        live_anomaly_queue.put(("anomaly", alert))
                
                last_anomaly_check = current_time
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Error in live anomaly worker: {e}")
            time.sleep(5)

def start_live_tracking():
    """Start live video tracking with camera"""
    global live_tracking_active, live_cap, live_frame_buffer, last_analysis_time, frame_accumulator, current_live_frame, live_reports_content
    
    try:
        # Reset reports
        live_reports_content = ""
        
        # Try different camera indices
        camera_indices = [0, 1, 2]
        live_cap = None
        
        for idx in camera_indices:
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    live_cap = test_cap
                    print(f"Successfully opened camera index {idx}")
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if live_cap is None:
            return None, "❌ Error: Could not access any camera. Please check camera permissions."
        
        # Configure camera
        live_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        live_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        live_cap.set(cv2.CAP_PROP_FPS, 30)
        
        live_tracking_active = True
        last_analysis_time = time.time()
        frame_accumulator = []
        current_live_frame = None
        
        # Start background workers
        capture_thread = threading.Thread(target=live_frame_capture_worker, daemon=True)
        analysis_thread = threading.Thread(target=live_analysis_worker, daemon=True)
        anomaly_thread = threading.Thread(target=live_anomaly_worker, daemon=True)
        
        capture_thread.start()
        analysis_thread.start()
        anomaly_thread.start()
        
        # Give threads time to start
        time.sleep(1)
        
        status_msg = "✅ Live tracking started! Camera activated.\n\n"
        status_msg += "📊 Status: Collecting frames...\n"
        status_msg += "🔄 Next analysis in 20 seconds\n"
        status_msg += "🚨 Anomaly detection: Active (every 5s)\n\n"
        status_msg += "📋 Analysis reports will appear in the reports section as they are generated."
        
        return get_current_live_frame(), status_msg
        
    except Exception as e:
        error_msg = f"❌ Error starting live tracking: {str(e)}"
        print(error_msg)
        return None, error_msg

def stop_live_tracking():
    """Stop live video tracking"""
    global live_tracking_active, live_cap, current_live_frame
    
    live_tracking_active = False
    
    if live_cap is not None:
        live_cap.release()
        live_cap = None
    
    current_live_frame = None
    
    return None, "🛑 Live tracking stopped. Camera released."

def get_current_live_frame():
    """Get current frame from live camera for display"""
    global current_live_frame
    
    if not live_tracking_active or current_live_frame is None:
        return None
    
    return current_live_frame

def get_live_updates():
    """Get live analysis and anomaly updates"""
    global live_reports_content
    
    # Get new updates from queues
    new_updates = []
    
    # Get analysis updates
    while not live_analysis_queue.empty():
        try:
            update_type, content = live_analysis_queue.get_nowait()
            new_updates.append(content)
        except queue.Empty:
            break
    
    # Get anomaly updates
    while not live_anomaly_queue.empty():
        try:
            update_type, content = live_anomaly_queue.get_nowait()
            new_updates.append(content)
        except queue.Empty:
            break
    
    # If there are new updates, add them to the global content
    if new_updates:
        for update in new_updates:
            live_reports_content = update + live_reports_content
    
    # Return the current reports content
    if live_reports_content:
        return live_reports_content
    else:
        return "📋 Live analysis reports will appear here...\n\n🔄 Automatic reports every 20 seconds\n🚨 Anomaly alerts as they happen\n📊 Manual analysis reports on demand"

def process_live_video_analysis():
    """Process analysis for live video (called when button pressed during live tracking)"""
    global frame_accumulator, live_reports_content
    
    if not live_tracking_active:
        return "❌ Live tracking is not active. Please start live tracking first."
    
    if len(frame_accumulator) < 5:
        return "⏳ Not enough frames collected yet. Please wait a few more seconds."
    
    try:
        # Take recent frames for immediate analysis
        recent_frames = frame_accumulator[-10:] if len(frame_accumulator) >= 10 else frame_accumulator
        
        print(f"Analyzing {len(recent_frames)} recent frames from live video...")
        analysis_result = analyze_video_with_vila(recent_frames, len(recent_frames) / 30.0)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        report = f"📹 INSTANT LIVE ANALYSIS [{timestamp}]\n"
        report += "=" * 45 + "\n"
        report += f"Frames analyzed: {len(recent_frames)}\n"
        report += f"Camera: Live camera feed\n\n"
        report += "🤖 VILA Analysis:\n"
        report += "-" * 20 + "\n"
        report += analysis_result + "\n\n"
        
        # Add to global reports
        live_reports_content = report + live_reports_content
        
        return live_reports_content
        
    except Exception as e:
        error_report = f"❌ Error analyzing live video: {str(e)}\n\n"
        live_reports_content = error_report + live_reports_content
        return live_reports_content

def process_live_anomaly_detection():
    """Process anomaly detection for live video (called when button pressed during live tracking)"""
    global frame_accumulator, live_reports_content
    
    if not live_tracking_active:
        return "❌ Live tracking is not active. Please start live tracking first."
    
    if len(frame_accumulator) < 5:
        return "⏳ Not enough frames collected yet. Please wait a few more seconds."
    
    try:
        # Take recent frames for immediate anomaly detection
        recent_frames = frame_accumulator[-15:] if len(frame_accumulator) >= 15 else frame_accumulator
        
        print(f"Checking {len(recent_frames)} recent frames for anomalies...")
        anomaly_result = detect_anomalies_with_vila(recent_frames, len(recent_frames) / 30.0)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        report = f"🚨 INSTANT ANOMALY CHECK [{timestamp}]\n"
        report += "=" * 45 + "\n"
        report += f"Frames analyzed: {len(recent_frames)}\n"
        report += f"Camera: Live camera feed\n\n"
        report += "🔍 Anomaly Detection Results:\n"
        report += "-" * 30 + "\n"
        report += anomaly_result + "\n\n"
        
        # Add to global reports
        live_reports_content = report + live_reports_content
        
        return live_reports_content
        
    except Exception as e:
        error_report = f"❌ Error detecting anomalies in live video: {str(e)}\n\n"
        live_reports_content = error_report + live_reports_content
        return live_reports_content

# ---- Video Processing Functions (Original) ----
def process_video(input_path):
    """Original video processing function for general analysis"""
    if input_path is None:
        return None, "Please upload a video file"
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(3)) or 640
        h = int(cap.get(4)) or 480
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

        print(f"Video info: {duration:.1f}s, {total_frames} frames, {fps} FPS, {w}x{h}")

        # Extract key frames for VILA analysis
        print("Extracting key frames for VILA analysis...")
        key_frames = extract_key_frames(cap, total_frames, num_frames=15)
        print(f"Extracted {len(key_frames)} key frames")

        if not key_frames:
            cap.release()
            return None, "Error: Could not extract frames from video"

        # Create output video
        output_path = "output.mp4"
        start_time = time.time()
        
        success = create_output_video(cap, output_path, fps, w, h, "analysis")
        cap.release()
        
        if not success:
            return None, "Error: Could not create output video"

        # Analyze video with VILA
        print("Analyzing video content with VILA...")
        vila_summary = analyze_video_with_vila(key_frames, duration)

        # Build Complete Summary
        summary = "🎥 VIDEO ANALYSIS REPORT\n"
        summary += "=" * 50 + "\n\n"
        summary += f"📊 Technical Details:\n"
        summary += f"• Duration: {duration:.2f} seconds\n"
        summary += f"• Total Frames: {total_frames}\n"
        summary += f"• Frame Rate: {fps} FPS\n"
        summary += f"• Resolution: {w}x{h}\n"
        summary += f"• Processing Time: {time.time() - start_time:.1f} seconds\n\n"
        
        summary += f"🤖 VILA AI Analysis:\n"
        summary += "-" * 30 + "\n"
        summary += vila_summary + "\n\n"
        
        summary += f"📝 Analysis Method:\n"
        summary += f"• Analyzed {len(key_frames)} key frames using VILA\n"
        summary += f"• AI Model: NVIDIA VILA (Vision-Language Assistant)\n"
        summary += f"• Frame sampling: Evenly distributed across video duration\n"

        print("Video analysis complete!")
        return output_path, summary
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(error_msg)
        return None, error_msg

def detect_video_anomalies(input_path):
    """Function specifically for anomaly detection"""
    if input_path is None:
        return None, "Please upload a video file for anomaly detection"
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(3)) or 640
        h = int(cap.get(4)) or 480
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0

        print(f"Anomaly Detection - Video info: {duration:.1f}s, {total_frames} frames, {fps} FPS, {w}x{h}")

        # Extract more key frames for better anomaly detection
        print("Extracting key frames for anomaly detection...")
        key_frames = extract_key_frames(cap, total_frames, num_frames=20)
        print(f"Extracted {len(key_frames)} key frames for anomaly analysis")

        if not key_frames:
            cap.release()
            return None, "Error: Could not extract frames from video"

        # Create output video with anomaly detection overlay
        output_path = "anomaly_output.mp4"
        start_time = time.time()
        
        success = create_output_video(cap, output_path, fps, w, h, "anomaly")
        cap.release()
        
        if not success:
            return None, "Error: Could not create anomaly output video"

        # Detect anomalies with VILA
        print("Detecting anomalies with VILA...")
        anomaly_report = detect_anomalies_with_vila(key_frames, duration)

        # Build Anomaly Report
        report = "🚨 ANOMALY DETECTION SUMMARY\n"
        report += "=" * 50 + "\n\n"
        report += f"📊 Video Details:\n"
        report += f"• Duration: {duration:.2f} seconds ({total_frames} frames)\n"
        report += f"• Resolution: {w}x{h} @ {fps} FPS\n"
        report += f"• Frames Analyzed: {len(key_frames)} key frames\n\n"
        
        # Add common anomaly types reference
        report += f"🔍 Anomaly Types Monitored:\n"
        common_anomalies_display = [
            "• Falling objects (boxes, equipment, items)",
            "• Person falls, trips, or stumbles", 
            "• Equipment malfunctions or breakdowns",
            "• Spills, leaks, or structural damage",
            "• Collisions or unusual impacts",
            "• Safety incidents or violations",
            "• Loitering in sensitive zones",
            "• Theft/shoplifting or concealment behavior",
            "• Unattended objects or packages",
            "• Crowd formations in restricted areas",
            "• Violence, fights, or altercations",
            "• Intrusion during non-operational hours",
            "• Suspicious or erratic movements",
            "• Vandalism or property damage",
            "• Camera blocking or tampering",
            "• Vehicle direction violations",
            "• Abandoned vehicles",
            "• Unusual speed incidents",
            "• Queue anomalies or overcrowding",
            "• Missing protective gear violations",
            "• Unauthorized carrying of items",
            "• Trespassing or fence climbing"
        ]
        report += "\n".join(common_anomalies_display) + "\n\n"
        
        report += f"🤖 VILA Analysis Summary:\n"
        report += "-" * 30 + "\n"
        report += anomaly_report + "\n\n"
        
        report += f"⚙️ Detection Method: NVIDIA VILA AI • Focus: Safety & Incident Detection\n"

        print("Anomaly detection complete!")
        return output_path, report
        
    except Exception as e:
        error_msg = f"Error in anomaly detection: {str(e)}"
        print(error_msg)
        return None, error_msg

# ---- Gradio UI (FIXED) ----
def create_interface():
    with gr.Blocks(title="VILA Video Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 VILA Video Analysis & Anomaly Detection")
        gr.Markdown("Choose your analysis mode: **Upload Video**, **Upload for Anomaly Detection**, or **Live Camera Tracking**!")
        
        with gr.Tabs():
            # Tab 1: Video Upload Analysis
            with gr.TabItem("📁 Upload Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Video input
                        inp = gr.Video(
                            label="📁 Upload Video", 
                            sources=["upload"]
                        )
                        
                        # Analysis buttons
                        analyze_btn = gr.Button("🎬 Analyze Video", variant="primary", size="lg")
                        
                        with gr.Accordion("ℹ️ Analysis Info", open=False):
                            gr.Markdown("""
                            **🎬 General Video Analysis:**
                            - Describes activities, people, and settings
                            - Provides natural language summary
                            - Identifies interactions and story flow
                            """)
                    
                    with gr.Column(scale=2):
                        # Output components
                        out_vid = gr.Video(label="📹 Processed Video")
                        out_txt = gr.Textbox(
                            label="📋 Analysis Report", 
                            lines=25, 
                            max_lines=35,
                            show_copy_button=True,
                            interactive=False
                        )

            # Tab 2: Anomaly Detection
            with gr.TabItem("🚨 Anomaly Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Video input for anomaly detection
                        inp_anomaly = gr.Video(
                            label="📁 Upload Video for Anomaly Detection", 
                            sources=["upload"]
                        )
                        
                        # Anomaly detection button
                        anomaly_btn = gr.Button("🚨 Detect Anomalies", variant="secondary", size="lg")
                        
                        with gr.Accordion("🔍 Anomaly Types", open=False):
                            gr.Markdown("""
                            **🚨 Anomaly Detection:**
                            - Detects falls, accidents, and unusual events
                            - Identifies objects falling or equipment failures
                            - Reports safety incidents and disruptions
                            - Provides severity assessment of anomalies
                            
                            **Monitored Anomalies:**
                            - Falling objects/people
                            - Equipment malfunctions
                            - Spills and collisions
                            - Theft and suspicious behavior
                            - Safety violations
                            - Unauthorized access
                            """)
                    
                    with gr.Column(scale=2):
                        # Output components for anomaly detection
                        out_vid_anomaly = gr.Video(label="📹 Anomaly Scan Video")
                        out_txt_anomaly = gr.Textbox(
                            label="🚨 Anomaly Report", 
                            lines=25, 
                            max_lines=35,
                            show_copy_button=True,
                            interactive=False
                        )

            # Tab 3: Live Camera Tracking (FIXED)
            with gr.TabItem("📹 Live Camera Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Live tracking controls
                        gr.Markdown("### 🔴 Live Camera Controls")
                        
                        with gr.Row():
                            start_live_btn = gr.Button("🔴 Start Live Tracking", variant="primary", size="lg")
                            stop_live_btn = gr.Button("🛑 Stop Live Tracking", variant="stop", size="lg")
                        
                        gr.Markdown("### 📊 Manual Analysis")
                        with gr.Row():
                            live_analyze_btn = gr.Button("🎬 Analyze Current Video", variant="primary")
                            live_anomaly_btn = gr.Button("🚨 Check for Anomalies", variant="secondary")
                        
                        # Live status
                        live_status = gr.Textbox(
                            label="📊 Live Tracking Status",
                            lines=8,
                            interactive=False,
                            value="🔴 Live tracking not started. Click 'Start Live Tracking' to begin."
                        )
                        
                        with gr.Accordion("📖 Live Tracking Guide", open=False):
                            gr.Markdown("""
                            **🔴 How Live Tracking Works:**
                            
                            1. **Start Tracking**: Opens your camera
                            2. **Automatic Analysis**: Every 20 seconds, analyzes recent activity
                            3. **Real-time Anomaly Detection**: Scans for unusual events every 5 seconds
                            4. **Manual Analysis**: Click buttons anytime for instant analysis
                            
                            **📊 Features:**
                            - Live video feed with real-time overlay
                            - 20-second automatic summaries
                            - 5-second anomaly monitoring
                            - Instant analysis on demand
                            - Continuous monitoring
                            
                            **⚠️ Notes:**
                            - Ensure camera permissions are enabled
                            - Good lighting improves accuracy
                            - Use 'Refresh' buttons to update displays
                            """)
                    
                    with gr.Column(scale=2):
                        # Live video feed
                        live_video = gr.Image(
                            label="📹 Live Camera Feed",
                            height=400,
                            show_download_button=False
                        )
                        
                        # Add refresh button for video feed
                        with gr.Row():
                            refresh_video_btn = gr.Button("🔄 Refresh Video Feed", size="sm", variant="secondary")
                            auto_refresh_btn = gr.Button("🔄 Auto Refresh ON/OFF", size="sm")
                        
                        # Live reports
                        live_reports = gr.Textbox(
                            label="📋 Live Analysis Reports",
                            lines=15,
                            max_lines=25,
                            show_copy_button=True,
                            interactive=False,
                            value="📋 Live analysis reports will appear here...\n\n🔄 Automatic reports every 20 seconds\n🚨 Anomaly alerts every 5 seconds\n📊 Manual analysis reports on demand"
                        )
                        
                        # Manual refresh for reports
                        refresh_reports_btn = gr.Button("🔄 Refresh Reports", size="sm", variant="secondary")

        # Tips section
        gr.Markdown("### 💡 Tips for Best Results:")
        gr.Markdown("• **Upload Analysis**: Good for detailed post-processing of recorded videos • **Anomaly Detection**: Best for safety monitoring and incident review • **Live Tracking**: Perfect for real-time monitoring and immediate response • Clear lighting and stable camera improve accuracy • Use refresh buttons to update live displays")

        # Event handlers for uploaded video analysis
        analyze_btn.click(
            fn=process_video, 
            inputs=[inp], 
            outputs=[out_vid, out_txt]
        )
        
        anomaly_btn.click(
            fn=detect_video_anomalies, 
            inputs=[inp_anomaly], 
            outputs=[out_vid_anomaly, out_txt_anomaly]
        )
        
        # Event handlers for live tracking
        start_live_btn.click(
            fn=start_live_tracking,
            inputs=[],
            outputs=[live_video, live_status]
        )
        
        stop_live_btn.click(
            fn=stop_live_tracking,
            inputs=[],
            outputs=[live_video, live_status]
        )
        
        # Event handlers for live analysis
        live_analyze_btn.click(
            fn=process_live_video_analysis,
            inputs=[],
            outputs=[live_reports]
        )
        
        live_anomaly_btn.click(
            fn=process_live_anomaly_detection,
            inputs=[],
            outputs=[live_reports]
        )
        
        # Refresh handlers
        refresh_reports_btn.click(
            fn=get_live_updates,
            inputs=[],
            outputs=[live_reports]
        )
        
        refresh_video_btn.click(
            fn=get_current_live_frame,
            inputs=[],
            outputs=[live_video]
        )

        # Auto-refresh setup using gr.Timer (if available) or manual refresh instructions
        try:
            # Try to set up auto-refresh timer for live video (every 100ms for smooth video)
            video_timer = gr.Timer(0.1)  # 100ms = 10 FPS display refresh
            video_timer.tick(
                fn=get_current_live_frame,
                inputs=[],
                outputs=[live_video]
            )
            
            # Timer for reports refresh (every 2 seconds)
            reports_timer = gr.Timer(2.0)  # 2 seconds
            reports_timer.tick(
                fn=get_live_updates,
                inputs=[],
                outputs=[live_reports]
            )
            
        except Exception as e:
            print("Auto-refresh timer not available, using manual refresh buttons")
            gr.Markdown("**Note**: Use the refresh buttons to update live displays manually.")

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False
    )