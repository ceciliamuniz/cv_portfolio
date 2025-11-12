"""
Module 5: Real-Time Object Tracker Backend
Standalone Flask app for marker-based, markerless, and SAM2 segmentation tracking.
"""

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import threading
import os

app = Flask(__name__, template_folder='templates')

# Global tracker state
tracker = None
tracker_mode = None
sam2_mask = None
tracking_active = False
lock = threading.Lock()

@app.route('/api/track/init', methods=['POST'])
def track_init():
    global tracker, tracker_mode, tracking_active
    data = request.json
    mode = data.get('mode')
    tracking_active = True
    if mode == 'marker':
        tracker_mode = 'marker'
        # Marker-based tracker setup (Aruco/QR/April)
        # TODO: Initialize marker detector
    elif mode == 'markerless':
        tracker_mode = 'markerless'
        # Markerless tracker setup (OpenCV tracker)
        # TODO: Initialize OpenCV tracker
    elif mode == 'sam2':
        tracker_mode = 'sam2'
        # SAM2 mask-based tracking
        # TODO: Load mask if available
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    return jsonify({'status': 'initialized', 'mode': tracker_mode})

@app.route('/api/track', methods=['POST'])
def track():
    global tracker, tracker_mode, sam2_mask, tracking_active
    if not tracking_active:
        return jsonify({'error': 'Tracker not initialized'}), 400
    frame = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    overlay = None
    if tracker_mode == 'marker':
        # Marker detection and overlay
        # TODO: Detect marker and draw overlay
        pass
    elif tracker_mode == 'markerless':
        # Markerless tracking and overlay
        # TODO: Track object and draw overlay
        pass
    elif tracker_mode == 'sam2':
        # Use sam2_mask for overlay
        # TODO: Overlay mask on frame
        pass
    else:
        return jsonify({'error': 'Invalid tracker mode'}), 400
    # Return overlay (dummy response for now)
    _, buf = cv2.imencode('.jpg', frame)
    return buf.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/api/track/stop', methods=['POST'])
def track_stop():
    global tracking_active
    tracking_active = False
    return jsonify({'status': 'stopped'})

@app.route('/api/sam2/upload', methods=['POST'])
def sam2_upload():
    global sam2_mask
    file = request.files['file']
    sam2_mask = np.load(file)
    return jsonify({'status': 'mask uploaded'})

@app.route('/api/sam2/mask', methods=['GET'])
def sam2_mask_get():
    global sam2_mask
    if sam2_mask is None:
        return jsonify({'error': 'No mask uploaded'}), 400
    # Return mask as image (dummy response)
    mask_img = (sam2_mask * 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', mask_img)
    return buf.tobytes(), 200, {'Content-Type': 'image/png'}


@app.route('/module5')
def module5_page():
    return render_template('module5.html')

@app.route('/')
def index():
    return '<h2>Module 5: Real-Time Object Tracker Backend</h2><p>Use /module5 for the UI.</p>'

if __name__ == '__main__':
    app.run(debug=True)
