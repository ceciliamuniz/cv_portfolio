"""
Module 5: Real-Time Object Tracker Backend
Flask blueprint for marker-based, markerless, and SAM2 segmentation tracking.
"""

from flask import Blueprint, Flask, request, jsonify, render_template
import cv2
import numpy as np
import threading
import os
import base64
from io import BytesIO
from pathlib import Path

# Use absolute paths for blueprint
template_dir = Path(__file__).parent / 'templates'
print(f"[DEBUG] Module 5 template directory: {template_dir}")
print(f"[DEBUG] Template exists: {(template_dir / 'module5.html').exists()}")

module5_bp = Blueprint('module5', __name__, 
                      template_folder=str(template_dir))

# Global tracker state and synchronization lock
tracker = None
tracker_mode = None
# sam2_mask stores a dummy NumPy array representing the segmentation mask (H, W)
sam2_mask = None 
tracking_active = False
lock = threading.Lock()
# Store the initial frame for markerless tracking initialization
initial_frame_data = None
# Store user-selected bounding box for markerless tracking
user_bbox = None

def create_dummy_mask(h, w):
    """Creates a simple circular placeholder mask in the center of the frame
    
    Args:
        h: Height of the mask
        w: Width of the mask
        
    Returns:
        numpy.ndarray: Binary mask with circular region in center
    """
    mask = np.zeros((h, w), dtype=np.float32)
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) // 4  # Radius is 1/4 of the smaller dimension
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask[dist_from_center <= radius] = 1.0
    print(f"[DEBUG] Created dummy mask: {h}x{w}, center=({center_x},{center_y}), radius={radius}")
    return mask

@module5_bp.route('/api/track/init', methods=['POST'])
def track_init():
    global tracker, tracker_mode, tracking_active, initial_frame_data, sam2_mask, user_bbox
    
    print(f"[DEBUG] Init tracker endpoint called")
    data = request.get_json() or {}
    mode = data.get('mode', 'marker')
    mask_option = data.get('mask_option', 'dummy')
    bbox_data = data.get('bbox')  # User-selected bounding box for markerless tracking
    print(f"[DEBUG] Mode: {mode}, Mask Option: {mask_option}, BBox: {bbox_data}")
    tracking_active = True
    tracker = None # Reset tracker instance
    initial_frame_data = None # Reset initial frame
    user_bbox = bbox_data  # Store user selection

    try:
        if mode == 'marker':
            tracker_mode = 'marker'
            # Initialize ArUco detector using the modern OpenCV API (cv2.aruco.ArucoDetector)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            aruco_params = cv2.aruco.DetectorParameters()
            # Store the detector instance directly
            tracker = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        elif mode == 'markerless':
            tracker_mode = 'markerless'
            
            # Check if user has made a selection
            if not bbox_data or len(bbox_data) != 4:
                return jsonify({'error': 'Please select a region with your mouse first. Switch to markerless mode, draw a selection box, then click Init Tracker.'}), 400
            
            # Initialize OpenCV tracker (use available trackers from opencv-contrib-python)
            tracker = None
            tracker_options = [
                ('MIL', cv2.TrackerMIL_create),
                ('DaSiamRPN', cv2.TrackerDaSiamRPN_create),
                ('GOTURN', cv2.TrackerGOTURN_create),
                ('Nano', cv2.TrackerNano_create)
            ]
            
            for name, create_func in tracker_options:
                try:
                    tracker = create_func()
                    print(f"[DEBUG] Successfully initialized {name} tracker")
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to initialize {name} tracker: {e}")
                    continue
            
            if tracker is None:
                raise Exception("No compatible OpenCV tracker found. Available trackers: MIL, DaSiamRPN, GOTURN, Nano")
            
        elif mode == 'sam2':
            tracker_mode = 'sam2'
            if mask_option == 'dummy' or sam2_mask is None:
                sam2_mask = create_dummy_mask(480, 640)
                print(f"[DEBUG] Using dummy mask for SAM2 mode")
            else:
                print(f"[DEBUG] Using uploaded mask for SAM2 mode")
        
        else:
            return jsonify({'error': 'Invalid tracking mode'}), 400
            
    except Exception as e:
        print(f"Initialization error in mode {mode}: {e}")
        return jsonify({'error': f'Tracker initialization failed: {str(e)}'}), 500
    
    print(f"[DEBUG] Tracker initialized in mode: {tracker_mode}")
    return jsonify({'status': 'initialized', 'mode': tracker_mode})

@module5_bp.route('/api/track', methods=['POST'])
def track():
    global tracker, tracker_mode, sam2_mask, tracking_active, initial_frame_data
    
    if not tracking_active or tracker_mode is None:
        return jsonify({'error': 'Tracker not initialized'}), 400
    
    try:
        # Decode frame from raw POST data
        frame_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Could not decode frame'}), 400
    except Exception as e:
        return jsonify({'error': f'Frame decoding error: {str(e)}'}), 400

    h, w = frame.shape[:2]

    with lock:
        try:
            if tracker_mode == 'marker':
                # --- Marker-Based Tracking (ArUco) ---
                
                # Check if the tracker is the ArucoDetector object
                if isinstance(tracker, cv2.aruco.ArucoDetector):
                    # Use the detector object's method to detect markers
                    corners, ids, rejected = tracker.detectMarkers(frame)
                else:
                    raise TypeError("Marker tracker not initialized correctly as ArucoDetector.")
                
                if ids is not None:
                    print(f"[DEBUG] Markers detected: IDs {ids.flatten()}")
                    
                    # Draw detected markers and their IDs
                    # Note: drawDetectedMarkers is still a static function
                    cv2.aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 255, 0))
                    
                    for i in range(len(ids)):
                        corner = corners[i][0]
                        # Draw center ID for clarity
                        center = np.mean(corner, axis=0).astype(int)
                        cv2.putText(frame, f'ID: {ids[i][0]}', 
                                    tuple(center + [10, 10]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    print("[DEBUG] No markers detected.")
                        
            elif tracker_mode == 'markerless':
                # --- Markerless Tracking (CSRT) ---
                if initial_frame_data is None:
                    # Check if we have a user-selected bounding box
                    if user_bbox and len(user_bbox) == 4:
                        # Initialize tracker with user selection - convert to integers for OpenCV
                        initial_frame_data = True
                        bbox = tuple(int(coord) for coord in user_bbox)
                        print(f"[DEBUG] Using user-selected bbox: {bbox}")
                        
                        tracker.init(frame, bbox)
                        
                        # Draw initial box and instruction
                        (x, y, bw, bh) = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                        cv2.putText(frame, 'Tracking selected region...', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        # Wait for user selection - don't initialize tracker yet
                        cv2.putText(frame, 'Please select an object with your mouse first, then reinitialize', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, 'Go back to markerless mode and draw a selection box', 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                else:
                    # Tracking phase
                    success, bbox = tracker.update(frame)
                    
                    if success:
                        # Draw bounding box
                        (x, y, bw, bh) = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                        cv2.putText(frame, 'Tracking', (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Tracking Lost - Reinitialize', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
            elif tracker_mode == 'sam2':
                # SAM2 segmentation with tracking overlay
                if sam2_mask is not None:
                    if initial_frame_data is None:
                        # Check if we have a user-selected bounding box for SAM2 tracking
                        if user_bbox and len(user_bbox) == 4:
                            # Initialize tracker with user selection
                            initial_frame_data = True
                            bbox = tuple(int(coord) for coord in user_bbox)
                            print(f"[DEBUG] SAM2 mode using user-selected bbox: {bbox}")
                            
                            # Initialize tracker for SAM2 mode
                            try:
                                sam2_tracker = cv2.TrackerMIL_create()
                                sam2_tracker.init(frame, bbox)
                                tracker = sam2_tracker  # Store tracker reference
                                print("[DEBUG] SAM2 tracker initialized successfully")
                            except Exception as e:
                                print(f"[DEBUG] SAM2 tracker init failed: {e}")
                        else:
                            # Show instruction to select region for SAM2 tracking
                            cv2.putText(frame, 'SAM2 Mode: Select region with mouse first, then reinitialize', 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frame, 'The mask will follow your selected object', 
                                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Tracking phase - update tracker and move mask accordingly
                        if tracker is not None:
                            success, bbox = tracker.update(frame)
                            
                            if success:
                                # Get current tracked position
                                (x, y, bw, bh) = [int(v) for v in bbox]
                                
                                # Create mask at tracked position
                                mask_at_position = np.zeros((h, w), dtype=np.float32)
                                
                                # Create circular mask centered in the tracked bounding box
                                center_x = x + bw // 2
                                center_y = y + bh // 2
                                radius = min(bw, bh) // 3
                                
                                cv2.circle(mask_at_position, (center_x, center_y), radius, 1.0, -1)
                                
                                # Apply mask overlay
                                mask_colored = np.zeros_like(frame)
                                mask_indices = mask_at_position > 0.5
                                mask_colored[mask_indices] = [0, 200, 200]  # Yellow-green mask
                                
                                alpha = 0.4
                                frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
                                
                                # Draw tracking box
                                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
                                cv2.putText(frame, 'SAM2 Tracking + Mask', (x, y - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                cv2.putText(frame, 'SAM2 Tracking Lost - Reinitialize', (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            # Fallback: static mask overlay
                            if sam2_mask.shape[:2] != (h, w):
                                mask_resized = cv2.resize(sam2_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_resized = sam2_mask
                            
                            mask_colored = np.zeros_like(frame)
                            mask_indices = mask_resized > 0.5
                            mask_colored[mask_indices] = [200, 200, 0]  # Light teal
                            
                            alpha = 0.5
                            frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
                            
                            cv2.putText(frame, 'SAM2 Static Mask (no tracking)', (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                else:
                    cv2.putText(frame, 'No SAM2 mask - using dummy mode!', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            cv2.putText(frame, f'Error: {str(e)[:50]}...', (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Processing error: {e}")
            
    # Encode processed frame back to JPEG
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@module5_bp.route('/api/track/stop', methods=['POST'])
def track_stop():
    global tracking_active, tracker, tracker_mode, initial_frame_data
    
    tracking_active = False
    tracker = None
    tracker_mode = None
    initial_frame_data = None
    print("Tracker stopped and reset.")
    return jsonify({'status': 'stopped'})

@module5_bp.route('/api/sam2/upload', methods=['POST'])
def sam2_upload():
    global sam2_mask
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # In real implementation, you would load the actual NPZ file:
            # data = np.load(file)
            # sam2_mask = data['mask']  # or whatever key contains the mask
            
            # For now, mock SAM2 mask loading - generate dummy mask for demo
            sam2_mask = create_dummy_mask(480, 640)
            print("SAM2 mask file uploaded and processed (currently mocked with dummy mask).")
            return jsonify({'status': 'NPZ file uploaded successfully! (Currently using dummy mask for demo)', 
                           'size': sam2_mask.shape,
                           'note': 'Real NPZ processing not implemented yet'})
        except Exception as e:
            print(f"SAM2 upload error: {e}")
            return jsonify({'error': f'Failed to process mask: {str(e)}'}), 500

@module5_bp.route('/api/sam2/mask', methods=['GET'])
def sam2_mask_get():
    global sam2_mask
    if sam2_mask is None:
        return jsonify({'error': 'No mask uploaded'}), 400
    # Return mask as image (dummy response)
    mask_img = (sam2_mask * 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', mask_img)
    return buf.tobytes(), 200, {'Content-Type': 'image/png'}


@module5_bp.route('/')
def module5_page():
    # Add cache-busting headers
    from flask import make_response
    import time
    
    response = make_response(render_template('module5.html', cache_buster=int(time.time())))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@module5_bp.route('/test')
def module5_test():
    """Test route to verify module is working"""
    return jsonify({
        'status': 'Module 5 is working!',
        'tracking_active': tracking_active,
        'tracker_mode': tracker_mode,
        'routes': [
            '/module5/ - Main tracking interface',
            '/module5/test - This test page',
            '/module5/api/track/init - Initialize tracker',
            '/module5/api/track - Process frames',
            '/module5/api/track/stop - Stop tracking',
            '/module5/api/sam2/upload - Upload SAM2 mask'
        ]
    })

@module5_bp.route('/info')
def info():
    return '<h2>Module 5: Real-Time Object Tracker Backend</h2><p>Integrated into main portfolio.</p>'

# Standalone app support
if __name__ == '__main__':
    app = Flask(__name__, template_folder='templates')
    app.register_blueprint(module5_bp, url_prefix='/')
    print("Module 5 running standalone on http://localhost:5011")
    app.run(debug=True, port=5011)
