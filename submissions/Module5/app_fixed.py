"""
Module 5: Real-Time Object Tracker Backend
Flask blueprint for marker-based, markerless, and SAM2 segmentation tracking.
"""

from flask import Blueprint, Flask, request, jsonify, render_template
import cv2
import numpy as np
import threading
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

# --- Utility Functions ---

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

# --- API Endpoints ---

@module5_bp.route('/api/track/init', methods=['POST'])
def track_init():
    global tracker, tracker_mode, tracking_active, initial_frame_data, sam2_mask
    
    print(f"[DEBUG] Init tracker endpoint called")
    data = request.get_json() or {}
    mode = data.get('mode', 'marker')
    mask_option = data.get('mask_option', 'dummy')
    print(f"[DEBUG] Mode: {mode}, Mask Option: {mask_option}, Data: {data}")
    tracking_active = True
    tracker = None # Reset tracker instance
    initial_frame_data = None # Reset initial frame

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
            # Initialize OpenCV CSRT tracker (robust and accurate)
            tracker = cv2.TrackerCSRT_create()
            
        elif mode == 'sam2':
            tracker_mode = 'sam2'
            if mask_option == 'dummy' or sam2_mask is None:
                # If no mask uploaded, generate a dummy one for demonstration
                sam2_mask = create_dummy_mask(480, 640)
                print(f"[DEBUG] Using dummy mask for SAM2 mode")
            else:
                print(f"[DEBUG] Using uploaded mask for SAM2 mode")
            
        else:
            return jsonify({'error': 'Invalid tracking mode specified'}), 400
            
    except Exception as e:
        # Log the detailed exception
        print(f"Initialization error in mode {mode}: {e}")
        return jsonify({'error': f'Tracker initialization failed for mode {mode}: {str(e)}'}), 500
    
    print(f"Tracker initialized in mode: {tracker_mode}")
    return jsonify({'status': 'initialized', 'mode': tracker_mode})

@module5_bp.route('/api/track', methods=['POST'])
def track():
    global tracker, tracker_mode, sam2_mask, tracking_active, initial_frame_data
    
    if not tracking_active or tracker_mode is None:
        return jsonify({'error': 'Tracker not initialized or stopped'}), 400
    
    # Decode frame from raw POST data
    try:
        # The frontend sends a raw JPEG blob, so we read it directly.
        frame_bytes = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Could not decode image frame'}), 400
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
                    # Initialization prompt for the user
                    initial_frame_data = True 
                    
                    # Define default center area for object placement
                    bbox = (w//4, h//4, w//2, h//2)
                    tracker.init(frame, bbox)
                    
                    # Draw initial box and instruction
                    (x, y, bw, bh) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                    cv2.putText(frame, 'Place object inside the BLUE box and hold steady.', 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
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
                # --- SAM2 Segmentation Mask Overlay ---
                if sam2_mask is not None:
                    # Ensure mask matches frame size
                    if sam2_mask.shape[:2] != (h, w):
                        mask_resized = cv2.resize(sam2_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = sam2_mask
                    
                    # Create colored overlay 
                    mask_colored = np.zeros_like(frame)
                    mask_indices = mask_resized > 0.5
                    
                    mask_colored[mask_indices] = [200, 200, 0] # Light Teal color
                    
                    # Blend with original frame
                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
                    
                    # Draw text indicator
                    cv2.putText(frame, 'SAM2 Mask Overlay Active', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                else:
                    cv2.putText(frame, 'No SAM2 mask uploaded!', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            # Catch errors during processing 
            cv2.putText(frame, f'Error: {str(e)[:50]}...', (10, h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Processing error: {e}")
            
    # Encode processed frame back to JPEG for streaming
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@module5_bp.route('/api/track/stop', methods=['POST'])
def track_stop():
    global tracking_active, tracker, tracker_mode, initial_frame_data
    tracking_active = False
    tracker = None # Destroy tracker instance
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
            # NOTE: Mocking the NPZ file upload by creating a dummy mask
            sam2_mask = create_dummy_mask(480, 640) 
            print("SAM2 mask mocked and loaded.")
            return jsonify({'status': 'Mask processed successfully (mocked).', 
                            'size': sam2_mask.shape})
        except Exception as e:
            # Handle potential file reading or format errors
            print(f"SAM2 upload error: {e}")
            return jsonify({'error': f'Failed to process mask file: {str(e)}'}), 500

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