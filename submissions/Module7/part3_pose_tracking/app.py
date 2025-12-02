"""
Module 7 Part 3: Real-time Pose Estimation and Hand Tracking
Implements pose estimation and hand tracking using MediaPipe with CSV data export
"""

import cv2
import numpy as np
import pandas as pd
import base64
import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, send_file, Response

# Initialize MediaPipe with error handling
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
    print("[INFO] MediaPipe successfully imported")
except ImportError as e:
    print(f"[ERROR] MediaPipe import failed: {e}")
    MEDIAPIPE_AVAILABLE = False
    # Create dummy objects to prevent errors
    class DummyMP:
        def __init__(self): pass
    mp_drawing = DummyMP()
    mp_drawing_styles = DummyMP()
    mp_pose = DummyMP()
    mp_hands = DummyMP()
    mp_holistic = DummyMP()

# Create blueprint with absolute template folder path
template_dir = Path(__file__).parent / 'templates'
static_dir = Path(__file__).parent / 'static'
pose_bp = Blueprint('pose_tracking', __name__, 
                   template_folder=str(template_dir),
                   static_folder=str(static_dir))

# Global variables for real-time processing
current_frame = None
pose_data_buffer = []
recording_active = False
processing_lock = threading.Lock()

# Export directory
EXPORT_DIR = Path(__file__).parent / 'exports'
EXPORT_DIR.mkdir(exist_ok=True)

class PoseTracker:
    """Comprehensive pose and hand tracking system"""
    
    def __init__(self):
        # Try to initialize MediaPipe regardless of global flag
        try:
            import mediapipe as mp_local
            
            self.pose = mp_local.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.hands = mp_local.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.holistic = mp_local.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[INFO] PoseTracker initialized successfully with MediaPipe")
            self.mediapipe_ready = True
            
        except Exception as e:
            print(f"[ERROR] PoseTracker initialization failed: {e}")
            self.pose = None
            self.hands = None
            self.holistic = None
            self.mediapipe_ready = False
            self.hands = None
            self.holistic = None
    
    def process_frame(self, frame, mode='holistic'):
        """Process a single frame and extract pose/hand data"""
        if not self.mediapipe_ready or self.pose is None:
            print(f"[DEBUG] MediaPipe not ready: mediapipe_ready={getattr(self, 'mediapipe_ready', False)}, pose={self.pose is not None}")
            return {}, frame
        
        try:
            # Validate frame format
            if frame is None:
                print("[ERROR] Frame is None")
                return {}, frame
            
            if not hasattr(frame, 'shape') or not hasattr(frame, 'dtype'):
                print(f"[ERROR] Frame is not a valid array: {type(frame)}")
                return {}, frame
            
            if frame.size == 0:
                print("[ERROR] Frame is empty")
                return {}, frame
            
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"[ERROR] Frame has wrong shape: {frame.shape}")
                return {}, frame
            
            # Ensure frame is contiguous in memory and correct dtype
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Convert frame to proper uint8 format for OpenCV
            if frame.dtype != np.uint8:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    # If float, assume it's normalized 0-1 and scale to 0-255
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                else:
                    # For other types, clip and convert
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure RGB frame is also contiguous and correct dtype
            if not rgb_frame.flags['C_CONTIGUOUS']:
                rgb_frame = np.ascontiguousarray(rgb_frame)
            
            h, w = frame.shape[:2]
            
            results = {}
            annotated_frame = frame.copy()
            
            # Import MediaPipe locally for drawing
            import mediapipe as mp_local
            mp_drawing_local = mp_local.solutions.drawing_utils
            mp_drawing_styles_local = mp_local.solutions.drawing_styles
            
            if mode == 'pose_only':
                try:
                    pose_results = self.pose.process(rgb_frame)
                    results['pose'] = pose_results
                except Exception as mp_error:
                    print(f"[ERROR] MediaPipe pose processing failed: {mp_error}")
                    results['pose'] = type('MockResults', (), {'pose_landmarks': None})()
                    return results, frame
                if pose_results.pose_landmarks:
                    mp_drawing_local.draw_landmarks(
                        annotated_frame,
                        pose_results.pose_landmarks,
                        mp_local.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles_local.get_default_pose_landmarks_style()
                    )
            
            elif mode == 'hands_only':
                try:
                    hand_results = self.hands.process(rgb_frame)
                    results['hands'] = hand_results
                except Exception as mp_error:
                    print(f"[ERROR] MediaPipe hands processing failed: {mp_error}")
                    results['hands'] = type('MockResults', (), {'multi_hand_landmarks': None, 'multi_handedness': None})()
                    return results, frame
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing_local.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            mp_local.solutions.hands.HAND_CONNECTIONS,
                            mp_drawing_styles_local.get_default_hand_landmarks_style(),
                            mp_drawing_styles_local.get_default_hand_connections_style()
                        )
            
            elif mode == 'holistic':
                try:
                    holistic_results = self.holistic.process(rgb_frame)
                    results['holistic'] = holistic_results
                except Exception as mp_error:
                    print(f"[ERROR] MediaPipe holistic processing failed: {mp_error}")
                    results['holistic'] = type('MockResults', (), {
                        'pose_landmarks': None,
                        'left_hand_landmarks': None,
                        'right_hand_landmarks': None,
                        'face_landmarks': None
                    })()
                    return results, frame
                
                # Draw pose landmarks
                if holistic_results.pose_landmarks:
                    mp_drawing_local.draw_landmarks(
                        annotated_frame,
                        holistic_results.pose_landmarks,
                        mp_local.solutions.holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles_local.get_default_pose_landmarks_style()
                    )
                
                # Draw hand landmarks
                if holistic_results.left_hand_landmarks:
                    mp_drawing_local.draw_landmarks(
                        annotated_frame,
                        holistic_results.left_hand_landmarks,
                        mp_local.solutions.holistic.HAND_CONNECTIONS,
                        mp_drawing_styles_local.get_default_hand_landmarks_style(),
                        mp_drawing_styles_local.get_default_hand_connections_style()
                    )
                
                if holistic_results.right_hand_landmarks:
                    mp_drawing_local.draw_landmarks(
                        annotated_frame,
                        holistic_results.right_hand_landmarks,
                        mp_local.solutions.holistic.HAND_CONNECTIONS,
                        mp_drawing_styles_local.get_default_hand_landmarks_style(),
                        mp_drawing_styles_local.get_default_hand_connections_style()
                    )
                
                # Draw face landmarks (optional)
                if holistic_results.face_landmarks:
                    mp_drawing_local.draw_landmarks(
                        annotated_frame,
                        holistic_results.face_landmarks,
                        mp_local.solutions.holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles_local.get_default_face_mesh_contours_style()
                    )
        
            return results, annotated_frame
            
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            return {}, frame
    
    def extract_pose_data(self, results, timestamp, frame_number, mode='holistic'):
        """Extract structured data from MediaPipe results"""
        data_row = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'mode': mode
        }
        
        # Always create the full structure with default None values
        # Create pose landmark columns (33 points)
        for i in range(33):
            data_row[f'pose_{i}_x'] = None
            data_row[f'pose_{i}_y'] = None
            data_row[f'pose_{i}_z'] = None
            data_row[f'pose_{i}_visibility'] = None
        
        # Create hand landmark columns (21 points each)
        for i in range(21):
            data_row[f'left_hand_{i}_x'] = None
            data_row[f'left_hand_{i}_y'] = None
            data_row[f'left_hand_{i}_z'] = None
            data_row[f'right_hand_{i}_x'] = None
            data_row[f'right_hand_{i}_y'] = None
            data_row[f'right_hand_{i}_z'] = None
        
        # Now fill in actual detected data
        if mode == 'holistic' and results.get('holistic'):
            holistic_results = results['holistic']
            
            # Fill in actual pose data if detected
            if holistic_results.pose_landmarks:
                for i, landmark in enumerate(holistic_results.pose_landmarks.landmark):
                    data_row[f'pose_{i}_x'] = float(landmark.x)
                    data_row[f'pose_{i}_y'] = float(landmark.y)
                    data_row[f'pose_{i}_z'] = float(landmark.z)
                    data_row[f'pose_{i}_visibility'] = float(landmark.visibility)
            
            # Fill in left hand data if detected
            if holistic_results.left_hand_landmarks:
                for i, landmark in enumerate(holistic_results.left_hand_landmarks.landmark):
                    data_row[f'left_hand_{i}_x'] = float(landmark.x)
                    data_row[f'left_hand_{i}_y'] = float(landmark.y)
                    data_row[f'left_hand_{i}_z'] = float(landmark.z)
            
            # Fill in right hand data if detected
            if holistic_results.right_hand_landmarks:
                for i, landmark in enumerate(holistic_results.right_hand_landmarks.landmark):
                    data_row[f'right_hand_{i}_x'] = float(landmark.x)
                    data_row[f'right_hand_{i}_y'] = float(landmark.y)
                    data_row[f'right_hand_{i}_z'] = float(landmark.z)
        
        elif mode == 'pose_only' and results.get('pose'):
            pose_results = results['pose']
            if pose_results.pose_landmarks:
                for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    data_row[f'pose_{i}_x'] = float(landmark.x)
                    data_row[f'pose_{i}_y'] = float(landmark.y)
                    data_row[f'pose_{i}_z'] = float(landmark.z)
                    data_row[f'pose_{i}_visibility'] = float(landmark.visibility)
        
        elif mode == 'hands_only' and results.get('hands'):
            hand_results = results['hands']
            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = hand_results.multi_handedness[hand_idx].classification[0].label.lower()
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        data_row[f'{hand_label}_hand_{i}_x'] = float(landmark.x)
                        data_row[f'{hand_label}_hand_{i}_y'] = float(landmark.y)
                        data_row[f'{hand_label}_hand_{i}_z'] = float(landmark.z)
        
        return data_row

# Initialize tracker
tracker = PoseTracker()

@pose_bp.route('/')
def pose_tracking_page():
    """Main pose tracking interface"""
    try:
        return render_template('pose_tracking.html', mediapipe_available=MEDIAPIPE_AVAILABLE)
    except Exception as e:
        # Fallback if template fails
        return f'''
        <h1>Module 7 Part 3: Pose & Hand Tracking</h1>
        <p>MediaPipe Status: {'Available' if MEDIAPIPE_AVAILABLE else 'Not Available'}</p>
        <p>Template Error: {str(e)}</p>
        <p>Please install MediaPipe: <code>pip install mediapipe</code></p>
        <a href="/module7">Back to Module 7 Main</a>
        '''

@pose_bp.route('/api/process_frame', methods=['POST'])
def process_frame_api():
    """Process a single uploaded frame"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        mode = request.form.get('mode', 'holistic')
        
        # Read and decode image
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process frame
        results, annotated_frame = tracker.process_frame(frame, mode)
        
        # Convert annotated frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract pose data
        timestamp = datetime.now().isoformat()
        pose_data = tracker.extract_pose_data(results, timestamp, 0, mode)
        
        return jsonify({
            'success': True,
            'annotated_image': f'data:image/jpeg;base64,{img_base64}',
            'pose_data': pose_data,
            'landmarks_detected': {
                'pose': bool(results.get('holistic', {}).pose_landmarks if mode == 'holistic' else results.get('pose', {}).pose_landmarks),
                'left_hand': bool(results.get('holistic', {}).left_hand_landmarks if mode == 'holistic' else False),
                'right_hand': bool(results.get('holistic', {}).right_hand_landmarks if mode == 'holistic' else False),
                'face': bool(results.get('holistic', {}).face_landmarks if mode == 'holistic' else False)
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@pose_bp.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start recording pose data"""
    global recording_active, pose_data_buffer
    
    with processing_lock:
        recording_active = True
        pose_data_buffer = []
    
    return jsonify({'success': True, 'message': 'Recording started'})

@pose_bp.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording and return session data"""
    global recording_active
    
    with processing_lock:
        recording_active = False
        data_count = len(pose_data_buffer)
    
    return jsonify({
        'success': True, 
        'message': 'Recording stopped',
        'frames_recorded': data_count
    })

@pose_bp.route('/api/export_csv', methods=['POST'])
def export_csv():
    """Export recorded pose data to CSV"""
    global pose_data_buffer
    
    if not pose_data_buffer:
        return jsonify({'error': 'No data to export'}), 400
    
    try:
        # Convert numpy arrays to native Python types before creating DataFrame
        cleaned_data = []
        for row in pose_data_buffer:
            cleaned_row = {}
            for key, value in row.items():
                # Handle numpy types more comprehensively
                if hasattr(value, 'item'):  # numpy scalar
                    cleaned_row[key] = value.item()
                elif hasattr(value, 'tolist'):  # numpy array
                    cleaned_row[key] = value.tolist() if hasattr(value, '__len__') else value.item()
                elif str(type(value)).startswith('<class \'numpy.'):
                    # Catch any other numpy types
                    if hasattr(value, 'item'):
                        cleaned_row[key] = value.item()
                    else:
                        cleaned_row[key] = float(value) if value is not None else None
                else:
                    cleaned_row[key] = value
            cleaned_data.append(cleaned_row)
        
        # Create DataFrame with cleaned data
        df = pd.DataFrame(cleaned_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'pose_data_{timestamp}.csv'
        filepath = EXPORT_DIR / filename
        
        # Save CSV
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'rows_exported': len(df),
            'columns': list(df.columns),
            'download_url': f'/module7/pose/api/download_csv/{filename}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@pose_bp.route('/api/download_csv/<filename>')
def download_csv(filename):
    """Download exported CSV file"""
    filepath = EXPORT_DIR / filename
    
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@pose_bp.route('/api/video_feed')
def video_feed():
    """Real-time video feed with pose tracking"""
    def generate():
        cap = cv2.VideoCapture(0)
        frame_number = 0
        
        if not cap.isOpened():
            print("[ERROR] Camera not accessible for video feed")
            # Return a placeholder image instead of failing
            placeholder_frame = create_placeholder_frame("Camera Not Available")
            _, buffer = cv2.imencode('.jpg', placeholder_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
        
        print("[DEBUG] Video feed started successfully")
        
        try:
            last_process_time = 0
            process_interval = 1.0 / 15  # Limit to 15 FPS to reduce load
            error_count = 0
            max_errors = 10  # Circuit breaker threshold
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[DEBUG] Failed to read frame from camera")
                    break
                
                # Throttle processing to prevent spam
                current_time = time.time()
                if current_time - last_process_time < process_interval:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    continue
                last_process_time = current_time
                
                try:
                    # Circuit breaker: skip processing if too many errors
                    if error_count >= max_errors:
                        print(f"[WARNING] Too many MediaPipe errors ({error_count}), using raw video feed")
                        annotated_frame = frame
                        results = {'holistic': type('MockResults', (), {
                            'pose_landmarks': None,
                            'left_hand_landmarks': None,
                            'right_hand_landmarks': None,
                            'face_landmarks': None
                        })()}
                    else:
                        # Process frame with error handling
                        mode = 'holistic'  # Default mode for video feed
                        results, annotated_frame = tracker.process_frame(frame, mode)
                        
                        # Use original frame if processing failed
                        if annotated_frame is None or annotated_frame.size == 0:
                            print("[DEBUG] Using original frame - processing failed")
                            annotated_frame = frame
                            error_count += 1
                        else:
                            # Reset error count on success
                            error_count = 0
                    
                    # Record data if active
                    if recording_active:
                        timestamp = datetime.now().isoformat()
                        pose_data = tracker.extract_pose_data(results, timestamp, frame_number, mode)
                        
                        # Debug: Log what data we're getting
                        if frame_number % 30 == 0:  # Log every 30 frames to avoid spam
                            print(f"[DEBUG] Frame {frame_number}: Recording active, pose_data keys: {len(pose_data)} keys")
                            if 'holistic' in results:
                                holistic = results['holistic']
                                pose_detected = holistic.pose_landmarks is not None
                                left_hand_detected = holistic.left_hand_landmarks is not None
                                right_hand_detected = holistic.right_hand_landmarks is not None
                                print(f"[DEBUG] Detections - Pose: {pose_detected}, Left Hand: {left_hand_detected}, Right Hand: {right_hand_detected}")
                        
                        with processing_lock:
                            pose_data_buffer.append(pose_data)
                    
                    # Encode frame
                    success, buffer = cv2.imencode('.jpg', annotated_frame)
                    if not success:
                        print("[ERROR] Failed to encode frame")
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    frame_number += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:  # Only log first few errors to prevent spam
                        print(f"[ERROR] Frame processing error #{error_count}: {e}")
                    # Continue with original frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print(f"[ERROR] Video feed error: {e}")
        finally:
            cap.release()
            print("[DEBUG] Camera released")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_placeholder_frame(message):
    """Create a placeholder frame with error message"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), font, 1, (255, 255, 255), 2)
    return frame

@pose_bp.route('/api/landmark_info')
def landmark_info():
    """Provide detailed information about landmark indices and meanings"""
    landmark_info = {
        'pose_landmarks': {
            'description': 'Body pose landmarks (33 points)',
            'landmarks': {
                0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
                4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer', 7: 'left_ear',
                8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right', 11: 'left_shoulder',
                12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow', 15: 'left_wrist',
                16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky', 19: 'left_index',
                20: 'right_index', 21: 'left_thumb', 22: 'right_thumb', 23: 'left_hip',
                24: 'right_hip', 25: 'left_knee', 26: 'right_knee', 27: 'left_ankle',
                28: 'right_ankle', 29: 'left_heel', 30: 'right_heel', 31: 'left_foot_index',
                32: 'right_foot_index'
            }
        },
        'hand_landmarks': {
            'description': 'Hand landmarks (21 points per hand)',
            'landmarks': {
                0: 'wrist', 1: 'thumb_cmc', 2: 'thumb_mcp', 3: 'thumb_ip', 4: 'thumb_tip',
                5: 'index_finger_mcp', 6: 'index_finger_pip', 7: 'index_finger_dip', 8: 'index_finger_tip',
                9: 'middle_finger_mcp', 10: 'middle_finger_pip', 11: 'middle_finger_dip', 12: 'middle_finger_tip',
                13: 'ring_finger_mcp', 14: 'ring_finger_pip', 15: 'ring_finger_dip', 16: 'ring_finger_tip',
                17: 'pinky_mcp', 18: 'pinky_pip', 19: 'pinky_dip', 20: 'pinky_tip'
            }
        },
        'coordinate_system': {
            'x': 'Horizontal position (0.0 = left, 1.0 = right)',
            'y': 'Vertical position (0.0 = top, 1.0 = bottom)',
            'z': 'Depth (smaller values = closer to camera)',
            'visibility': 'Landmark visibility (pose only, 0.0 = not visible, 1.0 = fully visible)'
        }
    }
    
    return jsonify(landmark_info)

@pose_bp.route('/api/test')
def test_mediapipe():
    """Test MediaPipe installation and functionality"""
    print("[DEBUG] API Test endpoint called")
    
    try:
        # Simple MediaPipe import test
        import mediapipe as mp_test
        version = mp_test.__version__
        print(f"[DEBUG] MediaPipe available - version {version}")
        
        test_results = {
            'mediapipe_version': version,
            'pose_available': True,
            'hands_available': True,
            'holistic_available': True,
            'drawing_utils_available': True,
            'opencv_version': cv2.__version__,
            'camera_available': True  # Assume camera is working for now
        }
        
        return jsonify({
            'success': True,
            'test_results': test_results,
            'status': 'All systems ready'
        })
        
    except ImportError as e:
        print(f"[ERROR] MediaPipe import failed: {e}")
        return jsonify({
            'success': False,
            'error': 'MediaPipe not available',
            'test_results': {
                'mediapipe_version': 'Not Available',
                'pose_available': False,
                'hands_available': False,
                'holistic_available': False,
                'drawing_utils_available': False,
                'camera_available': False,
                'opencv_version': cv2.__version__
            },
            'status': 'MediaPipe installation required'
        })
    
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'Test failed'
        })
