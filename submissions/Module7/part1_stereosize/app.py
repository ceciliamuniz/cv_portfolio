"""
Module 7 Part 1: Object Size Estimation using Calibrated Stereo
Flask blueprint for stereo size estimation
"""
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, render_template


stereo_bp = Blueprint('stereo_bp', __name__, template_folder='../../../templates')

def load_calibration_file(calibration_file):
    """Load stereo calibration from uploaded file (NPZ, JSON, XML, YAML)"""
    try:
        filename = calibration_file.filename.lower()
        file_content = calibration_file.read()
        
        if filename.endswith('.npz'):
            # NumPy archive format (most common for OpenCV stereo calibration)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                data = np.load(tmp_file.name)
                calibration = {
                    'camera_matrix_left': data.get('camera_matrix_left', DEFAULT_CALIBRATION['camera_matrix_left']),
                    'camera_matrix_right': data.get('camera_matrix_right', DEFAULT_CALIBRATION['camera_matrix_right']),
                    'dist_coeffs_left': data.get('dist_coeffs_left', DEFAULT_CALIBRATION['dist_coeffs_left']),
                    'dist_coeffs_right': data.get('dist_coeffs_right', DEFAULT_CALIBRATION['dist_coeffs_right']),
                    'rotation_matrix': data.get('rotation_matrix', DEFAULT_CALIBRATION['rotation_matrix']),
                    'translation_vector': data.get('translation_vector', DEFAULT_CALIBRATION['translation_vector'])
                }
                os.unlink(tmp_file.name)
                
        elif filename.endswith('.json'):
            # JSON format
            import json
            data = json.loads(file_content.decode('utf-8'))
            calibration = {
                'camera_matrix_left': np.array(data.get('camera_matrix_left', DEFAULT_CALIBRATION['camera_matrix_left'].tolist())),
                'camera_matrix_right': np.array(data.get('camera_matrix_right', DEFAULT_CALIBRATION['camera_matrix_right'].tolist())),
                'dist_coeffs_left': np.array(data.get('dist_coeffs_left', DEFAULT_CALIBRATION['dist_coeffs_left'].tolist())),
                'dist_coeffs_right': np.array(data.get('dist_coeffs_right', DEFAULT_CALIBRATION['dist_coeffs_right'].tolist())),
                'rotation_matrix': np.array(data.get('rotation_matrix', DEFAULT_CALIBRATION['rotation_matrix'].tolist())),
                'translation_vector': np.array(data.get('translation_vector', DEFAULT_CALIBRATION['translation_vector'].tolist()))
            }
        else:
            print(f"[WARNING] Unsupported calibration file format: {filename}")
            return DEFAULT_CALIBRATION
            
        # Extract derived parameters
        calibration['focal_length_x'] = calibration['camera_matrix_left'][0, 0]
        calibration['focal_length_y'] = calibration['camera_matrix_left'][1, 1]
        calibration['cx'] = calibration['camera_matrix_left'][0, 2]
        calibration['cy'] = calibration['camera_matrix_left'][1, 2]
        calibration['baseline'] = abs(calibration['translation_vector'][0])  # Baseline magnitude
        
        print(f"[DEBUG] Loaded calibration: fx={calibration['focal_length_x']:.1f}, baseline={calibration['baseline']:.1f}mm")
        return calibration
        
    except Exception as e:
        print(f"[ERROR] Failed to load calibration file: {e}")
        return DEFAULT_CALIBRATION

@stereo_bp.route('/test', methods=['GET'])
def stereo_test():
    return 'Module 7 blueprint is registered and working!'

def rectify_stereo_pair(left_img, right_img, calibration):
    """Rectify stereo image pair using calibration parameters"""
    try:
        # Get image dimensions
        h, w = left_img.shape[:2]
        
        # Compute rectification maps
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            calibration['camera_matrix_left'], calibration['dist_coeffs_left'],
            calibration['camera_matrix_right'], calibration['dist_coeffs_right'],
            (w, h), calibration['rotation_matrix'], calibration['translation_vector'],
            alpha=0  # 0 = crop to valid pixels, 1 = keep all pixels
        )
        
        # Generate rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            calibration['camera_matrix_left'], calibration['dist_coeffs_left'],
            R1, P1, (w, h), cv2.CV_16SC2
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            calibration['camera_matrix_right'], calibration['dist_coeffs_right'],
            R2, P2, (w, h), cv2.CV_16SC2
        )
        
        # Apply rectification
        left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
        
        print(f"[DEBUG] Stereo rectification completed. ROI1: {roi1}, ROI2: {roi2}")
        
        # Update calibration with rectified parameters
        rectified_calibration = calibration.copy()
        rectified_calibration['P1'] = P1
        rectified_calibration['P2'] = P2
        rectified_calibration['Q'] = Q
        rectified_calibration['roi1'] = roi1
        rectified_calibration['roi2'] = roi2
        
        return left_rectified, right_rectified, rectified_calibration
        
    except Exception as e:
        print(f"[WARNING] Rectification failed: {e}. Using original images.")
        return left_img, right_img, calibration

# Hardcoded stereo calibration using your phone's exact camera matrix from Module 1
# Camera matrix: [[640.84, 0, 294.25], [0, 648.80, 349.31], [0, 0, 1]]
DEFAULT_CALIBRATION = {
    # Left camera intrinsics (your exact phone calibration from Module 1)
    'camera_matrix_left': np.array([
        [640.8396063, 0, 294.24936703],      # Fx, 0, Ox
        [0, 648.80269311, 349.31369175],    # 0, Fy, Oy  
        [0, 0, 1]                           # 0, 0, 1
    ]),
    # Right camera intrinsics (same phone, assuming identical calibration)
    'camera_matrix_right': np.array([
        [640.8396063, 0, 294.24936703],      # Same as left camera
        [0, 648.80269311, 349.31369175],    
        [0, 0, 1]
    ]),
    # Minimal distortion coefficients for phone cameras
    'dist_coeffs_left': np.array([0.05, -0.1, 0, 0, 0]),    # k1, k2, p1, p2, k3
    'dist_coeffs_right': np.array([0.05, -0.1, 0, 0, 0]),   # Assuming similar distortion
    # Stereo extrinsics (camera separation and alignment)
    'rotation_matrix': np.eye(3),                            # Parallel stereo setup
    'translation_vector': np.array([65.0, 0, 0]),           # 6.5cm baseline (typical)
    
    # Direct parameters (extracted from your Module 1 calibration)
    'focal_length_x': 640.8396063,    # Fx from your phone calibration
    'focal_length_y': 648.80269311,   # Fy from your phone calibration
    'baseline': 65.0,                 # Stereo baseline in mm
    'cx': 294.24936703,               # Ox - principal point x
    'cy': 349.31369175                # Oy - principal point y
}

def compute_disparity(left_img, right_img):
    """Compute stereo disparity map with improved parameters and preprocessing"""
    print(f"[DEBUG] Computing disparity for images: left={left_img.shape}, right={right_img.shape}")
    
    # Ensure images are the same size
    if left_img.shape != right_img.shape:
        raise ValueError(f"Image size mismatch: left={left_img.shape}, right={right_img.shape}")
    
    # Preprocess images for better matching
    # Apply slight Gaussian blur to reduce noise
    left_smooth = cv2.GaussianBlur(left_img, (3, 3), 0)
    right_smooth = cv2.GaussianBlur(right_img, (3, 3), 0)
    
    # Create stereo matcher with optimized parameters
    stereo = cv2.StereoBM_create(numDisparities=80, blockSize=21)  # Increased for better accuracy
    
    # Fine-tune parameters for better matching
    stereo.setMinDisparity(0)
    stereo.setSpeckleWindowSize(200)   # Larger window for better speckle filtering
    stereo.setSpeckleRange(16)         # Tighter range for more consistent results
    stereo.setDisp12MaxDiff(5)         # Allow small differences for robustness
    stereo.setUniquenessRatio(15)      # Higher ratio for more confident matches
    stereo.setTextureThreshold(20)     # Higher threshold for textured regions
    
    # Compute disparity
    disparity = stereo.compute(left_smooth, right_smooth).astype(np.float32) / 16.0
    
    print(f"[DEBUG] Raw disparity range: min={np.min(disparity)}, max={np.max(disparity)}")
    
    # Post-process disparity map
    # Apply median filter to reduce noise
    disparity_filtered = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
    
    print(f"[DEBUG] Filtered disparity range: min={np.min(disparity_filtered)}, max={np.max(disparity_filtered)}")
    return disparity_filtered

def estimate_depth(disparity, focal_length_x, baseline):
    """Estimate depth using stereo disparity and calibrated parameters"""
    # Filter out invalid disparities (typically <= 0 or too large)
    valid_disparity = np.where((disparity > 1.0) & (disparity < 100), disparity, np.nan)
    
    # Calculate depth using stereo formula: Z = (f * B) / d
    # focal_length_x in pixels, baseline in mm, result in mm
    Z = (focal_length_x * baseline) / valid_disparity
    
    # Clip unrealistic depths (keep objects between 30cm and 10m)
    Z = np.clip(Z, 300, 10000)  # 300mm to 10000mm
    
    print(f"[DEBUG] Depth statistics: min={np.nanmin(Z):.1f}mm, max={np.nanmax(Z):.1f}mm, mean={np.nanmean(Z):.1f}mm")
    return Z

def create_automatic_mask(img):
    """Create automatic mask by detecting prominent objects in the image"""
    print(f"[DEBUG] Creating automatic mask for image shape: {img.shape}")
    
    # Method 1: Edge-based detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds for more edges
    
    # Method 2: Adaptive threshold for objects with different lighting
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both methods
    combined = cv2.bitwise_or(edges, adaptive_thresh)
    
    # Morphological operations to clean up and fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill holes
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel_fill, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from significant contours (not just the largest)
    mask = np.zeros(img.shape, dtype=np.uint8)
    if contours:
        # Sort by area and take contours that are significant
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Include multiple objects if they're substantial
        total_area = img.shape[0] * img.shape[1]
        for i, cnt in enumerate(contours[:5]):  # Max 5 objects
            area = cv2.contourArea(cnt)
            area_ratio = area / total_area
            
            # Include objects that are at least 0.5% of image area
            if area_ratio > 0.005 and area > 1000:
                cv2.fillPoly(mask, [cnt], 255)
                print(f"[DEBUG] Added object {i+1}: area={area:.0f} pixels ({area_ratio*100:.1f}% of image)")
        
        print(f"[DEBUG] Created mask with {np.sum(mask > 0)} white pixels")
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    return mask

def detect_shape_and_measure(img, mask, Z_measurement, Z_stereo, calibration, mode='stereo'):
    """Detect objects and measure their real-world dimensions using calibrated stereo vision"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    
    fx = calibration['focal_length_x']  # pixels
    fy = calibration['focal_length_y']  # pixels
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Increased minimum area for meaningful objects
            continue
            
        # Get object region and average depth
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract depth values for this object (ignore NaN values)
        z_measurement_region = Z_measurement[y:y+h, x:x+w]
        z_stereo_region = Z_stereo[y:y+h, x:x+w]
        
        # For stereo depths, filter NaN values
        valid_stereo_depths = z_stereo_region[~np.isnan(z_stereo_region)]
        
        if len(valid_stereo_depths) < 10:  # Need sufficient valid stereo depth points
            continue
            
        z_measurement_mean = np.mean(z_measurement_region[~np.isnan(z_measurement_region)])  # Depth used for measurement
        z_stereo_mean = np.mean(valid_stereo_depths)  # Stereo-calculated depth
        
        print(f"[DEBUG] Object measurement depth: {z_measurement_mean:.1f}mm, stereo depth: {z_stereo_mean:.1f}mm")
        
        shape = "polygon"
        dims = {}
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        
        if len(approx) == 4:
            shape = "rectangle"
            # Real-world size formula: real_size = (pixel_size × depth) / focal_length
            real_width = (w * z_mean) / fx   # mm
            real_height = (h * z_mean) / fy  # mm
            dims['width'] = f"{real_width:.1f} mm"
            dims['height'] = f"{real_height:.1f} mm"
            dims['area'] = f"{(real_width * real_height):.1f} mm²"
            
        elif len(approx) > 6:
            shape = "circle"
            (cx_obj, cy_obj), radius = cv2.minEnclosingCircle(cnt)
            # Real diameter in mm
            real_diameter = (2 * radius * z_mean) / fx
            dims['diameter'] = f"{real_diameter:.1f} mm"
            dims['area'] = f"{(np.pi * (real_diameter/2)**2):.1f} mm²"
            
        else:
            pts = approx.reshape(-1, 2)
            edges = []
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i+1)%len(pts)]
                pixel_dist = np.linalg.norm(p1 - p2)
                # Real edge length in mm
                real_edge = (pixel_dist * z_mean) / fx
                edges.append(f"{real_edge:.1f} mm")
            dims['edges'] = edges
            
        # Add depth information and comparison
        if mode == 'manual':
            dims['measurement_depth'] = f"{z_measurement_mean:.1f} mm (manual input)"
            dims['stereo_depth'] = f"{z_stereo_mean:.1f} mm (calculated)"
            depth_difference = abs(z_measurement_mean - z_stereo_mean)
            depth_error_percent = (depth_difference / z_stereo_mean) * 100
            dims['depth_comparison'] = f"Difference: {depth_difference:.1f} mm ({depth_error_percent:.1f}%)"
        else:
            dims['depth'] = f"{z_stereo_mean:.1f} mm (stereo calculated)"
            
        dims['pixel_area'] = f"{area} pixels"
        
        results.append({
            'shape': shape, 
            'dimensions': dims,
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
        })
        
    return results

@stereo_bp.route('/', methods=['GET'])
def stereo_page():
    return render_template('module7.html')

@stereo_bp.route('/api/stereo/estimate', methods=['POST'])
def stereo_estimate():
    try:
        print(f"[DEBUG] Request method: {request.method}")
        print(f"[DEBUG] Content type: {request.content_type}")
        print(f"[DEBUG] Files in request: {list(request.files.keys())}")
        
        left_file = request.files.get('left')
        right_file = request.files.get('right')
        calibration_file = request.files.get('calibration')
        manual_distance_str = request.form.get('manual_distance')
        
        # Parse manual distance if provided
        manual_distance = None
        if manual_distance_str and manual_distance_str.strip():
            try:
                manual_distance = float(manual_distance_str)
                print(f"[DEBUG] Manual distance provided: {manual_distance}mm")
            except ValueError:
                print(f"[WARNING] Invalid manual distance: {manual_distance_str}")
    
        if not left_file or not right_file:
            print(f"[ERROR] Missing files - left: {left_file}, right: {right_file}")
            return jsonify({'error': 'Both left and right images are required'}), 400
            
        # Load calibration (use uploaded file or default)
        if calibration_file and calibration_file.filename:
            print(f"[DEBUG] Loading calibration from: {calibration_file.filename}")
            calibration = load_calibration_file(calibration_file)
        else:
            print(f"[DEBUG] Using default calibration parameters")
            calibration = DEFAULT_CALIBRATION
            
        print(f"[DEBUG] Received files: left={left_file.filename if left_file else 'None'}, right={right_file.filename if right_file else 'None'}")
        print(f"[DEBUG] File sizes: left={len(left_file.read()) if left_file else 0}, right={len(right_file.read()) if right_file else 0}")
        
        # Reset file pointers after reading for size
        if left_file:
            left_file.seek(0)
        if right_file:
            right_file.seek(0)
        left_img = cv2.imdecode(np.frombuffer(left_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imdecode(np.frombuffer(right_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Validate images were decoded successfully
        if left_img is None:
            return jsonify({'error': 'Failed to decode left image. Please use JPG, PNG, or other standard image format.'}), 400
        if right_img is None:
            return jsonify({'error': 'Failed to decode right image. Please use JPG, PNG, or other standard image format.'}), 400
            
        print(f"[DEBUG] Original image shapes: left={left_img.shape}, right={right_img.shape}")
        
        # Ensure both images have the same dimensions (required for stereo matching)
        if left_img.shape != right_img.shape:
            print(f"[DEBUG] Resizing images to match dimensions")
            # Use the smaller dimensions to avoid upscaling
            target_height = min(left_img.shape[0], right_img.shape[0])
            target_width = min(left_img.shape[1], right_img.shape[1])
            
            left_img = cv2.resize(left_img, (target_width, target_height))
            right_img = cv2.resize(right_img, (target_width, target_height))
            print(f"[DEBUG] Resized to: left={left_img.shape}, right={right_img.shape}")
        else:
            print(f"[DEBUG] Images already have matching shapes: {left_img.shape}")
        
        # Rectify stereo pair for accurate disparity computation
        left_rect, right_rect, rect_calibration = rectify_stereo_pair(left_img, right_img, calibration)
        
        # Create automatic mask using edge detection and largest contour
        mask = create_automatic_mask(left_rect)
        print(f"[DEBUG] Using automatic mask, shape={mask.shape}")
        
        # Compute disparity on rectified images
        disparity = compute_disparity(left_rect, right_rect)
        Z_stereo = estimate_depth(disparity, rect_calibration['focal_length_x'], rect_calibration['baseline'])
        
        # Use manual distance if provided, otherwise use stereo-calculated distance
        if manual_distance is not None:
            print(f"[DEBUG] Using manual distance: {manual_distance}mm for measurements")
            Z_measurement = np.full_like(Z_stereo, manual_distance)  # Create array filled with manual distance
            measurement_mode = 'manual'
        else:
            print(f"[DEBUG] Using stereo-calculated distance for measurements")
            Z_measurement = Z_stereo
            measurement_mode = 'stereo'
            
        results = detect_shape_and_measure(left_rect, mask, Z_measurement, Z_stereo, rect_calibration, measurement_mode)
        
        response_data = {
            'success': True,
            'results': results,
            'processing_info': {
                'automatic_detection': True,
                'calibration_used': {
                    'source': 'uploaded' if calibration_file and calibration_file.filename else 'default',
                    'focal_length_x': rect_calibration['focal_length_x'],
                    'baseline': rect_calibration['baseline'],
                    'rectified': True
                },
                'measurement_mode': measurement_mode,
                'manual_distance': manual_distance if manual_distance else None,
                'image_shapes': {
                    'left': left_img.shape,
                    'right': right_img.shape,
                    'mask': mask.shape
                }
            }
        }
        print(f"[DEBUG] Returning success response with {len(results)} objects detected")
        return jsonify(response_data)
    except Exception as e:
        print(f"[ERROR] Exception in stereo estimation: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'error_type': type(e).__name__
        }), 500
