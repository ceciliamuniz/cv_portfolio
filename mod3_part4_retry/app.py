"""
Module 3 Part 4: ArUco Marker Detection & Segmentation - Clean Implementation
Detects ArUco markers and performs GrabCut segmentation on detected objects.

Author: Cecilia Muniz Siqueira
"""
from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import base64
import time
from pathlib import Path

# Import ArUco segmentation functions
try:
    from aruco_segmentation import ArucoSegmentation, IMAGE_FILE_NAMES
    ARUCO_AVAILABLE = True
    print("[INFO] ArUco segmentation module loaded successfully")
except ImportError as e:
    print(f"[WARN] ArUco segmentation not available: {e}")
    ARUCO_AVAILABLE = False
    ArucoSegmentation = None
    IMAGE_FILE_NAMES = []

app = Flask(__name__)

def img_to_base64(img, quality=90):
    """Convert image to base64 string"""
    encode_params = [cv.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv.imencode('.jpg', img, encode_params)
    return base64.b64encode(buf).decode('utf-8')

def process_aruco_detection(image, dictionary_type='DICT_6X6_1000'):
    """Process ArUco detection with segmentation"""
    if not ARUCO_AVAILABLE:
        raise Exception("ArUco segmentation module not available")
    
    # Convert dictionary name to OpenCV constant
    dictionary_map = {
        'DICT_4X4_50': cv.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv.aruco.DICT_6X6_1000,
    }
    
    dict_id = dictionary_map.get(dictionary_type, cv.aruco.DICT_6X6_1000)
    
    # Create ArUco segmentation instance
    segmenter = ArucoSegmentation(aruco_dict_type=dict_id)
    
    # Use the simplified grabcut segmentation method
    segmented_image, mask, metadata = segmenter.simplified_grabcut_segmentation(image)
    
    # Get marker detection info
    corners, ids = segmenter.detect_markers_comprehensive(image)
    
    # Create result dictionary
    result = {
        'markers_detected': len(ids) if ids is not None else 0,
        'annotated_image': segmented_image,
        'segmented_objects': [],
        'metadata': metadata
    }
    
    # If we have a valid mask, create segmented objects
    if mask is not None:
        result['segmented_objects'].append({
            'marker_id': 0,
            'mask': mask
        })
    
    return result

@app.route('/')
def index():
    """Module 3 Part 4 homepage"""
    return render_template('index.html')

@app.route('/api/detect-aruco', methods=['POST'])
def api_detect_aruco():
    """ArUco detection and segmentation API"""
    try:
        start_time = time.time()
        
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get dictionary parameter
        dictionary_type = request.form.get('dictionary', 'DICT_6X6_1000')
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Resize if too large
        max_dimension = 1200
        h, w = image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv.resize(image, (new_w, new_h))
        
        # Process ArUco detection
        print(f"[INFO] Processing ArUco detection with {dictionary_type}")
        result = process_aruco_detection(image, dictionary_type)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'processing_time': f'{processing_time:.2f}s',
            'dictionary_used': dictionary_type,
            'markers_detected': result.get('markers_detected', 0),
            'segmentation_results': []
        }
        
        # Convert result images to base64
        if 'annotated_image' in result and result['annotated_image'] is not None:
            response_data['annotated_image'] = f"data:image/jpeg;base64,{img_to_base64(result['annotated_image'])}"
        
        if 'segmented_objects' in result:
            for i, obj in enumerate(result['segmented_objects']):
                if 'mask' in obj and obj['mask'] is not None:
                    # Create masked object image
                    masked_img = cv.bitwise_and(image, image, mask=obj['mask'])
                    response_data['segmentation_results'].append({
                        'object_id': i,
                        'marker_id': obj.get('marker_id', i),
                        'segmented_object': f"data:image/jpeg;base64,{img_to_base64(masked_img)}",
                        'mask': f"data:image/jpeg;base64,{img_to_base64(cv.cvtColor(obj['mask'], cv.COLOR_GRAY2BGR))}"
                    })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] ArUco detection failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/process-hardcoded', methods=['POST'])
def api_process_hardcoded():
    """Process the 10 hardcoded images from the images folder"""
    try:
        start_time = time.time()
        
        if not ARUCO_AVAILABLE:
            return jsonify({'error': 'ArUco segmentation not available'}), 500
        
        # Get dictionary parameter
        dictionary_type = request.form.get('dictionary', 'DICT_6X6_1000')
        
        # Convert dictionary name to OpenCV constant
        dictionary_map = {
            'DICT_4X4_50': cv.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv.aruco.DICT_6X6_1000,
        }
        
        dict_id = dictionary_map.get(dictionary_type, cv.aruco.DICT_6X6_1000)
        segmenter = ArucoSegmentation(aruco_dict_type=dict_id)
        
        # Process hardcoded images
        images_dir = Path('images')
        batch_results = []
        total_markers = 0
        
        for i, filename in enumerate(IMAGE_FILE_NAMES, 1):
            image_path = images_dir / filename
            
            if not image_path.exists():
                batch_results.append({
                    'filename': filename,
                    'success': False,
                    'error': f'File not found: {filename}'
                })
                continue
            
            # Load and process image
            image = cv.imread(str(image_path))
            if image is None:
                batch_results.append({
                    'filename': filename,
                    'success': False,
                    'error': f'Could not load image: {filename}'
                })
                continue
            
            # Resize if too large
            max_dimension = 800
            h, w = image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv.resize(image, (new_w, new_h))
            
            try:
                # Process with ArUco segmentation
                segmented_image, mask, metadata = segmenter.simplified_grabcut_segmentation(image)
                corners, ids = segmenter.detect_markers_comprehensive(image)
                
                markers_found = len(ids) if ids is not None else 0
                total_markers += markers_found
                
                # Prepare result
                image_result = {
                    'filename': filename,
                    'markers_detected': markers_found,
                    'success': True,
                    'annotated_image': f"data:image/jpeg;base64,{img_to_base64(segmented_image)}"
                }
                
                # Add mask if available
                if mask is not None:
                    image_result['mask'] = f"data:image/jpeg;base64,{img_to_base64(cv.cvtColor(mask, cv.COLOR_GRAY2BGR))}"
                
                batch_results.append(image_result)
                print(f"[{i}/10] {filename}: {markers_found} markers detected")
                
            except Exception as e:
                batch_results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'processing_time': f'{processing_time:.2f}s',
            'dictionary_used': dictionary_type,
            'total_images': len(IMAGE_FILE_NAMES),
            'processed_images': len([r for r in batch_results if r['success']]),
            'total_markers_detected': total_markers,
            'results': batch_results
        })
        
    except Exception as e:
        print(f"[ERROR] Hardcoded processing failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Hardcoded processing failed: {str(e)}'}), 500

@app.route('/api/batch-process', methods=['POST'])
def api_batch_process():
    """Batch processing multiple images"""
    try:
        start_time = time.time()
        
        files = request.files.getlist('images')
        if len(files) < 1:
            return jsonify({'error': 'No images uploaded'}), 400
        
        dictionary_type = request.form.get('dictionary', 'DICT_6X6_1000')
        
        batch_results = []
        total_markers = 0
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            # Read image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Resize if needed
            max_dimension = 1000  # Smaller for batch processing
            h, w = image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv.resize(image, (new_w, new_h))
            
            # Process image
            try:
                result = process_aruco_detection(image, dictionary_type)
                markers_found = result.get('markers_detected', 0)
                total_markers += markers_found
                
                # Prepare result for this image
                image_result = {
                    'filename': file.filename,
                    'markers_detected': markers_found,
                    'success': True
                }
                
                # Add annotated image if available
                if 'annotated_image' in result and result['annotated_image'] is not None:
                    image_result['annotated_image'] = f"data:image/jpeg;base64,{img_to_base64(result['annotated_image'])}"
                
                batch_results.append(image_result)
                
            except Exception as e:
                batch_results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'processing_time': f'{processing_time:.2f}s',
            'dictionary_used': dictionary_type,
            'total_images': len(files),
            'processed_images': len(batch_results),
            'total_markers_detected': total_markers,
            'results': batch_results
        })
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Module 3 Part 4 - ArUco Detection")
    print("📍 Single image: /api/detect-aruco")
    print("📍 Batch processing: /api/batch-process")
    print("📍 Hardcoded 10 images: /api/process-hardcoded")
    print("🌐 Visit: http://localhost:5003")
    app.run(debug=True, port=5003)