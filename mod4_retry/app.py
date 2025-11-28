"""
Module 4: Image Stitching - Clean Implementation
Two assignments:
1. Simple OpenCV stitching 
2. Custom SIFT from scratch with RANSAC

Author: Cecilia Muniz Siqueira
"""
from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import base64
import time
from pathlib import Path

# Import custom SIFT implementation
try:
    import sift_scratch
    CUSTOM_SIFT_AVAILABLE = True
    print("[INFO] Custom SIFT implementation loaded successfully")
except ImportError as e:
    print(f"[WARN] Custom SIFT not available: {e}")
    CUSTOM_SIFT_AVAILABLE = False

app = Flask(__name__)

class ImageStitching:
    """Simple image stitching class for Assignment 2"""
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        self.use_custom_sift = CUSTOM_SIFT_AVAILABLE
        
        # Initialize OpenCV SIFT for comparison/fallback
        try:
            self.opencv_sift = cv.SIFT_create()
        except AttributeError:
            raise Exception("OpenCV SIFT not available")

    def registration(self, img1, img2):
        """Find homography using custom SIFT or OpenCV fallback"""
        if self.use_custom_sift:
            try:
                # Convert to grayscale
                gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
                gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
                
                # Use custom SIFT
                pyramid1 = sift_scratch.build_gaussian_pyramid(gray1)
                pyramid2 = sift_scratch.build_gaussian_pyramid(gray2)
                
                dog1 = sift_scratch.compute_dog(pyramid1)
                dog2 = sift_scratch.compute_dog(pyramid2)
                
                kps1 = sift_scratch.detect_keypoints(dog1)
                kps2 = sift_scratch.detect_keypoints(dog2)
                
                if len(kps1) < 4 or len(kps2) < 4:
                    raise Exception("Insufficient keypoints")
                
                kps1_oriented = sift_scratch.assign_orientations(kps1, pyramid1)
                kps2_oriented = sift_scratch.assign_orientations(kps2, pyramid2)
                
                desc1_custom = sift_scratch.compute_descriptors(kps1_oriented, pyramid1)
                desc2_custom = sift_scratch.compute_descriptors(kps2_oriented, pyramid2)
                
                if len(desc1_custom) < 4 or len(desc2_custom) < 4:
                    raise Exception("Insufficient descriptors")
                
                matches_custom = sift_scratch.match_descriptors(desc1_custom, desc2_custom)
                
                if len(matches_custom) >= self.min_match:
                    # Extract point correspondences
                    src_points = []
                    dst_points = []
                    
                    for match_idx, (i1, i2, dist) in enumerate(matches_custom):
                        if i1 < len(desc1_custom) and i2 < len(desc2_custom):
                            src_points.append(desc2_custom[i2]['pt'])
                            dst_points.append(desc1_custom[i1]['pt'])
                    
                    if len(src_points) >= 4:
                        H_custom, inliers = sift_scratch.enhanced_ransac_homography(
                            matches_custom, src_points, dst_points)
                        
                        if H_custom is not None:
                            print(f"[SUCCESS] Custom SIFT: {len(inliers)} inliers from {len(matches_custom)} matches")
                            return H_custom
                
                print("[FALLBACK] Custom SIFT insufficient matches, using OpenCV")
                
            except Exception as e:
                print(f"[ERROR] Custom SIFT failed: {e}")
        
        # OpenCV SIFT fallback
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        kp1, des1 = self.opencv_sift.detectAndCompute(gray1, None)
        kp2, des2 = self.opencv_sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None
            
        matcher = cv.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m1, m2 = match_pair
                if m1.distance < self.ratio * m2.distance:
                    good_points.append((m1.trainIdx, m1.queryIdx))
        
        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv.findHomography(image2_kp, image1_kp, cv.RANSAC, 5.0)
            
            if H is not None:
                print(f"[SUCCESS] OpenCV SIFT: {np.sum(status)} inliers")
            
            return H
        
        return None

    def blending(self, img1, img2):
        """Simple blending of two images"""
        H = self.registration(img1, img2)
        if H is None:
            return None
            
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        width_panorama = width_img1 + width_img2
        height_panorama = height_img1

        # Warp second image
        panorama = np.zeros((height_panorama, width_panorama, 3), dtype=np.uint8)
        panorama[0:height_img1, 0:width_img1] = img1
        
        warped = cv.warpPerspective(img2, H, (width_panorama, height_panorama))
        
        # Simple blending in overlap region
        for i in range(height_panorama):
            for j in range(width_panorama):
                if warped[i, j].any() and panorama[i, j].any():
                    panorama[i, j] = (panorama[i, j] * 0.5 + warped[i, j] * 0.5).astype(np.uint8)
                elif warped[i, j].any():
                    panorama[i, j] = warped[i, j]

        # Crop to content
        gray = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(c)
            panorama = panorama[y:y+h, x:x+w]
        
        return panorama

@app.route('/')
def index():
    """Module 4 homepage"""
    return render_template('index.html')

@app.route('/api/stitch-simple', methods=['POST'])
def api_stitch_simple():
    """Assignment 1: Simple OpenCV stitching"""
    try:
        start_time = time.time()
        files = request.files.getlist('images')
        
        if len(files) < 2:
            return jsonify({'error': 'Upload at least 2 images'}), 400
        
        # Read images
        imgs = []
        for file in files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': f'Invalid image file: {file.filename}'}), 400
            imgs.append(img)
        
        # Resize if too large
        max_dim = 800
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                imgs[i] = cv.resize(img, (new_w, new_h))
        
        # Use OpenCV stitcher
        stitcher = cv.Stitcher_create()
        status, result = stitcher.stitch(imgs)
        
        if status == cv.Stitcher_OK:
            processing_time = time.time() - start_time
            
            # Convert to base64
            _, buffer = cv.imencode('.jpg', result, [cv.IMWRITE_JPEG_QUALITY, 90])
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'panorama': f'data:image/jpeg;base64,{result_base64}',
                'statistics': {
                    'input_images': len(imgs),
                    'output_resolution': f'{result.shape[1]}x{result.shape[0]}',
                    'processing_time': f'{processing_time:.2f}s',
                    'method': 'OpenCV Stitcher (Assignment 1)'
                }
            })
        else:
            error_messages = {
                cv.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                cv.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
            }
            error_msg = error_messages.get(status, f"Stitching error (code: {status})")
            return jsonify({'error': f'Stitching failed: {error_msg}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Stitching failed: {str(e)}'}), 500

@app.route('/api/stitch-custom', methods=['POST'])
def api_stitch_custom():
    """Assignment 2: Custom SIFT with RANSAC"""
    try:
        start_time = time.time()
        files = request.files.getlist('images')
        
        if len(files) < 4:
            return jsonify({'error': 'Upload at least 4 images for custom SIFT'}), 400
        
        # Read images
        imgs = []
        for file in files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': f'Invalid image file: {file.filename}'}), 400
            imgs.append(img)
        
        # Resize if too large
        max_dim = 600  # Smaller for custom SIFT (slower)
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                imgs[i] = cv.resize(img, (new_w, new_h))
        
        # Use custom stitching
        stitcher = ImageStitching()
        
        # Sequential stitching
        result = imgs[0]
        for i in range(1, len(imgs)):
            temp_result = stitcher.blending(result, imgs[i])
            if temp_result is not None:
                result = temp_result
            else:
                print(f"[WARN] Failed to stitch image {i+1}")
                break
        
        if result is not None:
            processing_time = time.time() - start_time
            
            # Convert to base64
            _, buffer = cv.imencode('.jpg', result, [cv.IMWRITE_JPEG_QUALITY, 90])
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'panorama': f'data:image/jpeg;base64,{result_base64}',
                'statistics': {
                    'input_images': len(imgs),
                    'output_resolution': f'{result.shape[1]}x{result.shape[0]}',
                    'processing_time': f'{processing_time:.2f}s',
                    'method': 'Custom SIFT + RANSAC (Assignment 2)',
                    'sift_implementation': 'Custom SIFT from scratch' if stitcher.use_custom_sift else 'OpenCV SIFT fallback'
                }
            })
        else:
            return jsonify({'error': 'Stitching failed - insufficient feature matches'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Stitching failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Module 4 - Image Stitching")
    print("📍 Assignment 1: /api/stitch-simple (OpenCV)")
    print("📍 Assignment 2: /api/stitch-custom (Custom SIFT)")
    print("🌐 Visit: http://localhost:5004")
    app.run(debug=True, port=5004)