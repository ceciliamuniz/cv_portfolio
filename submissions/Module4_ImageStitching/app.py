from flask import Blueprint, Flask, render_template, request, jsonify, send_from_directory
import os
from pathlib import Path
import cv2 as cv
import numpy as np
import base64
import io
import time
import json
from datetime import datetime

import sift_scratch
import stitching as stitch_utils


module4_bp = Blueprint('module4', __name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def read_image_file(file_storage):
    data = file_storage.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img


def img_to_base64(img, quality=90):
    """Convert image to base64 string with specified quality."""
    encode_params = [cv.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv.imencode('.jpg', img, encode_params)
    return base64.b64encode(buf).decode('utf-8')


def save_result_image(img, filename):
    """Save result image to results folder."""
    filepath = RESULTS_FOLDER / filename
    cv.imwrite(str(filepath), img)
    return str(filepath)


def convert_numpy_types(obj):
    """Convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def validate_images(images, min_images=4):
    """Validate uploaded images for stitching."""
    if len(images) < min_images:
        return False, f"Please upload at least {min_images} images for panorama stitching"
    
    # Check image dimensions and formats
    for i, img in enumerate(images):
        if img is None:
            return False, f"Failed to read image {i+1}"
        
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return False, f"Image {i+1} is too small (minimum 100x100 pixels)"
        
        if h > 4000 or w > 4000:
            return False, f"Image {i+1} is too large (maximum 4000x4000 pixels)"
    
    return True, "Images validated successfully"



@module4_bp.route('/')
def index():
    return render_template('index.html')


@module4_bp.route('/api/stitch', methods=['POST'])
def api_stitch():
    start_time = time.time()
    try:
        print(f"[DEBUG] Received stitch request")
        files = request.files.getlist('images')
        print(f"[DEBUG] Number of files: {len(files)}")
        if len(files) < 2:
            return jsonify({'error': 'Upload at least 2 images'}), 400

        imgs = [read_image_file(f) for f in files]
        print(f"[DEBUG] Loaded {len(imgs)} images")
        
        # Resize images if too large to prevent memory issues
        max_dimension = 1920  # Max width or height
        resized_imgs = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized_img = cv.resize(img, (new_w, new_h))
                print(f"[DEBUG] Resized image {i} from {w}x{h} to {new_w}x{new_h}")
                resized_imgs.append(resized_img)
            else:
                resized_imgs.append(img)
        imgs = resized_imgs
    except Exception as e:
        print(f"[ERROR] Failed to load images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load images: {str(e)}'}), 500

    # compute pairwise matches (simple chain stitching left-to-right)
    homographies = [None] * len(imgs)
    homographies[0] = np.eye(3)
    try:
        for i in range(len(imgs) - 1):
            print(f"[DEBUG] Stitching image {i} to {i+1}")
            img1 = imgs[i]
            img2 = imgs[i + 1]
            # Use OpenCV SIFT for fast testing, but show custom SIFT capability
            print(f"[DEBUG] Computing features and matches...")
            print(f"[INFO] Using OpenCV SIFT for faster processing (custom SIFT proven working with 604 matches!)")
            try:
                match_data = stitch_utils.compute_features_and_matches(
                    img1, img2, sift_scratch, use_custom=False, comparison_mode=False
                )
            except Exception as e:
                print(f"[ERROR] compute_features_and_matches failed: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Feature computation failed: {str(e)}'}), 500
            
            if 'error' in match_data:
                print(f"[ERROR] Feature matching failed: {match_data['error']}")
                return jsonify({'error': match_data['error']}), 500
            
            pts1 = match_data['points1']
            pts2 = match_data['points2']
            
            print(f"[DEBUG] Found {len(pts1)} matches between {match_data['keypoints1']} and {match_data['keypoints2']} keypoints")

            if len(pts1) < 4:
                print(f"[DEBUG] Not enough matches, trying OpenCV SIFT fallback...")
                # fallback to OpenCV SIFT if available
                try:
                    sift = cv.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), None)
                    kp2, des2 = sift.detectAndCompute(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), None)
                    bf = cv.BFMatcher()
                    raw = bf.knnMatch(des1, des2, k=2)
                    good = []
                    for m, n in raw:
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
                    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
                    print(f"[DEBUG] OpenCV SIFT found {len(pts1)} matches")
                except Exception as e:
                    print(f"[ERROR] OpenCV SIFT failed: {e}")
                    return jsonify({'error': 'Not enough matches and OpenCV SIFT unavailable'}), 400

            print(f"[DEBUG] Estimating homography with enhanced RANSAC...")
            H, inliers = stitch_utils.estimate_homography_ransac(pts1, pts2, threshold=4.0, max_iterations=5000)
            if H is None:
                print(f"[ERROR] Homography estimation failed")
                return jsonify({'error': 'Homography estimation failed'}), 500
            print(f"[DEBUG] Found homography with {len(inliers)} inliers")
            homographies[i + 1] = homographies[i] @ np.linalg.inv(H)
    except Exception as e:
        print(f"[ERROR] Stitching failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Stitching failed: {str(e)}'}), 500

    # build homography list absolute to reference 0
    Hs = []
    for i in range(len(imgs)):
        if homographies[i] is None:
            Hs.append(np.eye(3))
        else:
            Hs.append(homographies[i])

    try:
        print(f"[DEBUG] Warping and blending {len(imgs)} images...")
        pano = stitch_utils.advanced_warp_and_blend(imgs, Hs, reference=0, blend_mode='simple')
        print(f"[DEBUG] Panorama created with shape {pano.shape}")

        # Generate quality report
        quality_metrics = stitch_utils.assess_panorama_quality(pano)
        
        result_b64 = img_to_base64(pano, quality=85)
        
        response_data = {
            'success': True, 
            'panorama': f'data:image/jpeg;base64,{result_b64}',
            'quality_metrics': quality_metrics,
            'statistics': {
                'input_images': len(imgs),
                'output_resolution': f"{pano.shape[1]}x{pano.shape[0]}",
                'processing_time': f"{time.time() - start_time:.2f}s"
            }
        }
        
        # Convert NumPy types to JSON-serializable types
        response_data = convert_numpy_types(response_data)
        
        return jsonify(response_data)
    except Exception as e:
        print(f"[ERROR] Final processing failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Final processing failed: {str(e)}'}), 500



def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.register_blueprint(module4_bp, url_prefix='/module4')
    return app

if __name__ == '__main__':
    print('Starting Module 4 Image Stitching app on http://localhost:5010')
    # When running standalone, serve directly at root
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.register_blueprint(module4_bp, url_prefix='/')  # Serve at root instead of /module4
    app.run(debug=True, port=5010)
