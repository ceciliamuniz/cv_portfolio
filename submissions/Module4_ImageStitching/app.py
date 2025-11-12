from flask import Blueprint, Flask, render_template, request, jsonify
import os
from pathlib import Path
import cv2 as cv
import numpy as np
import base64
import io

from sift_scratch import build_gaussian_pyramid, compute_dog, detect_keypoints, assign_orientations, compute_descriptors, match_descriptors
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


def img_to_base64(img):
    _, buf = cv.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')



@module4_bp.route('/')
def index():
    return render_template('index.html')


@module4_bp.route('/api/stitch', methods=['POST'])
def api_stitch():
    try:
        print(f"[DEBUG] Received stitch request")
        files = request.files.getlist('images')
        print(f"[DEBUG] Number of files: {len(files)}")
        if len(files) < 2:
            return jsonify({'error': 'Upload at least 2 images'}), 400

        imgs = [read_image_file(f) for f in files]
        print(f"[DEBUG] Loaded {len(imgs)} images")
    except Exception as e:
        print(f"[ERROR] Failed to load images: {e}")
        return jsonify({'error': f'Failed to load images: {str(e)}'}), 400

    # compute pairwise matches (simple chain stitching left-to-right)
    homographies = [None] * len(imgs)
    homographies[0] = np.eye(3)
    try:
        for i in range(len(imgs) - 1):
            print(f"[DEBUG] Stitching image {i} to {i+1}")
            img1 = imgs[i]
            img2 = imgs[i + 1]
            # use our simplified SIFT pipeline
            # For speed we call utilities directly
            print(f"[DEBUG] Building pyramids...")
            pyr1 = build_gaussian_pyramid(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))
            pyr2 = build_gaussian_pyramid(cv.cvtColor(img2, cv.COLOR_BGR2GRAY))
            dog1 = compute_dog(pyr1)
            dog2 = compute_dog(pyr2)
            print(f"[DEBUG] Detecting keypoints...")
            kps1 = detect_keypoints(dog1)
            kps2 = detect_keypoints(dog2)
            print(f"[DEBUG] Found {len(kps1)} and {len(kps2)} keypoints")
            kps1_o = assign_orientations(kps1, pyr1)
            kps2_o = assign_orientations(kps2, pyr2)
            print(f"[DEBUG] Computing descriptors...")
            desc1 = compute_descriptors(kps1_o, pyr1)
            desc2 = compute_descriptors(kps2_o, pyr2)
            print(f"[DEBUG] Got {len(desc1)} and {len(desc2)} descriptors")
            matches = match_descriptors(desc1, desc2)
            print(f"[DEBUG] Found {len(matches)} matches")

            pts1 = np.array([desc1[m[0]]['pt'] for m in matches], dtype=np.float32) if matches else np.zeros((0, 2), dtype=np.float32)
            pts2 = np.array([desc2[m[1]]['pt'] for m in matches], dtype=np.float32) if matches else np.zeros((0, 2), dtype=np.float32)

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

            print(f"[DEBUG] Estimating homography with RANSAC...")
            H, inliers = stitch_utils.estimate_homography_ransac(list(zip(pts1, pts2)), thresh=5.0, max_iters=2000)
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

    print(f"[DEBUG] Warping and blending {len(imgs)} images...")
    pano = stitch_utils.warp_and_blend(imgs, Hs, reference=0)
    print(f"[DEBUG] Panorama created with shape {pano.shape}")

    result_b64 = img_to_base64(pano)
    return jsonify({'success': True, 'panorama': f'data:image/jpeg;base64,{result_b64}'})



def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.register_blueprint(module4_bp, url_prefix='/module4')
    return app

if __name__ == '__main__':
    print('Starting Module 4 Image Stitching app on http://localhost:5010')
    app = create_app()
    app.run(debug=True, port=5010)
