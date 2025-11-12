"""
Module 7 Part 1: Object Size Estimation using Calibrated Stereo
Flask blueprint for stereo size estimation
"""
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, render_template


stereo_bp = Blueprint('stereo_bp', __name__, template_folder='../../../templates')

@stereo_bp.route('/test', methods=['GET'])
def stereo_test():
    return 'Module 7 blueprint is registered and working!'

# Dummy calibration parameters (replace with real values)
CALIBRATION = {
    'focal_length': 800,  # pixels
    'baseline': 0.1,      # meters
    'cx': 320,
    'cy': 240
}

def compute_disparity(left_img, right_img):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity

def estimate_depth(disparity, focal_length, baseline):
    disparity[disparity == 0] = 0.1
    Z = (focal_length * baseline) / disparity
    return Z

def detect_shape_and_measure(img, mask, Z, calibration):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        shape = "polygon"
        dims = {}
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            shape = "rectangle"
            x, y, w, h = cv2.boundingRect(approx)
            z_mean = np.mean(Z[y:y+h, x:x+w])
            dims['width'] = w * z_mean / calibration['focal_length']
            dims['height'] = h * z_mean / calibration['focal_length']
        elif len(approx) > 6:
            shape = "circle"
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            z_mean = np.mean(Z[int(y-radius):int(y+radius), int(x-radius):int(x+radius)])
            dims['diameter'] = 2 * radius * z_mean / calibration['focal_length']
        else:
            pts = approx.reshape(-1, 2)
            z_mean = np.mean([Z[y, x] for x, y in pts])
            edges = []
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i+1)%len(pts)]
                pixel_dist = np.linalg.norm(p1 - p2)
                edge_len = pixel_dist * z_mean / calibration['focal_length']
                edges.append(edge_len)
            dims['edges'] = edges
        results.append({'shape': shape, 'dimensions': dims})
    return results

@stereo_bp.route('/', methods=['GET'])
def stereo_page():
    return render_template('module7.html')

@stereo_bp.route('/api/stereo/estimate', methods=['POST'])
def stereo_estimate():
    left_file = request.files.get('left')
    right_file = request.files.get('right')
    mask_file = request.files.get('mask')
    if not left_file or not right_file or not mask_file:
        return jsonify({'error': 'Missing files'}), 400
    left_img = cv2.imdecode(np.frombuffer(left_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.frombuffer(right_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    disparity = compute_disparity(left_img, right_img)
    Z = estimate_depth(disparity, CALIBRATION['focal_length'], CALIBRATION['baseline'])
    results = detect_shape_and_measure(left_img, mask, Z, CALIBRATION)
    return jsonify({'results': results})
