"""
Module 1: Real-world Distance Measurement using Perspective Projection
Standalone Flask Application for Assignment Submission

This application demonstrates:
- Camera calibration with pre-computed intrinsic parameters
- Perspective projection mathematics for real-world distance calculation
- Interactive web interface for point selection and measurement
- Real-time distance computation in centimeters
"""

from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import base64
import io
import math
import json
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DistanceMeasurementEngine:
    def __init__(self):
        # Pre-calibrated camera intrinsic parameters from calibration process
        # Camera matrix: [[640.84, 0, 294.25], [0, 648.80, 349.31], [0, 0, 1]]
        self.camera_matrix = {
            'Fx': 640.8396063,    # Focal length in x direction
            'Fy': 648.80269311,   # Focal length in y direction  
            'Ox': 294.24936703,   # Principal point x coordinate
            'Oy': 349.31369175    # Principal point y coordinate
        }
    
    def get_camera_calibration(self):
        """Return pre-calibrated camera intrinsic parameters"""
        return self.camera_matrix
    
    def calculate_real_world_distance(self, point1, point2, z_distance, calibration_data):
        """
        Calculate real-world distance between two points using perspective projection
        
        Mathematical Formula:
        X_real = (x_pixel - Ox) * Z / Fx
        Y_real = (y_pixel - Oy) * Z / Fy
        
        Where:
        - (x_pixel, y_pixel) are image coordinates in pixels
        - (Ox, Oy) is the principal point (camera center)
        - (Fx, Fy) are focal lengths in pixels
        - Z is the depth (distance from camera to object plane)
        - (X_real, Y_real) are real-world coordinates
        
        Args:
            point1, point2: (x, y) coordinates in pixels
            z_distance: real-world distance from camera to objects (in cm)
            calibration_data: camera intrinsic parameters
        
        Returns:
            dict with x_distance, y_distance, euclidean_distance in cm
        """
        # Convert image coordinates to normalized coordinates
        x1_norm = (point1[0] - calibration_data['Ox']) / calibration_data['Fx']
        y1_norm = (point1[1] - calibration_data['Oy']) / calibration_data['Fy']
        
        x2_norm = (point2[0] - calibration_data['Ox']) / calibration_data['Fx']
        y2_norm = (point2[1] - calibration_data['Oy']) / calibration_data['Fy']
        
        # Convert to real-world coordinates using perspective projection
        x1_real = x1_norm * z_distance
        y1_real = y1_norm * z_distance
        
        x2_real = x2_norm * z_distance
        y2_real = y2_norm * z_distance
        
        # Calculate distances
        x_distance = abs(x2_real - x1_real)
        y_distance = abs(y2_real - y1_real)
        euclidean_distance = math.sqrt(x_distance**2 + y_distance**2)
        
        return {
            'x_distance': x_distance,
            'y_distance': y_distance,
            'euclidean_distance': euclidean_distance,
            'point1_real': (x1_real, y1_real),
            'point2_real': (x2_real, y2_real)
        }
    
    def process_distance_measurement(self, image, point1, point2, z_distance):
        """
        Process an image with two clicked points to measure real-world distance
        
        Args:
            image: input image (numpy array)
            point1, point2: (x, y) coordinates of clicked points
            z_distance: real-world distance from camera (in cm)
            
        Returns:
            dict with processed image and measurement results
        """
        calibration = self.get_camera_calibration()
        
        # Calculate real-world distance
        distance_result = self.calculate_real_world_distance(
            point1, point2, z_distance, calibration
        )
        
        # Draw points and line on image
        result_image = image.copy()
        cv.circle(result_image, tuple(map(int, point1)), 8, (0, 255, 0), -1)  # Green circle for point 1
        cv.circle(result_image, tuple(map(int, point2)), 8, (0, 0, 255), -1)  # Red circle for point 2
        cv.line(result_image, tuple(map(int, point1)), tuple(map(int, point2)), (255, 0, 0), 3)  # Blue line
        
        # Add measurement text
        text = f"Distance: {distance_result['euclidean_distance']:.1f}cm"
        cv.putText(result_image, text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv.putText(result_image, text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        
        # Add point labels
        cv.putText(result_image, "P1", (int(point1[0])-20, int(point1[1])-15), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(result_image, "P2", (int(point2[0])-20, int(point2[1])-15), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return {
            'processed_image': result_image,
            'measurements': distance_result,
            'calibration_used': calibration
        }

# Initialize distance measurement engine
distance_engine = DistanceMeasurementEngine()

# Routes
@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/calibration')
def api_calibration():
    """Get camera calibration data"""
    print("📷 Calibration API called")
    calibration = distance_engine.get_camera_calibration()
    print(f"📷 Returning calibration: {calibration}")
    return jsonify(calibration)

@app.route('/api/distance-measurement', methods=['POST'])
def api_distance_measurement():
    """Process distance measurement between two points"""
    try:
        print("📏 Distance measurement request received")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters from form
        point1_x = float(request.form.get('point1_x', 0))
        point1_y = float(request.form.get('point1_y', 0))
        point2_x = float(request.form.get('point2_x', 0))
        point2_y = float(request.form.get('point2_y', 0))
        z_distance = float(request.form.get('z_distance', 100))  # Default 100cm
        
        print(f"📍 Points: ({point1_x:.1f}, {point1_y:.1f}) to ({point2_x:.1f}, {point2_y:.1f})")
        print(f"📏 Z-distance: {z_distance}cm")
        
        # Read image
        image_array = np.frombuffer(file.read(), np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        print(f"🖼️ Image loaded: {image.shape if image is not None else 'Failed to load'}")
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Process distance measurement
        result = distance_engine.process_distance_measurement(
            image, 
            (point1_x, point1_y), 
            (point2_x, point2_y), 
            z_distance
        )
        
        # Convert result image to base64
        def img_to_base64(img):
            _, buffer = cv.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Measurement completed: {result['measurements']['euclidean_distance']:.2f}cm")
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{img_to_base64(result["processed_image"])}',
            'measurements': result['measurements'],
            'calibration': result['calibration_used'],
            'z_distance_used': z_distance
        })
        
    except Exception as e:
        print(f"❌ Error in distance measurement: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def api_test():
    """Simple test endpoint"""
    print("🧪 Test API called")
    return jsonify({
        "status": "working", 
        "message": "Module 1 Distance Measurement API is responding",
        "version": "1.0"
    })

if __name__ == '__main__':
    print("🚀 Starting Module 1: Distance Measurement Application")
    print("📐 Real-world distance measurement using perspective projection")
    print("🌐 Visit: http://localhost:5001")
    app.run(debug=True, port=5001)