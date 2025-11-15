"""
Module 1: Real-world Distance Measurement using Perspective Projection

This web application demonstrates:
- Camera calibration with pre-computed intrinsic parameters
- Perspective projection mathematics for real-world distance calculation
- Interactive web interface for point selection and measurement
- Real-time distance computation in centimeters
"""

from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import base64
import math
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
        
        Formula: ΔX_real = Δx_image * Z / fx
                ΔY_real = Δy_image * Z / fy
        
        Where:
        - Δx_image, Δy_image are pixel differences between the two points
        - Z is the distance from camera to object plane (in cm)
        - fx, fy are focal lengths in pixels
        - ΔX_real, ΔY_real are real-world distances (in cm)
        
        Args:
            point1, point2: (x, y) coordinates in pixels
            z_distance: real-world distance from camera to objects (in cm)
            calibration_data: camera intrinsic parameters
        
        Returns:
            dict with x_distance, y_distance, euclidean_distance in cm
        """
        # Calculate pixel differences
        dx_pixels = abs(point2[0] - point1[0])
        dy_pixels = abs(point2[1] - point1[1])
        
        # Get focal lengths
        fx = calibration_data['Fx']
        fy = calibration_data['Fy']
        
        # Apply the given formula: ΔX_real = Δx_image * Z / fx
        x_distance = (dx_pixels * z_distance) / fx
        y_distance = (dy_pixels * z_distance) / fy
        
        # Calculate euclidean distance
        euclidean_distance = math.sqrt(x_distance**2 + y_distance**2)
        
        # For reference, calculate absolute positions (not used for distance calculation)
        x1_real = (point1[0] - calibration_data['Ox']) * z_distance / fx
        y1_real = (point1[1] - calibration_data['Oy']) * z_distance / fy
        x2_real = (point2[0] - calibration_data['Ox']) * z_distance / fx
        y2_real = (point2[1] - calibration_data['Oy']) * z_distance / fy
        
        return {
            'x_distance': x_distance,
            'y_distance': y_distance,
            'euclidean_distance': euclidean_distance,
            'point1_real': (x1_real, y1_real),
            'point2_real': (x2_real, y2_real),
            'debug_info': {
                'dx_pixels': dx_pixels,
                'dy_pixels': dy_pixels,
                'fx': fx,
                'fy': fy,
                'z_distance': z_distance
            }
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
        
        # Adjust calibration for image scaling
        # Use your calculated scale factor: old (fx=640) became (810) with scale 1.266
        current_width = image.shape[1]
        current_height = image.shape[0]
        
        # Use your specific scale factor that gave better results
        your_scale_factor = 1.266  # 810/640 = 1.266 as you calculated
        
        # Scale the focal lengths and principal point
        adjusted_calibration = {
            'Fx': calibration['Fx'] * your_scale_factor,
            'Fy': calibration['Fy'] * your_scale_factor, 
            'Ox': calibration['Ox'] * your_scale_factor,
            'Oy': calibration['Oy'] * your_scale_factor
        }
        
        print(f"📐 Focal length scaling applied:")
        print(f"   Current image size: {current_width}x{current_height}")
        print(f"   Scale factor: {your_scale_factor:.3f}")
        print(f"   Original Fx: {calibration['Fx']:.1f} → Adjusted: {adjusted_calibration['Fx']:.1f}")
        print(f"   Original Fy: {calibration['Fy']:.1f} → Adjusted: {adjusted_calibration['Fy']:.1f}")
        
        # Calculate real-world distance using adjusted calibration
        distance_result = self.calculate_real_world_distance(
            point1, point2, z_distance, adjusted_calibration
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
            'calibration_used': adjusted_calibration,
            'original_calibration': calibration,
            'scaling_info': {
                'current_size': (current_width, current_height),
                'scale_factor': your_scale_factor,
                'original_fx': calibration['Fx'],
                'adjusted_fx': adjusted_calibration['Fx']
            }
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
        z_distance = float(request.form.get('z_distance'))  # Required parameter, no default
        
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