"""
Computer Vision Portfolio - Flask Web Application
Combines Module 1 and Module 2 assignments:
- Template Matching with Regional Blurring
- Gaussian Blur and Fourier Transform Recovery
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import base64
import io
import math
import json
from pathlib import Path
import json
import os


app = Flask(__name__)

# --- Module 4 (Image Stitching) integration ---
# Temporarily disabled blueprint to use direct route instead
# import sys as _sys
# from pathlib import Path
# _module4_path = Path(__file__).parent / 'submissions' / 'Module4_ImageStitching'
# try:
#     if _module4_path.exists():
#         from submissions.Module4_ImageStitching.app import module4_bp
#         app.register_blueprint(module4_bp, url_prefix='/module4')
#         print("[INFO] Module 4 blueprint registered at /module4")
#     else:
#         print("[INFO] Module 4 not found at:", _module4_path)
# except Exception as e:
#     print("[WARN] Failed to register Module 4 blueprint:", e)
print("[INFO] Module 4 using direct route instead of blueprint")

# --- Module 7 (Stereo Size Estimation) integration ---
_module7_part1_path = Path(__file__).parent / 'submissions' / 'Module7' / 'part1_stereosize'
try:
    if _module7_part1_path.exists():
        from submissions.Module7.part1_stereosize.app import stereo_bp
        app.register_blueprint(stereo_bp, url_prefix='/module7')
        print("[INFO] Module 7 Part 1 blueprint registered at /module7")
    else:
        print("[INFO] Module 7 Part 1 not found at:", _module7_part1_path)
except Exception as e:
    print("[WARN] Failed to register Module 7 Part 1 blueprint:", e)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Module 3 (Advanced Image Analysis) integration ---
# Register the Module 3 blueprint if available (with better error handling)
try:
    import sys as _sys
    _module3_path = Path(__file__).parent / 'submissions' / 'Module3' / 'web_integration'
    if _module3_path.exists():
        _sys.path.insert(0, str(_module3_path))
        try:
            from routes import module3_bp  # type: ignore
            app.register_blueprint(module3_bp)
            print("[INFO] Module 3 blueprint registered at /module3")
        except (ImportError, KeyboardInterrupt) as e:
            print(f"[WARN] Module 3 blueprint failed to load (missing dependencies): {type(e).__name__}")
            # Continue without Module 3 if dependencies are missing
    else:
        print("[INFO] Module 3 web integration not found at:", _module3_path)
except Exception as e:
    print("[WARN] Failed to register Module 3 blueprint:", e)

# --- Module 5 (Real-time Object Tracker) integration ---
_module5_path = Path(__file__).parent / 'submissions' / 'Module5'
try:
    if _module5_path.exists():
        from submissions.Module5.app import module5_bp
        app.register_blueprint(module5_bp, url_prefix='/module5')
        print("[INFO] Module 5 blueprint registered at /module5")
    else:
        print("[INFO] Module 5 not found at:", _module5_path)
except Exception as e:
    print(f"[WARN] Failed to register Module 5 blueprint: {e}")

class ComputerVisionEngine:
    def __init__(self):
        self.templates_path = Path("images/templates")
        self.scenes_path = Path("images/scenes")
        
    def load_templates(self):
        """Load all template images"""
        templates = {}
        if self.templates_path.exists():
            for template_file in self.templates_path.glob("*.jpg"):
                template = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[template_file.stem] = template
        return templates
    
    def template_matching_with_blur(self, scene_image, blur_detected=True, blur_sigma=3.0):
        """Perform template matching and optionally blur detected regions"""
        templates = self.load_templates()
        results = []
        
        # Convert scene to grayscale if needed
        if len(scene_image.shape) == 3:
            scene_gray = cv.cvtColor(scene_image, cv.COLOR_BGR2GRAY)
        else:
            scene_gray = scene_image.copy()
            
        scene_result = scene_image.copy()
        
        for template_name, template in templates.items():
            # Check if template is smaller than scene
            scene_h, scene_w = scene_gray.shape
            template_h, template_w = template.shape
            
            # Skip if template is larger than scene
            if template_h > scene_h or template_w > scene_w:
                print(f"Skipping {template_name}: template ({template_w}x{template_h}) larger than scene ({scene_w}x{scene_h})")
                continue
            
            # Template matching with normalized correlation
            result = cv.matchTemplate(scene_gray, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            # Debug: Print confidence scores
            print(f"Template {template_name}: confidence = {max_val:.3f}")
            
            # Set threshold for detection (lowered to see more matches)
            threshold = 0.3
            
            if max_val >= threshold:
                # Get bounding box
                h, w = template.shape
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # Draw rectangle
                cv.rectangle(scene_result, top_left, bottom_right, (0, 255, 0), 2)
                cv.putText(scene_result, f'{template_name}: {max_val:.2f}', 
                          (top_left[0], top_left[1] - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Apply blur to detected region if requested
                if blur_detected:
                    roi = scene_result[top_left[1]:bottom_right[1], 
                                     top_left[0]:bottom_right[0]]
                    
                    # Apply Gaussian blur
                    kernel_size = int(6 * blur_sigma + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    blurred_roi = cv.GaussianBlur(roi, (kernel_size, kernel_size), blur_sigma)
                    scene_result[top_left[1]:bottom_right[1], 
                               top_left[0]:bottom_right[0]] = blurred_roi
                
                results.append({
                    'template': template_name,
                    'confidence': float(max_val),
                    'bbox': [int(top_left[0]), int(top_left[1]), int(w), int(h)],
                    'blurred': blur_detected
                })
        
        return scene_result, results
    
    def gaussian_blur_recovery(self, image, sigma=3.0):
        """Demonstrate Gaussian blur and FFT recovery"""
        # Convert to grayscale and float
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        L = gray.astype(np.float64) / 255.0
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Apply blur
        L_b = cv.GaussianBlur(L, (kernel_size, kernel_size), sigma)
        
        # FFT-based recovery
        # Create Gaussian kernel for deconvolution
        gaussian_kernel_2d = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Pad kernel to image size
        padded_kernel = np.zeros_like(L)
        kh, kw = gaussian_kernel_2d.shape
        padded_kernel[:kh, :kw] = gaussian_kernel_2d
        
        # Shift zero frequency to center for proper FFT
        padded_kernel = np.fft.fftshift(padded_kernel)
        
        # FFT
        L_b_fft = np.fft.fft2(L_b)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Wiener deconvolution (with noise regularization)
        noise_level = 0.01
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        
        L_recovered_fft = (L_b_fft * kernel_conj) / (kernel_mag_sq + noise_level)
        L_recovered = np.real(np.fft.ifft2(L_recovered_fft))
        
        # Calculate PSNR
        psnr = self.calculate_psnr(L, L_recovered)
        
        return {
            'original': (L * 255).astype(np.uint8),
            'blurred': (L_b * 255).astype(np.uint8),
            'recovered': np.clip(L_recovered * 255, 0, 255).astype(np.uint8),
            'psnr': float(psnr),
            'sigma': float(sigma)
        }
    
    def create_gaussian_kernel(self, size, sigma):
        """Create 2D Gaussian kernel"""
        kernel_1d = cv.getGaussianKernel(size, sigma)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.sum()
    
    def calculate_psnr(self, original, recovered):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original - recovered) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Since we're working with 0-1 range
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def get_camera_calibration(self):
        """Return pre-calibrated camera intrinsic parameters"""
        # Camera matrix from Module 1:
        # [[640.8396063    0.         294.24936703]
        #  [  0.         648.80269311 349.31369175]
        #  [  0.           0.           1.        ]]
        return {
            'Fx': 640.8396063,
            'Fy': 648.80269311, 
            'Ox': 294.24936703,
            'Oy': 349.31369175
        }
    
    def calculate_real_world_distance(self, point1, point2, z_distance, calibration_data):
        """
        Calculate real-world distance between two points using perspective projection
        
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
            image: input image
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
        cv.circle(result_image, tuple(map(int, point1)), 5, (0, 255, 0), -1)
        cv.circle(result_image, tuple(map(int, point2)), 5, (0, 0, 255), -1)
        cv.line(result_image, tuple(map(int, point1)), tuple(map(int, point2)), (255, 0, 0), 2)
        
        # Add text with measurement
        text = f"Distance: {distance_result['euclidean_distance']:.1f}cm"
        cv.putText(result_image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            'processed_image': result_image,
            'measurements': distance_result,
            'calibration_used': calibration
        }

# Initialize CV engine
cv_engine = ComputerVisionEngine()

@app.route('/')
def index():
    """Main portfolio page"""
    return render_template('index.html')

@app.route('/module1')
def module1():
    """Module 1: Real-world Distance Measurement using Perspective Projection"""
    return render_template('module1_distance_measurement_clean.html')

@app.route('/module2-part1')
def module2_part1():
    """Module 2 Part 1: Template Matching with Regional Blurring"""
    return render_template('part1_template_matching.html')

@app.route('/module2-part2')
def module2_part2():
    """Module 2 Part 2: Gaussian Blur and Fourier Transform Recovery"""
    return render_template('part2_blur_recovery.html')

@app.route('/module4/')
@app.route('/module4')
def module4_direct():
    """Module 4: Image Stitching - Direct route using main template"""
    return render_template('module4.html')

@app.route('/api/template-matching', methods=['POST'])
def api_template_matching():
    """API endpoint for template matching with optional blurring"""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters
        blur_detected = request.form.get('blur_detected', 'false').lower() == 'true'
        blur_sigma = float(request.form.get('blur_sigma', 3.0))
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        # Perform template matching
        result_image, detections = cv_engine.template_matching_with_blur(
            image, blur_detected, blur_sigma
        )
        
        # Save result image
        result_path = os.path.join(RESULTS_FOLDER, 'template_matching_result.jpg')
        cv.imwrite(result_path, result_image)
        
        # Convert to base64 for web display
        _, buffer = cv.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{image_base64}',
            'detections': detections,
            'total_detected': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blur-recovery', methods=['POST'])
def api_blur_recovery():
    """API endpoint for Gaussian blur and recovery demonstration"""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters
        sigma = float(request.form.get('sigma', 3.0))
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        # Perform blur and recovery
        results = cv_engine.gaussian_blur_recovery(image, sigma)
        
        # Convert images to base64
        def img_to_base64(img):
            _, buffer = cv.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'original': f'data:image/jpeg;base64,{img_to_base64(results["original"])}',
            'blurred': f'data:image/jpeg;base64,{img_to_base64(results["blurred"])}',
            'recovered': f'data:image/jpeg;base64,{img_to_base64(results["recovered"])}',
            'psnr': results['psnr'],
            'sigma': results['sigma']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates')
def api_templates():
    """Get list of available templates"""
    templates = cv_engine.load_templates()
    template_list = []
    
    for name, template in templates.items():
        # Convert template to base64 for preview
        _, buffer = cv.imencode('.jpg', template)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        template_list.append({
            'name': name,
            'image': f'data:image/jpeg;base64,{image_base64}',
            'size': template.shape
        })
    
    return jsonify({'templates': template_list})

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
        
        print(f"📍 Points: ({point1_x}, {point1_y}) to ({point2_x}, {point2_y})")
        print(f"📏 Z-distance: {z_distance}cm")
        
        # Read image
        image_array = np.frombuffer(file.read(), np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        print(f"🖼️ Image loaded: {image.shape if image is not None else 'Failed to load'}")
        
        # Process distance measurement
        result = cv_engine.process_distance_measurement(
            image, 
            (point1_x, point1_y), 
            (point2_x, point2_y), 
            z_distance
        )
        
        # Convert result image to base64
        def img_to_base64(img):
            _, buffer = cv.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{img_to_base64(result["processed_image"])}',
            'measurements': result['measurements'],
            'calibration': result['calibration_used'],
            'z_distance_used': z_distance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibration')
def api_calibration():
    """Get camera calibration data"""
    print("📷 Calibration API called")
    calibration = cv_engine.get_camera_calibration()
    print(f"📷 Returning calibration: {calibration}")
    return jsonify(calibration)

@app.route('/api/test')
def api_test():
    """Simple test endpoint"""
    print("🧪 Test API called")
    return jsonify({"status": "working", "message": "API is responding"})

@app.route('/debug/routes')
def debug_routes():
    """Debug endpoint to show all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify({"routes": routes, "total": len(routes)})


if __name__ == '__main__':
    print("🚀 Starting Computer Vision Portfolio Website...")
    print("📁 Make sure you have images/templates/ and images/scenes/ folders")
    print("🌐 Visit: http://localhost:5000")
    app.run(debug=True, use_reloader=False)