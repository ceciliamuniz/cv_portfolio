"""
Module 2: Template Matching and Blur Recovery
Standalone Flask Application

This module combines two computer vision techniques:
1. Template Matching through Normalized Cross-Correlation with Regional Blurring
2. Gaussian Blur and FFT-based Recovery using Wiener Deconvolution

Mathematical Foundation:
- Template Matching: Uses normalized cross-correlation to find templates in scenes
- Gaussian Blur: Applied via convolution with 2D Gaussian kernel
- FFT Recovery: Uses Wiener deconvolution in frequency domain to recover blurred images

Author: Computer Vision Student
Course: Computer Vision Module 2
"""

from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import base64
import io
import os
from pathlib import Path

app = Flask(__name__)

# Configure upload and results folders
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class Module2Engine:
    """
    Computer Vision Engine for Module 2 operations:
    - Template Matching with Regional Blurring
    - Gaussian Blur and FFT-based Recovery
    """
    
    def __init__(self):
        self.templates_path = Path("images/templates")
        
    def load_templates(self):
        """Load all template images from the templates directory"""
        templates = {}
        if self.templates_path.exists():
            for template_file in self.templates_path.glob("*.jpg"):
                template = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[template_file.stem] = template
            # Also check for PNG files
            for template_file in self.templates_path.glob("*.png"):
                template = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[template_file.stem] = template
        return templates
    
    def template_matching_with_blur(self, scene_image, blur_detected=True, blur_sigma=3.0):
        """
        Perform template matching using normalized cross-correlation
        and optionally blur detected regions
        
        Args:
            scene_image: Input scene image (BGR format)
            blur_detected: Whether to blur detected regions
            blur_sigma: Standard deviation for Gaussian blur
            
        Returns:
            tuple: (result_image, detection_results)
        """
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
            
            # Template matching with normalized cross-correlation
            result = cv.matchTemplate(scene_gray, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            # Debug: Print confidence scores
            print(f"Template {template_name}: confidence = {max_val:.3f}")
            
            # Set threshold for detection
            threshold = 0.3
            
            if max_val >= threshold:
                # Get bounding box coordinates
                h, w = template.shape
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # Draw detection rectangle
                cv.rectangle(scene_result, top_left, bottom_right, (0, 255, 0), 2)
                cv.putText(scene_result, f'{template_name}: {max_val:.2f}', 
                          (top_left[0], top_left[1] - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Apply Gaussian blur to detected region if requested
                if blur_detected:
                    roi = scene_result[top_left[1]:bottom_right[1], 
                                     top_left[0]:bottom_right[0]]
                    
                    # Calculate kernel size based on sigma (6-sigma rule)
                    kernel_size = int(6 * blur_sigma + 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    # Apply Gaussian blur via convolution
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
        """
        Demonstrate Gaussian blur and FFT-based recovery using Wiener deconvolution
        
        Mathematical Process:
        1. Apply Gaussian blur: I_blurred = I * G(σ)
        2. DFT both image and kernel: F{I_blurred}, F{G}
        3. Wiener deconvolution: F{I_recovered} = F{I_blurred} * conj(F{G}) / (|F{G}|² + λ)
        4. IDFT to get recovered image
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            dict: Contains original, blurred, recovered images and PSNR
        """
        # Convert to grayscale and normalize to [0,1]
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        L = gray.astype(np.float32) / 255.0

        # Create and apply Gaussian blur
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply blur using OpenCV
        L_b = cv.GaussianBlur(L, (kernel_size, kernel_size), sigma)

        # Create 2D Gaussian kernel for deconvolution and pad to image size
        gaussian_kernel_2d = self.create_gaussian_kernel(kernel_size, sigma)
        h, w = L.shape
        kh, kw = gaussian_kernel_2d.shape
        pad_y = max(0, h - kh)
        pad_x = max(0, w - kw)
        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        psf = cv.copyMakeBorder(gaussian_kernel_2d, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=0)

        # DFT (complex) of blurred image and PSF
        dft_L_b = cv.dft(L_b, flags=cv.DFT_COMPLEX_OUTPUT)
        dft_psf = cv.dft(psf, flags=cv.DFT_COMPLEX_OUTPUT)

        # Wiener deconvolution in frequency domain
        noise_level = 0.01
        psf_mag_sq = dft_psf[:, :, 0] ** 2 + dft_psf[:, :, 1] ** 2
        denominator = psf_mag_sq + noise_level

        # Compute recovered spectrum: F(I) * conj(H) / (|H|^2 + K)
        recovered_real = (dft_L_b[:, :, 0] * dft_psf[:, :, 0] + dft_L_b[:, :, 1] * dft_psf[:, :, 1]) / denominator
        recovered_imag = (dft_L_b[:, :, 1] * dft_psf[:, :, 0] - dft_L_b[:, :, 0] * dft_psf[:, :, 1]) / denominator

        dft_recovered = np.stack([recovered_real, recovered_imag], axis=2)

        # Inverse DFT to get recovered image
        L_recovered = cv.idft(dft_recovered, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        if len(L_recovered.shape) > 2:
            L_recovered = L_recovered[:, :, 0]

        # Normalize recovered image to [0, 1]
        L_recovered_norm = L_recovered - L_recovered.min()
        if L_recovered_norm.max() > 0:
            L_recovered_norm = L_recovered_norm / L_recovered_norm.max()
        else:
            L_recovered_norm = np.zeros_like(L_recovered_norm)

        psnr = self.calculate_psnr(L, L_recovered_norm)

        return {
            'original': (L * 255).astype(np.uint8),
            'blurred': (L_b * 255).astype(np.uint8),
            'recovered': np.clip(L_recovered_norm * 255, 0, 255).astype(np.uint8),
            'psnr': float(psnr),
            'sigma': float(sigma)
        }
    def create_gaussian_kernel(self, size, sigma):
        """
        Create 2D Gaussian kernel using separable 1D kernels

        Args:
            size: Kernel size (should be odd)
            sigma: Standard deviation

        Returns:
            2D normalized Gaussian kernel
        """
        kernel_1d = cv.getGaussianKernel(size, sigma)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.sum()
    
    def calculate_psnr(self, original, recovered):
        """
        Calculate Peak Signal-to-Noise Ratio
        
        PSNR = 20 * log10(MAX_I / sqrt(MSE))
        where MSE = mean((original - recovered)²)
        
        Args:
            original: Original image (0-1 range)
            recovered: Recovered image (0-1 range)
            
        Returns:
            PSNR value in dB
        """
        mse = np.mean((original - recovered) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Since we're working with 0-1 range
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

# Initialize the computer vision engine
cv_engine = Module2Engine()

# Routes
@app.route('/')
def index():
    """Main page showing both Module 2 parts"""
    return render_template('index.html')

@app.route('/template-matching')
def template_matching():
    """Template Matching interface"""
    return render_template('template_matching.html')

@app.route('/blur-recovery')
def blur_recovery():
    """Blur Recovery interface"""
    return render_template('blur_recovery.html')

# API Endpoints
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

if __name__ == '__main__':
    print("🚀 Starting Module 2: Template Matching & Blur Recovery")
    print("📍 Template Matching: http://localhost:5000/template-matching")
    print("🌀 Blur Recovery: http://localhost:5000/blur-recovery")
    print("🏠 Home: http://localhost:5000/")
    print("\n💡 Make sure to place template images in 'images/templates/' directory")
    app.run(debug=True, port=5000, use_reloader=False)