"""
Module 2: Template Matching and Blur Recovery

This module combines two computer vision techniques:
1. Template Matching through Normalized Cross-Correlation with Regional Blurring
2. Gaussian Blur and FFT-based Recovery using Wiener Deconvolution

"""

from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import base64
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
    
    def template_matching(self, scene_image, blur_detected=True, blur_sigma=3.0):
        """
        Template Matching using Normalized Cross-Correlation (NCC)
        as required by the assignment.
        
        Detects objects using template matching, then applies Gaussian blur
        to the detected regions as specified in the assignment.

        Args:
            scene_image: input BGR scene image
            blur_detected: whether to blur detected regions
            blur_sigma: Gaussian blur sigma parameter

        Returns:
            result_image: scene with bounding boxes and blurred regions
            detections: list with template name, confidence, and bbox
        """

        templates = self.load_templates()
        detections = []

        # Convert to grayscale
        if len(scene_image.shape) == 3:
            scene_gray = cv.cvtColor(scene_image, cv.COLOR_BGR2GRAY)
        else:
            scene_gray = scene_image.copy()

        scene_result = scene_image.copy()

        # Loop through templates
        for template_name, template in templates.items():

            # Ensure template fits
            if template.shape[0] > scene_gray.shape[0] or template.shape[1] > scene_gray.shape[1]:
                continue

            h, w = template.shape

            # Perform NCC correlation
            heatmap = cv.matchTemplate(scene_gray, template, cv.TM_CCOEFF_NORMED)

            # Threshold (NCC correlation)
            threshold = 0.40
            yloc, xloc = np.where(heatmap >= threshold)

            # Collect raw detections
            boxes = []
            scores = []

            for (x, y) in zip(xloc, yloc):
                boxes.append([x, y, x + w, y + h])
                scores.append(float(heatmap[y, x]))

            if not boxes:
                print(f"No matches for: {template_name}")
                continue

            boxes = np.array(boxes)
            scores = np.array(scores)

            # Non-maximum suppression
            keep_indices = self.non_max_suppression(boxes, scores, overlapThresh=0.3)

            # Draw and store results
            for idx in keep_indices:
                x1, y1, x2, y2 = boxes[idx]
                score = scores[idx]

                # Apply blur to detected region if enabled
                if blur_detected:
                    # Extract the region
                    region = scene_result[y1:y2, x1:x2]
                    
                    # Apply Gaussian blur
                    ksize = int(6 * blur_sigma) + 1
                    if ksize % 2 == 0:
                        ksize += 1
                    blurred_region = cv.GaussianBlur(region, (ksize, ksize), blur_sigma)
                    
                    # Replace the region in the result image
                    scene_result[y1:y2, x1:x2] = blurred_region
                
                # Draw bounding box
                cv.rectangle(scene_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(scene_result, f"{template_name}: {score:.2f}",
                           (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)

                detections.append({
                    "template": template_name,
                    "confidence": float(score),
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "blurred": blur_detected
                })

        return scene_result, detections
    
    def non_max_suppression(self, boxes, scores, overlapThresh=0.3):
        """
        Fast NMS for template matching bounding boxes.
        boxes: [x1,y1,x2,y2]
        returns: indices of kept boxes
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= overlapThresh)[0]
            order = order[inds + 1]

        return keep
    
    def gaussian_blur_recovery(self, image, sigma=3.0, noise_level=1e-2, debug=False):
        """
        Robust Gaussian blur + Wiener deconvolution using numpy.fft (fft2 / ifft2).
        This avoids cv.dft/roll/fftshift pitfalls and prevents quadrant swaps.

        Args:
            image: BGR or grayscale image (uint8)
            sigma: gaussian kernel sigma (float)
            noise_level: K parameter in Wiener filter (float)
            debug: if True, prints diagnostics

        Returns:
            dict with 'original', 'blurred', 'recovered' as uint8 arrays (single-channel)
        """
        # 1) convert to grayscale float image in [0,1]
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        L = gray.astype(np.float32) / 255.0
        h, w = L.shape

        # 2) blur using OpenCV (to simulate observed blurred image)
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        L_b = cv.GaussianBlur(L, (ksize, ksize), sigma)

        # 3) create gaussian kernel (small) and pad to image size
        kernel_small = self.create_gaussian_kernel(ksize, sigma)  # normalized
        kh, kw = kernel_small.shape

        # pad kernel to image size centered in the array
        pad_top = (h - kh) // 2
        pad_bottom = h - kh - pad_top
        pad_left = (w - kw) // 2
        pad_right = w - kw - pad_left
        psf = np.pad(kernel_small, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        # 4) center PSF for frequency domain (move kernel center to [0,0] freq position)
        psf_shifted = np.fft.ifftshift(psf)   # critical

        # DEBUG checks
        if debug:
            print("L shape:", L.shape)
            print("psf small shape:", kernel_small.shape)
            print("psf padded shape:", psf.shape)
            print("psf sum (should be ~1):", psf.sum())
            # show where the largest PSF value is (should be near center of kernel_small before ifftshift)
            print("psf max index (padded):", np.unravel_index(np.argmax(psf), psf.shape))
            print("psf_shifted max index:", np.unravel_index(np.argmax(psf_shifted), psf_shifted.shape))

        # 5) FFTs using numpy
        F_blurred = np.fft.fft2(L_b)
        H = np.fft.fft2(psf_shifted)

        # 6) Wiener filter: G * conj(H) / (|H|^2 + K)
        H_conj = np.conjugate(H)
        H_mag_sq = (np.abs(H) ** 2)
        denom = H_mag_sq + noise_level
        F_recovered = (F_blurred * H_conj) / denom

        # 7) inverse FFT to get recovered image
        L_recovered = np.fft.ifft2(F_recovered).real

        # 8) clip / normalize to [0,1] robustly
        L_recovered = L_recovered - L_recovered.min()
        maxv = L_recovered.max()
        if maxv > 0:
            L_recovered = L_recovered / maxv
        else:
            L_recovered = np.zeros_like(L_recovered)

        # 9) compute PSNR (expecting original L in [0,1])
        psnr = self.calculate_psnr(L, L_recovered)

        # 10) return uint8 images for display
        return {
            'original': (np.clip(L * 255.0, 0, 255)).astype(np.uint8),
            'blurred': (np.clip(L_b * 255.0, 0, 255)).astype(np.uint8),
            'recovered': (np.clip(L_recovered * 255.0, 0, 255)).astype(np.uint8),
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
        
        # Perform template matching with blur
        result_image, detections = cv_engine.template_matching(
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