# Module 2: Template Matching and Blur Recovery

A standalone Flask web application implementing advanced computer vision techniques for object detection and image restoration.

## Demo Video
## Part 1
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod2_pt1_rec.gif)

## Part 2
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod2_pt2_rec.gif)

*Live demonstration of the template matching and blur recovery application showing image upload, template detection with confidence scoring, and FFT-based blur recovery with PSNR metrics.*

## 📋 Overview

This module combines two fundamental computer vision operations:

1. **Template Matching**: Object detection using normalized cross-correlation with optional regional blurring
2. **Blur Recovery**: Gaussian blur application and FFT-based recovery using Wiener deconvolution

## 🧮 Mathematical Foundation

### Template Matching - Normalized Cross-Correlation

Template matching uses normalized cross-correlation to find template patterns within scene images:

```
NCC(x,y) = Σ[T(x',y') × I(x+x',y+y')] / √[Σ T²(x',y') × Σ I²(x+x',y+y')]
```

Where:
- `T(x',y')` is the template image
- `I(x+x',y+y')` is the image region being compared
- The result ranges from -1 to 1, with 1 indicating perfect match

### Gaussian Blur - 2D Convolution

Gaussian blur is applied using 2D convolution with a Gaussian kernel:

```
G(x,y) = (1/2πσ²) × exp(-(x² + y²)/2σ²)
```

```
I_blurred(x,y) = I(x,y) ⊗ G(x,y)
```

Where:
- `σ` is the standard deviation controlling blur intensity
- `⊗` denotes convolution operation
- Kernel size = `6σ + 1` (ensuring 99.7% of Gaussian captured)

### FFT-Based Recovery - Wiener Deconvolution

Recovery uses Wiener deconvolution in the frequency domain:

```
H_wiener(ω) = H*(ω) / (|H(ω)|² + λ)
```

```
F{I_recovered} = F{I_blurred} × H_wiener(ω)
```

Where:
- `H*(ω)` is the complex conjugate of the blur kernel's FFT
- `|H(ω)|²` is the power spectrum of the blur kernel
- `λ` is the noise regularization parameter (0.01)
- `F{·}` denotes Fourier transform

### Quality Assessment - PSNR

Peak Signal-to-Noise Ratio measures recovery quality:

```
MSE = (1/MN) × Σ[I_original(i,j) - I_recovered(i,j)]²
```

```
PSNR = 20 × log₁₀(MAX_I / √MSE)
```

Where:
- `MAX_I` is the maximum possible pixel value (1.0 for normalized images)
- Higher PSNR values indicate better recovery quality
- Typical good values: 20-40 dB


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd Module2_TemplateMatching_BlurRecovery
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify template images:**
   - Template images should be in `images/templates/` directory
   - Supported formats: JPG, PNG
   - 10 sample templates are included

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   - Main page: http://localhost:5000
   - Template Matching: http://localhost:5000/template-matching
   - Blur Recovery: http://localhost:5000/blur-recovery

## 🎯 Features

### Template Matching Module

- **Multi-template Detection**: Simultaneously detect multiple object types
- **Confidence Scoring**: Normalized cross-correlation confidence values (0-1)
- **Regional Blurring**: Apply Gaussian blur to detected regions
- **Interactive Parameters**: Adjustable blur intensity and detection thresholds
- **Visual Results**: Bounding boxes with confidence scores overlay

#### Usage:
1. Upload a scene image containing objects to detect
2. Configure blur settings (optional)
3. Adjust blur intensity (σ = 0.5 to 10.0)
4. Run detection to find templates in the scene
5. View results with detection count and confidence scores

### Blur Recovery Module

- **Gaussian Blur**: Apply controlled blur using 2D convolution
- **FFT Processing**: Frequency domain operations for efficient computation
- **Wiener Deconvolution**: Optimal recovery with noise suppression
- **Quality Metrics**: PSNR calculation for recovery assessment
- **Side-by-side Comparison**: Original, blurred, and recovered images

#### Usage:
1. Upload any image for blur/recovery demonstration
2. Adjust Gaussian blur sigma parameter (σ = 0.5 to 10.0)
3. Run the blur and recovery process
4. Compare results across three tabs
5. Review PSNR quality metrics

## 📁 Project Structure

```
Module2_TemplateMatching_BlurRecovery/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # This documentation
├── templates/
│   ├── base.html                  # Base template with navigation
│   ├── index.html                 # Main landing page
│   ├── template_matching.html     # Template matching interface
│   └── blur_recovery.html         # Blur recovery interface
├── images/
│   └── templates/                 # Template images for detection
│       ├── IMG_3550.jpg          # Sample template 1
│       └── ...                   # Additional templates
└── static/
    ├── uploads/                   # Temporary uploaded images
    └── results/                   # Processing results
```

## 🔧 Technical Implementation

### Core Classes

#### `Module2Engine`
Main processing engine implementing computer vision algorithms:

- `load_templates()`: Load template images from directory
- `template_matching_with_blur()`: Perform detection and optional blurring
- `gaussian_blur_recovery()`: Apply blur and FFT-based recovery
- `create_gaussian_kernel()`: Generate 2D Gaussian kernels
- `calculate_psnr()`: Compute Peak Signal-to-Noise Ratio

### API Endpoints

#### Template Matching API
- **Endpoint**: `POST /api/template-matching`
- **Parameters**:
  - `image`: Scene image file
  - `blur_detected`: Boolean for regional blurring
  - `blur_sigma`: Blur intensity (0.5-10.0)
- **Response**: Detection results with confidence scores and result image

#### Blur Recovery API
- **Endpoint**: `POST /api/blur-recovery`
- **Parameters**:
  - `image`: Input image file
  - `sigma`: Gaussian blur parameter (0.5-10.0)
- **Response**: Original, blurred, recovered images with PSNR

#### Templates API
- **Endpoint**: `GET /api/templates`
- **Response**: List of available templates with previews

### Performance Characteristics

- **Template Matching**: Sub-second processing for images up to 2048x2048
- **Blur Recovery**: Processing time scales with image size (O(n log n) due to FFT)
- **Memory Usage**: Efficient processing with temporary arrays
- **Supported Formats**: JPG, PNG with automatic format detection

## 🧪 Algorithm Details

### Template Matching Process

1. **Template Loading**: Load all templates from `images/templates/` directory
2. **Grayscale Conversion**: Convert scene and templates to grayscale
3. **Size Validation**: Skip templates larger than scene image
4. **Cross-Correlation**: Apply `cv.matchTemplate()` with `TM_CCOEFF_NORMED`
5. **Threshold Filtering**: Accept detections with confidence ≥ 0.3
6. **Bounding Box Drawing**: Overlay rectangles with template names and confidence
7. **Optional Blurring**: Apply Gaussian blur to detected regions if enabled

### Blur Recovery Process

1. **Preprocessing**: Convert to grayscale and normalize to [0,1] range
2. **Gaussian Blur**: Apply 2D convolution with calculated kernel size
3. **Kernel Preparation**: Create 2D Gaussian kernel and pad to image size
4. **FFT Forward**: Transform blurred image and kernel to frequency domain
5. **Wiener Filtering**: Apply optimal linear filter with noise regularization
6. **FFT Inverse**: Transform back to spatial domain for recovered image
7. **Quality Assessment**: Calculate PSNR between original and recovered

### Key Parameters

#### Template Matching
- **Detection Threshold**: 0.3 (normalized correlation)
- **Blur Kernel Size**: `6σ + 1` pixels (ensuring odd dimensions)
- **Maximum Templates**: Unlimited (limited by directory contents)

#### Blur Recovery
- **Noise Regularization**: λ = 0.01 (prevents division by zero in frequency domain)
- **Sigma Range**: 0.5 to 10.0 (light to heavy blur)
- **PSNR Range**: Typically 15-40 dB for good recovery

## 📊 Expected Results

### Template Matching
- **Detection Rate**: 40–60% for well-matched templates under similar conditions; 20–40% for real-world scenarios
- **Confidence Scores**: Typically 0.4–0.6 depending on scene lighting and template similarity
- **Bounding Boxes**: Correctly overlay detected objects when found
- **Processing Time**: <2 seconds for images up to 2048×2048

#### Example Observations:
- **IMG_3550–IMG_3557**: Detections mostly above 0.5 confidence
- **Slight false positives** possible with low-contrast or occluded templates

#### Limitations / Why Some Results Aren't Perfect:
- **Low Contrast / Poor Lighting**: Templates with similar color or intensity to the background are harder to detect
- **Partial Occlusion**: Template matching requires the full template to be visible; cropped or blocked objects reduce confidence
- **Size Mismatch**: Templates larger than objects in the scene cannot be matched; the current implementation does not include multi-scale matching
- **Rotation / Perspective Changes**: Detection fails if the object is rotated or seen from a different angle than the template
- **Threshold Sensitivity**: A fixed confidence threshold may reject some valid detections or include weak false positives

### Blur Recovery
- **PSNR Values**:
  - Light blur (σ ≤ 2): 25-40 dB
  - Moderate blur (σ = 3-5): 20-30 dB
  - Heavy blur (σ ≥ 6): 15-25 dB
- **Visual Quality**: Good recovery for moderate blur levels

## 🚨 Troubleshooting

### Common Issues

1. **No Templates Found**
   - Ensure template images are in `images/templates/` directory
   - Check file formats (JPG, PNG supported)
   - Verify file permissions

2. **Poor Detection Results**
   - Try different scene images with clearer object visibility
   - Adjust detection threshold (currently fixed at 0.3)
   - Ensure templates are not larger than scene image

3. **Low PSNR Values**
   - Heavy blur (high σ) naturally results in lower PSNR
   - Very low values may indicate processing errors
   - Try with different images or lower blur levels

4. **Memory Issues**
   - Large images may require significant memory for FFT operations
   - Consider resizing images before processing
   - Close other memory-intensive applications

### Performance Optimization

- **Template Organization**: Keep template count reasonable (≤ 20 for best performance)
- **Image Sizing**: Resize large images to 1024x1024 or smaller
- **File Formats**: JPG provides good balance of quality and file size

## 🔬 Advanced Usage

### Custom Template Addition

1. Place new template images in `images/templates/` directory
2. Restart the Flask application
3. Templates will be automatically loaded and available for detection

### Parameter Tuning

#### Template Matching
- Modify detection threshold in `template_matching_with_blur()` method
- Adjust blur kernel size calculation for different blur characteristics
- Implement multi-scale template matching for scale invariance

#### Blur Recovery
- Modify noise regularization parameter (λ) for different noise conditions
- Implement adaptive Wiener filtering based on local image statistics
- Add edge-preserving regularization for better recovery

### Integration Examples

```python
# Create engine instance
engine = Module2Engine()

# Template matching example
scene_image = cv.imread('scene.jpg')
result_image, detections = engine.template_matching_with_blur(
    scene_image, blur_detected=True, blur_sigma=3.0
)

# Blur recovery example
input_image = cv.imread('image.jpg')
results = engine.gaussian_blur_recovery(input_image, sigma=2.5)
psnr = results['psnr']
recovered_image = results['recovered']
```

## 📚 Dependencies

- **Flask 2.3.3**: Web framework for user interface
- **OpenCV 4.8.1**: Computer vision operations and image processing
- **NumPy 1.24.3**: Array operations and FFT computations
- **Matplotlib 3.7.2**: Plotting backend (non-GUI mode)



---
**Author**: Cecilia Muniz Siqueira