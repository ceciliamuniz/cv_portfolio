# Module 4: Advanced Image Stitching with SIFT Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)

A comprehensive implementation of image stitching featuring a complete **SIFT (Scale-Invariant Feature Transform)** algorithm built from scratch, enhanced **RANSAC** optimization, and advanced blending techniques. This module demonstrates both theoretical understanding and practical implementation of computer vision panorama creation.

## 🎯 **Project Overview**

This module implements the complete image stitching pipeline with:

- **Complete SIFT Implementation from Scratch**: Gaussian pyramids, DoG computation, keypoint detection, orientation assignment, and 128D descriptor computation
- **Enhanced RANSAC Optimization**: Adaptive iteration calculation, confidence-based termination, and homography refinement
- **Advanced Blending Techniques**: Multi-band blending using Laplacian pyramids, feather blending, and seam optimization
- **Comprehensive Comparison Framework**: Side-by-side analysis with OpenCV SIFT and mobile device panoramas
- **Quality Assessment**: Automated panorama quality metrics including sharpness, exposure consistency, and seam visibility

## 📋 **Requirements**

### **Image Requirements**
- **Landscape Mode**: Minimum 4 images (horizontal orientation)
- **Portrait Mode**: Minimum 8 images (vertical orientation)
- **Resolution**: 100x100 to 4000x4000 pixels
- **Format**: JPEG, PNG, or other common formats
- **Overlap**: 20-40% overlap between consecutive images for best results

### **Software Dependencies**
```
Python 3.8+
OpenCV 4.5+ (with contrib modules)
NumPy 1.19+
Flask 2.0+
Pathlib (standard library)
```

## 🚀 **Quick Start**

### **1. Installation**
```bash
cd submissions/Module4_ImageStitching
pip install -r requirements.txt
```

### **2. Run the Application**
```bash
python app.py
```
Then visit `http://localhost:5010` in your browser.

### **3. Alternative: Integration Mode**
Run from the project root and visit `/module4` in the main application.

## 🏗️ **Architecture & Implementation**

### **Core Components**

#### **1. SIFT Implementation (`sift_scratch.py`)**
```python
# Complete SIFT pipeline
features = {
    'gaussian_pyramid': build_gaussian_pyramid(image),
    'dog_pyramid': compute_dog(gaussian_pyramid), 
    'keypoints': detect_keypoints(dog_pyramid),
    'orientations': assign_orientations(keypoints, gaussian_pyramid),
    'descriptors': compute_descriptors(oriented_keypoints, gaussian_pyramid)
}
```

**Key Features:**
- **Scale-Space Construction**: 4 octaves × 3 scales with proper σ sampling
- **Extrema Detection**: 26-connected neighborhood analysis with sub-pixel refinement
- **Edge Filtering**: Harris corner response for eliminating edge-like features
- **Orientation Assignment**: Gradient histogram analysis with peak interpolation
- **Descriptor Computation**: 4×4 spatial grid × 8 orientation bins = 128D vectors
- **Robust Matching**: Lowe's ratio test with adaptive thresholds

#### **2. Enhanced RANSAC (`stitching.py`)**
```python
H, inliers = estimate_homography_ransac(
    points1, points2,
    threshold=4.0,
    max_iterations=5000,
    confidence=0.995
)
```

**Enhancements:**
- **Adaptive Iterations**: Dynamically adjusts based on inlier ratio
- **Confidence-Based Termination**: Stops early when confidence threshold met
- **Homography Refinement**: Least-squares optimization using all inliers
- **Geometric Validation**: Robust error handling for degenerate cases

#### **3. Multi-Band Blending**
```python
panorama = advanced_warp_and_blend(
    images, homographies,
    reference=0,
    blend_mode='multiband'  # or 'feather', 'simple'
)
```

**Blending Options:**
- **Multi-band**: Laplacian pyramid blending for seamless transitions
- **Feather**: Distance-based weight blending
- **Simple**: Basic averaging in overlap regions

### **4. Quality Assessment**
```python
metrics = assess_panorama_quality(panorama)
# Returns: sharpness, contrast, seam_visibility, exposure_consistency, overall_quality
```

## 🔬 **Comparison Features**

### **SIFT Implementation Comparison**
- **Performance Metrics**: Processing time, keypoint count, match quality
- **Accuracy Analysis**: Feature repeatability and matching precision
- **Visual Comparison**: Side-by-side feature visualization

### **Mobile Panorama Comparison**
- **Quality Metrics**: Resolution, sharpness, exposure consistency
- **Algorithm Analysis**: Stitching approach differences
- **Performance Benchmarking**: Processing time and resource usage

## 📊 **Usage Examples**

### **Basic Panorama Creation**
```python
from sift_scratch import *
from stitching import *

# Load images
images = [cv.imread(f'image_{i}.jpg') for i in range(4)]

# Extract features and match
feature_data = []
for i in range(len(images)-1):
    matches = compute_features_and_matches(
        images[i], images[i+1], sift_scratch
    )
    feature_data.append(matches)

# Estimate homographies
homographies = [np.eye(3)]
for i, data in enumerate(feature_data):
    H, _ = estimate_homography_ransac(
        data['points1'], data['points2']
    )
    homographies.append(homographies[-1] @ np.linalg.inv(H))

# Create panorama
panorama = advanced_warp_and_blend(
    images, homographies, blend_mode='multiband'
)

# Assess quality
quality = assess_panorama_quality(panorama)
print(f"Quality Score: {quality['overall_quality']:.2f}")
```

### **SIFT Comparison Analysis**
```python
# Compare implementations
comparison = compare_with_opencv_sift(image1, image2)
print(f"Custom SIFT: {comparison['custom']['keypoints_count']} keypoints")
print(f"OpenCV SIFT: {comparison['opencv']['keypoints_count']} keypoints")
print(f"Speed ratio: {comparison['comparison']['speed_ratio']:.2f}x")
```

## 📈 **Performance Benchmarks**

### **Typical Performance (4 images, 1920×1080)**
- **Custom SIFT**: ~15-25 seconds
- **OpenCV SIFT**: ~2-5 seconds
- **Memory Usage**: ~500MB-1GB peak
- **Output Quality**: Comparable to professional tools

### **Quality Metrics Range**
- **Excellent**: Overall quality > 0.7
- **Good**: Overall quality 0.4-0.7
- **Needs Improvement**: Overall quality < 0.4

## 🛠️ **API Reference**

### **REST Endpoints**

#### `POST /api/stitch`
Main stitching endpoint

**Parameters:**
- `images`: Multi-part file upload (4+ images)
- `blend_mode`: 'multiband', 'feather', or 'simple'
- `use_custom_sift`: boolean (default: true)
- `comparison_mode`: boolean (default: false)

**Response:**
```json
{
  "success": true,
  "panorama": "data:image/jpeg;base64,...",
  "statistics": {
    "input_images": 4,
    "output_resolution": "3840x1080",
    "processing_time": "23.45s"
  },
  "quality_metrics": {
    "overall_quality": 0.78,
    "sharpness": 892.3,
    "seam_visibility": 45.2
  }
}
```

#### `POST /api/compare`
SIFT implementation comparison

#### `POST /api/mobile-compare`
Mobile panorama comparison

## 🔍 **Technical Deep Dive**

### **SIFT Algorithm Implementation**

1. **Scale-Space Construction**
   - Gaussian pyramid with 4 octaves, 3 scales per octave
   - Proper σ progression: σ = 1.6 × 2^(scale/3)
   - Image downsampling by factor of 2 per octave

2. **DoG Extrema Detection**
   - 26-connected neighborhood analysis
   - Sub-pixel localization using Taylor expansion
   - Edge response filtering via Harris corner measure

3. **Orientation Assignment**
   - 36-bin gradient histogram in keypoint neighborhood
   - Gaussian weighting window (σ = 1.5 × scale)
   - Peak detection with parabolic interpolation

4. **Descriptor Construction**
   - 16×16 pixel neighborhood around keypoint
   - 4×4 spatial subregions × 8 orientation bins
   - Trilinear interpolation for robust descriptors
   - Normalization and illumination invariance

### **RANSAC Optimization**

- **Adaptive Iterations**: N = log(1-p) / log(1-w^4)
  - p: desired confidence (0.995)
  - w: inlier ratio estimate
- **Early Termination**: Stop when >60% inliers found
- **Homography Refinement**: Least-squares on all inliers

## 📸 **Best Practices for Image Acquisition**

### **Recommended Capture Technique**
1. **Overlap**: 20-40% between consecutive images
2. **Exposure**: Use manual or exposure lock for consistency
3. **Focus**: Lock focus to prevent changes between shots
4. **Rotation**: Keep camera level, rotate around optical center
5. **Lighting**: Avoid dramatic lighting changes

### **Mobile Comparison Guidelines**
1. Use same scene and lighting conditions
2. Capture mobile panorama immediately after individual shots
3. Match approximate field of view
4. Save both results for quality comparison

## 🐛 **Troubleshooting**

### **Common Issues**

**"Insufficient matches" Error**
- Increase image overlap (>30%)
- Ensure consistent lighting
- Use images with rich texture/features

**"Homography estimation failed"**
- Check image sequence order
- Verify image quality and sharpness
- Reduce RANSAC threshold if needed

**Poor Quality Score (<0.4)**
- Improve image overlap and alignment
- Use manual exposure for consistency
- Try different blending modes

**Visible Seams**
- Switch to multi-band blending
- Ensure proper exposure matching
- Check for moving objects in scene

## 🚧 **Limitations & Future Improvements**

### **Current Limitations**
- Sequential stitching only (no global optimization)
- Limited to planar scenes (no parallax handling)
- No automatic exposure compensation
- Processing time slower than optimized implementations

### **Future Enhancements**
- Bundle adjustment for global optimization
- Parallax handling for closer objects
- GPU acceleration for real-time processing
- Advanced seam finding algorithms
- Automatic exposure and color correction

## 📚 **References & Theory**

- **Lowe, D.G.** (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*
- **Fischler, M.A. & Bolles, R.C.** (1981). Random Sample Consensus. *Communications of the ACM*
- **Brown, M. & Lowe, D.G.** (2007). Automatic Panoramic Image Stitching. *IJCV*
- **Szeliski, R.** (2010). Computer Vision: Algorithms and Applications. *Springer*

---

## 📝 **Assignment Compliance**

✅ **Complete SIFT Implementation**: All components implemented from scratch  
✅ **RANSAC Optimization**: Enhanced with adaptive iterations and refinement  
✅ **OpenCV Comparison**: Comprehensive performance and accuracy analysis  
✅ **4+ Landscape/8+ Portrait**: Flexible image count with validation  
✅ **Mobile Comparison**: Framework for comparing with device panoramas  
✅ **Quality Assessment**: Automated metrics and recommendations  

---

**Author**: Cecilia Muniz Siqueira  
**Course**: Computer Vision Module 4  
**Date**: November 2025  
**License**: MIT