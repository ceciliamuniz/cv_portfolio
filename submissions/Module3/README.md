# Module 3: Image Analysis - Gradients, LoG, Edge & Corner Detection

Complete implementation of image analysis algorithms for gradient computation, edge detection, corner detection, and object boundary detection.

## Demo Video
## Part 1
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod3_pt1_rec.gif)

## Part 2
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod3_pt2_rec.gif)

## Part 3
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod3_pt3_rec.gif)

## Part 4
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod3_pt4_rec.gif)

## Part 5
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod3_pt5_rec.gif)

## 📋 Requirements Implemented

### 1. Gradient Image Computation
- ✅ **Gradient Magnitude**: Computed using Sobel operators `sqrt(gx² + gy²)`
- ✅ **Gradient Angle**: Computed using `arctan2(gy, gx)` and mapped to [0, 180) degrees
- ✅ Both saved as normalized images for visualization

### 2. Laplacian of Gaussian (LoG)
- ✅ Gaussian blur applied first to reduce noise
- ✅ Laplacian operator applied to detect edges
- ✅ Comparison with gradient images provided

### 3. Edge Keypoint Detection
- ✅ **Canny Edge Detection Algorithm** implemented
  - Gaussian blur for noise reduction
  - Gradient computation using Sobel
  - Non-maximum suppression
  - Double thresholding (strong/weak edges)
  - Edge tracking by hysteresis
- ✅ Edge keypoints marked in **GREEN**

### 4. Corner Keypoint Detection
- ✅ **Harris Corner Detection Algorithm** implemented
  - Gradient computation (Ix, Iy)
  - Structure tensor M from Ix², Iy², Ix·Iy
  - Corner response: R = det(M) - k·trace(M)²
  - Thresholding (top 1% of responses)
  - Non-maximum suppression
- ✅ Corner keypoints marked in **RED**

### 5. Object Boundary Detection
- ✅ **Multi-technique boundary detection**:
  - Canny edge detection
  - Morphological operations (closing)
  - Contour detection with area filtering
  - Convex hull fitting
  - Bounding rectangles (axis-aligned and minimum area)
- ✅ No machine learning or deep learning used (pure OpenCV)

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python numpy
```

### Run the Complete Pipeline
```bash
cd submissions/Module3
python process_all.py
```

This will process all 10 images in the `images/` folder and generate:
- Gradient magnitude images
- Gradient angle images
- Laplacian of Gaussian images
- Edge keypoint visualizations
- Corner keypoint visualizations
- Combined keypoint visualizations
- Object boundary detections
- Comparison panels

## 📂 Output Structure

```
outputs/
├── gradients/
│   ├── magnitude/          # Gradient magnitude images
│   └── angle/              # Gradient angle images (0-180°)
├── log/                    # Laplacian of Gaussian images
├── edges/                  # Edge keypoints (green markers)
├── corners/                # Corner keypoints (red markers)
├── combined/               # Both edges and corners
├── boundaries/             # Object boundaries with:
│                          #   - Green: Contours
│                          #   - Blue: Convex hull
│                          #   - Red: Bounding rectangle
│                          #   - Yellow: Minimum area rectangle
├── comparison/             # Side-by-side comparison panels
```

## 🔬 Algorithm Details

### Gradient Computation
```python
# Sobel operators in x and y directions
gx = Sobel(image, dx=1, dy=0, ksize=3)
gy = Sobel(image, dx=0, dy=1, ksize=3)

# Magnitude
magnitude = sqrt(gx² + gy²)

# Angle (in degrees, unsigned)
angle = arctan2(gy, gx) * 180/π mod 180
```

### Laplacian of Gaussian (LoG)
```python
# Step 1: Gaussian blur
blurred = GaussianBlur(image, kernel=5x5, sigma=1.0)

# Step 2: Laplacian
LoG = Laplacian(blurred)
```

### Canny Edge Detection
```python
# 1. Noise reduction: Gaussian blur
# 2. Gradient calculation: Sobel operators
# 3. Non-maximum suppression: Thin edges
# 4. Double threshold: Classify strong/weak edges
# 5. Edge tracking: Hysteresis to connect edges
```

### Harris Corner Detection
```python
# 1. Compute gradients Ix, Iy
# 2. Build structure tensor:
#    M = [Σ(Ix²)    Σ(Ix·Iy)]
#        [Σ(Ix·Iy)  Σ(Iy²)  ]
# 3. Corner response:
#    R = det(M) - k·trace(M)²
# 4. Threshold: Keep R > 0.01 * max(R)
# 5. Non-maximum suppression
```

### Object Boundary Detection
```python
# 1. Preprocess: Gaussian blur
# 2. Edge detection: Canny
# 3. Morphological closing: Fill gaps
# 4. Find contours: External contours only
# 5. Filter by area: Remove noise
# 6. Fit shapes: Convex hull, rectangles
```

## 📊 Comparison: Gradient vs LoG

### Gradient Magnitude
- Shows **all intensity changes** in the image
- Good for detecting edges, textures, and details
- Magnitude is always positive
- Angle provides edge orientation

### Laplacian of Gaussian (LoG)
- Highlights **rapid intensity changes** (edges)
- More sensitive to fine details and noise
- Can be positive or negative (visualized as absolute value)
- Zero-crossings indicate edges
- Better for detecting blobs and ridges

**Key Difference**: 
- Gradient uses **first derivative** (rate of change)
- LoG uses **second derivative** (rate of change of rate of change)
- LoG is more sensitive to fine details
- Gradient provides directional information

## 🎨 Visualization Legend

### Keypoint Markers
- 🟢 **Green dots**: Edge keypoints (Canny)
- 🔴 **Red dots**: Corner keypoints (Harris)

### Boundary Detection
- 🟢 **Green lines**: Object contours
- 🔵 **Blue lines**: Convex hull
- 🔴 **Red rectangles**: Axis-aligned bounding boxes
- 🟡 **Yellow rectangles**: Minimum area bounding boxes

## 📝 Parameters Used

### Gradient Computation
- Sobel kernel size: 3×3
- Output: Normalized to [0, 255] for visualization

### LoG Computation
- Gaussian kernel: 5×5
- Gaussian sigma: 1.0
- Laplacian kernel: 5×5

### Canny Edge Detection
- Low threshold: 50
- High threshold: 150
- L2 gradient: Enabled

### Harris Corner Detection
- Block size: 2
- Sobel kernel: 3×3
- Harris parameter k: 0.04
- Response threshold: 1% of maximum

### Boundary Detection
- Canny thresholds: 50, 150
- Morphological kernel: 5×5 rectangle
- Minimum contour area: 100 pixels

## 🔍 Understanding the Results

### When to use each detector:

**Edge Detection (Canny)**
- Detecting object boundaries
- Finding lines and curves
- Extracting contours
- Many keypoints along edges

**Corner Detection (Harris)**
- Feature matching
- Object tracking
- Image alignment
- Sparse keypoints at corners

**LoG**
- Blob detection
- Scale-space analysis
- Fine detail enhancement
- Zero-crossing edge detection

## 📚 References

- Canny, J. (1986). "A Computational Approach to Edge Detection"
- Harris, C. & Stephens, M. (1988). "A Combined Corner and Edge Detector"
- Marr, D. & Hildreth, E. (1980). "Theory of Edge Detection"

## 🎓 Learning Outcomes

This implementation demonstrates:
1. **Gradient-based edge detection** principles
2. **Structure tensor** analysis for corner detection
3. **Multi-scale analysis** with LoG
4. **Contour detection** and shape fitting
5. **Classical computer vision** without ML/DL

---

**Author**: Cecilia Muniz Siqueira
