# Part 2: Edge and Corner Detection

This part implements keypoint detection algorithms for edges and corners.

## Algorithms Implemented

### Edge Detection (Canny Algorithm)
Detects edge keypoints using the Canny edge detection algorithm:
1. **Gaussian blur** to reduce noise
2. **Gradient computation** using Sobel operators
3. **Non-maximum suppression** to thin edges
4. **Double thresholding** to classify strong/weak edges
5. **Edge tracking by hysteresis** to connect edges

### Corner Detection (Harris Algorithm)
Detects corner keypoints using the Harris corner detector:
1. Compute image **gradients** Ix and Iy
2. Build **structure tensor** M from Ix², Iy², and Ix·Iy
3. Calculate **corner response**: R = det(M) - k·trace(M)²
4. **Threshold** to keep strong corners (1% of max response)
5. **Non-maximum suppression** to get local maxima

## Quick Start

1. **Add your dataset** (optional):
   - Place 10+ images in `data/images/`
   - Or use the auto-generated synthetic examples

2. **Process the dataset**:
   ```bash
   python src/process_dataset.py
   ```
   Creates:
   - `outputs/edges/` - Edge keypoints overlay (green)
   - `outputs/corners/` - Corner keypoints overlay (red)
   - `outputs/combined/` - Both edges and corners
   - `outputs/comparison/` - Side-by-side comparison panels

3. **Launch the web app**:
   ```bash
   python web/app.py
   ```
   Visit: http://localhost:5001

## Visualization

- **Green markers** = Edge keypoints
- **Red markers** = Corner keypoints
- **Combined view** shows both on the same image

## Parameters

**Canny Edge Detection:**
- Low threshold: 50
- High threshold: 150
- Gaussian kernel: 5×5, σ=1.4

**Harris Corner Detection:**
- Block size: 2
- Sobel kernel: 3
- Harris parameter k: 0.04
- Response threshold: 1% of maximum
