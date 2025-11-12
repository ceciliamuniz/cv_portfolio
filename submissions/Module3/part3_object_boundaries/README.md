# Part 3: Object Boundary Detection

This part implements exact object boundary detection using classical computer vision techniques (no ML/DL).

## Methods Implemented

### 1. Contour-Based Detection
Uses thresholding and morphological operations:
1. **Grayscale conversion**
2. **Otsu's thresholding** - Automatic optimal threshold selection
3. **Morphological operations** - Close and open to remove noise
4. **Contour detection** - Border following algorithm
5. **Shape fitting** - Bounding box, convex hull, min circle, ellipse

### 2. GrabCut Segmentation
Graph-cut based iterative segmentation:
1. **Initialize ROI** - Assume object is centered
2. **Build GMMs** - Gaussian Mixture Models for foreground/background
3. **Graph construction** - Pixels as nodes, similarity as edges
4. **Min-cut optimization** - Iteratively refine segmentation
5. **Extract boundaries** - Find contours from final mask

### 3. Boundary Metrics
Computed for each detected object:
- **Area** - Total pixels enclosed
- **Perimeter** - Boundary length
- **Bounding box** - Minimal rectangle
- **Min enclosing circle** - Smallest circle
- **Fitted ellipse** - Best-fit ellipse
- **Centroid** - Geometric center

## Quick Start

1. **Add your dataset** (optional):
   - Place 10+ object images in `data/images/`
   - Or use auto-generated synthetic examples

2. **Process the dataset**:
   ```bash
   python src/process_dataset.py
   ```
   Creates:
   - `outputs/contours/` - Contour detection with shapes
   - `outputs/grabcut/` - GrabCut segmentation results
   - `outputs/boundaries/` - Precise boundaries with metrics
   - `outputs/comparison/` - Side-by-side comparison

3. **Launch the web app**:
   ```bash
   python web/app.py
   ```
   Visit: http://localhost:5002

## Visualization Legend

**Color coding:**
- **Green** - Exact contour outline
- **Blue** - Bounding rectangle
- **Cyan** - Minimum enclosing circle
- **Yellow** - Convex hull
- **Magenta** - Fitted ellipse

## Technical Notes

- **No machine learning** - All classical CV algorithms
- **Works on various objects** - Shapes, tools, everyday items
- **Handles noise** - Morphological operations clean up
- **Multiple methods** - Compare different segmentation approaches
- **Detailed metrics** - Quantitative boundary measurements

## Parameters

**Otsu Thresholding:**
- Automatic threshold from image histogram

**Morphological Operations:**
- Kernel: 5×5 ellipse
- Closing: 2 iterations
- Opening: 1 iteration

**GrabCut:**
- ROI margin: 15% from edges
- Iterations: 5
- Mode: Rectangle initialization
