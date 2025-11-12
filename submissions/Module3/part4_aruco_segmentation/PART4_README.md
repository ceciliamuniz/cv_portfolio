# Part 4: ArUco Marker-Based Object Segmentation

## Overview
Implements object segmentation using ArUco markers placed on the boundary of non-rectangular objects. The system detects markers and uses their positions to identify and segment the object from the background.

## Methodology

### 1. ArUco Marker Detection
- **Dictionary**: DICT_4X4_50 (50 unique 4×4 markers)
- **Detection**: OpenCV's `ArucoDetector` for robust marker identification
- **Output**: Marker corners, IDs, and center positions

### 2. Segmentation Methods
The system implements three segmentation approaches:

#### A. Convex Hull Method
- Creates the smallest convex polygon containing all marker centers
- **Pros**: Simple, robust, works well for convex objects
- **Cons**: Cannot handle concave object boundaries

#### B. Contour Method
- Creates circular regions around each marker
- Dilates regions to connect nearby markers
- Finds external contour of connected regions
- **Pros**: Better handles concave regions
- **Cons**: Sensitive to dilation kernel size

#### C. Alpha Shape Method (Optional)
- Uses Delaunay triangulation of marker points
- Filters triangles by edge length (alpha parameter)
- Creates more accurate boundary for complex shapes
- **Pros**: Best for non-convex objects
- **Cons**: Requires scipy, sensitive to alpha parameter

### 3. Metrics
For each segmentation:
- Number of markers detected
- Segmented area (pixels)
- Perimeter length (pixels)
- Marker IDs detected

## Usage

### Step 1: Generate ArUco Markers
```bash
cd part4_aruco_segmentation
python aruco_segmentation.py
```

This generates 20 printable markers in `aruco_markers/`:
- Files: `aruco_marker_00.png` through `aruco_marker_19.png`
- Each marker has a white border for easier cutting/mounting
- Print on white paper for best results

### Step 2: Prepare Your Object
1. Choose a **non-rectangular object** (e.g., curved vase, irregular shape, organic object)
2. Print 4-10 ArUco markers
3. Stick markers **around the boundary** of the object
   - Space them evenly
   - Ensure markers are flat and visible
   - Use markers with different IDs

### Step 3: Capture Images
Capture **at least 10 images** with variations:
- **Distances**: Close-up, medium range, far
- **Angles**: Front view, side views, top view, oblique angles
- **Lighting**: Ensure markers are clearly visible
- **Resolution**: Higher is better (min 640×480)

Save images to: `part4_aruco_segmentation/images/`

### Step 4: Process Images
```bash
python aruco_segmentation.py
```

**Outputs** (saved to `outputs/`):
- `convex_hull/`: Results using convex hull method
- `contour/`: Results using contour method
- For each image:
  - `*_mask.png`: Binary segmentation mask
  - `*_segmentation.jpg`: Visualization with markers and boundary
- `processing_summary.json`: Detailed results for all images

## File Structure
```
part4_aruco_segmentation/
├── aruco_segmentation.py      # Main processing script
├── README.md                  # This file
├── images/                    # Input images (add yours here)
├── aruco_markers/            # Generated printable markers
│   ├── aruco_marker_00.png
│   ├── aruco_marker_01.png
│   └── ...
└── outputs/                  # Results
    ├── convex_hull/
    │   ├── image1_mask.png
    │   ├── image1_segmentation.jpg
    │   └── processing_summary.json
    └── contour/
        └── ...
```
