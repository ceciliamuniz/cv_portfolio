# Part 4: ArUco Marker-Based Object Segmentation

Segment non-rectangular objects using ArUco fiducial markers placed on the boundary.

## Overview

This part implements object segmentation for non-rectangular objects by:
1. Detecting ArUco markers placed along the object boundary
2. Using marker positions as boundary hints
3. Refining segmentation with classical CV techniques (GrabCut, morphological operations)
4. Extracting precise object boundaries

**No machine learning or deep learning is used - only classical CV algorithms.**

## Quick Start (With Real Images)

### Step 1: Generate Printable Markers
```bash
python src/generate_markers.py
```
This creates 20 ArUco markers in `data/markers/` that you can print.

### Step 2: Prepare Your Object
1. Print the generated markers (preferably one marker per page)
2. Cut out the markers carefully
3. Choose a **non-rectangular object** (e.g., bottle, vase, toy, tool)
4. Attach 4-8 markers around the object's boundary
   - Distribute markers evenly around the perimeter
   - Make sure markers are visible from all angles
   - Attach securely (tape works well)

### Step 3: Capture Images
Capture at least 10 images of your object:
- **Different angles:** Front, side, top, angled views
- **Different distances:** Close-up and farther away
- **Good lighting:** Avoid shadows on markers
- **Steady camera:** Reduce motion blur

Save all images to `data/images/`

### Step 4: Process Dataset
```bash
python src/process_dataset.py
```
This will:
- Detect ArUco markers in each image
- Segment the object using marker positions
- Calculate boundary metrics
- Generate visualizations

### Step 5: View Results
```bash
python web/app.py
```
Visit: http://localhost:5003

## Quick Start (With Synthetic Test Images)

If you don't have physical markers yet, you can test with synthetic images:

```bash
# Just run the processor - it will auto-generate test images
python src/process_dataset.py

# View results
python web/app.py  # http://localhost:5003
```

## How It Works

### ArUco Marker Detection
ArUco markers are binary square patterns optimized for fast, robust detection:

1. **Edge detection** - Find quadrilaterals in the image
2. **Perspective correction** - Unwarp candidates using homography
3. **Bit extraction** - Read binary pattern from square grid
4. **Error correction** - Validate using Hamming code
5. **ID matching** - Match pattern to dictionary (DICT_4X4_50)

### Marker-Based Segmentation
Using detected markers to segment the object:

1. **Marker centers** - Extract (x,y) coordinates of each marker
2. **Convex hull** - Create initial boundary from marker positions
3. **Hull expansion** - Expand by 30% to ensure object coverage
4. **GrabCut refinement** - Iterative graph-cut segmentation (3 iterations)
5. **Morphological cleanup** - Close holes, remove noise
6. **Boundary extraction** - Find final precise contour

### Boundary Analysis
Compute metrics for the segmented object:
- **Area** - Total pixels enclosed by boundary
- **Perimeter** - Total boundary length
- **Bounding box** - Minimal axis-aligned rectangle
- **Centroid** - Geometric center of mass

## Outputs

All results saved to `outputs/`:

- `detected_markers/` - Images with detected markers highlighted
- `segmentation/` - Segmentation result with marker overlay
- `boundary/` - Clean boundary with metrics
- `comparison/` - Side-by-side comparison of all steps

## Visualization Legend

**Colors:**
- **Green outline** - Detected marker borders
- **Green circles** - Marker center points
- **Magenta polygon** - Convex hull of markers
- **Yellow dots** - Marker positions
- **Green overlay** - Segmented object region
- **Blue rectangle** - Bounding box
- **White text** - Metrics (area, perimeter, etc.)

## Tips for Best Results

### Marker Placement
✅ **Do:**
- Place 4-8 markers around the boundary
- Distribute evenly around perimeter
- Ensure markers are flat and visible
- Use consistent lighting

❌ **Don't:**
- Place markers too close together
- Cover markers with shadows
- Use torn or damaged markers
- Place markers inside the object area

### Image Capture
✅ **Do:**
- Capture from multiple angles (front, side, top, diagonal)
- Vary distance (near and far)
- Use good lighting (natural or bright indoor)
- Keep camera steady

❌ **Don't:**
- Use motion-blurred images
- Capture with extreme perspective distortion
- Include cluttered backgrounds
- Use very low resolution

## Technical Details

**ArUco Dictionary:** DICT_4X4_50
- 4x4 bit pattern
- 50 unique markers (IDs 0-49)
- Hamming distance for error correction

**Segmentation Parameters:**
- Hull expansion ratio: 1.3 (30% larger)
- GrabCut iterations: 3
- Morphological kernel: 5x5 ellipse
- Closing iterations: 2
- Opening iterations: 1

**Requirements:**
- opencv-python >= 4.8.0 (includes aruco module)
- numpy >= 1.26.0
- scipy >= 1.11.0
- Flask >= 3.0.0

## Troubleshooting

**No markers detected:**
- Check marker print quality
- Ensure good lighting
- Try different camera angles
- Verify markers are DICT_4X4_50 format

**Poor segmentation:**
- Add more markers (6-8 recommended)
- Improve marker distribution around boundary
- Use higher resolution images
- Ensure contrast between object and background

**Boundary not accurate:**
- Adjust `expand_ratio` in `segment_object_from_markers()` (try 1.2-1.5)
- Increase GrabCut iterations (5-7)
- Check that markers are actually on the boundary

## Example Workflow

```bash
# 1. Generate markers
python src/generate_markers.py

# 2. Print markers 0-7 from data/markers/

# 3. Attach to your object (e.g., a coffee mug)

# 4. Capture 10 images from different angles
#    Save to data/images/

# 5. Process
python src/process_dataset.py

# 6. View results
python web/app.py
# Open http://localhost:5003

# 7. Click on any image to see:
#    - Detected markers with IDs
#    - Segmentation result
#    - Boundary with metrics
#    - Complete processing pipeline
```

## Assignment Deliverables

When submitting Part 4:
1. ✅ 10+ images with ArUco markers on non-rectangular object
2. ✅ Processed results showing marker detection
3. ✅ Object segmentation visualizations
4. ✅ Boundary metrics (area, perimeter, bounding box)
5. ✅ Web interface screenshots or demo

## References

- ArUco: Original paper by Garrido-Jurado et al., 2014
- OpenCV ArUco documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- GrabCut: "GrabCut" Interactive Foreground Extraction (Rother et al., 2004)
