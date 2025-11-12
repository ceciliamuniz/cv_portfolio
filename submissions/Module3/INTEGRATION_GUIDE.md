# Module 3 Parts 4 & 5: Integration Guide for CV Portfolio

## Overview
This guide explains how to integrate the ArUco segmentation (Part 4) and SAM2 comparison (Part 5) into your CV Portfolio website.

## Files Created

### Part 4: ArUco Segmentation
```
part4_aruco_segmentation/
├── aruco_segmentation.py      # Main processing script
├── PART4_README.md           # Documentation
├── images/                    # Input images (your captures)
├── aruco_markers/            # Generated printable markers
└── outputs/                  # Segmentation results
    ├── convex_hull/
    └── contour/
```

### Part 5: SAM2 Comparison
```
part5_sam2_comparison/
├── sam2_comparison.py        # Comparison script
├── README.md                 # Documentation
├── checkpoints/             # SAM2 model weights (download)
└── comparison_results/      # Comparison outputs
```

### Web Integration
```
web_integration/
├── routes.py                # Flask Blueprint routes
├── templates/
│   ├── module3_index.html  # Main Module 3 page
│   └── module3_part4.html  # ArUco upload interface
└── static/
    ├── results/            # Processed images
    └── uploads/            # Temporary uploads
```

## Integration Steps

### Step 1: Add Blueprint to Main App

In your main `app.py` (in `CV_Module2/submissions/Module2_TemplateMatching_BlurRecovery/`):

```python
from pathlib import Path
import sys

# Add Module 3 to path
module3_path = Path(__file__).parent.parent / 'Module3' / 'web_integration'
sys.path.insert(0, str(module3_path))

from routes import module3_bp

# Register blueprint
app.register_blueprint(module3_bp)
```

### Step 2: Update Navigation

Add Module 3 link to your main navigation menu:

```html
<nav>
    <a href="/">Home</a>
    <a href="/module1">Module 1: Distance Measurement</a>
    <a href="/module2">Module 2: Template Matching & Blur Recovery</a>
    <a href="/module3">Module 3: Advanced Image Analysis</a>
</nav>
```

### Step 3: Install Dependencies

```bash
cd CV_Module2/submissions/Module3
pip install -r requirements_parts45.txt
```

For SAM2 (Part 5):
```bash
# Install PyTorch (choose based on your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU
# OR
pip install torch torchvision  # CPU

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download checkpoint (choose one):
# SAM2 Large (best quality): https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
# SAM2 Small (faster): https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
# SAM2 Tiny (fastest): https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

# Save to: part5_sam2_comparison/checkpoints/
```

### Step 4: Prepare ArUco Markers

Generate markers for printing:
```bash
cd part4_aruco_segmentation
python aruco_segmentation.py
```

This creates printable markers in `aruco_markers/`. Print these on white paper.

### Step 5: Capture Images

1. **Choose a non-rectangular object** (e.g., vase, bottle, organic shape)
2. **Print and stick 4-10 ArUco markers** around the boundary
3. **Capture 10+ images** with variations:
   - Distances: close, medium, far (3-4 each)
   - Angles: front, side, top, oblique (mix)
   - Good lighting, markers visible
4. **Save to** `part4_aruco_segmentation/images/`

### Step 6: Process ArUco Segmentation

```bash
python aruco_segmentation.py
```

Results saved to `outputs/convex_hull/` and `outputs/contour/`.

### Step 7: Run SAM2 Comparison (Optional)

If you have SAM2 installed and checkpoint downloaded:

```bash
cd ../part5_sam2_comparison
python sam2_comparison.py
```

Results saved to `comparison_results/`.

### Step 8: Launch Web Interface

```bash
cd ../../submissions/Module2_TemplateMatching_BlurRecovery
python app.py
```

Navigate to: `http://localhost:5000/module3`

## URL Structure

Once integrated, your CV Portfolio will have these URLs:

### Main Pages
- `/module3` - Module 3 overview and navigation
- `/module3/gallery` - Gallery view of all results

### Individual Parts
- `/module3/part1-gradient-log` - Gradient & LoG visualizations
- `/module3/part2-keypoints` - Edge & corner keypoints
- `/module3/part3-boundaries` - Object boundary detection
- `/module3/part4-aruco` - ArUco segmentation (with upload)
- `/module3/part5-sam2-comparison` - SAM2 comparison (with upload)

### API
- `/module3/api/stats` - JSON statistics for all parts

## Features

### Part 4: ArUco Segmentation
- ✅ Upload image with ArUco markers
- ✅ Choose segmentation method (convex hull or contour)
- ✅ Real-time processing
- ✅ Display metrics (markers detected, area, perimeter)
- ✅ Show segmentation visualization and binary mask
- ✅ List detected marker IDs

### Part 5: SAM2 Comparison
- ✅ Upload same images from Part 4
- ✅ Run both ArUco and SAM2 segmentation
- ✅ Calculate comparison metrics (IoU, Dice, Precision, Recall)
- ✅ 4-panel visualization (original, ArUco, SAM2, overlap)
- ✅ Color-coded overlap analysis

### Gallery View
- ✅ Display results from all 5 parts
- ✅ Grid layout with thumbnails
- ✅ Quick navigation between parts
- ✅ Statistics dashboard

## Customization

### Changing ArUco Dictionary
In `aruco_segmentation.py`:
```python
# Default: DICT_4X4_50 (50 unique 4×4 markers)
# Options: DICT_4X4_100, DICT_5X5_50, DICT_6X6_50, etc.
aruco_dict_type = cv.aruco.DICT_4X4_50
```

### Adjusting Segmentation Methods
In `aruco_segmentation.py`, class `ArucoSegmentation`, method `segment_object()`:

**Convex Hull**: Works as-is for convex objects
**Contour**: Adjust dilation kernel size
```python
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))  # Increase for larger gaps
```

**Alpha Shape**: Adjust alpha parameter
```python
alpha = 100.0  # Increase for more convex, decrease for more concave
```

### SAM2 Model Selection
In `sam2_comparison.py`:
```python
# Choose model size (tiny, small, large)
model_cfg = "sam2_hiera_l.yaml"  # large (best quality)
# model_cfg = "sam2_hiera_s.yaml"  # small (faster)
# model_cfg = "sam2_hiera_t.yaml"  # tiny (fastest)

checkpoint_path = "checkpoints/sam2_hiera_large.pt"
```

### Styling
Modify templates in `web_integration/templates/`:
- Colors: Change `#667eea` and `#764ba2` (gradients)
- Layout: Adjust grid columns in `.stats-grid`, `.metrics`
- Fonts: Change `font-family` in `<style>` blocks

## Deployment Considerations

### For Production
1. **File Uploads**: Set maximum file size
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
   ```

2. **Security**: Validate file types and sanitize filenames
   ```python
   from werkzeug.utils import secure_filename
   filename = secure_filename(file.filename)
   ```

3. **Storage**: Use cloud storage for processed images
   - AWS S3
   - Google Cloud Storage
   - Azure Blob Storage

4. **SAM2 Performance**:
   - Use GPU for faster inference
   - Cache model in memory (don't reload per request)
   - Consider batch processing

5. **Rate Limiting**: Prevent abuse
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   @limiter.limit("10 per minute")
   ```

### Static File Serving
For production, serve static files (images) via:
- **Nginx**: Reverse proxy with static file caching
- **CDN**: CloudFlare, AWS CloudFront
- **Flask**: Use `send_from_directory()` with caching headers

## Troubleshooting

### ArUco Markers Not Detected
- **Check dictionary type**: Must match generated markers
- **Improve lighting**: Markers need high contrast
- **Increase resolution**: Higher quality images
- **Verify marker visibility**: Ensure not blurred or occluded

### SAM2 Out of Memory
- **Use smaller model**: Switch to `sam2_hiera_t` (tiny)
- **Resize images**: Reduce to 800×600 or 1024×768
- **Use CPU**: Set `device='cpu'` in SAM2Segmentation

### Low IoU Scores
- **Check alignment**: Ensure same image used for both methods
- **Add more markers**: 6-8 markers recommended
- **Even spacing**: Distribute markers uniformly
- **Try different method**: Contour vs. convex hull

### Import Errors
```bash
# If "module not found" errors:
export PYTHONPATH="${PYTHONPATH}:/path/to/CV_Module2/submissions/Module3"

# Or in code:
import sys
sys.path.insert(0, '/path/to/module3')
```

## Testing

### Unit Tests
Create `tests/test_aruco.py`:
```python
import pytest
from aruco_segmentation import ArucoSegmentation

def test_marker_detection():
    segmenter = ArucoSegmentation()
    # Add test image with known markers
    # Assert correct number detected
```

### Integration Tests
Create `tests/test_routes.py`:
```python
def test_upload_route(client):
    response = client.post('/module3/part4-aruco', 
                          data={'file': test_image})
    assert response.status_code == 200
```

## Performance Benchmarks

Expected processing times (on mid-range CPU):

| Operation | Time | Notes |
|-----------|------|-------|
| ArUco Detection | 50-200ms | Fast, CPU-only |
| Convex Hull | 10-50ms | Very fast |
| Contour Method | 100-300ms | Morphology operations |
| SAM2 (CPU) | 5-15s | Slow, use GPU |
| SAM2 (GPU) | 500ms-2s | Much faster |

## Next Steps

1. **Capture your 10+ images** with ArUco markers on object boundary
2. **Process with ArUco segmentation**
3. **Optional: Compare with SAM2** (requires model download)
4. **Integrate into main app** using this guide
5. **Deploy to CV Portfolio** website

## Resources

- **ArUco Documentation**: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- **SAM2 GitHub**: https://github.com/facebookresearch/segment-anything-2
- **Flask Blueprints**: https://flask.palletsprojects.com/en/latest/blueprints/
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

---

## Summary

You now have a complete implementation of:
- ✅ **Part 4**: ArUco marker-based object segmentation
- ✅ **Part 5**: SAM2 model comparison
- ✅ **Web Interface**: Flask routes and templates
- ✅ **Documentation**: READMEs and integration guide
- ✅ **Structure**: Ready for CV Portfolio integration

The system is ready to process your captured images and compare classical CV (ArUco) with state-of-the-art AI (SAM2) for object segmentation!
