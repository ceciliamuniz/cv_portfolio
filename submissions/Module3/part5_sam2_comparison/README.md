# Part 5: SAM2 Comparison

## Overview
Compares ArUco marker-based segmentation (Part 4) with Meta's **Segment Anything Model 2 (SAM2)**, a state-of-the-art foundation model for image segmentation.

## SAM2 Background

### What is SAM2?
- **Developer**: Meta AI (Facebook AI Research)
- **Release**: 2024
- **Purpose**: Universal image and video segmentation
- **Key Feature**: Segments any object with minimal prompts (points, boxes, or masks)
- **Architecture**: Vision Transformer-based with prompt encoder

### Why Compare with SAM2?
- SAM2 represents cutting-edge deep learning segmentation
- ArUco method uses classical computer vision (markers + geometry)
- Comparison shows trade-offs between traditional and ML approaches
- Validates ArUco segmentation quality against ground truth

## Methodology

### 1. Segmentation Approaches

#### ArUco Method (Part 4)
- **Input**: Image with visible ArUco markers on object boundary
- **Process**: Detect markers → Calculate centers → Create boundary
- **Advantages**: 
  - No training data needed
  - Fast processing
  - Interpretable results
  - Works offline
- **Limitations**:
  - Requires physical marker placement
  - Limited to marker-visible objects
  - Geometric constraints (convex/concave)

#### SAM2 Method
- **Input**: Image + prompts (point coordinates from ArUco centers)
- **Process**: Encode image → Apply prompts → Generate segmentation
- **Advantages**:
  - No markers needed in deployment
  - Handles complex shapes naturally
  - Trained on vast datasets
  - State-of-the-art accuracy
- **Limitations**:
  - Requires large model download (~2GB)
  - GPU recommended for speed
  - Black-box model (less interpretable)

### 2. Comparison Metrics

#### Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}} = \frac{|A \cap B|}{|A \cup B|}$$

- Measures overlap between two masks
- Range: 0 (no overlap) to 1 (perfect match)
- **Good score**: IoU > 0.7

#### Dice Coefficient
$$\text{Dice} = \frac{2 \times \text{Intersection}}{\text{Total}} = \frac{2|A \cap B|}{|A| + |B|}$$

- Similar to IoU but more sensitive to size differences
- Range: 0 to 1
- **Good score**: Dice > 0.8

#### Precision & Recall
$$\text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}$$
$$\text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}$$

- Precision: How much of SAM2's prediction is correct?
- Recall: How much of ArUco's segmentation did SAM2 capture?

### 3. Visualization
Four-panel comparison for each image:
1. **Original Image**: Source image with object
2. **ArUco Segmentation**: Green overlay showing marker-based segmentation
3. **SAM2 Segmentation**: Red overlay showing AI-based segmentation
4. **Overlap Analysis**:
   - 🟢 Green: ArUco only (SAM2 missed)
   - 🔴 Red: SAM2 only (ArUco missed)
   - 🟡 Yellow: Both agree (overlap)

## Installation

### Prerequisites
```bash
# Python 3.8+
# Part 4 completed with ArUco segmentation results
```

### Install PyTorch
```bash
# For CUDA 11.8 (GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision
```

### Install SAM2
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Download SAM2 Checkpoint
Choose a model size and download checkpoint:

| Model | Size | Accuracy | Speed | Download Link |
|-------|------|----------|-------|---------------|
| SAM2 Tiny | ~150MB | Good | Fast | [sam2_hiera_t.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt) |
| SAM2 Small | ~180MB | Better | Medium | [sam2_hiera_s.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt) |
| **SAM2 Large** | **~220MB** | **Best** | **Slower** | **[sam2_hiera_l.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)** |

Save checkpoint to: `part5_sam2_comparison/checkpoints/`

## Usage

### Step 1: Verify ArUco Results
Ensure Part 4 is complete:
```bash
ls part4_aruco_segmentation/outputs/convex_hull/
# Should show *_mask.png and *_segmentation.jpg files
```

### Step 2: Run SAM2 Comparison
```python
cd part5_sam2_comparison

# Edit sam2_comparison.py to set checkpoint path:
checkpoint = "checkpoints/sam2_hiera_large.pt"

# Run comparison
python sam2_comparison.py
```

### Step 3: View Results
Results saved to `comparison_results/`:
- `*_comparison.jpg`: 4-panel visualization
- `*_sam2_mask.png`: SAM2 segmentation mask
- `comparison_summary.json`: Metrics for all images

## Implementation Details

### SAM2 Initialization
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
predictor = build_sam2(
    config="sam2_hiera_l.yaml",
    checkpoint="checkpoints/sam2_hiera_large.pt",
    device="cuda"  # or "cpu"
)
```

### Point-Based Prompting
```python
# Use ArUco marker centers as positive prompts
predictor.set_image(image_rgb)
masks, scores, logits = predictor.predict(
    point_coords=aruco_marker_centers,  # Nx2 array
    point_labels=np.ones(N),  # All foreground points
    multimask_output=False
)
```

### Metric Calculation
```python
# IoU
intersection = np.logical_and(mask_aruco, mask_sam2).sum()
union = np.logical_or(mask_aruco, mask_sam2).sum()
iou = intersection / union

# Dice
dice = 2 * intersection / (mask_aruco.sum() + mask_sam2.sum())
```

## Expected Results

### Typical Metric Ranges

| Scenario | IoU | Dice | Interpretation |
|----------|-----|------|----------------|
| Excellent agreement | 0.85-0.95 | 0.90-0.97 | Methods nearly identical |
| Good agreement | 0.70-0.85 | 0.82-0.90 | Minor differences |
| Moderate agreement | 0.50-0.70 | 0.67-0.82 | Noticeable differences |
| Poor agreement | <0.50 | <0.67 | Significant differences |

### Common Patterns

#### SAM2 More Accurate (ArUco Under-segments)
- **Cause**: Convex hull misses concave regions
- **Evidence**: Low recall, precision OK
- **Visual**: More green (ArUco-only) in overlap panel
- **Solution**: Use contour method in Part 4

#### ArUco More Accurate (SAM2 Over-segments)
- **Cause**: SAM2 includes background or shadows
- **Evidence**: Low precision, recall OK
- **Visual**: More red (SAM2-only) in overlap panel
- **Solution**: Fine-tune SAM2 prompts or post-process mask

#### High Agreement
- **Cause**: Simple object, good markers, clear boundaries
- **Evidence**: High IoU and Dice (>0.8)
- **Visual**: Mostly yellow (overlap) in comparison
- **Conclusion**: ArUco method validated!

## Analysis Questions

When analyzing results, consider:

1. **When does ArUco outperform SAM2?**
   - Simple, well-defined shapes
   - High-contrast boundaries
   - Even marker distribution

2. **When does SAM2 outperform ArUco?**
   - Complex, non-convex shapes
   - Fuzzy or gradual boundaries
   - Shadows and occlusions

3. **Trade-offs**:
   - **Setup**: ArUco needs physical markers vs SAM2 needs model download
   - **Speed**: ArUco faster (~ms) vs SAM2 slower (~seconds)
   - **Accuracy**: Depends on object complexity
   - **Generalization**: ArUco limited to marked objects vs SAM2 works on anything

## Troubleshooting

### SAM2 Installation Issues
```bash
# If git clone fails:
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# If CUDA not found:
# Install CPU version of PyTorch instead
```

### Out of Memory (GPU)
```python
# Use smaller model (Tiny instead of Large)
# Or process images at lower resolution:
image_resized = cv.resize(image, (800, 600))
```

### Low IoU Scores
- Check that ArUco and SAM2 are using same image
- Verify ArUco markers are properly detected
- Try different SAM2 prompting strategies
- Ensure masks are binary (0 or 255)

## File Structure
```
part5_sam2_comparison/
├── sam2_comparison.py          # Main comparison script
├── README.md                   # This file
├── checkpoints/               # SAM2 model weights
│   └── sam2_hiera_large.pt   # Download here
└── comparison_results/        # Output
    ├── image1_comparison.jpg
    ├── image1_sam2_mask.png
    └── comparison_summary.json
```

## Integration with CV Portfolio

The results from Parts 4 & 5 can be added to your CV Portfolio website with:
- Interactive image gallery
- Side-by-side comparisons
- Metric visualizations (IoU/Dice charts)
- Method explanations
- Live demo (if deploying SAM2)

See `web_integration/` folder for Flask routes and templates.

---

## References

1. **SAM2 Paper**: [Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
2. **SAM2 GitHub**: https://github.com/facebookresearch/segment-anything-2
3. **ArUco Markers**: [Detection of ArUco Markers (OpenCV Docs)](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
4. **IoU Metric**: [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index)
