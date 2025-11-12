# Part 1: Gradient and LoG Analysis

This part analyzes a dataset of 10 images to compute gradient magnitude, gradient angle, and Laplacian of Gaussian (LoG) filtered versions.

## Quick Start

1. **Add your dataset** (optional):
   - Place 10+ images of the same object (different angles/distances) in `data/images/`
   - Supported formats: jpg, jpeg, png, bmp, tif, tiff
   - If no images are provided, synthetic examples will be generated

2. **Process the dataset**:
   ```bash
   python src/process_dataset.py
   ```
   This creates:
   - `outputs/grad_mag/` - Gradient magnitude images
   - `outputs/grad_angle/` - Gradient angle images  
   - `outputs/log/` - Laplacian of Gaussian filtered images
   - `outputs/comparison/` - Side-by-side comparison panels

3. **Launch the web app**:
   ```bash
   python web/app.py
   ```
   Visit: http://localhost:5000

## Processing Details

- **Gradient Magnitude**: `sqrt(gx² + gy²)` using Sobel operators
- **Gradient Angle**: `arctan2(gy, gx)` mapped to [0,180) degrees
- **LoG**: Laplacian applied after Gaussian blur (σ=1.0, kernel=5x5)

All outputs are normalized to [0,255] for visualization.
