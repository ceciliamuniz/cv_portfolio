"""
Module 3: Complete Image Analysis Pipeline
Processes 10 uploaded images to compute:
1. Gradient magnitude and angle
2. Laplacian of Gaussian (LoG)
3. Edge detection keypoints
4. Corner detection keypoints
5. Object boundary detection

Author: Computer Vision Student
Course: Computer Vision Module 3
"""

import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / 'images'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create output directories
(OUTPUT_DIR / 'gradients' / 'magnitude').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'gradients' / 'angle').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'log').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'comparison').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'edges').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'corners').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'combined').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'boundaries').mkdir(parents=True, exist_ok=True)


def compute_gradient_magnitude_angle(gray):
    """
    Compute gradient magnitude and angle using Sobel operators.
    
    Returns:
        mag: Gradient magnitude (normalized to 0-255)
        angle: Gradient angle in degrees [0, 180) (normalized to 0-255 for visualization)
    """
    # Sobel gradients (float64 for precision)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude: sqrt(gx² + gy²)
    mag = np.sqrt(gx**2 + gy**2)
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Angle: arctan2(gy, gx) in degrees, mapped to [0, 180)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0
    angle_normalized = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mag_normalized, angle_normalized


def compute_laplacian_of_gaussian(gray, ksize=5, sigma=1.0):
    """
    Compute Laplacian of Gaussian (LoG) for edge detection response.
    
    Process:
    1. Apply Gaussian blur to reduce noise
    2. Apply Laplacian operator to detect edges
    
    Returns:
        log_viz: LoG response (normalized to 0-255)
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    
    # Laplacian
    log_response = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    
    # Normalize for visualization
    log_abs = np.abs(log_response)
    log_viz = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return log_viz


def detect_edge_keypoints(gray, low_thresh=50, high_thresh=150):
    """
    Detect EDGE keypoints using Canny edge detection algorithm.
    
    Algorithm steps:
    1. Gaussian blur to reduce noise
    2. Compute gradients using Sobel
    3. Non-maximum suppression to thin edges
    4. Double thresholding (strong/weak edges)
    5. Edge tracking by hysteresis
    
    Returns:
        edges: Binary edge map
        edge_keypoints: List of (x, y) edge keypoint coordinates
    """
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_thresh, high_thresh, L2gradient=True)
    
    # Extract edge keypoints (all non-zero pixels in edge map)
    edge_keypoints = np.argwhere(edges > 0)
    # Convert from (row, col) to (x, y)
    edge_keypoints = [(int(pt[1]), int(pt[0])) for pt in edge_keypoints]
    
    return edges, edge_keypoints


def detect_corner_keypoints(gray, block_size=2, ksize=3, k=0.04, threshold_ratio=0.01):
    """
    Detect CORNER keypoints using Harris corner detection algorithm.
    
    Algorithm steps:
    1. Compute image gradients Ix and Iy
    2. Build structure tensor M from Ix², Iy², and Ix·Iy
    3. Calculate corner response: R = det(M) - k·trace(M)²
    4. Threshold to keep strong corners
    5. Non-maximum suppression to get local maxima
    
    Returns:
        corner_response: Harris corner response map
        corner_keypoints: List of (x, y) corner keypoint coordinates
    """
    # Harris corner detection
    corner_response = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilate to mark corners
    corner_response = cv2.dilate(corner_response, None)
    
    # Threshold: keep only strong corners (top 1% of max response)
    threshold = threshold_ratio * corner_response.max()
    corner_mask = corner_response > threshold
    
    # Extract corner keypoints
    corner_keypoints = np.argwhere(corner_mask)
    # Convert from (row, col) to (x, y)
    corner_keypoints = [(int(pt[1]), int(pt[0])) for pt in corner_keypoints]
    
    return corner_response, corner_keypoints


def detect_object_boundaries(gray, original):
    """
    Detect exact boundaries of objects using multiple techniques:
    1. Canny edge detection
    2. Morphological operations
    3. Contour detection
    4. Convex hull fitting
    
    Returns:
        boundary_img: Image with boundaries drawn
        contours: List of detected contours
    """
    # Preprocessing: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Method 1: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Method 2: Morphological closing to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Method 3: Find contours
    contours, hierarchy = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (remove very small noise)
    min_area = 100
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create output image
    boundary_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original.copy()
    
    # Draw contours and bounding information
    for i, contour in enumerate(significant_contours):
        # Draw contour boundary (green)
        cv2.drawContours(boundary_img, [contour], 0, (0, 255, 0), 2)
        
        # Draw convex hull (blue)
        hull = cv2.convexHull(contour)
        cv2.drawContours(boundary_img, [hull], 0, (255, 0, 0), 1)
        
        # Draw bounding rectangle (red)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(boundary_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        # Draw minimum area rectangle (yellow)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Use np.intp instead of deprecated np.int0
        cv2.drawContours(boundary_img, [box], 0, (0, 255, 255), 1)
    
    return boundary_img, significant_contours


def create_comparison_panel(images, titles):
    """Create side-by-side comparison panel with titles."""
    # Ensure all images are same height
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        scale = target_h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))
    
    # Convert grayscale to BGR for consistency
    colored = []
    for img in resized:
        if len(img.shape) == 2:
            colored.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        else:
            colored.append(img)
    
    # Add title bars
    pad_h = 40
    panels = []
    for img, title in zip(colored, titles):
        pad = np.ones((pad_h, img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(pad, title, (10, pad_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([pad, img]))
    
    # Concatenate horizontally
    return np.hstack(panels)


def visualize_keypoints(original, edge_keypoints, corner_keypoints):
    """
    Create visualizations with keypoints marked.
    
    Returns:
        edges_viz: Image with edge keypoints (green)
        corners_viz: Image with corner keypoints (red)
        combined_viz: Image with both edge and corner keypoints
    """
    # Create BGR copies
    edges_viz = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original.copy()
    corners_viz = edges_viz.copy()
    combined_viz = edges_viz.copy()
    
    # Draw edge keypoints (green circles) - sample for visibility
    sample_edges = edge_keypoints[::20]  # Sample every 20th point for visibility
    for pt in sample_edges:
        cv2.circle(edges_viz, pt, 1, (0, 255, 0), -1)
        cv2.circle(combined_viz, pt, 1, (0, 255, 0), -1)
    
    # Draw corner keypoints (red circles)
    for pt in corner_keypoints:
        cv2.circle(corners_viz, pt, 3, (0, 0, 255), -1)
        cv2.circle(combined_viz, pt, 3, (0, 0, 255), -1)
    
    return edges_viz, corners_viz, combined_viz


def process_image(image_path):
    """Process a single image through the complete pipeline."""
    print(f"\nProcessing: {image_path.name}")
    
    # Read image
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"  ERROR: Could not read image")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    name = image_path.stem
    
    # 1. Compute gradient magnitude and angle
    print("  Computing gradients...")
    grad_mag, grad_angle = compute_gradient_magnitude_angle(gray)
    cv2.imwrite(str(OUTPUT_DIR / 'gradients' / 'magnitude' / f'{name}_grad_mag.png'), grad_mag)
    cv2.imwrite(str(OUTPUT_DIR / 'gradients' / 'angle' / f'{name}_grad_angle.png'), grad_angle)
    
    # 2. Compute Laplacian of Gaussian
    print("  Computing LoG...")
    log_img = compute_laplacian_of_gaussian(gray)
    cv2.imwrite(str(OUTPUT_DIR / 'log' / f'{name}_log.png'), log_img)
    
    # 3. Detect edge keypoints
    print("  Detecting edge keypoints...")
    edges, edge_keypoints = detect_edge_keypoints(gray)
    print(f"    Found {len(edge_keypoints)} edge keypoints")
    
    # 4. Detect corner keypoints
    print("  Detecting corner keypoints...")
    corner_response, corner_keypoints = detect_corner_keypoints(gray)
    print(f"    Found {len(corner_keypoints)} corner keypoints")
    
    # 5. Visualize keypoints
    print("  Creating keypoint visualizations...")
    edges_viz, corners_viz, combined_viz = visualize_keypoints(gray, edge_keypoints, corner_keypoints)
    cv2.imwrite(str(OUTPUT_DIR / 'edges' / f'{name}_edges.png'), edges_viz)
    cv2.imwrite(str(OUTPUT_DIR / 'corners' / f'{name}_corners.png'), corners_viz)
    cv2.imwrite(str(OUTPUT_DIR / 'combined' / f'{name}_combined.png'), combined_viz)
    
    # 6. Detect object boundaries
    print("  Detecting object boundaries...")
    boundary_img, contours = detect_object_boundaries(gray, original)
    cv2.imwrite(str(OUTPUT_DIR / 'boundaries' / f'{name}_boundaries.png'), boundary_img)
    print(f"    Found {len(contours)} significant contours")
    
    # 7. Create comparison panel
    print("  Creating comparison panel...")
    panel = create_comparison_panel(
        [gray, grad_mag, grad_angle, log_img],
        ["Original", "Gradient Mag", "Gradient Angle", "LoG"]
    )
    cv2.imwrite(str(OUTPUT_DIR / 'comparison' / f'{name}_comparison.png'), panel)
    
    print("  ✓ Complete")


def main():
    """Main processing function."""
    print("=" * 60)
    print("Module 3: Image Analysis Pipeline")
    print("=" * 60)
    
    # Find all images
    image_files = list(IMAGES_DIR.glob('*.jpeg')) + list(IMAGES_DIR.glob('*.jpg')) + \
                  list(IMAGES_DIR.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {IMAGES_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each image
    for img_path in sorted(image_files):
        try:
            process_image(img_path)
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nOutputs saved to:")
    print(f"  Gradient Magnitude:  {OUTPUT_DIR / 'gradients' / 'magnitude'}")
    print(f"  Gradient Angle:      {OUTPUT_DIR / 'gradients' / 'angle'}")
    print(f"  Laplacian of Gaussian: {OUTPUT_DIR / 'log'}")
    print(f"  Edge Keypoints:      {OUTPUT_DIR / 'edges'}")
    print(f"  Corner Keypoints:    {OUTPUT_DIR / 'corners'}")
    print(f"  Combined Keypoints:  {OUTPUT_DIR / 'combined'}")
    print(f"  Object Boundaries:   {OUTPUT_DIR / 'boundaries'}")
    print(f"  Comparison Panels:   {OUTPUT_DIR / 'comparison'}")


if __name__ == '__main__':
    main()
