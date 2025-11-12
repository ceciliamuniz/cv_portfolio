import os
from glob import glob
import cv2
import numpy as np
from image_utils import (
    detect_edges_canny, 
    detect_corners_harris,
    draw_keypoints_on_image,
    save_image, 
    panel_compare
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def list_images(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    files.sort()
    return files


def maybe_generate_synthetic(folder, count=10):
    """Generate synthetic images if dataset is empty."""
    os.makedirs(folder, exist_ok=True)
    images = list_images(folder)
    if len(images) >= count:
        return images
    
    # Generate simple synthetic images with edges and corners
    for i in range(count - len(images)):
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        
        # Draw rectangles (has both edges and corners)
        num_rects = np.random.randint(2, 5)
        for _ in range(num_rects):
            pt1 = (np.random.randint(20, 400), np.random.randint(20, 300))
            pt2 = (pt1[0] + np.random.randint(40, 120), pt1[1] + np.random.randint(40, 100))
            color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
            thickness = np.random.choice([2, 3, -1])  # -1 fills the rectangle
            cv2.rectangle(img, pt1, pt2, color, thickness)
        
        # Draw circles (has edges but no sharp corners)
        num_circles = np.random.randint(1, 3)
        for _ in range(num_circles):
            center = (np.random.randint(60, 420), np.random.randint(60, 300))
            radius = np.random.randint(20, 60)
            color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
            cv2.circle(img, center, radius, color, 2)
        
        # Draw lines (edges)
        num_lines = np.random.randint(1, 3)
        for _ in range(num_lines):
            pt1 = (np.random.randint(20, 460), np.random.randint(20, 340))
            pt2 = (np.random.randint(20, 460), np.random.randint(20, 340))
            color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
            cv2.line(img, pt1, pt2, color, 2)
        
        # Add noise for realism
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        path = os.path.join(folder, f'synth_{i+1:02d}.png')
        cv2.imwrite(path, noisy)
    
    return list_images(folder)


def process_all():
    """Process all images in the dataset to detect edges and corners."""
    os.makedirs(OUT_DIR, exist_ok=True)
    images = maybe_generate_synthetic(DATA_DIR, count=10)
    
    if len(images) < 10:
        print(f"Warning: found only {len(images)} images; synthetic fillers added.")

    for idx, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {idx+1}/{len(images)}: {name}")
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  Skipped (failed to read): {path}")
            continue
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edge_keypoints = detect_edges_canny(gray, low_thresh=50, high_thresh=150)
        
        # Detect corners using Harris
        corner_response, corner_keypoints = detect_corners_harris(gray, threshold=0.01)
        
        # Create visualizations with keypoints overlaid on original image
        edges_viz = draw_keypoints_on_image(bgr, edge_keypoints, color=(0, 255, 0), marker_size=1)  # Green edges
        corners_viz = draw_keypoints_on_image(bgr, corner_keypoints, color=(0, 0, 255), marker_size=3)  # Red corners
        
        # Combined view: edges (green) + corners (red)
        combined_viz = draw_keypoints_on_image(bgr, edge_keypoints, color=(0, 255, 0), marker_size=1)
        combined_viz = draw_keypoints_on_image(combined_viz, corner_keypoints, color=(0, 0, 255), marker_size=3)
        
        # Save outputs
        save_image(os.path.join(OUT_DIR, 'edges', f'{name}_edges.png'), edges_viz)
        save_image(os.path.join(OUT_DIR, 'corners', f'{name}_corners.png'), corners_viz)
        save_image(os.path.join(OUT_DIR, 'combined', f'{name}_combined.png'), combined_viz)
        
        # Create comparison panel
        panel = panel_compare(
            (bgr, edges_viz, corners_viz, combined_viz),
            ("Original", "Edge Keypoints (Green)", "Corner Keypoints (Red)", "Combined (Edges+Corners)")
        )
        save_image(os.path.join(OUT_DIR, 'comparison', f'{name}_comparison.png'), panel)
        
        print(f"  Detected {np.sum(edge_keypoints == 255)} edge keypoints")
        print(f"  Detected {np.sum(corner_keypoints == 255)} corner keypoints")

    print("Done. Outputs saved to:")
    print(f"  {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    process_all()
