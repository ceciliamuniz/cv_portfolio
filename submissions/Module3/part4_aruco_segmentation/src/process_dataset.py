import os
from glob import glob
import cv2
import numpy as np
from image_utils import (
    detect_aruco_markers,
    get_marker_centers,
    segment_object_from_markers,
    draw_markers,
    draw_segmentation_result,
    generate_aruco_marker,
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


def create_synthetic_test_image(index: int) -> np.ndarray:
    """
    Create a synthetic test image with a non-rectangular object and ArUco markers.
    This simulates what a real capture would look like.
    """
    # Create background
    bg = np.random.randint(40, 80, (480, 640, 3), dtype=np.uint8)
    
    # Create a non-rectangular object (irregular polygon, ellipse, or blob)
    obj_type = ['ellipse', 'polygon', 'irregular'][index % 3]
    
    # Object mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    
    if obj_type == 'ellipse':
        center = (320 + np.random.randint(-40, 40), 240 + np.random.randint(-30, 30))
        axes = (np.random.randint(120, 180), np.random.randint(80, 120))
        angle = np.random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
    elif obj_type == 'polygon':
        num_points = np.random.randint(6, 10)
        center = (320, 240)
        radius = np.random.randint(100, 150)
        angles = np.sort(np.random.uniform(0, 2*np.pi, num_points))
        points = []
        for angle in angles:
            r = radius + np.random.randint(-30, 30)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
    else:  # irregular
        # Create irregular blob
        center = (320, 240)
        for _ in range(8):
            cx = center[0] + np.random.randint(-50, 50)
            cy = center[1] + np.random.randint(-50, 50)
            radius = np.random.randint(60, 100)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    # Create object with texture
    obj_color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
    img = bg.copy()
    img[mask == 255] = obj_color
    
    # Add texture to object
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Find object contour to place markers on boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img
    
    contour = contours[0]
    perimeter_points = contour.reshape(-1, 2)
    
    # Place 4-8 ArUco markers along the boundary
    num_markers = np.random.randint(4, 9)
    marker_indices = np.linspace(0, len(perimeter_points)-1, num_markers, dtype=int)
    
    marker_size = 40
    for i, idx in enumerate(marker_indices):
        px, py = perimeter_points[idx]
        
        # Generate small ArUco marker
        marker_id = i
        marker_img = generate_aruco_marker(marker_id, size=marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Place marker at boundary position
        x1 = max(0, px - marker_size//2)
        y1 = max(0, py - marker_size//2)
        x2 = min(640, x1 + marker_size)
        y2 = min(480, y1 + marker_size)
        
        # Adjust if out of bounds
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            img[y1:y2, x1:x2] = cv2.resize(marker_bgr, (w, h))
    
    return img


def maybe_generate_synthetic(folder, count=10):
    """Generate synthetic test images with ArUco markers if dataset is empty."""
    os.makedirs(folder, exist_ok=True)
    images = list_images(folder)
    if len(images) >= count:
        return images
    
    print("No images found. Generating synthetic test images...")
    print("(Replace these with real images captured with ArUco markers)")
    
    for i in range(count - len(images)):
        img = create_synthetic_test_image(i)
        path = os.path.join(folder, f'synth_aruco_{i+1:02d}.png')
        cv2.imwrite(path, img)
        print(f"  Generated: synth_aruco_{i+1:02d}.png")
    
    return list_images(folder)


def process_all():
    """Process all images to detect ArUco markers and segment objects."""
    os.makedirs(OUT_DIR, exist_ok=True)
    images = maybe_generate_synthetic(DATA_DIR, count=10)
    
    if len(images) < 10:
        print(f"Warning: found only {len(images)} images.")

    print(f"\nProcessing {len(images)} images for ArUco-based segmentation...\n")

    for idx, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {idx+1}/{len(images)}: {name}")
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  Skipped (failed to read): {path}")
            continue

        # Step 1: Detect ArUco markers
        print("  - Detecting ArUco markers...")
        corners, ids, rejected = detect_aruco_markers(bgr)
        
        if len(corners) == 0:
            print("  ⚠ No ArUco markers detected! Skipping segmentation.")
            # Save visualization anyway
            no_markers_img = bgr.copy()
            cv2.putText(no_markers_img, "No markers detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            save_image(os.path.join(OUT_DIR, 'detected_markers', f'{name}_markers.png'), no_markers_img)
            continue
        
        # Visualize detected markers
        markers_viz = draw_markers(bgr, corners, ids)
        save_image(os.path.join(OUT_DIR, 'detected_markers', f'{name}_markers.png'), markers_viz)
        
        # Get marker center positions
        marker_centers = get_marker_centers(corners)
        print(f"  ✓ Detected {len(marker_centers)} ArUco markers")
        
        # Step 2: Segment object using marker positions
        print("  - Segmenting object from marker positions...")
        mask, boundary = segment_object_from_markers(bgr, marker_centers, expand_ratio=1.3)
        
        if len(boundary) == 0:
            print("  ⚠ Segmentation failed (no boundary found)")
            continue
        
        # Create segmentation visualization
        segmentation_viz = draw_segmentation_result(bgr, mask, boundary, marker_centers)
        save_image(os.path.join(OUT_DIR, 'segmentation', f'{name}_segmentation.png'), segmentation_viz)
        
        # Extract segmented object
        segmented_obj = cv2.bitwise_and(bgr, bgr, mask=mask)
        
        # Draw clean boundary outline
        boundary_viz = bgr.copy()
        cv2.drawContours(boundary_viz, [boundary], -1, (0, 255, 0), 3)
        
        # Add metrics
        area = cv2.contourArea(boundary)
        perimeter = cv2.arcLength(boundary, True)
        x, y, w, h = cv2.boundingRect(boundary)
        cv2.rectangle(boundary_viz, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.putText(boundary_viz, f"Area: {int(area)} px", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(boundary_viz, f"Perimeter: {int(perimeter)} px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(boundary_viz, f"Markers: {len(marker_centers)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_image(os.path.join(OUT_DIR, 'boundary', f'{name}_boundary.png'), boundary_viz)
        
        # Create comparison panel
        panel = panel_compare(
            (bgr, markers_viz, mask, segmentation_viz, boundary_viz),
            ("Original", "Detected Markers", "Segmentation Mask", "Result with Markers", "Boundary + Metrics")
        )
        save_image(os.path.join(OUT_DIR, 'comparison', f'{name}_comparison.png'), panel)
        
        print(f"  ✓ Segmented object: area={int(area)}, perimeter={int(perimeter)}")
        print(f"  ✓ Bounding box: {w}x{h} px\n")

    print("✓ Done. Outputs saved to:")
    print(f"  {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    process_all()
