import os
from glob import glob
import cv2
import numpy as np
from image_utils import (
    find_boundaries_contour,
    find_boundaries_grabcut,
    draw_contours_with_info,
    extract_largest_object,
    find_precise_boundaries,
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
    """Generate synthetic images with clear objects for boundary detection."""
    os.makedirs(folder, exist_ok=True)
    images = list_images(folder)
    if len(images) >= count:
        return images
    
    # Generate images with distinct objects on contrasting backgrounds
    for i in range(count - len(images)):
        # Create background
        bg_color = np.random.randint(20, 80)
        img = np.ones((400, 600, 3), dtype=np.uint8) * bg_color
        
        # Choose object type
        obj_type = np.random.choice(['rectangle', 'circle', 'polygon', 'ellipse'])
        obj_color = (np.random.randint(180, 255), np.random.randint(180, 255), np.random.randint(180, 255))
        
        if obj_type == 'rectangle':
            w, h = np.random.randint(100, 200), np.random.randint(80, 150)
            x = (600 - w) // 2 + np.random.randint(-50, 50)
            y = (400 - h) // 2 + np.random.randint(-30, 30)
            cv2.rectangle(img, (x, y), (x+w, y+h), obj_color, -1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
        elif obj_type == 'circle':
            radius = np.random.randint(60, 100)
            cx = 300 + np.random.randint(-50, 50)
            cy = 200 + np.random.randint(-30, 30)
            cv2.circle(img, (cx, cy), radius, obj_color, -1)
            cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
            
        elif obj_type == 'ellipse':
            cx = 300 + np.random.randint(-50, 50)
            cy = 200 + np.random.randint(-30, 30)
            axes = (np.random.randint(80, 120), np.random.randint(50, 80))
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, obj_color, -1)
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (255, 255, 255), 2)
            
        else:  # polygon
            num_points = np.random.randint(5, 8)
            cx, cy = 300, 200
            radius = np.random.randint(70, 100)
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            points = []
            for angle in angles:
                r = radius + np.random.randint(-20, 20)
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                points.append([x, y])
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [points], obj_color)
            cv2.polylines(img, [points], True, (255, 255, 255), 2)
        
        # Add slight noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        path = os.path.join(folder, f'synth_{i+1:02d}.png')
        cv2.imwrite(path, noisy)
    
    return list_images(folder)


def process_all():
    """Process all images to find object boundaries using multiple methods."""
    os.makedirs(OUT_DIR, exist_ok=True)
    images = maybe_generate_synthetic(DATA_DIR, count=10)
    
    if len(images) < 10:
        print(f"Warning: found only {len(images)} images; synthetic fillers added.")

    for idx, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nProcessing {idx+1}/{len(images)}: {name}")
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  Skipped (failed to read): {path}")
            continue

        # Method 1: Contour-based boundary detection (Otsu thresholding)
        print("  - Finding boundaries with contour detection...")
        binary_otsu, contours_otsu, _ = find_boundaries_contour(bgr, method='otsu')
        contour_viz = draw_contours_with_info(bgr, contours_otsu, min_area=500)
        
        # Extract largest object
        largest_obj, largest_mask = extract_largest_object(bgr, contours_otsu)
        
        # Method 2: GrabCut segmentation
        print("  - Finding boundaries with GrabCut...")
        try:
            grabcut_mask, grabcut_fg = find_boundaries_grabcut(bgr, margin_percent=0.15)
            # Apply mask to original image
            grabcut_result = cv2.bitwise_and(bgr, bgr, mask=grabcut_fg)
            # Find contours from GrabCut mask
            gc_contours, _ = cv2.findContours(grabcut_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            grabcut_viz = draw_contours_with_info(bgr, gc_contours, min_area=500)
        except Exception as e:
            print(f"    GrabCut failed: {e}")
            grabcut_result = bgr.copy()
            grabcut_viz = bgr.copy()
            grabcut_fg = np.zeros(bgr.shape[:2], dtype=np.uint8)
        
        # Get precise boundary information
        precise_contours, boundary_info = find_precise_boundaries(largest_mask)
        
        # Create detailed boundary visualization
        boundary_detail = bgr.copy()
        if len(precise_contours) > 0:
            cnt = precise_contours[0]
            # Draw exact boundary (green)
            cv2.drawContours(boundary_detail, [cnt], -1, (0, 255, 0), 3)
            
            # Draw bounding shapes
            x, y, w, h = boundary_info['bounding_box']
            cv2.rectangle(boundary_detail, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(boundary_detail, f"Box: {w}x{h}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            center, radius = boundary_info['min_enclosing_circle']
            cv2.circle(boundary_detail, center, radius, (255, 255, 0), 2)
            
            if boundary_info['ellipse'] is not None:
                cv2.ellipse(boundary_detail, boundary_info['ellipse'], (255, 0, 255), 2)
            
            # Add metrics text
            y_offset = 30
            cv2.putText(boundary_detail, f"Area: {int(boundary_info['area'])} px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(boundary_detail, f"Perimeter: {int(boundary_info['perimeter'])} px", (10, y_offset+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(boundary_detail, f"Centroid: {boundary_info['centroid']}", (10, y_offset+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save outputs
        save_image(os.path.join(OUT_DIR, 'contours', f'{name}_contours.png'), contour_viz)
        save_image(os.path.join(OUT_DIR, 'grabcut', f'{name}_grabcut.png'), grabcut_viz)
        save_image(os.path.join(OUT_DIR, 'boundaries', f'{name}_boundaries.png'), boundary_detail)
        
        # Create comparison panel
        panel = panel_compare(
            (bgr, binary_otsu, contour_viz, grabcut_viz, boundary_detail),
            ("Original", "Binary Mask (Otsu)", "Contour Detection", "GrabCut Segmentation", "Precise Boundaries")
        )
        save_image(os.path.join(OUT_DIR, 'comparison', f'{name}_comparison.png'), panel)
        
        # Print statistics
        print(f"  ✓ Found {len(contours_otsu)} contours (Otsu method)")
        if boundary_info:
            print(f"  ✓ Largest object: area={int(boundary_info['area'])}, perimeter={int(boundary_info['perimeter'])}")
            print(f"  ✓ Bounding box: {boundary_info['bounding_box'][2]}x{boundary_info['bounding_box'][3]} px")

    print("\n✓ Done. Outputs saved to:")
    print(f"  {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    process_all()
