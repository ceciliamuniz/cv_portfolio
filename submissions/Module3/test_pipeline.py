"""
Quick test script to verify the image processing pipeline works correctly.
"""

import cv2
import numpy as np
from pathlib import Path

# Import from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from process_all import (
    compute_gradient_magnitude_angle,
    compute_laplacian_of_gaussian,
    detect_edge_keypoints,
    detect_corner_keypoints,
    detect_object_boundaries
)

def test_pipeline():
    """Test the processing pipeline with a simple synthetic image."""
    print("Testing image processing pipeline...")
    
    # Create a simple test image with shapes
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw rectangle
    cv2.rectangle(img, (100, 100), (300, 250), (255, 255, 255), -1)
    
    # Draw circle
    cv2.circle(img, (450, 200), 80, (255, 255, 255), -1)
    
    # Draw triangle
    pts = np.array([[350, 80], [450, 280], [250, 280]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("\n1. Testing gradient computation...")
    try:
        grad_mag, grad_angle = compute_gradient_magnitude_angle(gray)
        print(f"   ✓ Gradient magnitude shape: {grad_mag.shape}")
        print(f"   ✓ Gradient angle shape: {grad_angle.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n2. Testing LoG computation...")
    try:
        log_img = compute_laplacian_of_gaussian(gray)
        print(f"   ✓ LoG image shape: {log_img.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n3. Testing edge detection...")
    try:
        edges, edge_kpts = detect_edge_keypoints(gray)
        print(f"   ✓ Found {len(edge_kpts)} edge keypoints")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n4. Testing corner detection...")
    try:
        corner_resp, corner_kpts = detect_corner_keypoints(gray)
        print(f"   ✓ Found {len(corner_kpts)} corner keypoints")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n5. Testing boundary detection...")
    try:
        boundary_img, contours = detect_object_boundaries(gray, img)
        print(f"   ✓ Found {len(contours)} contours")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    print("\nYou can now run: python process_all.py")
    return True

if __name__ == '__main__':
    test_pipeline()
