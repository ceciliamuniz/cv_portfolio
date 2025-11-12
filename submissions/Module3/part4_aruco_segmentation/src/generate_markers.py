"""
Generate printable ArUco markers.

Run this script to create marker images that you can print and attach to your object.
"""
import os
import cv2
import numpy as np
from image_utils import generate_aruco_marker, save_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'markers')


def generate_marker_sheet(num_markers: int = 10, marker_size: int = 150):
    """
    Generate individual ArUco markers for printing.
    
    Args:
        num_markers: Number of unique markers to generate
        marker_size: Size of each marker in pixels
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {num_markers} ArUco markers...")
    
    for i in range(num_markers):
        marker_img = generate_aruco_marker(i, size=marker_size)
        
        # Add white border for easier cutting
        border = 20
        bordered = np.ones((marker_size + 2*border, marker_size + 2*border), dtype=np.uint8) * 255
        bordered[border:border+marker_size, border:border+marker_size] = marker_img
        
        # Add ID label
        cv2.putText(bordered, f"ID: {i}", (border, border-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        
        path = os.path.join(OUTPUT_DIR, f'marker_{i:02d}.png')
        cv2.imwrite(path, bordered)
        print(f"  Saved: marker_{i:02d}.png")
    
    print(f"\n✓ Markers saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nUsage Instructions:")
    print("1. Print the marker images (one marker per page for best results)")
    print("2. Cut out the markers carefully")
    print("3. Attach markers to the boundary of your non-rectangular object")
    print("4. Use at least 4-6 markers distributed around the object perimeter")
    print("5. Capture 10+ images from different angles and distances")
    print("6. Save images to part4_aruco_segmentation/data/images/")


if __name__ == '__main__':
    generate_marker_sheet(num_markers=20, marker_size=150)
