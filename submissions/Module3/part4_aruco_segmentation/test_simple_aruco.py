#!/usr/bin/env python3
"""
Test the simplified ArUco segmentation approach.
"""

import cv2 as cv
import os
from aruco_segmentation import ArucoSegmentation

def main():
    # Initialize detector
    detector = ArucoSegmentation()
    
    # Test images
    image_dir = "images"
    output_dir = "outputs/simple_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    print("Testing simplified ArUco segmentation...")
    print("=" * 60)
    
    successful_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv.imread(img_path)
        
        if image is not None:
            print(f"\nProcessing: {img_file}")
            
            # Use the simplified approach
            segmentation_viz, status = detector.aruco_segment_object_simple(image)
            
            print(f"  Status: {status}")
            
            if "Success" in status:
                successful_count += 1
                
                # Save result
                output_path = os.path.join(output_dir, f"{img_file}_segmented.png")
                cv.imwrite(output_path, segmentation_viz)
                print(f"  ✓ Saved: {output_path}")
            else:
                print(f"  ✗ Failed")
        else:
            print(f"✗ Could not load: {img_file}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {successful_count}/{len(image_files)} images successfully processed")
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()