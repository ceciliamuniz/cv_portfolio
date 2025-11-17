#!/usr/bin/env python3
"""
Systematic ArUco dictionary detection based on OpenCV documentation.
Tests all predefined dictionaries to find which one matches the user's markers.
"""

import cv2 as cv
import numpy as np

def test_all_dictionaries(image_path):
    """
    Test an image against all predefined ArUco dictionaries.
    Based on OpenCV documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    """
    
    # All predefined dictionaries as per OpenCV documentation
    dictionaries = [
        ("DICT_4X4_50", cv.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv.aruco.DICT_4X4_100), 
        ("DICT_4X4_250", cv.aruco.DICT_4X4_250),
        ("DICT_4X4_1000", cv.aruco.DICT_4X4_1000),
        ("DICT_5X5_50", cv.aruco.DICT_5X5_50),
        ("DICT_5X5_100", cv.aruco.DICT_5X5_100),
        ("DICT_5X5_250", cv.aruco.DICT_5X5_250), 
        ("DICT_5X5_1000", cv.aruco.DICT_5X5_1000),
        ("DICT_6X6_50", cv.aruco.DICT_6X6_50),
        ("DICT_6X6_100", cv.aruco.DICT_6X6_100),
        ("DICT_6X6_250", cv.aruco.DICT_6X6_250),
        ("DICT_6X6_1000", cv.aruco.DICT_6X6_1000),
        ("DICT_7X7_50", cv.aruco.DICT_7X7_50),
        ("DICT_7X7_100", cv.aruco.DICT_7X7_100),
        ("DICT_7X7_250", cv.aruco.DICT_7X7_250),
        ("DICT_7X7_1000", cv.aruco.DICT_7X7_1000),
        ("DICT_ARUCO_ORIGINAL", cv.aruco.DICT_ARUCO_ORIGINAL)
    ]
    
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    print(f"Testing {image_path} against all ArUco dictionaries...")
    print("=" * 80)
    
    best_results = []
    
    for dict_name, dict_type in dictionaries:
        try:
            # Create dictionary and detector
            aruco_dict = cv.aruco.getPredefinedDictionary(dict_type)
            parameters = cv.aruco.DetectorParameters()
            
            # Use robust detection parameters
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 100
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.minMarkerPerimeterRate = 0.005
            parameters.maxMarkerPerimeterRate = 8.0
            parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
            
            detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            marker_count = len(ids) if ids is not None else 0
            
            if marker_count > 0:
                marker_ids = ids.flatten().tolist()
                print(f"✓ {dict_name:18} -> {marker_count} markers: {marker_ids}")
                best_results.append((dict_name, dict_type, marker_count, marker_ids))
            else:
                print(f"✗ {dict_name:18} -> No markers detected")
                
        except Exception as e:
            print(f"✗ {dict_name:18} -> Error: {str(e)}")
    
    print("=" * 80)
    
    if best_results:
        print("SUMMARY - Successful Detections:")
        for dict_name, dict_type, count, ids in best_results:
            print(f"  {dict_name}: {count} markers {ids}")
        
        # Return the dictionary with most detections
        best_dict = max(best_results, key=lambda x: x[2])
        print(f"\\nRECOMMENDED: {best_dict[0]} (detected {best_dict[2]} markers)")
        return best_dict[0], best_dict[1]
    else:
        print("No markers detected in any dictionary.")
        return None, None

def main():
    # Test images that we know have markers
    test_images = [
        "images/IMG_3790.jpeg",  # Known to have 2 markers
        "images/IMG_3797.jpeg"   # Known to have 1 marker  
    ]
    
    for image_path in test_images:
        print(f"\\n{'='*80}")
        print(f"TESTING: {image_path}")
        print('='*80)
        
        best_dict_name, best_dict_type = test_all_dictionaries(image_path)
        
        if best_dict_name:
            print(f"\\n🎯 RESULT: Use {best_dict_name} for this image")
        else:
            print("\\n❌ RESULT: No compatible dictionary found")

if __name__ == "__main__":
    main()