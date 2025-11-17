#!/usr/bin/env python3
"""
Robust ArUco detection based on official OpenCV documentation.
Systematically tests dictionaries and uses proper DetectorParameters.
"""

import cv2 as cv
import numpy as np
import os

def create_robust_detector_params():
    """
    Create DetectorParameters with robust settings based on OpenCV docs.
    """
    params = cv.aruco.DetectorParameters()
    
    # Adaptive thresholding parameters
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53  # Should be odd
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    
    # Contour filtering parameters
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 3
    
    # Corner refinement (improves accuracy)
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1
    
    # Error correction
    params.errorCorrectionRate = 0.6
    
    return params

def test_dictionaries_systematically(image_path):
    """
    Test image against common ArUco dictionaries in order of likelihood.
    """
    # Order by most common usage
    dictionaries_to_test = [
        ("DICT_6X6_250", cv.aruco.DICT_6X6_250),   # Very common
        ("DICT_4X4_50", cv.aruco.DICT_4X4_50),     # Simple and common
        ("DICT_5X5_250", cv.aruco.DICT_5X5_250),   # Good balance
        ("DICT_6X6_100", cv.aruco.DICT_6X6_100),   # Medium size
        ("DICT_4X4_100", cv.aruco.DICT_4X4_100),   # Larger 4x4 set
        ("DICT_4X4_250", cv.aruco.DICT_4X4_250),   # Large 4x4 set
        ("DICT_6X6_50", cv.aruco.DICT_6X6_50),     # Smaller 6x6 set
    ]
    
    # Load and prepare image
    image = cv.imread(image_path)
    if image is None:
        return None, None, None
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detector_params = create_robust_detector_params()
    
    print(f"Testing {os.path.basename(image_path)}:")
    
    best_result = None
    best_count = 0
    
    for dict_name, dict_type in dictionaries_to_test:
        try:
            # Create dictionary and detector
            dictionary = cv.aruco.getPredefinedDictionary(dict_type)
            detector = cv.aruco.ArucoDetector(dictionary, detector_params)
            
            # Detect markers
            marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)
            
            marker_count = len(marker_ids) if marker_ids is not None else 0
            
            if marker_count > 0:
                ids_list = marker_ids.flatten().tolist()
                print(f"  ✓ {dict_name}: {marker_count} markers {ids_list}")
                
                if marker_count > best_count:
                    best_count = marker_count
                    best_result = (dict_name, dict_type, marker_corners, marker_ids, rejected_candidates)
            else:
                print(f"  ✗ {dict_name}: No markers")
                
        except Exception as e:
            print(f"  ✗ {dict_name}: Error - {e}")
    
    if best_result:
        dict_name, dict_type, corners, ids, rejected = best_result
        print(f"  🎯 BEST: {dict_name} with {best_count} markers")
        return dict_name, dict_type, (corners, ids, rejected)
    else:
        print(f"  ❌ No markers found in any dictionary")
        return None, None, None

def segment_with_official_approach(image_path):
    """
    Segment object using official OpenCV ArUco approach.
    """
    # Test dictionaries to find the best one
    best_dict_name, best_dict_type, detection_result = test_dictionaries_systematically(image_path)
    
    if best_dict_name is None:
        return None, "No ArUco markers detected in any dictionary"
    
    # Load image and extract detection results
    image = cv.imread(image_path)
    marker_corners, marker_ids, rejected_candidates = detection_result
    
    print(f"\\nUsing {best_dict_name} for segmentation...")
    
    # Check minimum markers for segmentation
    if len(marker_ids) < 2:
        return None, f"Need at least 2 markers for segmentation (found {len(marker_ids)})"
    
    # Extract all corner points for convex hull
    all_points = []
    for corners in marker_corners:
        # corners is shape (1, 4, 2) - 4 corners with (x,y) coordinates
        # Add all 4 corners of each marker
        for corner in corners[0]:
            all_points.append([int(corner[0]), int(corner[1])])
    
    all_points = np.array(all_points)
    
    # Create convex hull around all marker corners
    hull_points = cv.convexHull(all_points)
    
    # Create visualization
    output_image = image.copy()
    
    # Draw detected markers (official OpenCV function)
    cv.aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids)
    
    # Draw convex hull
    cv.polylines(output_image, [hull_points], True, (0, 255, 255), 3)
    
    # GrabCut segmentation within hull region
    try:
        # Get bounding rectangle of hull
        x, y, w, h = cv.boundingRect(hull_points)
        
        # Ensure bounds are within image
        x = max(0, x)
        y = max(0, y) 
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)
        
        # Initialize mask for GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        rect = (x, y, w, h)
        
        # Background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Run GrabCut
        cv.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)
        
        # Create final mask
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to create segmentation
        segmented = image * final_mask[:, :, np.newaxis]
        
        # Combine with visualization
        alpha = 0.7
        output_image = cv.addWeighted(output_image, alpha, segmented, 1-alpha, 0)
        
        status = f"Success: {best_dict_name}, {len(marker_ids)} markers, GrabCut segmentation"
        
    except Exception as e:
        status = f"Markers detected ({best_dict_name}) but GrabCut failed: {e}"
    
    return output_image, status

def main():
    # Test on images we know have markers
    test_images = ["images/IMG_3790.jpeg", "images/IMG_3797.jpeg"]
    
    output_dir = "outputs/official_approach"
    os.makedirs(output_dir, exist_ok=True)
    
    print("ArUco Segmentation - Official OpenCV Approach")
    print("=" * 60)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\\nProcessing {image_path}:")
            print("-" * 40)
            
            result_image, status = segment_with_official_approach(image_path)
            
            if result_image is not None:
                # Save result
                output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_result.png")
                cv.imwrite(output_path, result_image)
                print(f"✓ {status}")
                print(f"  Saved: {output_path}")
            else:
                print(f"✗ {status}")
        else:
            print(f"✗ File not found: {image_path}")

if __name__ == "__main__":
    main()