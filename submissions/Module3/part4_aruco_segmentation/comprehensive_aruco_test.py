#!/usr/bin/env python3
"""
Comprehensive ArUco detection test based on complete OpenCV documentation.
Tests all dictionaries with timing, shows rejected candidates, and debug info.
"""

import cv2 as cv
import numpy as np
import time
import os

def get_all_predefined_dictionaries():
    """
    Return all predefined dictionaries with their parameter values from OpenCV docs.
    """
    return [
        (0, "DICT_4X4_50", cv.aruco.DICT_4X4_50),
        (1, "DICT_4X4_100", cv.aruco.DICT_4X4_100),
        (2, "DICT_4X4_250", cv.aruco.DICT_4X4_250),
        (3, "DICT_4X4_1000", cv.aruco.DICT_4X4_1000),
        (4, "DICT_5X5_50", cv.aruco.DICT_5X5_50),
        (5, "DICT_5X5_100", cv.aruco.DICT_5X5_100),
        (6, "DICT_5X5_250", cv.aruco.DICT_5X5_250),
        (7, "DICT_5X5_1000", cv.aruco.DICT_5X5_1000),
        (8, "DICT_6X6_50", cv.aruco.DICT_6X6_50),
        (9, "DICT_6X6_100", cv.aruco.DICT_6X6_100),
        (10, "DICT_6X6_250", cv.aruco.DICT_6X6_250),
        (11, "DICT_6X6_1000", cv.aruco.DICT_6X6_1000),
        (12, "DICT_7X7_50", cv.aruco.DICT_7X7_50),
        (13, "DICT_7X7_100", cv.aruco.DICT_7X7_100),
        (14, "DICT_7X7_250", cv.aruco.DICT_7X7_250),
        (15, "DICT_7X7_1000", cv.aruco.DICT_7X7_1000),
        (16, "DICT_ARUCO_ORIGINAL", cv.aruco.DICT_ARUCO_ORIGINAL),
        (17, "DICT_APRILTAG_16h5", cv.aruco.DICT_APRILTAG_16h5),
        (18, "DICT_APRILTAG_25h9", cv.aruco.DICT_APRILTAG_25h9),
        (19, "DICT_APRILTAG_36h10", cv.aruco.DICT_APRILTAG_36h10),
        (20, "DICT_APRILTAG_36h11", cv.aruco.DICT_APRILTAG_36h11),
    ]

def create_optimized_detector_params():
    """
    Create DetectorParameters optimized for various conditions.
    """
    params = cv.aruco.DetectorParameters()
    
    # Adaptive thresholding (key for various lighting)
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    
    # Contour filtering
    params.minMarkerPerimeterRate = 0.005  # Very permissive for small/distant markers
    params.maxMarkerPerimeterRate = 8.0     # Very permissive for large/close markers
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 1
    
    # Corner refinement for accuracy
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1
    
    # Marker validation
    params.minOtsuStdDev = 2.0
    params.errorCorrectionRate = 0.6
    
    return params

def comprehensive_aruco_test(image_path):
    """
    Comprehensive test following OpenCV documentation exactly.
    """
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"\\nTesting: {os.path.basename(image_path)}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print("=" * 80)
    
    # Get detector parameters
    detector_params = create_optimized_detector_params()
    
    all_results = []
    total_time = 0
    total_iterations = 0
    
    # Test each dictionary
    for param_val, dict_name, dict_type in get_all_predefined_dictionaries():
        try:
            # Create dictionary and detector (as per documentation)
            dictionary = cv.aruco.getPredefinedDictionary(dict_type)
            detector = cv.aruco.ArucoDetector(dictionary, detector_params)
            
            # Time the detection (as shown in documentation)
            tick = cv.getTickCount()
            
            # Detect markers (exact function signature from docs)
            marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(image)
            
            # Calculate timing
            current_time = (cv.getTickCount() - tick) / cv.getTickFrequency()
            total_time += current_time
            total_iterations += 1
            
            # Count results
            num_markers = len(marker_ids) if marker_ids is not None else 0
            num_rejected = len(rejected_candidates) if rejected_candidates is not None else 0
            
            if num_markers > 0:
                ids_list = marker_ids.flatten().tolist()
                status = "✓"
                result_info = f"IDs: {ids_list}"
            else:
                status = "✗"
                result_info = "No markers"
            
            print(f"{status} {param_val:2d} {dict_name:20} -> {num_markers} markers, {num_rejected} rejected, {current_time*1000:.1f}ms | {result_info}")
            
            if num_markers > 0:
                all_results.append({
                    'param': param_val,
                    'name': dict_name,
                    'type': dict_type,
                    'markers': num_markers,
                    'ids': ids_list,
                    'corners': marker_corners,
                    'rejected': rejected_candidates,
                    'time_ms': current_time * 1000
                })
                
        except Exception as e:
            print(f"✗ {param_val:2d} {dict_name:20} -> ERROR: {str(e)}")
    
    # Summary
    print("=" * 80)
    if all_results:
        print(f"SUCCESSFUL DETECTIONS: {len(all_results)}")
        print(f"Average detection time: {(total_time/total_iterations)*1000:.1f} ms")
        
        # Sort by number of markers detected
        best_results = sorted(all_results, key=lambda x: x['markers'], reverse=True)
        
        print(f"\\nTOP RESULTS:")
        for i, result in enumerate(best_results[:3]):
            print(f"  {i+1}. {result['name']} (param={result['param']}): {result['markers']} markers {result['ids']} - {result['time_ms']:.1f}ms")
        
        return best_results[0]  # Return best result
    else:
        print("NO MARKERS DETECTED in any dictionary")
        print(f"Average processing time: {(total_time/total_iterations)*1000:.1f} ms")
        return None

def create_visualization_with_rejected(image_path, detection_result):
    """
    Create visualization showing detected markers and rejected candidates.
    """
    if detection_result is None:
        return None
    
    image = cv.imread(image_path)
    output_image = image.copy()
    
    # Draw detected markers (official OpenCV function)
    if len(detection_result['corners']) > 0:
        # Convert IDs back to the format expected by drawDetectedMarkers
        ids_array = np.array(detection_result['ids']).reshape(-1, 1)
        cv.aruco.drawDetectedMarkers(output_image, detection_result['corners'], ids_array)
    
    # Draw rejected candidates in red (as shown in documentation)
    if detection_result['rejected'] and len(detection_result['rejected']) > 0:
        cv.aruco.drawDetectedMarkers(output_image, detection_result['rejected'], 
                                   np.array([]), (100, 0, 255))  # Red color for rejected
    
    # Add text overlay with detection info
    text = f"{detection_result['name']}: {detection_result['markers']} markers"
    cv.putText(output_image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return output_image

def main():
    # Test images
    test_images = [
        "images/IMG_3790.jpeg",  # Known to have markers
        "images/IMG_3797.jpeg",  # Known to have 1 marker
        "images/IMG_3789.jpeg"   # Test case
    ]
    
    output_dir = "outputs/comprehensive_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("COMPREHENSIVE ARUCO DETECTION TEST")
    print("Following OpenCV Documentation Exactly")
    print("=" * 80)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            # Run comprehensive test
            best_result = comprehensive_aruco_test(image_path)
            
            if best_result:
                # Create visualization
                viz = create_visualization_with_rejected(image_path, best_result)
                if viz is not None:
                    output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_result.png")
                    cv.imwrite(output_path, viz)
                    print(f"\\n💾 Saved visualization: {output_path}")
        else:
            print(f"\\n❌ File not found: {image_path}")
    
    print(f"\\n🎯 All results saved to: {output_dir}/")

if __name__ == "__main__":
    main()