#!/usr/bin/env python3
"""
Comprehensive Test Script for Module 4 Image Stitching

This script provides automated testing and demonstration of the complete
image stitching pipeline including SIFT implementation, RANSAC optimization,
and quality assessment.

Author: Cecilia Muniz Siqueira
Module: CV_Module4_ImageStitching
"""

import cv2 as cv
import numpy as np
import time
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Import our modules
import sift_scratch
import stitching as stitch_utils


def create_test_images(output_dir="test_images", image_type="landscape"):
    """
    Create synthetic test images for stitching demonstration.
    
    Args:
        output_dir: Directory to save test images
        image_type: 'landscape' or 'portrait'
    
    Returns:
        List of created image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Image dimensions
    if image_type == "landscape":
        width, height = 800, 600
        num_images = 4
        shift = 200  # Horizontal shift for overlap
    else:
        width, height = 600, 800
        num_images = 8
        shift = 160  # Vertical shift for overlap
    
    image_paths = []
    
    print(f"Creating {num_images} {image_type} test images...")
    
    for i in range(num_images):
        # Create a base pattern
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    (x + i * 50) % 255,
                    (y + i * 30) % 255,
                    ((x + y) + i * 40) % 255
                ]
        
        # Add geometric patterns for feature detection
        # Circles
        for j in range(5):
            center_x = (width // 6) * (j + 1)
            center_y = height // 3 + (i * 20) % (height // 3)
            radius = 30 + (i * 10) % 30
            color = (255 - i * 40, 255, i * 60)
            cv.circle(image, (center_x, center_y), radius, color, 3)
        
        # Rectangles
        for j in range(3):
            x1 = (width // 4) * j + i * 20
            y1 = 2 * height // 3 + (i * 15) % (height // 6)
            x2 = x1 + 80
            y2 = y1 + 60
            color = (i * 50, 255 - i * 30, 200)
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Text labels
        text = f"Image {i+1}"
        font = cv.FONT_HERSHEY_SIMPLEX
        text_size = cv.getTextSize(text, font, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv.putText(image, text, (text_x, text_y), font, 2, (255, 255, 255), 3)
        
        # Add some noise for realistic features
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        filename = f"test_{image_type}_{i+1:02d}.jpg"
        filepath = output_path / filename
        cv.imwrite(str(filepath), image)
        image_paths.append(str(filepath))
        
        print(f"  Created: {filename}")
    
    return image_paths


def run_stitching_test(image_paths, output_dir="results", blend_modes=None):
    """
    Run comprehensive stitching tests with different configurations.
    
    Args:
        image_paths: List of image file paths
        output_dir: Directory for results
        blend_modes: List of blending modes to test
    
    Returns:
        Dictionary with test results
    """
    if blend_modes is None:
        blend_modes = ['simple', 'feather', 'multiband']
    
    results_path = Path(output_dir)
    results_path.mkdir(exist_ok=True)
    
    # Load images
    print(f"Loading {len(image_paths)} images...")
    images = []
    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load {path}")
    
    if len(images) < 2:
        raise ValueError("Need at least 2 valid images for stitching")
    
    print(f"Successfully loaded {len(images)} images")
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'input_images': len(images),
        'image_resolutions': [f"{img.shape[1]}x{img.shape[0]}" for img in images],
        'blend_tests': {},
        'sift_comparison': None,
        'performance_metrics': {}
    }
    
    # Test different blending modes
    for blend_mode in blend_modes:
        print(f"\n--- Testing {blend_mode} blending ---")
        
        try:
            blend_start_time = time.time()
            
            # Step 1: Feature extraction and matching
            print("1. Computing features and matches...")
            feature_start = time.time()
            
            homographies = [np.eye(3)]  # Reference image
            match_info = []
            
            for i in range(len(images) - 1):
                print(f"   Processing pair {i}-{i+1}")
                
                # Get matches using custom SIFT
                match_data = stitch_utils.compute_features_and_matches(
                    images[i], images[i+1], sift_scratch, use_custom=True
                )
                
                if 'error' in match_data:
                    raise Exception(f"Feature matching failed: {match_data['error']}")
                
                pts1 = match_data['points1']
                pts2 = match_data['points2']
                
                match_info.append({
                    'pair': f"{i}-{i+1}",
                    'matches': len(pts1),
                    'keypoints1': match_data['keypoints1'],
                    'keypoints2': match_data['keypoints2']
                })
                
                if len(pts1) < 4:
                    raise Exception(f"Insufficient matches: {len(pts1)} < 4")
                
                # Estimate homography
                H, inliers = stitch_utils.estimate_homography_ransac(
                    pts1, pts2, threshold=4.0, max_iterations=5000
                )
                
                if H is None:
                    raise Exception("Homography estimation failed")
                
                print(f"   Found {len(inliers)}/{len(pts1)} inliers")
                homographies.append(homographies[-1] @ np.linalg.inv(H))
            
            feature_time = time.time() - feature_start
            
            # Step 2: Warping and blending
            print("2. Warping and blending...")
            blend_start = time.time()
            
            panorama = stitch_utils.advanced_warp_and_blend(
                images, homographies, reference=0, blend_mode=blend_mode
            )
            
            blend_time = time.time() - blend_start
            total_time = time.time() - blend_start_time
            
            # Step 3: Quality assessment
            print("3. Assessing quality...")
            quality_metrics = stitch_utils.assess_panorama_quality(panorama)
            
            # Save result
            result_filename = f"panorama_{blend_mode}_{datetime.now().strftime('%H%M%S')}.jpg"
            result_path = results_path / result_filename
            cv.imwrite(str(result_path), panorama)
            
            # Store results
            test_results['blend_tests'][blend_mode] = {
                'success': True,
                'output_file': str(result_path),
                'output_resolution': f"{panorama.shape[1]}x{panorama.shape[0]}",
                'processing_times': {
                    'feature_extraction': feature_time,
                    'blending': blend_time,
                    'total': total_time
                },
                'match_info': match_info,
                'quality_metrics': quality_metrics
            }
            
            print(f"   Success! Saved to {result_filename}")
            print(f"   Quality score: {quality_metrics['overall_quality']:.3f}")
            print(f"   Processing time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"   Failed: {e}")
            test_results['blend_tests'][blend_mode] = {
                'success': False,
                'error': str(e)
            }
    
    # SIFT comparison test (using first two images)
    if len(images) >= 2:
        print(f"\n--- SIFT Implementation Comparison ---")
        try:
            comparison = sift_scratch.compare_with_opencv_sift(images[0], images[1])
            test_results['sift_comparison'] = comparison
            
            print("Custom SIFT:")
            print(f"  Keypoints: {comparison['custom']['keypoints_count']}")
            print(f"  Matches: {comparison['custom']['matches_count']}")
            print(f"  Time: {comparison['custom']['processing_time']:.2f}s")
            
            if 'opencv' in comparison and 'error' not in comparison['opencv']:
                print("OpenCV SIFT:")
                print(f"  Keypoints: {comparison['opencv']['keypoints_count']}")
                print(f"  Matches: {comparison['opencv']['matches_count']}")
                print(f"  Time: {comparison['opencv']['processing_time']:.2f}s")
                
                if 'comparison' in comparison:
                    comp = comparison['comparison']
                    print(f"Speed ratio (OpenCV/Custom): {comp['speed_ratio']:.2f}")
                    print(f"Keypoint ratio: {comp['keypoint_ratio']:.2f}")
                    print(f"Match ratio: {comp['match_ratio']:.2f}")
        
        except Exception as e:
            print(f"SIFT comparison failed: {e}")
            test_results['sift_comparison'] = {'error': str(e)}
    
    return test_results


def save_test_report(results, output_file="test_report.json"):
    """Save test results to JSON file with pretty formatting."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTest report saved to: {output_file}")


def print_test_summary(results):
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"Timestamp: {results['timestamp']}")
    print(f"Input images: {results['input_images']}")
    
    print(f"\nBlending Mode Results:")
    for mode, result in results['blend_tests'].items():
        if result['success']:
            quality = result['quality_metrics']['overall_quality']
            time_taken = result['processing_times']['total']
            print(f"  {mode:12} | Quality: {quality:.3f} | Time: {time_taken:6.2f}s | ✓")
        else:
            print(f"  {mode:12} | {result['error']:30} | ✗")
    
    if results['sift_comparison'] and 'error' not in results['sift_comparison']:
        comp = results['sift_comparison']
        print(f"\nSIFT Comparison:")
        print(f"  Custom keypoints:  {comp['custom']['keypoints_count']:6}")
        print(f"  OpenCV keypoints:  {comp['opencv']['keypoints_count']:6}")
        print(f"  Speed ratio:       {comp['comparison']['speed_ratio']:6.2f}x")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Test Module 4 Image Stitching Implementation')
    parser.add_argument('--images', '-i', nargs='+', help='Input image paths')
    parser.add_argument('--create-test', choices=['landscape', 'portrait'], 
                       help='Create synthetic test images')
    parser.add_argument('--output-dir', '-o', default='test_results', 
                       help='Output directory for results')
    parser.add_argument('--blend-modes', nargs='+', 
                       choices=['simple', 'feather', 'multiband'],
                       default=['simple', 'feather', 'multiband'],
                       help='Blending modes to test')
    parser.add_argument('--report-file', default='test_report.json',
                       help='Output file for test report')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get image paths
    if args.create_test:
        print(f"Creating {args.create_test} test images...")
        image_paths = create_test_images(
            output_dir=output_path / "test_images", 
            image_type=args.create_test
        )
    elif args.images:
        image_paths = args.images
        print(f"Using provided images: {len(image_paths)} files")
    else:
        # Look for existing test images
        test_dir = Path("test_images")
        if test_dir.exists():
            image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            image_paths = [str(p) for p in sorted(image_paths)]
            if image_paths:
                print(f"Found {len(image_paths)} test images")
            else:
                print("No test images found. Use --create-test or --images")
                return
        else:
            print("No images specified. Use --create-test or --images")
            return
    
    # Validate minimum image count
    if len(image_paths) < 2:
        print("Error: Need at least 2 images for stitching")
        return
    
    # Run tests
    print(f"\nStarting stitching tests with {len(image_paths)} images...")
    results = run_stitching_test(
        image_paths, 
        output_dir=args.output_dir,
        blend_modes=args.blend_modes
    )
    
    # Save and display results
    report_path = output_path / args.report_file
    save_test_report(results, report_path)
    print_test_summary(results)
    
    # Display success status
    successful_tests = sum(1 for test in results['blend_tests'].values() if test['success'])
    total_tests = len(results['blend_tests'])
    
    print(f"\nTesting completed: {successful_tests}/{total_tests} successful")
    
    if successful_tests > 0:
        print(f"Results saved in: {args.output_dir}")
        print("View panoramas and check quality metrics in the test report.")


if __name__ == "__main__":
    main()