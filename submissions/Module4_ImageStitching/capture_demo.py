#!/usr/bin/env python3
"""
Image Acquisition Guidelines and Demo Script for Module 4

This script provides guidelines for capturing images for panorama stitching
and includes a demo mode for testing with webcam or existing images.

Author: Cecilia Muniz Siqueira
Module: CV_Module4_ImageStitching
"""

import cv2 as cv
import numpy as np
import time
from pathlib import Path
import argparse


class ImageAcquisitionDemo:
    """Interactive demo for proper image acquisition techniques."""
    
    def __init__(self):
        self.images = []
        self.capture_count = 0
        self.target_overlap = 0.3  # 30% overlap target
        
    def display_guidelines(self):
        """Display image acquisition best practices."""
        guidelines = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        IMAGE ACQUISITION GUIDELINES                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📷 CAMERA SETUP:                                                            ║
║    • Use manual focus (lock focus on first image)                           ║
║    • Set manual exposure or use exposure lock                               ║
║    • Use tripod or steady handheld technique                                ║
║    • Shoot in RAW or highest JPEG quality                                   ║
║                                                                              ║
║  📐 COMPOSITION REQUIREMENTS:                                                ║
║    • LANDSCAPE: Minimum 4 images (recommended 6-8)                          ║
║    • PORTRAIT: Minimum 8 images (recommended 12-16)                         ║
║    • Overlap: 20-40% between consecutive images                             ║
║    • Keep camera level and rotate around optical center                     ║
║                                                                              ║
║  🌅 LIGHTING CONDITIONS:                                                     ║
║    • Avoid dramatic lighting changes between shots                          ║
║    • Best results in even, diffused lighting                               ║
║    • Avoid backlighting and strong shadows                                 ║
║    • Golden hour provides excellent conditions                             ║
║                                                                              ║
║  🎯 SCENE SELECTION:                                                         ║
║    • Choose scenes with good texture and features                           ║
║    • Avoid repetitive patterns (water, sky, walls)                         ║
║    • Include foreground elements for depth                                 ║
║    • Minimize moving objects (people, vehicles, water)                     ║
║                                                                              ║
║  📱 MOBILE COMPARISON:                                                       ║
║    • Capture mobile panorama of same scene immediately after               ║
║    • Use same approximate viewpoint and field of view                      ║
║    • Save both results for quality comparison                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(guidelines)
        
    def estimate_overlap(self, img1, img2):
        """
        Estimate overlap between two consecutive images.
        
        Returns approximate overlap percentage.
        """
        try:
            # Convert to grayscale
            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            
            # Use SIFT for quick feature matching
            sift = cv.SIFT_create(nfeatures=500)  # Limited features for speed
            
            kp1, desc1 = sift.detectAndCompute(gray1, None)
            kp2, desc2 = sift.detectAndCompute(gray2, None)
            
            if desc1 is None or desc2 is None:
                return 0
            
            # Quick matching
            bf = cv.BFMatcher(cv.NORM_L2)
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Estimate overlap based on matching keypoints and their distribution
            if len(good_matches) < 10:
                return 0
            
            # Get matched points
            pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
            
            # Estimate overlap based on horizontal distribution of matches
            width1, width2 = img1.shape[1], img2.shape[1]
            
            # Simple heuristic: overlap based on match distribution
            x_range1 = np.max(pts1[:, 0]) - np.min(pts1[:, 0])
            x_range2 = np.max(pts2[:, 0]) - np.min(pts2[:, 0])
            
            # Estimate overlap percentage
            overlap_estimate = min(x_range1 / width1, x_range2 / width2) * 1.5
            return min(overlap_estimate, 1.0)
            
        except Exception:
            return 0
    
    def webcam_capture_demo(self, output_dir="captured_images"):
        """Interactive webcam capture with overlap feedback."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize webcam
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n🎥 WEBCAM CAPTURE MODE")
        print("=" * 50)
        print("Instructions:")
        print("• Position camera for first shot")
        print("• Press SPACE to capture image")  
        print("• Rotate camera slightly (20-40% overlap)")
        print("• Capture next image when overlap indicator is GREEN")
        print("• Press 'q' to quit")
        print("• Press 'r' to reset and start over")
        
        last_image = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create display frame with overlay
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Add capture count
            cv.putText(display_frame, f"Images captured: {self.capture_count}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add guidelines overlay
            cv.putText(display_frame, "SPACE: Capture | Q: Quit | R: Reset", 
                      (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Check overlap if we have a previous image
            if last_image is not None:
                overlap = self.estimate_overlap(last_image, frame)
                
                # Color code overlap feedback
                if 0.2 <= overlap <= 0.4:
                    color = (0, 255, 0)  # Green - good overlap
                    status = "GOOD OVERLAP"
                elif overlap < 0.2:
                    color = (0, 165, 255)  # Orange - too little overlap
                    status = "MORE OVERLAP NEEDED"
                else:
                    color = (0, 0, 255)  # Red - too much overlap
                    status = "TOO MUCH OVERLAP"
                
                cv.putText(display_frame, f"Overlap: {overlap:.1%} - {status}", 
                          (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show target overlap zone
            zone_width = int(w * 0.3)  # 30% target overlap
            cv.rectangle(display_frame, (w - zone_width, 0), (w, h), (0, 255, 255), 2)
            cv.putText(display_frame, "TARGET", (w - zone_width + 10, 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv.imshow('Image Acquisition Demo', display_frame)
            
            key = cv.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to capture
                filename = f"capture_{self.capture_count:03d}.jpg"
                filepath = output_path / filename
                cv.imwrite(str(filepath), frame)
                
                print(f"Captured: {filename}")
                self.capture_count += 1
                last_image = frame.copy()
                
                # Check if we have enough images
                if self.capture_count >= 4:
                    print(f"✓ Captured {self.capture_count} images - sufficient for stitching!")
                
            elif key == ord('r'):  # Reset
                self.capture_count = 0
                last_image = None
                print("Reset - starting over")
                
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv.destroyAllWindows()
        
        if self.capture_count >= 4:
            print(f"\n✓ Capture complete! {self.capture_count} images saved to {output_dir}")
            print("You can now use these images for stitching:")
            print(f"python test_stitching.py --images {output_dir}/*.jpg")
        else:
            print(f"\n⚠ Only {self.capture_count} images captured. Need at least 4 for stitching.")
    
    def analyze_existing_images(self, image_paths):
        """Analyze existing images for stitching suitability."""
        print("\n📊 IMAGE ANALYSIS")
        print("=" * 50)
        
        if len(image_paths) < 4:
            print(f"❌ Insufficient images: {len(image_paths)} (need at least 4)")
            return False
        
        # Load images
        images = []
        for path in image_paths:
            img = cv.imread(path)
            if img is not None:
                images.append((img, path))
            else:
                print(f"❌ Could not load: {path}")
        
        if len(images) < 4:
            print(f"❌ Only {len(images)} valid images loaded")
            return False
        
        print(f"✓ Loaded {len(images)} images")
        
        # Analyze each image
        print("\\nImage Analysis:")
        total_overlap = 0
        valid_pairs = 0
        
        for i, (img, path) in enumerate(images):
            h, w = img.shape[:2]
            aspect = w / h
            
            print(f"  {i+1:2d}. {Path(path).name}")
            print(f"      Resolution: {w}x{h} (aspect: {aspect:.2f})")
            
            # Check consecutive overlap
            if i < len(images) - 1:
                next_img = images[i + 1][0]
                overlap = self.estimate_overlap(img, next_img)
                total_overlap += overlap
                valid_pairs += 1
                
                if overlap >= 0.2:
                    print(f"      Overlap with next: {overlap:.1%} ✓")
                else:
                    print(f"      Overlap with next: {overlap:.1%} ❌ (too low)")
        
        # Overall assessment
        avg_overlap = total_overlap / valid_pairs if valid_pairs > 0 else 0
        print(f"\\nOverall Assessment:")
        print(f"  Average overlap: {avg_overlap:.1%}")
        
        if avg_overlap >= 0.2:
            print("  ✓ Images appear suitable for stitching")
            return True
        else:
            print("  ❌ Low overlap detected - stitching may fail")
            print("  💡 Consider capturing additional images with more overlap")
            return False


def main():
    parser = argparse.ArgumentParser(description='Image Acquisition Demo for Panorama Stitching')
    parser.add_argument('--mode', choices=['guidelines', 'webcam', 'analyze'], 
                       default='guidelines',
                       help='Demo mode: show guidelines, capture with webcam, or analyze existing images')
    parser.add_argument('--images', nargs='+', help='Image paths to analyze (for analyze mode)')
    parser.add_argument('--output-dir', default='captured_images', 
                       help='Output directory for captured images')
    
    args = parser.parse_args()
    
    demo = ImageAcquisitionDemo()
    
    if args.mode == 'guidelines':
        demo.display_guidelines()
        
    elif args.mode == 'webcam':
        demo.display_guidelines()
        input("\\nPress Enter to start webcam capture demo...")
        demo.webcam_capture_demo(args.output_dir)
        
    elif args.mode == 'analyze':
        if not args.images:
            print("Error: --images required for analyze mode")
            return
        
        demo.analyze_existing_images(args.images)
    
    print("\\n💡 Tips for best results:")
    print("   • Use consistent lighting and exposure")
    print("   • Keep the camera steady and level")
    print("   • Rotate around the camera's optical center") 
    print("   • Include rich textures for feature detection")
    print("   • Avoid moving objects in the scene")


if __name__ == "__main__":
    main()