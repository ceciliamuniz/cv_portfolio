"""
Part 5: SAM2 (Segment Anything Model 2) Comparison
Compare ArUco-based segmentation with Meta's SAM2 model.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import sys

# Try to import SAM2 dependencies
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not installed. Install with:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")


class SAM2Segmentation:
    """SAM2 model wrapper for object segmentation."""
    
    def __init__(self, model_cfg: str = "sam2_hiera_l.yaml", 
                 checkpoint_path: str = None,
                 device: str = None):
        """
        Initialize SAM2 model.
        
        Args:
            model_cfg: SAM2 model configuration
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not installed")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing SAM2 on device: {self.device}")
        
        # Build SAM2 model
        if checkpoint_path is None:
            # Download checkpoint if not provided
            checkpoint_path = self._download_checkpoint(model_cfg)
        
        self.predictor = build_sam2(model_cfg, checkpoint_path, device=self.device)
        
    def _download_checkpoint(self, model_cfg: str) -> str:
        """Download SAM2 checkpoint if needed."""
        # Implement checkpoint download logic
        # For now, raise error if checkpoint not provided
        raise ValueError(
            "Please provide checkpoint_path or download from:\n"
            "https://github.com/facebookresearch/segment-anything-2#model-checkpoints"
        )
    
    def segment_with_points(self, 
                           image: np.ndarray,
                           points: np.ndarray,
                           labels: np.ndarray = None) -> np.ndarray:
        """
        Segment object using point prompts (e.g., ArUco marker centers).
        
        Args:
            image: Input image (RGB)
            points: Nx2 array of point coordinates
            labels: Point labels (1=foreground, 0=background)
            
        Returns:
            Segmentation mask
        """
        if labels is None:
            labels = np.ones(len(points), dtype=np.int32)  # All foreground
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict with points
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        # Return best mask
        return (masks[0] * 255).astype(np.uint8)
    
    def segment_with_box(self,
                        image: np.ndarray,
                        box: np.ndarray) -> np.ndarray:
        """
        Segment object using bounding box prompt.
        
        Args:
            image: Input image (RGB)
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Segmentation mask
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )
        
        return (masks[0] * 255).astype(np.uint8)


class SegmentationComparison:
    """Compare ArUco segmentation with SAM2 results."""
    
    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            IoU score (0-1)
        """
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    @staticmethod
    def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Dice coefficient between two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Dice score (0-1)
        """
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        total = (mask1 > 0).sum() + (mask2 > 0).sum()
        
        if total == 0:
            return 0.0
        
        return float(2 * intersection / total)
    
    @staticmethod
    def calculate_precision_recall(pred_mask: np.ndarray, 
                                   gt_mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate precision and recall.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            
        Returns:
            Tuple of (precision, recall)
        """
        true_positive = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        false_positive = np.logical_and(pred_mask > 0, gt_mask == 0).sum()
        false_negative = np.logical_and(pred_mask == 0, gt_mask > 0).sum()
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        return float(precision), float(recall)
    
    @staticmethod
    def visualize_comparison(image: np.ndarray,
                           aruco_mask: np.ndarray,
                           sam2_mask: np.ndarray,
                           metrics: Dict) -> np.ndarray:
        """
        Create side-by-side comparison visualization.
        
        Args:
            image: Original image
            aruco_mask: ArUco-based segmentation mask
            sam2_mask: SAM2 segmentation mask
            metrics: Dictionary of comparison metrics
            
        Returns:
            Comparison visualization image
        """
        h, w = image.shape[:2]
        
        # Create 4-panel visualization
        panel_width = w
        panel_height = h
        
        # Create blank canvas
        canvas = np.ones((panel_height * 2, panel_width * 2, 3), dtype=np.uint8) * 255
        
        # Panel 1: Original image
        canvas[0:h, 0:w] = image
        
        # Panel 2: ArUco segmentation
        aruco_overlay = image.copy()
        aruco_colored = np.zeros_like(image)
        aruco_colored[:, :, 1] = aruco_mask  # Green
        aruco_overlay = cv.addWeighted(aruco_overlay, 0.6, aruco_colored, 0.4, 0)
        canvas[0:h, w:w*2] = aruco_overlay
        
        # Panel 3: SAM2 segmentation
        sam2_overlay = image.copy()
        sam2_colored = np.zeros_like(image)
        sam2_colored[:, :, 2] = sam2_mask  # Red
        sam2_overlay = cv.addWeighted(sam2_overlay, 0.6, sam2_colored, 0.4, 0)
        canvas[h:h*2, 0:w] = sam2_overlay
        
        # Panel 4: Overlap visualization
        overlap = image.copy()
        # Green = ArUco only, Red = SAM2 only, Yellow = Both
        aruco_only = np.logical_and(aruco_mask > 0, sam2_mask == 0)
        sam2_only = np.logical_and(aruco_mask == 0, sam2_mask > 0)
        both = np.logical_and(aruco_mask > 0, sam2_mask > 0)
        
        overlap_colored = np.zeros_like(image)
        overlap_colored[aruco_only, 1] = 255  # Green
        overlap_colored[sam2_only, 2] = 255   # Red
        overlap_colored[both, 1] = 255        # Yellow
        overlap_colored[both, 2] = 255
        overlap = cv.addWeighted(overlap, 0.6, overlap_colored, 0.4, 0)
        canvas[h:h*2, w:w*2] = overlap
        
        # Add labels and metrics
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 0, 0)
        
        cv.putText(canvas, "Original", (10, 30), font, font_scale, color, thickness)
        cv.putText(canvas, "ArUco Segmentation", (w + 10, 30), font, font_scale, color, thickness)
        cv.putText(canvas, "SAM2 Segmentation", (10, h + 30), font, font_scale, color, thickness)
        cv.putText(canvas, "Overlap Analysis", (w + 10, h + 30), font, font_scale, color, thickness)
        
        # Add metrics text
        metrics_text = [
            f"IoU: {metrics.get('iou', 0):.3f}",
            f"Dice: {metrics.get('dice', 0):.3f}",
            f"Precision: {metrics.get('precision', 0):.3f}",
            f"Recall: {metrics.get('recall', 0):.3f}"
        ]
        
        y_offset = h + 60
        for i, text in enumerate(metrics_text):
            cv.putText(canvas, text, (w + 10, y_offset + i * 30), 
                      font, 0.6, (0, 100, 0), 2)
        
        # Add legend
        legend_y = h + 60
        cv.circle(canvas, (30, legend_y), 10, (0, 255, 0), -1)
        cv.putText(canvas, "ArUco only", (50, legend_y + 5), font, 0.5, color, 1)
        
        cv.circle(canvas, (30, legend_y + 30), 10, (0, 0, 255), -1)
        cv.putText(canvas, "SAM2 only", (50, legend_y + 35), font, 0.5, color, 1)
        
        cv.circle(canvas, (30, legend_y + 60), 10, (0, 255, 255), -1)
        cv.putText(canvas, "Both (overlap)", (50, legend_y + 65), font, 0.5, color, 1)
        
        return canvas


def compare_segmentations(aruco_results_dir: Path,
                         images_dir: Path,
                         output_dir: Path,
                         sam2_checkpoint: str = None) -> List[Dict]:
    """
    Compare ArUco segmentations with SAM2 on all images.
    
    Args:
        aruco_results_dir: Directory with ArUco segmentation results
        images_dir: Directory with original images
        output_dir: Directory to save comparison results
        sam2_checkpoint: Path to SAM2 checkpoint
        
    Returns:
        List of comparison results
    """
    if not SAM2_AVAILABLE:
        print("❌ SAM2 is not available. Please install:")
        print("   pip install torch torchvision")
        print("   pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return []
    
    # Initialize SAM2
    try:
        sam2 = SAM2Segmentation(checkpoint_path=sam2_checkpoint)
    except Exception as e:
        print(f"❌ Failed to initialize SAM2: {e}")
        return []
    
    comparison = SegmentationComparison()
    results = []
    
    # Find all ArUco mask files
    mask_files = list(aruco_results_dir.glob("*_mask.png"))
    print(f"Found {len(mask_files)} ArUco segmentation masks")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, mask_path in enumerate(sorted(mask_files), 1):
        print(f"\n[{i}/{len(mask_files)}] Processing: {mask_path.name}")
        
        # Load ArUco mask
        aruco_mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        
        # Find corresponding original image
        image_name = mask_path.stem.replace('_mask', '')
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = images_dir / f"{image_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            print(f"  ⚠️  Original image not found for {mask_path.name}")
            continue
        
        # Load original image
        image = cv.imread(str(image_path))
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Load ArUco marker centers from JSON (if available)
        # For SAM2, we'll use the ArUco mask to generate point prompts
        aruco_points = np.argwhere(aruco_mask > 0)
        if len(aruco_points) > 100:
            # Sample points if too many
            indices = np.random.choice(len(aruco_points), 100, replace=False)
            aruco_points = aruco_points[indices]
        aruco_points = aruco_points[:, [1, 0]]  # Convert to (x, y)
        
        # Run SAM2 segmentation
        try:
            sam2_mask = sam2.segment_with_points(image_rgb, aruco_points)
        except Exception as e:
            print(f"  ❌ SAM2 failed: {e}")
            continue
        
        # Calculate metrics
        iou = comparison.calculate_iou(aruco_mask, sam2_mask)
        dice = comparison.calculate_dice(aruco_mask, sam2_mask)
        precision, recall = comparison.calculate_precision_recall(sam2_mask, aruco_mask)
        
        metrics = {
            "image": image_path.name,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "aruco_area": int((aruco_mask > 0).sum()),
            "sam2_area": int((sam2_mask > 0).sum())
        }
        
        print(f"  ✓ IoU: {iou:.3f}, Dice: {dice:.3f}")
        
        # Create comparison visualization
        vis = comparison.visualize_comparison(image, aruco_mask, sam2_mask, metrics)
        
        # Save results
        vis_path = output_dir / f"{image_name}_comparison.jpg"
        cv.imwrite(str(vis_path), vis)
        
        sam2_mask_path = output_dir / f"{image_name}_sam2_mask.png"
        cv.imwrite(str(sam2_mask_path), sam2_mask)
        
        results.append(metrics)
    
    # Save summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print overall statistics
    if results:
        avg_iou = np.mean([r['iou'] for r in results])
        avg_dice = np.mean([r['dice'] for r in results])
        print(f"\n📊 Overall Statistics:")
        print(f"  Average IoU: {avg_iou:.3f}")
        print(f"  Average Dice: {avg_dice:.3f}")
        print(f"  Total images compared: {len(results)}")
    
    return results


if __name__ == "__main__":
    # Setup paths
    script_dir = Path(__file__).parent
    aruco_dir = script_dir.parent / "part4_aruco_segmentation"
    aruco_results = aruco_dir / "outputs" / "convex_hull"
    images_dir = aruco_dir / "images"
    output_dir = script_dir / "comparison_results"
    
    # Check if ArUco results exist
    if not aruco_results.exists():
        print("❌ ArUco segmentation results not found.")
        print("Please run part4_aruco_segmentation/aruco_segmentation.py first.")
        sys.exit(1)
    
    # Run comparison
    print("=" * 80)
    print("SAM2 vs ArUco Segmentation Comparison")
    print("=" * 80)
    
    # Note: User needs to provide SAM2 checkpoint path
    checkpoint = None  # TODO: Set path to downloaded SAM2 checkpoint
    
    if checkpoint is None:
        print("\n⚠️  SAM2 checkpoint not specified.")
        print("Download a checkpoint from:")
        print("https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
        print("\nThen update the 'checkpoint' variable in this script.")
    else:
        results = compare_segmentations(aruco_results, images_dir, output_dir, checkpoint)
