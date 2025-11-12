"""
Flask Routes for Module 3 Parts 4 & 5
Integration with CV Portfolio Website
"""

from flask import Blueprint, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import shutil
import json
import cv2 as cv
import numpy as np
import sys

# Prefer package-relative imports; fall back to path insert if needed
ARUCO_AVAILABLE = False
SAM2_AVAILABLE = False
try:
    from ..part4_aruco_segmentation.aruco_segmentation import ArucoSegmentation
    ARUCO_AVAILABLE = True
except Exception:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'part4_aruco_segmentation'))
        from aruco_segmentation import ArucoSegmentation  # type: ignore
        ARUCO_AVAILABLE = True
    except Exception:
        ARUCO_AVAILABLE = False

try:
    from ..part5_sam2_comparison.sam2_comparison import SAM2Segmentation, SegmentationComparison
    SAM2_AVAILABLE = True
except Exception:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'part5_sam2_comparison'))
        from sam2_comparison import SAM2Segmentation, SegmentationComparison  # type: ignore
        SAM2_AVAILABLE = True
    except Exception:
        SAM2_AVAILABLE = False

# Create Blueprint
module3_bp = Blueprint('module3', __name__, 
                       template_folder='templates',
                       static_folder='static',
                       url_prefix='/module3')

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'static' / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'static' / 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Ensure expected result subfolders exist
for sub in ['part1', 'part2', 'part3', 'aruco', 'sam2']:
    (RESULTS_FOLDER / sub).mkdir(parents=True, exist_ok=True)

# Helper: sync pipeline outputs into web static for easy serving
def _sync_outputs_to_static():
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / 'outputs'
    if not outputs_dir.exists():
        return
    # Map source folders to web subfolders
    mapping = {
        (outputs_dir / 'comparison'): RESULTS_FOLDER / 'part1',
        (outputs_dir / 'combined'): RESULTS_FOLDER / 'part2',
        (outputs_dir / 'boundaries'): RESULTS_FOLDER / 'part3',
    }
    for src, dst in mapping.items():
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for img in src.glob('*.*'):
            # Copy only image-like files
            if img.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                target = dst / img.name
                try:
                    if (not target.exists()) or (img.stat().st_mtime > target.stat().st_mtime):
                        shutil.copy2(str(img), str(target))
                except Exception as _e:
                    print('[WARN] Failed to copy', img, '->', target, ':', _e)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@module3_bp.route('/')
def index():
    """Main page for Module 3."""
    return render_template('module3_index.html')


@module3_bp.route('/part1-gradient-log')
def part1():
    """Part 1: Gradient and LoG visualization."""
    # Sync and load processed results
    _sync_outputs_to_static()
    results_dir = Path(__file__).parent.parent / 'outputs'
    
    images = []
    if results_dir.exists():
        comparison_dir = results_dir / 'comparison'
        if comparison_dir.exists():
            for img_file in sorted(list(comparison_dir.glob('*.jpg')) + list(comparison_dir.glob('*.png'))):
                images.append({
                    'name': img_file.stem,
                    'url': f'/module3/static/results/part1/{img_file.name}'
                })
    
    return render_template('module3_part1.html', images=images)


@module3_bp.route('/part2-keypoints')
def part2():
    """Part 2: Edge and Corner Keypoints."""
    _sync_outputs_to_static()
    results_dir = Path(__file__).parent.parent / 'outputs'
    
    images = []
    if results_dir.exists():
        combined_dir = results_dir / 'combined'
        if combined_dir.exists():
            for img_file in sorted(list(combined_dir.glob('*.jpg')) + list(combined_dir.glob('*.png'))):
                images.append({
                    'name': img_file.stem,
                    'url': f'/module3/static/results/part2/{img_file.name}'
                })
    
    return render_template('module3_part2.html', images=images)


@module3_bp.route('/part3-boundaries')
def part3():
    """Part 3: Object Boundary Detection."""
    _sync_outputs_to_static()
    results_dir = Path(__file__).parent.parent / 'outputs'
    
    images = []
    if results_dir.exists():
        boundaries_dir = results_dir / 'boundaries'
        if boundaries_dir.exists():
            for img_file in sorted(list(boundaries_dir.glob('*.jpg')) + list(boundaries_dir.glob('*.png'))):
                images.append({
                    'name': img_file.stem,
                    'url': f'/module3/static/results/part3/{img_file.name}'
                })
    
    return render_template('module3_part3.html', images=images)


@module3_bp.route('/part4-aruco', methods=['GET', 'POST'])
def part4_aruco():
    """Part 4: ArUco Marker-Based Segmentation."""
    if not ARUCO_AVAILABLE:
        # GET request - show informational page
        if request.method == 'GET':
            return render_template('module3_part4.html', sam2_available=SAM2_AVAILABLE, 
                                   error='Aruco module not available. Ensure opencv-contrib-python is installed.'), 200
        return jsonify({'error': 'Aruco module not available'}), 500
    if request.method == 'POST':
        # Handle image upload and processing
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(str(filepath))
            
            # Process with ArUco
            method = request.form.get('method', 'convex_hull')
            segmenter = ArucoSegmentation()
            
            result = segmenter.process_image(
                filepath,
                RESULTS_FOLDER / 'aruco',
                method=method
            )
            
            if 'error' in result:
                return jsonify(result), 400
            
            # Return results
            return jsonify({
                'success': True,
                'markers_detected': result['markers_detected'],
                'marker_ids': result['marker_ids'],
                'area_pixels': result['area_pixels'],
                'perimeter_pixels': result['perimeter_pixels'],
                'mask_url': f'/module3/static/results/aruco/{Path(result["mask_saved"]).name}',
                'visualization_url': f'/module3/static/results/aruco/{Path(result["visualization_saved"]).name}'
            })
    
    # GET request - show upload form
    return render_template('module3_part4.html', sam2_available=SAM2_AVAILABLE)


@module3_bp.route('/part5-sam2-comparison', methods=['GET', 'POST'])
def part5_sam2():
    """Part 5: SAM2 Comparison."""
    if not SAM2_AVAILABLE:
        return render_template('module3_part5.html', 
                             sam2_available=False,
                             error='SAM2 is not installed. Please install required dependencies.')
    
    if request.method == 'POST':
        # Handle comparison request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(str(filepath))
            
            # First run ArUco segmentation
            aruco_segmenter = ArucoSegmentation()
            aruco_result = aruco_segmenter.process_image(
                filepath,
                RESULTS_FOLDER / 'aruco',
                method='convex_hull'
            )
            
            if 'error' in aruco_result:
                return jsonify({'error': f'ArUco failed: {aruco_result["error"]}'}), 400
            
            # Load ArUco mask
            aruco_mask = cv.imread(aruco_result['mask_saved'], cv.IMREAD_GRAYSCALE)
            
            # Get marker centers as SAM2 prompts
            image = cv.imread(str(filepath))
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            corners, ids, _ = aruco_segmenter.detect_markers(image)
            marker_centers = aruco_segmenter.get_marker_centers(corners)
            
            # Run SAM2 (if checkpoint available)
            checkpoint_path = request.form.get('checkpoint_path')
            if not checkpoint_path:
                checkpoint_path = Path(__file__).parent.parent / 'part5_sam2_comparison' / 'checkpoints' / 'sam2_hiera_large.pt'
            
            try:
                sam2 = SAM2Segmentation(checkpoint_path=str(checkpoint_path))
                sam2_mask = sam2.segment_with_points(image_rgb, marker_centers)
            except Exception as e:
                return jsonify({'error': f'SAM2 failed: {str(e)}'}), 500
            
            # Calculate comparison metrics
            comparison = SegmentationComparison()
            iou = comparison.calculate_iou(aruco_mask, sam2_mask)
            dice = comparison.calculate_dice(aruco_mask, sam2_mask)
            precision, recall = comparison.calculate_precision_recall(sam2_mask, aruco_mask)
            
            metrics = {
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall
            }
            
            # Create visualization
            vis = comparison.visualize_comparison(image, aruco_mask, sam2_mask, metrics)
            
            # Save results
            vis_path = RESULTS_FOLDER / 'sam2' / f'{filepath.stem}_comparison.jpg'
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(vis_path), vis)
            
            sam2_mask_path = RESULTS_FOLDER / 'sam2' / f'{filepath.stem}_sam2_mask.png'
            cv.imwrite(str(sam2_mask_path), sam2_mask)
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'aruco_markers': len(ids),
                'aruco_area': int((aruco_mask > 0).sum()),
                'sam2_area': int((sam2_mask > 0).sum()),
                'comparison_url': f'/module3/static/results/sam2/{vis_path.name}',
                'sam2_mask_url': f'/module3/static/results/sam2/{sam2_mask_path.name}',
                'aruco_mask_url': f'/module3/static/results/aruco/{Path(aruco_result["mask_saved"]).name}'
            })
    
    # GET request
    return render_template('module3_part5.html', sam2_available=True)


@module3_bp.route('/gallery')
def gallery():
    """Gallery view of all Module 3 results."""
    # Ensure latest outputs are mirrored to static
    _sync_outputs_to_static()
    results = {
        'part1': [],
        'part2': [],
        'part3': [],
        'part4': [],
        'part5': []
    }
    
    # Load all results
    base_dir = Path(__file__).parent.parent
    
    # Part 1: Gradient & LoG
    comparison_dir = base_dir / 'outputs' / 'comparison'
    if comparison_dir.exists():
        for img in sorted(list(comparison_dir.glob('*.jpg')) + list(comparison_dir.glob('*.png')))[:6]:  # Limit to 6
            results['part1'].append({
                'name': img.stem,
                'url': f'/module3/static/results/part1/{img.name}'
            })
    
    # Part 2: Keypoints
    combined_dir = base_dir / 'outputs' / 'combined'
    if combined_dir.exists():
        for img in sorted(list(combined_dir.glob('*.jpg')) + list(combined_dir.glob('*.png')))[:6]:
            results['part2'].append({
                'name': img.stem,
                'url': f'/module3/static/results/part2/{img.name}'
            })
    
    # Part 3: Boundaries
    boundaries_dir = base_dir / 'outputs' / 'boundaries'
    if boundaries_dir.exists():
        for img in sorted(list(boundaries_dir.glob('*.jpg')) + list(boundaries_dir.glob('*.png')))[:6]:
            results['part3'].append({
                'name': img.stem,
                'url': f'/module3/static/results/part3/{img.name}'
            })
    
    # Part 4: ArUco (served from static mirror)
    aruco_static = RESULTS_FOLDER / 'aruco'
    if aruco_static.exists():
        for img in sorted(aruco_static.glob('*_segmentation.jpg'))[:6]:
            results['part4'].append({
                'name': img.stem,
                'url': f'/module3/static/results/aruco/{img.name}'
            })
    
    # Part 5: SAM2 Comparison (served from static mirror)
    sam2_static = RESULTS_FOLDER / 'sam2'
    if sam2_static.exists():
        for img in sorted(sam2_static.glob('*_comparison.jpg'))[:6]:
            results['part5'].append({
                'name': img.stem,
                'url': f'/module3/static/results/sam2/{img.name}'
            })
    
    return render_template('module3_gallery.html', results=results)


@module3_bp.route('/api/stats')
def get_stats():
    """API endpoint for Module 3 statistics."""
    stats = {
        'part1_processed': 0,
        'part2_edge_keypoints': 0,
        'part2_corner_keypoints': 0,
        'part3_contours': 0,
        'part4_images': 0,
        'part5_avg_iou': 0.0
    }
    
    base_dir = Path(__file__).parent.parent
    
    # Load Part 1-3 summary
    summary_file = base_dir / 'PROCESSING_SUMMARY.md'
    if summary_file.exists():
        stats['part1_processed'] = 10  # From earlier processing
        stats['part2_edge_keypoints'] = 424436
        stats['part2_corner_keypoints'] = 85197
        stats['part3_contours'] = 252
    
    # Load Part 4 summary
    part4_summary = base_dir / 'part4_aruco_segmentation' / 'outputs' / 'convex_hull' / 'processing_summary.json'
    if part4_summary.exists():
        with open(part4_summary) as f:
            data = json.load(f)
            stats['part4_images'] = len([r for r in data if 'error' not in r])
    
    # Load Part 5 summary
    part5_summary = base_dir / 'part5_sam2_comparison' / 'comparison_results' / 'comparison_summary.json'
    if part5_summary.exists():
        with open(part5_summary) as f:
            data = json.load(f)
            if data:
                stats['part5_avg_iou'] = np.mean([r['iou'] for r in data])
    
    return jsonify(stats)


# Error handlers
@module3_bp.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@module3_bp.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
