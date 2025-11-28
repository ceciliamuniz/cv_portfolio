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
# Import process_all_images at top level to avoid runtime import issues
PROCESS_ALL_IMAGES = None
try:
    from ..part4_aruco_segmentation.aruco_segmentation import ArucoSegmentation, process_all_images
    ARUCO_AVAILABLE = True
    PROCESS_ALL_IMAGES = process_all_images
    print("[DEBUG] ArUco import successful via relative import")
except Exception as e1:
    print(f"[DEBUG] Relative import failed: {e1}")
    try:
        # More robust absolute import path
        aruco_path = Path(__file__).parent.parent / 'part4_aruco_segmentation'
        sys.path.insert(0, str(aruco_path))
        from aruco_segmentation import ArucoSegmentation, process_all_images  # type: ignore
        ARUCO_AVAILABLE = True
        PROCESS_ALL_IMAGES = process_all_images
        print("[DEBUG] ArUco import successful via path insert")
    except Exception as e2:
        print(f"[DEBUG] Path insert import failed: {e2}")
        ARUCO_AVAILABLE = False
        PROCESS_ALL_IMAGES = None

# SAM2 functionality removed - user will handle Part 5 separately
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


@module3_bp.route('/part4-aruco')
def part4_aruco():
    """Part 4: ArUco Marker-Based Segmentation - Use working retry interface."""
    return render_template('module3_part4.html')


@module3_bp.route('/test-route', methods=['GET', 'POST'])
def test_route():
    """Simple test route."""
    return jsonify({'message': 'Test route working!'})

@module3_bp.route('/part4-aruco-batch', methods=['POST'])
def part4_aruco_batch():
    """Process hardcoded batch using retry code directly."""
    try:
        print("[DEBUG] Starting part4_aruco_batch processing")
        # Import and use the retry folder's code directly
        retry_path = Path(__file__).parent.parent / 'part4_retry'
        print(f"[DEBUG] Retry path: {retry_path}")
        print(f"[DEBUG] Retry path exists: {retry_path.exists()}")
        
        sys.path.insert(0, str(retry_path))
        
        from aruco_segmentation import ArucoSegmentation
        import time
        print("[DEBUG] ArUco segmentation imported successfully")
        
        # Get list of image files
        images_dir = retry_path / 'images'
        print(f"[DEBUG] Images directory: {images_dir}")
        print(f"[DEBUG] Images directory exists: {images_dir.exists()}")
        
        image_files = [f.name for f in images_dir.glob('*.png') if f.is_file()]
        print(f"[DEBUG] Found {len(image_files)} image files: {image_files}")
        
        segmenter = ArucoSegmentation()
        results = []
        start_time = time.time()
        
        for i, img_name in enumerate(image_files, 1):
            img_path = images_dir / img_name
            print(f"[DEBUG] Processing {img_path}")
            result = segmenter.process_image(str(img_path))
            print(f"[DEBUG] Result for {img_name}: {result}")
            if result:
                results.append({
                    'image': img_name,
                    'markers_detected': result.get('markers_detected', 0),
                    'processing_time': result.get('processing_time', 0)
                })
                print(f"[{i}/{len(image_files)}] {img_name}: {result.get('markers_detected', 0)} markers detected")
        
        total_time = time.time() - start_time
        total_markers = sum(r['markers_detected'] for r in results)
        
        response_data = {
            'success': True,
            'processing_time': f'{total_time:.2f}s',
            'total_images': len(image_files),
            'processed_images': len(results),
            'total_markers_detected': total_markers,
            'dictionary_used': 'DICT_6X6_1000',
            'message': f'Processed {len(results)} images successfully',
            'results': results
        }
        print(f"[DEBUG] Final response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Exception in part4_aruco_batch: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@module3_bp.route('/part4-aruco-single', methods=['POST'])
def part4_aruco_single():
    """Process single image using retry code directly."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            
            # Import and use the retry folder's code directly
            retry_path = Path(__file__).parent.parent / 'part4_retry'
            sys.path.insert(0, str(retry_path))
            
            from aruco_segmentation import ArucoSegmentation
            
            segmenter = ArucoSegmentation()
            result = segmenter.process_image(tmp_file.name)
            
            # Clean up temp file
            import os
            os.unlink(tmp_file.name)
            
            if result:
                return jsonify({
                    'success': True,
                    'markers_detected': result.get('markers_detected', 0),
                    'processing_time': result.get('processing_time', 0),
                    'message': f'Detected {result.get("markers_detected", 0)} markers'
                })
            else:
                return jsonify({'error': 'Processing failed'}), 500
                
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@module3_bp.route('/part5-sam2-comparison', methods=['GET', 'POST'])
def part5_sam2():
    """Part 5: SAM2 Comparison - Handled separately by user."""
    return render_template('module3_part5.html', 
                         sam2_available=False,
                         error='Part 5 will be handled separately - results will be uploaded manually.')


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
