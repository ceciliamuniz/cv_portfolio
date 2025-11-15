"""
Main Flask Application for Module 3 - Complete Image Analysis
Integrates all 5 parts into a single web application
"""

from flask import Flask, render_template, redirect, url_for, flash
from pathlib import Path
import sys
import os

# Add web_integration to path
web_integration_path = Path(__file__).parent / 'web_integration'
sys.path.insert(0, str(web_integration_path))

try:
    # Import the Module 3 blueprint
    from routes import module3_bp
    BLUEPRINT_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import module3 blueprint: {e}")
    BLUEPRINT_LOADED = False

# Create Flask app
app = Flask(__name__, 
           template_folder='web_integration/templates',
           static_folder='web_integration/static')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'cv_module3_demo_key_2024'

# Register the Module 3 blueprint if available
if BLUEPRINT_LOADED:
    app.register_blueprint(module3_bp)
else:
    print("Running in fallback mode without advanced features...")

@app.route('/')
def home():
    """Main page for Module 3."""
    if BLUEPRINT_LOADED:
        return redirect(url_for('module3.index'))
    else:
        # Fallback route
        return render_template('module3_index.html')

@app.route('/gallery')
def gallery_fallback():
    """Gallery fallback route."""
    if BLUEPRINT_LOADED:
        return redirect(url_for('module3.gallery'))
    else:
        return render_template('module3_gallery.html', results={
            'part1': [], 'part2': [], 'part3': [], 'part4': [], 'part5': []
        })

@app.route('/part1-gradient-log')
def part1_fallback():
    """Part 1 fallback route."""
    return render_template('module3_part1.html', images=[])

@app.route('/part2-keypoints')
def part2_fallback():
    """Part 2 fallback route."""
    return render_template('module3_part2.html', images=[])

@app.route('/part3-boundaries')
def part3_fallback():
    """Part 3 fallback route."""
    return render_template('module3_part3.html', images=[])

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)  
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 Module 3: Complete Image Analysis Web Application")
    print("=" * 60)
    print("\n📋 Available Parts:")
    print("  Part 1: Gradient & Laplacian of Gaussian")
    print("  Part 2: Edge & Corner Keypoint Detection") 
    print("  Part 3: Object Boundary Detection")
    print("  Part 4: ArUco Marker-Based Segmentation")
    print("  Part 5: SAM2 Model Comparison")
    print("\n🌐 Starting web server...")
    print("📍 Open your browser to: http://localhost:5000")
    print("🎨 Or try the gallery view: http://localhost:5000/gallery")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)