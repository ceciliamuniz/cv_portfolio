import os
from glob import glob
from flask import Flask, render_template, send_from_directory
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'images'
OUT_DIR = ROOT / 'outputs'

app = Flask(__name__)


def list_images(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    files = []
    for e in exts:
        files.extend(Path(folder).glob(e))
    return sorted(files)


@app.route('/')
def index():
    imgs = list_images(DATA_DIR)
    return render_template('index.html', images=[p.name for p in imgs])


@app.route('/image/<name>')
def image_detail(name):
    base = name.rsplit('.', 1)[0]
    paths = {
        'original': f'/data/{name}',
        'contours': f'/out/contours/{base}_contours.png',
        'grabcut': f'/out/grabcut/{base}_grabcut.png',
        'boundaries': f'/out/boundaries/{base}_boundaries.png',
        'comparison': f'/out/comparison/{base}_comparison.png',
    }
    return render_template('detail.html', name=name, paths=paths)


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


@app.route('/out/<path:subpath>')
def serve_out(subpath):
    return send_from_directory(OUT_DIR, subpath)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
