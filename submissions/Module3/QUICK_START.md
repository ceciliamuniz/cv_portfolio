# Module 3 Parts 4 & 5: Quick Reference Card

## 🚀 Quick Start (30 Minutes)

### Step 1: Print Markers (5 min)
```bash
# Markers already generated! Location:
# CV_Module2/submissions/Module3/part4_aruco_segmentation/aruco_markers/

# Print aruco_marker_00.png through aruco_marker_10.png
# Use white paper, normal quality
```

### Step 2: Setup Object (5 min)
- Choose NON-RECTANGULAR object (vase, bottle, curved item)
- Stick 6-8 markers around the boundary
- Space evenly
- Ensure markers are flat and visible

### Step 3: Capture Images (10 min)
Capture 10+ images with variety:
- **3 close-up** (fill frame with object)
- **4 medium range** (object + some background)
- **3 far** (object in context)
- **Mix angles**: front, side, top, oblique

Save to: `part4_aruco_segmentation/images/`

### Step 4: Process (5 min)
```bash
cd C:\Users\cecil\OneDrive\CompVision\CV_Module2\submissions\Module3\part4_aruco_segmentation
python aruco_segmentation.py
```

### Step 5: View Results (5 min)
Check: `outputs/convex_hull/` and `outputs/contour/`
- `*_segmentation.jpg` - Visualization
- `*_mask.png` - Binary mask
- `processing_summary.json` - Metrics

---

## 📁 File Locations

```
📂 Module3/
├── 📂 part4_aruco_segmentation/
│   ├── 📄 aruco_segmentation.py       ← Run this
│   ├── 📂 aruco_markers/              ← Print these
│   ├── 📂 images/                     ← Add your photos here
│   └── 📂 outputs/                    ← Results appear here
│
├── 📂 part5_sam2_comparison/          ← Optional (needs SAM2)
│   ├── 📄 sam2_comparison.py
│   └── 📂 checkpoints/                ← Download SAM2 weights
│
└── 📂 web_integration/                ← For CV Portfolio
    ├── 📄 routes.py                   ← Flask Blueprint
    └── 📂 templates/                  ← HTML pages
```

---

## 🎯 Requirements Checklist

### Part 4: ArUco Segmentation
- [ ] Non-rectangular object chosen
- [ ] 6-8 ArUco markers printed
- [ ] Markers attached to object boundary
- [ ] 10+ images captured (various distances/angles)
- [ ] Images saved to `images/` folder
- [ ] Processed with `aruco_segmentation.py`
- [ ] Results reviewed in `outputs/`

### Part 5: SAM2 Comparison (Optional)
- [ ] PyTorch installed
- [ ] SAM2 package installed
- [ ] SAM2 checkpoint downloaded
- [ ] Part 4 completed first
- [ ] Processed with `sam2_comparison.py`
- [ ] Comparison results reviewed

### Web Integration (For Portfolio)
- [ ] Read `INTEGRATION_GUIDE.md`
- [ ] Blueprint added to main app
- [ ] Navigation updated
- [ ] Routes tested
- [ ] Results accessible via browser

---

## 🛠️ Commands Cheat Sheet

```bash
# Navigate to Part 4
cd C:\Users\cecil\OneDrive\CompVision\CV_Module2\submissions\Module3\part4_aruco_segmentation

# Generate markers (already done!)
python aruco_segmentation.py

# Process your images
python aruco_segmentation.py

# Navigate to Part 5
cd ..\part5_sam2_comparison

# Run SAM2 comparison (if installed)
python sam2_comparison.py

# Check results
explorer outputs                      # Part 4
explorer comparison_results          # Part 5
```

---

## 📊 Expected Output

### For Each Image:
**Part 4 Files**:
- `imagename_segmentation.jpg` - Visual result with markers
- `imagename_mask.png` - Binary mask (white=object)

**Part 5 Files** (if SAM2 used):
- `imagename_comparison.jpg` - 4-panel comparison
- `imagename_sam2_mask.png` - SAM2 segmentation

### Summary Files:
- `processing_summary.json` - All metrics
- `comparison_summary.json` - SAM2 vs ArUco metrics

---

## 🎨 Marker Setup Tips

### Good Practices ✅
- Use 6-8 markers (sweet spot)
- Even spacing around boundary
- Flat against surface
- Good lighting
- High contrast (white background)

### Avoid ❌
- Too few markers (<4)
- Uneven distribution
- Wrinkled or bent markers
- Shadows covering markers
- Blurry photos

---

## 📸 Image Capture Strategy

### Distance Distribution
```
Close-up (3 images)    [====    ]  30%
Medium (4 images)      [======  ]  40%  
Far (3 images)         [====    ]  30%
```

### Angle Distribution
```
Front view (4 images)  [======  ]  40%
Side views (3 images)  [====    ]  30%
Top/Oblique (3 images) [====    ]  30%
```

### Quality Checklist
- [ ] All markers visible
- [ ] Sharp focus (not blurry)
- [ ] Good lighting
- [ ] Object fills frame appropriately
- [ ] No shadows on markers
- [ ] Varied perspectives

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No markers detected | Check dictionary type (DICT_4X4_50), improve lighting |
| Only some markers detected | Add more markers, improve visibility |
| Poor segmentation | Try contour method instead of convex_hull |
| Segmentation too large | Background included, adjust method |
| Segmentation too small | Add more boundary markers |
| SAM2 import error | Install: `pip install torch` and SAM2 package |
| Out of memory (SAM2) | Use smaller model or resize images |

---

## 📈 Quality Metrics

### Good Results
- ✅ 90-100% markers detected per image
- ✅ Clean boundary following object shape
- ✅ Consistent across different views
- ✅ IoU > 0.7 (if comparing with SAM2)

### Poor Results
- ❌ <70% markers detected
- ❌ Jagged or incomplete boundaries
- ❌ Large variation between views
- ❌ IoU < 0.5

---

## 🌐 Web Integration (Optional)

### Add to Main App
In `CV_Module2/submissions/Module2_TemplateMatching_BlurRecovery/app.py`:

```python
# Add at top
from pathlib import Path
import sys
module3_path = Path(__file__).parent.parent / 'Module3' / 'web_integration'
sys.path.insert(0, str(module3_path))
from routes import module3_bp

# Add after app creation
app.register_blueprint(module3_bp)
```

### Test URLs
- http://localhost:5000/module3
- http://localhost:5000/module3/part4-aruco
- http://localhost:5000/module3/part5-sam2-comparison
- http://localhost:5000/module3/gallery

---

## 🎓 Key Concepts

### ArUco Markers
- Fiducial markers for pose estimation
- Dictionary: predefined pattern set
- Detection: Find corners → ID decode
- Applications: AR, robotics, calibration

### Segmentation Methods
- **Convex Hull**: Smallest convex shape containing points
- **Contour**: Morphology-based boundary detection  
- **Alpha Shape**: Concave hull with parameter α

### SAM2 (Segment Anything)
- Foundation model for segmentation
- Prompt-based (points, boxes, masks)
- Zero-shot generalization
- Trained on 11M images

### Metrics
- **IoU**: Overlap measure (0-1)
- **Dice**: Similar to IoU, more balanced
- **Precision**: How accurate is the prediction?
- **Recall**: How complete is the prediction?

---

## 💡 Pro Tips

1. **Marker Placement**: Imagine you're outlining the object with a marker pen - that's where they should go

2. **Camera Settings**: Use portrait mode or macro for close-ups, ensure markers stay sharp

3. **Lighting**: Soft, diffuse lighting is best - avoid harsh shadows

4. **Object Choice**: Organic shapes work great (fruits, bottles, vases)

5. **Processing Time**: ArUco is instant, SAM2 can take 5-15s per image on CPU

6. **Batch Processing**: Process all images at once - the script handles it!

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `PARTS_4_5_SUMMARY.md` | This summary (you are here) |
| `INTEGRATION_GUIDE.md` | Full integration instructions |
| `part4_aruco_segmentation/PART4_README.md` | Part 4 details |
| `part5_sam2_comparison/README.md` | Part 5 + SAM2 info |

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Print markers | 5 min |
| Setup object | 5 min |
| Capture images | 10-15 min |
| Process Part 4 | 1-2 min |
| Review results | 5 min |
| Install SAM2 | 10-15 min |
| Process Part 5 | 2-5 min |
| Web integration | 10-15 min |
| **TOTAL** | **~60 min** |

---

## 🎉 Success Indicators

You're done when you have:
✅ 10+ processed images in `outputs/`
✅ All images showing detected markers
✅ Clean segmentation boundaries
✅ Processing summary JSON
✅ (Optional) SAM2 comparison results
✅ (Optional) Web interface integrated

---

**Ready to start? Print those markers and capture some images! 📸**

See `PARTS_4_5_SUMMARY.md` for detailed information.
See `INTEGRATION_GUIDE.md` for web integration steps.
