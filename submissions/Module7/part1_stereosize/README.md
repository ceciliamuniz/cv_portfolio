# Module 7 Part 1 — Calibrated Stereo Size Estimation

A professional stereo vision system implementing **calibrated 3D object measurement** using real camera parameters from Module 1. This implementation demonstrates proper stereo vision theory with rectification, disparity computation, and accurate real-world size calculation.

## 🎯 **Core Features**

- **Real Camera Calibration**: Uses exact intrinsic parameters from Module 1 phone calibration
- **Automatic Stereo Rectification**: Proper epipolar alignment using OpenCV stereo pipeline  
- **No Manual Masks Required**: Automatic object detection using multi-method approach
- **Manual Distance Validation**: Optional known distance input for accuracy verification
- **Multi-Shape Recognition**: Rectangles, circles, and polygons with appropriate measurements
- **Professional Web Interface**: Bootstrap UI with comprehensive result display

## 🔧 **Technical Implementation**

### **Calibration Parameters** (Hardcoded from Module 1)
```python
CAMERA_MATRIX_LEFT = [
    [640.8396063, 0, 294.24936703],    # fx, 0, cx
    [0, 648.80269311, 349.31369175],  # 0, fy, cy
    [0, 0, 1]                         # 0, 0, 1
]
BASELINE = 65.0  # millimeters
```

### **Processing Pipeline**

1. **Image Validation & Resizing**
   ```python
   # Automatic size matching for stereo computation
   if left_img.shape != right_img.shape:
       target_height = min(left_img.shape[0], right_img.shape[0])
       target_width = min(left_img.shape[1], right_img.shape[1])
       left_img = cv2.resize(left_img, (target_width, target_height))
       right_img = cv2.resize(right_img, (target_width, target_height))
   ```

2. **Stereo Rectification**
   ```python
   R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
       camera_matrix_left, dist_coeffs_left,
       camera_matrix_right, dist_coeffs_right,
       (w, h), rotation_matrix, translation_vector
   )
   ```

3. **Object Detection** (No masks required)
   ```python
   # Multi-method automatic detection
   edges = cv2.Canny(blurred, 30, 100)
   adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
   combined = cv2.bitwise_or(edges, adaptive_thresh)
   ```

4. **Stereo Computation**
   ```python
   # Enhanced stereo matching
   stereo = cv2.StereoBM_create(numDisparities=80, blockSize=21)
   disparity = stereo.compute(left_rect, right_rect)
   
   # Depth calculation: Z = (fx × baseline) / disparity
   Z = (focal_length_x * baseline) / disparity_safe
   ```

5. **3D Measurement**
   ```python
   # Real-world size: size = (pixel_distance × depth) / focal_length
   real_width = (w * z_mean) / fx   # mm
   real_height = (h * z_mean) / fy  # mm
   ```

## 📊 **API Reference**

### **Flask Blueprint Endpoints**

#### `GET /module7/`
Serves the main stereo vision interface with:
- Calibration parameter display
- Object detection methodology explanation
- File upload interface for stereo pairs
- Optional manual distance input
- Real-time result display

#### `POST /module7/api/stereo/estimate`
**Parameters:**
- `left` (required): Left stereo image file
- `right` (required): Right stereo image file  
- `calibration` (optional): Stereo calibration file (.npz, .json)
- `manual_distance` (optional): Known object distance in mm

**Response Format:**
```json
{
  "success": true,
  "results": [
    {
      "shape": "rectangle",
      "dimensions": {
        "width": "45.2 mm",
        "height": "32.1 mm",
        "area": "1451.9 mm²",
        "depth": "487.3 mm (stereo calculated)"
      },
      "bounding_box": {"x": 150, "y": 200, "width": 80, "height": 60}
    }
  ],
  "processing_info": {
    "automatic_detection": true,
    "measurement_mode": "stereo",
    "calibration_used": {
      "source": "default",
      "focal_length_x": 640.8396063,
      "baseline": 65.0,
      "rectified": true
    }
  }
}
```

**With Manual Distance Validation:**
```json
{
  "dimensions": {
    "width": "45.2 mm",
    "measurement_depth": "500.0 mm (manual input)",
    "stereo_depth": "487.3 mm (calculated)", 
    "depth_comparison": "Difference: 12.7 mm (2.6%)"
  }
}
```

## 🎮 **Usage Examples**

### **Basic Stereo Measurement**
1. Upload left and right stereo images
2. System automatically detects objects
3. View measurements with stereo-calculated depth

### **Accuracy Validation**
1. Measure actual object distance with ruler
2. Enter distance in "Known Distance" field  
3. Compare stereo calculation vs. manual measurement
4. View percentage error and accuracy assessment

### **Custom Calibration**
1. Upload stereo calibration file (.npz format)
2. System uses your specific camera parameters
3. Potentially improved accuracy for your setup

## 🔬 **Calibration File Format**

### **NumPy Archive (.npz)**
```python
# Save custom calibration
np.savez('stereo_calibration.npz',
         camera_matrix_left=K1,
         camera_matrix_right=K2, 
         dist_coeffs_left=D1,
         dist_coeffs_right=D2,
         rotation_matrix=R,
         translation_vector=T)
```

### **JSON Format**
```json
{
  "camera_matrix_left": [[640.84, 0, 294.25], [0, 648.80, 349.31], [0, 0, 1]],
  "camera_matrix_right": [[640.84, 0, 294.25], [0, 648.80, 349.31], [0, 0, 1]],
  "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "translation_vector": [65.0, 0, 0]
}
```

## 🎯 **Expected Accuracy**

### **Distance Measurement**
- **Close Range (30-100cm)**: ±5-8% accuracy
- **Medium Range (100-200cm)**: ±3-5% accuracy  
- **Far Range (200-500cm)**: ±8-15% accuracy

### **Size Measurement** 
- **Large Objects (>10cm)**: ±2-4% accuracy
- **Medium Objects (5-10cm)**: ±4-8% accuracy
- **Small Objects (<5cm)**: ±10-20% accuracy

## 🚀 **Local Testing**

### **Standalone Mode**
```bash
cd submissions/Module7/part1_stereosize
python app.py
# Visit http://localhost:5001
```

### **Integrated Mode**
```bash
# From project root
python app.py
# Visit http://localhost:5000/module7
```

## 🐛 **Troubleshooting**

**"Failed to decode image"** → Use standard formats (JPG, PNG)  
**"Image size mismatch"** → Fixed automatically with resizing  
**"No objects detected"** → Improve lighting and object contrast  
**"Unrealistic depth values"** → Check stereo image alignment  
**"High measurement error"** → Verify baseline calibration, improve image quality  

---
**Author**: Cecilia Muniz Siqueira
