# Module 7 — Stereo Size Estimation & Pose Tracking

## Demo Video
## Part 1: Stereo Size Estimation
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod7_pt1_rec.gif)

## Part 3: Pose Tracking
![Application Demo](https://github.com/ceciliamuniz/cv_portfolio_recordings/blob/main/screen_recordings/mod7_pt3_rec.gif)

A comprehensive computer vision system implementing **calibrated stereo vision** for accurate 3D object measurement and **real-time pose tracking** using MediaPipe. This module demonstrates advanced 3D computer vision techniques with practical applications in measurement and human motion analysis.

## 🎯 **Project Overview**

This module implements two core components:

### **Part 1: Calibrated Stereo Vision System**
- **Real Camera Calibration**: Uses exact intrinsic parameters from Module 1 phone calibration
- **Stereo Rectification**: Proper image rectification using OpenCV stereo pipeline
- **Automatic Object Detection**: Multi-method object detection without requiring masks
- **Accurate 3D Measurement**: Real-world dimension calculation using calibrated parameters
- **Manual Distance Validation**: Optional manual distance input for accuracy verification
- **Multi-Shape Support**: Rectangles, circles, and polygons with appropriate dimension calculations

### **Part 3: Real-time Pose Tracking**
- **MediaPipe Integration**: 33-point body pose estimation with graceful fallback
- **Dual-hand Tracking**: 21 landmarks per hand (42 total hand points)
- **Real-time Processing**: Live webcam feed with comprehensive data recording
- **CSV Export**: Timestamped pose data export for analysis
- **System Diagnostics**: Real-time performance monitoring and status display

## 📋 **Requirements**

### **Stereo Vision Requirements**
- **Stereo Image Pairs**: Left and right images from calibrated stereo setup
- **Image Formats**: JPEG, PNG, BMP, TIFF supported
- **Resolution**: Any resolution (automatic resizing for matching dimensions)
- **Optional**: Stereo calibration files (.npz, .json formats)
- **Optional**: Known object distance for validation

### **Pose Tracking Requirements**
- **Webcam**: Any standard USB or integrated camera
- **MediaPipe**: Optional but recommended for full functionality
- **Browser**: Modern browser with WebRTC support

### **Software Dependencies**
```
Python 3.8+
OpenCV 4.5+ (with contrib modules)
NumPy 1.19+
Flask 2.0+
MediaPipe 0.10+ (optional)
Pandas 1.3+ (for CSV export)
```

## 🚀 **Quick Start**

### **Run from Main Application**
```bash
# From project root
python app.py
# Visit http://localhost:5000/module7
```

### **Standalone Mode**
```bash
# Part 1: Stereo Vision
cd submissions/Module7/part1_stereosize
python app.py
# Visit http://localhost:5001

# Part 3: Pose Tracking  
cd submissions/Module7/part3_pose_tracking
python app.py
# Visit http://localhost:5002
```

## 🔬 **Stereo Vision Technical Details**

### **Calibration System**
**Hardcoded Phone Calibration (from Module 1):**
```python
CAMERA_MATRIX = [
    [640.84,    0,    294.25],
    [   0,   648.80, 349.31], 
    [   0,      0,       1  ]
]
```

**Key Parameters:**
- **fx = 640.8396063** pixels (horizontal focal length)
- **fy = 648.80269311** pixels (vertical focal length)
- **cx = 294.24936703** pixels (principal point x)
- **cy = 349.31369175** pixels (principal point y)
- **baseline = 65.0** mm (stereo camera separation)

### **Processing Pipeline**

1. **Image Preprocessing**
   - Automatic size validation and resizing
   - Stereo rectification using calibration parameters
   - Lens distortion correction

2. **Object Detection**
   - **Edge Detection**: Canny edge detector with adaptive thresholds
   - **Adaptive Thresholding**: Handles varying lighting conditions
   - **Contour Analysis**: Identifies closed object boundaries
   - **Size Filtering**: Only processes objects >1000 pixels

3. **Stereo Computation**
   - **Disparity Calculation**: Enhanced StereoBM with optimized parameters
   - **Depth Estimation**: `Z = (fx × baseline) / disparity`
   - **Validation**: Clips unrealistic depths (30cm to 10m)

4. **3D Measurement**
   - **Size Calculation**: `Real_size = (pixel_distance × depth) / focal_length`
   - **Shape Classification**: Rectangles, circles, polygons
   - **Multi-object Support**: Measures all detected objects

### **Measurement Modes**

#### **Automatic Stereo Mode** (Default)
- Uses stereo disparity for depth calculation
- Shows: `"depth: 1250.3 mm (stereo calculated)"`

#### **Manual Validation Mode**
- User provides known distance for comparison
- Shows both stereo and manual calculations
- Displays accuracy percentage and error analysis

## 🎬 **Pose Tracking Technical Details**

### **MediaPipe Integration**
```python
class PoseTracker:
    def __init__(self, mode='holistic'):
        # holistic: Full body + hands
        # pose: Body only (33 landmarks)
        # hands: Hands only (42 landmarks)
```

### **Landmark Detection**
- **Body Pose**: 33 key points (head, torso, arms, legs)
- **Left Hand**: 21 landmarks (finger joints, palm)
- **Right Hand**: 21 landmarks (finger joints, palm)
- **Total**: Up to 75 simultaneous tracking points

### **Data Export Format**
```csv
timestamp,landmark_type,landmark_id,x,y,z,visibility
2025-11-27_15:30:45.123,pose,0,0.5234,0.3456,0.0012,0.9876
2025-11-27_15:30:45.123,left_hand,4,0.4567,0.2345,0.0034,0.8765
```

## 📊 **Usage Examples**

### **Stereo Measurement**
1. **Upload Images**: Select left and right stereo images
2. **Optional**: Upload calibration file or enter known distance
3. **Process**: System automatically detects and measures objects
4. **Results**: View measurements with accuracy information

### **Pose Tracking**
1. **Start Session**: Click "Start Recording"
2. **Perform Actions**: Move in front of camera
3. **Monitor Status**: View real-time landmark detection
4. **Export Data**: Download CSV with timestamped pose data

## 🔧 **API Reference**

### **Stereo Vision Endpoints**

#### `POST /module7/api/stereo/estimate`
**Parameters:**
- `left`: Left stereo image file
- `right`: Right stereo image file 
- `calibration`: Optional calibration file (.npz/.json)
- `manual_distance`: Optional known distance (mm)

**Response:**
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
      }
    }
  ],
  "processing_info": {
    "measurement_mode": "stereo",
    "calibration_used": {
      "source": "default",
      "focal_length_x": 640.84,
      "baseline": 65.0
    }
  }
}
```

### **Pose Tracking Endpoints**

#### `POST /module7/pose/api/start_recording`
#### `POST /module7/pose/api/stop_recording` 
#### `POST /module7/pose/api/export_csv`
#### `GET /module7/pose/api/video_feed` - Real-time video stream

## 🎯 **Best Practices**

### **Stereo Image Capture**
1. **Baseline**: Use consistent camera separation (6-10cm)
2. **Alignment**: Keep cameras parallel and level
3. **Lighting**: Ensure consistent illumination
4. **Objects**: Place objects with sufficient texture and contrast
5. **Distance**: Keep objects 30cm to 5m from cameras

### **Pose Tracking**
1. **Lighting**: Ensure good, even lighting on subject
2. **Background**: Use contrasting background for better detection
3. **Distance**: Stay 1-3 meters from camera for optimal tracking
4. **Clothing**: Avoid loose clothing that might obscure landmarks

## 🔍 **Validation & Accuracy**

### **Measurement Accuracy Factors**
- **Calibration Quality**: Uses precise phone calibration from Module 1
- **Baseline Accuracy**: 65mm baseline assumption (adjustable)
- **Disparity Precision**: Sub-pixel stereo matching
- **Object Detection**: Multi-method approach for robustness

### **Expected Accuracy**
- **Distance**: ±5-10% at 0.5-2m range
- **Size**: ±2-5% for objects >5cm
- **Pose**: MediaPipe accuracy (typically >95% landmark detection)

## 🐛 **Troubleshooting**

### **Stereo Vision Issues**
**"Image size mismatch"** → Fixed automatically with resizing  
**"No objects detected"** → Increase lighting, improve object contrast  
**"Unrealistic measurements"** → Check baseline calibration, verify image alignment  
**"High depth error %"** → Improve stereo camera alignment, check calibration

### **Pose Tracking Issues**
**"MediaPipe not available"** → Install: `pip install mediapipe`  
**"No landmarks detected"** → Improve lighting, check camera permissions  
**"Choppy video feed"** → Reduce resolution, close other applications  

## 🚧 **Limitations & Future Improvements**

### **Current Limitations**
- **Stereo**: Assumes planar objects, limited baseline flexibility
- **Pose**: Requires MediaPipe installation for full functionality
- **Processing**: Real-time constraints on complex calculations

### **Future Enhancements**
- **Stereo**: Bundle adjustment, multi-view geometry
- **Pose**: 3D pose estimation, action recognition
- **Integration**: Combined stereo-pose analysis for spatial understanding

---

## 📝 **Assignment Compliance**

✅ **Part 1 - Stereo Size Estimation**: Calibrated stereo vision with real camera parameters  
✅ **Part 3 - Pose & Hand Tracking**: MediaPipe integration with comprehensive data export  
✅ **Real Camera Calibration**: Uses exact Module 1 phone calibration matrix  
✅ **Automatic Object Detection**: No manual mask input required  
✅ **Validation Framework**: Manual distance comparison for accuracy verification  
✅ **Real-time Performance**: Live pose tracking with CSV data export  

---
**Author**: Cecilia Muniz Siqueira