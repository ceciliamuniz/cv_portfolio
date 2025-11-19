# Module 5 — Real-time Object Tracker

A comprehensive real-time object tracking system with multiple tracking algorithms and an interactive web interface.

## 🎯 Overview

This module implements three different object tracking approaches:
- **Marker-based Tracking**: Uses ArUco markers, QR codes, and AprilTags for robust detection
- **Markerless Tracking**: OpenCV-based object tracking with user-selectable regions of interest
- **SAM2 Segmentation**: Advanced segmentation-based tracking using Meta's Segment Anything Model 2

## ✨ Features

- **Real-time Processing**: Live webcam feed with immediate tracking feedback
- **Interactive Selection**: Click-and-drag interface for selecting objects to track
- **Multiple Algorithms**: Switch between tracking modes dynamically
- **Visual Feedback**: Real-time bounding boxes and tracking status indicators
- **RESTful API**: Backend endpoints for programmatic access

## 🚀 Demo Videos

### Part 1: Marker-based Tracking
![Marker-based Tracking Demo](https://github.com/ceciliamuniz/cv_portfolio/blob/main/screen_recordings/mod5_pt1_rec.gif)
*Demonstrates detection and tracking of ArUco markers in real-time*

### Part 2: Markerless Tracking
![Markerless Tracking Demo](https://github.com/ceciliamuniz/cv_portfolio/blob/main/screen_recordings/mod5_pt2_rec.gif)
*Shows interactive object selection and OpenCV-based tracking*

### Part 3: SAM2 Segmentation
![SAM2 Tracking Demo](https://github.com/ceciliamuniz/cv_portfolio/blob/main/screen_recordings/mod5_pt3_rec.gif)
*Advanced segmentation-based tracking with precise object boundaries*

## 🛠️ Technical Implementation

### Backend Architecture
- **Flask Blueprint**: Modular endpoint organization
- **OpenCV Integration**: Computer vision processing pipeline
- **SAM2 Support**: Optional advanced segmentation capabilities
- **Real-time Streaming**: Efficient frame processing and delivery

### Frontend Features
- **Responsive UI**: Bootstrap-based interface design
- **Canvas Overlay**: Interactive selection and visual feedback
- **Mode Switching**: Dynamic algorithm selection
- **Status Monitoring**: Real-time tracking performance indicators

### API Endpoints
- `POST /module5/api/track/init` - Initialize tracking session
- `POST /module5/api/track` - Process frame and return tracked results
- `POST /module5/api/track/stop` - Stop tracking session
- `POST /module5/api/sam2/upload` - Upload SAM2 mask files

## 📋 Usage

1. **Select Tracking Mode**: Choose from the dropdown menu
2. **For Markerless/SAM2**: Click and drag to select object
3. **Initialize Tracker**: Click "Init Tracker" to begin
4. **Monitor Performance**: View real-time tracking status
5. **Stop When Done**: Use "Stop Tracker" to end session

## 🔧 Dependencies

- OpenCV (cv2) - Computer vision operations
- Flask - Web framework
- NumPy - Numerical computations
- SAM2 (optional) - Advanced segmentation

---
**Author**: Cecilia Muniz Siqueira  