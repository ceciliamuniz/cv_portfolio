# Module 7 Part 3: Real-time Pose Estimation and Hand Tracking

This implementation provides comprehensive pose estimation and hand tracking using MediaPipe with real-time processing and CSV data export capabilities.

## Features

### Real-time Processing
- Live webcam feed with pose and hand tracking overlay
- Multiple processing modes:
  - **Holistic**: Full body pose + hands + face landmarks
  - **Pose Only**: Body pose estimation (33 landmarks)
  - **Hands Only**: Hand tracking (up to 2 hands, 21 landmarks each)

### Data Recording & Export
- Start/stop recording of pose data during live feed
- Automatic timestamping of all recorded frames
- CSV export with comprehensive landmark data
- Downloadable files with session timestamps

### Static Image Analysis
- Upload and analyze individual images
- Visual overlay showing detected landmarks
- Instant pose data extraction

### System Monitoring
- MediaPipe installation verification
- Camera availability checking
- Real-time system status monitoring

## CSV Data Format

The exported CSV contains the following columns for each frame:

### Basic Information
- `timestamp`: ISO format timestamp of frame capture
- `frame_number`: Sequential frame number in recording session
- `mode`: Processing mode used (holistic, pose_only, hands_only)

### Pose Landmarks (33 points)
- `pose_{0-32}_x`: Horizontal position (0.0 = left, 1.0 = right)
- `pose_{0-32}_y`: Vertical position (0.0 = top, 1.0 = bottom) 
- `pose_{0-32}_z`: Depth value (smaller = closer to camera)
- `pose_{0-32}_visibility`: Landmark visibility score (0.0-1.0)

### Hand Landmarks (21 points per hand)
- `left_hand_{0-20}_x/y/z`: Left hand landmark coordinates
- `right_hand_{0-20}_x/y/z`: Right hand landmark coordinates

## Landmark Definitions

### Pose Landmarks (0-32)
0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer
4: right_eye_inner, 5: right_eye, 6: right_eye_outer, 7: left_ear
8: right_ear, 9: mouth_left, 10: mouth_right, 11: left_shoulder
12: right_shoulder, 13: left_elbow, 14: right_elbow, 15: left_wrist
16: right_wrist, 17: left_pinky, 18: right_pinky, 19: left_index
20: right_index, 21: left_thumb, 22: right_thumb, 23: left_hip
24: right_hip, 25: left_knee, 26: right_knee, 27: left_ankle
28: right_ankle, 29: left_heel, 30: right_heel, 31: left_foot_index
32: right_foot_index

### Hand Landmarks (0-20)
0: wrist, 1: thumb_cmc, 2: thumb_mcp, 3: thumb_ip, 4: thumb_tip
5: index_finger_mcp, 6: index_finger_pip, 7: index_finger_dip, 8: index_finger_tip
9: middle_finger_mcp, 10: middle_finger_pip, 11: middle_finger_dip, 12: middle_finger_tip
13: ring_finger_mcp, 14: ring_finger_pip, 15: ring_finger_dip, 16: ring_finger_tip
17: pinky_mcp, 18: pinky_pip, 19: pinky_dip, 20: pinky_tip

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a working webcam connected

3. The blueprint is automatically registered with the main Flask app at `/module7/pose`

## Usage

1. Navigate to `/module7/pose` in your web browser
2. Grant camera permissions when prompted
3. Use the control panel to:
   - Start/stop recording pose data
   - Export recorded data to CSV
   - Test system components
   - Upload images for static analysis

## API Endpoints

- `GET /module7/pose/` - Main interface
- `POST /module7/pose/api/process_frame` - Process uploaded image
- `POST /module7/pose/api/start_recording` - Start data recording
- `POST /module7/pose/api/stop_recording` - Stop data recording
- `POST /module7/pose/api/export_csv` - Export recorded data
- `GET /module7/pose/api/video_feed` - Real-time video stream
- `GET /module7/pose/api/landmark_info` - Landmark documentation
- `GET /module7/pose/api/test` - System status check

## Technical Implementation

- **MediaPipe**: Google's ML framework for pose/hand detection
- **OpenCV**: Computer vision operations and video processing
- **Flask**: Web framework with blueprint architecture
- **Pandas**: Data manipulation and CSV export
- **Real-time Processing**: Threaded video capture with landmark extraction
- **Data Export**: Structured CSV format with comprehensive metadata

---
**Author**: Cecilia Muniz Siqueira  
**Module**: 7 Part 3 - Real-time Pose Estimation and Hand Tracking
