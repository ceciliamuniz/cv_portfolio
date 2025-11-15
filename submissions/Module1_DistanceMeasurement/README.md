# Module 1: Real-world Distance Measurement

## Assignment Overview
This application demonstrates real-world distance measurement using **perspective projection mathematics** and **camera calibration**. Students can upload images, click two points, and calculate actual distances in centimeters.

## Demo Video

![Application Demo](https://github.com/ceciliamuniz/cv_portfolio/blob/main/screen_recordings/mod1_rec.gif)

*Live demonstration of the distance measurement application showing image upload, point selection, and real-time distance calculation.*

## Features
- ✨ **Interactive Web Interface** - User-friendly point-and-click measurement
- 📐 **Perspective Projection** - Mathematical conversion from pixels to real-world coordinates  
- 📷 **Pre-calibrated Camera Matrix** - Uses computed intrinsic parameters
- 🎯 **Real-time Calculations** - Instant distance computation in centimeters
- 📊 **Visual Results** - Annotated images with measurement overlays

## Technical Implementation

### Mathematical Foundation
The application uses perspective projection to convert 2D pixel coordinates to 3D real-world coordinates:

```
X_real = (x_pixel - Ox) × Z / Fx
Y_real = (y_pixel - Oy) × Z / Fy
Distance = √(ΔX² + ΔY²)
```

Where:
- `(x_pixel, y_pixel)` = Image coordinates in pixels
- `(Ox, Oy)` = Principal point (camera center)
- `(Fx, Fy)` = Focal lengths in pixels  
- `Z` = Depth (perpendicular distance from camera to object plane)
- `(X_real, Y_real)` = Real-world coordinates

### Camera Calibration Parameters
Pre-computed intrinsic camera matrix:
```
Fx = 640.84 pixels
Fy = 648.80 pixels  
Ox = 294.25 pixels
Oy = 349.31 pixels
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

### Step 3: Access the Web Interface
Open your browser and navigate to: `http://localhost:5001`


## Usage Instructions

1. **Upload Image**: Click "Choose Image File" or drag & drop an image
2. **Set Z-Distance**: Enter the perpendicular distance from camera to objects (in cm)
3. **Select Points**: Click two points on the uploaded image
4. **Measure**: Click "Measure Distance" to calculate real-world distance
5. **View Results**: See the calculated distance and visual annotations

## File Structure
```
Module1_DistanceMeasurement/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── templates/
│   ├── base.html              # Base HTML template
│   └── index.html             # Main application interface
└── static/                    # Static files (auto-created)
```

## Key Components

### Backend (`app.py`)
- **DistanceMeasurementEngine**: Core calculation engine
- **Flask Routes**: API endpoints for calibration and measurement
- **Image Processing**: OpenCV-based point annotation and visualization

### Frontend (`templates/index.html`)
- **Interactive Canvas**: Point selection and image display
- **Real-time Feedback**: Live coordinate updates and validation
- **Bootstrap UI**: Responsive design with professional styling

## API Endpoints

### `GET /api/calibration`
Returns camera intrinsic parameters
```json
{
  "Fx": 640.8396063,
  "Fy": 648.80269311,
  "Ox": 294.24936703,
  "Oy": 349.31369175
}
```

### `POST /api/distance-measurement`
Processes distance measurement request
**Parameters:**
- `image`: Uploaded image file
- `point1_x`, `point1_y`: First point coordinates
- `point2_x`, `point2_y`: Second point coordinates  
- `z_distance`: Camera-to-object distance in cm

**Response:**
```json
{
  "success": true,
  "processed_image": "data:image/jpeg;base64,/9j/4AA...",
  "measurements": {
    "x_distance": 15.42,
    "y_distance": 8.73,
    "euclidean_distance": 17.74,
    "point1_real": [12.34, -5.67],
    "point2_real": [27.76, 3.06]
  },
  "calibration": { ... },
  "z_distance_used": 100
}
```

### `GET /api/test`
Simple connectivity test endpoint

## Assignment Submission Notes

### What This Demonstrates
1. **Camera Calibration Understanding** - Proper use of intrinsic parameters
2. **Perspective Projection Math** - Correct implementation of coordinate transformation
3. **Computer Vision Pipeline** - End-to-end image processing workflow
4. **Web Development Skills** - Professional Flask application with interactive UI
5. **Error Handling** - Robust validation and user feedback

### Academic Requirements Met
- ✅ Real-world distance calculation using camera parameters
- ✅ Interactive point selection interface  
- ✅ Mathematical accuracy in perspective projection
- ✅ Professional documentation and code organization
- ✅ Working web application ready for demonstration

## Troubleshooting

### Common Issues
1. **Port Already in Use**: Change port in `app.py` (line: `app.run(debug=True, port=5001)`)
2. **Module Import Errors**: Ensure all requirements are installed via `pip install -r requirements.txt`
3. **Image Upload Fails**: Check file permissions and ensure `static/uploads` directory exists
4. **Calibration Not Loading**: Verify Flask server is running and API endpoints are accessible

### Testing the Application
1. Use the built-in "Test API Connection" button
2. Try with different image sizes and Z-distance values
3. Test with images containing clear, measurable objects
4. Verify distance calculations make sense relative to Z-distance input

## Development Notes
- Application runs on port **5001** to avoid conflicts with other modules
- Debug mode enabled for development (disable for production)  
- All calculations performed server-side for accuracy
- Client-side validation ensures proper user input

---
**Author**: Cecilia Muniz Siqueira
