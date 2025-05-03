# Hand Tracking Project

This project implements hand gesture recognition using computer vision techniques. It can detect various hand gestures including fist, pointing, victory sign, and open palm.

## Files

- `main.py`: The main application with an enhanced GUI interface using Tkinter. Shows all processing steps in a single window.
- `test.py`: Original version of the hand tracking implementation (backup).
- `filters.py`: Contains additional image processing filters and utilities.
- `all-filters.py`: Demonstrates various image processing filters for hand detection.
- `requirements.txt`: Lists all required Python packages.

## Features

- Real-time hand gesture recognition
- Multiple visualization windows showing different processing steps
- FPS counter and gesture status display
- Region of Interest (ROI) for focused hand tracking
- Support for various gestures:
  - Fist
  - Pointing/One finger
  - Victory/Two fingers
  - Three fingers
  - Four fingers
  - Open palm/Five fingers

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pillow (for GUI)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

The application will open a single window showing:
- Original camera feed with ROI
- Region of Interest (ROI) view
- HSV color space
- Binary mask
- Opened mask
- Closed mask
- Contour visualization
- Convex hull
- Defect points

The status bar at the bottom shows:
- Current FPS
- Detected gesture and finger count

## Controls

- Close the window to exit the application

## Notes

- The ROI (Region of Interest) is set to a 200x200 pixel area in the center of the frame
- Skin color detection uses HSV color space
- The application uses morphological operations to clean up the hand mask
- Convex hull and defect analysis are used for finger counting

## Academic Details

### Image Processing Techniques

#### 1. Color Space Transformation
- RGB to HSV conversion for better skin color detection
- HSV ranges for skin detection:
  - Hue: [0, 20] (captures skin tones)
  - Saturation: [30, 150] (excludes low saturation values)
  - Value: [60, 255] (excludes very dark regions)

#### 2. Morphological Operations
- Opening operation (erosion followed by dilation):
  - Kernel size: 5×5
  - Purpose: Removes small noise and gaps
  - Formula: \( A \circ B = (A \ominus B) \oplus B \)
- Closing operation (dilation followed by erosion):
  - Kernel size: 5×5
  - Purpose: Fills small holes and connects nearby components
  - Formula: \( A \bullet B = (A \oplus B) \ominus B \)

#### 3. Contour Analysis
- Contour detection using OpenCV's findContours
- Area threshold: 1500 pixels
- Convex Hull algorithm:
  - Graham scan implementation
  - Time complexity: O(n log n)
  - Purpose: Creates the smallest convex polygon containing all points

#### 4. Defect Analysis
- Convexity defects detection
- Angle calculation using cosine law:
  \[ \cos(\theta) = \frac{b^2 + c^2 - a^2}{2bc} \]
  where:
  - a = distance between start and end points
  - b = distance between start and far points
  - c = distance between end and far points
- Defect filtering:
  - Maximum angle: 85°
  - Minimum depth: 15.0 units

#### 5. Gesture Classification
- Fist detection using fill ratio:
  \[ \text{fill\_ratio} = \frac{\text{contour\_area}}{\text{hull\_area}} \]
  - Threshold: 0.85
- Finger counting based on defect analysis:
  - Number of fingers = Number of valid defects + 1
  - Valid defects: angle < 85° and depth > 15.0

### Libraries and Tools

#### OpenCV (cv2)
- Core image processing operations
- Contour detection and analysis
- Convex hull computation
- Real-time video capture and processing

#### NumPy
- Efficient array operations
- Mathematical computations
- Matrix manipulations for image processing

#### Tkinter and PIL
- GUI implementation
- Real-time image display
- Window management
- Image format conversion

### Performance Considerations
- Frame processing rate: 30+ FPS
- ROI size: 200×200 pixels
- Memory efficient processing using NumPy arrays
- Real-time visualization with minimal latency

### Future Improvements
- Machine learning integration for better gesture recognition
- Multi-hand tracking capability
- 3D hand pose estimation
- Gesture sequence recognition
