import cv2
import numpy as np
import math
import time

# Constants
WEBCAM_INDEX = 0
ROI_Y1, ROI_Y2 = 100, 300
ROI_X1, ROI_X2 = 350, 550
ROI_W = ROI_X2 - ROI_X1
ROI_H = ROI_Y2 - ROI_Y1

# Colors
RECT_COLOR = (0, 255, 0)
HAND_CONTOUR_COLOR = (255, 0, 0)
HULL_COLOR = (0, 255, 0)
DEFECT_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 255)

# Drawing parameters
CONTOUR_THICKNESS = 2
HULL_THICKNESS = 2
TEXT_FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# Processing parameters
LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)
MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS_OPEN = 1
MORPH_ITERATIONS_CLOSE = 1
MIN_CONTOUR_AREA = 1500
MIN_DEFECT_DEPTH = 15.0
MAX_DEFECT_ANGLE = 85.0
FIST_FILL_RATIO_THRESHOLD = 0.85

def create_display_window(title, image, position):
    """Create a named window and move it to the specified position"""
    cv2.namedWindow(title)
    cv2.moveWindow(title, position[0], position[1])
    cv2.imshow(title, image)

def process_frame(frame):
    """Process a single frame and return all intermediate results"""
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    
    # Draw ROI rectangle
    cv2.rectangle(frame_copy, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), RECT_COLOR, 2)
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    
    # Color processing
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    
    # Morphological operations
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS_OPEN)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE)
    
    # Contour processing
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > MIN_CONTOUR_AREA else None
    
    # Create copies for different visualizations
    roi_with_contour = roi.copy()
    roi_with_hull = roi.copy()
    roi_with_defects = roi.copy()
    mask_with_hull_defects = cv2.cvtColor(mask_cleaned.copy(), cv2.COLOR_GRAY2BGR)
    
    gesture = "No Hand"
    finger_count = 0
    
    if max_contour is not None:
        # Draw contour
        cv2.drawContours(roi_with_contour, [max_contour], -1, HAND_CONTOUR_COLOR, CONTOUR_THICKNESS)
        
        # Calculate hull and defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            hull_points = max_contour[hull_indices[:, 0]]
            cv2.drawContours(roi_with_hull, [hull_points], -1, HULL_COLOR, HULL_THICKNESS)
            cv2.drawContours(mask_with_hull_defects, [hull_points], -1, HULL_COLOR, HULL_THICKNESS)
            
            defects = cv2.convexityDefects(max_contour, hull_indices)
            if defects is not None:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    
                    depth = d / 256.0
                    a = math.dist(start, end)
                    b = math.dist(start, far)
                    c = math.dist(end, far)
                    
                    angle = 0
                    if b * c != 0:
                        cos_angle_arg = (b**2 + c**2 - a**2) / (2 * b * c)
                        cos_angle_arg = max(-1.0, min(1.0, cos_angle_arg))
                        try:
                            angle = math.degrees(math.acos(cos_angle_arg))
                        except ValueError:
                            pass
                    
                    if angle < MAX_DEFECT_ANGLE and depth > MIN_DEFECT_DEPTH:
                        count_defects += 1
                        cv2.circle(roi_with_defects, far, 5, DEFECT_COLOR, -1)
                        cv2.circle(mask_with_hull_defects, far, 5, DEFECT_COLOR, -1)
                
                finger_count = count_defects + 1
                
                # Determine gesture
                if count_defects == 0:
                    area_contour = cv2.contourArea(max_contour)
                    area_hull = cv2.contourArea(hull_points)
                    fill_ratio = area_contour / area_hull if area_hull > 0 else 0
                    if fill_ratio > FIST_FILL_RATIO_THRESHOLD:
                        gesture = "Fist"
                        finger_count = 0
                    else:
                        gesture = "Point / One"
                        finger_count = 1
                elif count_defects == 1: gesture = "Victory / Two"
                elif count_defects == 2: gesture = "Three"
                elif count_defects == 3: gesture = "Four"
                elif count_defects == 4: gesture = "Open Palm / Five"
                else: gesture = "Unknown (>5?)"
    
    # Calculate FPS
    current_time = time.time()
    fps = 0
    if hasattr(process_frame, 'prev_time') and process_frame.prev_time > 0:
        fps = 1 / (current_time - process_frame.prev_time)
    process_frame.prev_time = current_time
    
    # Add FPS and gesture text
    cv2.putText(frame_copy, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECT_COLOR, 2)
    cv2.putText(frame_copy, f"Gesture: {gesture} ({finger_count})", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    
    return {
        'frame': frame_copy,
        'roi': roi,
        'hsv': hsv,
        'mask': mask,
        'mask_opened': mask_opened,
        'mask_cleaned': mask_cleaned,
        'roi_with_contour': roi_with_contour,
        'roi_with_hull': roi_with_hull,
        'roi_with_defects': roi_with_defects,
        'mask_with_hull_defects': mask_with_hull_defects
    }

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {WEBCAM_INDEX}")
        return
    
    print("Starting Enhanced Hand Tracking...")
    print("Press 'q' to quit.")
    
    # Window positions
    window_positions = {
        'Original Frame': (50, 50),
        'ROI': (50, 400),
        'HSV': (250, 400),
        'Mask': (450, 400),
        'Opened': (650, 400),
        'Closed': (50, 600),
        'Contour': (250, 600),
        'Hull': (450, 600),
        'Defects': (650, 600)
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        
        results = process_frame(frame)
        
        # Display all windows
        create_display_window('Original Frame', results['frame'], window_positions['Original Frame'])
        create_display_window('ROI', results['roi'], window_positions['ROI'])
        create_display_window('HSV', results['hsv'], window_positions['HSV'])
        create_display_window('Mask', results['mask'], window_positions['Mask'])
        create_display_window('Opened', results['mask_opened'], window_positions['Opened'])
        create_display_window('Closed', results['mask_cleaned'], window_positions['Closed'])
        create_display_window('Contour', results['roi_with_contour'], window_positions['Contour'])
        create_display_window('Hull', results['roi_with_hull'], window_positions['Hull'])
        create_display_window('Defects', results['roi_with_defects'], window_positions['Defects'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 