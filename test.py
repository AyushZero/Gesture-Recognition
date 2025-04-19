import cv2
import numpy as np
import math
import time

WEBCAM_INDEX = 0

ROI_Y1, ROI_Y2 = 100, 300
ROI_X1, ROI_X2 = 350, 550
RECT_COLOR = (0, 255, 0)
HAND_CONTOUR_COLOR = (255, 0, 0)
HULL_COLOR = (0, 255, 0)
DEFECT_COLOR = (0, 0, 255)

LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS_OPEN = 1
MORPH_ITERATIONS_CLOSE = 1

MIN_CONTOUR_AREA = 1500

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}")
    exit()

print("Starting Step-by-Step Hand Processing Visualization...")
print("Press 'n' for next step, 'q' to quit.")

process_step = 0
frame = None
roi = None
hsv = None
mask = None
mask_opened = None
mask_cleaned = None
contours = None
max_contour = None
hull_indices = None
hull_points = None
defects = None
max_steps = 7  # Increased total steps
step_printed = [False] * (max_steps + 1)
windows_created = {}

def create_window(window_name, image):
    cv2.imshow(window_name, image)
    windows_created[window_name] = True

def destroy_window_if_exists(window_name, step):
    if window_name in windows_created and process_step < step:
        cv2.destroyWindow(window_name)
        del windows_created[window_name]
        step_printed[step] = False

while True:
    ret, current_frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    current_frame = cv2.flip(current_frame, 1)
    frame_copy = current_frame.copy()
    cv2.rectangle(frame_copy, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), RECT_COLOR, 2)
    current_roi = current_frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    if process_step >= 1:
        hsv = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
    if process_step >= 2:
        mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    if process_step >= 3:
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS_OPEN)
    if process_step >= 4:
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE)
    if process_step >= 5 and mask_cleaned is not None:
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
                cv2.drawContours(current_roi, [max_contour], -1, HAND_CONTOUR_COLOR, 2)
    if process_step >= 6 and max_contour is not None:
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        if hull_indices is not None:
            hull_points = max_contour[hull_indices[:, 0]]
            cv2.drawContours(current_roi, [hull_points], -1, HULL_COLOR, 2)
    if process_step >= 7 and max_contour is not None and hull_indices is not None and len(hull_indices) > 3:
        defects = cv2.convexityDefects(max_contour, hull_indices)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = tuple(max_contour[f][0])
                cv2.circle(current_roi, far, 5, DEFECT_COLOR, -1)

    # Display
    cv2.imshow("Original Frame", frame_copy)

    if process_step >= 1 and hsv is not None:
        create_window("Step 1: HSV Conversion (ROI)", hsv)
        if process_step == 1 and not step_printed[1]:
            print("Step 1: Converted ROI to HSV color space.")
            step_printed[1] = True
    destroy_window_if_exists("Step 1: HSV Conversion (ROI)", 1)

    if process_step >= 2 and mask is not None:
        create_window("Step 2: Color Filtered Mask", mask)
        if process_step == 2 and not step_printed[2]:
            print("Step 2: Applied color filtering using cv2.inRange() to create a binary mask.")
            step_printed[2] = True
    destroy_window_if_exists("Step 2: Color Filtered Mask", 2)

    if process_step >= 3 and mask_opened is not None:
        create_window("Step 3: Mask After Opening", mask_opened)
        if process_step == 3 and not step_printed[3]:
            print("Step 3: Applied morphological 'opening' (erosion followed by dilation) to the mask.")
            step_printed[3] = True
    destroy_window_if_exists("Step 3: Mask After Opening", 3)

    if process_step >= 4 and mask_cleaned is not None:
        create_window("Step 4: Mask After Closing", mask_cleaned)
        if process_step == 4 and not step_printed[4]:
            print("Step 4: Applied morphological 'closing' (dilation followed by erosion) to the opened mask.")
            step_printed[4] = True
    destroy_window_if_exists("Step 4: Mask After Closing", 4)

    if process_step >= 5 and max_contour is not None:
        contour_display = current_roi.copy()
        cv2.drawContours(contour_display, [max_contour], -1, HAND_CONTOUR_COLOR, 2)
        create_window("Step 5: Largest Hand Contour", contour_display)
        if process_step == 5 and not step_printed[5]:
            print("Step 5: Found and drew the largest hand contour.")
            step_printed[5] = True
    destroy_window_if_exists("Step 5: Largest Hand Contour", 5)

    if process_step >= 6 and hull_points is not None:
        hull_display = current_roi.copy()
        cv2.drawContours(hull_display, [hull_points], -1, HULL_COLOR, 2)
        create_window("Step 6: Convex Hull", hull_display)
        if process_step == 6 and not step_printed[6]:
            print("Step 6: Calculated and drew the convex hull of the hand contour.")
            step_printed[6] = True
    destroy_window_if_exists("Step 6: Convex Hull", 6)

    if process_step >= 7 and defects is not None:
        defects_display = current_roi.copy()
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = tuple(max_contour[f][0])
                cv2.circle(defects_display, far, 5, DEFECT_COLOR, -1)
        create_window("Step 7: Convexity Defects", defects_display)
        if process_step == 7 and not step_printed[7]:
            print("Step 7: Detected and visualized convexity defects (potential finger valleys).")
            step_printed[7] = True
    destroy_window_if_exists("Step 7: Convexity Defects", 7)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        if process_step < max_steps:
            process_step += 1
            print(f"Showing step: {process_step}")
        else:
            print("Reached the final visualization step.")
    elif key == ord('q'):
        break

print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()