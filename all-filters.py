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
TEXT_COLOR = (0, 0, 255)
TEXT_FONT_SCALE = 0.6  # Increased font scale
TEXT_THICKNESS = 1

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

print("Starting Simultaneous Step-by-Step Hand Processing Visualization...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), RECT_COLOR, 2)
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS_OPEN)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > MIN_CONTOUR_AREA else None
    hull_indices = cv2.convexHull(max_contour, returnPoints=False) if max_contour is not None and len(max_contour) > 3 else None

    roi_with_hull = roi.copy()
    roi_with_defects = roi.copy()
    roi_with_contour = roi.copy()
    mask_with_hull_defects = cv2.cvtColor(mask_cleaned.copy(), cv2.COLOR_GRAY2BGR)

    if max_contour is not None:
        cv2.drawContours(roi_with_contour, [max_contour], -1, HAND_CONTOUR_COLOR, 3) # Increased thickness

    if hull_indices is not None and max_contour is not None:
        hull_pts = max_contour[hull_indices[:, 0]]
        cv2.drawContours(roi_with_hull, [hull_pts], -1, HULL_COLOR, 3) # Increased thickness
        cv2.drawContours(mask_with_hull_defects, [hull_pts], -1, HULL_COLOR, 3) # Increased thickness

        defects = cv2.convexityDefects(max_contour, hull_indices)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                far = tuple(max_contour[f][0])
                cv2.circle(roi_with_defects, far, 5, DEFECT_COLOR, -1) # Increased radius
                cv2.circle(mask_with_hull_defects, far, 5, DEFECT_COLOR, -1) # Increased radius

    resized_roi = cv2.resize(roi, (160, 120)) # Increased size
    resized_hsv = cv2.resize(hsv, (160, 120)) # Increased size
    resized_mask = cv2.cvtColor(cv2.resize(mask, (160, 120)), cv2.COLOR_GRAY2BGR) # Increased size
    resized_mask_opened = cv2.cvtColor(cv2.resize(mask_opened, (160, 120)), cv2.COLOR_GRAY2BGR) # Increased size
    resized_mask_cleaned = cv2.cvtColor(cv2.resize(mask_cleaned, (160, 120)), cv2.COLOR_GRAY2BGR) # Increased size
    resized_contour = cv2.resize(roi_with_contour, (160, 120)) # Increased size
    resized_hull = cv2.resize(roi_with_hull, (160, 120)) # Increased size
    resized_defects = cv2.resize(roi_with_defects, (160, 120)) # Increased size
    resized_hull_on_mask = cv2.resize(mask_with_hull_defects, (160, 120)) # Increased size

    # Create a 3x3 display window by stacking the resized images
    row1 = np.hstack((resized_roi, resized_hsv, resized_mask))
    row2 = np.hstack((resized_mask_opened, resized_mask_cleaned, resized_contour))
    row3 = np.hstack((resized_hull, resized_defects, resized_hull_on_mask))

    final_display = np.vstack((row1, row2, row3))

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 20
    x_offset = 5

    cv2.putText(final_display, "ROI", (x_offset, y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "HSV", (160 + x_offset, y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Mask", (320 + x_offset, y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Opened", (x_offset, 120 + y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Closed", (160 + x_offset, 120 + y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Contour (on ROI)", (320 + x_offset - 20, 120 + y_offset), font, TEXT_FONT_SCALE - 0.1, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Hull (on ROI)", (x_offset, 240 + y_offset), font, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Defects (on ROI)", (160 + x_offset - 10, 240 + y_offset), font, TEXT_FONT_SCALE - 0.1, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(final_display, "Hull/\nDefects\n(on Mask)", (320 + x_offset - 20, 240 + y_offset - 10), font, TEXT_FONT_SCALE - 0.1, TEXT_COLOR, TEXT_THICKNESS)


    cv2.imshow("Hand Processing Steps", final_display)
    cv2.imshow("Original Frame", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()