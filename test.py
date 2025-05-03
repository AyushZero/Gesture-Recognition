import cv2
import numpy as np
import math
import time

WEBCAM_INDEX = 0

ROI_Y1, ROI_Y2 = 100, 300
ROI_X1, ROI_X2 = 350, 550
ROI_W = ROI_X2 - ROI_X1
ROI_H = ROI_Y2 - ROI_Y1

LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS_OPEN = 1
MORPH_ITERATIONS_CLOSE = 1

MIN_CONTOUR_AREA = 1500

MIN_DEFECT_DEPTH = 15.0
MAX_DEFECT_ANGLE = 85.0

FIST_FILL_RATIO_THRESHOLD = 0.85

RECT_COLOR = (0, 255, 0)
HAND_CONTOUR_COLOR = (255, 0, 0)
HULL_COLOR = (0, 255, 0)
DEFECT_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 255)
CONTOUR_THICKNESS = 2
HULL_THICKNESS = 2
TEXT_FONT_SCALE = 0.8
TEXT_THICKNESS = 2

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}")
    exit()

print("Starting Hand Gesture Recognition...")
print("Press 'q' to quit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), RECT_COLOR, 2)

    if ROI_Y1 >= ROI_Y2 or ROI_X1 >= ROI_X2:
        print("Error: Invalid ROI coordinates. Skipping frame.")
        continue
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS_OPEN)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gesture = "No Hand"
    finger_count = 0
    max_contour = None
    hull_points = None

    if contours:
        try:
            max_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:

                hull_indices = cv2.convexHull(max_contour, returnPoints=False)
                hull_points = max_contour[hull_indices[:, 0]]

                if hull_indices is not None and len(hull_indices) > 3:
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
                                cv2.circle(roi, far, 5, DEFECT_COLOR, -1)

                        finger_count = count_defects + 1

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

                    else: # Defects is None
                        area_contour = cv2.contourArea(max_contour)
                        area_hull = cv2.contourArea(hull_points)
                        fill_ratio = area_contour / area_hull if area_hull > 0 else 0
                        if fill_ratio > FIST_FILL_RATIO_THRESHOLD:
                            gesture = "Fist"
                            finger_count = 0
                        else:
                            gesture = "Point / One?"
                            finger_count = 1

                else: # Hull indices invalid
                    gesture = "Hull Invalid"

                cv2.drawContours(roi, [max_contour], -1, HAND_CONTOUR_COLOR, CONTOUR_THICKNESS)
                cv2.drawContours(roi, [hull_points], -1, HULL_COLOR, HULL_THICKNESS)

            else: # Contour area too small
                 gesture = "Hand Too Small"

        except Exception as e:
            print(f"Error processing contours/defects: {e}")
            gesture = "Processing Error"

    current_time = time.time()
    fps = 0
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECT_COLOR, 2)
    cv2.putText(frame, f"Gesture: {gesture} ({finger_count})", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    cv2.imshow("Hand Tracking", frame)
    cv2.imshow("Cleaned Mask", mask_cleaned)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Releasing resources...")
cap.release()
cv2.destroyAllWindows() 