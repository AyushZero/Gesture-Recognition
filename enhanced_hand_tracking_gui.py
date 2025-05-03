import cv2
import numpy as np
import math
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

class HandTrackingGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Enhanced Hand Tracking")
        self.window.configure(bg='#2b2b2b')
        
        # Create main frame
        self.main_frame = ttk.Frame(window)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Create style
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#2b2b2b')
        self.style.configure('Custom.TLabel', background='#2b2b2b', foreground='white')
        self.style.configure('Status.TLabel', background='#2b2b2b', foreground='#00ff00', font=('Arial', 12, 'bold'))
        
        # Create status bar
        self.status_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.status_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0", style='Status.TLabel')
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.gesture_label = ttk.Label(self.status_frame, text="Gesture: No Hand (0)", style='Status.TLabel')
        self.gesture_label.pack(side=tk.LEFT, padx=10)
        
        # Create frames for each section
        self.create_frames()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam {WEBCAM_INDEX}")
            return
        
        # Initialize variables
        self.prev_time = 0
        self.photo_images = {}  # Store PhotoImage objects
        
        # Start video processing
        self.process_video()
        
    def create_frames(self):
        # Create frames for each section
        self.frames = {}
        positions = {
            'Original Frame': (0, 0),
            'ROI': (0, 1),
            'HSV': (0, 2),
            'Mask': (1, 0),
            'Opened': (1, 1),
            'Closed': (1, 2),
            'Contour': (2, 0),
            'Hull': (2, 1),
            'Defects': (2, 2)
        }
        
        for name, (row, col) in positions.items():
            frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
            frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Add label
            label = ttk.Label(frame, text=name, style='Custom.TLabel')
            label.pack()
            
            # Add image label
            image_label = ttk.Label(frame)
            image_label.pack()
            
            self.frames[name] = {'frame': frame, 'label': image_label}
        
        # Configure grid weights
        for i in range(3):
            self.main_frame.grid_rowconfigure(i, weight=1)
            self.main_frame.grid_columnconfigure(i, weight=1)
    
    def process_frame(self, frame):
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
        if self.prev_time > 0:
            fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Update status labels
        self.fps_label.configure(text=f"FPS: {int(fps)}")
        self.gesture_label.configure(text=f"Gesture: {gesture} ({finger_count})")
        
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
    
    def update_image(self, cv_img, label):
        """Convert OpenCV image to PhotoImage and update label"""
        # Resize image to fit the label
        height, width = cv_img.shape[:2]
        max_size = 200
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        cv_img = cv2.resize(cv_img, (new_width, new_height))
        
        # Convert to RGB for PIL
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(cv_img)
        photo = ImageTk.PhotoImage(image=img)
        
        # Update label
        label.configure(image=photo)
        label.image = photo  # Keep a reference
    
    def process_video(self):
        """Process video frames and update GUI"""
        ret, frame = self.cap.read()
        if ret:
            results = self.process_frame(frame)
            
            # Update all frames
            self.update_image(results['frame'], self.frames['Original Frame']['label'])
            self.update_image(results['roi'], self.frames['ROI']['label'])
            self.update_image(results['hsv'], self.frames['HSV']['label'])
            self.update_image(results['mask'], self.frames['Mask']['label'])
            self.update_image(results['mask_opened'], self.frames['Opened']['label'])
            self.update_image(results['mask_cleaned'], self.frames['Closed']['label'])
            self.update_image(results['roi_with_contour'], self.frames['Contour']['label'])
            self.update_image(results['roi_with_hull'], self.frames['Hull']['label'])
            self.update_image(results['roi_with_defects'], self.frames['Defects']['label'])
        
        # Schedule next update
        self.window.after(10, self.process_video)
    
    def __del__(self):
        """Cleanup when the window is closed"""
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    root = tk.Tk()
    app = HandTrackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 