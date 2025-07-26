import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect white and yellow lines
    white = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
    yellow = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    edges = cv2.Canny(mask, 50, 150)
    cropped = region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=100)
    line_image = np.zeros_like(frame)

    if lines is not None:
        left_lines, right_lines = [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 0.01)
                if abs(slope) < 0.5: continue
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        def draw_average(lines, img, color):
            if lines:
                x = []
                y = []
                for x1, y1, x2, y2 in lines:
                    x += [x1, x2]
                    y += [y1, y2]
                poly = np.polyfit(y, x, deg=1)
                y1 = img.shape[0]
                y2 = int(y1 * 0.6)
                x1 = int(poly[0] * y1 + poly[1])
                x2 = int(poly[0] * y2 + poly[1])
                cv2.line(img, (x1, y1), (x2, y2), color, 8)

        draw_average(left_lines, line_image, (255, 0, 0))
        draw_average(right_lines, line_image, (0, 255, 0))

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def select_video():
    path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi")])
    if path:
        threading.Thread(target=play_video, args=(path,)).start()

def play_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = detect_lanes(frame)
        processed = cv2.resize(processed, (600, 400))
        img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.config(image=img)
        video_label.image = img
        root.update()
    cap.release()

#GUI
root = tk.Tk()
root.title("Road Lane Detection System")
root.geometry("700x500")
root.config(bg="#121212")

title = tk.Label(root, text="Smart Road Lane Detection", font=("Segoe UI", 18, "bold"), fg="white", bg="#121212")
title.pack(pady=10)

select_btn = tk.Button(root, text="Choose Road Video", font=("Arial", 14), command=select_video, bg="#ff6b6b", fg="white", padx=10, pady=5)
select_btn.pack(pady=10)

video_label = tk.Label(root)
video_label.pack(pady=10)

footer = tk.Label(root, text="Developed with ❤️ using OpenCV & Tkinter", font=("Segoe UI", 10), fg="gray", bg="#121212")
footer.pack(side=tk.BOTTOM, pady=5)

root.mainloop()
