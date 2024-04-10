from ultralytics import YOLO
import cv2

model = YOLO("yolov8l.pt")
results = model("C:\\Users\\User\\Documents\\extract-image\\1236.jpg", show=True)
cv2.waitKey(0)