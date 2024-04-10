# Importing the Dependencies
import cv2
from ultralytics import YOLO
import cvzone
import math
import threading
import numpy as np
from sort import *


# Load the name of the objects
classname = ['person', 'bicycle', 'car', 'motorcycle',  'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'cup', 'water bottle',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def object_detection():
    # Load settings
    limits = [289, 468, 650, 158]
    totalCountId = []

    car_count = 0
    global model, classname
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture('Datasets\japanese_transport_edit.mp4')
    #Tracking
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

    # Set the size when using webcam
    # cap.set(3, 1920)
    # cap.set(4, 1080)
    
    # Main loop
    while True:
        _, img = cap.read()
        results = model(img, stream=True)
        detection_list = []
        for r in results:
            # Init the bounding box
            boxes = r.boxes
            # Draw the bounding box
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                #cvzone.cornerRect(img, (x1, y1, w, h), l=8)
                conf = (math.ceil(box.conf[0]*100)) / 100
                cls = int(box.cls[0])
                # Check if the current class is car, truck, bus, motorcycle, or bicycle and if confidence is greater than 0.3
                current_class = classname[cls]
                if current_class in ['car', 'truck', 'bus','motorcycle','bicycle'] and conf > 0.3:
                    detection_list.append([x1, y1, x2, y2, conf])
                    #cvzone.putTextRect(img, f'{classname[cls]} {conf}', (max(0, x1), max(20, y1)), scale=0.6, thickness=1, offset=5)
                detections_array = np.array(detection_list)   
        
        resultsTracker = tracker.update(detections_array) # Update the tracker with the current detections
        #Vẽ đường thẳng giới hạn
        cv2.line(img, (limits[0]-30, limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        for results in resultsTracker:
            x1,y1,x2,y2,id = results.astype(int)
            print(results)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 255))
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(20, y1)), scale=0.6, thickness=1, offset=5)
            cx, cy = x1 + w//2, y1 + h//2
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                car_count += 1
        # Display the car count on the image
        cv2.putText(img, f'Car Count: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", img)
        cv2.waitKey(1)
        print(img.shape)

# Initialize a separate thread to run object_detection()
detection_thread = threading.Thread(target=object_detection)
detection_thread.start()
