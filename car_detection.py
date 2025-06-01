from ultralytics import YOLO
import cv2

class CarDetector:
    def __init__(self):
        # Charger YOLOv8 pré-entraîné sur COCO (inclut la classe 'car')
        self.model = YOLO("yolov8n.pt")  # ou yolov8s.pt, yolov8m.pt selon la précision voulue
    
    def detect(self, image):
        results = self.model(image)[0]
        cars = []
        for box in results.boxes:
            if int(box.cls) == 2:  # Class 2 = 'car' dans COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cars.append((x1, y1, x2, y2))
        return cars