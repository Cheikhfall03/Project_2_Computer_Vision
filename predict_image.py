import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class ImDetector:
    def __init__(self, filepath, confidence_threshold=0.5, show_cars=True):
        self.filepath = filepath
        self.confidence = confidence_threshold
        self.show_cars = show_cars
        
        # Load models
        self.plate_model = YOLO("runs/detect/train3/weights/best.pt")  # Modèle plaques
        self.car_model = YOLO("yolov8n.pt") if show_cars else None     # Modèle COCO pour voitures
        
        # Colors for visualization
        self.colors = {
            'car': (255, 0, 0),     # Bleu
            'plate': (0, 255, 0),   # Vert
            'text': (255, 255, 255),# Blanc
            'link': (0, 0, 255)     # Rouge pour les lignes d’association
        }

    def associate_plates_to_cars(self, plate_boxes, car_boxes):
        associations = []
        
        for plate_box in plate_boxes:
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0].tolist())
            plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            
            min_dist = float("inf")
            best_car = None

            for car_box in car_boxes:
                cx1, cy1, cx2, cy2 = map(int, car_box.xyxy[0].tolist())
                car_center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)

                # Si la plaque est dans la boîte de la voiture
                if cx1 <= plate_center[0] <= cx2 and cy1 <= plate_center[1] <= cy2:
                    best_car = car_box
                    break
                
                # Sinon, chercher la voiture la plus proche
                dist = np.sqrt((plate_center[0] - car_center[0])**2 + (plate_center[1] - car_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_car = car_box
            
            associations.append({'plate': plate_box, 'car': best_car})
        
        return associations

    def forward(self):
        # Charger l’image
        img = cv2.imread(self.filepath)
        if img is None:
            raise ValueError(f"Could not read image from {self.filepath}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated = img_rgb.copy()
        
        # Détection voitures
        car_detections = []
        if self.show_cars and self.car_model:
            cars = self.car_model(img_rgb, conf=0.25)[0]
            for box in cars.boxes:
                class_id = int(box.cls)
                if class_id in [2, 5, 7]:  # car, bus, truck
                    car_detections.append(box)
        
        # Détection plaques
        plates = self.plate_model(img_rgb, conf=self.confidence)[0]
        plate_detections = plates.boxes
        
        # Dessiner voitures
        for box in car_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf)
            class_id = int(box.cls)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.colors['car'], 2)
            label = f"{self.car_model.names[class_id]} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1-10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2,
                        cv2.LINE_AA)
        
        # Dessiner plaques
        for box in plate_detections:
            if float(box.conf) > self.confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.colors['plate'], 2)
                label = f"plate {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1-10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2,
                            cv2.LINE_AA)

        # Associer plaques ↔ voitures
        associations = self.associate_plates_to_cars(plate_detections, car_detections)

        # Tracer lignes entre plaque et voiture associée
        for pair in associations:
            plate_box = pair['plate']
            car_box = pair['car']
            if plate_box is not None and car_box is not None:
                px1, py1, px2, py2 = map(int, plate_box.xyxy[0].tolist())
                cx1, cy1, cx2, cy2 = map(int, car_box.xyxy[0].tolist())
                plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                car_center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                cv2.line(annotated, plate_center, car_center, self.colors['link'], 2)

        return {
            'original_image': img_rgb,
            'annotated_image': annotated,
            'car_detections': car_detections,
            'plate_detections': plate_detections,
            'plate_car_pairs': associations
        }

    def plot(self):
        results = self.forward()
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(results['original_image'])
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(results['annotated_image'])
        plt.title('Detection Results (Blue: Cars, Green: Plates, Red: Links)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
