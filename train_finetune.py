from ultralytics import YOLO
model= YOLO("yolov8n.pt ")
results=model.train(data="/home/students-asn33/Téléchargements/Licence_Plate_CV/License Plate Detector.v2i.yolov8/data.yaml",epochs=50)