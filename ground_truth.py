import os
import csv
from PIL import Image

labels_dir = "/home/students-asn33/Téléchargements/Licence_Plate_CV/License Plate Detector.v2i.yolov8/test/labels"
images_dir = "/home/students-asn33/Téléchargements/Licence_Plate_CV/License Plate Detector.v2i.yolov8/test/images"
output_csv = "ground_truth.csv"

def yolo_to_bbox(x_center_norm, y_center_norm, width_norm, height_norm, img_w, img_h):
    x_center = x_center_norm * img_w
    y_center = y_center_norm * img_h
    width = width_norm * img_w
    height = height_norm * img_h
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Colonnes : frame, image, class, x1, y1, x2, y2
    writer.writerow(['frame', 'image', 'class', 'x1', 'y1', 'x2', 'y2'])
    
    for frame_idx, label_filename in enumerate(sorted(os.listdir(labels_dir))):
        if not label_filename.endswith(".txt"):
            continue
        
        label_path = os.path.join(labels_dir, label_filename)
        image_filename = label_filename.replace('.txt', '.jpg')  # adapter si .png
        
        # Chargement de l'image pour récupérer sa taille
        img_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Image non trouvée : {img_path}")
            continue
        
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        # Lecture du fichier annotation
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                x1, y1, x2, y2 = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
                
                writer.writerow([frame_idx, image_filename, cls_id, x1, y1, x2, y2])

print(f"✅ Ground truth exporté dans {output_csv}")
