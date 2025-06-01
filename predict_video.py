import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os
import csv
import pygame
import motmetrics as mm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def euclidean_distances(a, b, max_d=np.inf):
    """Calcule les distances euclidiennes entre deux ensembles de points"""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    a = np.array(a)
    b = np.array(b)
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    if max_d < np.inf:
        dists[dists > max_d] = np.nan
    return dists

class Detector:
    def __init__(self, filepath):
        self.filepath = filepath

        # Initialize models
        self.car_model = YOLO("yolov8n.pt")
        self.plate_model = YOLO("runs/detect/train3/weights/best.pt")
        self.tracker = Sort()

        # MOT metrics
        mm.lap.default_solver = 'lap'
        self.accumulator = mm.MOTAccumulator(auto_id=True)
        
        # Journaux et métriques
        self.detections_log = []
        self.heatmap_points = []
        self.tracking_log = []

        # Output files
        self.output_video = "output/detection_output.mp4"
        self.output_csv = "output/detection_logs.csv"
        self.output_metrics = "output/mot_metrics.csv"
        self.output_heatmap = "output/heatmap.png"

        os.makedirs("output", exist_ok=True)

        # Initialize audio
        pygame.init()
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound("/home/students-asn33/Téléchargements/Licence_Plate_CV/alert.mp3")
        except:
            print("⚠️ Fichier audio non trouvé - alerte désactivée")
            self.alert_sound = None

    def associate_plate_to_car(self, plate_box, car_boxes):
        """Associe une plaque d'immatriculation à un véhicule"""
        px1, py1, px2, py2 = plate_box
        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)

        min_dist = float("inf")
        best_car = None

        for car_box in car_boxes:
            cx1, cy1, cx2, cy2 = car_box
            car_center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)

            # Vérifier si la plaque est à l'intérieur du véhicule
            if cx1 <= plate_center[0] <= cx2 and cy1 <= plate_center[1] <= cy2:
                return car_box

            # Sinon, trouver le véhicule le plus proche
            dist = np.linalg.norm(np.array(plate_center) - np.array(car_center))
            if dist < min_dist:
                min_dist = dist
                best_car = car_box

        return best_car

    def log_detection(self, frame_count, track_id, bbox, class_name, confidence, vehicle_bbox=None):
        """Enregistre une détection dans le journal"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = bbox
        
        detection_entry = {
            "timestamp": timestamp,
            "frame": frame_count,
            "track_id": track_id,
            "class": class_name,
            "bbox": f"({x1}, {y1}, {x2}, {y2})",
            "confidence": round(confidence, 2),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2
        }
        
        if vehicle_bbox is not None:
            vx1, vy1, vx2, vy2 = vehicle_bbox
            detection_entry["vehicle_bbox"] = f"({vx1}, {vy1}, {vx2}, {vy2})"
            detection_entry["vehicle_center_x"] = (vx1 + vx2) / 2
            detection_entry["vehicle_center_y"] = (vy1 + vy2) / 2
        
        self.detections_log.append(detection_entry)
        
        # Ajouter le point central pour la heatmap
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.heatmap_points.append(center)

    def update_mot_metrics(self, gt_boxes, gt_ids, pred_boxes, pred_ids):
        """Met à jour les métriques MOT"""
        # Centres des boîtes ground truth et prédites
        gt_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in gt_boxes]
        pred_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in pred_boxes]
        
        # Calculer les distances
        distances = euclidean_distances(gt_centers, pred_centers, max_d=100)
        
        # Mettre à jour l'accumulateur
        self.accumulator.update(gt_ids, pred_ids, distances)

    def generate_heatmap(self):
        """Génère une heatmap des positions détectées"""
        if not self.heatmap_points:
            return None
            
        try:
            x_coords = [pt[0] for pt in self.heatmap_points]
            y_coords = [pt[1] for pt in self.heatmap_points]

            # Créer l'histogramme 2D
            heatmap, xedges, yedges = np.histogram2d(y_coords, x_coords, bins=(100, 100))
            heatmap = np.rot90(heatmap)
            heatmap = np.flipud(heatmap)

            # Créer la visualisation
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap, cmap='inferno', cbar=True, 
                       xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title("Heatmap - Zones de détection des plaques d'immatriculation", 
                        fontsize=14, fontweight='bold')
            ax.axis("off")

            # Sauvegarder
            plt.savefig(self.output_heatmap, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Heatmap sauvegardée : {self.output_heatmap}")
            return True
        except Exception as e:
            print(f"❌ Erreur génération heatmap : {e}")
            return False

    def save_mot_metrics(self):
        """Sauvegarde les métriques MOT"""
        try:
            mh = mm.metrics.create()
            summary = mh.compute(self.accumulator, 
                               metrics=mm.metrics.motchallenge_metrics, 
                               name='DetectionResults')
            
            if not summary.empty:
                summary_rounded = summary.round(4)
                summary_rounded.to_csv(self.output_metrics)
                print(f"✅ Métriques MOT sauvegardées : {self.output_metrics}")
                return summary_rounded
            else:
                print("⚠️ Aucune métrique MOT disponible")
                return None
        except Exception as e:
            print(f"❌ Erreur sauvegarde métriques MOT : {e}")
            return None

    def save_detection_logs(self):
        """Sauvegarde le journal des détections"""
        if not self.detections_log:
            print("⚠️ Aucune détection à sauvegarder")
            return
            
        try:
            # Sauvegarder en CSV
            log_csv = "output/detection_journal.csv"
            fieldnames = self.detections_log[0].keys()
            
            with open(log_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detections_log)
            
            print(f"✅ Journal des détections sauvegardé : {log_csv}")
            print(f"📊 Nombre total de détections : {len(self.detections_log)}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde journal : {e}")

    def forward(self):
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        csv_file = open(self.output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2', 'class', 'confidence'])

        frame_count = 0
        total_detected_plates = 0
        total_vehicles_with_plate = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detection_active = False

            # --- Détecter les véhicules ---
            car_results = self.car_model(frame_rgb)[0]
            car_boxes = []
            for box in car_results.boxes:
                if int(box.cls) in [2, 5, 7]:  # car, bus, truck
                    car_boxes.append(box.xyxy[0].cpu().numpy())

            # --- Détecter les plaques ---
            plate_detections = []
            gt_boxes = []  # Pour les métriques MOT
            gt_ids = []
            
            for car_box in car_boxes:
                x1, y1, x2, y2 = map(int, car_box)
                car_roi = frame_rgb[y1:y2, x1:x2]

                plate_results = self.plate_model(car_roi)[0]
                for box in plate_results.boxes:
                    if float(box.conf) > 0.3:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        global_coords = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]
                        plate_detections.append(global_coords + [
                            float(box.conf),
                            int(box.cls)
                        ])
                        
                        # Pour les métriques MOT
                        gt_boxes.append(global_coords)
                        gt_ids.append(len(gt_boxes) - 1)

            if plate_detections:
                # Suivi des objets
                dets = np.array([d[:4] + [d[4]] for d in plate_detections])
                tracks = self.tracker.update(dets)
                total_detected_plates += len(tracks)

                pred_boxes = []
                pred_ids = []

                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track[:5])
                    conf = track[4] if len(track) > 5 else 0.5
                    
                    pred_boxes.append([x1, y1, x2, y2])
                    pred_ids.append(int(track_id))

                    # Associer la plaque au véhicule
                    associated_car = self.associate_plate_to_car([x1, y1, x2, y2], car_boxes)
                    vehicle_bbox = None
                    
                    if associated_car is not None:
                        detection_active = True
                        total_vehicles_with_plate += 1
                        vehicle_bbox = list(map(int, associated_car))

                        # Alerte sonore
                        if self.alert_sound and not pygame.mixer.get_busy():
                            self.alert_sound.play()

                        cx1, cy1, cx2, cy2 = vehicle_bbox
                        plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        car_center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                        cv2.line(frame, plate_center, car_center, (0, 0, 255), 2)

                        # Dessiner le véhicule
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
                        cv2.putText(frame, "Vehicle", (cx1, cy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Dessiner la plaque
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Plate {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Enregistrer dans le journal
                    class_name = "license_plate"
                    if len(plate_detections) > 0 and len(self.plate_model.names) > plate_detections[0][5]:
                        class_name = self.plate_model.names[plate_detections[0][5]]
                    
                    self.log_detection(frame_count, int(track_id), [x1, y1, x2, y2], 
                                     class_name, conf, vehicle_bbox)

                    # CSV original
                    csv_writer.writerow([
                        frame_count, track_id, x1, y1, x2, y2, class_name, conf
                    ])

                # Mettre à jour les métriques MOT
                if gt_boxes and pred_boxes:
                    self.update_mot_metrics(gt_boxes, gt_ids, pred_boxes, pred_ids)

            # Affichage du statut
            if detection_active:
                cv2.putText(frame, "Détection active : Véhicule avec plaque",
                            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Aucune détection",
                            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            # Afficher les métriques
            cv2.putText(frame, f"Plaques détectées : {total_detected_plates}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Véhicules associés : {total_vehicles_with_plate}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame : {frame_count}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)
            
            # IMPORTANT: Yield chaque frame pour Streamlit
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Nettoyage et sauvegarde finale
        cap.release()
        writer.release()
        csv_file.close()

        print(f"\n🎬 Traitement vidéo terminé !")
        print(f"📁 Vidéo de sortie : {self.output_video}")
        print(f"📊 CSV des détections : {self.output_csv}")

        # Sauvegarder tous les résultats
        self.save_detection_logs()
        mot_metrics = self.save_mot_metrics()
        self.generate_heatmap()

        # Afficher un résumé final
        print(f"\n📈 RÉSUMÉ FINAL :")
        print(f"• Frames traitées : {frame_count}")
        print(f"• Détections totales : {len(self.detections_log)}")
        print(f"• Points heatmap : {len(self.heatmap_points)}")
        
        # Yield final avec les métriques (tuple pour signaler la fin)
        yield (None, mot_metrics)