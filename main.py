import argparse
import os
from predict_image import ImDetector
from predict_video import Detector

def parse_args():
    parser = argparse.ArgumentParser(description="Détection Voitures & Plaques (Image/Video)")
    parser.add_argument("--filepath", type=str, required=True,
                      help="Chemin vers le fichier image/video")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Seuil de confiance (défaut=0.5)")
    parser.add_argument("--show-cars", action="store_true",
                      help="Afficher les détections de voitures")
    parser.add_argument("--output", type=str, default="output",
                      help="Dossier de sortie pour les résultats")
    return parser.parse_args()

def is_image(filepath):
    return filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def is_video(filepath):
    return filepath.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output)
    
    if not os.path.exists(args.filepath):
        print(f"❌ Fichier introuvable: {args.filepath}")
        return
    
    if is_image(args.filepath):
        print("🔍 Traitement d'une image...")
        detector = ImDetector(args.filepath, args.threshold, args.show_cars)
        
        results = detector.forward()
        car_count = len(results['car_detections'])
        plate_count = len([box for box in results['plate_detections'] 
                         if float(box.conf[0]) > args.threshold])
        
        print(f"🚗 Voitures détectées: {car_count}")
        print(f"🚘 Plaques détectées: {plate_count}")
        
        detector.plot()
        print(f"✅ Résultats sauvegardés dans: {output_dir}")
        
    elif is_video(args.filepath):
        print("🎥 Traitement d'une vidéo...")
        detector = Detector(args.filepath)
        
        print("⏳ Analyse en cours...")
        for frame in detector.forward():
            if isinstance(frame, tuple):
                metrics = frame[1]
                print("\n📊 Métriques finales:")
                print(metrics)
                break
        
        print(f"✅ Résultats vidéo sauvegardés dans: {output_dir}")
    else:
        print("❌ Format non supporté. Formats acceptés: .jpg, .png, .mp4, .avi")

if __name__ == '__main__':
    main()