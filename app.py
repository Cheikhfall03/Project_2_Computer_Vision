import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import os
from predict_image import ImDetector
from predict_video import Detector
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-container {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .stats-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    show_stats = st.checkbox("Afficher les statistiques", value=True)
    show_heatmap = st.checkbox("Générer la heatmap", value=True)
    show_journal = st.checkbox("Afficher le journal", value=True)

# App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🚗 License Plate Detection System</h1>', unsafe_allow_html=True)

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Résultats", "🔥 Heatmap", "📝 Journal"])

with tab1:
    # File Upload
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📁 Sélectionnez votre fichier")
    uploaded_file = st.file_uploader("Choisissez une image ou vidéo", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        process_btn = st.button("🚀 Traiter", key="process", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Effacer", key="clear", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Session State
if 'results' not in st.session_state:
    st.session_state.results = None
if 'status' not in st.session_state:
    st.session_state.status = ""
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def is_image(file):
    return file.lower().endswith(('.jpg', '.jpeg', '.png'))

def is_video(file):
    return file.lower().endswith(('.mp4', '.avi'))

# Clear button functionality
if clear_btn:
    st.session_state.results = None
    st.session_state.status = ""
    st.session_state.detector = None
    st.session_state.processing = False
    st.rerun()

# Processing
if process_btn and uploaded_file and not st.session_state.processing:
    st.session_state.processing = True
    temp_path = f"temp_{uploaded_file.name}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        if is_image(temp_path):
            with tab1:
                st.markdown('<div class="stats-box">📸 Traitement de l\'image en cours...</div>', unsafe_allow_html=True)
            
            detector = ImDetector(temp_path, confidence)
            results = detector.forward()
            
            # Process image results
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].imshow(results['original_image'])
            ax[0].set_title('Image Originale')
            ax[0].axis('off')
            
            ax[1].imshow(results['annotated_image'])
            ax[1].set_title('Détections')
            ax[1].axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            st.session_state.results = Image.open(buf)
            st.session_state.status = "✅ Image traitée avec succès"
            
        elif is_video(temp_path):
            with tab1:
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                video_placeholder = st.empty()
                
                status_placeholder.markdown('<div class="stats-box">🎬 Traitement de la vidéo en cours...</div>', unsafe_allow_html=True)
            
            detector = Detector(temp_path)
            st.session_state.detector = detector
            
            frame_count = 0
            total_frames = int(cv2.VideoCapture(temp_path).get(cv2.CAP_PROP_FRAME_COUNT))
            
            for frame_data in detector.forward():
                if isinstance(frame_data, tuple):
                    # Fin du traitement
                    break
                
                frame_count += 1
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
                # Afficher chaque 10ème frame pour éviter la surcharge
                if frame_count % 10 == 0:
                    video_placeholder.image(frame_data, channels="RGB", use_container_width=True)
                
                status_placeholder.markdown(
                    f'<div class="stats-box">🎬 Frame {frame_count}/{total_frames} - Détections: {len(detector.detections_log)}</div>', 
                    unsafe_allow_html=True
                )
            
            st.session_state.status = "✅ Vidéo traitée avec succès"
            progress_bar.progress(1.0)
            
            # Vérifier si les fichiers de sortie existent
            if os.path.exists("output/detection_output.mp4"):
                st.session_state.results = "output/detection_output.mp4"
            
    except Exception as e:
        st.session_state.status = f"❌ Erreur: {str(e)}"
        with tab1:
            st.markdown(f'<div class="error-box">❌ Erreur: {str(e)}</div>', unsafe_allow_html=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.processing = False

# Display Status
with tab1:
    if st.session_state.status:
        if "✅" in st.session_state.status:
            st.markdown(f'<div class="success-box">{st.session_state.status}</div>', unsafe_allow_html=True)
        elif "❌" in st.session_state.status:
            st.markdown(f'<div class="error-box">{st.session_state.status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="stats-box">{st.session_state.status}</div>', unsafe_allow_html=True)

# Tab 2: Résultats
with tab2:
    if st.session_state.results:
        st.subheader("🎯 Résultats de détection")
        
        if isinstance(st.session_state.results, Image.Image):
            st.image(st.session_state.results, use_container_width=True)
        elif isinstance(st.session_state.results, str):
            st.video(st.session_state.results)
        
        # Afficher les statistiques si disponibles
        if show_stats and st.session_state.detector and hasattr(st.session_state.detector, 'detections_log'):
            detector = st.session_state.detector
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Détections totales", len(detector.detections_log))
            with col2:
                st.metric("Points heatmap", len(detector.heatmap_points))
            with col3:
                unique_ids = len(set([d['track_id'] for d in detector.detections_log]))
                st.metric("Objets uniques", unique_ids)
                
        # Télécharger les fichiers de résultats
        st.subheader("📥 Téléchargements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists("output/detection_output.mp4"):
                with open("output/detection_output.mp4", "rb") as f:
                    st.download_button("📹 Vidéo annotée", f.read(), "detection_output.mp4", "video/mp4")
        
        with col2:
            if os.path.exists("output/detection_journal.csv"):
                df = pd.read_csv("output/detection_journal.csv")
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("📊 Journal CSV", csv_data, "detection_journal.csv", "text/csv")
        
        with col3:
            if os.path.exists("output/mot_metrics.csv"):
                with open("output/mot_metrics.csv", "rb") as f:
                    st.download_button("📈 Métriques MOT", f.read(), "mot_metrics.csv", "text/csv")

# Tab 3: Heatmap
with tab3:
    if show_heatmap and os.path.exists("output/heatmap.png"):
        st.subheader("🔥 Heatmap des détections")
        st.image("output/heatmap.png", use_container_width=True)
        st.info("Cette heatmap montre les zones où les plaques d'immatriculation sont le plus fréquemment détectées.")
        
        # Bouton de téléchargement pour la heatmap
        with open("output/heatmap.png", "rb") as f:
            st.download_button("💾 Télécharger la heatmap", f.read(), "heatmap.png", "image/png")
    else:
        st.info("🔥 La heatmap sera générée après le traitement d'une vidéo.")

# Tab 4: Journal
with tab4:
    if show_journal and os.path.exists("output/detection_journal.csv"):
        st.subheader("📝 Journal des détections")
        
        df = pd.read_csv("output/detection_journal.csv")
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            unique_classes = df['class'].unique() if 'class' in df.columns else []
            selected_class = st.            selected_class = st.multiselect("Filtrer par classe", options=unique_classes, default=unique_classes)
        
        with col2:
            if 'track_id' in df.columns:
                unique_ids = df['track_id'].unique()
                selected_ids = st.multiselect("Filtrer par ID", options=unique_ids, default=unique_ids)
        
        # Appliquer les filtres
        if selected_class:
            df = df[df['class'].isin(selected_class)]
        if 'track_id' in df.columns and selected_ids:
            df = df[df['track_id'].isin(selected_ids)]
        
        # Afficher le tableau
        st.dataframe(df, height=500, use_container_width=True)
        
        # Statistiques
        st.subheader("📊 Statistiques")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total des entrées", len(df))
        
        with col2:
            if 'class' in df.columns:
                class_counts = df['class'].value_counts()
                st.write("Répartition par classe:")
                st.bar_chart(class_counts)
        
        with col3:
            if 'timestamp' in df.columns:
                st.write("Détections par minute:")
                df['minute'] = pd.to_datetime(df['timestamp']).dt.floor('min')
                minute_counts = df['minute'].value_counts().sort_index()
                st.line_chart(minute_counts)
    else:
        st.info("📝 Le journal sera disponible après le traitement d'une vidéo.")

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #6c757d;">
        <hr>
        <p>License Plate Detection System © 2023</p>
    </div>
""", unsafe_allow_html=True)