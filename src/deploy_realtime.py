import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time

# Load model with proper path handling
model_path = os.path.join('..', 'runs', 'detect', 'train', 'weights', 'best.pt')
if not os.path.exists(model_path):
    model_path = 'runs/detect/train/weights/best.pt'

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

st.title("🎭 Real-Time Face Expression Detection")
st.write("Choose between camera modes for real-time facial expression detection")

# App mode selection
app_mode = st.selectbox("Choose Mode",
                       ["Photo Upload", "Camera Snapshot", "Live Camera Feed"])

if app_mode == "Photo Upload":
    st.subheader("📸 Upload Image Mode")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)
        
        # Run inference
        with st.spinner("Detecting expressions..."):
            results = model(img)
        
        with col2:
            st.subheader("Detection Results")
            # Plot results on image
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)
        
        # Display detection details
        if len(results[0].boxes) > 0:
            st.subheader("Detection Details")
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                st.write(f"**Detection {i+1}:** {class_name} (Confidence: {conf:.2f})")
        else:
            st.write("No faces detected in the image.")

elif app_mode == "Camera Snapshot":
    st.subheader("📱 Camera Snapshot Mode")
    st.write("Take a photo using your camera to detect expressions")
    
    # Camera input
    camera_photo = st.camera_input("Take a photo")
    
    if camera_photo is not None:
        # Convert to PIL Image
        img = Image.open(camera_photo)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(img, use_container_width=True)
        
        # Run inference
        with st.spinner("Detecting expressions..."):
            results = model(img)
        
        with col2:
            st.subheader("Detection Results")
            # Plot results on image
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)
        
        # Display detection details
        if len(results[0].boxes) > 0:
            st.subheader("Detection Details")
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                st.write(f"**Detection {i+1}:** {class_name} (Confidence: {conf:.2f})")
        else:
            st.write("No faces detected in the image.")

elif app_mode == "Live Camera Feed":
    st.subheader("🎥 Live Camera Feed Mode")
    st.write("Real-time facial expression detection from your camera")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🎬 Start Camera")
    with col2:
        stop_button = st.button("⏹️ Stop Camera")
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Session state for camera
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    
    if start_button:
        st.session_state.camera_active = True
    
    if stop_button:
        st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        # Create placeholder for video feed
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # OpenCV camera capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open camera. Please check if camera is connected and not being used by another application.")
        else:
            st.success("Camera started! Press 'Stop Camera' to end the session.")
            
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run YOLO inference
                results = model(rgb_frame, conf=confidence_threshold)
                
                # Plot results
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Display frame
                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Display statistics
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                stats_placeholder.metric("Live Stats", f"FPS: {fps:.1f} | Detections: {detections}")
                
                # Detection details
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    expressions = []
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = results[0].names[cls]
                        expressions.append(f"{class_name} ({conf:.2f})")
                    
                    if expressions:
                        st.sidebar.subheader("Current Detections")
                        for expr in expressions:
                            st.sidebar.write(f"• {expr}")
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.1)
            
            cap.release()
            video_placeholder.empty()
            stats_placeholder.empty()
            st.info("Camera stopped.")

# Sidebar information
st.sidebar.header("ℹ️ Information")
st.sidebar.write("**Model:** YOLOv8 Facial Expression Detection")
st.sidebar.write("**Classes:** 8 expressions")
st.sidebar.write("- Neutral")
st.sidebar.write("- Happy")
st.sidebar.write("- Sad")
st.sidebar.write("- Surprise")
st.sidebar.write("- Fear")
st.sidebar.write("- Disgust")
st.sidebar.write("- Anger")
st.sidebar.write("- Contempt")

st.sidebar.header("🔧 Tips")
st.sidebar.write("• Ensure good lighting for better detection")
st.sidebar.write("• Position face clearly in frame")
st.sidebar.write("• Adjust confidence threshold if needed")
st.sidebar.write("• Use Chrome/Firefox for best camera support")
