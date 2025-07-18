import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Load model with proper path handling
model_path = os.path.join('..', 'runs', 'detect', 'train', 'weights', 'best.pt')
if not os.path.exists(model_path):
    model_path = 'runs/detect/train/weights/best.pt'

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

st.title("🎭 Face Expression Detection")
st.write("Upload an image to detect facial expressions")

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
