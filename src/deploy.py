import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')
st.title("Deteksi Ekspresi Wajah")
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    results = model(img)
    st.image(results.render()[0], caption="Hasil Deteksi")
