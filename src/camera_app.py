import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Expression Detection",
    page_icon="🎭",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join('..', 'runs', 'detect', 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = 'runs/detect/train/weights/best.pt'
    
    # Load model and verify class names
    model = YOLO(model_path)
    
    # Ensure class names match training data
    expected_classes = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
    if hasattr(model, 'names') and model.names:
        actual_classes = list(model.names.values())
        st.sidebar.write(f"**Model Classes:** {actual_classes}")
        if actual_classes != expected_classes:
            st.sidebar.warning("⚠️ Model classes don't match expected classes!")
    
    return model

model = load_model()

st.title("🎭 Real-Time Facial Expression Detection")

# Sidebar controls
st.sidebar.header("🎛️ Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05)
show_fps = st.sidebar.checkbox("Show FPS", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_all_detections = st.sidebar.checkbox("Show All Detections", value=False)

# Image preprocessing options
st.sidebar.header("📸 Image Settings")
image_size = st.sidebar.selectbox("Image Size", [320, 416, 640, 832], index=2)
enhance_contrast = st.sidebar.checkbox("Enhance Contrast", value=False)
denoise = st.sidebar.checkbox("Denoise Image", value=False)

# Camera controls
col1, col2 = st.columns([1, 1])
with col1:
    start_camera = st.button("🎬 Start Camera", type="primary")
with col2:
    stop_camera = st.button("⏹️ Stop Camera")

# Initialize session state
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if start_camera:
    st.session_state.camera_running = True

if stop_camera:
    st.session_state.camera_running = False

# Main camera feed
if st.session_state.camera_running:
    # Create containers for the video feed and info
    video_container = st.container()
    info_container = st.container()
    
    with video_container:
        video_placeholder = st.empty()
    
    with info_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            fps_placeholder = st.empty()
        with col2:
            detection_placeholder = st.empty()
        with col3:
            expression_placeholder = st.empty()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size * 0.75))  # 4:3 aspect ratio
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        st.error("❌ Cannot access camera. Please check camera permissions and try again.")
        st.session_state.camera_running = False
    else:
        st.success("✅ Camera started successfully!")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Image preprocessing
                if enhance_contrast:
                    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                
                if denoise:
                    frame = cv2.bilateralFilter(frame, 9, 75, 75)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference with additional parameters
                results = model(rgb_frame, 
                              conf=confidence_threshold, 
                              iou=iou_threshold,
                              imgsz=image_size,
                              verbose=False)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Display frame
                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Update info
                if show_fps:
                    fps_placeholder.metric("FPS", f"{current_fps:.1f}")
                
                # Detection info
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    detection_placeholder.metric("Faces Detected", len(detections))
                    
                    # Sort detections by confidence
                    sorted_detections = sorted(zip(detections.conf, detections.cls, detections.xyxy), 
                                             key=lambda x: x[0], reverse=True)
                    
                    if show_all_detections:
                        # Show all detections
                        detection_text = []
                        for i, (conf, cls, bbox) in enumerate(sorted_detections):
                            conf_val = float(conf)
                            class_id = int(cls)
                            class_name = results[0].names[class_id]
                            detection_text.append(f"{i+1}. {class_name} ({conf_val:.2f})")
                        
                        expression_placeholder.text("\n".join(detection_text))
                    else:
                        # Show primary (highest confidence) detection
                        best_conf, best_cls, best_bbox = sorted_detections[0]
                        best_conf_val = float(best_conf)
                        best_class_id = int(best_cls)
                        best_expression = results[0].names[best_class_id]
                        
                        expression_placeholder.metric(
                            "Primary Expression", 
                            f"{best_expression}",
                            delta=f"{best_conf_val:.2f}" if show_confidence else None
                        )
                        
                        # Show confidence color coding
                        if best_conf_val > 0.7:
                            confidence_color = "🟢"  # Green for high confidence
                        elif best_conf_val > 0.5:
                            confidence_color = "🟡"  # Yellow for medium confidence
                        else:
                            confidence_color = "🔴"  # Red for low confidence
                        
                        st.sidebar.write(f"**Current Detection:** {confidence_color} {best_expression} ({best_conf_val:.2f})")
                        
                else:
                    detection_placeholder.metric("Faces Detected", 0)
                    expression_placeholder.metric("Primary Expression", "None")
                    st.sidebar.write("**Current Detection:** No faces detected")
                
                # Small delay
                time.sleep(0.05)
                
        except Exception as e:
            st.error(f"Error during camera operation: {str(e)}")
        finally:
            cap.release()
            st.session_state.camera_running = False
            st.info("Camera stopped.")

else:
    st.info("👆 Click 'Start Camera' to begin real-time facial expression detection")
    
    # Show example image or instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **Click 'Start Camera'** to begin real-time detection
    2. **Position your face** clearly in the camera view
    3. **Adjust confidence threshold** in the sidebar if needed
    4. **Ensure good lighting** for better detection accuracy
    
    ### 🎭 Detectable Expressions:
    - 😐 Neutral
    - 😊 Happy  
    - 😢 Sad
    - 😲 Surprise
    - 😨 Fear
    - 🤢 Disgust
    - 😠 Anger
    - 😒 Contempt
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"**Model:** YOLOv8")
st.sidebar.write(f"**Classes:** 8 expressions")
st.sidebar.write(f"**Training mAP@50:** 0.82")
st.sidebar.write(f"**Inference Speed:** ~0.6ms per image")

st.sidebar.markdown("---")
st.sidebar.header("� Troubleshooting")
st.sidebar.write("**If predictions seem off:**")
st.sidebar.write("• Lower confidence threshold to 0.2-0.3")
st.sidebar.write("• Try different lighting conditions")
st.sidebar.write("• Ensure face is clearly visible")
st.sidebar.write("• Check if expressions are exaggerated enough")
st.sidebar.write("• Enable 'Show All Detections' to see alternatives")

st.sidebar.markdown("---")
st.sidebar.header("�💡 Tips")
st.sidebar.write("• Use good lighting")
st.sidebar.write("• Keep face centered")
st.sidebar.write("• Minimize background clutter")
st.sidebar.write("• Try different expressions!")
st.sidebar.write("• Higher confidence = more reliable detection")
