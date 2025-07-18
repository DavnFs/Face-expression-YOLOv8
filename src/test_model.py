import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_model():
    """Test the model with a simple camera capture to debug predictions"""
    
    # Load model
    model_path = os.path.join('..', 'runs', 'detect', 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = 'runs/detect/train/weights/best.pt'
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Print model info
    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Press 'q' to quit, 's' to save a test image")
    print("Testing model predictions...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Test every 10th frame to reduce processing
        if frame_count % 10 == 0:
            # Run inference
            results = model(frame, conf=0.3, verbose=False)
            
            # Print detection results
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                print(f"\n--- Frame {frame_count} ---")
                for i, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = results[0].names[cls]
                    bbox = box.xyxy[0].cpu().numpy()
                    print(f"Detection {i+1}: {class_name} (confidence: {conf:.3f})")
                    print(f"  Bounding box: {bbox}")
            else:
                print(f"Frame {frame_count}: No detections")
        
        # Display frame with annotations
        annotated_frame = results[0].plot() if frame_count % 10 == 0 else frame
        cv2.imshow('Model Test', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame for analysis
            cv2.imwrite(f'test_frame_{frame_count}.jpg', frame)
            print(f"Saved test_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")

if __name__ == "__main__":
    test_model()
