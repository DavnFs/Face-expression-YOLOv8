from ultralytics import YOLO
import os

def diagnose_model():
    """Diagnose model and check its configuration"""
    
    # Load model
    model_path = os.path.join('..', 'runs', 'detect', 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = 'runs/detect/train/weights/best.pt'
    
    print(f"🔍 Diagnosing model: {model_path}")
    print(f"📁 Model exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Check model info
        print(f"\n📊 Model Information:")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
        
        # Expected classes based on your affectnet.yaml
        expected_classes = {
            0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 
            4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'
        }
        
        print(f"\n🎯 Expected classes: {expected_classes}")
        
        # Check if classes match
        if model.names == expected_classes:
            print("✅ Model classes match expected classes!")
        else:
            print("⚠️  Model classes don't match expected classes!")
            print("   This might be causing prediction issues.")
        
        # Model architecture info
        print(f"\n🏗️  Model Architecture:")
        print(f"   Model type: {type(model.model).__name__}")
        
        # Check if model has been trained
        if hasattr(model, 'ckpt') and model.ckpt:
            print(f"   Training info available: ✅")
        else:
            print(f"   Training info: ❌")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
    # Check training results
    results_path = os.path.join('..', 'runs', 'detect', 'train', 'results.csv')
    if not os.path.exists(results_path):
        results_path = 'runs/detect/train/results.csv'
    
    if os.path.exists(results_path):
        print(f"\n📈 Training Results Available: {results_path}")
        try:
            with open(results_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) > 7:
                        final_map50 = last_line[7]
                        print(f"   Final mAP@50: {final_map50}")
                        print(f"   Total epochs: {len(lines) - 1}")
        except Exception as e:
            print(f"   Error reading results: {e}")
    
    print(f"\n🔧 Troubleshooting Tips:")
    print(f"   1. Lower confidence threshold to 0.2-0.3")
    print(f"   2. Check if your face expressions are clear enough")
    print(f"   3. Ensure good lighting conditions")
    print(f"   4. Model was trained on AffectNet dataset - expressions should be similar")
    print(f"   5. Try exaggerated expressions first (big smile, wide eyes, etc.)")

if __name__ == "__main__":
    diagnose_model()
