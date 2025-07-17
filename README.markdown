# Face-Expression-Detection-YOLOv8

A facial expression detection project using the YOLOv8 framework with the AffectNet dataset, fully implemented in a Jupyter Notebook, including training, evaluation, and Streamlit deployment.

## Overview
This repository contains a machine learning solution for detecting and classifying facial expressions (neutral, happy, sad, surprise, fear, disgust, anger, contempt) using the YOLOv8 object detection model. The project is developed and executed within a single Jupyter Notebook (`facial_expression_detection.ipynb`), trained on the AffectNet dataset with 8 expression classes, and deployed as a web application using Streamlit. The model achieves a mean Average Precision (mAP@50) of 0.82 with an inference time of 0.6ms per image.

## Features
- Detects 8 facial expressions with mAP@50 of 0.82.
- Fast inference time of 0.6ms per image, suitable for real-time applications.
- Complete workflow (data preparation, training, evaluation, and deployment) in a single notebook.
- Includes pre-trained model weights and a deployable Streamlit app.

## Installation

### Prerequisites
- Python 3.8+
- Git
- pip
- Jupyter Notebook (optional for local execution)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/DavnFs/Face-expression-YOLOv8.git
   cd Face-Expression-Detection-YOLOv8
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the AffectNet dataset (YOLO format):
   - Obtain the dataset from [your dataset source] (e.g., Kaggle or a custom link).
   - Extract and place it in the `data/YOLO_format` directory.

4. Download the pre-trained model weights:
   - The `best.pt` file is available in the `models/runs/detect/train/weights/` directory after running the notebook or can be downloaded from the repository releases.
   - Place it in `models/runs/detect/train/weights/`.

## Usage

### Running the Notebook
1. Open the notebook:
   - In Kaggle: Upload the dataset and run `facial_expression_detection.ipynb` directly.
   - Locally: Install Jupyter Notebook (`pip install notebook`), then run:
     ```bash
     jupyter notebook
     ```
     Open `notebooks/facial_expression_detection.ipynb` in your browser.

2. Execute cells sequentially:
   - Cell 1: Install dependencies.
   - Cell 2: Import libraries.
   - Cell 3: Prepare the dataset.
   - Cell 4: Train the model.
   - Cell 5: Evaluate the model.
   - Cell 6: Export the model to ONNX.
   - Cell 7: Save Streamlit code for deployment.

### Deployment
- After running the notebook, a `deploy.py` file will be generated.
- Run the Streamlit app locally:
  ```bash
  streamlit run deploy.py
  ```
  - Open your browser at `http://localhost:8501` to upload images and see detection results.

- **Deploy to Hugging Face Spaces**:
  1. Connect this repository to a Hugging Face Space (see [Hugging Face Spaces](https://huggingface.co/spaces)).
  2. Ensure the `best.pt` file is tracked with Git LFS.
  3. Rebuild the Space to deploy the app at a public URL (e.g., `https://huggingface.co/spaces/[your-username]/Face-Expression-Detection-YOLOv8`).

## Results
- **Overall Metrics**:
  - Precision: 0.723
  - Recall: 0.755
  - mAP@50: 0.82
  - mAP@50:95: 0.82
- **Per Class**:
  - Neutral: mAP@50: 0.827
  - Happy: mAP@50: 0.84
  - Sad: mAP@50: 0.799
  - Surprise: mAP@50: 0.839
  - Fear: mAP@50: 0.951
  - Disgust: mAP@50: 0.702
  - Anger: mAP@50: 0.758
  - Contempt: mAP@50: 0.845
- **Speed**: 0.6ms inference per image

## Project Structure
```
Face-Expression-Detection-YOLOv8/
├── runs/                    # Training and validation results
│   └── detect/
│       ├── train/           # Training outputs
│       │   ├── weights/     # Model weights (best.pt, last.pt, best.onnx)
│       │   └── ...          # Training metrics and visualizations
│       └── val/             # Validation outputs
│           └── ...          # Validation metrics and visualizations
├── src/                     # Source code
│   ├── deploy.py            # Streamlit deployment script
│   └── facial_expression_detection.ipynb  # Main notebook
├── affectnet.yaml           # Dataset configuration
├── yolo11n.pt               # YOLO11 pre-trained weights
├── yolov8n.pt               # YOLOv8 pre-trained weights
├── README.markdown          # This file
└── LICENSE                  # Project license (Apache 2.0)
```

**Note**: The `data/` directory containing the AffectNet dataset is not included in the repository. You can access it on Kaggle: https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss.

## License
This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments
- **Dataset**: AffectNet (https://affectnet-database.org/)
- **Framework**: Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)
- **Deployment**: Streamlit (https://streamlit.io/) and Hugging Face Spaces

