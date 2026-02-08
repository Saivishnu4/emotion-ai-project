# Emotion AI - Dual-Model Facial Expression Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready deep learning system for real-time emotion detection and facial keypoint prediction using ResNet architecture with TensorFlow Serving.

## 🎯 Overview

This project implements a dual-model Emotion AI system that:
- **Detects 15 facial keypoints** from 96x96 grayscale images
- **Classifies 5 emotion categories**: Angry, Disgust, Sad, Happy, Surprise from 48x48 pixel facial images
- Uses **ResNet architecture** with residual blocks and skip connections
- Provides **REST API endpoints** for real-time inference
- Achieves production-ready deployment with **TensorFlow Serving**

## 🏗️ Architecture

### Model 1: Facial Keypoint Detection
- **Input**: 96x96 grayscale images
- **Output**: 15 facial keypoint coordinates
- **Architecture**: ResNet with residual blocks
- **Training Data**: 2,000+ annotated images

### Model 2: Emotion Classification
- **Input**: 48x48 pixel facial images
- **Output**: 5 emotion categories (Angry, Disgust, Sad, Happy, Surprise)
- **Architecture**: CNN with convolutional blocks, identity blocks, batch normalization, max pooling, dropout
- **Training Data**: 20,000+ labeled facial expression images
- **Base Model**: ImageNet-pretrained ResNet

### Key Features
- ✅ Skip connections to prevent vanishing gradient issues
- ✅ Batch normalization for stable training
- ✅ Dropout layers for regularization
- ✅ Max pooling for feature extraction
- ✅ Transfer learning from ImageNet-pretrained models

## 📁 Project Structure

```
emotion-ai-project/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Preprocessed data
│   └── README.md              # Data documentation
│
├── models/                    # Trained models
│   ├── keypoint_detector/     # Facial keypoint model
│   ├── emotion_classifier/    # Emotion classification model
│   └── README.md              # Model documentation
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data processing
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   ├── keypoint_model.py
│   │   └── emotion_model.py
│   ├── training/              # Training scripts
│   │   ├── __init__.py
│   │   ├── train_keypoints.py
│   │   └── train_emotions.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
│
├── api/                       # REST API
│   ├── __init__.py
│   ├── app.py                 # Flask/FastAPI application
│   ├── inference.py           # Inference logic
│   └── schemas.py             # API schemas
│
├── deployment/                # Deployment configs
│   ├── tensorflow_serving/    # TF Serving configs
│   │   ├── model_config.conf
│   │   └── batching_config.txt
│   ├── docker/                # Docker files
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/            # K8s manifests
│       └── deployment.yaml
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── api_documentation.md
│   └── deployment_guide.md
│
└── assets/                    # Images, diagrams
    └── demo/
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
CUDA 11.x (for GPU support)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/emotion-ai-project.git
cd emotion-ai-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models** (if available)
```bash
python scripts/download_models.py
```

## 💻 Usage

### Training Models

**Train Keypoint Detection Model:**
```bash
python src/training/train_keypoints.py \
    --data_path data/processed/keypoints \
    --epochs 100 \
    --batch_size 32 \
    --output_dir models/keypoint_detector
```

**Train Emotion Classification Model:**
```bash
python src/training/train_emotions.py \
    --data_path data/processed/emotions \
    --epochs 50 \
    --batch_size 64 \
    --pretrained imagenet \
    --output_dir models/emotion_classifier
```

### Running Inference

**Python API:**
```python
from src.models.keypoint_model import KeypointDetector
from src.models.emotion_model import EmotionClassifier

# Load models
keypoint_model = KeypointDetector.load('models/keypoint_detector')
emotion_model = EmotionClassifier.load('models/emotion_classifier')

# Predict
import cv2
image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

keypoints = keypoint_model.predict(image)
emotion = emotion_model.predict(image)

print(f"Detected emotion: {emotion}")
print(f"Facial keypoints: {keypoints}")
```

**REST API:**
```bash
# Start the API server
python api/app.py

# Make prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

### TensorFlow Serving Deployment

```bash
# Start TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/models,target=/models \
  -e MODEL_NAME=emotion_ai \
  -t tensorflow/serving

# Make inference request
curl -X POST http://localhost:8501/v1/models/emotion_ai:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"input": [[...image_data...]]}]}'
```

## 📊 Model Performance

### Keypoint Detection Model
- **Training Images**: 2,000+
- **Input Size**: 96x96 grayscale
- **Output**: 15 keypoint coordinates
- **Architecture**: ResNet with residual blocks

### Emotion Classification Model
- **Training Images**: 20,000+
- **Input Size**: 48x48 pixels
- **Classes**: 5 (Angry, Disgust, Sad, Happy, Surprise)
- **Architecture**: CNN with ImageNet-pretrained ResNet
- **Accuracy**: [Add your accuracy metrics]

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Model Architecture**: ResNet, CNN
- **API**: Flask/FastAPI
- **Serving**: TensorFlow Serving
- **Deployment**: Docker, Kubernetes (optional)
- **Data Processing**: NumPy, OpenCV, Pandas
- **Visualization**: Matplotlib, Seaborn

## 📈 Training Details

### Data Augmentation
- Random rotation (±15°)
- Horizontal flipping
- Random brightness adjustment
- Gaussian noise addition

### Training Strategy
- **Optimizer**: Adam
- **Loss Functions**: 
  - Keypoint Detection: MSE
  - Emotion Classification: Categorical Crossentropy
- **Learning Rate**: Adaptive with ReduceLROnPlateau
- **Early Stopping**: Patience of 10 epochs
- **Batch Normalization**: Applied after each conv layer
- **Dropout**: 0.3-0.5 for regularization

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t emotion-ai:latest -f deployment/docker/Dockerfile .

# Run container
docker-compose -f deployment/docker/docker-compose.yml up
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# With coverage
pytest --cov=src tests/
```

## 📝 API Documentation

### Endpoints

#### 1. Predict Emotion
```
POST /api/v1/predict/emotion
```
**Request:**
```json
{
  "image": "base64_encoded_image",
  "model_version": "1"
}
```
**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.95,
  "probabilities": {
    "Angry": 0.01,
    "Disgust": 0.01,
    "Sad": 0.02,
    "Happy": 0.95,
    "Surprise": 0.01
  }
}
```

#### 2. Detect Keypoints
```
POST /api/v1/predict/keypoints
```
**Request:**
```json
{
  "image": "base64_encoded_image",
  "model_version": "1"
}
```
**Response:**
```json
{
  "keypoints": [
    {"x": 45.2, "y": 32.1},
    {"x": 54.8, "y": 31.9},
    ...
  ],
  "num_keypoints": 15
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **M Saivishnu Sarode** - [LinkedIn](https://www.linkedin.com/in/saivishnu2002) | [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- ImageNet pre-trained models
- TensorFlow and Keras teams
- Open-source community

## 📧 Contact

For questions or feedback, please reach out:
- Email: saivishnusarode@gmail.com
- LinkedIn: [linkedin.com/in/saivishnu2002](https://www.linkedin.com/in/saivishnu2002)

## 🔮 Future Enhancements

- [ ] Add support for real-time video emotion detection
- [ ] Implement multi-face detection
- [ ] Add more emotion categories
- [ ] Mobile deployment (TFLite)
- [ ] Web-based demo interface
- [ ] Model quantization for edge deployment
- [ ] A/B testing framework for model versions

---

⭐ If you find this project useful, please consider giving it a star!
