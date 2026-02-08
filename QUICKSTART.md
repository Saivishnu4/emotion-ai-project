# Quick Start Guide

Get up and running with Emotion AI in 5 minutes!

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/emotion-ai-project.git
cd emotion-ai-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Using Python API

```python
from src.models.emotion_model import EmotionClassifier
from src.models.keypoint_model import KeypointDetector
import cv2

# Load models
emotion_model = EmotionClassifier()
emotion_model.compile_model()

# Load and preprocess image
image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (48, 48))

# Predict emotion
result = emotion_model.predict(image)
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Using REST API

```bash
# Start the server
python api/app.py

# In another terminal, make a request
curl -X POST http://localhost:8000/api/v1/predict/emotion \
  -H "Content-Type: application/json" \
  -d '{"image": "YOUR_BASE64_IMAGE"}'
```

### 3. Using Docker

```bash
# Build and run
docker-compose -f deployment/docker/docker-compose.yml up

# Access API at http://localhost:8000
```

## Training Your Own Model

```bash
python src/training/train_emotions.py \
    --data_path data/processed/emotions \
    --epochs 50 \
    --batch_size 64 \
    --output_dir models/my_emotion_model
```

## Next Steps

- Check out the [Jupyter notebooks](notebooks/) for detailed examples
- Read the [full documentation](docs/)
- Explore the [API documentation](docs/api_documentation.md)

## Need Help?

- 📖 [Read the docs](docs/)
- 💬 [Open an issue](https://github.com/yourusername/emotion-ai-project/issues)
- 📧 Email: saivishnusarode@gmail.com
