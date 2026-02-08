"""
Emotion AI - Dual-Model Facial Expression Recognition System
"""

__version__ = "1.0.0"
__author__ = "M Saivishnu Sarode"
__email__ = "saivishnusarode@gmail.com"

from .models.emotion_model import EmotionClassifier
from .models.keypoint_model import KeypointDetector

__all__ = ['EmotionClassifier', 'KeypointDetector']
