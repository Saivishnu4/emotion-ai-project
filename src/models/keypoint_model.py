"""
Facial Keypoint Detection Model
Detects 15 facial keypoints from 96x96 grayscale images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .resnet import build_resnet_backbone, ResidualBlock


class KeypointDetector:
    """Facial Keypoint Detection using ResNet"""
    
    def __init__(self, input_shape=(96, 96, 1), num_keypoints=15):
        """
        Initialize Keypoint Detector
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_keypoints: Number of facial keypoints to detect
        """
        self.input_shape = input_shape
        self.num_keypoints = num_keypoints
        self.model = None
    
    def build_model(self):
        """Build the keypoint detection model"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Residual blocks
        x = ResidualBlock(64, stride=2)(x)
        x = ResidualBlock(128, stride=2)(x)
        x = ResidualBlock(256, stride=2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output layer: num_keypoints * 2 (x, y coordinates)
        outputs = layers.Dense(num_keypoints * 2, activation='linear')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='keypoint_detector')
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training keypoints
            X_val: Validation images
            y_val: Validation keypoints
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            History object
        """
        if self.model is None:
            self.compile_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/keypoint_detector/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs/keypoint_detector'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image):
        """
        Predict keypoints for a single image
        
        Args:
            image: Input image (96x96)
            
        Returns:
            Array of keypoint coordinates [(x1, y1), (x2, y2), ...]
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        
        # Reshape to (num_keypoints, 2)
        keypoints = predictions.reshape(-1, self.num_keypoints, 2)
        
        return keypoints[0]
    
    def save(self, filepath):
        """Save the model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model"""
        instance = cls()
        instance.model = keras.models.load_model(filepath)
        return instance


if __name__ == "__main__":
    # Test the model
    detector = KeypointDetector()
    model = detector.build_model()
    model.summary()
    
    # Test prediction
    test_image = np.random.rand(96, 96, 1)
    detector.compile_model()
    keypoints = detector.predict(test_image)
    print(f"\nPredicted keypoints shape: {keypoints.shape}")
    print(f"Keypoints: {keypoints}")
