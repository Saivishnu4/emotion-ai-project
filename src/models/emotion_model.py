"""
Emotion Classification Model
Classifies 5 emotions from 48x48 facial images using ResNet
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .resnet import load_pretrained_resnet, ResidualBlock, IdentityBlock


class EmotionClassifier:
    """Emotion Classification using CNN with ResNet"""
    
    EMOTIONS = ['Angry', 'Disgust', 'Sad', 'Happy', 'Surprise']
    
    def __init__(self, input_shape=(48, 48, 1), num_classes=5, use_pretrained=True):
        """
        Initialize Emotion Classifier
        
        Args:
            input_shape: Input image shape
            num_classes: Number of emotion classes
            use_pretrained: Whether to use ImageNet pre-trained weights
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.model = None
    
    def build_model(self):
        """Build the emotion classification model"""
        inputs = keras.Input(shape=self.input_shape)
        
        # If grayscale, convert to 3 channels for pre-trained models
        if self.input_shape[-1] == 1 and self.use_pretrained:
            x = layers.Conv2D(3, 1, padding='same')(inputs)
        else:
            x = inputs
        
        # Convolutional blocks
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual blocks
        x = ResidualBlock(128, stride=2)(x)
        x = IdentityBlock(128)(x)
        x = layers.Dropout(0.3)(x)
        
        x = ResidualBlock(256, stride=2)(x)
        x = IdentityBlock(256)(x)
        x = layers.Dropout(0.4)(x)
        
        x = ResidualBlock(512, stride=2)(x)
        x = IdentityBlock(512)(x)
        x = layers.Dropout(0.5)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='emotion_classifier')
        return self.model
    
    def build_with_pretrained(self):
        """Build model using ImageNet pre-trained ResNet"""
        # Adjust input for pre-trained model
        if self.input_shape[-1] == 1:
            temp_input = (48, 48, 3)
        else:
            temp_input = self.input_shape
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB if needed
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, 1, padding='same')(inputs)
        else:
            x = inputs
        
        # Load pre-trained ResNet
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=temp_input,
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        x = base_model(x)
        
        # Add custom classification head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='emotion_classifier_pretrained')
        self.base_model = base_model
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            if self.use_pretrained:
                self.build_with_pretrained()
            else:
                self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels
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
                monitor='val_accuracy',
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
                'models/emotion_classifier/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs/emotion_classifier'
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
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Fine-tune the pre-trained model"""
        if not self.use_pretrained:
            print("Model is not using pre-trained weights. Use train() instead.")
            return
        
        # Unfreeze base model
        self.base_model.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        return history
    
    def predict(self, image):
        """
        Predict emotion for a single image
        
        Args:
            image: Input image (48x48)
            
        Returns:
            dict: Emotion probabilities
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        
        # Get probabilities
        probabilities = {
            emotion: float(prob) 
            for emotion, prob in zip(self.EMOTIONS, predictions[0])
        }
        
        # Get predicted class
        predicted_class = np.argmax(predictions[0])
        predicted_emotion = self.EMOTIONS[predicted_class]
        confidence = float(predictions[0][predicted_class])
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
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
    classifier = EmotionClassifier(use_pretrained=False)
    model = classifier.build_model()
    model.summary()
    
    # Test prediction
    test_image = np.random.rand(48, 48, 1)
    classifier.compile_model()
    result = classifier.predict(test_image)
    print(f"\nPrediction result: {result}")
