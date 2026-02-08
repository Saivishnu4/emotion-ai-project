"""
Example Training Script for Emotion Classification
"""

import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.emotion_model import EmotionClassifier
from tensorflow.keras.utils import to_categorical


def load_data(data_path):
    """
    Load training data
    Modify this function based on your data format
    """
    # Example: Load from numpy files
    X_train = np.load(f"{data_path}/X_train.npy")
    y_train = np.load(f"{data_path}/y_train.npy")
    X_val = np.load(f"{data_path}/X_val.npy")
    y_val = np.load(f"{data_path}/y_val.npy")
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=5)
    y_val = to_categorical(y_val, num_classes=5)
    
    return X_train, y_train, X_val, y_val


def main(args):
    """Main training function"""
    
    print("=" * 50)
    print("Emotion AI - Training Emotion Classifier")
    print("=" * 50)
    
    # Load data
    print("\n[1/4] Loading data...")
    X_train, y_train, X_val, y_val = load_data(args.data_path)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Image shape: {X_train.shape[1:]}")
    
    # Create model
    print("\n[2/4] Building model...")
    classifier = EmotionClassifier(
        input_shape=X_train.shape[1:],
        num_classes=5,
        use_pretrained=args.pretrained
    )
    
    if args.pretrained:
        classifier.build_with_pretrained()
    else:
        classifier.build_model()
    
    classifier.compile_model(learning_rate=args.learning_rate)
    print("  Model built successfully")
    classifier.model.summary()
    
    # Train model
    print("\n[3/4] Training model...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    print("\n[4/4] Saving model...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classifier.save(output_path / "emotion_model.h5")
    
    # Save training history
    np.save(output_path / "history.npy", history.history)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {output_path}")
    print("=" * 50)
    
    # Print final metrics
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    print(f"Final Validation Loss: {final_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Classification Model")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/emotion_classifier",
        help="Output directory for saved model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pre-trained weights"
    )
    
    args = parser.parse_args()
    main(args)
