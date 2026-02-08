"""
ResNet Architecture Implementation for Emotion AI
Implements residual blocks with skip connections
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    """Residual block with skip connections"""
    
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Main path
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Skip connection
        if stride != 1:
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, strides=stride, padding='same'),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
        self.relu2 = layers.ReLU()
    
    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        shortcut = self.shortcut(inputs)
        
        # Add and activate
        x = layers.add([x, shortcut])
        x = self.relu2(x)
        
        return x


class IdentityBlock(layers.Layer):
    """Identity block - residual block with identity shortcut"""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Add skip connection
        x = layers.add([x, inputs])
        x = self.relu2(x)
        
        return x


def build_resnet_backbone(input_shape, num_residual_blocks=3):
    """
    Build ResNet backbone with residual blocks
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_residual_blocks: Number of residual blocks per stage
        
    Returns:
        keras.Model: ResNet backbone model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Stage 1: 64 filters
    for _ in range(num_residual_blocks):
        x = ResidualBlock(64)(x)
    
    # Stage 2: 128 filters
    x = ResidualBlock(128, stride=2)(x)
    for _ in range(num_residual_blocks - 1):
        x = IdentityBlock(128)(x)
    
    # Stage 3: 256 filters
    x = ResidualBlock(256, stride=2)(x)
    for _ in range(num_residual_blocks - 1):
        x = IdentityBlock(256)(x)
    
    # Stage 4: 512 filters
    x = ResidualBlock(512, stride=2)(x)
    for _ in range(num_residual_blocks - 1):
        x = IdentityBlock(512)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='resnet_backbone')
    return model


def load_pretrained_resnet(input_shape, weights='imagenet'):
    """
    Load ImageNet pre-trained ResNet model
    
    Args:
        input_shape: Input image shape
        weights: 'imagenet' or None
        
    Returns:
        keras.Model: Pre-trained ResNet model
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='avg'
    )
    
    return base_model


if __name__ == "__main__":
    # Test the model
    model = build_resnet_backbone((96, 96, 1))
    model.summary()
    
    print("\nPre-trained ResNet:")
    pretrained = load_pretrained_resnet((96, 96, 3), weights=None)
    pretrained.summary()
