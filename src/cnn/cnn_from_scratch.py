import numpy as np
from typing import List, Tuple, Dict, Any
import tensorflow as tf

class Conv2DLayer:
    def __init__(self, filters: int, kernel_size: Tuple[int, int], 
                 stride: Tuple[int, int] = (1, 1), padding: str = 'valid', 
                 activation: str = 'relu'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.weights = None
        self.bias = None
        
    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        """Load weights from Keras model"""
        self.weights = weights  # Shape: (kernel_h, kernel_w, input_channels, output_channels)
        self.bias = bias        # Shape: (output_channels,)
        
    def _pad_input(self, inputs: np.ndarray) -> np.ndarray:
        """Apply padding to input"""
        if self.padding == 'valid':
            return inputs
        elif self.padding == 'same':
            batch_size, height, width, channels = inputs.shape
            kernel_h, kernel_w = self.kernel_size
            
            # Calculate padding
            pad_h = max(0, kernel_h - 1)
            pad_w = max(0, kernel_w - 1)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Apply padding
            padded = np.pad(inputs, 
                          ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=0)
            return padded
        else:
            return inputs
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:  # linear
            return x
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for Conv2D layer
        Args: inputs (batch_size, height, width, channels)
        Returns: (batch_size, out_height, out_width, filters)
        """
        if self.weights is None:
            raise ValueError("Weights not loaded")
        
        # Pad input
        padded_inputs = self._pad_input(inputs)
        batch_size, input_h, input_w, input_c = padded_inputs.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        
        # Calculate output dimensions
        out_h = (input_h - kernel_h) // stride_h + 1
        out_w = (input_w - kernel_w) // stride_w + 1
        
        # Initialize output
        outputs = np.zeros((batch_size, out_h, out_w, self.filters))
        
        # Convolution operation
        for b in range(batch_size):
            for f in range(self.filters):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract region
                        start_i = i * stride_h
                        start_j = j * stride_w
                        region = padded_inputs[b, start_i:start_i+kernel_h, 
                                            start_j:start_j+kernel_w, :]
                        
                        # Convolution
                        conv_sum = np.sum(region * self.weights[:, :, :, f])
                        outputs[b, i, j, f] = conv_sum + self.bias[f]
        
        # Apply activation
        return self._apply_activation(outputs)


class PoolingLayer:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), 
                 stride: Tuple[int, int] = (2, 2), 
                 pool_type: str = 'max'):
        self.pool_size = pool_size
        self.stride = stride
        self.pool_type = pool_type
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for pooling layer
        Args: inputs (batch_size, height, width, channels)
        Returns: (batch_size, out_height, out_width, channels)
        """
        batch_size, input_h, input_w, channels = inputs.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        
        # Calculate output dimensions
        out_h = (input_h - pool_h) // stride_h + 1
        out_w = (input_w - pool_w) // stride_w + 1
        
        # Initialize output
        outputs = np.zeros((batch_size, out_h, out_w, channels))
        
        # Pooling operation
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract region
                        start_i = i * stride_h
                        start_j = j * stride_w
                        region = inputs[b, start_i:start_i+pool_h, 
                                      start_j:start_j+pool_w, c]
                        
                        # Apply pooling
                        if self.pool_type == 'max':
                            outputs[b, i, j, c] = np.max(region)
                        elif self.pool_type == 'average':
                            outputs[b, i, j, c] = np.mean(region)
                        
        return outputs


class FlattenLayer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Flatten layer
        Args: inputs (batch_size, height, width, channels)
        Returns: (batch_size, height*width*channels)
        """
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)


class DenseLayer:
    def __init__(self, units: int, activation: str = 'linear'):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        
    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        """Load weights from Keras model"""
        self.weights = weights
        self.bias = bias
        
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation == 'softmax':
            x_shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:  # linear
            return x
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for dense layer
        Args: inputs (batch_size, input_features)
        Returns: (batch_size, units)
        """
        if self.weights is None:
            raise ValueError("Weights not loaded")
            
        outputs = np.dot(inputs, self.weights) + self.bias
        return self._apply_activation(outputs)


class CNNModel:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        CNN Model for image classification
        Args:
            input_shape: (height, width, channels)
            num_classes: number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        
    def add_conv2d(self, filters: int, kernel_size: Tuple[int, int], 
                   stride: Tuple[int, int] = (1, 1), padding: str = 'valid', 
                   activation: str = 'relu'):
        """Add Conv2D layer"""
        layer = Conv2DLayer(filters, kernel_size, stride, padding, activation)
        self.layers.append(layer)
        return self
        
    def add_pooling(self, pool_size: Tuple[int, int] = (2, 2), 
                    stride: Tuple[int, int] = (2, 2), pool_type: str = 'max'):
        """Add pooling layer"""
        layer = PoolingLayer(pool_size, stride, pool_type)
        self.layers.append(layer)
        return self
        
    def add_flatten(self):
        """Add flatten layer"""
        layer = FlattenLayer()
        self.layers.append(layer)
        return self
        
    def add_dense(self, units: int, activation: str = 'linear'):
        """Add dense layer"""
        layer = DenseLayer(units, activation)
        self.layers.append(layer)
        return self
        
    def load_keras_weights(self, keras_model_path: str):
        """Load weights from Keras .h5 model"""
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        conv_idx = 0
        dense_idx = 0
        
        for layer in self.layers:
            if isinstance(layer, Conv2DLayer):
                # Find corresponding conv layer in Keras model
                keras_layer = None
                for k_layer in keras_model.layers:
                    if 'conv' in k_layer.name.lower():
                        if conv_idx == 0:
                            keras_layer = k_layer
                            conv_idx += 1
                            break
                        conv_idx -= 1
                        
                if keras_layer:
                    weights = keras_layer.get_weights()
                    layer.load_weights(weights[0], weights[1])
                    
            elif isinstance(layer, DenseLayer):
                # Find corresponding dense layer in Keras model
                keras_layer = None
                for k_layer in keras_model.layers:
                    if 'dense' in k_layer.name.lower():
                        if dense_idx == 0:
                            keras_layer = k_layer
                            dense_idx += 1
                            break
                        dense_idx -= 1
                        
                if keras_layer:
                    weights = keras_layer.get_weights()
                    layer.load_weights(weights[0], weights[1])
        
        print(f"Weights loaded from {keras_model_path}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire model
        Args: inputs (batch_size, height, width, channels)
        Returns: (batch_size, num_classes)
        """
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=-1)
        
    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.forward(inputs)
        
    def summary(self):
        """Print model summary"""
        print("CNN Model Summary:")
        print("=" * 50)
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            if isinstance(layer, Conv2DLayer):
                print(f"Layer {i}: {layer_type} - Filters: {layer.filters}, "
                      f"Kernel: {layer.kernel_size}, Activation: {layer.activation}")
            elif isinstance(layer, PoolingLayer):
                print(f"Layer {i}: {layer_type} - Pool Size: {layer.pool_size}, "
                      f"Type: {layer.pool_type}")
            elif isinstance(layer, DenseLayer):
                print(f"Layer {i}: {layer_type} - Units: {layer.units}, "
                      f"Activation: {layer.activation}")
            else:
                print(f"Layer {i}: {layer_type}")
        print("=" * 50)