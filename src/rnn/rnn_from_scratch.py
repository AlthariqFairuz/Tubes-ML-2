import numpy as np
from typing import List,  Dict

class EmbeddingLayer:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = None
        
    def load_weights(self, weights: np.ndarray):
        self.weights = weights
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args: inputs (batch_size, sequence_length) - token indices
        Returns: (batch_size, sequence_length, embedding_dim)
        """
        if self.weights is None:
            raise ValueError("Weights not loaded")
            
        batch_size, seq_length = inputs.shape
        outputs = np.zeros((batch_size, seq_length, self.embedding_dim))
        
        for i in range(batch_size):
            for j in range(seq_length):
                token_idx = int(inputs[i, j])
                if 0 <= token_idx < self.vocab_size:
                    outputs[i, j] = self.weights[token_idx]
                    
        return outputs


class SimpleRNNCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = None  # input to hidden
        self.W_hh = None  # hidden to hidden
        self.b_h = None   # bias
        
    def load_weights(self, W_ih: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_h = b_h
        
    def forward_step(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
        Returns: h_new: (batch_size, hidden_size)
        """
        if self.W_ih is None:
            raise ValueError("Weights not loaded")
            
        linear = np.dot(x, self.W_ih) + np.dot(h_prev, self.W_hh) + self.b_h
        h_new = np.tanh(linear)
        return h_new


class SimpleRNNLayer:
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.forward_cell = SimpleRNNCell(input_size, hidden_size)
        if bidirectional:
            self.backward_cell = SimpleRNNCell(input_size, hidden_size)
            
    def load_weights(self, weights_dict: Dict[str, np.ndarray]):
        self.forward_cell.load_weights(
            weights_dict['forward_W_ih'],
            weights_dict['forward_W_hh'], 
            weights_dict['forward_b_h']
        )
        
        if self.bidirectional:
            self.backward_cell.load_weights(
                weights_dict['backward_W_ih'],
                weights_dict['backward_W_hh'],
                weights_dict['backward_b_h']
            )
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args: inputs (batch_size, sequence_length, input_size)
        Returns: (batch_size, sequence_length, hidden_size * num_directions)
        """
        batch_size, seq_length, _ = inputs.shape
        
        # Forward direction
        forward_outputs = []
        h_forward = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            h_forward = self.forward_cell.forward_step(inputs[:, t, :], h_forward)
            forward_outputs.append(h_forward)
            
        forward_outputs = np.stack(forward_outputs, axis=1)
        
        if not self.bidirectional:
            return forward_outputs
            
        # Backward direction
        backward_outputs = []
        h_backward = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length - 1, -1, -1):
            h_backward = self.backward_cell.forward_step(inputs[:, t, :], h_backward)
            backward_outputs.append(h_backward)
            
        backward_outputs = np.stack(backward_outputs[::-1], axis=1)
        
        # Concatenate forward and backward
        outputs = np.concatenate([forward_outputs, backward_outputs], axis=-1)
        return outputs


class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'linear'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = None
        self.bias = None
        
    def load_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights not loaded")
            
        # Handle 2D and 3D inputs
        original_shape = inputs.shape
        if len(original_shape) == 3:
            batch_size, seq_length, input_size = original_shape
            inputs = inputs.reshape(-1, input_size)
            
        # Linear transformation
        outputs = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation
        if self.activation == 'softmax':
            outputs = self._softmax(outputs)
        elif self.activation == 'relu':
            outputs = np.maximum(0, outputs)
        elif self.activation == 'sigmoid':
            outputs = 1 / (1 + np.exp(-np.clip(outputs, -250, 250)))
        
        # Reshape back if needed
        if len(original_shape) == 3:
            outputs = outputs.reshape(batch_size, seq_length, self.output_size)
            
        return outputs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SimpleRNNModel:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_sizes: List[int], 
                 num_classes: int, bidirectional: bool = False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Initialize layers
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.rnn_layers = []
        
        # Create RNN layers
        input_size = embedding_dim
        for hidden_size in hidden_sizes:
            rnn_layer = SimpleRNNLayer(input_size, hidden_size, bidirectional)
            self.rnn_layers.append(rnn_layer)
            input_size = hidden_size * (2 if bidirectional else 1)
            
        self.dense = DenseLayer(input_size, num_classes, activation='softmax')
        
    def load_keras_weights(self, keras_model_path: str):
        """Load weights from Keras .h5 model"""
        import tensorflow as tf
        
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        # Load embedding weights
        embedding_weights = keras_model.get_layer('embedding').get_weights()[0]
        self.embedding.load_weights(embedding_weights)
        
        # Load RNN weights
        rnn_layer_idx = 0
        for layer in keras_model.layers:
            if ('rnn' in layer.name.lower() or 'bidirectional' in layer.name.lower()) and rnn_layer_idx < len(self.rnn_layers):
                weights = layer.get_weights()
                
                if 'bidirectional' in layer.name.lower():
                    weights_dict = {
                        'forward_W_ih': weights[0],
                        'forward_W_hh': weights[1], 
                        'forward_b_h': weights[2],
                        'backward_W_ih': weights[3],
                        'backward_W_hh': weights[4],
                        'backward_b_h': weights[5]
                    }
                else:
                    weights_dict = {
                        'forward_W_ih': weights[0],
                        'forward_W_hh': weights[1],
                        'forward_b_h': weights[2]
                    }
                
                self.rnn_layers[rnn_layer_idx].load_weights(weights_dict)
                rnn_layer_idx += 1
        
        # Load dense weights
        dense_weights = keras_model.get_layer('dense').get_weights()
        self.dense.load_weights(dense_weights[0], dense_weights[1])
        
        print(f"Weights loaded from {keras_model_path}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args: inputs (batch_size, sequence_length) - token indices
        Returns: (batch_size, num_classes) - class probabilities
        """
        # Embedding
        x = self.embedding.forward(inputs)
        
        # RNN layers
        for rnn_layer in self.rnn_layers:
            x = rnn_layer.forward(x)
            
        # Use last timestep for classification
        x = x[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        # Dense layer
        outputs = self.dense.forward(x)
        
        return outputs
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=-1)
    
    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.forward(inputs)


# # Testing function
# def test_rnn_implementation():
#     """Quick test of the implementation"""
#     # Create model
#     model = SimpleRNNModel(
#         vocab_size=1000,
#         embedding_dim=64,
#         hidden_sizes=[32, 16],
#         num_classes=3,
#         bidirectional=False
#     )
    
#     # Create dummy weights for testing
#     np.random.seed(42)
    
#     # Embedding weights
#     embedding_weights = np.random.randn(1000, 64) * 0.1
#     model.embedding.load_weights(embedding_weights)
    
#     # RNN weights for each layer
#     for i, layer in enumerate(model.rnn_layers):
#         input_size = 64 if i == 0 else 32
#         hidden_size = model.hidden_sizes[i]
        
#         W_ih = np.random.randn(input_size, hidden_size) * 0.1
#         W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
#         b_h = np.zeros(hidden_size)
        
#         layer.load_weights({
#             'forward_W_ih': W_ih,
#             'forward_W_hh': W_hh,
#             'forward_b_h': b_h
#         })
    
#     # Dense weights
#     dense_weights = np.random.randn(16, 3) * 0.1
#     dense_bias = np.zeros(3)
#     model.dense.load_weights(dense_weights, dense_bias)
    
#     # Test forward pass
#     test_input = np.random.randint(0, 1000, (5, 20))  # batch=5, seq_len=20
#     output = model.forward(test_input)
#     predictions = model.predict(test_input)
    
#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Predictions: {predictions}")
#     print("✅ RNN from scratch test passed!")


# if __name__ == "__main__":
#     test_rnn_implementation()