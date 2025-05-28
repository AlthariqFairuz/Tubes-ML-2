import numpy as np
from typing import List, Dict

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


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = None  # input to hidden
        self.W_hh = None  # hidden to hidden
        self.b_ih = None  # input bias
        self.b_hh = None  # hidden bias
        
    def load_weights(self, W_ih: np.ndarray, W_hh: np.ndarray, b_ih: np.ndarray, b_hh: np.ndarray):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
        
    def forward_step(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple:
        """
        Args:
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)
        Returns: (h_new, c_new): (batch_size, hidden_size), (batch_size, hidden_size)
        """
        if self.W_ih is None:
            raise ValueError("Weights not loaded")
            
        batch_size = x.shape[0]
        
        # Calculate gates: input, forget, cell, output
        gates = np.dot(x, self.W_ih) + self.b_ih + np.dot(h_prev, self.W_hh) + self.b_hh
        
        # Split into 4 parts for i, f, g, o gates
        i, f, g, o = np.split(gates, 4, axis=1)
        
        # activations
        i = 1 / (1 + np.exp(-i))  # input gate: sigmoid
        f = 1 / (1 + np.exp(-f))  # forget gate: sigmoid
        g = np.tanh(g)            # cell gate: tanh
        o = 1 / (1 + np.exp(-o))  # output gate: sigmoid
        
        # Update cell and hidden state
        c_new = f * c_prev + i * g
        h_new = o * np.tanh(c_new)
        
        return h_new, c_new


class LSTMLayer:
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.forward_cell = LSTMCell(input_size, hidden_size)
        if bidirectional:
            self.backward_cell = LSTMCell(input_size, hidden_size)
            
    def load_weights(self, weights_dict: Dict[str, np.ndarray]):
        self.forward_cell.load_weights(
            weights_dict['forward_W_ih'],
            weights_dict['forward_W_hh'], 
            weights_dict['forward_b_ih'],
            weights_dict['forward_b_hh']
        )
        
        if self.bidirectional:
            self.backward_cell.load_weights(
                weights_dict['backward_W_ih'],
                weights_dict['backward_W_hh'],
                weights_dict['backward_b_ih'],
                weights_dict['backward_b_hh']
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
        c_forward = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            h_forward, c_forward = self.forward_cell.forward_step(inputs[:, t, :], h_forward, c_forward)
            forward_outputs.append(h_forward)
            
        forward_outputs = np.stack(forward_outputs, axis=1)
        
        if not self.bidirectional:
            return forward_outputs
            
        # Backward direction
        backward_outputs = []
        h_backward = np.zeros((batch_size, self.hidden_size))
        c_backward = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length - 1, -1, -1):
            h_backward, c_backward = self.backward_cell.forward_step(inputs[:, t, :], h_backward, c_backward)
            backward_outputs.append(h_backward)
            
        backward_outputs = np.stack(backward_outputs[::-1], axis=1)
        
        # Concatenate forward and backward (default keras behavior is to concat)
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
            
        original_shape = inputs.shape
        if len(original_shape) == 3:
            batch_size, seq_length, input_size = original_shape
            inputs = inputs.reshape(-1, input_size)
            
        outputs = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'softmax':
            outputs = self._softmax(outputs)
        elif self.activation == 'relu':
            outputs = np.maximum(0, outputs)
        elif self.activation == 'sigmoid':
            outputs = 1 / (1 + np.exp(-(outputs)))
        
        if len(original_shape) == 3:
            outputs = outputs.reshape(batch_size, seq_length, self.output_size)
            
        return outputs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class LSTMModel:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_sizes: List[int], 
                 num_classes: int, bidirectional: bool = False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.lstm_layers = []
        
        input_size = embedding_dim
        for hidden_size in hidden_sizes:
            lstm_layer = LSTMLayer(input_size, hidden_size, bidirectional)
            self.lstm_layers.append(lstm_layer)
            input_size = hidden_size * (2 if bidirectional else 1)
            
        self.dense = DenseLayer(input_size, num_classes, activation='softmax')
        
    def load_keras_weights(self, keras_model_path: str):
        """Load weights from Keras .h5 model"""
        import tensorflow as tf
        
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        embedding_weights = keras_model.get_layer('embedding').get_weights()[0]
        self.embedding.load_weights(embedding_weights)
        
        lstm_layer_idx = 0
        for layer in keras_model.layers:
            if ('lstm' in layer.name.lower() or 'bidirectional' in layer.name.lower()) and lstm_layer_idx < len(self.lstm_layers):
                weights = layer.get_weights()
                hidden_size = self.hidden_sizes[lstm_layer_idx]
                
                if 'bidirectional' in layer.name.lower():
                    # For bidirectional layers with 6 weight arrays:
                    # [fw_W_ih, fw_W_hh, fw_bias, bw_W_ih, bw_W_hh, bw_bias]
                    if len(weights) == 6:
                        fw_W_ih, fw_W_hh, fw_bias = weights[0], weights[1], weights[2]
                        bw_W_ih, bw_W_hh, bw_bias = weights[3], weights[4], weights[5]
                        
                        # in keras, the bias is stored as one vector containing all gate biases
                        # i use the full bias as b_ih and set b_hh to zeros
                        # because keras combines what PyTorch separates into b_ih and b_hh

                        weights_dict = {
                            'forward_W_ih': fw_W_ih,
                            'forward_W_hh': fw_W_hh,
                            'forward_b_ih': fw_bias,
                            'forward_b_hh': np.zeros_like(fw_bias),
                            'backward_W_ih': bw_W_ih,
                            'backward_W_hh': bw_W_hh,
                            'backward_b_ih': bw_bias,  
                            'backward_b_hh': np.zeros_like(bw_bias)  
                        }
                    else:
                        raise ValueError(f"Unexpected bidirectional weight structure with {len(weights)} arrays")
                else:
                    # For unidirectional LSTM layers
                    # [W_ih, W_hh, bias]
                    if len(weights) >= 3:
                        weights_dict = {
                            'forward_W_ih': weights[0],
                            'forward_W_hh': weights[1],
                            'forward_b_ih': weights[2],  
                            'forward_b_hh': np.zeros_like(weights[2]) 
                        }
                    else:
                        raise ValueError(f"Unexpected unidirectional weight structure with {len(weights)} arrays")
                
                self.lstm_layers[lstm_layer_idx].load_weights(weights_dict)
                lstm_layer_idx += 1
        
        # Load dense weights
        for layer in keras_model.layers:
            if 'dense' in layer.name.lower():
                dense_weights = layer.get_weights()
                self.dense.load_weights(dense_weights[0], dense_weights[1])
                break
        
        print(f"Weights loaded from {keras_model_path}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Args: inputs (batch_size, sequence_length) - token indices
        Returns: (batch_size, num_classes) - class probabilities
        """
        # Embedding
        x = self.embedding.forward(inputs)
        
        # LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer.forward(x)
            
        # Get the last hidden state of the forward pass (from the last time step)
        forward_state = x[:, -1, :self.hidden_sizes[-1]]
        
        # Get the last hidden state of the backward pass (from the first time step)
        backward_state = x[:, 0, self.hidden_sizes[-1]:] if self.bidirectional else None
        
        # Concatenate to get the final bidirectional state
        if self.bidirectional:
            x = np.concatenate([forward_state, backward_state], axis=-1)
        else:
            x = forward_state
        
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

