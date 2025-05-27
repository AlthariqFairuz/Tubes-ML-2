import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Bidirectional, Dense, Dropout
from sklearn.metrics import f1_score

import sys
sys.path.append('../../src/rnn')

def create_keras_rnn(vocab_size, num_classes, config):
    """Create Keras RNN model"""
    model = Sequential([Embedding(vocab_size, 128, input_length=100, name='embedding')])
    
    # Add RNN layers
    for i, units in enumerate(config['hidden_sizes']):
        return_sequences = (i < len(config['hidden_sizes']) - 1)
        
        if config['bidirectional']:
            layer = Bidirectional(SimpleRNN(units, return_sequences=return_sequences), 
                                name=f'bidirectional_rnn_{i}')
        else:
            layer = SimpleRNN(units, return_sequences=return_sequences, name=f'simple_rnn_{i}')
        
        model.add(layer)
        model.add(Dropout(0.2, name=f'dropout_{i}'))
    
    model.add(Dense(num_classes, activation='softmax', name='dense'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
def train_and_save_models(data, configs):
    """Train models dan save"""

    results = {}
    
    for name, config in configs.items():
        print(f"\nTraining {name}...")
        
        # Create and train model
        model = create_keras_rnn(data['vocab_size'], data['num_classes'], config)
        
        history = model.fit(
            data['train_X'], data['train_y'],
            validation_data=(data['val_X'], data['val_y']),
            epochs=10, batch_size=32, verbose=1
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(data['test_X']), axis=1)
        macro_f1 = f1_score(data['test_y'], y_pred, average='macro')
        
        # Save model
        model.save(f'../../models/{name}.h5')
        
        results[name] = {
            'config': config,
            'macro_f1': macro_f1,
            'history': history.history
        }
        
        print(f"Macro F1-score: {macro_f1:.4f}")
    
    return results
