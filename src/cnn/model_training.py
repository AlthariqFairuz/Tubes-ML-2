import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, 
                                   Flatten, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from cnn.cnn_from_scratch import CNNModel

def load_and_preprocess_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                          np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess CIFAR-10 dataset
    Split into train (40k), validation (10k), test (10k)
    """
    # Load CIFAR-10
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    
    # Split training data into train (40k) and validation (10k)
    # Use the first 40k for training and last 10k for validation
    x_train = x_train_full[:40000]
    y_train = y_train_full[:40000]
    x_val = x_train_full[40000:]
    y_val = y_train_full[40000:]
    
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def train_and_evaluate_model(model: tf.keras.Model, model_name: str, 
                           data: Tuple, epochs: int = 20) -> Dict[str, Any]:
    """
    Train and evaluate a single model
    """
    x_train, y_train, x_val, y_val, x_test, y_test = data
    
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    # Display model architecture
    model.summary()
    
    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Get predictions for F1 score
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate macro F1 score
    macro_f1 = f1_score(y_test, y_pred_classes, average='macro')
    
    # Save model
    model.save(f'../../models/cnn_{model_name}.h5')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    return {
        'model': model,
        'history': history.history,
        'test_accuracy': test_acc,
        'macro_f1': macro_f1,
        'model_name': model_name
    }

def plot_training_history(results: Dict[str, Dict[str, Any]], 
                        experiment_type: str, save_path: str = None):
    """
    Plot training and validation loss/accuracy for comparison
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        color = colors[i % len(colors)]
        
        # Training and validation loss
        ax1.plot(history['loss'], color=color, linestyle='-', 
                label=f'{name} (train)', alpha=0.7)
        ax1.plot(history['val_loss'], color=color, linestyle='--', 
                label=f'{name} (val)', alpha=0.7)
        
        # Training and validation accuracy
        ax2.plot(history['accuracy'], color=color, linestyle='-', 
                label=f'{name} (train)', alpha=0.7)
        ax2.plot(history['val_accuracy'], color=color, linestyle='--', 
                label=f'{name} (val)', alpha=0.7)
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1 scores comparison
    names = list(results.keys())
    f1_scores = [results[name]['macro_f1'] for name in names]
    
    ax3.bar(names, f1_scores, color=colors[:len(names)])
    ax3.set_title('Macro F1 Score Comparison')
    ax3.set_ylabel('F1 Score')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, axis='y')
    
    # Test accuracy comparison
    test_accs = [results[name]['test_accuracy'] for name in names]
    
    ax4.bar(names, test_accs, color=colors[:len(names)])
    ax4.set_title('Test Accuracy Comparison')
    ax4.set_ylabel('Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, axis='y')
    
    plt.suptitle(f'CNN Experiment: {experiment_type}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()