# Tugas Besar Pembelajaran Mesing 2 - Feed Forward Implementation from Scratch

## Project Overview

This project implements three fundamental deep learning architectures **completely from scratch** using only NumPy, and compares their performance with TensorFlow/Keras implementations. The goal is to deeply understand how these neural networks work at the mathematical level while conducting systematic experiments on their hyperparameters.

### What We're Building

**Convolutional Neural Network (CNN)** - For image classification using the CIFAR-10 dataset, exploring how different architectural choices affect performance on visual recognition tasks.

**Simple Recurrent Neural Network (RNN)** - For sentiment analysis using the NusaX-Sentiment dataset (Indonesian text), investigating how sequence processing capabilities vary with different configurations.

**Long Short-Term Memory (LSTM)** - Also for sentiment analysis, examining how this advanced RNN variant handles long-term dependencies in text data.

### The Research Questions We're Answering

Each implementation explores specific research questions through controlled experiments:

**For CNN:**
- How does the depth of the network (number of convolutional layers) affect learning?
- What impact does the number of filters per layer have on feature extraction?
- How do different filter sizes change the model's ability to recognize patterns?
- Does the choice between max pooling and average pooling matter for performance?

**For RNN:**
- How does stacking multiple RNN layers affect sequence modeling capability?
- What's the relationship between the number of hidden units and model performance?
- How much better are bidirectional RNNs compared to unidirectional ones?

**For LSTM:**
- Do the same architectural principles that apply to simple RNNs also work for LSTMs?
- How do LSTMs compare to simple RNNs on the same tasks?
- What's the trade-off between model complexity and performance improvement?

## Project Structure

Understanding the codebase organization will help you navigate and extend the project:

```
├── src/
│   ├── utils/
│   │   └── data_preprocessing.py     # Text preprocessing utilities
│   ├── cnn/
│   │   ├── cnn_from_scratch.py       # CNN implementation from scratch
│   │   ├── model_training.py         # CNN training and evaluation
│   │   └── cnn.ipynb                 # CNN experiments notebook
│   ├── rnn/
│   │   ├── rnn_from_scratch.py       # RNN implementation from scratch
│   │   ├── model_training.py         # RNN training utilities
│   │   └── rnn.ipynb                 # RNN experiments notebook
│   └── lstm/
│       ├── lstm_from_scratch.py      # LSTM implementation from scratch
│       ├── model_training.py         # LSTM training utilities
│       └── lstm.ipynb                # LSTM experiments notebook
├── datasets/
│   ├── train.csv                     # Training data for text classification
│   ├── valid.csv                     # Validation data for text classification
│   └── test.csv                      # Test data for text classification
├── models/                           # Saved trained models
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup and Installation

### Prerequisites

You'll need Python 3.8 or higher with the following key libraries. The implementation philosophy is to use minimal dependencies - our from-scratch implementations rely only on NumPy for mathematical operations.

### Installation Steps

1. **Clone the repository and navigate to the project directory:**
```bash
git clone <repository-url>
cd Tubes-ML-2
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv ml_project_env
source ml_project_env/bin/activate  # On Windows: ml_project_env\Scripts\activate
```

3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

The main dependencies are:
- **TensorFlow**: For Keras baseline models and CIFAR-10 dataset loading
- **NumPy**: The foundation for all our from-scratch implementations
- **Pandas**: For data manipulation and CSV handling
- **Scikit-learn**: For evaluation metrics (F1-score) and preprocessing utilities
- **Matplotlib**: For visualizing training progress and results

## How to Run the Experiments

Each neural network has its own notebook that guides you through the complete experimental process. Here's how to approach each one:

### CNN Experiments (Image Classification)

```bash
cd src/cnn/
jupyter notebook cnn.ipynb
```

**What this notebook does:**
- Automatically downloads and preprocesses the CIFAR-10 dataset (60,000 32×32 color images across 10 classes)
- Splits the data into training (40,000), validation (10,000), and test (10,000) sets
- Trains multiple CNN variants with different architectural choices
- Implements the same architectures from scratch and compares results
- Generates comparison plots and performance metrics

**Expected runtime:** 2-4 hours depending on your hardware (experiments run multiple models with 20 epochs each)

### RNN Experiments (Text Classification)

```bash
cd src/rnn/
jupyter notebook rnn.ipynb
```

**What this notebook does:**
- Loads and preprocesses the NusaX-Sentiment dataset (Indonesian sentiment analysis)
- Converts text to numerical sequences using tokenization
- Trains RNN models with different configurations (layers, units, directions)
- Implements RNN forward propagation from scratch using only NumPy
- Validates that scratch implementations produce identical results to Keras

**Expected runtime:** 1-2 hours (text models train faster than image models)

### LSTM Experiments (Advanced Text Classification)

```bash
cd src/lstm/
jupyter notebook lstm.ipynb
```

**What this notebook does:**
- Uses the same text dataset as RNN experiments for direct comparison
- Implements the complex LSTM cell mathematics from scratch (forget gates, input gates, output gates)
- Compares LSTM performance against simple RNN on identical tasks
- Analyzes whether the additional complexity of LSTM provides meaningful improvements

**Expected runtime:** 1-2 hours

## Understanding the Implementation

### The From-Scratch Philosophy

Our implementations are designed to be educational and transparent. Here's what makes them special:

**Mathematical Transparency**: Every operation is implemented explicitly. For example, our CNN convolution operation shows exactly how filters slide across images, rather than using optimized library functions.

**Modular Design**: Each layer type (Conv2D, RNN Cell, LSTM Cell) is implemented as a separate class with clear `forward()` methods, making it easy to understand data flow.

**Weight Compatibility**: Our from-scratch models can load weights directly from trained Keras models, allowing precise validation of our implementations.

### Key Implementation Details

**CNN Implementation**: Includes explicit convolution operations, padding calculations, and activation functions. The pooling layers show clearly how spatial dimensions are reduced.

**RNN Implementation**: Demonstrates the recurrent connection where hidden states are passed between time steps. The bidirectional implementation shows how forward and backward passes are combined.

**LSTM Implementation**: Implements all four gates (input, forget, cell, output) with their specific activation functions, showing how the cell state flows through time.

## Interpreting the Results

### What to Look For

**Training Curves**: Compare how training and validation loss evolve. Diverging curves indicate overfitting, while parallel curves suggest good generalization.

**F1-Score Comparisons**: We use macro F1-score (average across all classes) rather than simple accuracy because it's more robust to class imbalance.

**Implementation Validation**: The maximum difference between Keras and from-scratch predictions should be very small (< 0.001), confirming implementation correctness.

### Understanding the Experiments

**Layer Depth Experiments**: Generally, deeper networks can learn more complex patterns but are harder to train and more prone to overfitting.

**Width Experiments** (filters/units): More filters or units increase model capacity but also computational cost and overfitting risk.

**Bidirectional vs Unidirectional**: Bidirectional models can use future context for predictions, often improving performance on sequence classification tasks.

## Technical Notes and Troubleshooting

### Memory Considerations

The from-scratch implementations are not optimized for memory efficiency. If you encounter memory issues:
- Reduce batch sizes in the training functions
- Use smaller model architectures for initial testing
- Consider running experiments sequentially rather than in parallel

### Performance Expectations

Our NumPy implementations will be slower than optimized frameworks like TensorFlow. This is expected and acceptable for educational purposes. The focus is on correctness and understanding, not speed.

### Validation Strategy

Each experiment includes validation that our from-scratch implementation produces results very close to the Keras version (typically within 0.001 difference). If you see larger differences, check:
- Weight loading is working correctly
- Input preprocessing is identical between implementations
- Activation functions are implemented correctly

## Extending the Project

### Adding New Experiments

To add new hyperparameter experiments:
1. Create new configuration dictionaries in the notebook
2. Add them to the training loop
3. Include them in the comparison plots

### Implementing New Architectures

The modular design makes it easy to add new layer types:
1. Create a new class following the same pattern (with `load_weights()` and `forward()` methods)
2. Add weight loading logic for the new layer type
3. Include it in the model's forward pass

### Improving Performance

While maintaining the educational focus, you could optimize the implementations:
- Vectorize operations more efficiently
- Add batch processing capabilities
- Implement more efficient convolution algorithms

## Learning Outcomes

By working through this project, you'll gain deep understanding of:

**Neural Network Fundamentals**: How gradients flow, how weights are updated, and how different architectures process different types of data.

**Implementation Skills**: Experience translating mathematical concepts into working code, debugging complex systems, and validating implementations.

**Experimental Design**: How to systematically vary hyperparameters, control for confounding variables, and draw meaningful conclusions from results.

**Model Comparison**: Techniques for fairly comparing different approaches and understanding when additional complexity is worthwhile.

This hands-on experience with implementation details provides intuition that's difficult to gain from using high-level frameworks alone, making you a more effective deep learning practitioner.