# Tugas Besar Pembelajaran Mesing 2 - Feed Forward Implementation from Scratch

## Project Overview

This project implements three fundamental deep learning architectures **completely from scratch** using only NumPy, and compares their performance with TensorFlow/Keras implementations. The goal is to deeply understand how these neural networks work at the mathematical level while conducting systematic experiments on their hyperparameters.

### What We're Building

**Convolutional Neural Network (CNN)** - For image classification using the CIFAR-10 dataset, exploring how different architectural choices affect performance on visual recognition tasks.

**Simple Recurrent Neural Network (RNN)** - For sentiment analysis using the NusaX-Sentiment dataset (Indonesian text), investigating how sequence processing capabilities vary with different configurations.

**Long Short-Term Memory (LSTM)** - Also for sentiment analysis, examining how this advanced RNN variant handles long-term dependencies in text data.

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

You'll need Python 3.8 or higher with the following key libraries.

### Installation Steps

1. **Clone the repository and navigate to the project directory:**
```bash
git clone https://github.com/AlthariqFairuz/Tubes-ML-2.git
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
- **Matplotlib and Seaborn**: For visualizing training progress and results

## How to Run the Experiments

Each neural network has its own notebook that guides you through the complete experimental process. Here's how to approach each one:

### CNN Experiments (Image Classification)

```bash
cd src/cnn/
jupyter notebook cnn.ipynb
```

**What this notebook does:**
- Automatically downloads and preprocesses the CIFAR-10 dataset (60,000 32×32 color images across 10 classes)
- Trains multiple CNN variants with different architectural choices
- Implements the same architectures from scratch and compares results
- Generates comparison plots and performance metrics

### RNN Experiments (Text Classification)

```bash
cd src/rnn/
jupyter notebook rnn.ipynb
```

**What this notebook does:**
- Loads and preprocesses the NusaX-Sentiment dataset (Indonesian sentiment analysis)
- Converts text to numerical sequences using tokenization
- Trains RNN models with different configurations (layers, units, directions)
- Validates that scratch implementations produce identical results to Keras

### LSTM Experiments (Advanced Text Classification)

```bash
cd src/lstm/
jupyter notebook lstm.ipynb
```

**What this notebook does:**
- Uses the same text dataset as RNN experiments for direct comparison
- Implements the complex LSTM cell mathematics from scratch (forget gates, input gates, output gates)
- Analyzes whether the additional complexity of LSTM provides meaningful improvements

### What to Look For

**Training Curves**: Compare how training and validation loss evolve. Diverging curves indicate overfitting, while parallel curves suggest good generalization.

**F1-Score Comparisons**: We use macro F1-score (average across all classes) rather than simple accuracy because it's more robust to class imbalance.

**Implementation Validation**: The maximum difference between Keras and from-scratch predictions should be very small (< 0.001), confirming implementation correctness.

### Understanding the Experiments

**Layer Depth Experiments**: Generally, deeper networks can learn more complex patterns but are harder to train and more prone to overfitting.

**Width Experiments** (filters/units): More filters or units increase model capacity but also computational cost and overfitting risk.

**Bidirectional vs Unidirectional**: Bidirectional models can use future context for predictions, often improving performance on sequence classification tasks.

