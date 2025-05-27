import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

class TextPreprocessor:
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = None
        self.text_vectorizer = None
        self.label_encoder = LabelEncoder()
    
    def build_tokenizer(self, texts: list):
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(texts)
        
    def build_text_vectorizer(self, texts: list):
        self.text_vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_sequence_length=self.max_length,
            standardize='lower_and_strip_punctuation'
        )
        self.text_vectorizer.adapt(texts)
        
    def encode_texts_tokenizer(self, texts: list) -> np.ndarray:
        """Encode texts pake Tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer belum di-build. Call build_tokenizer() first.")
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length, padding='post')
    
    def encode_texts_vectorizer(self, texts: list) -> np.ndarray:
        """Encode texts pake TextVectorization"""
        if self.text_vectorizer is None:
            raise ValueError("TextVectorizer belum di-build. Call build_text_vectorizer() first.")
            
        return self.text_vectorizer(texts).numpy()
    
    def encode_labels(self, labels: list) -> np.ndarray:
        """Encode labels untuk sparse categorical crossentropy"""
        return self.label_encoder.fit_transform(labels)
    
    def transform_labels(self, labels: list) -> np.ndarray:
        """Transform labels using fitted encoder"""
        return self.label_encoder.transform(labels)
    
    def get_vocab_size(self) -> int:
        if self.tokenizer:
            return min(len(self.tokenizer.word_index) + 1, self.vocab_size)
        elif self.text_vectorizer:
            return self.text_vectorizer.vocabulary_size()
        else:
            return self.vocab_size
    
    def get_num_classes(self) -> int:
        return len(self.label_encoder.classes_)
    
    def preprocess_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, use_vectorizer: bool = True) -> Dict[str, Any]:
        """
        All in one preprocessing pipeline
        Returns: dictionary containing processed data and metadata
        """
        # extract texts and labels
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()
        test_texts = test_df['text'].tolist()
        
        train_labels = train_df['label'].tolist()
        val_labels = val_df['label'].tolist()
        test_labels = test_df['label'].tolist()
        
        # encode labels
        train_labels_encoded = self.encode_labels(train_labels)
        val_labels_encoded = self.transform_labels(val_labels)
        test_labels_encoded = self.transform_labels(test_labels)
        
        # apply text encoding
        if use_vectorizer:
            self.build_text_vectorizer(train_texts)
            train_sequences = self.encode_texts_vectorizer(train_texts)
            val_sequences = self.encode_texts_vectorizer(val_texts)
            test_sequences = self.encode_texts_vectorizer(test_texts)
        else:
            self.build_tokenizer(train_texts)
            train_sequences = self.encode_texts_tokenizer(train_texts)
            val_sequences = self.encode_texts_tokenizer(val_texts)
            test_sequences = self.encode_texts_tokenizer(test_texts)
        
        return {
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'train_labels': train_labels_encoded,
            'val_labels': val_labels_encoded,
            'test_labels': test_labels_encoded,
            'vocab_size': self.get_vocab_size(),
            'num_classes': self.get_num_classes(),
            'max_length': self.max_length,
            'label_encoder': self.label_encoder,
            'tokenizer': self.tokenizer if not use_vectorizer else None,
            'text_vectorizer': self.text_vectorizer if use_vectorizer else None
        }