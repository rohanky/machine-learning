# main.py
import os
import pandas as pd
from data.load_data import load_data
from utils.clean_data import clean_data
from utils.pre_processing import apply_text_processing, encode_labels, extract_features, split_data


def preprocess_data(file_path, text_column, label_column):
    """
    Full preprocessing pipeline for the data.
    """
    # Load data
    data = load_data(file_path)

    # Clean data
    data = clean_data(data)

    # Preprocess text column
    data = apply_text_processing(data, text_column)

    # Encode labels
    data, encoder = encode_labels(data, label_column)

    # Extract features (TF-IDF)
    features, vectorizer = extract_features(data, text_column)

    # Split data into train and test sets
    labels = data[label_column].values
    X_train, X_test, y_train, y_test = split_data(features, labels)

    return X_train, X_test, y_train, y_test, encoder, vectorizer

if __name__ == "__main__":
    # Set file path and column names
    file_path = 'data/spam_data.csv'
    text_column = 'text'
    label_column = 'label'

    # Preprocess data
    X_train, X_test, y_train, y_test, encoder, vectorizer = preprocess_data(file_path, text_column, label_column)
    print(f"Data Preprocessing Complete! Training data size: {len(X_train)}")
