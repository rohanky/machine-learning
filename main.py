# main.py
import os
import pandas as pd
from data.load_data import load_data

from utils.pre_processing import apply_text_processing, encode_labels, extract_features, split_data
from models.ml_models import train_and_evaluate_model

def preprocess_data(file_path, text_column, label_column):
    """
    Full preprocessing pipeline for the data.
    """
    # Load data
    data = load_data(file_path)

    
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
    file_path = 'data/spam_ham_dataset.csv'
    text_column = 'text'
    label_column = 'label_num'

    # Preprocess data
    X_train, X_test, y_train, y_test, encoder, vectorizer = preprocess_data(file_path, text_column, label_column)

    # Train and evaluate SVM model
    model_name = "svm"  # Change this flag to try different models
    model, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name=model_name)
    print(f"{model_name.upper()} Model Metrics:", metrics)
