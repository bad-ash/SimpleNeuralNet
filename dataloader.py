"""
Data Loader Module for MNIST Binary Classification

This module provides functionality to load and preprocess the MNIST dataset
for binary classification tasks (specifically distinguishing between digits 0 and 1).
The data is fetched, filtered, normalized, and split into training and testing sets.
"""

from sklearn.datasets import fetch_openml
import numpy as np

def processMNIST():
    """
    Load and preprocess MNIST dataset for binary classification (0s vs 1s).
    
    This function:
    1. Fetches the MNIST dataset from OpenML
    2. Filters to only include digits 0 and 1
    3. Normalizes pixel values to [0, 1] range
    4. Splits data into 80% training and 20% testing sets
    
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
            - x_train: Training features (normalized pixel values)
            - x_test: Testing features (normalized pixel values)  
            - y_train: Training labels (0 or 1)
            - y_test: Testing labels (0 or 1)
    """

    # Fetch MNIST dataset from OpenML repository
    # MNIST contains 70,000 28x28 grayscale images of handwritten digits
    mnist = fetch_openml('mnist_784', version=1)

    # Convert data to 32-bit float for memory efficiency and numerical stability
    # Convert target labels to integers (0-9) for proper classification
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    # Create binary classification dataset by filtering only digits 0 and 1
    # This reduces the 10-class problem to a 2-class problem
    mask = ((y == 0) | (y == 1))
    X = X[mask]  # Apply mask to features
    y = y[mask]  # Apply mask to labels

    # Normalize pixel values from [0, 255] to [0, 1] range
    # This helps with neural network training convergence and numerical stability
    X /= 255.0

    # Calculate split point for 80/20 train/test split
    train_length = int(0.8 * X.shape[0])
    
    # Split dataset into training and testing sets
    # Training set: first 80% of data
    # Testing set: remaining 20% of data
    x_train, x_test = X[:train_length], X[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]

    return x_train, x_test, y_train, y_test