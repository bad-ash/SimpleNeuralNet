from sklearn.datasets import fetch_openml
import numpy as np

def processMNIST():

  # Fetch MNIST data
  mnist = fetch_openml('mnist_784', version=1)

  # cast data as 32bit, target as an int representing 0-9
  X = mnist.data.astype(np.float32)
  y = mnist.target.astype(int)

  # only 0s and 1s
  mask = ((y == 0) | (y == 1))
  X = X[mask]
  y = y[mask]

  # Normalize
  X /= 255.0

  train_length = int(0.8 * X.shape[0])
  # 4) Split into train/test
  x_train, x_test = X[:train_length], X[train_length:]
  y_train, y_test = y[:train_length], y[train_length:]

  return x_train, x_test, y_train, y_test