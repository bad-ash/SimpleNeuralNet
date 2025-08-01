from dataloader import processMNIST
from Simple1LayerMLP import Simple1LayerMLP
import numpy as np
import matplotlib.pyplot as plt

def batches(X, y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

x_train, x_test, y_train, y_test = processMNIST()
x_train = x_train.values
y_train = y_train.values
x_test  = x_test.values
y_test  = y_test.values

model = Simple1LayerMLP(D_in=784, D_out=128)
# 3) Training loop
train_losses = []
n_epochs = 10
batch_size = 64
learning_rate = 1e-3

for epoch in range(1, n_epochs + 1):
    epoch_losses = []
    for Xb, yb in batches(x_train, y_train, batch_size):
        loss = model.train_step(Xb, yb, learning_rate)
        epoch_losses.append(loss)
    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}/{n_epochs} — Training Loss: {avg_loss:.4f}")

# Evaluation on test set
probs = model.forward(x_test)               # shape (N_test,)
preds = (probs >= 0.5).astype(int)          # binary predictions
accuracy = np.mean(preds == y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot the training loss curve
plt.figure()
plt.plot(range(1, n_epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()