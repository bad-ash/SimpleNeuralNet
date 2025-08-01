import numpy as np

class Simple1LayerMLP:
    def __init__(self, D_in, D_out):
        # dims
        self.D_in  = D_in
        self.D_out = D_out

        # params: W1:(D_in×D_out), b1:(D_out,), W2:(D_out,), b2:scalar
        self.W1 = np.random.randn(D_in, D_out) * np.sqrt(2/D_in) # He initialization
        self.b1 = np.zeros(D_out)
        self.W2 = np.random.randn(D_out)    * np.sqrt(2/D_out) # He initialization
        self.b2 = 0.0

    def forward(self, X, y=None):
        """
        X: (BxD_in)
        y: (B,) binary targets, optional
        Stores all intermediates on self for backward().
        Returns: 
          P = sigmoid(z2)    shape (B,)
          optionally loss if y is given
        """
        self.X = X
        # hidden pre‐act
        self.z1 = X.dot(self.W1) + self.b1        # (B×D_out)
        # hidden post‐act ReLU
        self.H1 = np.maximum(0, self.z1)
        # output logit
        self.z2 = self.H1.dot(self.W2) + self.b2  # (B,)
        # prob
        self.P  = 1/(1 + np.exp(-self.z2))        # (B,)

        if y is not None:
            self.y = y
            # binary cross‐entropy
            eps = 1e-8
            self.loss = -np.mean(
                y*np.log(self.P+eps) + (1-y)*np.log(1-self.P+eps)
            )
            return self.P, self.loss

        return self.P

    def backward(self):
        """
        Assumes self.X, self.z1, self.H1, self.z2, self.P, self.y exist.
        Computes and stores:
          self.dW1, self.db1, self.dW2, self.db2
        """
        B = self.X.shape[0]

        # 1) dJ/dz2  (shape (B,))
        dz2 = (self.P - self.y) / B

        # 2) dJ/dW2  = H1.T @ dz2    → (D_out,)
        self.dW2 = self.H1.T.dot(dz2)    # (D_out,)

        # 3) dJ/db2  = sum(dz2)
        self.db2 = np.sum(dz2)           # scalar

        # 4) dJ/dH1  = dz2[:,None] * W2[None,:]  → (B×D_out)
        dH1 = dz2[:, None] * self.W2[None, :]

        # 5) dJ/dz1  = dH1 * (z1>0)            → (B×D_out)
        dZ1 = dH1 * (self.z1 > 0)

        # 6) dJ/dW1  = X.T @ dZ1               → (D_in×D_out)
        self.dW1 = self.X.T.dot(dZ1)

        # 7) dJ/db1  = sum over batch of dZ1   → (D_out,)
        self.db1 = np.sum(dZ1, axis=0)

    def update(self, lr):
        """Simple SGD step."""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def train_step(self, X, y, lr):
        """One forward+backward+update; returns loss."""
        _, loss = self.forward(X, y)
        self.backward()
        self.update(lr)
        return loss
