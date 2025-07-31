import numpy as np

# Inputs (2 features)
X = np.array([[0.5, 0.1]])  # shape (1, 2)

# True output (label)
y_true = np.array([[1]])  # shape (1, 1)

# Weights for input → hidden (2 inputs to 2 neurons)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

# Weights for hidden → output
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
# Layer 1
z1 = X.dot(W1) + b1       # Linear part
a1 = relu(z1)             # Activation

# Output layer
z2 = a1.dot(W2) + b2
y_pred = z2               # No activation for regression (or use sigmoid for classification)

loss = np.mean((y_true - y_pred) ** 2)
print("Loss:", loss)

# Derivative of loss w.r.t prediction
d_loss_y_pred = 2 * (y_pred - y_true)

# Output layer gradients
d_z2 = d_loss_y_pred            # because d(y_pred)/d(z2) = 1
d_W2 = a1.T.dot(d_z2)
d_b2 = np.sum(d_z2, axis=0, keepdims=True)

# Hidden layer gradients
d_a1 = d_z2.dot(W2.T)
d_z1 = d_a1 * relu_derivative(z1)
d_W1 = X.T.dot(d_z1)
d_b1 = np.sum(d_z1, axis=0, keepdims=True)

learning_rate = 0.1

W2 -= learning_rate * d_W2
b2 -= learning_rate * d_b2
W1 -= learning_rate * d_W1
b1 -= learning_rate * d_b1
