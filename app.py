import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# 1. CREATE SYNTHETIC DATA
# -------------------------------------------------------
# We'll create 100 data points in 2D space.
# Then we'll label them "1" if x1 + x2 > 0, else "0".
# This way, the data is somewhat linearly separable.
torch.manual_seed(0)  # For reproducible results
X = torch.randn(100, 2)  # Shape: [100 samples, 2 features]
y = (X[:, 0] + X[:, 1] > 0).long()  # Shape: [100 samples], values 0 or 1

# x is a 2D tensor with the shape 100x2.
# Each row in our 2D tensor serve as a data point
# Each colum value represents a feature
# So, X[i] is a 1D tensor with 2 elements, corresponding to the two features of the i-th sample.

# y = tensor([0, 0, 1, 0, 0...])
# y is a 1D tensor with 100 values, each will serve to be a label for each data point (row) in X.
# y[i] is an integer label for the i-th sample.


# -------------------------------------------------------
# 2. DEFINE A SIMPLE NEURAL NETWORK
# -------------------------------------------------------
# We'll build a small feed-forward (fully connected) network.
# Sequential means we just stack layers: Linear -> ReLU -> Linear.
model = nn.Sequential(
    nn.Linear(2, 4),  # Input size = 2, hidden layer size = 4
    nn.ReLU(),        # Activation function
    nn.Linear(4, 2)   # Output layer: 2 classes
)

# -------------------------------------------------------
# 3. CHOOSE A LOSS FUNCTION AND OPTIMIZER
# -------------------------------------------------------
# - CrossEntropyLoss is common for classification tasks.
# - We'll use a simple SGD (stochastic gradient descent) optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------
# We'll run several epochs of training. Each epoch:
#   a) Clears old gradients.
#   b) Feeds 'X' into the model to get predictions y_pred.
#   c) Computes loss using y_pred and true labels 'y'.
#   d) Backpropagates to compute gradients.
#   e) Updates the model's parameters using the optimizer.
num_epochs = 1000
for epoch in range(num_epochs):
    # a) Zero out the gradients from the previous iteration
    optimizer.zero_grad()

    # b) Forward pass: get the model's predictions for X
    y_pred = model(X)

    # c) Compute the cross-entropy loss between predictions and labels
    loss = loss_fn(y_pred, y)

    # d) Backward pass: calculate gradients
    loss.backward()

    # e) Optimize the parameters
    optimizer.step()

    # Print the loss every 200 epochs
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# -------------------------------------------------------
# 5. TEST THE MODEL
# -------------------------------------------------------
# Let's see what the model predicts for a couple of simple inputs.
# our test samples are represented using a 2D tensor with two rows serving as sample and 2 columns whose value serve as features.
test_samples = torch.tensor([[-1.0, -1.0],  # Should be negative sum => class 0, based on how we trained our model
                             [ 2.0,  1.0]]) # Should be positive sum => class 1
model.eval()  # Switch to evaluation mode (e.g. turns off dropout if any)
with torch.no_grad():   # When making predictions, we don't need to compute gradients

    test_preds = model(test_samples) # Feeds test_samples (the test data) into the trained model to get predictions.
    # test_preds will be logits (unnormalized scores).
    # The highest-scoring class in each row is the predicted label.
    predicted_labels = test_preds.argmax(dim=1) # Here based on the values of test preds, we determine the labels
    # Since test_preds contains logits, the highest value represents the most likely class.

    print("\nTest samples:")
    print(test_samples)
    print("Model logits:", test_preds)
    print("Predicted labels:", predicted_labels)













