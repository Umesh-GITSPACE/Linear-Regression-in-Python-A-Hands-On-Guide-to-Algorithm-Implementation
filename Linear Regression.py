# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:07:42 2024

@author: Umesh
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize parameters
m = np.random.randn(1)
c = np.random.randn(1)

# Define the model prediction function
def predict(X, m, c):
    return m * X + c

# Define the cost function (Mean Squared Error)
def compute_cost(X, y, m, c):
    predictions = predict(X, m, c)
    return np.mean((predictions - y) ** 2)

# Gradient Descent implementation
def gradient_descent(X, y, m, c, learning_rate, iterations):
    n = len(y)
    for i in range(iterations):
        y_pred = predict(X, m, c)
        dm = (-2/n) * np.sum(X * (y - y_pred))
        dc = (-2/n) * np.sum(y - y_pred)
        m = m - learning_rate * dm
        c = c - learning_rate * dc
    return m, c

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Train the model
m, c = gradient_descent(X, y, m, c, learning_rate, iterations)

# Output the optimized parameters
print(f"Optimized slope (m): {m}")
print(f"Optimized intercept (c): {c}")

# Evaluate the final model
y_pred = predict(X, m, c)
cost = compute_cost(X, y, m, c)
print(f"Final cost: {cost}")

# Visualize the results
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


