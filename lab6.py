import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_data = pd.read_csv('logisticX.csv', header=None)
Y_data = pd.read_csv('logisticY.csv', header=None)

X = (X_data - X_data.mean()) / X_data.std()
Y = Y_data.values.flatten()

X = np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, Y, theta):
    m = len(Y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    return cost


def gradient_descent(X, Y, theta, learning_rate, num_iterations):
    m = len(Y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - Y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, Y, theta))

    return theta, cost_history


theta = np.zeros(X.shape[1])
learning_rate = 0.1
num_iterations = 1000

theta, cost_history = gradient_descent(X, Y, theta, learning_rate, num_iterations)

plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history, label="Learning Rate = 0.1")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function vs Iteration")
plt.legend()
plt.savefig("Cost function vs Iteration.png", dpi=300, bbox_inches="tight")
plt.show()


def plot_decision_boundary(X, Y, theta):
    plt.figure(figsize=(8, 6))

    class0 = (Y == 0)
    class1 = (Y == 1)
    plt.scatter(X[class0, 1], X[class0, 2], label="Class 0", marker='x')
    plt.scatter(X[class1, 1], X[class1, 2], label="Class 1", marker='o')

    x1_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]
    plt.plot(x1_vals, x2_vals, color='red', label="Decision Boundary")

    plt.xlabel("Normalized Feature 1")
    plt.ylabel("Normalized Feature 2")
    plt.title("Decision Boundary")
    plt.legend()
    plt.savefig("decision_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_decision_boundary(X, Y, theta)

learning_rates = [0.1, 5]
iternations = 100

cost_histories = {}

for lr in learning_rates:
    theta_temp = np.zeros(X.shape[1])
    _, cost_history_lr = gradient_descent(X, Y, theta_temp, lr, iternations)
    cost_histories[lr] = cost_history_lr

plt.figure(figsize=(8, 6))
for lr, cost_history in cost_histories.items():
    plt.plot(range(len(cost_history)), cost_history, label=f"Learning Rate = {lr}")

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function vs Iteration (Learning Rates Comparison)")
plt.legend()
plt.savefig("cost_vs_iteration.png", dpi=300, bbox_inches="tight")
plt.show()


def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return probabilities >= 0.5


predictions = predict(X, theta)
TP = np.sum((predictions == 1) & (Y == 1))
TN = np.sum((predictions == 0) & (Y == 0))
FP = np.sum((predictions == 1) & (Y == 0))
FN = np.sum((predictions == 0) & (Y == 1))

confusion_matrix = np.array([[TP, FP], [FN, TN]])

accuracy = (TP + TN) / len(Y)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Confusion Matrix:")
print(confusion_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
