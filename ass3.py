import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data_x = pd.read_csv('LinearX.csv', header=None).values.flatten()
data_y = pd.read_csv('LinearY.csv', header=None).values.flatten()

# Standardize feature
feature_x = (data_x - np.mean(data_x)) / np.std(data_x)

# Prepare dataset
features = np.c_[np.ones((len(feature_x), 1)), feature_x]  # Add bias
labels = data_y.reshape(-1, 1)


# Compute error
def calculate_loss(features, labels, params):
    num_samples = len(labels)
    predictions = features @ params
    loss = (1 / (2 * num_samples)) * np.sum((predictions - labels) ** 2)
    return loss


# Gradient descent implementation
def optimize_params(features, labels, params, step_size, num_iterations):
    num_samples = len(labels)
    loss_track = []

    for _ in range(num_iterations):
        adjustments = (1 / num_samples) * (features.T @ (features @ params - labels))
        params -= step_size * adjustments
        loss = calculate_loss(features, labels, params)
        loss_track.append(loss)

    return params, loss_track


# Training using gradient descent
step_size = 0.5
num_iterations = 50
parameters = np.zeros((features.shape[1], 1))
final_params, loss_progression = optimize_params(features, labels, parameters, step_size, num_iterations)

# Loss plot
plt.figure()
plt.plot(range(len(loss_progression)), loss_progression, label="Loss Reduction", color='blue')
plt.xlabel("Iteration Count")
plt.ylabel("Loss Value")
plt.title("Loss vs Iterations (Step Size: 0.5)")
plt.legend()
plt.grid(True)
plt.show()

# Plot regression results
plt.figure()
plt.scatter(feature_x, data_y, label='Dataset Points', color='blue')
plt.plot(feature_x, features @ final_params, label='Best Fit Line', color='red')
plt.xlabel("Normalized Input Feature")
plt.ylabel("Target Variable")
plt.title("Data and Regression Line")
plt.legend()
plt.grid(True)
plt.show()

# Exploring multiple step sizes
step_options = [0.005, 0.5, 5]
plt.figure()
for rate in step_options:
    temp_params = np.zeros((features.shape[1], 1))
    _, loss_path = optimize_params(features, labels, temp_params, rate, num_iterations)
    plt.plot(range(len(loss_path)), loss_path, label=f'Rate={rate}')

plt.xlabel("Iteration Count")
plt.ylabel("Loss Value")
plt.title("Loss vs Iterations for Different Step Sizes")
plt.legend()
plt.grid(True)
plt.show()


# Stochastic and Mini-Batch Gradient Descent

def stochastic_optimization(features, labels, params, step_size, num_iterations):
    num_samples = len(labels)
    loss_history = []
    for _ in range(num_iterations):
        for _ in range(num_samples):
            rand_idx = np.random.randint(num_samples)
            x_rand = features[rand_idx:rand_idx + 1]
            y_rand = labels[rand_idx:rand_idx + 1]
            update = x_rand.T @ (x_rand @ params - y_rand)
            params -= step_size * update
        loss = calculate_loss(features, labels, params)
        loss_history.append(loss)
    return params, loss_history


def mini_batch_optimization(features, labels, params, step_size, num_iterations, mini_batch_size):
    num_samples = len(labels)
    loss_record = []
    for _ in range(num_iterations):
        shuffled_indices = np.random.permutation(num_samples)
        shuffled_x = features[shuffled_indices]
        shuffled_y = labels[shuffled_indices]
        for batch_start in range(0, num_samples, mini_batch_size):
            x_batch = shuffled_x[batch_start:batch_start + mini_batch_size]
            y_batch = shuffled_y[batch_start:batch_start + mini_batch_size]
            batch_update = (1 / len(y_batch)) * x_batch.T @ (x_batch @ params - y_batch)
            params -= step_size * batch_update
        loss = calculate_loss(features, labels, params)
        loss_record.append(loss)
    return params, loss_record


# Initialize parameters for different methods
stochastic_params = np.zeros((features.shape[1], 1))
mini_batch_params = np.zeros((features.shape[1], 1))
batch_size = 10

# Execute Gradient Descent Variants
_, loss_stochastic = stochastic_optimization(features, labels, stochastic_params, step_size=0.1, num_iterations=50)
_, loss_mini_batch = mini_batch_optimization(features, labels, mini_batch_params, step_size=0.1, num_iterations=50,
                                             mini_batch_size=batch_size)

# Compare Loss Progressions
plt.figure()
plt.plot(range(len(loss_stochastic)), loss_stochastic, label="Stochastic GD", linestyle="--", color='green')
plt.plot(range(len(loss_mini_batch)), loss_mini_batch, label="Mini-Batch GD", linestyle="-.", color='orange')
plt.plot(range(len(loss_progression)), loss_progression, label="Full Batch GD", linestyle="-", color='blue')
plt.xlabel("Iteration Count")
plt.ylabel("Loss Value")
plt.title("Loss vs Iterations for Different Gradient Descent Variants")
plt.legend()
plt.grid(True)
plt.savefig('gradient_descent_comparison_variant.png')
plt.show()
