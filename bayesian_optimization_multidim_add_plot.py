import numpy as np
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

# Objective function: Negative of the sum of distances to a set of predefined points (simulating optimal sensor locations)
def sensor_objective(X, sensor_locations):
    distances = np.linalg.norm(X - sensor_locations, axis=1)
    return -np.sum(distances)

# generating the data
x1, x2 = np.linspace(0, 10, 11).reshape(-1, 1), np.linspace(0, 10, 11).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)
X = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

y = np.sin(np.linalg.norm(X, axis=1))/2 - ((10 - np.linalg.norm(X, axis=1))**2)/50 + 2

# assembling initial training set
X_initial, y_initial = X[:10], y[:10]

# # Generate initial sensor locations and corresponding objective values
# initial_sensor_locations = np.array([[2, 3], [8, 8]])
# X_initial = np.random.rand(10, 2) * 10  # Random initial sensor locations
# y_initial = [sensor_objective(x, initial_sensor_locations) for x in X_initial]

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)

optimizer = BayesianOptimizer(
    estimator=GaussianProcessRegressor(kernel=kernel),
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=max_EI)

num_iterations = 20
num_query_points = 5

for _ in range(num_iterations):
    query_idx, query_inst = optimizer.query(X)
    optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx])
    # query_idx, query_inst = optimizer.query(np.random.rand(num_query_points, 2) * 10)
    # query_inst_flat = query_inst.flatten()
    # query_objectives = [sensor_objective(query_inst[i, :], initial_sensor_locations) for i in range(num_query_points)]
    # optimizer.teach(query_inst_flat, query_objectives)

# Visualize the optimization process
plt.figure(figsize=(12, 5))

# Plot the true function
plt.subplot(1, 2, 1)
plt.title('True Function')
plt.contourf(x1, x2, y.reshape(x1.shape), cmap='viridis')
plt.colorbar()

# Plot the optimized function
plt.subplot(1, 2, 2)
plt.title('Optimized Function')
predicted_mean = optimizer.predict(X)
plt.contourf(x1, x2, predicted_mean.reshape(x1.shape), cmap='viridis')
plt.colorbar()

plt.show()

# # Visualize the optimized sensor locations
# optimized_sensor_locations = optimizer.X_training
# print("Optimized Sensor Locations:")
# print(optimized_sensor_locations)

# # Plot the optimized sensor locations
# plt.scatter(initial_sensor_locations[:, 0], initial_sensor_locations[:, 1], label='Initial Sensor Locations', marker='x')
# plt.scatter(optimized_sensor_locations[:, 0], optimized_sensor_locations[:, 1], label='Optimized Sensor Locations', marker='o')
# plt.title('Optimized Sensor Locations')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.show()