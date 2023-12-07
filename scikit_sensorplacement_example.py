# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:26:48 2023

@author: eliam
"""

import numpy as np
from skopt import BayesSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt import Optimizer

# Generate synthetic data for demonstration
np.random.seed(42)

# Number of initial sensor locations
initial_sensor_locations = 5
# Total number of sensor locations
total_sensor_locations = 20

# Generate random sensor locations
sensor_locations = np.random.rand(total_sensor_locations, 1)

# Generate synthetic measurements
def generate_data(locations):
    return np.sin(2 * np.pi * locations) + 0.1 * np.random.randn(locations.shape[0])

data = generate_data(sensor_locations)

# Define the Gaussian process regressor with RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
model = GaussianProcessRegressor(kernel=kernel)

# Define the objective function to be minimized
def objective_function(x):
    return x[0]**2 + (x[0] - 2)**2

# Create an optimizer with the objective function and search space
opt = Optimizer([(-5.0, 5.0)], base_estimator="gp", acq_func="gp_hedge", n_random_starts=5)

# Perform Bayesian optimization
result = gp_minimize(objective_function, opt.space, acq_func="gp_hedge", n_calls=20)

optimal_params = result.x

# Print the optimal parameters and minimum value
print("Optimal parameters:", result.x)
print("Minimum value:", result.fun)

# Plot the original sensor locations
plt.scatter(sensor_locations, np.full_like(sensor_locations, objective_function(sensor_locations)), c='blue', label='Original Locations')

# Plot the optimal sensor location
plt.scatter(optimal_params[0], objective_function(optimal_params), c='red', marker='x', label='Optimal Location')

plt.title('Sensor Locations')
plt.legend()
plt.show()