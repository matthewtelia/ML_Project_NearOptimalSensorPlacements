# Install required libraries
# pip install numpy scikit-learn modAL

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import matplotlib.pyplot as plt

# Generate synthetic air quality data for demonstration
np.random.seed(42)

# Number of initial sensor locations
initial_sensor_locations = 5
# Total number of sensor locations
total_sensor_locations = 20

# Generate random sensor locations
sensor_locations = np.random.rand(total_sensor_locations, 2)

# Generate synthetic air quality measurements
def generate_air_quality_data(locations):
    return np.sin(2 * np.pi * locations[:, 0]) + np.cos(2 * np.pi * locations[:, 1]) + 0.1 * np.random.randn(locations.shape[0])

air_quality_data = generate_air_quality_data(sensor_locations)

# Define the Gaussian process regressor with RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
model = GaussianProcessRegressor(kernel=kernel)

# Initialize the active learner
learner = ActiveLearner(
    estimator=model,
    X_training=sensor_locations[:initial_sensor_locations],
    y_training=air_quality_data[:initial_sensor_locations],
    query_strategy=uncertainty_sampling
)

# Number of iterations for active learning
num_iterations = total_sensor_locations - initial_sensor_locations

# Active learning loop
for _ in range(num_iterations):
    query_idx, query_instance = learner.query(sensor_locations)
    learner.teach(sensor_locations[query_idx].reshape(1, -1), air_quality_data[query_idx].reshape(1, -1))
    
# Plot the final sensor locations
plt.scatter(sensor_locations[:, 0], sensor_locations[:, 1], c='blue', label='All Locations')
plt.scatter(sensor_locations[learner.query_strategy.queried_idx, 0], 
            sensor_locations[learner.query_strategy.queried_idx, 1], 
            c='red', marker='x', label='Selected Locations')
plt.title('Optimized Sensor Locations')
plt.legend()
plt.show()
