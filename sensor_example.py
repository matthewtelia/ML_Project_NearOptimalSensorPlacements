import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import entropy_sampling

# Generate synthetic data for demonstration
np.random.seed(42)

# Number of initial sensor locations
initial_sensor_locations = 5
# Total number of sensor locations
total_sensor_locations = 20

# Generate random sensor locations
sensor_locations = np.random.rand(total_sensor_locations, 2)

# Generate synthetic measurements
def generate_data(locations):
    return np.sin(2 * np.pi * locations[:, 0]) + np.cos(2 * np.pi * locations[:, 1]) + 0.1 * np.random.randn(locations.shape[0], 1)

data = generate_data(sensor_locations)

# Define the Gaussian process regressor with RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
model = RandomForestClassifier()

# assembling initial training set
n_initial = 5
initial_idx = np.random.choice(range(len(sensor_locations)), size=n_initial, replace=False)
X_initial, y_initial = sensor_locations[initial_idx], data[initial_idx]

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    return np.argmax(std,axis = 0)

# y_initial = y_initial.reshape(-1,1)

# Initialize the active learner
learner = ActiveLearner(
    estimator=model,
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=GP_regression_std  
)

# Number of iterations for active learning
num_iterations = total_sensor_locations - initial_sensor_locations

# Active learning loop
for _ in range(num_iterations):
    query_idx = learner.query(sensor_locations)
    query_location = sensor_locations[query_idx[0]]

    # Simulate measuring the value at the queried location
    query_value = generate_data(query_location)

    # Update the active learner with the new measurement
    learner.teach(query_location, query_value)

# Plot the final sensor locations
plt.scatter(sensor_locations[:, 0], sensor_locations[:, 1], c='blue', label='All Locations')

# Get the queried indices from the learner object
queried_idx = learner.query_strategy.query_instance if hasattr(learner.query_strategy, 'queried_idx') else []

plt.scatter(sensor_locations[queried_idx, 0], sensor_locations[queried_idx, 1], c='red', marker='x', label='Selected Locations')
plt.title('Optimized Sensor Locations')
plt.legend()
plt.show()