import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
n_timesteps = 50
state = 0
state_variance = 1  # Initial state variance
process_variance = 0.1  # Process noise variance
measurement_variance = 1  # Measurement noise variance

# State transition and measurement matrices
F = 1  # State transition (identity in this simple case)
H = 1  # Measurement matrix (identity for direct observation)

# Lists to store results
states = []
measurements = []

# Run Kalman filter
for t in range(n_timesteps):
    # True state with process noise
    state += np.random.normal(0, np.sqrt(process_variance))
    measurement = state + np.random.normal(0, np.sqrt(measurement_variance))

    # Prediction step
    predicted_state = F * state
    predicted_variance = state_variance * F * F + process_variance

    # Update step with measurement
    kalman_gain = predicted_variance * H / (H * predicted_variance * H + measurement_variance)
    updated_state = predicted_state + kalman_gain * (measurement - H * predicted_state)
    updated_variance = (1 - kalman_gain * H) * predicted_variance

    # Store results
    states.append(updated_state)
    measurements.append(measurement)

    # Update state and variance for the next iteration
    state = updated_state
    state_variance = updated_variance

# Plot results
plt.plot(states, label='Estimated State')
plt.plot(measurements, label='Measurements', linestyle='dotted')
plt.legend()
plt.title('Kalman Filter with Dynamic Covariance')
plt.show()
