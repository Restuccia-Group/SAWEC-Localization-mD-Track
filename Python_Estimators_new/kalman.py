import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Simulated measurements
measurements = np.array([1, 1.2, 1.3, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8])

# Create a Kalman filter
kf = KalmanFilter(initial_state_mean=1.0,  # Initial value
                  initial_state_covariance=1.0,  # Initial uncertainty
                  observation_covariance=0.1,  # Measurement noise
                  transition_covariance=0.01)  # Process noise

# Apply the Kalman filter to the measurements
(filtered_state_means, _) = kf.filter(measurements)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(measurements, label='Measurements', marker='o')
plt.plot(filtered_state_means, label='Filtered', marker='x')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Kalman Filter for Tracking Changing Values')
plt.grid(True)
plt.show()
