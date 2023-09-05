import numpy as np
from scipy.integrate import quad


# Define the Gaussian mask gα(θ)
def gaussian_mask(theta, alpha, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(theta - alpha) ** 2 / (2 * sigma ** 2))


# Define the given values
P_values = [-14, 17, 45, 54, 65, 46, 47, -26, 28, 33, 39, 42, 38, 34, 41]
alpha = 45
variance = 30
k = 1 / np.sum(P_values)  # Normalizing constant

# Calculate η for each P value
etas = []

for p in P_values:
    P_prime = k * p
    sigma = np.sqrt(variance)

    # Define the integrands for the numerator and denominator
    numerator_integrand = lambda theta: gaussian_mask(theta, alpha, sigma) * P_prime
    denominator_integrand = lambda theta: (1 - gaussian_mask(theta, alpha, sigma)) * P_prime

    # Numerical integration using quad
    numerator, _ = quad(numerator_integrand, -np.inf, np.inf)
    denominator, _ = quad(denominator_integrand, -np.inf, np.inf)

    eta = numerator / denominator
    etas.append(eta)

# Print the results
for i, p in enumerate(P_values):
    print(f"P({i}) = {p}, η({i}) = {etas[i]}")

a =1
