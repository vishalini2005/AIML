import numpy as np
# Parameters initialization
p_X_given_Z0 = 0.6  # P(X=1 | Z=0)
p_X_given_Z1 = 0.8  # P(X=1 | Z=1)
p_Z = 0.5           # P(Z=1)
# Observed data (incomplete data) where some Z values are missing
X_data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
def E_step(X_data, p_X_given_Z0, p_X_given_Z1, p_Z):
    """
    E-step: Estimate the expectation of the log-likelihood function with respect to the current estimates.
    """
    responsibilities = []
    for x in X_data:
        # Calculate P(Z=1 | X=x) using Bayes' rule
        p_X_given_Z = p_X_given_Z1 if x == 1 else (1 - p_X_given_Z1)
        p_Z1_given_X = p_Z * p_X_given_Z / (p_Z * p_X_given_Z + (1 - p_Z) * (1 - p_X_given_Z))
        responsibilities.append(p_Z1_given_X)
    return np.array(responsibilities)

def M_step(X_data, responsibilities):
    """
    M-step: Maximize the expectation of the complete log-likelihood function.
    """
    # Update p_Z
    p_Z_new = np.mean(responsibilities)

    # Update p_X_given_Z1
    p_X_given_Z1_new = np.sum(responsibilities * X_data) / np.sum(responsibilities)

    # Update p_X_given_Z0
    p_X_given_Z0_new = np.sum((1 - responsibilities) * X_data) / np.sum(1 - responsibilities)

    return p_X_given_Z0_new, p_X_given_Z1_new, p_Z_new

# Run EM algorithm
num_iterations = 10
for i in range(num_iterations):
    # E-step
    responsibilities = E_step(X_data, p_X_given_Z0, p_X_given_Z1, p_Z)

    # M-step
    p_X_given_Z0, p_X_given_Z1, p_Z = M_step(X_data, responsibilities)

    # Display updated parameters
    print(f"Iteration {i+1}")
    print(f"p(X=1 | Z=0): {p_X_given_Z0}")
    print(f"p(X=1 | Z=1): {p_X_given_Z1}")
    print(f"p(Z=1): {p_Z}")
    print()

print("Final Parameters:")
print(f"p(X=1 | Z=0): {p_X_given_Z0}")
print(f"p(X=1 | Z=1): {p_X_given_Z1}")
print(f"p(Z=1): {p_Z}")