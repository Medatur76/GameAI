import numpy as np

def generate_action_vectors(actions, size=2):
    action_vectors = []
    
    for mean_std in actions:
        mean, std_dev = mean_std
        # Generate a vector from a normal distribution with given mean and std deviation
        vector = np.random.normal(mean, std_dev, size)
        action_vectors.append(vector)
    
    return action_vectors

# Example usage
actions = [
    [0, 0.01],  # Mean = 0, Small standard deviation to get close to 0
    [1, 0.01],  # Mean = 1, Small standard deviation to get close to 1
    [1, 0.01],  # Mean = 1, Small standard deviation to get close to 1
    [0, 0.01]   # Mean = 0, Small standard deviation to get close to 0
]
action_vectors = generate_action_vectors(actions, 1)

# Printing the action vectors
for i in range(len(action_vectors)):
    print(f"Action Vector {i}: {action_vectors[i]}")

print(f"Test Vector: {np.round(np.random.normal([0, 1], [0.001, 0.001], 2))}")