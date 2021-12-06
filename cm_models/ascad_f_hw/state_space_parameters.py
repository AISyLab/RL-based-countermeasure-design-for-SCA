output_states = 9  # Number of Classes/Output layer size
input_size = 700  # Amount of measurements before they enter network/input layer size

countermeasures_budget = 5

# Epsilon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
epsilon_schedule = [[1.0, 500],  # 1500],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 150],
                    [0.5, 150],
                    [0.4, 150],
                    [0.3, 150],
                    [0.2, 150],
                    [0.1, 100]]

# Q-Learning Hyper parameters
# Q Learning omega polynomial parameter (α = 1 / t^ω) where t is the iteration step and α is the learning rate from Eq 3
# This learning rate was based on theoretical and experimental results (Even-Dar and Mansour, 2003)
learning_rate_omega = 0.85
discount_factor = 1.0  # Q Learning discount factor (gamma from Equation 3)
replay_number = 128  # Number trajectories to sample for replay at each iteration

init_utility = 0.3
