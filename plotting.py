


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For smoothing

# Load the CSV files
exploration_csv_path = 'exploration_K2.csv'  # Update with your correct file path
fixed_epsilon_csv_path = 'fixedepsilon.csv'  # Update with your correct file path
decaying_csv_path = 'training_performance_decaying.csv'  # Update with your correct file path

# Read the CSV files
exploration_data = pd.read_csv(exploration_csv_path)
fixed_epsilon_data = pd.read_csv(fixed_epsilon_csv_path)
decaying_data = pd.read_csv(decaying_csv_path)

# Filter data (for example, consider the first 10,000 episodes
exploration_data_filtered = exploration_data[exploration_data['episode'] <= 850000]
fixed_epsilon_data_filtered = fixed_epsilon_data[fixed_epsilon_data['episode'] <= 850000]
decaying_data_filtered = decaying_data[decaying_data['episode'] <= 850000]

# Calculate the maximum cumulative reward observed as the "optimal" reward
max_reward = max(
    exploration_data['cumulative_reward'].max(),
    fixed_epsilon_data['cumulative_reward'].max(),
    decaying_data['cumulative_reward'].max()
)
print(f"Max Reward: {max_reward}")

# Function to calculate regret
def calculate_regret(data, max_reward):
    data['regret'] = max_reward - data['cumulative_reward']
    return data

# Apply the regret calculation to all strategies
exploration_data_filtered = calculate_regret(exploration_data_filtered, max_reward)
fixed_epsilon_data_filtered = calculate_regret(fixed_epsilon_data_filtered, max_reward)
decaying_data_filtered = calculate_regret(decaying_data_filtered, max_reward)

# Function to plot the continuous regret using Savitzky-Golay smoothing
def plot_smooth_regret(data, label, color):
    # Interpolating between episodes for smooth regret curve
    episodes_continuous = np.linspace(data['episode'].min(), data['episode'].max(), 100)
    regret_continuous = np.interp(episodes_continuous, data['episode'], data['regret'])
    
    # Apply Savitzky-Golay filter to smooth the curve
    smooth_regret = savgol_filter(regret_continuous, window_length=100, polyorder=3)
    
    # Plotting the smoothed regret
    plt.plot(episodes_continuous, smooth_regret, label=label, color=color, linewidth=3)

# Plot for all strategies in one plot
plt.figure(figsize=(10, 6))

# Plot each strategy's regret on the same plot
plot_smooth_regret(exploration_data_filtered, "Exploration Strategy", 'blue')
plot_smooth_regret(fixed_epsilon_data_filtered, "Fixed Epsilon", 'orange')
plot_smooth_regret(decaying_data_filtered, "Decaying Epsilon", 'green')

# Labels and Title
plt.xlabel('Episode')
plt.ylabel('Cumulative Regret')
plt.title('Comparison of Cumulative Regret Across Strategies')

# Add a legend
plt.legend()

# Add a grid
plt.grid(True)
plt.savefig('regret_comparison.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
