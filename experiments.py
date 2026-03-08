import numpy as np
import matplotlib.pyplot as plt
from bandit_environment import Bandit
from agents import Random_Agent, Epsilon_Greedy_Agent, Softmax_Agent, UCB_Agent

# -------------------------------
# Experiment Parameters
# -------------------------------
k = 10             # Number of arms
steps = 1000       # Steps per run
epsilon = 0.1      # For epsilon-greedy
tau = 0.1          # For softmax
c = 0.2            # For UCB

# -------------------------------
# Initialize bandit and agents
# -------------------------------
bandit = Bandit(k)
agents = {
    "Random": Random_Agent(k),
    "Epsilon-Greedy": Epsilon_Greedy_Agent(k, epsilon),
    "Softmax": Softmax_Agent(k, tau),
    "UCB": UCB_Agent(k, c)
}

# Identify optimal action
optimal_action = np.argmax(bandit.true_values)

# -------------------------------
# Run experiment
# -------------------------------
avg_rewards = {name: [] for name in agents}
optimal_action_counts = {name: [] for name in agents}

for step in range(steps):
    for name, agent in agents.items():
        # UCB requires passing step index
        if name == "UCB":
            action = agent.select_action(step + 1)  # +1 to avoid log(0)
        else:
            action = agent.select_action()
        
        reward = bandit.pull(action)
        agent.update(action, reward)
        
        # Track cumulative average reward
        if step == 0:
            avg_rewards[name].append(reward)
        else:
            avg_rewards[name].append(avg_rewards[name][-1] + (reward - avg_rewards[name][-1])/(step + 1))
        
        # Track % optimal action
        if step == 0:
            optimal_action_counts[name].append(1 if action == optimal_action else 0)
        else:
            optimal_action_counts[name].append(optimal_action_counts[name][-1] + ((1 if action == optimal_action else 0) - optimal_action_counts[name][-1])/(step + 1))

# -------------------------------
# Plot Average Reward
# -------------------------------
plt.figure(figsize=(12,5))
for name in agents:
    plt.plot(avg_rewards[name], label=name)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Multi-Armed Bandit: Average Reward")
plt.legend()
plt.show()

# -------------------------------
# Plot % Optimal Action
# -------------------------------
plt.figure(figsize=(12,5))
for name in agents:
    plt.plot(np.array(optimal_action_counts[name])*100, label=name)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Multi-Armed Bandit: % Optimal Action")
plt.legend()
plt.show()