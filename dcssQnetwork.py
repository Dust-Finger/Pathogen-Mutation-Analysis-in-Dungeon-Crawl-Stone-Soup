import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the neural network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the RL environment
class DCSSRLAgentEnvironment:
    def __init__(self, log_data):
        self.log_data = log_data
        self.current_index = 0

    def reset(self):
        """Resets the environment to the first log entry."""
        self.current_index = 0
        return self.log_data[self.current_index]['state']

    def step(self, action):
        """
        Steps through the log data based on the action.
        Returns the next state, reward, and done flag.
        """
        if self.current_index >= len(self.log_data) - 1:
            return None, 0, True

        current_log = self.log_data[self.current_index]
        next_log = self.log_data[self.current_index + 1]

        # Simulate rewards and penalties based on parsed data
        reward = current_log['reward'] if current_log['action'] == action else -10
        done = (self.current_index >= len(self.log_data) - 1)

        self.current_index += 1
        return next_log['state'], reward, done
# Define the reinforcement learning agent
class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # Random action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()  # Best action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).detach()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Main training loop
def train_rl_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(list(state.values()))
            next_state, reward, done = env.step(action)
            agent.remember(list(state.values()), action, reward, list(next_state.values()), done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay()
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Initialize environment and agent
state_size = len(RoguelikeEnvironment({}).get_initial_state())
action_size = 5  # Example: move, attack, use item, etc.
log_file = 'dcss_game_log.txt'
parser = DCSSLogParser(log_file)
parsed_log_data = parser.parse_log()

# Initialize environment and RL agent
state_size = len(parsed_log_data[0]['state']) if parsed_log_data else 0
action_size = 5  # Example: move, attack, use item, etc.

env = DCSSRLAgentEnvironment(parsed_log_data)
agent = RLAgent(state_size, action_size)

# Train the RL agent using the parsed game log
train_rl_agent(env, agent)

# Train the agent
train_rl_agent(env, agent)
