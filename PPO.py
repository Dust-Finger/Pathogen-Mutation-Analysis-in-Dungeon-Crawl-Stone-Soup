class PPOPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, action_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.action_layer(x), self.value_layer(x)

def train_ppo(env, policy_network, optimizer, gamma, epsilon_clip, num_epochs=10):
    states, actions, rewards, dones, log_probs = [], [], [], [], []

    # Rollout phase
    state = env.reset()
    for _ in range(1000):  # Example max steps
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = policy_network(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done = env.step(action.item())
        log_prob = dist.log_prob(action)

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        state = next_state
        if done:
            break

    # Compute returns
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + gamma * G * (1 - done)
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    # Training phase
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.cat(log_probs)

    for _ in range(num_epochs):
        logits, values = policy_network(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - values.detach().squeeze()
        policy_loss = -torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        ).mean()

        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
