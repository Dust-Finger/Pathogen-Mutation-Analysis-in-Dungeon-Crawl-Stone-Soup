def train_agent(env, dqn_network, ppo_policy, replay_buffer, dqn_optimizer, ppo_optimizer, episodes, gamma, epsilon_strategy):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        epsilon = epsilon_strategy.get_epsilon(episode)

        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.action_space - 1)  # Random action
            else:
                action = torch.argmax(dqn_network(torch.FloatTensor(state))).item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Update DQN
            dqn_loss = train_dqn(env, replay_buffer, dqn_network, target_network, dqn_optimizer, batch_size=64, gamma=gamma)

            # Update PPO
            train_ppo(env, ppo_policy, ppo_optimizer, gamma, epsilon_clip=0.2)

            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
