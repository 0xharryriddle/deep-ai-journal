import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
# Try importing clear_output, but provide a fallback if not available
try:
    from IPython.display import clear_output
except ImportError:
    # Define a no-op clear_output function if IPython is not available
    def clear_output(wait=False):
        pass

env_id = 'Taxi-v3'
env = gym.make(env_id, render_mode='rgb_array')

def initialize_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space))
    return q_table

def greedy_policy(q_table, state):
    action = np.argmax(q_table[state, :])
    return action

def epsilon_greedy_policy(q_table, state, epsilon):
    rand_n = float(np.random.uniform(0, 1))

    if rand_n > epsilon:
        action = greedy_policy(q_table, state)
    else:
        action = np.random.choice(q_table.shape[1])

    return action

n_training_episodes = 30000
n_eval_episodes = 100
lr = 0.7

max_steps = 99
gamma = 0.95
eval_seed = range(n_eval_episodes)

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

def train(env, max_steps, q_table, n_training_episodes, min_epsilon, max_epsilon, decay_rate, lr, gamma):
    # Keep track of metrics for visualization
    rewards_per_episode = []
    
    for episode in range(n_training_episodes):
        # Show progress every 1000 episodes
        if episode % 1000 == 0:
            print(f"Episode {episode}/{n_training_episodes}")
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        total_episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(q_table, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_episode_reward += reward
            
            # Update Q-table
            q_table[state, action] += lr * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            if terminated or truncated:
                break

            state = next_state
            
        # Store the rewards for this episode
        rewards_per_episode.append(total_episode_reward)
        
        # Plot training progress every 5000 episodes
        if episode % 5000 == 0 and episode > 0:
            # Calculate rolling average of rewards
            rolling_length = 500
            if len(rewards_per_episode) >= rolling_length:
                rolling_avg = np.mean(rewards_per_episode[-rolling_length:])
                print(f"Episode {episode}/{n_training_episodes}, Average Reward (last {rolling_length}): {rolling_avg:.2f}")

    # Plot the training rewards at the end
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.show()

    return q_table

def evaluate_agent(env, max_steps, n_eval_episodes, q_table, seed):
    """
    Evaluate the agent's performance after training
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        
        total_rewards_ep = 0
        
        for step in range(max_steps):
            # Select the action with highest value given a state
            action = greedy_policy(q_table, state)
            
            # Take the action and observe the outcome
            new_state, reward, terminated, truncated, info = env.step(action)
            
            total_rewards_ep += reward
            
            if terminated or truncated:
                break
            
            state = new_state
        
        episode_rewards.append(total_rewards_ep)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward

def render_episode(env, q_table, max_steps):
    """Render one episode of the agent's performance"""
    state, info = env.reset()
    total_reward = 0
    
    plt.figure(figsize=(8, 6))
    
    for step in range(max_steps):
        action = greedy_policy(q_table, state)
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print information about the current step
        print(f"Step: {step+1}")
        print(f"State: {state}, Action: {action}, Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        
        # Render the environment
        frame = env.render()
        plt.clf()  # Clear the current figure
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Step {step+1}, Total Reward: {total_reward}")
        plt.draw()
        plt.pause(0.1)  # Short pause to update the plot
        time.sleep(0.4)  # Additional sleep to slow down the visualization
        
        if terminated or truncated:
            print("Episode finished!")
            break
            
        state = new_state
    
    # Keep the final frame visible for a few seconds
    plt.pause(2)
    plt.close()

state_space = 500
action_space = 6

q_table = initialize_q_table(state_space=state_space, action_space=action_space)

print("Training the agent...")
trained_q_table = train(env, max_steps, q_table, n_training_episodes, min_epsilon, max_epsilon, decay_rate, lr, gamma)

# Evaluate the trained agent
print("Evaluating the trained agent...")
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, trained_q_table, eval_seed)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Render a single episode
print("\nRendering an episode with the trained agent...")
render_episode(env, trained_q_table, max_steps)

# Display a visualization of the Q-table
print("\nVisualizing the Q-table...")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
im = ax.imshow(trained_q_table)
plt.colorbar(im)
plt.xlabel("Actions")
plt.ylabel("States")
plt.title("Q-table")
plt.show()
