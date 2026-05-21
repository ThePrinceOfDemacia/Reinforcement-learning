from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os


gamma = 0.99
batch_size = 1
max_episodes = 20000

model_path = "cartpole_reinforce_pi_3.pt"

load_existing_model = True
play_only = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# POLICY NETWORK
# ============================================================

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def act(self, state):
        """
        Sample action từ policy
        """
        x = torch.from_numpy(state.astype(np.float32)).to(device)

        logits = self.forward(x)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def act_greedy(self, state):
        """
        Dùng khi test
        """
        x = torch.from_numpy(state.astype(np.float32)).to(device)

        with torch.no_grad():
            logits = self.forward(x)
            action = torch.argmax(logits).item()

        return action


# ============================================================
# SAVE / LOAD
# ============================================================

def save_model(pi, path):
    torch.save(pi.state_dict(), path)
    print(f"Saved model to: {path}")


def load_model(pi, path):
    state_dict = torch.load(path, map_location=device)
    pi.load_state_dict(state_dict)
    pi.to(device)
    pi.eval()
    print(f"Loaded model from: {path}")


# ============================================================
# RETURN
# ============================================================

def compute_returns(rewards):
    T = len(rewards)
    returns = np.empty(T, dtype=np.float32)

    future_return = 0.0

    for t in reversed(range(T)):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return

    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    return returns


# ============================================================
# LOSS
# ============================================================

def compute_episode_loss(log_probs, rewards):

    returns = compute_returns(rewards)

    # Normalize để giảm variance
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)

    loss = -log_probs * returns
    loss = loss.sum()

    return loss


# ============================================================
# TRAIN
# ============================================================

def train_policy(pi, env):

    optimizer = optim.Adam(pi.parameters(), lr=1e-2)

    episode_count = 0

    # lưu 400 rewards gần nhất
    recent_rewards = deque(maxlen=400)

    while episode_count < max_episodes:

        pi.train()

        batch_losses = []
        batch_rewards = []

        for _ in range(batch_size):

            state, _ = env.reset()

            episode_log_probs = []
            episode_rewards = []

            for t in range(500):

                action, log_prob = pi.act(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)

                state = next_state

                if done:
                    break

            episode_loss = compute_episode_loss(
                episode_log_probs,
                episode_rewards,
            )

            total_reward = sum(episode_rewards)

            batch_losses.append(episode_loss)
            batch_rewards.append(total_reward)

            # thêm vào moving window
            recent_rewards.append(total_reward)

            episode_count += 1

            if episode_count >= max_episodes:
                break

        batch_loss = torch.stack(batch_losses).mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        avg_reward = np.mean(batch_rewards)

        # moving average 400 episodes
        avg_reward_400 = np.mean(recent_rewards)

        print(
            f"Episodes: {episode_count}, "
            f"loss: {batch_loss.item():.4f}, "
            f"batch_avg_reward: {avg_reward:.2f}, "
            f"avg_reward_400: {avg_reward_400:.2f}"
        )

        save_model(pi, model_path)

        # chỉ check khi đủ 400 episodes
        if len(recent_rewards) == 400 and avg_reward_400 >= 475:
            print("Solved!")
            break

    return pi


# ============================================================
# PLAY
# ============================================================

def play(pi, num_episodes=5):

    env = gym.make("CartPole-v1", render_mode="human")

    pi.eval()

    for episode in range(num_episodes):

        state, _ = env.reset()
        total_reward = 0

        for t in range(500):

            action = pi.act_greedy(state)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            time.sleep(0.01)

            if done:
                break

        print(f"Episode {episode+1}, reward: {total_reward}")

    env.close()


# ============================================================
# MAIN
# ============================================================

def main():

    env = gym.make("CartPole-v1")

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    pi = Pi(in_dim, out_dim).to(device)

    if load_existing_model and os.path.exists(model_path):
        load_model(pi, model_path)
    else:
        print("Training from scratch.")

    if not play_only:
        pi = train_policy(pi, env)
        save_model(pi, model_path)

    env.close()

    print("Now playing...")
    play(pi)


if __name__ == "__main__":
    main()