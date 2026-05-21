from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import os
from torch.utils.tensorboard import SummaryWriter

# =============================
# HYPERPARAM
# =============================
gamma = 0.99
max_episodes = 20000

model_path = "cartpole_actor_critic.pt"

load_existing_model = False
play_only = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# MODEL
# =============================
class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)).to(device)

        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def act_greedy(self, state):
        x = torch.from_numpy(state.astype(np.float32)).to(device)
        with torch.no_grad():
            logits, _ = self.forward(x)
            action = torch.argmax(logits).item()
        return action

# =============================
# SAVE / LOAD
# =============================
def save_model(model):
    torch.save(model.state_dict(), model_path)

def load_model(model):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Loaded model")

# =============================
# LOSS
# =============================
def compute_losses(log_probs, values, rewards, next_values, dones):

    values = torch.stack(values).squeeze()
    next_values = torch.stack(next_values).squeeze()
    log_probs = torch.stack(log_probs)

    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # ===== TD target =====
    targets = rewards + gamma * next_values * (1 - dones)

    # ===== Advantage =====
    advantages = targets - values

    # ===== DEBUG PRINT =====
    print("\n===== DEBUG CRITIC =====")
    for i in range(len(values)):
        print(f"[Step {i}]")
        print(f" V(s): {values[i].item():.4f}")
        print(f" V(s_next): {next_values[i].item():.4f}")
        print(f" Reward: {rewards[i]:.2f}")
        print(f" Target: {targets[i].item():.4f}")
        print(f" Advantage: {advantages[i].item():.4f}")

    # ===== LOSSES =====
    critic_loss = (values - targets.detach()).pow(2).mean()
    actor_loss = -(log_probs * advantages.detach()).mean()

    print(f"Critic loss: {critic_loss.item():.4f}")
    print(f"Actor loss: {actor_loss.item():.4f}")

    return actor_loss, critic_loss

# =============================
# TRAIN
# =============================
def train(model, env):

    optimizer_actor = optim.Adam(model.actor.parameters(), lr=1e-3)
    optimizer_critic = optim.Adam(model.critic.parameters(), lr=1e-3)

    recent_rewards = deque(maxlen=100)

    # Initialize TensorBoard writer
    log_dir = os.path.join("runs", f"CartPole_A2C_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging active. Logging to {log_dir}")
    print("To view training progress, run: tensorboard --logdir runs")

    for episode in range(max_episodes):

        model.train()

        state, _ = env.reset()

        log_probs = []
        values = []
        rewards = []
        next_values = []
        dones = []

        total_reward = 0

        for t in range(500):

            action, log_prob, value = model.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # next value
            next_state_tensor = torch.from_numpy(next_state.astype(np.float32)).to(device)
            with torch.no_grad():
                _, next_value = model(next_state_tensor)

            # store
            log_probs.append(log_prob)
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            dones.append(done)

            total_reward += reward
            state = next_state

            if done:
                break

        actor_loss, critic_loss = compute_losses(
            log_probs, values, rewards, next_values, dones
        )

        # ===== UPDATE CRITIC =====
        optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        optimizer_critic.step()
        print(">>> Critic updated")

        # ===== UPDATE ACTOR =====
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        print(">>> Actor updated")

        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)

        # Log to TensorBoard
        writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
        writer.add_scalar("Loss/Critic", critic_loss.item(), episode)
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Reward/Avg100", avg_reward, episode)

        print(
            f"\nEpisode {episode+1}, reward: {total_reward}, avg100: {avg_reward:.2f}"
        )

        save_model(model)

        if len(recent_rewards) == 100 and avg_reward > 475:
            print("Solved!")
            break

    # Close the TensorBoard writer
    writer.close()

# =============================
# PLAY
# =============================
def play(model, num_episodes=5):

    env = gym.make("CartPole-v1", render_mode="human")

    model.eval()

    for ep in range(num_episodes):

        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):

            action = model.act_greedy(state)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            time.sleep(0.01)

            if done:
                break

        print(f"Episode {ep+1}, reward: {total_reward}")

    env.close()

# =============================
# MAIN
# =============================
def main():

    env = gym.make("CartPole-v1")

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    model = ActorCritic(in_dim, out_dim).to(device)

    if load_existing_model and os.path.exists(model_path):
        load_model(model)
    else:
        print("Training from scratch")

    if not play_only:
        train(model, env)

    env.close()

    print("Playing...")
    play(model)

if __name__ == "__main__":
    main()