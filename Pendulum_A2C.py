from torch.distributions import Normal
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
batch_size = 20

scale_factor = 2.0
min_value = -2.0
max_value = 2.0
entropy_bonus_parameter = 0.01

model_path = "checkpoints/pendulum_a2c_swingup_ep20000.pt"

load_existing_model = True
play_only = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# =============================
# MODEL
# =============================
class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.actor_mean = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(out_dim))

        self.critic = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128,1)            
        )

    def  forward(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)).to(device)
        mean, std, value = self.forward(x)
        dist = torch.distributions.Normal(loc = mean, scale = std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis = -1)
        entropy = dist.entropy().sum(axis = -1)

        action_scaled = action * scale_factor
        action_clipped = torch.clamp(action_scaled, min_value, max_value)

        return action_clipped.cpu().numpy(), log_prob, value, entropy

    def act_greedy(self, state):
        x = torch.from_numpy(state.astype(np.float32)).to(device)

        with torch.no_grad():
            mean, std, value = self.forward(x) #action will be mean, which is the best action
            action_scaled = mean * scale_factor
            action_clipped = torch.clamp(action_scaled, min_value, max_value)

        return action_clipped.cpu().numpy()



# =============================
# SAVE / LOAD
# =============================
def save_model(model, path=model_path):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Loaded model")



# =============================
# LOSS
# =============================
def compute_losses(log_probs, values, rewards, next_values, dones, entropies):

    values = torch.stack(values).squeeze(-1)
    next_values = torch.stack(next_values).squeeze(-1)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    gamma = 0.99
    lam = 0.95   # GAE lambda

    # ===== TD residual (delta) =====
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # ===== GAE Advantage =====
    advantages = torch.zeros_like(deltas).to(device)

    gae = 0

    for t in reversed   (range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    # ===== Value target =====
    targets = advantages + values

    # GAE mà không normalize → rất dễ rung, nên thêm:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


    # ===== DEBUG =====
    print("\n===== DEBUG GAE =====")
    for i in range(len(values)):
        print(f"[Step {i}]")
        print(f" V(s): {values[i].item():.4f}")
        print(f" V(s_next): {next_values[i].item():.4f}")
        print(f" Reward: {rewards[i]:.2f}")
        print(f" Delta: {deltas[i].item():.4f}")
        print(f" Advantage(GAE): {advantages[i].item():.4f}")

    # ===== LOSSES =====
    critic_loss = (values - targets.detach()).pow(2).mean()

    entropy_loss = entropy_bonus_parameter * entropies.mean()
    actor_loss = -((log_probs * advantages.detach()).mean() + entropy_loss)


    print(f"Critic loss: {critic_loss.item():.4f}")
    print(f"Actor loss: {actor_loss.item():.4f}")

    return actor_loss, critic_loss


# =============================
# TRAIN
# =============================
def train(model, env):

    optimizer_actor = optim.Adam(
        list(model.actor_mean.parameters()) + [model.log_std],
        lr=3e-4
    )

    optimizer_critic = optim.Adam(model.critic.parameters(), lr=1e-3)

    recent_rewards = deque(maxlen=100)

    # Initialize TensorBoard writer
    log_dir = os.path.join("runs", f"Pendulumn_A2C_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging active. Logging to {log_dir}")
    print("To view training progress, run: tensorboard --logdir runs")

    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    for episode in range(max_episodes):

        model.train()

        state, _ = env.reset()

        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []

        total_reward = 0

        for t in range(500):

            action, log_prob, value, entropy = model.act(state)

            next_state, _, terminated, truncated, _ = env.step(action)

            cos_theta, sin_theta, theta_dot = state

            theta = np.arctan2(sin_theta, cos_theta)

            # Custumize reward function
            reward = - (theta**2 + 0.1 * theta_dot**2 + 0.001 * action[0]**2)
            done = terminated or truncated

            # store
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy)

            total_reward += reward
            state = next_state

            if done:
                break

        # Compute next_values by shifting values and calculating V(s_T) only for the final state
        final_state_tensor = torch.from_numpy(state.astype(np.float32)).to(device)
        with torch.no_grad():
            _, _, final_value = model(final_state_tensor)
            if done:
                final_value = torch.zeros_like(final_value)

        next_values = values[1:] + [final_value]

        actor_loss, critic_loss = compute_losses(
            log_probs, values, rewards, next_values, dones, entropies
        )

        (actor_loss / batch_size).backward()
        (critic_loss / batch_size).backward()

        if (episode + 1) % batch_size == 0:
            optimizer_actor.step()
            optimizer_critic.step()
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()


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

        # Save checkpoint every 500 episodes to track training progress
        if (episode + 1) % 500 == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"pendulum_a2c_swingup_ep{episode+1}.pt")
            save_model(model, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if len(recent_rewards) == 100 and avg_reward > 475:
            print("Solved!")
            break

    # Close the TensorBoard writer
    writer.close()




# =============================
# PLAY
# =============================
def play(model, num_episodes=5):

    env = gym.make("Pendulum-v1", render_mode="human")

    model.eval()

    for ep in range(num_episodes):

        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):

            action = model.act_greedy(state)

            state, _, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            cos_theta, sin_theta, theta_dot = state

            theta = np.arctan2(sin_theta, cos_theta)

            # Custumize reward function
            reward = - (theta**2 + 0.1 * theta_dot**2 + 0.001 * action[0]**2)

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

    env = gym.make("Pendulum-v1")

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]

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

