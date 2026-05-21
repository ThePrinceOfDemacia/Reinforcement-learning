from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os


# ============================================================
# CONFIG
# ============================================================

algorithm = "sarsa"   # "sarsa" or "qlearning"

gamma = 0.99
lr = 1e-3

batch_size = 64
memory_size = 10000

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

target_update_freq = 20

max_episodes = 2000

model_path = f"models/cartpole_{algorithm}.pt"

load_existing_model = False
play_only = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Q NETWORK
# ============================================================

class QNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.model(x)

    def act(self, state, epsilon):
        """
        epsilon-greedy action
        """

        if random.random() < epsilon:
            return random.randint(0, out_dim - 1)

        x = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            q_values = self.forward(x)

        action = torch.argmax(q_values).item()

        return action

    def act_greedy(self, state):

        x = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            q_values = self.forward(x)

        return torch.argmax(q_values).item()


# ============================================================
# REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, experience):

        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ============================================================
# SAVE / LOAD
# ============================================================

def save_model(model, path):

    torch.save(model.state_dict(), path)

    print(f"Saved model to: {path}")


def load_model(model, path):

    state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)

    print(f"Loaded model from: {path}")


# ============================================================
# TRAIN STEP
# ============================================================

def train_step(
    q_net,
    target_net,
    optimizer,
    memory,
):

    if len(memory) < batch_size:
        return None

    batch = memory.sample(batch_size)

    states = torch.tensor(
        np.array([x[0] for x in batch]),
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        [x[1] for x in batch],
        dtype=torch.long,
        device=device
    )

    rewards = torch.tensor(
        [x[2] for x in batch],
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        np.array([x[3] for x in batch]),
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        [x[4] for x in batch],
        dtype=torch.float32,
        device=device
    )

    next_actions = torch.tensor(
        [x[5] for x in batch],
        dtype=torch.long,
        device=device
    )

    # -----------------------------------------
    # Q(s,a)
    # -----------------------------------------

    q_values = q_net(states)

    current_q = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # -----------------------------------------
    # TARGET
    # -----------------------------------------

    with torch.no_grad():

        next_q_values = target_net(next_states)

        if algorithm == "sarsa":

            # SARSA:
            # Q(s', a')

            next_q = next_q_values.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

        elif algorithm == "qlearning":

            # Q-learning:
            # max_a Q(s', a)

            next_q = next_q_values.max(dim=1)[0]

        else:
            raise ValueError("Unknown algorithm")

        target_q = rewards + gamma * (1 - dones) * next_q

    # -----------------------------------------
    # LOSS
    # -----------------------------------------

    loss_fn = nn.MSELoss()

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


# ============================================================
# TRAIN
# ============================================================

def train(q_net, target_net, env):

    global epsilon

    optimizer = optim.Adam(
        q_net.parameters(),
        lr=lr
    )

    memory = ReplayMemory(memory_size)

    for episode in range(max_episodes):

        state, _ = env.reset()

        total_reward = 0

        action = q_net.act(state, epsilon)

        for t in range(500):

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            total_reward += reward

            # -------------------------------------
            # next action
            # -------------------------------------

            next_action = q_net.act(
                next_state,
                epsilon
            )

            # -------------------------------------
            # store transition
            # -------------------------------------

            memory.push((
                state,
                action,
                reward,
                next_state,
                done,
                next_action
            ))

            # -------------------------------------
            # train
            # -------------------------------------

            loss = train_step(
                q_net,
                target_net,
                optimizer,
                memory
            )

            state = next_state
            action = next_action

            if done:
                break

        # -----------------------------------------
        # epsilon decay
        # -----------------------------------------

        epsilon = max(
            epsilon * epsilon_decay,
            epsilon_min
        )

        # -----------------------------------------
        # target update
        # -----------------------------------------

        if episode % target_update_freq == 0:
            target_net.load_state_dict(
                q_net.state_dict()
            )

        print(
            f"Episode: {episode}, "
            f"reward: {total_reward}, "
            f"epsilon: {epsilon:.3f}, "
            f"loss: {loss}"
        )

        save_model(q_net, model_path)

        if total_reward >= 500:
            print("Solved!")
            break

    return q_net


# ============================================================
# PLAY
# ============================================================

def play(q_net, num_episodes=5):

    env = gym.make(
        "CartPole-v1",
        render_mode="human"
    )

    q_net.eval()

    for episode in range(num_episodes):

        state, _ = env.reset()

        total_reward = 0

        for _ in range(500):

            action = q_net.act_greedy(state)

            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            total_reward += reward

            time.sleep(0.01)

            if done:
                break

        print(
            f"Play episode {episode + 1}, "
            f"reward: {total_reward}"
        )

    env.close()


# ============================================================
# MAIN
# ============================================================

def main():

    train_env = gym.make("CartPole-v1")

    global out_dim

    in_dim = train_env.observation_space.shape[0]

    out_dim = train_env.action_space.n

    q_net = QNet(
        in_dim,
        out_dim
    ).to(device)

    target_net = QNet(
        in_dim,
        out_dim
    ).to(device)

    target_net.load_state_dict(
        q_net.state_dict()
    )

    if load_existing_model and os.path.exists(model_path):

        load_model(q_net, model_path)

        target_net.load_state_dict(
            q_net.state_dict()
        )

    else:
        print("Training from scratch.")

    if not play_only:

        train(
            q_net,
            target_net,
            train_env
        )

        save_model(q_net, model_path)

    train_env.close()

    print("Now playing...")

    play(q_net)


if __name__ == "__main__":
    main()