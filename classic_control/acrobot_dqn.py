import gymnasium as gym
from tinygrad import nn, Tensor
import random
from collections import deque
import numpy as np
from typeguard import typechecked
import os
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from logger import Logger


class MSELoss:
    """
    Mean Squared Error Loss
    """

    @typechecked
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        assert (
            pred.shape == target.shape
        ), f"Shape mismatch: {pred.shape} vs {target.shape}"
        return ((pred - target) ** 2).mean()  # Average over batch


class ReplayBuffer:
    """
    Replay Buffer stores transitions (state: (6,), action: (1,), reward: (1,), next_state: (6,), done: (1,))
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    @typechecked
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    @typechecked
    def sample(self, batch_size) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Get random indices
        indices = random.sample(range(len(self.buffer)), batch_size)

        # Convert all batch items to numpy arrays first
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []

        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            next_states_list.append(next_state)
            dones_list.append(float(done))

        # Create tensors with proper shapes
        states = Tensor(np.array(states_list, dtype=np.float32))
        actions = Tensor(np.array(actions_list, dtype=np.int32).reshape(-1, 1))
        rewards = Tensor(np.array(rewards_list, dtype=np.float32).reshape(-1, 1))
        next_states = Tensor(np.array(next_states_list, dtype=np.float32))
        dones = Tensor(np.array(dones_list, dtype=np.float32).reshape(-1, 1))

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQN:
    def __init__(self, hidden_size=128):
        self.l1 = nn.Linear(6, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 256)
        self.l4 = nn.Linear(256, 3)

    @typechecked
    def __call__(self, x: Tensor) -> Tensor:
        """
        x: Tensor of (batch_size, 6)
        Returns: Tensor of (batch_size, 3)
        """
        x = self.l1(x).leaky_relu()
        x = self.l2(x).leaky_relu()
        x = self.l3(x).leaky_relu()
        return self.l4(x)


class Agent:
    def __init__(
        self,
        learning_rate=0.001,
        epsilon_decay=500,
        hidden_size=128,
        gamma=0.99,
        sync_interval=1000,
    ):
        self.dqn = DQN(hidden_size=hidden_size)
        self.target_dqn = DQN(hidden_size=hidden_size)
        load_state_dict(self.target_dqn, get_state_dict(self.dqn))

        self.optim = nn.optim.Adam(nn.state.get_parameters(self.dqn), lr=learning_rate)
        self.loss = MSELoss()

        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.sync_interval = sync_interval
        self.model_path = "acrobot_dqn_model.safetensors"
        self.logger = Logger()

    @typechecked
    def select_action(self, state: Tensor) -> int:
        assert state.shape == (6,), f"Select action input shape: {state.shape}"
        # calculate epsilon after decay
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        # Log epsilon
        self.logger.log_step(epsilon=eps)

        self.steps_done += 1
        if random.random() < eps:
            return random.randint(0, 2)  # 0, 1, or 2
        else:
            Tensor.training = False
            q_values = self.dqn(state)
            Tensor.training = True

            # Log Q-values
            self.logger.log_step(q_value=float(q_values.max().numpy()))

            return int(q_values.argmax().numpy())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Get batch Tensors
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values
        q_preds = self.dqn(states)
        assert q_preds.shape == (self.batch_size, 3), f"Q-values shape: {q_preds.shape}"
        assert actions.shape == (self.batch_size, 1), f"Actions shape: {actions.shape}"
        current_q_values = q_preds.gather(1, actions).squeeze()  # shape (64,)

        # Next Q-values
        Tensor.training = False
        next_q_values = self.target_dqn(next_states)
        next_q_max = next_q_values.max(axis=1, keepdim=True)
        target_q = rewards + self.gamma * next_q_max * (1 - dones)
        Tensor.training = True

        # Compute loss
        loss = self.loss(current_q_values, target_q.squeeze())

        self.optim.zero_grad()
        loss.backward()

        # Calculate gradient norm
        grad_norm = 0.0
        for p in nn.state.get_parameters(self.dqn):
            if p.grad is not None:
                grad_norm += float((p.grad**2).sum().numpy())
        grad_norm = np.sqrt(grad_norm)

        # Log metrics
        self.logger.log_step(
            loss=float(loss.numpy()),
            grad_norm=grad_norm,
            future_reward=float(next_q_max.mean().numpy()),
        )

        self.optim.step()

    def sync_target(self):
        load_state_dict(self.target_dqn, get_state_dict(self.dqn))

    def save_model(self):
        state_dict = get_state_dict(self.dqn)
        safe_save(state_dict, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        if os.path.exists(self.model_path):
            state_dict = safe_load(self.model_path)
            load_state_dict(self.dqn, state_dict)
            load_state_dict(self.target_dqn, state_dict)
            print(f"Model loaded from {self.model_path}")
            return True
        return False

    def play_game(self, env):
        state, info = env.reset()
        state_t = Tensor(state)
        total_reward = 0
        done = False

        while not done:
            Tensor.training = False
            q_values = self.dqn(state_t)
            action = int(q_values.argmax().numpy())

            next_state, reward, terminated, truncated, info = env.step(action)
            state_t = Tensor(next_state)
            total_reward += reward
            done = terminated or truncated

        return total_reward


def train_and_evaluate(hyperparams, num_eval_episodes=5):
    # Create agent with specified hyperparameters
    agent = Agent(
        learning_rate=hyperparams["learning_rate"],
        epsilon_decay=hyperparams["epsilon_decay"],
        hidden_size=hyperparams["hidden_size"],
        gamma=hyperparams["gamma"],
        sync_interval=hyperparams["sync_interval"],
    )

    # Training environment
    training_env = gym.make("Acrobot-v1")

    # Training loop
    total_steps = 0
    num_episodes = 200  # Reduced for grid search
    max_steps = 200

    with Tensor.train():
        for ep in range(1, num_episodes + 1):
            state, info = training_env.reset()
            state_t = Tensor(state)
            episode_reward = 0

            for t in range(max_steps):
                action = agent.select_action(state_t)
                next_state, reward, terminated, truncated, info = training_env.step(
                    action
                )
                next_state_t = Tensor(next_state)
                agent.replay_buffer.push(state, action, reward, next_state, terminated)
                agent.update()

                # Log reward
                agent.logger.log_step(reward=reward)

                episode_reward += reward
                state = next_state
                state_t = next_state_t
                total_steps += 1

                if total_steps % agent.sync_interval == 0:
                    agent.sync_target()

                if terminated or truncated:
                    break

            # End episode in logger
            agent.logger.end_episode()

            if ep % 50 == 0:
                print(f"Episode {ep} finished with reward {episode_reward}")

    # Save and plot metrics
    agent.logger.save_metrics()
    agent.logger.plot_metrics()

    # Evaluate the trained agent
    eval_env = gym.make("Acrobot-v1")
    eval_rewards = []

    for _ in range(num_eval_episodes):
        state, info = eval_env.reset()
        state_t = Tensor(state)
        episode_reward = 0
        done = False

        while not done:
            Tensor.training = False
            q_values = agent.dqn(state_t)
            action = int(q_values.argmax().numpy())

            next_state, reward, terminated, truncated, info = eval_env.step(action)
            state_t = Tensor(next_state)
            episode_reward += reward
            done = terminated or truncated

        eval_rewards.append(episode_reward)

    avg_reward = sum(eval_rewards) / len(eval_rewards)
    return avg_reward, agent


if __name__ == "__main__":
    # Define hyperparameters
    hyperparams = {
        "learning_rate": 1e-4,
        "epsilon_decay": 1e-4,
        "hidden_size": 128,
        "gamma": 0.99,
        "sync_interval": 1000,
    }

    print(f"Training with hyperparameters: {hyperparams}")

    # Train and evaluate with these hyperparameters
    avg_reward, agent = train_and_evaluate(hyperparams)

    print(f"Average evaluation reward: {avg_reward}")

    # Save the model
    model_path = "acrobot_dqn_model.safetensors"
    state_dict = get_state_dict(agent.dqn)
    safe_save(state_dict, model_path)
    print(f"Model saved to {model_path}")

    # Play a demo game with the trained agent
    print("\nPlaying a game with the trained agent:")
    demo_env = gym.make("Acrobot-v1", render_mode="human")
    reward = agent.play_game(demo_env)
    print(f"Game finished with reward {reward}")
    demo_env.close()
