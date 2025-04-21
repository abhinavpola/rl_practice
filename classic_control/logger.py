import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


class Logger:
    """Logger to store training metrics for DQN"""

    def __init__(self):
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "grad_norms": [],
            "epsilons": [],
            "avg_q_values": [],
            "avg_future_rewards": [],
        }

        # For storing per-episode metrics
        self.current_episode = {
            "rewards": [],
            "losses": [],
            "grad_norms": [],
            "epsilons": [],
            "q_values": [],
            "future_rewards": [],
        }

        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/run_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

    def log_step(
        self,
        loss: float = None,
        grad_norm: float = None,
        epsilon: float = None,
        q_value: float = None,
        future_reward: float = None,
        reward: float = None,
    ):
        """Log metrics for a single training step"""
        if loss is not None:
            self.current_episode["losses"].append(loss)
        if grad_norm is not None:
            self.current_episode["grad_norms"].append(grad_norm)
        if epsilon is not None:
            self.current_episode["epsilons"].append(epsilon)
        if q_value is not None:
            self.current_episode["q_values"].append(q_value)
        if future_reward is not None:
            self.current_episode["future_rewards"].append(future_reward)
        if reward is not None:
            self.current_episode["rewards"].append(reward)

    def end_episode(self):
        """Store metrics for the completed episode"""
        # Store episode total reward and length
        self.metrics["episode_rewards"].append(sum(self.current_episode["rewards"]))
        self.metrics["episode_lengths"].append(len(self.current_episode["rewards"]))

        # Store averages of other metrics
        if self.current_episode["losses"]:
            self.metrics["losses"].append(np.mean(self.current_episode["losses"]))
        if self.current_episode["grad_norms"]:
            self.metrics["grad_norms"].append(
                np.mean(self.current_episode["grad_norms"])
            )
        if self.current_episode["epsilons"]:
            self.metrics["epsilons"].append(np.mean(self.current_episode["epsilons"]))
        if self.current_episode["q_values"]:
            self.metrics["avg_q_values"].append(
                np.mean(self.current_episode["q_values"])
            )
        if self.current_episode["future_rewards"]:
            self.metrics["avg_future_rewards"].append(
                np.mean(self.current_episode["future_rewards"])
            )

        # Reset current episode metrics
        self.current_episode = {key: [] for key in self.current_episode}

    def save_metrics(self):
        """Save metrics to disk"""
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f)

    def plot_metrics(self):
        """Plot all tracked metrics"""
        metrics_to_plot = [
            ("episode_rewards", "Episode Rewards"),
            ("episode_lengths", "Episode Lengths"),
            ("losses", "Average Loss per Episode"),
            ("grad_norms", "Average Gradient Norm per Episode"),
            ("epsilons", "Average Epsilon per Episode"),
            ("avg_q_values", "Average Q-Values per Episode"),
            ("avg_future_rewards", "Average Future Rewards per Episode"),
        ]

        fig, axes = plt.subplots(
            len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot))
        )
        fig.suptitle("Training Metrics")

        for (metric_name, title), ax in zip(metrics_to_plot, axes):
            if self.metrics[metric_name]:  # Only plot if we have data
                ax.plot(self.metrics[metric_name])
                ax.set_title(title)
                ax.set_xlabel("Episode")
                ax.grid(True)

        plt.tight_layout()
        plot_file = os.path.join(self.log_dir, "training_plots.png")
        plt.savefig(plot_file)
        plt.close()
