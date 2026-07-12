import csv
import os

import matplotlib.pyplot as plt

import config
from agent import create_agent, load_model
from env_wrapper import make_env
from evaluation import evaluate_policy


def plot_history(history_path=config.HISTORY_PATH):
    # Load the training CSV and draw reward and epsilon curves.
    if not os.path.exists(history_path):
        print(f"NO HISTORY FILE FOUND: {history_path}")
        return

    with open(history_path, newline="") as history_file:
        rows = list(csv.DictReader(history_file))
    if not rows:
        print(f"NO EPISODES FOUND IN: {history_path}")
        return

    episodes = [int(row["Episode"]) for row in rows]
    rewards = [float(row["Train_Reward"]) for row in rows]
    epsilons = [float(row["Epsilon"]) for row in rows]
    eval_episodes = []
    eval_rewards = []
    eval_stds = []

    for row in rows:
        value = row.get("Eval_Mean", row.get("Eval_Reward", ""))
        if value:
            eval_episodes.append(int(row["Episode"]))
            eval_rewards.append(float(value))
            eval_stds.append(float(row.get("Eval_Std", 0.0) or 0.0))

    window = min(50, len(rewards))
    rolling_rewards = [
        sum(rewards[max(0, index - window + 1) : index + 1])
        / min(index + 1, window)
        for index in range(len(rewards))
    ]

    figure, reward_axis = plt.subplots(figsize=(11, 5))
    reward_axis.plot(episodes, rewards, color="tab:blue", alpha=0.2, label="Train reward")
    reward_axis.plot(
        episodes,
        rolling_rewards,
        color="tab:blue",
        label=f"Train reward ({window}-episode mean)",
    )
    if eval_rewards:
        reward_axis.errorbar(
            eval_episodes,
            eval_rewards,
            yerr=eval_stds,
            fmt="o-",
            color="tab:green",
            capsize=3,
            label="Fixed-seed evaluation mean +/- std",
        )
    reward_axis.set_xlabel("Episode")
    reward_axis.set_ylabel("Reward", color="tab:blue")
    reward_axis.tick_params("y", colors="tab:blue")
    reward_axis.grid(alpha=0.2)

    epsilon_axis = reward_axis.twinx()
    epsilon_axis.plot(episodes, epsilons, "r--", label="Epsilon")
    epsilon_axis.set_ylabel("Epsilon", color="tab:red")
    epsilon_axis.tick_params("y", colors="tab:red")

    reward_handles, reward_labels = reward_axis.get_legend_handles_labels()
    epsilon_handles, epsilon_labels = epsilon_axis.get_legend_handles_labels()
    figure.legend(
        reward_handles + epsilon_handles,
        reward_labels + epsilon_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Assault Training Learning Curve", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()


def get_num_actions():
    # Read the action space size from a live environment.
    env = make_env(render_mode=None)
    try:
        return int(env.action_space.n)
    finally:
        env.close()


def evaluate_agent(
    model_path=config.BEST_MODEL_PATH,
    num_episodes=config.EVALUATION_EPISODES,
    render_mode=None,
    seed=config.SEED + 10_000,
):
    # Load the best checkpoint and print a fixed evaluation summary.
    agent = create_agent(get_num_actions())
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL NOT FOUND: {model_path}")
    load_model(agent, model_path)

    result = evaluate_policy(
        agent,
        num_episodes=num_episodes,
        seed=seed,
        render_mode=render_mode,
        collect_diagnostics=True,
    )
    print("ASSAULT EVALUATION SUMMARY")
    print(f"AVERAGE REWARD: {result['mean_reward']:.2f}")
    print(f"REWARD STD DEV: {result['std_reward']:.2f}")
    print(f"BEST REWARD: {max(result['rewards']):.2f}")
    print(f"WORST REWARD: {min(result['rewards']):.2f}")
    print(
        "AVERAGE STEPS: "
        f"{sum(result['episode_lengths']) / len(result['episode_lengths']):.1f}"
    )
    return result


def watch_agent(model_path=config.BEST_MODEL_PATH, num_episodes=1):
    # Open the environment in human mode and run one evaluation episode.
    return evaluate_agent(model_path, num_episodes, render_mode="human")


if __name__ == "__main__":
    plot_history()
    evaluate_agent()
