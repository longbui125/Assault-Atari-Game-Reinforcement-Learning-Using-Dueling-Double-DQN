from collections import Counter, deque

import numpy as np

from agent import select_action
from env_wrapper import ASSAULT_ACTIONS, make_env


def get_action_meanings(env):
    # Use the environment's action names when they are available.
    meanings = env.unwrapped.get_action_meanings()
    if len(meanings) == env.action_space.n:
        return meanings
    return [ASSAULT_ACTIONS.get(action, str(action)) for action in range(env.action_space.n)]


def evaluate_policy(
    agent,
    num_episodes,
    seed,
    render_mode=None,
    collect_diagnostics=False,
):
    # Run full evaluation episodes without exploration.
    env = make_env(render_mode=render_mode)
    action_meanings = get_action_meanings(env)
    rewards = []
    episode_lengths = []
    action_counts = Counter()
    positive_sequences = Counter()

    try:
        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            steps = 0
            recent_actions = deque(maxlen=3)

            while not done:
                # Track action usage only when diagnostics are requested.
                action = select_action(agent, state, exploration=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if collect_diagnostics:
                    action_name = action_meanings[action]
                    action_counts[action_name] += 1
                    recent_actions.append(action_name)
                    if reward > 0 and len(recent_actions) == 3:
                        positive_sequences[" -> ".join(recent_actions)] += 1

                state = next_state
                total_reward += float(reward)
                steps += 1

            rewards.append(total_reward)
            episode_lengths.append(steps)
    finally:
        env.close()

    return {
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "action_counts": action_counts,
        "positive_sequences": positive_sequences,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }
