import random

import numpy as np
import tensorflow as tf

import config
from agent import create_agent, select_action, train_step, update_epsilon, update_target_network
from callbacks import (
    create_checkpoint_state,
    log_episode,
    reset_history,
    save_evaluation_checkpoints,
)
from env_wrapper import make_env
from evaluation import evaluate_policy
from replay_buffer import add_experience, create_replay_buffer, sample_batch, update_priorities


def train():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.keras.utils.set_random_seed(config.SEED)

    env = make_env(render_mode=None)
    env.action_space.seed(config.SEED)
    agent = create_agent(int(env.action_space.n))
    replay = create_replay_buffer(
        capacity=config.REPLAY_CAPACITY,
        alpha=config.PRIORITY_ALPHA,
        max_priority=config.MAX_PRIORITY,
    )
    checkpoint_state = create_checkpoint_state()
    reset_history(config.HISTORY_PATH)

    environment_steps = 0
    training_steps = 0
    print("STARTING FRESH TRAINING RUN")

    try:
        for episode in range(1, config.EPISODES + 1):
            state, _ = env.reset(seed=config.SEED if episode == 1 else None)
            total_reward = 0.0
            losses = []
            done = False

            while not done:
                if replay["size"] < config.WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = select_action(agent, state, exploration=True)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                add_experience(
                    replay, state, action, reward, next_state, float(done), episode
                )
                state = next_state
                total_reward += float(reward)
                environment_steps += 1
                update_epsilon(agent, environment_steps)

                if replay["size"] >= config.WARMUP_STEPS:
                    beta_progress = min(
                        1.0, training_steps / config.PRIORITY_BETA_STEPS
                    )
                    beta = config.PRIORITY_BETA_START + beta_progress * (
                        1.0 - config.PRIORITY_BETA_START
                    )
                    batch = sample_batch(replay, config.BATCH_SIZE, beta)
                    loss, td_errors = train_step(agent, batch)
                    update_priorities(replay, batch["indices"], td_errors)
                    losses.append(loss)
                    training_steps += 1

                    if training_steps % config.TARGET_UPDATE_STEPS == 0:
                        update_target_network(agent)

            evaluation = None
            if (
                episode % config.EVALUATION_INTERVAL == 0
                or episode == config.EPISODES
            ):
                evaluation = evaluate_policy(
                    agent,
                    num_episodes=config.EVALUATION_EPISODES,
                    seed=config.SEED + 10_000,
                )
                is_best = save_evaluation_checkpoints(
                    agent,
                    evaluation,
                    checkpoint_state,
                    config.MODEL_PATH,
                    config.BEST_MODEL_PATH,
                )
                print(
                    f"EVAL: mean={evaluation['mean_reward']:.2f} | "
                    f"std={evaluation['std_reward']:.2f} | best={is_best}"
                )

            mean_loss = float(np.mean(losses)) if losses else 0.0
            log_episode(
                config.HISTORY_PATH,
                episode,
                total_reward,
                agent["epsilon"],
                environment_steps,
                training_steps,
                mean_loss,
                replay["size"],
                evaluation,
            )
            print(
                f"EPISODE: {episode} | reward={total_reward:.2f} | "
                f"epsilon={agent['epsilon']:.3f} | loss={mean_loss:.4f} | "
                f"replay={replay['size']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    train()
