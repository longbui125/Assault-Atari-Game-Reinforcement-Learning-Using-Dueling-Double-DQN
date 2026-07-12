import random

import numpy as np
import tensorflow as tf

import config
from network import build_dqn


def create_agent(num_actions):
    # Build the online and target networks and keep them in a plain dict.
    main_net = build_dqn(num_actions=num_actions)
    target_net = build_dqn(num_actions=num_actions)
    target_net.set_weights(main_net.get_weights())

    return {
        "num_actions": num_actions,
        "main_net": main_net,
        "target_net": target_net,
        "optimizer": tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            epsilon=1e-4,
            clipnorm=1.0,
        ),
        "loss_fn": tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE
        ),
        "epsilon": config.EPSILON_START,
    }


def update_epsilon(agent, environment_steps):
    # Linearly decay epsilon over the configured number of environment steps.
    progress = min(1.0, environment_steps / config.EPSILON_DECAY_STEPS)
    agent["epsilon"] = config.EPSILON_START + progress * (
        config.EPSILON_FINAL - config.EPSILON_START
    )


def select_action(agent, state, exploration=True):
    # Use random actions during exploration and greedy actions otherwise.
    if exploration and random.random() < agent["epsilon"]:
        return random.randrange(agent["num_actions"])

    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    q_values = agent["main_net"](state_tensor, training=False)
    return int(tf.argmax(q_values[0]))


def train_step(agent, batch):
    # Prepare tensors for the Double DQN update.
    states = tf.convert_to_tensor(batch["states"])
    actions = tf.cast(tf.squeeze(tf.convert_to_tensor(batch["actions"]), axis=1), tf.int32)
    rewards = tf.clip_by_value(tf.cast(batch["rewards"], tf.float32), -1.0, 1.0)
    next_states = tf.convert_to_tensor(batch["next_states"])
    dones = tf.cast(batch["dones"], tf.float32)

    # Select next actions with the online network and evaluate them with the target network.
    next_online_q = agent["main_net"](next_states, training=False)
    next_actions = tf.argmax(next_online_q, axis=1, output_type=tf.int32)
    next_target_q = agent["target_net"](next_states, training=False)
    next_q = tf.gather(next_target_q, next_actions, axis=1, batch_dims=1)
    target_q = tf.stop_gradient(
        tf.clip_by_value(
            rewards + config.GAMMA * tf.expand_dims(next_q, axis=1) * (1.0 - dones),
            -config.MAX_ABS_Q_VALUE,
            config.MAX_ABS_Q_VALUE,
        )
    )

    # Compute the Huber loss and update the online network.
    with tf.GradientTape() as tape:
        q_predictions = agent["main_net"](states, training=True)
        current_q = tf.expand_dims(
            tf.gather(q_predictions, actions, axis=1, batch_dims=1), axis=1
        )
        per_sample_loss = agent["loss_fn"](target_q, current_q)
        loss = tf.reduce_mean(per_sample_loss)

    if not bool(tf.math.is_finite(loss)):
        raise FloatingPointError("DQN loss became non-finite.")

    gradients = tape.gradient(loss, agent["main_net"].trainable_variables)
    agent["optimizer"].apply_gradients(
        zip(gradients, agent["main_net"].trainable_variables)
    )
    return float(loss)


def update_target_network(agent):
    # Copy weights from the online network into the target network.
    agent["target_net"].set_weights(agent["main_net"].get_weights())


def save_model(agent, filepath):
    # Save only the online network weights.
    agent["main_net"].save_weights(filepath)


def load_model(agent, filepath):
    # Load weights into the online network and sync the target network.
    agent["main_net"].load_weights(filepath)
    agent["target_net"].set_weights(agent["main_net"].get_weights())
