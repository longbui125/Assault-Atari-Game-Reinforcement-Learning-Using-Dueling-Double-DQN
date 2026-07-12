import tensorflow as tf
from tensorflow.keras import Model, layers


def build_dqn(input_shape=(4, 84, 84), num_actions=7):
    # Normalize stacked frames before they go into the convolution layers.
    frames = layers.Input(shape=input_shape, dtype=tf.uint8, name="frames")
    x = layers.Lambda(lambda value: tf.cast(value, tf.float32) / 255.0)(frames)

    # Extract visual features with the standard Atari DQN backbone.
    x = layers.Conv2D(
        32, kernel_size=8, strides=4, activation="relu", data_format="channels_first"
    )(x)

    x = layers.Conv2D(
        64, kernel_size=4, strides=2, activation="relu", data_format="channels_first"
    )(x)

    x = layers.Conv2D(
        64, kernel_size=3, strides=1, activation="relu", data_format="channels_first"
    )(x)

    features = layers.Flatten()(x)

    # Split into value and advantage streams for a dueling DQN head.
    value = layers.Dense(512, activation="relu", name="value_hidden")(features)
    value = layers.Dense(1, name="value")(value)

    advantage = layers.Dense(512, activation="relu", name="advantage_hidden")(features)
    advantage = layers.Dense(num_actions, name="advantage")(advantage)
    q_values = layers.Lambda(lambda values: values[0] + values[1] - tf.reduce_mean(values[1], axis=1, keepdims=True),
        name="q_values",
    )([value, advantage])

    return Model(frames, q_values, name="dueling_dqn")
