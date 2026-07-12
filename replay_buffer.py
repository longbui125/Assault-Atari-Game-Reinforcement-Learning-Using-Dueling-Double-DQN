import numpy as np


def create_replay_buffer(capacity, state_shape=(4, 84, 84)):
    # Store one frame per transition and rebuild 4-frame stacks when sampling.
    stack_size, height, width = state_shape
    return {
        "capacity": capacity,
        "stack_size": stack_size,
        "frames": np.empty((capacity, height, width), dtype=np.uint8),
        "next_frames": np.empty((capacity, height, width), dtype=np.uint8),
        "actions": np.empty((capacity, 1), dtype=np.int32),
        "rewards": np.empty((capacity, 1), dtype=np.float32),
        "dones": np.empty((capacity, 1), dtype=np.float32),
        "sequence_ids": np.full(capacity, -1, dtype=np.int64),
        "episode_ids": np.full(capacity, -1, dtype=np.int64),
        "pointer": 0,
        "size": 0,
        "total_added": 0,
    }


def add_experience(buffer, state, action, reward, next_state, done, episode_id):
    # Insert one transition at the current circular-buffer position.
    index = buffer["pointer"]
    buffer["frames"][index] = state[-1]
    buffer["next_frames"][index] = next_state[-1]
    buffer["actions"][index] = action
    buffer["rewards"][index] = reward
    buffer["dones"][index] = done
    buffer["sequence_ids"][index] = buffer["total_added"]
    buffer["episode_ids"][index] = episode_id

    buffer["pointer"] = (index + 1) % buffer["capacity"]
    buffer["size"] = min(buffer["size"] + 1, buffer["capacity"])
    buffer["total_added"] += 1


def is_valid_history_index(buffer, index, sequence_id, episode_id):
    # A frame is valid only if it belongs to the same episode and sequence.
    return (
        buffer["sequence_ids"][index] == sequence_id
        and buffer["episode_ids"][index] == episode_id
    )


def build_state_stack(buffer, index):
    # Reconstruct a 4-frame state stack without crossing episode boundaries.
    sequence_id = buffer["sequence_ids"][index]
    episode_id = buffer["episode_ids"][index]
    stack_size = buffer["stack_size"]
    earliest_frame = buffer["frames"][index]

    for offset in range(1, stack_size):
        candidate = (index - offset) % buffer["capacity"]
        if not is_valid_history_index(
            buffer, candidate, sequence_id - offset, episode_id
        ):
            break
        earliest_frame = buffer["frames"][candidate]

    frames = []
    for offset in range(stack_size - 1, -1, -1):
        candidate = (index - offset) % buffer["capacity"]
        if is_valid_history_index(buffer, candidate, sequence_id - offset, episode_id):
            frames.append(buffer["frames"][candidate])
        else:
            frames.append(earliest_frame)
    return np.stack(frames, axis=0)


def sample_batch(buffer, batch_size):
    # Sample transitions uniformly from the replay buffer.
    if buffer["size"] < batch_size:
        raise ValueError("Not enough transitions to sample a batch.")

    indices = np.random.randint(0, buffer["size"], size=batch_size, dtype=np.int32)
    stack_shape = (batch_size, buffer["stack_size"], *buffer["frames"].shape[1:])
    states = np.empty(stack_shape, dtype=np.uint8)
    next_states = np.empty_like(states)

    for batch_index, index in enumerate(indices):
        state = build_state_stack(buffer, index)
        states[batch_index] = state
        next_states[batch_index] = np.concatenate(
            (state[1:], buffer["next_frames"][index][None]), axis=0
        )

    return {
        "states": states,
        "actions": buffer["actions"][indices],
        "rewards": buffer["rewards"][indices],
        "next_states": next_states,
        "dones": buffer["dones"][indices],
    }
