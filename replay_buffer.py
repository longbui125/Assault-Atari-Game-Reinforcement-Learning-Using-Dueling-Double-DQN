import numpy as np


def create_sum_tree(capacity):
    leaf_count = 1
    while leaf_count < capacity:
        leaf_count *= 2
    return {
        "leaf_count": leaf_count,
        "values": np.zeros(2 * leaf_count, dtype=np.float64),
    }


def sum_tree_total(tree):
    return float(tree["values"][1])


def update_sum_tree(tree, index, priority):
    node = index + tree["leaf_count"]
    delta = priority - tree["values"][node]
    while node >= 1:
        tree["values"][node] += delta
        node //= 2


def find_sum_tree_index(tree, value):
    total = sum_tree_total(tree)
    value = min(max(value, np.finfo(np.float64).tiny), np.nextafter(total, 0.0))
    node = 1
    while node < tree["leaf_count"]:
        left = node * 2
        if value < tree["values"][left]:
            node = left
        else:
            value -= tree["values"][left]
            node = left + 1
    return node - tree["leaf_count"], float(tree["values"][node])


def create_replay_buffer(capacity, state_shape=(4, 84, 84), alpha=0.6, max_priority=100.0):
    stack_size, height, width = state_shape
    return {
        "capacity": capacity,
        "stack_size": stack_size,
        "alpha": alpha,
        "max_allowed_priority": max_priority,
        "frames": np.empty((capacity, height, width), dtype=np.uint8),
        "next_frames": np.empty((capacity, height, width), dtype=np.uint8),
        "actions": np.empty((capacity, 1), dtype=np.int32),
        "rewards": np.empty((capacity, 1), dtype=np.float32),
        "dones": np.empty((capacity, 1), dtype=np.float32),
        "sequence_ids": np.full(capacity, -1, dtype=np.int64),
        "episode_ids": np.full(capacity, -1, dtype=np.int64),
        "tree": create_sum_tree(capacity),
        "pointer": 0,
        "size": 0,
        "total_added": 0,
        "max_priority": 1.0,
    }


def add_experience(buffer, state, action, reward, next_state, done, episode_id):
    index = buffer["pointer"]
    buffer["frames"][index] = state[-1]
    buffer["next_frames"][index] = next_state[-1]
    buffer["actions"][index] = action
    buffer["rewards"][index] = reward
    buffer["dones"][index] = done
    buffer["sequence_ids"][index] = buffer["total_added"]
    buffer["episode_ids"][index] = episode_id
    update_sum_tree(
        buffer["tree"], index, buffer["max_priority"] ** buffer["alpha"]
    )

    buffer["pointer"] = (index + 1) % buffer["capacity"]
    buffer["size"] = min(buffer["size"] + 1, buffer["capacity"])
    buffer["total_added"] += 1


def is_valid_history_index(buffer, index, sequence_id, episode_id):
    return (
        buffer["sequence_ids"][index] == sequence_id
        and buffer["episode_ids"][index] == episode_id
    )


def build_state_stack(buffer, index):
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


def sample_batch(buffer, batch_size, beta):
    if buffer["size"] < batch_size:
        raise ValueError("Not enough transitions to sample a batch.")

    total_priority = sum_tree_total(buffer["tree"])
    if not np.isfinite(total_priority) or total_priority <= 0:
        raise RuntimeError("Replay priorities are invalid.")

    indices = np.empty(batch_size, dtype=np.int32)
    probabilities = np.empty(batch_size, dtype=np.float32)
    segment = total_priority / batch_size

    for sample_number in range(batch_size):
        value = np.random.uniform(segment * sample_number, segment * (sample_number + 1))
        index, priority = find_sum_tree_index(buffer["tree"], value)
        if index >= buffer["capacity"] or priority <= 0:
            raise RuntimeError("Replay sampled an invalid priority leaf.")
        indices[sample_number] = index
        probabilities[sample_number] = priority / total_priority

    stack_shape = (batch_size, buffer["stack_size"], *buffer["frames"].shape[1:])
    states = np.empty(stack_shape, dtype=np.uint8)
    next_states = np.empty_like(states)
    for batch_index, index in enumerate(indices):
        state = build_state_stack(buffer, index)
        states[batch_index] = state
        next_states[batch_index] = np.concatenate(
            (state[1:], buffer["next_frames"][index][None]), axis=0
        )

    probabilities = np.maximum(probabilities, np.finfo(np.float32).tiny)
    weights = np.power(buffer["size"] * probabilities, -beta)
    weights /= max(float(weights.max()), np.finfo(np.float32).tiny)
    return {
        "states": states,
        "actions": buffer["actions"][indices],
        "rewards": buffer["rewards"][indices],
        "next_states": next_states,
        "dones": buffer["dones"][indices],
        "indices": indices,
        "importance_weights": weights.astype(np.float32).reshape(-1, 1),
    }


def update_priorities(buffer, indices, td_errors):
    for index, td_error in zip(indices, td_errors):
        priority = float(
            np.clip(
                np.nan_to_num(
                    abs(float(td_error)),
                    nan=buffer["max_allowed_priority"],
                    posinf=buffer["max_allowed_priority"],
                    neginf=buffer["max_allowed_priority"],
                )
                + 1e-4,
                1e-4,
                buffer["max_allowed_priority"],
            )
        )
        buffer["max_priority"] = max(buffer["max_priority"], priority)
        update_sum_tree(buffer["tree"], int(index), priority ** buffer["alpha"])
