# Assault Atari Reinforcement Learning Using Dueling Double DQN

This repository trains an agent to play `ALE/Assault-v5` with a Dueling Double DQN implementation in TensorFlow/Keras.

The current training code uses:

- Dueling DQN network head
- Double DQN target calculation
- Target network updates
- Epsilon-greedy exploration
- Uniform replay buffer
- Reward clipping for learning
- Fixed-seed evaluation checkpoints

## Project Structure

```text
agent.py          Agent creation, action selection, Double DQN training step
callbacks.py      CSV history logging and checkpoint saving
config.py         Training hyperparameters and output file paths
env_wrapper.py    Assault Atari environment preprocessing
evaluate.py       Plot history, evaluate checkpoint, watch agent
evaluation.py     Fixed-seed evaluation loop
network.py        Dueling DQN model
replay_buffer.py  Uniform replay buffer
train.py          Main training loop
```

## Requirements

Install the main Python packages:

```bash
pip install tensorflow gymnasium ale-py matplotlib numpy
```

Depending on your Gymnasium/ALE version, you may also need Atari ROM support installed in your environment.

## Train

Run:

```bash
python train.py
```

Training starts from scratch and writes:

- `training_history_assault_1.csv`
- `dqn_assault.weights_1.h5`
- `dqn_assault_best.weights_1.h5`

The main settings are in `config.py`.

## Evaluate

Run:

```bash
python evaluate.py
```

This plots the training curve and evaluates the best saved model.

To watch the trained agent, call `watch_agent()` from `evaluate.py`.

## Notes

The environment reward comes from Atari/ALE. The code logs raw game reward, but clips reward to `[-1, 1]` inside the DQN training target for stability.
