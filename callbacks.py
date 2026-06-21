import csv
import math

from agent import save_model


def reset_history(history_path):
    with open(history_path, "w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(
            [
                "Episode",
                "Train_Reward",
                "Epsilon",
                "Environment_Steps",
                "Training_Steps",
                "Mean_Loss",
                "Replay_Size",
                "Eval_Mean",
                "Eval_Std",
            ]
        )


def log_episode(
    history_path,
    episode,
    reward,
    epsilon,
    environment_steps,
    training_steps,
    mean_loss,
    replay_size,
    evaluation=None,
):
    with open(history_path, "a", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(
            [
                episode,
                reward,
                epsilon,
                environment_steps,
                training_steps,
                mean_loss,
                replay_size,
                "" if evaluation is None else evaluation["mean_reward"],
                "" if evaluation is None else evaluation["std_reward"],
            ]
        )


def create_checkpoint_state():
    return {"best_score": float("-inf")}


def save_evaluation_checkpoints(agent, evaluation, state, latest_path, best_path):
    save_model(agent, latest_path)
    conservative_score = evaluation["mean_reward"] - (
        evaluation["std_reward"] / math.sqrt(len(evaluation["rewards"]))
    )
    if conservative_score <= state["best_score"]:
        return False

    state["best_score"] = conservative_score
    save_model(agent, best_path)
    return True
