import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

ENV_ID = "ALE/Assault-v5"

ASSAULT_ACTIONS = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "RIGHTFIRE",
    6: "LEFTFIRE",
}


def make_env(env_id=ENV_ID, render_mode=None):
    # Build the Assault environment with Atari preprocessing and frame stacking.
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=20,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )

    env = FrameStackObservation(env, stack_size=4)

    return env


if __name__ == "__main__":
    # Print the wrapped observation and action space for a quick sanity check.
    test_env = make_env()
    obs, info = test_env.reset()

    print("Size of state space:", test_env.observation_space.shape)
    print("Data type:", obs.dtype)
    print("Action space:", test_env.action_space.n)

    test_env.close()
