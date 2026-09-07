"""Play a Go2 Tumbler checkpoint with its own encoder-history length.

Example:
    python legged_gym/scripts/play_go2_tumbler.py --checkpoint_path \\
      logs/random_dog/<run>/stage1_nn/last.pt --num_envs 1
"""

import os

import isaacgym  # noqa: F401: must be imported before torch/legged_gym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2_tumbler.checkpoint import infer_history_length
from legged_gym.utils import get_args, task_registry


def play(args):
    if not args.checkpoint_path:
        raise ValueError("Pass a Tumbler checkpoint with --checkpoint_path <path-to-last.pt>.")
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    env_cfg, train_cfg = task_registry.get_cfgs(name="go2_tumbler")
    history_length = infer_history_length(checkpoint, env_cfg.env.num_observations)
    print(f"[go2_tumbler] checkpoint history length: {history_length}")

    # The environment and actor must be built with exactly the width encoded in
    # the checkpoint.  Changing only Encoder.HistoryLen is not sufficient:
    # the environment owns the rolling proprioception buffer.
    env_cfg.env.num_histroy_obs = history_length
    train_cfg.Encoder.HistoryLen = history_length
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = False

    env, _ = task_registry.make_env(name="go2_tumbler", args=args, env_cfg=env_cfg)
    play_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "go2_tumbler_play")
    runner, _ = task_registry.make_alg_runner(
        env=env, name="go2_tumbler", args=args, train_cfg=train_cfg, log_root=play_log_root
    )
    runner.load(checkpoint_path, load_optimizer=False)
    policy, _ = runner.get_inference_policy(device=env.device)
    obs_dict = env.get_observations()

    steps = args.max_steps if args.max_steps is not None else int(env.max_episode_length)
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play(get_args())
