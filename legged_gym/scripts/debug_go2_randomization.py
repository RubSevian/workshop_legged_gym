"""Create a few go2_stand environments and validate domain randomization.

Run before a long baseline training:

    python legged_gym/scripts/debug_go2_randomization.py \
        --task=go2_stand --headless

Environment creation prints friction, masses, CoM offsets, PD multipliers,
motor-zero offsets and the torque produced for an identical synthetic state.
The process raises RuntimeError if controller randomization does not change the
probe torque across environments.
"""

import os
import sys

import isaacgym  # noqa: F401 - must be imported before torch in Isaac Gym

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from legged_gym.envs import *  # noqa: F401,F403 - registers environments
from legged_gym.utils import get_args, task_registry


def main(args):
    if args.task != "go2_stand":
        raise ValueError("Use this diagnostic with --task=go2_stand")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = int(os.environ.get("GO2_DEBUG_ENVS", "4"))
    env_cfg.domain_rand.debug_randomization = True
    env_cfg.domain_rand.debug_randomization_envs = env_cfg.env.num_envs
    env_cfg.domain_rand.push_robots = False
    task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[go2_stand] domain-randomization diagnostic completed successfully")


if __name__ == "__main__":
    main(get_args())
