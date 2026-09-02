"""Evaluate a trained go2_stand policy using task-level metrics.

Unlike the training return, these metrics are difficult for a policy to game and
make regressions between checkpoints visible.  Configuration is intentionally
kept out of the global Legged Gym argument parser:

    GO2_EVAL_ENVS=128 GO2_EVAL_EPISODES=3 \
      python legged_gym/scripts/evaluate_go2_stand.py \
      --task=go2_stand --headless --load_run=<run> --checkpoint=<checkpoint>

The JSON report is written to ``GO2_EVAL_OUTPUT`` (default:
``go2_stand_evaluation.json``).
"""

import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import isaacgym  # noqa: F401 - must be imported before torch in Isaac Gym
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers environments
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.isaacgym_utils import get_euler_xyz


class MetricAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.square_sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name, values):
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.sums[name] += values.sum().item()
        self.square_sums[name] += torch.square(values).sum().item()
        self.counts[name] += values.numel()

    def mean(self, name):
        count = self.counts[name]
        return self.sums[name] / count if count else None

    def rmse(self, name):
        count = self.counts[name]
        return (self.square_sums[name] / count) ** 0.5 if count else None


def intended_pitch_target(env):
    """The stand-up ramp intended by the config, including negative targets."""
    target = float(env.cfg.commands.pitch)
    duration = float(env.cfg.commands.standup_duration)
    ramp = env.episode_length_buf.float() * env.dt * target / duration
    return torch.clamp(ramp, min=min(0.0, target), max=max(0.0, target))


def add_command_bucket_metrics(metrics, env, valid, x_error, y_error, yaw_error):
    linear_dead = float(env.cfg.commands.linear_locomotion_threshold)
    yaw_dead = float(env.cfg.commands.yaw_locomotion_threshold)
    command_x = env.commands[:, 0]
    command_y = env.commands[:, 1]
    command_yaw = env.commands[:, 2]
    buckets = {
        "forward": command_x > linear_dead,
        "backward": command_x < -linear_dead,
        "strafe_left": (command_x.abs() <= linear_dead) & (command_y > linear_dead),
        "strafe_right": (command_x.abs() <= linear_dead) & (command_y < -linear_dead),
        "turn_left": (torch.norm(env.commands[:, :2], dim=1) <= linear_dead) & (command_yaw > yaw_dead),
        "turn_right": (torch.norm(env.commands[:, :2], dim=1) <= linear_dead) & (command_yaw < -yaw_dead),
        "idle": ~env._is_locomotion_command(),
    }
    for name, bucket_mask in buckets.items():
        selected = valid & bucket_mask
        metrics.add(f"bucket/{name}/abs_x_error", x_error[selected])
        metrics.add(f"bucket/{name}/abs_y_error", y_error[selected])
        metrics.add(f"bucket/{name}/abs_yaw_error", yaw_error[selected])


def evaluate(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.task != "go2_stand":
        raise ValueError("This evaluator is specific to --task=go2_stand")

    num_envs = int(os.environ.get("GO2_EVAL_ENVS", "128"))
    evaluation_episodes = int(os.environ.get("GO2_EVAL_EPISODES", "3"))
    output_path = os.environ.get("GO2_EVAL_OUTPUT", "go2_stand_evaluation.json")
    use_randomization = os.environ.get("GO2_EVAL_RANDOMIZATION", "0") == "1"

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_envs)
    env_cfg.terrain.curriculum = False
    if not use_randomization:
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_pd_gains = False
        env_cfg.domain_rand.randomize_motor_zero_offset = False
        env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    observations = env.get_observations()

    metrics = MetricAccumulator()
    episode_reward_sums = defaultdict(float)
    completed_episodes = 0
    failures = 0
    timeouts = 0
    previous_actions = torch.zeros_like(env.actions)
    total_steps = evaluation_episodes * int(env.max_episode_length)

    for _ in range(total_steps):
        previous_lengths = env.episode_length_buf.clone()
        with torch.no_grad():
            actions = policy(observations.detach())
            observations, _, rewards, dones, infos = env.step(actions.detach())

        dones = dones.bool()
        valid = ~dones
        metrics.add("reward_per_step", rewards[valid])

        heading_lin_vel = env._get_heading_frame_lin_vel()
        linear_error = env.commands[:, :2] - heading_lin_vel[:, :2]
        x_error = linear_error[:, 0].abs()
        y_error = linear_error[:, 1].abs()
        yaw_error = (env.commands[:, 2] - env.root_states[:, 12]).abs()
        metrics.add("abs_x_velocity_error", x_error[valid])
        metrics.add("abs_y_velocity_error", y_error[valid])
        metrics.add("linear_velocity_error", torch.norm(linear_error[valid], dim=1))
        metrics.add("abs_yaw_rate_error", yaw_error[valid])
        add_command_bucket_metrics(metrics, env, valid, x_error, y_error, yaw_error)

        euler = get_euler_xyz(env.root_states[:, 3:7])
        pitch_error = (intended_pitch_target(env) - euler[:, 1]).abs()
        roll_error = (float(env.cfg.commands.roll) - euler[:, 0]).abs()
        base_height = torch.mean(
            env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1
        )
        height_error = (base_height - float(env.cfg.rewards.base_height_target)).abs()
        metrics.add("abs_pitch_error", pitch_error[valid])
        metrics.add("abs_roll_error", roll_error[valid])
        metrics.add("abs_base_height_error", height_error[valid])
        if env.body_masses is not None:
            metrics.add(
                "com_over_support_reward", env._reward_com_over_support()[valid]
            )

        gait_stance = env._get_gait_phase()
        rear_contact = (
            env.contact_forces[:, env.desired_contact_indices, 2]
            > env.cfg.rewards.rear_contact_force
        )
        valid_feet = valid.unsqueeze(1).expand_as(rear_contact)
        metrics.add("gait_contact_match", (rear_contact == gait_stance)[valid_feet])
        metrics.add("stance_contact", rear_contact[valid_feet & gait_stance])
        metrics.add("swing_airborne", (~rear_contact)[valid_feet & ~gait_stance])

        undesired_contact = (
            torch.norm(
                env.contact_forces[:, env.undesired_contact_indices, :], dim=2
            ) > env.cfg.rewards.undesired_contact_force
        )
        metrics.add(
            "undesired_body_contact",
            undesired_contact[valid.unsqueeze(1).expand_as(undesired_contact)],
        )
        metrics.add("any_undesired_contact", undesired_contact.any(dim=1)[valid])

        rear_xy_speed = torch.norm(
            env.rigid_state[:, env.desired_contact_indices, 7:9], dim=2
        )
        metrics.add("contact_foot_slip_speed", rear_xy_speed[valid_feet & rear_contact])
        rear_height = (
            env.rigid_state[:, env.desired_contact_indices, 2]
            - env.env_origins[:, 2].unsqueeze(1)
            - env.cfg.rewards.foot_radius
        )
        clearance_error = (
            rear_height - float(env.cfg.rewards.target_foot_height)
        ).abs()
        metrics.add("swing_clearance_error", clearance_error[valid_feet & ~gait_stance])

        metrics.add(
            "mechanical_power_abs",
            torch.sum(torch.abs(env.torques * env.dof_vel), dim=1)[valid],
        )
        metrics.add(
            "action_delta_l2",
            torch.norm(actions - previous_actions, dim=1)[valid],
        )
        previous_actions.copy_(actions)
        previous_actions[dones] = 0.0

        stable = (
            (pitch_error < 0.25)
            & (roll_error < 0.20)
            & (height_error < 0.10)
            & ~undesired_contact.any(dim=1)
        )
        metrics.add("stable_upright", stable[valid])

        done_count = int(dones.sum().item())
        if done_count:
            completed_episodes += done_count
            current_timeouts = env.time_out_buf.bool() & dones
            timeout_count = int(current_timeouts.sum().item())
            timeouts += timeout_count
            failures += done_count - timeout_count
            episode_durations = (previous_lengths[dones].float() + 1.0) * env.dt
            metrics.add("episode_duration_s", episode_durations)
            for name, value in infos.get("episode", {}).items():
                if name.startswith("rew_"):
                    episode_reward_sums[name] += float(value.item()) * done_count

    report = {
        "configuration": {
            "task": args.task,
            "num_envs": env.num_envs,
            "evaluation_steps": total_steps,
            "policy_dt_s": env.dt,
            "domain_randomization": use_randomization,
            "history_length": env.cfg.env.history_length,
            "actor_observation_size": env.cfg.env.num_observations,
        },
        "episodes": {
            "completed": completed_episodes,
            "failures": failures,
            "timeouts": timeouts,
            "failure_rate": failures / completed_episodes if completed_episodes else None,
            "mean_duration_s": metrics.mean("episode_duration_s"),
        },
        "task_metrics": {
            "x_velocity_mae_m_s": metrics.mean("abs_x_velocity_error"),
            "x_velocity_rmse_m_s": metrics.rmse("abs_x_velocity_error"),
            "y_velocity_mae_m_s": metrics.mean("abs_y_velocity_error"),
            "yaw_rate_mae_rad_s": metrics.mean("abs_yaw_rate_error"),
            "pitch_mae_rad": metrics.mean("abs_pitch_error"),
            "roll_mae_rad": metrics.mean("abs_roll_error"),
            "base_height_mae_m": metrics.mean("abs_base_height_error"),
            "com_over_support_mean_reward": metrics.mean("com_over_support_reward"),
            "gait_contact_match_rate": metrics.mean("gait_contact_match"),
            "stance_contact_rate": metrics.mean("stance_contact"),
            "swing_airborne_rate": metrics.mean("swing_airborne"),
            "any_undesired_contact_rate": metrics.mean("any_undesired_contact"),
            "contact_foot_slip_mean_m_s": metrics.mean("contact_foot_slip_speed"),
            "swing_clearance_mae_m": metrics.mean("swing_clearance_error"),
            "stable_upright_rate": metrics.mean("stable_upright"),
            "mean_abs_mechanical_power_w": metrics.mean("mechanical_power_abs"),
            "mean_action_delta_l2": metrics.mean("action_delta_l2"),
            "mean_reward_per_step": metrics.mean("reward_per_step"),
        },
        "command_buckets": {},
        "mean_episode_reward_terms_per_second": {
            name: value / completed_episodes
            for name, value in sorted(episode_reward_sums.items())
        } if completed_episodes else {},
    }
    for bucket in (
        "forward", "backward", "strafe_left", "strafe_right",
        "turn_left", "turn_right", "idle",
    ):
        key = f"bucket/{bucket}/abs_x_error"
        report["command_buckets"][bucket] = {
            "samples": metrics.counts[key],
            "x_velocity_mae_m_s": metrics.mean(key),
            "y_velocity_mae_m_s": metrics.mean(f"bucket/{bucket}/abs_y_error"),
            "yaw_rate_mae_rad_s": metrics.mean(
                f"bucket/{bucket}/abs_yaw_error"
            ),
        }

    with open(output_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    evaluate(get_args())
