"""Minimal headless Isaac-Gym-to-MuJoCo smoke test for a Go2 Tumbler policy.

This intentionally tests only the inference path used by the actor:
45 proprioceptive values plus their history.  It uses real MuJoCo contacts and
the training PD gains, but does not attempt to reproduce privileged terrain or
domain-randomization signals.
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

# Running a script by its path adds ``legged_gym/scripts`` to sys.path, not
# the repository root.  Keep this standalone smoke test usable before an
# editable install of the project.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tumbler_rl.modules import ActorCritic


DEFAULT_DOF_POS = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}


def rotate_inverse(quat_wxyz, vector):
    """Rotate a world vector into the base frame, matching Isaac Gym."""
    quat_vec = quat_wxyz[1:]
    quat_w = quat_wxyz[0]
    return (vector * (2.0 * quat_w * quat_w - 1.0)
            - np.cross(quat_vec, vector) * quat_w * 2.0
            + quat_vec * np.dot(quat_vec, vector) * 2.0)


def history_length_from_checkpoint(checkpoint, num_obs=45):
    weight = checkpoint["actor_state_dict"]["dm_encoder.encoder.0.weight"]
    if weight.shape[1] % num_obs:
        raise ValueError(f"Unsupported encoder input width {weight.shape[1]}.")
    return weight.shape[1] // num_obs


def build_actor(checkpoint, history_length):
    actor = ActorCritic(
        45, 12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        priv_info=True,
        priv_info_dim=222,
        HistoryLen=history_length,
        encoder_mlp_units=[256, 128, 6],
        decoder_mlp_units=[256, 128, 6],
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--scene",
        default="/home/ruben/go2_deploy/workhop_rl/src/unitree_mujoco/unitree_robots/go2/scene.xml",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    history_length = history_length_from_checkpoint(checkpoint)
    actor = build_actor(checkpoint, history_length)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0 or model.nu != 12:
        raise RuntimeError("Expected the 12-actuator Go2 MuJoCo model.")

    # Set the same default pose that was used for training.
    for joint_name, angle in DEFAULT_DOF_POS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    data.qpos[2] = 0.34
    mujoco.mj_forward(model, data)

    action_by_joint = np.zeros(12, dtype=np.float32)
    history = deque(maxlen=history_length)
    action_abs_max = 0.0
    for step in range(args.steps):
        quat = data.xquat[base_id].copy()  # MuJoCo uses w, x, y, z.
        angular_velocity = rotate_inverse(quat, data.qvel[3:6])
        projected_gravity = rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        dof_pos, dof_vel = [], []
        for joint_name in DEFAULT_DOF_POS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof_pos.append(data.qpos[model.jnt_qposadr[joint_id]])
            dof_vel.append(data.qvel[model.jnt_dofadr[joint_id]])
        dof_pos = np.asarray(dof_pos, dtype=np.float32)
        dof_vel = np.asarray(dof_vel, dtype=np.float32)
        default_pos = np.asarray(list(DEFAULT_DOF_POS.values()), dtype=np.float32)
        obs = np.concatenate((
            angular_velocity * 0.25,
            projected_gravity,
            np.zeros(3, dtype=np.float32),  # command: standing still
            dof_pos - default_pos,
            dof_vel * 0.05,
            action_by_joint,
        )).astype(np.float32)
        if obs.shape != (45,):
            raise RuntimeError(f"Expected 45 actor observations, got {obs.shape}.")
        if not history:
            history.extend([obs.copy()] * history_length)
        else:
            history.append(obs.copy())

        # The critic needs privileged data, but the actor does not.  A zero
        # placeholder lets the shared implementation run without affecting its
        # action output.
        obs_dict = {
            "obs": torch.from_numpy(obs).unsqueeze(0),
            "proprio_hist": torch.from_numpy(np.concatenate(history)).unsqueeze(0),
            "privileged_info": torch.zeros(1, 222),
        }
        with torch.inference_mode():
            action_by_joint = actor.act_inference(obs_dict).squeeze(0).numpy()
        if not np.isfinite(action_by_joint).all():
            raise RuntimeError("Policy returned non-finite actions.")
        action_abs_max = max(action_abs_max, float(np.abs(action_by_joint).max()))

        # MuJoCo actuator order differs from the policy order; match by joint
        # name, then apply the training-time position PD controller (20 / 1).
        target_by_joint = {
            name: DEFAULT_DOF_POS[name] + 0.25 * action_by_joint[index]
            for index, name in enumerate(DEFAULT_DOF_POS)
        }
        for actuator_id in range(model.nu):
            joint_id = model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos = data.qpos[model.jnt_qposadr[joint_id]]
            qvel = data.qvel[model.jnt_dofadr[joint_id]]
            torque = 20.0 * (target_by_joint[joint_name] - qpos) - 1.0 * qvel
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(torque, low, high)
        for _ in range(4):  # 0.005 s MuJoCo step -> 0.02 s policy step
            mujoco.mj_step(model, data)

    print(
        "MuJoCo smoke test passed: "
        f"steps={args.steps}, history={history_length}, "
        f"base_z={data.qpos[2]:.3f}, max_abs_action={action_abs_max:.3f}"
    )


if __name__ == "__main__":
    main()
