# Go2 walk baseline

PPO baseline for four-legged velocity locomotion. The reward set is adapted
from `unitree_go2_sim2real/policies/walker_policy`, while the actor input and
command sampler remain compatible with this Legged-Gym project.

## Train from scratch

Observation and reward dimensions changed, so start from a new checkpoint.

```bash
cd /home/ruben/go2_deploy/workshop_legged_gym
source /home/ruben/miniconda3/etc/profile.d/conda.sh
conda activate workshop_gym
python legged_gym/scripts/train.py --task=go2_walk --headless
```

The paper uses 8192 environments. On a smaller GPU override this without
changing the config:

```bash
python legged_gym/scripts/train.py --task=go2_walk --headless --num_envs=4096
```

Monitor the run:

```bash
tensorboard --logdir ./logs/go2_walk --port 6006
```

Watch `rew_tracking_lin_vel`, `rew_tracking_ang_vel`, `rew_feet_height`,
`rew_feet_air_time`, and `rew_foot_slip` first.

## Play and export

```bash
python legged_gym/scripts/play.py --task=go2_walk
```

To select a run explicitly:

```bash
python legged_gym/scripts/play.py --task=go2_walk \
  --load_run <run-directory-name> --checkpoint <iteration>
```

To test the stand rule independently of terrain, pushes, reset velocity, and
domain randomization, run one 20-second zero-command episode:

```bash
python legged_gym/scripts/play.py --task=go2_walk \
  --load_run <run-directory-name> --checkpoint <iteration> \
  --zero_command --max_steps 1000
```

At the end it prints horizontal displacement and final body-frame velocity.

`play.py` exports the actor as a JIT policy under
`logs/go2_walk/exported/policies/`.

## Actor input

The actor has no gait clock and no terrain heights. One frame has 45 values:
base angular velocity (3), projected gravity (3), command (3), joint position
offsets (12), joint velocities (12), and previous action (12). Five frames are
stacked, therefore the deployed policy input is 225 values. The policy runs at
50 Hz (`sim.dt=0.005`, `decimation=4`).

## Main tuning controls

All behavior parameters are in `go2_config.py`:

- `target_foot_height = 0.10 m` and `foot_radius = 0.022 m`: the source
  policy's desired clearance and the Go2 URDF foot-sphere radius.  The
  `feet_height` term checks the peak of a swing at touchdown; `feet_clearance`
  supplies a continuous shaping signal while the foot moves.
- `min_feet_air_time = 0.10 s`: source touchdown threshold. It is deliberately
  much smaller than the stock Legged-Gym `0.5 s`, so it rewards normal Go2
  steps instead of forcing unnaturally long swings.
- `feet_air_time = 0.1`, `feet_height = -0.2`, `feet_clearance = -2.0`, and
  `foot_slip = -0.1` are the source reward coefficients. The remaining source
  coefficients are copied in `rewards.scales` too.

Commands are sampled by the standard parent Legged-Gym sampler from
`vx = [-1.5, 1.5] m/s`, `vy = [-0.75, 0.75] m/s`, and
`yaw = [-0.5, 0.5] rad/s`; small planar commands (`norm(vx, vy) <= 0.2`) are
set to zero. `commands.zero_command_probability = 0.15` additionally reserves
15% of complete intervals for the exact command `[0, 0, 0]`; the remaining
85% retain that standard distribution. There is no phase clock, sin/cos
observation, workspace, or foot-placement reward.

During these zero-command intervals, `stand_still` penalizes the L1 distance
of all joints from `init_state.default_joint_angles`. It uses a bounded
`tanh(mean_abs_error / stand_joint_position_tolerance)` cost, so reward
clipping does not erase the difference between bad standing poses. Its maximum
strength is `rewards.stand_joint_position_weight * rewards.scales.stand_still`
(default `2.0 * -1.0`).
