# Go2 quadruped walk

This task trains all four Go2 legs to track planar velocity commands with a
diagonal trot. Exactly zero commands are a separate stand mode: all feet must
remain in contact and leg motion is penalized.

## Train from scratch

The gait-clock semantics and rewards changed, so do not resume an older
checkpoint for the first baseline run.

```bash
cd /home/ruben/go2_deploy/workshop_legged_gym
source /home/ruben/miniconda3/etc/profile.d/conda.sh
conda activate workshop_gym
python legged_gym/scripts/train.py --task=go2_walk --headless
```

Monitor the run:

```bash
tensorboard --logdir ./logs/go2_walk --port 6006
```

The most useful reward curves are `rew_tracking_lin_vel`,
`rew_tracking_ang_vel`, `rew_stand_still`, `rew_foot_placement`, and
`rew_rear_foot_extension`.

## Play and export

```bash
python legged_gym/scripts/play.py --task=go2_walk
```

To select a run explicitly:

```bash
python legged_gym/scripts/play.py --task=go2_walk \
  --load_run <run-directory-name> --checkpoint <iteration>
```

`play.py` exports the actor as a JIT policy under
`logs/go2_walk/exported/policies/`.

## Gait clock contract for deployment

The actor observation remains 47 values per frame and five frames of history,
so its tensor shape is unchanged. The last two values in each frame are still
`sin(2*pi*phase)` and `cos(2*pi*phase)`.

The controller that runs the exported policy must reproduce this clock:

1. Classify motion as
   `norm([vx, vy]) > 0.1 or abs(yaw_rate) > 0.1`.
2. While the command is zero, set `phase = 0`; the actor then receives
   `sin = 0`, `cos = 1`.
3. On the first moving control step keep `phase = 0`.
4. On subsequent moving steps update
   `phase = (phase + policy_dt / 0.5) % 1`, where `policy_dt = 0.02 s`.

If deployment keeps advancing the old wall-clock phase while stopped, the
policy will receive observations it did not see in stand mode and may move its
legs periodically.

## Main tuning controls

All behavior parameters are in `go2_config.py`:

- `commands.stand_probability`: fraction of five-second command intervals
  devoted to standing; default `0.25`.
- `rewards.scales.stand_still`: strength of zero-command stillness; default
  `-2.0`.
- `rewards.scales.foot_placement`: late-swing velocity-aware touchdown target;
  default `-0.5`.
- `rewards.scales.rear_foot_extension`: one-sided penalty for rear feet going
  excessively behind the base; default `-0.4`.
- `rewards.rear_foot_max_extension`: allowed rearward travel from the nominal
  rear foothold; default `0.14 m`.
- `domain_rand.push_standing`: whether random training pushes are allowed in
  stand mode; default `False` because pushes and command resampling otherwise
  occur on the same five-second boundary.

Change one scale at a time and compare both reward curves and videos. If rear
steps remain too far back, first reduce `rear_foot_max_extension` toward
`0.11-0.12 m`; only then increase `rear_foot_extension` magnitude.
