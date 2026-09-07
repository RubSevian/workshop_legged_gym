# Go2 TumblerNet task

This task is an isolated port of the HKU bipedal locomotion implementation from
`arclab-hku/bipedal_locomotion_for_quadrupedal_robots`, branch `dev`, commit
`ea502ca`. The environment/config snapshot is the repository's
`outputs/random_dog/Imi/test_reward` version; it has internally consistent
45/222 observation dimensions.

The task is registered as `go2_tumbler` and can be started independently:

```bash
conda activate workshop_gym
python legged_gym/scripts/train.py --task=go2_tumbler --headless
```

For a short launch check:

```bash
python legged_gym/scripts/train.py \
  --task=go2_tumbler --headless --num_envs=4 --max_iterations=1
```

## Policy playback

Use the dedicated player, rather than the generic `play.py`: this task returns
an observation dictionary and a four-value `step()` result.  The player reads
the encoder input width from the checkpoint and sets both the environment
history buffer and `Encoder.HistoryLen` before constructing the policy.  Thus
checkpoints trained with a different number of history frames do not fail with
a matrix-shape error.

```bash
conda activate workshop_gym
python legged_gym/scripts/play_go2_tumbler.py \
  --checkpoint_path logs/random_dog/<run>/stage1_nn/last.pt \
  --num_envs 1
```

For a bounded smoke test add `--headless --max_steps 100`.  The script prints
the detected history length; for the current checkpoint it is 4
(`4 * 45 = 180` encoder inputs).

The copied algorithm keeps the original settings:

- action dimension: 12;
- actor observation: 45;
- privileged observation: 222;
- history: 4 frames (180 estimator inputs);
- estimator: 180 -> 256 -> 128 -> 6;
- actor input: 45 + 6 = 51;
- critic input: 45 + 222 = 267;
- simulation timestep: 0.005 s;
- control decimation: 4 (policy timestep 0.02 s);
- action scale: 0.25;
- PD gains: Kp 30.0, Kd 0.8;
- PPO, rewards, randomization, commands and terrain settings are copied from
  the HKU snapshot.

Go1-to-Go2 substitutions are limited to the asset directory/name and runtime
checks for Go2 joint/body ordering. The HKU repository does not include the
referenced imitation pickle file. Its loading is skipped when all imitation
reference rewards are disabled, as they are in this configuration; this does
not change an active reward or training target.
