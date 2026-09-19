# Training

`oz train` trains the shared MAPPO actor and centralized critic on
`CollisionAvoidanceEnv`.

```sh
.venv/bin/oz train --config configs/mappo_smoke.json --output runs/mappo_smoke
.venv/bin/oz train --resume runs/mappo_smoke
```

Without `--output`, a new directory `runs/<timestamp>_mappo_seed<N>` is created.
Existing output paths are refused. `--resume` reads `config.json` from the run
directory and continues from its latest checkpoint until `training.total_updates`.

## Scenario source

`environment.scenario` selects where episodes come from:

- `development`: the fixed four-satellite, one-debris fixture, reset with each seed.
- `generated`: a new scenario per episode from real orbits and real close calls; see
  [TRAINING_SCENARIOS.md](TRAINING_SCENARIOS.md).

The number of spacecraft must equal `environment.num_agents`.

## Curriculum

Train the stages in order; each copies the previous stage's actor through
`training.initial_actor_checkpoint`:

```sh
.venv/bin/oz train --config configs/mappo_stage1.json --output runs/stage1
.venv/bin/oz train --config configs/mappo_stage2.json --output runs/stage2
.venv/bin/oz train --config configs/mappo_stage3.json --output runs/stage3
```

## Training loop

Each update collects complete episodes until at least `training.rollout_steps`
environment steps are stored, then runs one PPO update. Episode `i` resets with
seed `seed + i`, so a run is fully determined by its configuration.

- A **collision** ends an episode as a true terminal state.
- Reaching the **horizon** is a time limit, not a terminal state: the last stored
  reward becomes `r + gamma * V(final state)` before GAE treats the step as done.

Rollouts always end on an episode boundary, so a resumed run reproduces an
uninterrupted run exactly.

## Run directory

| Path | Contents |
| --- | --- |
| `config.json` | Exact experiment configuration |
| `environment_info.json` | Python, PyTorch, platform, and device |
| `metrics.csv` | One row per update |
| `tensorboard/` | The same metrics as TensorBoard scalars |
| `checkpoints/latest.pt` | MAPPO weights, optimizer state, and counters |
| `checkpoints/rng_state.pt` | NumPy and PyTorch random-number states |
| `training_state.json` | Completed updates, environment steps, next episode index |

Checkpoints are written every `training.checkpoint_interval` updates and after the
final update. On resume, metric rows newer than the checkpoint are discarded.

## Metrics

| Column | Meaning |
| --- | --- |
| `update`, `environment_steps`, `episodes` | Progress counters |
| `mean_episode_return` | Mean per-agent undiscounted return |
| `mean_episode_length` | Decision steps per episode |
| `collision_rate` | Fraction of episodes ended by a collision |
| `mean_unsafe_agent_steps` | Agent-steps involved in an unsafe conjunction, per episode |
| `mean_delta_v_per_agent_mps` | Delta-v used per agent per episode |
| `minimum_separation_meters` | Closest approach seen during the update |
| `actor_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_fraction` | PPO diagnostics |
| `update_seconds` | Wall-clock time of the update |

## Configurations

- `configs/mappo_smoke.json`: two tiny updates on the development fixture, for
  checking the pipeline.
- `configs/mappo_generated_smoke.json`: two tiny updates on generated scenarios.
- `configs/mappo_stage{1,2,3}.json`: the training curriculum.
- `configs/mappo_toy.json`: architecture defaults. It is not a thesis experiment
  until the training scenarios and maneuver sizing are fixed.
