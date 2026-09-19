# Training scenarios

With `environment.scenario = "generated"`, every training episode is built on the
fly from its seed (`seed + episode index`): real satellites on real orbits, facing
close calls copied from the 496 real conjunctions found by the calibration. The same
seed always rebuilds the same episode, so runs are reproducible and resumable.

## Recipe

1. **Date.** A random time within `epoch_window_seconds` (72 h) after the catalog's
   latest TLE epoch.
2. **Situations.** Draw situations from `situation_weights` until every agent slot is
   used, never exceeding the extra-object slots:

   | Situation | Agents | Extra objects | Meaning |
   | --- | ---: | ---: | --- |
   | `debris` | 1 | 1 | A non-maneuverable threat |
   | `satellite_pair` | 2 | 0 | Two agents on a collision course; both can maneuver |
   | `double_threat` | 1 | 2 | Two threats, the second 1–4 min after the first |
   | `harmless` | 1 | 1 | An object passing 2–10 km away; the right answer is to coast |
   | `quiet` | 1 | 0 | Nothing nearby |

3. **Agents.** Real agent-candidate satellites from `data/full`, propagated with SGP4
   to the date and converted from TEME to EME2000, the frame the environment uses.
4. **Threats, built backwards.** For each threat, fly the agent to a meeting time
   (8–16.5 min, matching the measured warning times), place the threat there using a
   real close call's miss vector and relative velocity (both in the agent's RSW
   frame), then propagate the threat backwards to the episode start. Propagation uses
   the environment's own J2 gravity model, so if nobody maneuvers the close call
   happens as planned (within a few metres).
5. **Background.** Unused extra slots are filled with real debris from the catalog.
6. **Start check.** If any unplanned pair starts closer than 10 km, the scenario is
   redrawn.

Each environment carries `env.scenario` with a readable manifest: date, agents,
situations, and planned encounters (meeting time, miss, and source close call).

## Train and test split

Satellites, background objects, and close-call shapes are split by a fixed hash of
their NORAD ID or event ID: 80% train, 20% test (`held_out_fraction`). Training uses
the train split; `oz evaluate` automatically draws from the test split.

| Split | Agent satellites | Background objects | Close-call shapes |
| --- | ---: | ---: | ---: |
| Train | 12,569 | 7,197 | 397 |
| Test | 3,169 | 1,787 | 99 |

## Curriculum

| Config | Agents | Objects | Situations | Starts from |
| --- | ---: | ---: | --- | --- |
| `configs/mappo_stage1.json` | 16 | 32 | Single threats, harmless, quiet | Random actor |
| `configs/mappo_stage2.json` | 64 | 128 | Balanced mix | Stage 1 actor |
| `configs/mappo_stage3.json` | 150 | 300 | Balanced mix | Stage 2 actor |

The critic's input size changes with the population, so each stage starts a new
critic and copies only the actor (`training.initial_actor_checkpoint`).

Measured on the M1 MacBook Air with a random policy:

| Stage | Episode (25 decisions) | Samples per second |
| ---: | ---: | ---: |
| 1 | 2.2 s | 184 |
| 2 | 7.7 s | 209 |
| 3 | 27.5 s | 137 |

## Inputs

- `data/full/catalog.tle` and `data/full/objects.csv` (agent and background pools)
- `configs/maneuver_sizing.json`, which points to the calibration catalog and the
  reference conjunctions in `runs/k_dt_calibration_20k_72h/`

Both are local and gitignored; tests that need them are skipped when absent.

## Limitations

- Threats are synthetic objects on real close-call geometry, not catalog objects.
  Final evaluation on unmodified catalog windows uses `oz scale`.
- Satellite-pair partners fly constructed orbits rather than their own real ones.
- Drag is off.
