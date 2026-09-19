# Scalability evaluation

`oz scale` runs a frozen decentralized policy at catalog scale and measures
safety and compute cost as the population grows. The critic is never used.

```sh
.venv/bin/oz scale --config configs/scalability.json \
  --policy rule --policy mappo=runs/<run>/checkpoints/latest.pt \
  --output runs/scalability
```

`--sweep catalog|agents|both` (default `both`) selects the sweeps. The no-op
policy always runs first as the reference. `configs/scalability_smoke.json` is a
small, fast version.

## Sweeps

| Sweep | Objects | Agents |
| --- | --- | --- |
| `catalog` | `catalog_sizes` (for example 1k–20k): every agent candidate plus a seeded sample of other objects | The metadata agent candidates (349 in the frozen catalog) |
| `agents` | The full retained catalog | `hypothetical_agent_counts` (for example 1k–10k) seeded LEO payloads, treated as maneuverable |

Populations are nested: each smaller population is a subset of the next. The
`agents` sweep is hypothetical: it asks what happens if every satellite ran the
policy, regardless of its real maneuverability.

## Simulation model

- **Reference motion.** Every object follows its SGP4 trajectory (TEME frame)
  from the latest TLE epoch, as in the calibration. With the no-op policy the
  simulation therefore reproduces the calibration's reference motion exactly.
- **Maneuvers.** Each executed action is an impulsive delta-v at the decision
  epoch. An agent's offset from its SGP4 track is propagated with the
  Clohessy–Wiltshire equations, which agree with Orekit to about 0.1 m over 10
  minutes for a 1 m/s burn (see `tests/test_collision_avoidance_env.py`). Fuel,
  burn-duration limits, and rejected actions follow the maneuver contract.
- **Observations.** Every agent is ranked against every object with the same
  linear screen as training and calibration. `encode_local_observations`
  produces rows identical to the training encoder, so the actor sees exactly
  the inputs it was trained on.
- **Conjunctions.** Agent-involved pairs are screened every `fine_step_seconds`
  with a k-d tree, then refined by linear closest approach within the sample
  window. Samples below the safe separation are merged into one event per
  encounter. Catalog-only pairs are never compared.

## Outputs

| File | Contents |
| --- | --- |
| `results.csv` | One row per scenario and policy |
| `events.csv` | Every conjunction, with NORAD IDs, time of closest approach, and miss distance |
| `scalability_config.json`, `experiment_config.json` | Exact configurations |
| `scalability_info.json` | Start epoch, retained objects, policies, and environment |

Key `results.csv` columns:

| Column | Meaning |
| --- | --- |
| `conjunctions`, `collisions` | Agent-involved events below the safe separation and below the combined radius |
| `resolved_reference_conjunctions` | No-op events that no longer occur under this policy |
| `secondary_conjunctions` | Events under this policy that the no-op run did not have (maneuver-induced) |
| `total_delta_v_mps`, `maneuvers`, `rejected_actions` | Maneuver accounting |
| `<stage>_seconds_per_decision` | Propagation, observation, policy, maneuver, and screening time |
| `observation_microseconds_per_agent_decision` | Per-agent observation cost; flat in agent count, linear in catalog size |
| `process_peak_rss_mb` | Process peak memory so far |
| `mean_slot_offset_m`, `max_slot_offset_m` | Agents' final offset from their SGP4 track (the Clohessy–Wiltshire offset) |
| `mean_return_delta_v_mps` | Calculated two-burn delta-v to return to the track; see [EVALUATION.md](EVALUATION.md#drift-from-the-nominal-slot) |

Events match across policies when they involve the same pair within
`match_tolerance_seconds`.

## Implicit coordination

Satellites never coordinate explicitly: there is no priority rule and no
communication. Any coordination must emerge from each satellite acting on its own
local observation. To measure it, every satellite-to-satellite conjunction is
classified by how many of its two satellites maneuvered (`none`, `one`, `both`) and
whether it was resolved. Columns `pair_<group>_maneuvered` count conjunctions and
`pair_<group>_resolved` count those resolved.

In `oz scale`, the conjunctions classified are the no-op run's agent-agent
events. A member counts as maneuvering if it burned within the screening horizon
before the event's time of closest approach; the event is resolved if it no longer
occurs under the policy.

## Limitations

- Maneuvers are impulsive, and the Clohessy–Wiltshire offset is linear around
  the SGP4 track; the offset error grows over many orbits.
- The observation screen is exact but dense: its cost is proportional to agents
  times catalog objects.
- Peak memory is reported for the whole process, not per scenario.
