# Evaluation

`oz evaluate` compares decentralized policies on the same held-out episodes.
Every policy chooses actions from local observations only; the critic and global
state are never used.

```sh
.venv/bin/oz evaluate --config configs/mappo_smoke.json \
  --policy noop --policy rule --policy mappo=runs/<run>/checkpoints/latest.pt \
  --episodes 20 --output runs/evaluation_smoke
```

Without `--policy`, the no-op and rule-based baselines are compared.

## Policies

| Spec | Policy |
| --- | --- |
| `noop` | Never maneuvers. The lower bound every result is compared against. |
| `rule` | Rule-based Clohessy–Wiltshire avoidance (below). |
| `PATH` or `NAME=PATH` | A trained MAPPO checkpoint, run deterministically (most likely action). |

### Rule-based baseline

For each agent, the rule reads only the top-ranked neighbour block of its own
observation. If that threat is valid, unsafe (curved-orbit predicted miss at or below
the safe separation), and still approaching, it reconstructs both objects' states
from the observation and predicts the miss vector at closest approach along curved
J2 orbits. It then evaluates each of the six burn directions with the
Clohessy–Wiltshire displacement across the relative velocity, as in the maneuver
sizing study, and takes the burn that most increases the predicted miss.
Otherwise it coasts.

On 20 held-out stage-1 scenarios it cut real close approaches from 8.55 to 0.45 per
episode (95%) using 0.53 m/s per agent, consistent with the sizing study.

The rule uses the same information as the trained actor, so it answers whether
learning adds anything beyond a physics-aware single-threat heuristic.

## Episodes

Evaluation seeds start at `seed + 1,000,000`, disjoint from the training seeds
(`seed + episode index`). All policies play the same seeds. Generated scenarios
are drawn from the held-out test split of satellites and close-call shapes.

## Output

| File | Contents |
| --- | --- |
| `config.json` | Experiment configuration used |
| `evaluation_info.json` | Policy specs, seeds, and execution environment |
| `episodes.csv` | One row per policy and episode |
| `summary.csv` | One row per policy |

`mean_return` includes the reward-shaping offset, which depends only on each
episode's start state; it is comparable across policies because every policy plays
the same seeds.

`summary.csv` columns: `collision_rate`, `mean_return`, `mean_unsafe_agent_steps`,
`mean_final_unsafe_agents` (agents still in an unsafe conjunction when the episode
ended), `mean_rejected_actions`, `mean_delta_v_per_agent_mps`,
`mean_minimum_separation_meters`, `minimum_separation_meters` (worst episode), and the drift columns below.

## Drift from the nominal slot

Returning to the assigned orbit is out of scope: the policy decides when and how to
avoid, and restoring the orbit is left to standard station-keeping. The drift that
avoidance causes is measured instead:

- `mean_slot_offset_m` and `max_slot_offset_m`: each agent's distance at the end of
  the episode from where it would be had it never maneuvered.
- `mean_return_delta_v_mps`: the smallest two-burn delta-v that would bring it back,
  computed with the Clohessy–Wiltshire equations over a coast of up to one orbit and
  a transfer of up to two orbits. It is calculated, never flown.

For reference, one 0.5 m/s burn followed by 50 minutes of coasting leaves about
5 km of drift and a return cost of about 1.05 m/s (prograde), 0.26 m/s (radial),
or 0.5 m/s (cross-track).

## Implicit coordination

Satellites never coordinate explicitly: there is no priority rule and no
communication. Any coordination must emerge from each satellite acting on its own
local observation. To measure it, every satellite-to-satellite conjunction is
classified by how many of its two satellites maneuvered (`none`, `one`, `both`) and
whether it was resolved. Columns `pair_<group>_maneuvered` count conjunctions and
`pair_<group>_resolved` count those resolved.

In `oz evaluate`, a pair is tracked from the first decision at which it is unsafe;
a member counts as maneuvering if it burned while the pair was unsafe. The pair is
resolved only if it stopped being unsafe while its predicted closest approach was
still more than one decision interval away, so flying past each other does not
count.
