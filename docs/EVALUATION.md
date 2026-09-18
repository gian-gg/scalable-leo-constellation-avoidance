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
observation. If that threat is valid, unsafe (predicted miss at or below the safe
separation), and still approaching, it predicts the threat's miss vector at time of
closest approach from the linear relative motion. It then evaluates each of the six
burn directions with the Clohessy–Wiltshire equations for an impulsive burn of the
configured delta-v, and takes the burn that most increases the predicted miss
distance. Otherwise it coasts.

The rule uses the same information as the trained actor, so it answers whether
learning adds anything beyond a physics-aware single-threat heuristic.

## Episodes

Evaluation seeds start at `seed + 1,000,000`, disjoint from the training seeds
(`seed + episode index`). All policies play the same seeds in the same
environment.

## Output

| File | Contents |
| --- | --- |
| `config.json` | Experiment configuration used |
| `evaluation_info.json` | Policy specs, seeds, and execution environment |
| `episodes.csv` | One row per policy and episode |
| `summary.csv` | One row per policy |

`summary.csv` columns: `collision_rate`, `mean_return`, `mean_unsafe_agent_steps`,
`mean_final_unsafe_agents` (agents still in an unsafe conjunction when the episode
ended), `mean_rejected_actions`, `mean_delta_v_per_agent_mps`,
`mean_minimum_separation_meters`, and `minimum_separation_meters` (worst episode).
