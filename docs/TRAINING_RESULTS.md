# Training results

First full run of the curriculum and its evaluation against the baselines. Settings
were chosen in [TRAINING_TRIALS.md](TRAINING_TRIALS.md); scenarios are described in
[TRAINING_SCENARIOS.md](TRAINING_SCENARIOS.md).

## Headline result

One shared 22-input, 7-action actor (about 11,700 weights), trained once through the
16 → 64 → 150 agent curriculum, resolves about **90% of the close approaches a
coasting constellation would suffer**, matching a physics-based rule and slightly
beating it at the two larger sizes. No policy caused a collision.

| Agents | No-op close approaches per episode | Rule | Trained actor |
| ---: | ---: | ---: | ---: |
| 16 | 8.55 | 0.45 (95%) | 0.60 (93%) |
| 64 | 43.6 | 4.30 (90%) | **4.00 (91%)** |
| 150 | 101.2 | 10.40 (90%) | **9.95 (90%)** |

## Training run

Curriculum: `configs/mappo_stage{1,2,3}.json`, each stage starting from the previous
stage's actor (a new critic each time, since its input grows with the population).

| Stage | Agents / objects | Updates | Environment steps | Median update | Wall clock |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 16 / 32 | 125 | 31,250 | 26.2 s | 23:54 → 00:49 |
| 2 | 64 / 128 | 110 | 11,000 | 34.6 s | 00:49 → 01:53 |
| 3 | 150 / 300 | 135 | 6,750 | 56.5 s | 01:53 → 04:01 |

Total 4 h 7 min on an Apple M1 MacBook Air (8 GB), about 0.87 GB peak memory, no
collisions in training. Stage 1 reproduced trial 6 exactly, confirming that a run is
fully determined by its configuration and seed.

Situation mixes: stage 1 used single threats only (50% debris, 25% harmless, 25%
quiet); stages 2 and 3 used the full mix (30% debris, 25% satellite pair, 15% double
threat, 15% harmless, 15% quiet).

## Evaluation

`oz evaluate`, 20 held-out episodes per size (seeds 1,000,042–1,000,061), drawn from
the test split of satellites and close-call shapes. All policies played the same
episodes; the actor ran deterministically with no critic.

| Size | Policy | Close approaches | Closest (m) | Mean shortfall | Delta-v (m/s) | Slot drift (m) | Return delta-v (m/s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | No-op | 8.55 | 41.0 | 0.395 | 0.000 | 1 | 0.00 |
| 16 | Rule | 0.45 | 653.2 | 0.033 | 0.533 | 3,103 | 0.61 |
| 16 | Actor | 0.60 | 653.2 | 0.081 | 0.763 | 2,912 | 0.44 |
| 64 | No-op | 43.60 | 40.2 | 0.385 | 0.000 | 1 | 0.00 |
| 64 | Rule | 4.30 | 289.1 | 0.169 | 0.676 | 3,681 | 0.73 |
| 64 | Actor | 4.00 | 182.9 | 0.235 | 0.847 | 3,225 | 0.49 |
| 150 | No-op | 101.20 | 40.6 | 0.378 | 0.000 | 1 | 0.00 |
| 150 | Rule | 10.40 | 206.9 | 0.217 | 0.670 | 3,632 | 0.73 |
| 150 | Actor | 9.95 | 211.3 | 0.228 | 0.848 | 3,214 | 0.50 |

Shortfall is how far inside the 1 km safe separation a close approach fell (0 at the
threshold, 1 at contact). No collisions occurred in any run.

### Delta-v including the return to the slot

Station-keeping is out of scope for the policy, but the drift it causes is measured.
Adding the calculated two-burn return makes the total cost comparable:

| Size | Rule (avoid + return) | Actor (avoid + return) |
| ---: | ---: | ---: |
| 64 | 0.676 + 0.725 = 1.401 | 0.847 + 0.493 = **1.340** |
| 150 | 0.670 + 0.726 = 1.396 | 0.848 + 0.501 = **1.349** |

The actor spends more delta-v avoiding but leaves satellites closer to their nominal
orbits, so the total is slightly lower.

### Implicit coordination in satellite-to-satellite conjunctions

No satellite communicates or follows a priority rule. Each agent-agent conjunction is
classified by how many of its two satellites maneuvered.

| Size | Policy | Neither (resolved) | One (resolved) | Both (resolved) |
| ---: | --- | --- | --- | --- |
| 64 | Rule | 49 (48) | 0 (0) | 40 (40) |
| 64 | Actor | 62 (61) | **7 (7)** | 41 (34) |
| 150 | Rule | 127 (126) | 0 (0) | 75 (72) |
| 150 | Actor | 155 (150) | **15 (14)** | 79 (61) |

The rule always makes both satellites maneuver. The trained actor sometimes resolves
a conjunction with one satellite maneuvering alone, and every such case was resolved
(7/7 and 14/15). When both of its satellites maneuver it is less reliable than the
rule (34/41 and 61/79 against 40/40 and 72/75), which is the clearest remaining
weakness.

## Diagnosis of the remaining close approaches

Every planned conjunction under 1 km in the held-out episodes was replayed under both
policies and recorded with its situation type, warning time, relative speed, planned
miss and the actions both satellites took (`runs/trials/09_diagnosis.json`, 3,860
records: 20 episodes at 64 agents, 10 at 150).

| Category | Rule cleared | Actor cleared |
| --- | ---: | ---: |
| Debris (single threat) | 93% | 92% |
| Double threat | 79% | **84%** |
| Satellite pair | **99%** | 96% |
| Planned miss under 200 m | 63% | **76%** |
| Planned miss 500–1000 m | 96% | 95% |
| Flagged 5–10 min ahead | **80%** | 75% |
| Flagged over 10 min ahead | 92% | **94%** |

(150 agents; the 64-agent table has the same shape.)

The actor is better on the hard cases — two simultaneous threats and deep conjunctions
— and worse on short warning and on satellite pairs. Failures are not caused by
coasting: the actor burned in all but 3 of its 106 failures, and burned *more* often
than the rule on average (2.79 against 2.16 decision steps).

### Root cause of the paired-maneuver weakness

Splitting satellite-to-satellite conjunctions by the two satellites' first burns:

| First burns | Count | Cleared |
| --- | ---: | ---: |
| Opposite directions | 418 | 417 (99.8%) |
| Same direction | 18 | 2 (11%) |

The rule picks opposite directions in 280 of 290 pairs because each satellite signs its
burn by its own miss vector. The shared actor picks the same direction in 18 cases, and
those are almost the whole paired-maneuver deficit: 9 of its 12 pair failures at 150
agents are both satellites burning −S. They happen on tighter conjunctions (median
planned miss 316 m against 619 m) where the two observations are closest to mirror
images of each other, so one shared policy maps them to one shared action and the two
burns cancel.

## Comparison with the trials

Stage 1 alone plateaued at 72–77% of close approaches resolved (trials 3–6). The same
actor after stages 2 and 3 resolves 93% on stage-1 scenarios and about 90% on the
larger mixes. The curriculum, not further stage-1 tuning, closed that gap.

## Reproducibility

```sh
runs/run_curriculum.sh        # stage 1 -> 2 -> 3
.venv/bin/oz evaluate --config configs/mappo_stage3.json \
  --policy noop --policy rule --policy final=runs/stage3/checkpoints/latest.pt \
  --episodes 20 --output runs/final_eval_stage3
```

| Item | SHA-256 |
| --- | --- |
| `runs/stage1/checkpoints/latest.pt` | `0b7d0a933d22da74e7f33ed1100be9131c2cdb20978c05018ae63dbea1463a5e` |
| `runs/stage2/checkpoints/latest.pt` | `b1289824fc8dd7846f0be5c48832ebf762640d39154082d65ede665e4033f0b3` |
| `runs/stage3/checkpoints/latest.pt` | `6d06a4c1802acd1596f3f79b56683bc5e569b30d96ca9d24dc94d19b5c61ccea` |

Code revision `4b7e19a`; seed 42; Python 3.11.4, NumPy 1.24.4, PyTorch 2.4.1;
macOS 26.0 on Apple M1.

## Limitations and open work

- **One seed.** Every number here comes from a single training run. Seed repeats are
  needed before the results are reported as robust.
- **Twenty episodes per size.** Differences of a few tenths of a close approach per
  episode are within noise; more episodes would tighten the comparison.
- **Paired maneuvers.** When both satellites of a pair maneuver, the actor resolves
  77–83% against the rule's 96–100%, because near-mirror observations make the shared
  policy pick the same burn direction for both satellites.
- **Short warning.** Threats first flagged 5–10 minutes ahead are cleared 75% of the
  time against the rule's 80%.
- **Worst case.** Closest approaches of 183–211 m remain at the larger sizes for both
  policies.
- **Scale not yet measured.** The real-catalog sweeps (`oz scale`, up to 20,000
  objects and 10,000 agents) have not been run; they are the evidence for the
  scalability claim.
- **Avoidance only.** Returning to the nominal orbit is out of scope; drift and the
  calculated return cost are reported instead.
