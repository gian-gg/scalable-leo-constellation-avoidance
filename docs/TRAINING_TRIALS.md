# Training trials

Short runs made before the main curriculum to check that the pipeline works and to
choose the reward weights and exploration settings. Every trial was evaluated with
`oz evaluate` on the same 20 held-out stage-1 scenarios (seeds 1,000,042–1,000,061,
test split). Artifacts are in `runs/trials/`.

## Trial 1 — can the scenarios be solved?

No training; the no-op and rule-based baselines only.

The first run exposed that the straight-line miss prediction flagged real close
calls only about 5.5 minutes ahead (median), leaving the rule too few decisions. The
top threat's time-to-closest-approach and miss features were changed to a curved J2
prediction, which flags them about 11 minutes ahead. See
[COLLISION_AVOIDANCE_ENVIRONMENT.md](COLLISION_AVOIDANCE_ENVIRONMENT.md).

| Policy | Close calls per episode | Closest (m) | Delta-v per agent (m/s) |
| --- | ---: | ---: | ---: |
| No-op | 8.55 | 41.0 | 0.00 |
| Rule, straight-line prediction | 5.95 | 83.1 | 0.78 |
| Rule, curved prediction | 0.45 | 653.2 | 0.53 |

## Trial 2 — original settings, 60 updates

Entropy coefficient 0.01, actor learning rate 3e-4, close-approach penalty −10,
shaping weight 10.

The policy first learned to stop wasting fuel, then its entropy fell from 1.94 to
0.05 within 20 updates and it settled on never maneuvering. Evaluated, it was
identical to no-op (8.55 close calls per episode).

## Trial 3 — three variants, 60 updates each

All three used entropy coefficient 0.05 and actor learning rate 1e-4.

| Variant | Change | Entropy at update 28 | Close calls per episode | Closest (m) | Delta-v per agent (m/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | More exploration only | 0.04 | 8.55 | 41.0 | 0.00 |
| B | A + close-approach penalty −30 and shaping weight 30 | 0.28 | **1.95** | 62.7 | 1.11 |
| C | A + 80% threatened agents, no quiet agents | 0.05 | 8.55 | 41.0 | 0.00 |

Only B learned to avoid: it resolved 77% of the no-op close calls after 60 updates,
against 95% for the rule, using about twice the rule's delta-v.

## Decision

The stage configurations (`configs/mappo_stage{1,2,3}.json`) adopt variant B:

| Setting | Value |
| --- | ---: |
| `rewards.close_approach_penalty` | −30 |
| `rewards.shaping_weight` | 30 |
| `rewards.collision_penalty` | −100 (unchanged) |
| `rewards.delta_v_penalty_per_mps` | 1 (unchanged) |
| `training.entropy_coefficient` | 0.05 |
| `training.actor_learning_rate` | 1e-4 |

With these weights a 500 m close approach costs 15 and one 0.5 m/s burn costs 0.5,
so a successful avoidance clearly outweighs its fuel, while a collision still costs
more than any near miss.

## Limitations

- Each trial used one seed; differences between variants have not been tested for
  seed variance.
- The weights were chosen from three variants, not a full sensitivity analysis.
- Trials 2 and 3 ran for 60 updates, about half of stage 1.
