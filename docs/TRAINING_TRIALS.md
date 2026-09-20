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

## Trial 4 — variant B for a full stage 1, 125 updates

Improvement flattened after about update 70. Evaluated: 2.40 close approaches per
episode (72% of the no-op total resolved) with 0.83 m/s per agent — slightly fewer
resolved than trial 3B but cheaper, so the extra updates traded safety for fuel.
One crash (a satellite whose TLE could not be propagated to the drawn epoch) was
fixed by redrawing the scenario, and the run resumed from its checkpoint.

## Trial 5 — adding the predicted miss direction, 125 updates

Three features were added to each neighbour block: the predicted miss vector at
closest approach. Learning started 5–10 updates earlier, and the close approaches it
failed to clear were much milder (closest 212 m against 63 m, mean shortfall 0.16
against 0.25, 0.68 m/s), but it resolved slightly fewer of them (68%).

## Trial 6 — adding a flat penalty per close approach, 125 updates

A diagnosis of trial 5 found the actor ignored *mild* conjunctions: of 46 close
approaches the rule cleared and the actor did not, it never maneuvered in 24, and
their planned miss was about 770 m. A flat −10 for any close approach under 1 km was
added so that even a shallow one outweighs a 0.5 m/s burn. Evaluated: 2.05 close
approaches (76%) with 1.02 m/s.

The actor also agreed with the rule's chosen direction only 49% of the time, and
coasted on 29% of flagged threats, preferring one default direction.

## Trial 7 — unit miss direction, stages 1-2

A diagnosis of the trained actor (see
[TRAINING_RESULTS.md](TRAINING_RESULTS.md)) found that satellite pairs failed when
both satellites burned the same way: opposite-direction burns cleared 417 of 418
conjunctions, same-direction burns 2 of 18. The cause was the miss-direction
feature, which was stored scaled by the miss vector and so shrank to a median
magnitude of 0.36 in those encounters against 0.71 in the ones that worked, while
the two observations stayed distinguishable (median L2 distance 1.51). The actor
fell back to one default burn.

The feature became a unit vector; the miss distance was already separate, so
nothing was lost. Stages 1 and 2 were retrained into `runs/sym_stage{1,2}` and
evaluated on the same 20 held-out stage-2 scenarios. The rule reproduced its
earlier numbers exactly, which confirms the comparison.

| Pair outcome (240 encounters) | Before | After |
| --- | ---: | ---: |
| Same direction | 18 (11% cleared) | **0** |
| Opposite | 418 (99.8% cleared) | 185 (95% cleared) |
| One satellite only | 56 | 36 (100% cleared) |
| Neither burned | 1 | **19 (0% cleared)** |

| Metric, 64 agents | Trial 6 actor | Rule | Trial 7 actor |
| --- | ---: | ---: | ---: |
| Close approaches | 4.00 | 4.30 | 4.10 |
| Closest (m) | 182.9 | 289.1 | **420.0** |
| Mean shortfall | 0.235 | 0.169 | **0.128** |
| Delta-v (m/s) | 0.847 | 0.676 | 0.815 |
| Slot drift (m) | 3,225 | 3,681 | 3,860 |

Double threats rose from 86% to 89% and debris from 92% to 94%. The remaining
close approaches are much shallower: the worst case beats the rule and the mean
shortfall is the lowest of the three.

The regression is 19 pair encounters where neither satellite burned, at a median
planned miss of 968 m. These are the mild conjunctions of trial 5 returning, and
they are the target of the next reward change rather than of this feature.

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
| `rewards.close_approach_flat_penalty` | −10 (trial 6) |

With these weights a 500 m close approach costs 15 and one 0.5 m/s burn costs 0.5,
so a successful avoidance clearly outweighs its fuel, while a collision still costs
more than any near miss.

## Limitations

- Each trial used one seed; differences between variants have not been tested for
  seed variance.
- The weights were chosen from three variants, not a full sensitivity analysis.
- Trials 2 and 3 ran for 60 updates, about half of stage 1.
- Stage-1 performance plateaued at 72–77% in trials 3–6; the gap to the rule closed
  only after stages 2 and 3, so stage-1 trials are a poor predictor of final quality.
  See [TRAINING_RESULTS.md](TRAINING_RESULTS.md).
