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

## Trial 8 — shaping that previews the flat penalty, stages 1-2

Trial 7 left 19 pair encounters where neither satellite burned, all with a planned
miss between 900 m and 995 m. They were not a perception failure: every one was
flagged, a median of 9.6 minutes ahead.

The cause was a mismatch between the two safety signals. A realized 968 m approach
costs the flat penalty plus its shortfall, about -10.96, while the shaping potential
valued clearing the same threat at 30 x 0.032, about +0.96. The dense signal the
actor learns from understated the sparse cost tenfold, so coasting was rational.

The potential now previews both penalties: a threat predicted inside the safe
separation carries the flat term as well as its shortfall. The flat fraction is
`close_approach_flat_penalty / close_approach_penalty`, so no new weight is
introduced, and when the shaping weight equals the penalty scale the preview equals
the cost avoided exactly.

| Pair outcome (240 encounters) | Trial 6 | Trial 7 | Trial 8 |
| --- | ---: | ---: | ---: |
| Same direction | 18 (11% cleared) | 0 | 0 |
| Neither burned | 1 | 19 (0% cleared) | **0** |
| Opposite | 418 (99.8%) | 185 (95%) | 210 (99%) |
| One satellite only | 56 | 36 (100%) | 30 (100%) |
| Mild pairs (>= 900 m) | 48/49 | 32/49 | **49/49** |

| Metric, 64 agents | Rule | Trial 7 | Trial 8 |
| --- | ---: | ---: | ---: |
| Close approaches | 4.30 | 4.10 | **3.10** |
| Closest (m) | 289.1 | 420.0 | 420.0 |
| Mean shortfall | 0.169 | 0.128 | 0.158 |
| Delta-v (m/s) | 0.676 | 0.815 | 0.965 |
| Slot drift (m) | 3,681 | 3,860 | 4,326 |
| Satellite pairs cleared | 99% | 88% | 99% |
| Double threats cleared | 80% | 89% | 89% |

Close approaches fall 28% below the rule, the first clear win rather than a match.
Pairs return to the rule's level while double threats stay ahead of it. The cost is
fuel: delta-v rises 43% above the rule and drift with it. Mean shortfall rises
slightly because the encounters now prevented were the shallow ones, which leaves a
deeper-skewed remainder rather than worse behaviour.

Training diagnostics also improved: minimum separation during training rose to
2,093 m from 1,380 m, at unchanged unsafe-step counts and with entropy at 0.171.

## Trial 9 — shorter warning times in the training mix, stages 1-2

Threats first flagged 5-10 minutes ahead were the one category where the rule still
beat the actor (81% against 79%). Conjunctions were generated to occur 480-990 s
into an episode, so short-notice encounters were rare in training. The floor moved
to 360 s.

Held-out episodes are generated from the same configuration as training, so widening
the mix would also move the test set. `configs/eval_stage{1,2,3}.json` were frozen
at the previous generator settings and are now the evaluation configurations; the
rule reproduced 794/880 on them, which confirms the benchmark did not move.

| Cleared, by warning time at first flag | Rule | Trial 8 | Trial 9 |
| --- | ---: | ---: | ---: |
| 5-10 min | 81% | 79% | **83%** |
| Over 10 min | 93% | 96% | **97%** |
| Overall | 90% | 93% | **95%** |

| Metric, 64 agents | Rule | Trial 8 | Trial 9 |
| --- | ---: | ---: | ---: |
| Close approaches | 4.30 | 3.10 | **2.40** |
| Closest (m) | 289.1 | 420.0 | 420.0 |
| Mean shortfall | 0.169 | 0.158 | **0.131** |
| Delta-v (m/s) | **0.676** | 0.965 | 1.119 |
| Slot drift (m) | **3,681** | 4,326 | 5,938 |
| Delta-v including return (m/s) | **1.401** | 1.708 | 2.264 |

Close approaches fall 44% below the rule, and the short-warning gap closes without
costing anything in the long-warning band. Paired across the same 20 episodes the
improvement over trial 8 is -0.70 per episode (p = 0.003), better in 12 episodes and
worse in 2.

The cost is fuel. Delta-v is 65% above the rule and drift 61% above it, and the
total including the return burn is 2.264 against 1.401. Trial 6 was cheaper than the
rule overall because low drift offset a higher avoidance burn; that advantage is
gone. Reducing delta-v is the next task, on a policy trained through stage 3.

On the widened training distribution itself, where conjunctions are harder, the rule
falls to 83% while the actor holds 87%.

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

Trials 7 to 9 changed three further things, each validated through stage 2 against
the frozen benchmark:

| Change | Trial | Effect at 64 agents |
| --- | --- | --- |
| Miss direction as a unit vector | 7 | same-direction pair burns 18 to 0 |
| Shaping previews the flat penalty | 8 | close approaches 4.10 to 3.10 |
| Meeting-time floor 480 s to 360 s | 9 | close approaches 3.10 to 2.40 |

Against the rule's 4.30 close approaches per episode, the actor now reaches 2.40,
at 65% more delta-v.

With these weights a 500 m close approach costs 15 and one 0.5 m/s burn costs 0.5,
so a successful avoidance clearly outweighs its fuel, while a collision still costs
more than any near miss.

## Trial artifacts

Stage-2 checkpoints for trials 7 to 9, all evaluated with
`oz evaluate --config configs/eval_stage2.json --episodes 20`:

| Trial | Run directory | Checkpoint SHA-256 (first 16) |
| --- | --- | --- |
| 7 | `runs/sym_stage2` | `20746aa083c027a5` |
| 8 | `runs/pot_stage2` | `f36a372c06214852` |
| 9 | `runs/lead_stage2` | `09f67424d3c57859` |

Diagnostics are in `runs/trials/1{1,2,4}_*.json`. Code revision `5ec089b`; seed 42.

## Limitations

- Each trial used one seed; differences between variants have not been tested for
  seed variance.
- The weights were chosen from three variants, not a full sensitivity analysis.
- Trials 2 and 3 ran for 60 updates, about half of stage 1.
- Stage-1 performance plateaued at 72–77% in trials 3–6; the gap to the rule closed
  only after stages 2 and 3, so stage-1 trials are a poor predictor of final quality.
  See [TRAINING_RESULTS.md](TRAINING_RESULTS.md).
- Trials 7 to 9 stop at stage 2, so none of their numbers describe a 150-agent
  policy. They were run to choose settings, not to report results.
- Episode returns are not comparable across trials 7 to 9: trial 8 changed the
  reward function, which shifts the return scale independently of behaviour.
- Every trial is one training run. The paired tests quoted measure variation
  between episodes, not between training seeds.
