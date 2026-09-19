# Maneuver sizing findings

## Final result

- Commanded delta-v per maneuver: **0.5 m/s**
- Maximum thrust: **7 N** (specific impulse 300 s, maximum burn 60 s)
- Selection rule: the smallest candidate delta-v that clears at least 95% of the
  calibration's reference conjunctions, when each maneuvering satellite burns at
  every decision from its own first k = 1 detection until closest approach
- Resolved at 0.5 m/s: **96.6%** (479 of 496 conjunctions)
- Orekit spot-check: 9 of 9 sampled burns landed within 5% of the 1 km safe
  separation

These values are the defaults in `thesis/config.py`, `configs/mappo_toy.json`,
and `configs/mappo_smoke.json`. Method details are in
[MANEUVER_SIZING.md](MANEUVER_SIZING.md).

## Inputs

| Input | Value |
| --- | --- |
| Conjunctions | 496 reference conjunctions from the 20,000-object, 72-hour calibration |
| Calibration catalog | `configs/k_dt_calibration_72h.json` |
| Neighbourhood and decision interval | k = 1, 120 s |
| Safe separation | 1,000 m |
| Candidate delta-v (m/s) | 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5 |
| Target resolved fraction | 95% |
| Satellite masses for thrust | 300 and 800 kg |

## Measured warning time

Every reference conjunction was detected by its agent's k = 1 view before closest
approach.

| Statistic | Warning |
| --- | ---: |
| Minimum | 0.6 min |
| 5th percentile | 7.9 min |
| 25th percentile | 12.5 min |
| Median | 14.9 min |
| 75th percentile | 15.8 min |
| 95th percentile | 16.4 min |
| Maximum | 29.1 min |

| Warning | Conjunctions |
| --- | ---: |
| Under 6 min | 11 |
| 6–10 min | 40 |
| 10–15 min | 216 |
| 15–20 min | 227 |
| 20–31 min | 2 |

Threats rarely reach the top of the ranking at the full 30-minute screening
horizon; most appear about 15 minutes before closest approach.

## Resolved fraction by delta-v per burn

| Scenario | 0.1 | 0.2 | 0.5 | 1 | 2 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Measured warning, burn every decision (**adopted**) | 50.0% | 79.0% | **96.6%** | 98.6% | 99.4% | 99.8% |
| Measured warning, single burn | 14.7% | 29.0% | 60.3% | 92.9% | 99.0% | 99.8% |
| Guaranteed minimum warning (6 min), 3 burns | 11.5% | 23.6% | 52.2% | 85.1% | 100.0% | 100.0% |
| Guaranteed minimum warning (6 min), 1 burn | 5.4% | 11.3% | 29.0% | 52.0% | 85.3% | 100.0% |

Per-burn delta-v needed with the adopted rule: median 0.10 m/s, 90th percentile
0.27 m/s, 95th percentile 0.39 m/s, 99th percentile 1.07 m/s.

## Why the measured-warning rule was adopted

The guaranteed-minimum rule assumes every threat is first seen only three
decisions before closest approach. It selects 2 m/s and needs 10–27 N, but only 11
of 496 conjunctions actually give that little warning. The measured-warning rule
uses each conjunction's real first detection under the calibrated k and delta t,
and cuts the per-maneuver delta-v fourfold.

## Required thrust

A 0.5 m/s maneuver must finish within the 60 s maximum burn:

| Satellite mass | Minimum thrust |
| ---: | ---: |
| 300 kg | 2.5 N |
| 800 kg | 6.7 N |

7 N covers both. This is small chemical or cold-gas thruster class, above the
low-thrust electric propulsion typical of large LEO constellations; the thesis
therefore assumes a satellite able to perform short-notice avoidance maneuvers.

## Orekit spot-check

| Event | Delta-v per burn (m/s) | Coasting miss (m) | Maneuvered miss (m) |
| ---: | ---: | ---: | ---: |
| 0 | 0.442 | 704 | 994 |
| 41 | 0.065 | 964 | 1,000 |
| 123 | 0.965 | 290 | 972 |
| 164 | 0.247 | 872 | 998 |
| 205 | 0.324 | 780 | 997 |
| 287 | 0.743 | 467 | 983 |
| 328 | 0.600 | 600 | 991 |
| 369 | 0.856 | 422 | 979 |
| 410 | 0.515 | 688 | 992 |

Finite burns land slightly short of the impulsive prediction, by up to about 3%
for the largest burns.

## Reproducibility

```sh
.venv/bin/oz size-maneuvers --config configs/maneuver_sizing.json --output runs/maneuver_sizing
```

| Item | SHA-256 |
| --- | --- |
| `sizing_config.json` | `49c70b658b28c9d976bb6bd081aeae73bfaa197a6876f9c0db89c02836adb177` |
| `recommendation.json` | `839f37dc0b3afadf39ede58e3a5d975fefd89c382a7b3ee99d9fe9ca07076603` |
| `sizing_table.csv` | `bbaea0dd0dbfab80b85410e847fbed565047e9846e7a5ea3e4f8f6618f9573d4` |
| `detection_leads.csv` | `bec74913afb4fadbfaf28ac3dfc60ddce7390bfe6c90789e8f9f314a51cd15c6` |
| `requirements.csv` | `417f9c0421c2abdd1c166c6c6d4ecd302570d511b8e99011bb690a79330d490f` |
| `spot_checks.csv` | `1851236025621625edaaf5a1d69ef1cc5f9b4b58cd5e9757ee96a273d8c5c418` |

Run on 2026-09-19 on top of revision `b9379e0` with the sizing code in the working
tree, on the Apple M1 MacBook Air used for the calibration.

## Limitations and paper disclosures

- The rule was changed from the guaranteed-minimum warning to the measured warning
  after the first results were seen. Both are reported above.
- The adopted rule assumes the agent burns from its first detection; a policy that
  waits needs larger burns.
- In satellite-to-satellite conjunctions only one satellite maneuvers in the
  calculation, which is conservative.
- Burns are impulsive in the calculation and relative motion is linear during the
  encounter; the few slow (below 1 km/s) conjunctions are least accurate.
- The candidate grid is coarse: the adopted rule needs 0.39 m/s at the 95th
  percentile and selects the next candidate, 0.5 m/s.
- Satellite masses (300 and 800 kg) are assumptions, not catalog data.
