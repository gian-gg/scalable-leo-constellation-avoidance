# Neighborhood-size and decision-interval calibration

## Final result

- Selected neighborhood size: `k = 1`
- Selected decision interval: `delta t = 120 seconds`
- Primary evidence: 20,000-object, 72-hour run
- Primary-run status: accepted on calibration and held-out validation seeds
- Training implication: each actor observes its own features and the single
  highest-ranked neighbor block, and may select an action every 120 seconds

The same pair was selected by the 5,000-object, 10,000-object, 20,000-object
24-hour, and 20,000-object 72-hour experiments.

## Research question

The calibration determines the smallest constant-size local neighborhood `k`
and the largest acceptable decision interval `delta t` that preserve threat
detection. It is performed before policy training so these values are chosen
from orbital geometry rather than reinforcement-learning performance.

The actor input remains fixed in size when the catalog grows: one own-state
feature block, `k` ranked neighbor blocks, and the neighbor padding mask. The
centralized critic may still receive the global training state.

## Method

This is an offline, propagation-only calibration. It does not train a policy,
apply collision-avoidance maneuvers, or alter any orbit.

1. Validate and join the frozen TLE catalog and object metadata by NORAD ID.
2. Start all propagation at the latest TLE epoch in that catalog.
3. Apply the 200–2,000 km LEO altitude filter at the common start epoch.
4. Select deterministic nested populations of 16, 64, and 256 eligible agents.
5. Propagate the catalog at 60-second resolution and screen only agent-to-catalog
   pairs using a `cKDTree`; catalog-only pairs are not compared.
6. Fine-propagate candidate encounter windows at 10-second resolution and refine
   time of closest approach between samples.
7. Treat a refined approach below 1,000 m as a reference conjunction.
8. Rank possible neighbors once at shared decision epochs, retaining ranks up to
   the largest candidate `k`.
9. Evaluate all prefixes `k = 1, 2, 4, 8, 16` and all decision intervals
   `delta t = 60, 120, 300, 600 seconds` without rerunning orbital propagation.
10. Pool integer event counts across agent counts and seeds, apply the declared
    thresholds to calibration seeds, and audit the selected pair on validation
    seeds.

Propagation uses SGP4 with WGS-72 gravity. States use the TEME frame and are
converted internally to SI units: metres and metres per second.

## Exact experimental configuration

| Parameter | Value |
| --- | --- |
| Catalog start rule | Latest TLE epoch |
| LEO altitude range | 200–2,000 km |
| Maximum TLE age within snapshot | 14 days |
| Coarse propagation step | 60 s |
| Fine/reference propagation step | 10 s |
| Fine-window padding | 60 s on each side |
| Conservative maximum relative speed | 20,000 m/s |
| Safe-separation threshold | 1,000 m |
| Screening horizon | 1,800 s |
| Agent populations | 16, 64, 256 |
| Candidate `k` | 1, 2, 4, 8, 16 |
| Candidate `delta t` | 60, 120, 300, 600 s |
| Calibration seeds | 0–9 |
| Validation seeds | 100–104 |
| Random-number generator | NumPy PCG64 |
| Required threat recall | at least 99.9% |
| Required timely-detection fraction | at least 99% |
| Timely-detection requirement | at least 3 decisions before TCA |

For each seed, one deterministic permutation is generated from the eligible
agent pool. The 16-agent set is a prefix of the 64-agent set, which is a prefix
of the 256-agent set. Calibration therefore contains 30 pooled samples
(10 seeds x 3 populations), while validation contains 15 samples
(5 seeds x 3 populations).

The selection rule first minimizes `k`. Among passing candidates at that `k`,
it selects the largest `delta t`. Validation data do not influence selection;
they are used only to accept or reject the calibration-selected pair. No
fallback candidate is selected after a validation failure.

## Metric definitions

Let `N_ref` be the pooled number of reference conjunction occurrences across
the relevant seeds and agent populations.

```text
threat recall = N_detected / N_ref
timely-detection fraction = N_detected_with_at_least_3_decisions / N_ref
```

A reference threat is detected when its other object appears within the first
`k` ranked neighbors while its TCA is in the future and within the 1,800-second
screening horizon. A zero-reference sample is recorded as zero, not as a
vacuous perfect score. Rates are computed after summing event counts rather
than by averaging per-sample percentages.

`Unique reference conjunctions` below refers to distinct events in the shared
reference manifest. `Pooled reference occurrences` counts those events again
where they apply to different seeds and nested agent populations. The latter is
the denominator used for the reported calibration and validation rates.

## Catalogs

| Catalog | Payloads | Debris | Agent candidates | TLE epoch range (UTC) | Common start epoch (UTC) |
| --- | ---: | ---: | ---: | --- | --- |
| 5,000 | 3,183 | 1,817 | 349 | 2026-09-01 16:52:40 to 2026-09-15 12:20:13 | 2026-09-15 12:20:13 |
| 10,000 | 6,366 | 3,634 | 349 | 2026-09-01 13:58:42 to 2026-09-15 12:20:13 | 2026-09-15 12:20:13 |
| 20,000 | 12,732 | 7,268 | 349 | 2026-09-01 13:58:42 to 2026-09-15 11:40:47 | 2026-09-15 11:40:47 |

All records in the final 20,000-object catalog passed strict TLE parsing,
freshness validation, the start-epoch LEO filter, and a 72-hour SGP4 preflight.
No record was stale-filtered or altitude-filtered in the completed 20,000-object
runs.

The 5,000-object catalog is an exact subset of the 10,000-object catalog. The
cleaned 20,000-object catalog contains 9,973 of the 10,000-object records; 27
records were replaced during the 72-hour data-quality cleanup. The 20,000-object
24-hour and 72-hour runs use exactly the same catalog. Because the 20,000-object
catalog has a slightly earlier common start epoch, catalog-size comparisons are
sensitivity checks rather than a perfectly controlled scaling experiment.

### Agent-candidate composition

The metadata field `is_agent_candidate` is the authoritative eligibility flag.
The same 349 candidates were retained in all catalog sizes:

| Constellation label | Candidates |
| --- | ---: |
| Starlink | 247 |
| Unspecified/blank | 69 |
| OneWeb | 14 |
| Kuiper | 9 |
| Qianfan | 5 |
| Planet | 3 |
| Spire | 2 |

Space stations, attached modules, docked vehicles, debris, and other
non-maneuverable objects are explicitly non-agents. Debris remains present as a
possible threat.

### Radius metadata

No radius was blank in the archived calibration catalogs, so the configured
1.0 m default was not applied. The final 20,000-object radius distribution is:

| Radius | Objects |
| ---: | ---: |
| 0.10 m | 61 |
| 0.15 m | 66 |
| 0.20 m | 153 |
| 0.30 m | 7,179 |
| 0.40 m | 89 |
| 1.00 m | 29 |
| 1.50 m | 3,108 |
| 2.00 m | 9,315 |

Radii determine whether a reference conjunction is also labeled a physical
collision. They do not change the 1,000 m conjunction threshold.

## Results across catalog sizes and durations

| Dataset | Runtime | Candidate windows | Unique conjunctions | Debris conjunctions | Selected `k` | Selected `delta t` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 objects, 24 h | 682.66 s | 22,346 | 6 | 0 | 1 | 120 s |
| 10,000 objects, 24 h | 1,410.62 s | 46,060 | 12 | 1 | 1 | 120 s |
| 20,000 objects, 24 h | 3,006.18 s | 92,743 | 29 | 4 | 1 | 120 s |
| 20,000 objects, 72 h | 9,036.17 s | 297,014 | 496 | 13 | 1 | 120 s |

Every completed full calibration was accepted. No reference event was labeled
a physical collision.

### 5,000 objects, 24 hours

- Status: accepted
- Selected pair: `k = 1`, `delta t = 120 seconds`
- Calibration recall/timely fraction: 100% / 100%
- Validation recall/timely fraction: 100% / 100%
- Reference events: 6 satellite-to-satellite, 0 debris-to-satellite
- Limitation: only six unique events, so this run is preliminary evidence

At `k = 1`, 60 and 120 seconds passed; 300 and 600 seconds failed the timely
threshold. At `k >= 2`, 300 seconds passed. Every `k` failed at 600 seconds.

### 10,000 objects, 24 hours

- Status: accepted
- Selected pair: `k = 1`, `delta t = 120 seconds`
- Calibration recall/timely fraction: 100% / 100%
- Validation recall/timely fraction: 100% / 100%
- Reference events: 11 satellite-to-satellite, 1 debris-to-satellite

At `k = 1`, 60 and 120 seconds passed. At 300 seconds, calibration timely
detection was 89.17% and validation timely detection was 93.44%, so it failed.
At `k >= 2`, 300 seconds passed. Every `k` failed at 600 seconds.

### 20,000 objects, 24 hours

- Status: accepted
- Runtime: 3,006.18 seconds (50 min 6 s)
- Candidate encounter windows: 92,743
- Unique reference events: 29
- Event types: 25 satellite-to-satellite and 4 debris-to-satellite
- Physical collisions: 0
- Selected pair: `k = 1`, `delta t = 120 seconds`
- Calibration: 284/284 detected and 284/284 timely
- Validation: 144/144 detected and 144/144 timely
- Calibration recall/timely fraction: 100% / 100%
- Validation recall/timely fraction: 100% / 100%

At `k = 1`, 60 and 120 seconds passed. The timely fraction at 300 seconds was
71.13% for calibration and 72.92% for validation. At 300 seconds, `k >= 4`
passed both splits. Every `k` failed at 600 seconds.

### 20,000 objects, 72 hours — primary experiment

- Status: accepted
- Runtime: 9,036.17 seconds (2 h 30 min 36 s)
- Candidate encounter windows: 297,014
- Unique reference events: 496
- Event types: 483 satellite-to-satellite and 13 debris-to-satellite
- Physical collisions: 0
- Selected pair: `k = 1`, `delta t = 120 seconds`
- Calibration: 4,555/4,555 detected and 4,531/4,555 timely
- Validation: 2,433/2,433 detected and 2,417/2,433 timely
- Calibration recall/timely fraction: 100% / 99.473%
- Validation recall/timely fraction: 100% / 99.342%

Complete primary-run sweep:

| `k` | `delta t` | Calibration recall | Calibration timely | Calibration pass | Validation recall | Validation timely | Validation pass |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |
| 1 | 60 s | 100.000% | 99.802% | yes | 100.000% | 99.753% | yes |
| 1 | 120 s | 100.000% | 99.473% | yes | 100.000% | 99.342% | yes |
| 1 | 300 s | 99.649% | 78.310% | no | 99.548% | 76.737% | no |
| 1 | 600 s | 97.519% | 0.373% | no | 96.630% | 0.247% | no |
| 2 | 60 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 2 | 120 s | 100.000% | 99.802% | yes | 100.000% | 99.753% | yes |
| 2 | 300 s | 100.000% | 93.041% | no | 100.000% | 91.163% | no |
| 2 | 600 s | 99.715% | 0.373% | no | 99.383% | 0.247% | no |
| 4 | 60 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 4 | 120 s | 100.000% | 99.802% | yes | 100.000% | 99.753% | yes |
| 4 | 300 s | 100.000% | 98.573% | no | 100.000% | 98.027% | no |
| 4 | 600 s | 99.802% | 0.373% | no | 99.753% | 0.247% | no |
| 8 | 60 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 8 | 120 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 8 | 300 s | 100.000% | 99.363% | yes | 100.000% | 98.931% | no |
| 8 | 600 s | 99.802% | 0.373% | no | 99.753% | 0.247% | no |
| 16 | 60 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 16 | 120 s | 100.000% | 100.000% | yes | 100.000% | 100.000% | yes |
| 16 | 300 s | 100.000% | 99.627% | yes | 100.000% | 99.630% | yes |
| 16 | 600 s | 99.802% | 0.373% | no | 99.753% | 0.247% | no |

At the smallest neighborhood, both 60 and 120 seconds passed, so the selection
rule chose the less frequent 120-second interval. A 300-second interval required
`k = 16` to pass both calibration and validation. No 600-second combination
passed.

## Interpretation

The selected `k = 1` means one correctly ranked local threat was sufficient to
retain every reference conjunction in both splits of the primary experiment.
It does not mean that the catalog contained one nearby object or that only one
pair was screened. All 20,000 objects remained eligible threats during
agent-to-catalog screening; `k` limits only the actor's visible ranked-neighbor
blocks.

The selected `delta t = 120 seconds` was the largest passing interval at the
smallest passing neighborhood. Increasing the interval to 300 seconds sharply
reduced the number of threats seen with three decisions remaining, even when
raw recall stayed high. The result therefore reflects maneuver decision lead
time, not only whether a conjunction eventually became visible.

The 72-hour experiment is the strongest evidence because it contains 496 unique
events and 6,988 pooled reference occurrences across calibration and validation,
while reproducing the same selection as every smaller experiment.

## Reproducibility

Primary commands:

```sh
.venv/bin/oz calibrate \
  --config configs/k_dt_calibration.json \
  --output runs/k_dt_calibration_20k_24h

.venv/bin/oz calibrate \
  --config configs/k_dt_calibration_72h.json \
  --output runs/k_dt_calibration_20k_72h
```

Each run directory contains the resolved configuration, catalog summary,
deterministic agent selections, reference events, per-sample metrics, pooled
metrics, recommendation, and human-readable summary.

| Item | SHA-256 |
| --- | --- |
| 5k TLE catalog | `12da3f0397fe8c5ba10c245746753b5f562effff95ae1160ac721c28a545b970` |
| 5k metadata CSV | `1ad79cab5109d68c1b4875f4bd3cbd88774f6c09ffed1a3ac93f735f51fc3fc6` |
| 10k TLE catalog | `0301100e199e94a138e6be9f3d9b6cf7fab354e657d8f11293fa524eee72a5d2` |
| 10k metadata CSV | `b8b08d3899905013712cdc4c36193f7aaf6dfd7a0ad41f1b4e9ecf4b57fc5ed5` |
| 20k TLE catalog | `ea307c39bf996d382ed1712f9074646055b4a982cbafd54aabe5395a91129a16` |
| 20k metadata CSV | `7078ead1169393b607bcfe030def44b1a6a266490e6d8f9da1f59adf31f5093a` |
| 24 h resolved configuration | `a0c3ab97ef5f487548ec86581327faba6007bf12b871202bc6f2cf454c33dedd` |
| 24 h pooled metrics | `963f5f162001c1f0fbf13f3a07bb5655b6798e1b8bc660964ad8124dace58c61` |
| 24 h recommendation | `b4ca46db04b10ce13534add772382289e951e621c6105ba44c94c3aefca7c57c` |
| 72 h resolved configuration | `2a625031951b42fe0b4dfdbe4bbd4d1a6d9c32d115e98246c3359829fd67991a` |
| 72 h pooled metrics | `e55358819d4edb21c0bca86fca23f792789962566fc34696bbe43908696bf9f3` |
| 72 h recommendation | `a57d9c74cfda207ef57637b795f93d1abffc5e71c307c24ad3bce3ec00a6e875` |

- Calibration implementation revision: `45725cf6795a475fa5f670fcb53e51821c371d4f`
- Python: 3.11.4
- NumPy: 1.24.4
- SciPy: 1.15.3
- sgp4: 2.27
- Runtime hardware: Apple M1 MacBook Air, 8 CPU cores, 8 GB memory
- Runtime platform: macOS 26.0, ARM64

## Limitations and paper disclosures

- The experiment uses one frozen catalog snapshot. Validation holds out agent
  selections, not a later TLE snapshot. Temporal generalization is untested.
- The 72-hour run propagates a fixed TLE snapshot for three days. It does not
  ingest newly published orbital elements during the window.
- Some TLEs were already nearly 14 days old at the common start; by the end of
  the 72-hour window, their effective extrapolation age was greater.
- SGP4/TLE propagation is suitable for screening experiments but is not a
  high-fidelity operational conjunction assessment with covariance data.
- The reference threshold is deterministic miss distance below 1,000 m. The
  study does not estimate collision probability or model position uncertainty.
- Object radii are metadata estimates. No reference event crossed the combined
  radius threshold, so the experiments provide conjunction-detection evidence,
  not examples of physical impacts.
- The 20,000-object catalog includes 69 agent candidates with a blank
  constellation label. Their `is_agent_candidate=true` flag made them eligible;
  the missing labels should be reviewed before describing the pool as belonging
  only to named constellations.
- Runtime measurements are wall-clock measurements from one machine, not a
  controlled performance benchmark with repeated timing trials.
- The smoke runs are engineering checks and are excluded from scientific
  results. They used reduced scenarios and produced too few reference events.

## Metadata still required before paper submission

The repository records the TLE contents and their epochs, but it does not record
the following provenance. Add these items from the teammate who supplied the
files before writing the final dataset subsection:

- TLE provider or database name and URL
- Catalog download/access date and time
- Query or filtering procedure used to obtain the original catalog
- Source and rule used for `object_type` classifications
- Source and rule used for agent eligibility labels
- Source or estimation rule for object radii
- Explanation or correction of the 69 blank constellation labels
- Redistribution or citation requirements for the catalog

These missing provenance fields do not change the computed calibration result,
but they are necessary for a complete and defensible thesis methodology.
