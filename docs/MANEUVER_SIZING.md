# Maneuver sizing

`oz size-maneuvers` chooses the delta-v of one discrete maneuver from the 496
reference conjunctions found by the calibration. The adopted result is recorded in
[MANEUVER_SIZING_FINDINGS.md](MANEUVER_SIZING_FINDINGS.md). It is a calculation, not a
simulation: it runs in seconds, then spot-checks a sample in Orekit.

```sh
.venv/bin/oz size-maneuvers --config configs/maneuver_sizing.json --output runs/maneuver_sizing
```

## Method

1. For each reference conjunction, propagate both objects with SGP4 to the time of
   closest approach and shift them to the exact linear closest approach. The
   screened agent is the maneuvering satellite; in satellite-to-satellite
   conjunctions only one satellite maneuvers, which is conservative.
2. Assume `burns` equal burns at consecutive decisions, the first `lead` seconds
   before closest approach. The Clohessy–Wiltshire equations give the agent's
   displacement at closest approach per m/s of delta-v, for each of the six
   directions.
3. Only the displacement across the relative velocity changes the miss distance of
   a fast crossing, so the along-path component is removed.
4. Solve for the smallest per-burn delta-v that moves the miss to the safe
   separation, and keep the cheapest direction.
5. Tabulate the fraction of conjunctions each candidate delta-v resolves, for one
   burn and for every available burn at each lead time.

## Selection rule

Declared before the results: the smallest candidate that resolves at least
`target_fraction` of conjunctions with `selection_burns` burns starting
`selection_lead_seconds` before closest approach. The defaults (95%, 3 burns,
360 s) use the warning the calibration guarantees: at least three decisions
before closest approach.

The minimum thrust for each satellite mass is `mass * delta_v / maximum burn
duration`, so the chosen delta-v can be compared with real thrusters.

## Measured warning time

The calibration only guarantees at least three decisions of warning. The study also
measures each conjunction's actual warning: it replays the maneuvering agent's
k = 1 view on the calibration's 120-second decision grid and records the first
decision at which the threat is its top-ranked neighbour. The agent may then burn
at every remaining decision before closest approach (or only once, reported for
comparison). `detection_leads.csv` lists each warning and requirement, and the
`detected` rows of `sizing_table.csv` and `detected_warning` in
`recommendation.json` give the resulting selection.

## Orekit spot-check

A sample of conjunctions is flown in Orekit (two-body gravity) from `lead` seconds
before closest approach, once coasting and once with the computed burns as finite
constant-thrust maneuvers. The requirement is recomputed from Orekit's own coasting
geometry, and the maneuvered miss must land within `spot_check_tolerance_fraction`
of the safe separation.

## Outputs

| File | Contents |
| --- | --- |
| `sizing_table.csv` | Resolved fraction for every lead time, burn count, and candidate delta-v |
| `requirements.csv` | Per-conjunction required delta-v and best direction |
| `spot_checks.csv` | Orekit coasting and maneuvered miss for each sampled conjunction |
| `recommendation.json` | Rule, selected delta-v, resolved fraction, and minimum thrust by mass |

## Limitations

- Burns are impulsive in the calculation; real finite burns land slightly short
  (about 3% at 1 m/s in the spot-check).
- Relative motion is treated as linear during the encounter, which is least
  accurate for the few slow (below 1 km/s) conjunctions.
- The measured warning assumes the agent reacts at its first detection; a policy
  that waits needs more delta-v per burn.
