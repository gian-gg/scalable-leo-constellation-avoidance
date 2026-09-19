# Discrete maneuver contract

The shared actor selects one discrete maneuver ID per decision point. The
maneuver layer converts that ID into a finite Orekit `ConstantThrustManeuver`.

## Action primitives

| ID | Action | RSW unit direction |
| ---: | --- | --- |
| 0 | no-op | `(0, 0, 0)` |
| 1 | prograde | `(0, +1, 0)` |
| 2 | retrograde | `(0, -1, 0)` |
| 3 | radial-out | `(+1, 0, 0)` |
| 4 | radial-in | `(-1, 0, 0)` |
| 5 | cross-track-positive | `(0, 0, +1)` |
| 6 | cross-track-negative | `(0, 0, -1)` |

`R`, `S`, and `W` denote radial, along-track, and cross-track directions.

## Finite-burn model

The action primitive is a fixed **commanded delta-v magnitude**, not a fixed
burn time. For every non-no-op action, the contract uses the spacecraft's
current mass, maximum thrust, and specific impulse to derive the finite burn
duration needed to approximate that delta-v:

\[
m_f = m_0 e^{-\Delta v / (I_{sp}g_0)}
\]

\[
t_{burn} = \frac{m_0-m_f}{F/(I_{sp}g_0)}
\]

The command is infeasible and is rejected when it requires more available
propellant than the spacecraft has or exceeds the configured maximum burn
duration. It is never silently clipped.

## Time model

The **decision interval** and **burn duration** are distinct:

```text
decision interval: how often the policy chooses an action
burn duration:     how long a non-no-op action applies thrust
```

OrbitZoo now accepts a per-spacecraft `maneuver_durations` mapping when calling
`env.step`. A spacecraft can burn briefly and coast through the rest of the
propagation interval.

## Accounting

The maneuver command records its intended delta-v and expected propellant. The
environment will record actual fuel use from pre/post-burn mass. Actual delta-v
is then calculated using the same rocket equation; cumulative actual delta-v is
an evaluation metric, while the per-maneuver commanded delta-v is a tunable
design hyperparameter.

The adopted values are 0.5 m/s per maneuver at up to 7 N; see
[MANEUVER_SIZING_FINDINGS.md](MANEUVER_SIZING_FINDINGS.md).
