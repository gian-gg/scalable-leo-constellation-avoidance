# Thesis implementation

This directory contains the implementation for **Scalable Collision Avoidance
in Large Low Earth Orbit Constellations**. It is built on the local OrbitZoo
fork, but thesis-specific components live under `src/orbitzoo/thesis/` so the
generic orbital-dynamics library remains reusable.

## Architecture

The planned system uses discrete MAPPO under Centralized Training with
Decentralized Execution (CTDE):

```text
local k-neighbor observation -> shared actor -> one maneuver per satellite
full training-only constellation state -> centralized critic -> learning signal
```

The actor will use seven actions: no-op, prograde, retrograde, radial-out,
radial-in, cross-track positive, and cross-track negative. The critic is used
only while training; the deployed/evaluated policy uses the shared actor and
each satellite's local observation only.

## Repository layout

```text
src/orbitzoo/                 reusable OrbitZoo code
src/orbitzoo/thesis/          thesis-specific source code
configs/                      versioned JSON experiment configurations
tests/                        automated tests
runs/                         generated metrics, metadata, and TensorBoard logs (ignored)
checkpoints/                  generated model files (ignored)
```

## Reproducible runs

Every run must save:

- `config.json`: the exact experiment configuration;
- `environment_info.json`: device, platform, Python, and PyTorch details;
- `metrics.csv` and TensorBoard data when training is added;
- model checkpoints when MAPPO is added.

`orbitzoo.thesis.runtime.select_device()` selects CUDA when available, then
Apple Metal (MPS), otherwise CPU. The choice is recorded for each run.

The initial configuration is [configs/mappo_toy.json](../configs/mappo_toy.json).
It fixes the architectural defaults—not final experimental values—to 16 agents,
four local neighbors, seven discrete actions, and a 300-second decision interval.
Its maneuver values are provisional development defaults and will be selected by
bounded sensitivity analysis before final experiments.

## Current status

The discrete MAPPO implementation is validated independently of orbital physics.
See [MAPPO.md](MAPPO.md) for its CTDE design and rollout API. The deterministic
[toy environment](TOY_ENVIRONMENT.md) validates end-to-end shared-policy learning
before the policy is connected to orbital simulation.

The [maneuver contract](MANEUVER_CONTRACT.md) defines the discrete action IDs,
their RSW thrust directions, finite-burn execution, and delta-v accounting.

The [collision-avoidance environment](COLLISION_AVOIDANCE_ENVIRONMENT.md)
connects that contract to OrbitZoo propagation, deterministic conjunction
screening, rewards, episode termination, and diagnostics.

The actor now receives fixed-width, threat-ranked local observations containing
`k` relative-neighbour blocks with explicit padding masks. The critic receives
a separate, deterministic full-system training state. This makes the deployed
actor input independent of constellation population size.

The versioned [calibration configuration](K_DT_CALIBRATION.md) defines the
catalog inputs, propagation window, candidate values, deterministic seeds, and
passing thresholds that will be used to select `k` and the decision interval
before MAPPO training. Its strict catalog loader now validates two-line and
three-line TLE input, records NORAD IDs and UTC epochs, applies the configured
freshness cutoff, and joins optional object metadata. Validated calibration data
models provide SI-unit Cartesian frames, deterministic agent selections,
ID-based conjunctions and neighbor rankings, combination metrics, and final
recommendations without coupling reference results to the runtime safety model.
The [propagation layer](K_DT_CALIBRATION.md#two-resolution-propagation) runs a
60-second pass whose spatial index contains the full catalog but is queried only
from selected-agent positions. It excludes self and catalog-only pairs,
deduplicates agent-agent pairs, merges conservative candidate encounter windows,
and produces 10-second SGP4 TEME states only for the involved pairs. This avoids
both all-pairs Python screening and a full day of 10-second catalog states while
preserving SI units and deterministic ordering.

Agent population selection now operates on the post-altitude-filter payload
pool. Explicit PCG64 permutations make the 16-, 64-, and 256-agent populations
nested and reproducible for every calibration and validation seed, with the
selected NORAD IDs stored in a validated versioned manifest.
The largest nested populations across all seeds can be screened once, and each
smaller population reuses the resulting windows through an ID-based filter.
Fine pair trajectories are converted into versioned reference conjunction
truth using bounded TCA refinement between 10-second samples. Coarse false
positives are removed at the configured safe separation, collision status is
derived from catalog radii, overlapping detections are deduplicated, and nested
agent selections reuse the shared truth through deterministic ID filtering.
