# Neighborhood and decision-interval calibration

Before MAPPO training, an offline TLE calibration will select the neighborhood
size `k` and decision interval `delta t`. The calibration uses propagated orbit
states and conjunction timing only; it does not execute maneuvers or train a
policy.

## Versioned configuration

The initial configuration is
[`configs/k_dt_calibration.json`](../configs/k_dt_calibration.json). It defines:

- paths to one frozen TLE catalog and its object metadata;
- the propagation time model and LEO altitude filter;
- conjunction-screening thresholds;
- agent counts, candidate values, and deterministic seeds;
- the passing thresholds used to choose a parameter pair.

Catalog paths are resolved relative to the configuration file, not the process
working directory. The checked-in paths therefore resolve to files under
`data/tle/`. Configuration validation checks the schema and internal
consistency; `load_catalog` reports missing or unreadable input files.

## TLE catalog input

The loader accepts ordinary two-line records and three-line records with a name
above the element lines. Blank lines are ignored, and a leading `0 ` on a name
line is removed. Each element line must be exactly 69 ASCII characters with a
valid checksum. The two lines must contain the same catalog ID. Both numeric
NORAD IDs and the Space-Track Alpha-5 representation are supported.

Loading fails on malformed element fields, duplicate NORAD IDs, duplicate
metadata IDs, or metadata rows for IDs absent from the TLE file. Every accepted
epoch is returned as a timezone-aware UTC value. Records more than
`maximum_tle_age_days` older than the latest epoch in that frozen catalog are
excluded; a record exactly on the cutoff is retained. The original TLE order is
preserved. LEO altitude filtering is intentionally deferred until propagation,
when an actual state vector exists.

Metadata is UTF-8 CSV. These columns are required:

- `norad_id`: positive decoded NORAD integer;
- `object_type`: `payload`, `rocket_body`, `debris`, or `unknown` (case,
  spaces, and hyphens are normalized);
- `is_agent_candidate`: exactly `true` or `false`, case-insensitive.

The optional columns are `name`, `radius_meters`, and `constellation`; other
named columns are ignored. Radius values must be finite and positive. Only a
payload may be an agent candidate. A TLE without metadata is retained as an
unknown, non-agent object with `default_radius_meters`. Display names use
metadata first, then the TLE name, then `NORAD-<id>`.

## Calibration data records

The public records in `orbitzoo.thesis.calibration` define the boundaries
between catalog loading, propagation, selection, screening, measurement, and
recommendation:

| Record | Purpose |
|---|---|
| `CatalogObject` | Validated TLE and joined metadata for one NORAD ID |
| `CartesianStateFrame` | All retained Cartesian states at one UTC epoch |
| `AgentSelection` | Deterministic agent population for one seed |
| `ReferenceConjunction` | Propagated closest approach for one canonical ID pair |
| `RankedNeighbor` | One agent-neighbor threat rank at a decision epoch |
| `CombinationMetrics` | Counts and runtime for one population, seed, `k`, and delta-t |
| `CalibrationRecommendation` | Selected pair, thresholds, pass status, and supporting metrics |

Cartesian frames store positions in metres and velocities in metres per second
as finite, read-only `float64` arrays with shape `(object_count, 3)`. Calibration
records are never normalized; conversion to normalized `float32` values happens
only when observations are encoded for the policy. All record field names carry
their units (`_m`, `_meters`, `_mps`, or `_seconds`), and every epoch is
normalized to UTC.

NORAD IDs, rather than display names, connect records. Conjunction IDs must be
in ascending order so a pair has only one representation. Recall and timely
detection fractions are derived from integer counts, and a sample containing no
reference conjunctions receives zero rather than a vacuous passing score.
Recommendation pass status is derived by pooling the integer event counts across
its metric rows, so it cannot contradict the stored evidence. Timely counts are
produced using the recommendation's `minimum_decisions_before_tca` threshold.
Ranked-neighbor records are designed to be streamed into metric accumulation;
persisting every decision is optional diagnostic output rather than a pipeline
requirement.

## Propagation start

`start_epoch_mode` is `latest_tle_epoch`. The catalog loader exposes the latest
epoch among all validated TLE records as the common propagation start, including
the epoch used to establish the freshness cutoff. The exact resolved UTC
timestamp must be stored with the run outputs. This keeps the checked-in
configuration usable with different frozen catalog snapshots without making the
start time ambiguous within a run.

## SGP4 propagation

`propagate_catalog` initializes SGP4 from each accepted TLE and creates a
repeatable, streaming `SGP4Propagation`. Its timeline includes both the common
start epoch and the configured end epoch, with frames separated by
`reference_step_seconds`. The default one-day, ten-second configuration therefore
produces 8,641 frames.

SGP4 produces True Equator Mean Equinox (TEME) positions in kilometres and
velocities in kilometres per second. Each emitted `CartesianStateFrame` converts
these values to `float64` metres and metres per second. Any nonzero SGP4 status
fails the run with the affected NORAD ID, UTC epoch, and decoded error reason;
partial frames are never emitted.

The configured altitude limits are applied once at the common start epoch using
geocentric distance minus the WGS-72 equatorial radius. Filtering once keeps the
same ordered NORAD IDs and array shape in every frame. The excluded IDs remain
available through `altitude_filtered_norad_ids`, and propagation fails if the
filter removes the entire catalog.

Frames are calculated in bounded batches and yielded one at a time. This avoids
holding the full `(time, object, state)` trajectory in memory when calibrating
large catalogs. Iterating the propagation object again deterministically reruns
the same trajectory. This phase performs no maneuvers.

## Default sweep

The first calibration evaluates agent populations of 16, 64, and 256; neighbor
counts of 1, 2, 4, 8, and 16; and decision intervals of 60, 120, 300, and 600
seconds. Seeds 0 through 9 are reserved for selection, while seeds 100 through
104 are held out for validation.

The selected pair must achieve at least 99.9% threat recall. At least 99% of
reference threats must be detected with three or more decision opportunities
remaining before time of closest approach. Among passing pairs, the later
calibration runner will choose the smallest `k` and then the largest decision
interval.

## Python API

From the repository root, validate and summarize the configured real-data
catalog with one command:

```sh
oz catalog
```

Use `oz catalog --config path/to/calibration.json --limit 20` for another
configuration or a longer object preview. Set `--limit 0` for summary-only
output. The command exits unsuccessfully and prints the input location when
validation fails.

The equivalent Python API is:

```python
from orbitzoo.thesis.calibration import (
    CalibrationConfig,
    load_catalog,
    propagate_catalog,
)

path = "configs/k_dt_calibration.json"
config = CalibrationConfig.load(path)
catalog = load_catalog(config, path)
propagation = propagate_catalog(catalog, config)

for frame in propagation:
    print(frame.epoch_utc, frame.positions_m.shape)
```

Saving a configuration validates it and emits deterministic, sorted JSON. The
schema version is currently `1`; unsupported versions are rejected rather than
silently interpreted.
