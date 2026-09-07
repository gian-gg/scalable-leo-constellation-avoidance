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
| `EncounterWindow` | Merged coarse candidate interval for one object pair |
| `FineEncounterTrajectory` | Fine pair states inside one encounter window |
| `AgentSelection` | Deterministic agent population for one seed |
| `AgentSelectionManifest` | Versioned calibration and validation selections |
| `ReferenceConjunction` | Propagated closest approach for one canonical ID pair |
| `ReferenceConjunctionManifest` | Versioned fine-pass conjunction truth |
| `RankedNeighbor` | One agent-neighbor threat rank at a decision epoch |
| `RankedNeighborFrame` | Reusable top ranks for all screened agents at one epoch |
| `ThreatDetection` | First visibility and decision lead time for one reference event |
| `CombinationMetrics` | Counts and runtime for one population, seed, `k`, and delta-t |
| `PooledCombinationMetrics` | Count-weighted pass result for one split and pair |
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

`propagate_catalog` is the low-level propagation primitive. It initializes SGP4
from each accepted TLE and creates a repeatable, streaming `SGP4Propagation` at
an explicit timestep. Its timeline includes both the common start epoch and the
configured end epoch. A direct one-day, ten-second full-catalog stream would
contain 8,641 frames, so calibration uses the two-resolution layer below.

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

## Two-resolution propagation

`build_two_resolution_propagation` performs the production calibration pass:

1. It propagates every altitude-filtered object at `coarse_step_seconds`.
2. At each coarse frame, a `cKDTree` indexes every catalog object, but
   `query_ball_point` is called only for explicitly selected agent positions.
   Catalog-only pairs are therefore never generated or filtered after the fact.
3. Self-pairs are excluded. Pairs between two selected agents are converted to
   canonical NORAD-ID order and deduplicated, while payloads, rocket bodies,
   debris, and unknown objects all remain possible threats to an agent.
4. Candidate intervals receive `fine_window_padding_seconds` on both sides and
   overlapping intervals for the same canonical NORAD pair are merged.
5. Only the two objects in each merged interval are propagated at
   `reference_step_seconds`; each fine trajectory is yielded independently.

The conservative coarse query radius is:

```text
safe_separation_meters
+ maximum_relative_speed_mps * coarse_step_seconds
```

This includes pairs that could cross the safety boundary between coarse samples,
not only pairs already close at a sample. Nearby pairs then pass through a
linear relative-motion check. Its acceptance threshold adds a conservative
two-body curvature margin derived from the configured minimum altitude, Earth
gravity, and the coarse interval. This prevents parallel objects within the
large spatial-query radius from creating unnecessary fine windows.

Every coarse frame checks that twice the largest observed object speed remains
below the configured relative-speed bound; calibration stops if the bound is too
small. With the default values, the full catalog has 1,441 coarse frames instead
of 8,641 fine frames. Fine work then depends only on the number and duration of
candidate encounter windows.

`screen_agent_selections` takes the saved selection manifest and screens the
union of the largest nested population for every calibration and validation
seed. The resulting `CandidateScreeningResult` stores that exact agent set.
Smaller populations call `windows_for(selection)` to filter the saved windows,
so the catalog does not need to be propagated and spatially queried again for
the 16- and 64-agent prefixes. This path does not use the environment's Python
all-pairs conjunction implementation.

## Reference conjunction generation

`generate_reference_conjunctions` converts the pair-only fine trajectories into
the reference events used to measure neighborhood recall. Within every
fine-resolution interval it computes a bounded linear closest-approach time from
the relative TEME position and velocity. This captures a closest approach that
falls between the 10-second SGP4 samples instead of rounding it to a frame.

Coarse candidates whose refined miss distance exceeds
`safe_separation_meters` are discarded. Retained records contain canonical
NORAD IDs, UTC TCA, miss distance, relative speed, and the combined metadata
radii in SI units. `ReferenceConjunction.is_collision` is derived by comparing
miss distance with the combined radii. Overlapping windows for the same pair
form one event, while non-overlapping later encounters remain separate.

The versioned `ReferenceConjunctionManifest` records the catalog epoch,
reference timestep, safety threshold, and screened agent IDs. Its
`conjunctions_for(selection)` method associates the shared reference pass with
each nested population. The manifest can be saved and loaded as deterministic
JSON for the later `(k, delta t)` sweep.

## Neighbor ranking at decision epochs

`build_decision_schedule` creates a timeline for every candidate decision
interval. Decision epochs include the propagation start and exclude the terminal
epoch, where no further action can be taken. Their union is propagated once, so
the 120-, 300-, and 600-second candidates reuse matching epochs from the
60-second timeline.

At each shared epoch, `rank_neighbors_at_epoch` compares only selected agents
against catalog objects. It never creates satellite-debris or debris-debris
pairs that cannot appear in an actor observation. Agent batches use NumPy
broadcasting for relative position and velocity, closest-approach time, miss
distance, combined radius, collision status, and unsafe status. This avoids the
environment's full Python all-pairs loop while retaining every payload, rocket
body, debris object, and unknown object as a possible neighbor.

The ranking order matches the actor observation semantics: current collision,
predicted unsafe encounter, smaller predicted miss distance, earlier TCA, and
finally NORAD ID. Self-pairs are masked before sorting. Only the largest
configured `k` is retained per agent; smaller candidate values use
`RankedNeighborFrame.rankings_for(selection, k)`. Frames are streamed so a
full-day catalog run does not retain every decision and agent in memory.

## Joint k/delta-t evaluation

`evaluate_joint_combinations` consumes the ranking-frame iterator once. At each
shared epoch it activates only the candidate decision-interval schedules that
contain that timestamp. A neighbor at rank `r` updates every configured
neighborhood prefix where `k >= r`, so no ranking or orbital propagation is
repeated for an individual `(k, delta t)` pair.

A visible neighbor is matched to a canonical reference pair only when its TCA
is still in the future and within the configured screening horizon. If a pair
has more than one reference event, the predicted absolute TCA selects the
closest eligible event. Either direction of an agent-agent ranking can detect
the shared event, but the event is emitted only once per selection and
combination.

Each `ThreatDetection` records the reference event, first visible decision
epoch, and number of actionable decisions before TCA. Undetected events use a
null visibility epoch and zero remaining decisions. For a detected event the
count is `ceil((TCA - first_visible) / delta_t)`, and timely status is derived
from the configured minimum. The evaluator verifies that every shared epoch is
present exactly once and that ranking frames retain the largest candidate `k`.

## Metrics and passing thresholds

`aggregate_detection_metrics` groups the event evidence by split, seed, agent
count, `k`, and `delta t`. It emits the complete configured grid, including
explicit zero-count rows for samples with no reference conjunctions. It also
verifies that every combination within a sample contains the same reference
events, preventing missing evaluator output from being mistaken for perfect
detection or an empty scenario.

Each `CombinationMetrics` row derives missed events, threat recall, and timely
detection fraction from integer counts. Optional per-case runtimes can be
attached without changing the detection results. `pool_combination_metrics`
then sums counts across seeds and agent populations before calculating rates;
it never averages percentages from differently sized samples.

`PooledCombinationMetrics.passed` applies the versioned minimum recall and
timely-detection thresholds separately to calibration and validation. A pooled
case with no reference events reports both rates as zero and cannot pass, even
if thresholds were configured as zero. Final parameter selection remains a
separate phase so validation evidence cannot influence the calibration search.

## Deterministic agent selection

Agent populations are selected only after TLE freshness and start-epoch altitude
filtering. The eligible pool contains every propagated object whose metadata has
`is_agent_candidate=true`; `constellation` remains descriptive and does not add
another eligibility filter.

`select_agent_populations` sorts eligible NORAD IDs before randomization, then
uses an explicit NumPy `PCG64` generator for each configured seed. Each seed
creates one permutation and all population sizes are prefixes of it:

```text
16 agents  = permutation[:16]
64 agents  = permutation[:64]
256 agents = permutation[:256]
```

This makes the smaller scenario an exact ordered subset of every larger one.
The same catalog epoch, configuration, and seed reproduce the same IDs even if
the source TLE order changes. Selection stops before calibration if the
post-filter pool cannot satisfy the largest configured population.

`save_agent_selections` writes a deterministic, schema-versioned JSON manifest
containing the catalog epoch, sorted eligible IDs, RNG algorithm, and every
calibration and validation selection. `load_agent_selections` validates the
saved IDs, disjoint seed groups, consistent population sizes, and nesting before
returning the manifest.

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
    aggregate_and_pool_detections,
    build_decision_schedule,
    build_two_resolution_propagation,
    evaluate_joint_combinations,
    generate_reference_conjunctions,
    iter_ranked_neighbor_frames,
    load_catalog,
    save_agent_selections,
    save_reference_conjunctions,
    select_agent_populations,
)

path = "configs/k_dt_calibration.json"
config = CalibrationConfig.load(path)
catalog = load_catalog(config, path)
propagation = build_two_resolution_propagation(catalog, config)
selections = select_agent_populations(propagation.coarse_propagation, config)
save_agent_selections(selections, "runs/calibration/agent_selections.json")
screening = propagation.screen_agent_selections(selections)

for selection in selections.calibration_selections:
    windows = screening.windows_for(selection)
    print(selection.seed, selection.requested_agent_count, len(windows))

references = generate_reference_conjunctions(
    propagation.iter_fine_trajectories(screening.encounter_windows),
    catalog,
    config,
    screened_agent_norad_ids=screening.screened_agent_norad_ids,
)
save_reference_conjunctions(
    references,
    "runs/calibration/reference_conjunctions.json",
)

for conjunction in references.conjunctions:
    print(conjunction.tca_epoch_utc, conjunction.miss_distance_meters)

schedule = build_decision_schedule(
    propagation.coarse_propagation.start_epoch_utc,
    config,
)
ranking_frames = iter_ranked_neighbor_frames(
    propagation.coarse_propagation,
    config,
    agent_norad_ids=screening.screened_agent_norad_ids,
)
detections = evaluate_joint_combinations(
    ranking_frames,
    references,
    selections,
    config,
    schedule=schedule,
)
for detection in detections:
    print(detection.detected, detection.decisions_remaining)

metrics, pooled_results = aggregate_and_pool_detections(
    detections,
    config,
)
for result in pooled_results:
    print(
        result.evaluation_split.value,
        result.neighborhood_size,
        result.decision_interval_seconds,
        result.threat_recall,
        result.timely_detection_fraction,
        result.passed,
    )
```

Saving a configuration validates it and emits deterministic, sorted JSON. The
schema version is currently `1`; unsupported versions are rejected rather than
silently interpreted.
