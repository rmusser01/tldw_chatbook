# Console workspace Tree performance evidence

Frozen baseline capture: 2026-08-22

Mounted new-path capture: 2026-08-23

Task: TASK-20937.2 baseline and TASK-20937.6 new-path verification

Phase: immutable old-projection baseline plus mounted/settled new-path capture

## Protocol

The deterministic fixture matrix uses seed `20937`: small = 3 named workspaces × 4 conversations + 4 Default/unassigned records; representative = 12 × 12 + 20; stress = 50 × 75 + 75. The active workspace and respectively 0, 2, or 9 additional workspaces are expanded. Titles produce the approved 25%, 10%, and 10% search hit ratios.

Each operation received three unreported warmups followed by 20 measured iterations using `time.perf_counter()`: initial projection, a run-marker replacement affecting `ceil(5%)` of service records, search apply plus clear, and active-row selection. The old projection performed one full reconcile/recompose for initial, marker, and selection operations and two for search apply/clear, for 100 reconciles and 100 recomposes per dataset across the measured iterations.

Environment: Python 3.12.11, Textual 8.2.8, macOS 15.6 arm64 (`macOS-15.6-arm64-arm-64bit`), terminal 180 × 52 cells, source `5729439e5ad4fe0959b59a1fe699ef9ee3ebb2f8`. Fixture SHA-256: `8b3a04b4af657e6419a4fb0d72df83c501acad75a5ba2b9abe97655ffecc177c`. Frozen JSON SHA-256: `140db572a9284b4cb6871483eab0ed720a2f2b417fb6a3d3ed08e1f26c909f34`.

Capture command:

```text
../../.venv/bin/python -B -m pytest Tests/UI/test_console_workspace_tree_performance.py::test_old_projection_baseline_is_reproducible -q -s
```

Frozen-baseline validation result: `1 passed, 1 warning in 3.06s`
(slowest setup: 1.06s). The warning was the environment's existing
Requests dependency-version warning. The frozen summary and raw samples follow
the new-path comparison section below.

## New workspace Tree path (mounted/settled capture)

TASK-20937.6 repeated the same fixture generator and timing protocol against
the pure workspace projection plus the native Tree adapter in a mounted Textual
harness at exactly 180 × 52 cells. Each timed operation waits for deferred
refresh, layout, and paint work to settle, then forces compositor strip
rendering. The initial sample builds the projection and mounts its keyed Tree
nodes. Marker and selection samples update the mounted Tree. Search measures
apply and clear as two logical updates.

Outside search, the projection materializes conversation children only for the
active workspace and respectively 0, 2, or 9 additional expanded workspaces.
Each expanded workspace uses the production 75-row page limit, with `Load
more` only when additional children exist. The stress fixture has exactly 75
children per workspace, so those branches have no `Load more` node. Full-scope
matches are materialized only while search is active.
The benchmark records the observed reconcile, native node-refresh, and
recompose counters; its CI assertions remain deterministic and do not impose a
wall-clock threshold.

Environment: Python 3.12.11, Textual 8.2.8, macOS 15.6 arm64
(`macOS-15.6-arm64-arm-64bit`), terminal 180 × 52 cells. The frozen baseline
still validates source `5729439e5ad4fe0959b59a1fe699ef9ee3ebb2f8`, fixture
SHA-256 `8b3a04b4af657e6419a4fb0d72df83c501acad75a5ba2b9abe97655ffecc177c`,
and JSON SHA-256
`140db572a9284b4cb6871483eab0ed720a2f2b417fb6a3d3ed08e1f26c909f34`.

The source for this capture is clean branch HEAD
`25be0541956745896b9e86a3d07bdbc5a5948c04`, a descendant of the merged
TASK-20937 implementation commit `a581f28e0`. The subsequent closeout commits
change only tests and evidence records, not the benchmark harness or measured
product path, so this remains the source-equivalent mounted capture.

Capture command:

```text
../../.venv/bin/python -B -m pytest Tests/UI/test_console_workspace_tree_performance.py::test_new_workspace_tree_benchmark_is_deterministic -q -s
```

Result: `1 passed, 1 warning in 17.11s`; the measured test was the slowest call
at 14.42s. The warning was the environment's existing Requests
dependency-version warning.

### New-path summary

| Dataset | Service records | Ordinary materialized Tree nodes | Reconciles / recomposes | Native node refreshes I / M / S / Sel | Initial median / p95 ms | Marker median / p95 ms | Search apply+clear median / p95 ms | Selection median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 16 | 9 | 100 / 0 | 0 / 20 / 0 / 40 | 47.313167 / 113.149250 | 22.332459 / 22.866916 | 67.205062 / 67.860000 | 21.999354 / 22.735417 |
| Representative | 164 | 57 | 100 / 0 | 0 / 40 / 0 / 40 | 48.020354 / 111.726291 | 22.668396 / 23.051042 | 70.492230 / 71.849750 | 22.166125 / 22.741875 |
| Stress | 3,825 | 840 | 100 / 0 | 0 / 760 / 0 / 40 | 102.561104 / 213.959750 | 27.032916 / 47.017875 | 84.449458 / 208.919625 | 25.780792 / 26.179667 |

The instrumented 100 reconciles per dataset are five calls per measured
iteration: one initial, one marker, search apply and clear, and one selection.
Every individual logical update therefore reconciled exactly once in this
capture. The native node-refresh totals in operation order initial / marker /
search / selection are small 0 / 20 / 0 / 40, representative 0 / 40 / 0 / 40,
and stress 0 / 760 / 0 / 40. The instrumented recompose total is zero for every
dataset.

### Representative comparison and investigation

| Operation | Frozen projection-only median ms | Mounted/settled new median ms | Diagnostic change |
| --- | ---: | ---: | ---: |
| Initial projection/mount | 0.524375 | 48.020354 | +9,057.6% |
| 5% marker update | 0.535958 | 22.668396 | +4,129.5% |
| Search apply/clear | 0.955604 | 70.492230 | +7,276.7% |
| Active-row selection | 0.531958 | 22.166125 | +4,066.9% |

Every representative operation exceeds the 20% investigation threshold. The
mounted/settled new path is deliberately not like-for-like with the frozen
projection-only baseline: it mounts a Textual application, settles deferred
refresh/layout/paint, forces compositor strip rendering, maintains native Tree
nodes, and temporarily materializes full-scope matches during search. The old
baseline measures projection and row materialization without that mounted
rendering lifecycle. Consequently, the percentage comparisons are diagnostic
only and cannot support a relative speed claim.

The investigation also confirms the ordinary new projection is bounded to
expanded-workspace children rather than materializing all 3,825 stress records,
and that the observed update work uses native node refreshes without Textual
recomposition. Under AC3, the representative diagnostic differences are
explicitly accepted with mounted/settled medians of 48.020 ms for initial
projection/mount, 22.668 ms for the 5% marker update, 70.492 ms for search
apply/clear, and 22.166 ms for active-row selection. No claim that either path
is faster is made.

### New-path raw samples (milliseconds)

The arrays below are the 20 measured samples from the mounted/settled capture;
three preceding warmups per dataset were intentionally not reported.

```json
{
  "small": {
    "initial_projection_mount": [64.66525, 45.749209, 64.55475, 45.396917, 113.14925, 46.946333, 64.506333, 47.362125, 42.896375, 45.567458, 64.45, 46.591291, 64.30625, 46.066666, 114.275334, 46.202625, 64.363083, 45.896417, 64.039292, 47.264208],
    "marker_update_5_percent": [21.262542, 22.343708, 22.304833, 22.397833, 22.313625, 22.490875, 22.94475, 22.08375, 22.327125, 21.471583, 22.506083, 22.429584, 22.276542, 22.514416, 22.329792, 21.834584, 22.452792, 22.335125, 22.313542, 22.866916],
    "search_apply_clear": [66.811459, 67.729458, 67.061, 67.545417, 48.6285, 67.302833, 67.181959, 67.831875, 49.5275, 67.19775, 67.86, 67.274666, 67.165083, 66.617375, 48.914834, 67.212375, 68.764375, 66.908209, 67.850958, 67.480291],
    "active_row_selection": [21.231417, 22.637334, 21.160083, 22.5185, 22.041334, 22.735417, 21.319166, 22.011083, 21.987625, 22.452, 21.393833, 21.973958, 20.980125, 22.338459, 21.369333, 22.657875, 21.138083, 22.750125, 21.389166, 22.142584]
  },
  "representative": {
    "initial_projection_mount": [65.124958, 47.64275, 48.063917, 47.383375, 48.657459, 64.382375, 46.879042, 125.352083, 65.593041, 47.231583, 46.838333, 49.74, 45.763167, 65.177375, 47.536667, 111.726291, 47.976791, 65.38225, 47.939917, 47.539334],
    "marker_update_5_percent": [22.681125, 22.796625, 22.948792, 22.043542, 22.380334, 22.575042, 22.535584, 21.926334, 22.952333, 22.228875, 22.632334, 22.877416, 22.811083, 23.051042, 22.585209, 21.570792, 22.848459, 22.88775, 23.08625, 22.655666],
    "search_apply_clear": [71.350834, 71.84975, 71.884375, 70.258292, 69.423083, 69.177291, 69.9655, 69.550292, 70.40875, 71.20975, 70.608625, 68.8355, 68.592375, 71.793042, 70.746625, 53.09775, 70.575709, 70.336166, 70.983333, 70.675625],
    "active_row_selection": [22.309333, 21.633209, 21.396708, 22.145833, 21.9785, 22.268583, 22.245459, 22.741875, 22.085625, 21.932625, 22.4235, 22.944417, 22.186417, 22.301291, 21.396, 22.110125, 22.690708, 22.494042, 21.866583, 22.035959]
  },
  "stress": {
    "initial_projection_mount": [48.207959, 82.633459, 48.234042, 103.145084, 161.934042, 103.031333, 102.874333, 102.628375, 102.493833, 213.95975, 82.075667, 84.960666, 49.753125, 104.249583, 82.565708, 104.304375, 83.727333, 103.017625, 85.368792, 220.519167],
    "marker_update_5_percent": [45.655958, 25.497542, 46.5895, 27.015041, 27.95925, 26.552833, 26.756958, 26.889833, 26.6045, 26.942375, 26.225125, 156.271375, 47.017875, 27.759333, 27.344125, 27.050791, 25.773625, 27.488291, 26.2185, 29.926792],
    "search_apply_clear": [79.976375, 184.336875, 83.733292, 82.555791, 81.326958, 96.420292, 191.596583, 96.543542, 82.820208, 82.286917, 83.081458, 83.394542, 85.376708, 234.214458, 84.904042, 98.059083, 208.919625, 85.2195, 83.359958, 83.994875],
    "active_row_selection": [25.595834, 24.79775, 25.027125, 25.26525, 25.439125, 26.179667, 25.532917, 26.241584, 25.3425, 25.054792, 26.062542, 26.026333, 25.997208, 26.050334, 26.135333, 25.893709, 25.811625, 25.999875, 25.749959, 25.420292]
  }
}
```

## Frozen old-projection summary

| Dataset | Service records | Materialized rows | Initial median / p95 ms | Marker median / p95 ms | Search apply+clear median / p95 ms | Selection median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 16 | 10 | 0.063209 / 0.065416 | 0.063625 / 0.064375 | 0.118146 / 0.153250 | 0.064167 / 0.074500 |
| Representative | 164 | 60 | 0.524375 / 0.545542 | 0.535958 / 0.570667 | 0.955604 / 1.088333 | 0.531958 / 0.575625 |
| Stress | 3,825 | 144 | 11.740625 / 12.137500 | 11.534563 / 13.153750 | 20.628667 / 22.516292 | 11.673000 / 13.326708 |

These values are a report-only old-projection baseline. TASK-20937.2 makes no speed claim.

## Frozen old-projection raw samples (milliseconds)

The following arrays are copied verbatim from the capture output. The machine-readable source of truth is `Tests/UI/fixtures/console_workspace_tree_old_baseline.json`; its reproducibility test checks the whole-file checksum, fixture checksum, metadata, counts, sample cardinality, median, and p95 without invoking changed projection code.

```json
{
  "small": {
    "initial_projection": [0.065792, 0.064083, 0.065416, 0.063958, 0.064083, 0.06325, 0.062916, 0.063333, 0.062834, 0.062791, 0.063167, 0.063333, 0.063167, 0.062958, 0.063375, 0.062834, 0.06375, 0.063083, 0.063042, 0.062959],
    "marker_update_5_percent": [0.064084, 0.064375, 0.063666, 0.063333, 0.063584, 0.063709, 0.063792, 0.063542, 0.063625, 0.063625, 0.063416, 0.0635, 0.063334, 0.06325, 0.063375, 0.063625, 0.063292, 0.064166, 0.065459, 0.064167],
    "search_apply_clear": [0.15325, 0.118709, 0.116583, 0.115959, 0.118459, 0.115833, 0.115958, 0.115958, 0.11575, 0.11575, 0.115875, 0.146667, 0.124666, 0.129042, 0.1155, 0.117834, 0.1245, 0.121875, 0.20725, 0.139458],
    "active_row_selection": [0.066417, 0.065834, 0.064708, 0.0745, 0.066625, 0.057125, 0.07125, 0.0665, 0.063458, 0.063625, 0.06325, 0.079584, 0.060625, 0.059584, 0.0635, 0.070084, 0.068208, 0.063417, 0.0635, 0.0635]
  },
  "representative": {
    "initial_projection": [0.545542, 0.5295, 0.577291, 0.544584, 0.539875, 0.517459, 0.517833, 0.51825, 0.517083, 0.518875, 0.517, 0.536667, 0.535541, 0.524625, 0.530083, 0.524, 0.524125, 0.523916, 0.523875, 0.525041],
    "marker_update_5_percent": [0.525833, 0.526958, 0.526375, 0.552875, 0.536125, 0.548208, 0.754792, 0.569667, 0.550583, 0.570667, 0.549084, 0.536292, 0.535333, 0.535625, 0.536083, 0.535833, 0.5355, 0.533833, 0.535208, 0.534083],
    "search_apply_clear": [0.99775, 0.955041, 0.947625, 0.946, 0.943375, 0.967, 0.952708, 0.94875, 1.039833, 1.012959, 0.956167, 1.045125, 0.944917, 0.945292, 0.94225, 0.95075, 1.005, 1.482458, 1.088333, 0.985292],
    "active_row_selection": [0.548375, 0.563916, 0.537917, 0.54475, 0.525583, 0.524625, 0.527167, 0.5255, 0.525542, 0.587791, 0.533125, 0.575625, 0.538167, 0.530042, 0.532041, 0.530375, 0.531875, 0.552125, 0.527208, 0.525042]
  },
  "stress": {
    "initial_projection": [11.991416, 11.987708, 11.974459, 12.1375, 11.780792, 11.515209, 11.745375, 11.428958, 11.783417, 11.616458, 11.956208, 11.671458, 11.388666, 11.735875, 53.100875, 11.763167, 11.669583, 11.456125, 11.320042, 11.473208],
    "marker_update_5_percent": [11.407125, 11.398125, 12.094, 12.076625, 13.15375, 13.177709, 12.014125, 11.945833, 11.489291, 11.454459, 11.737792, 11.757666, 12.048959, 11.558292, 11.362416, 11.316875, 11.312125, 11.154875, 11.471375, 11.510834],
    "search_apply_clear": [20.535125, 20.39875, 20.210083, 20.142584, 20.411833, 21.377208, 64.370959, 20.768041, 20.450875, 20.601333, 20.356958, 21.259041, 21.336292, 22.516292, 21.167167, 20.637042, 21.029708, 20.409375, 20.735042, 20.620292],
    "active_row_selection": [11.589167, 11.215292, 11.478458, 11.864958, 11.526708, 11.698875, 11.733458, 11.656834, 11.549167, 11.371875, 11.443917, 11.658292, 50.707167, 11.687708, 11.816, 11.851042, 11.737625, 11.632667, 11.738542, 13.326708]
  }
}
```
