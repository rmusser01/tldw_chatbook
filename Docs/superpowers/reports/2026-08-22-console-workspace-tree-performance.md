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

Frozen-baseline validation result: `1 passed, 1 warning in 2.58s`
(measurement call: 1.54s). The warning was the environment's existing
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

The working-tree source for this capture is branch HEAD
`c9dfaac284bcdc47ee9e362546c10c791fa406cd` plus the uncommitted Task 6 changes;
the commit alone does not contain the complete captured implementation.

Capture command:

```text
../../.venv/bin/python -B -m pytest Tests/UI/test_console_workspace_tree_performance.py::test_new_workspace_tree_benchmark_is_deterministic -q -s
```

Result: `1 passed, 1 warning in 17.27s`; the measured test was the slowest call
at 15.96s. The warning was the environment's existing Requests
dependency-version warning.

### New-path summary

| Dataset | Service records | Ordinary materialized Tree nodes | Reconciles / recomposes | Native node refreshes I / M / S / Sel | Initial median / p95 ms | Marker median / p95 ms | Search apply+clear median / p95 ms | Selection median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 16 | 9 | 100 / 0 | 0 / 20 / 0 / 40 | 47.777938 / 116.413167 | 23.319625 / 42.398209 | 69.209709 / 73.932167 | 23.377479 / 23.968583 |
| Representative | 164 | 57 | 100 / 0 | 0 / 40 / 0 / 40 | 49.450541 / 126.136667 | 23.317833 / 24.467792 | 72.452062 / 74.626583 | 23.644042 / 43.788333 |
| Stress | 3,825 | 840 | 100 / 0 | 0 / 760 / 0 / 40 | 109.209438 / 227.922625 | 29.390938 / 51.514458 | 113.008688 / 258.748417 | 26.553896 / 28.154583 |

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
| Initial projection/mount | 0.524375 | 49.450541 | +9,330.4% |
| 5% marker update | 0.535958 | 23.317833 | +4,250.7% |
| Search apply/clear | 0.955604 | 72.452062 | +7,481.8% |
| Active-row selection | 0.531958 | 23.644042 | +4,344.7% |

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
explicitly accepted with mounted/settled medians of 49.451 ms for initial
projection/mount, 23.318 ms for the 5% marker update, 72.452 ms for search
apply/clear, and 23.644 ms for active-row selection. No claim that either path
is faster is made.

### New-path raw samples (milliseconds)

The arrays below are the 20 measured samples from the mounted/settled capture;
three preceding warmups per dataset were intentionally not reported.

```json
{
  "small": {
    "initial_projection_mount": [46.447333, 45.840125, 47.48475, 46.400708, 47.984291, 116.413167, 54.60475, 37.886542, 35.092666, 65.83, 46.251041, 66.036208, 48.0205, 47.028666, 48.845209, 123.189875, 45.183667, 65.559333, 49.42125, 47.571584],
    "marker_update_5_percent": [23.676708, 23.328333, 23.530583, 24.032167, 23.985292, 43.573417, 23.322333, 42.398209, 23.630333, 23.316917, 23.111667, 22.9665, 22.474209, 22.881625, 23.108041, 23.248834, 23.263583, 22.880583, 23.906083, 22.762],
    "search_apply_clear": [68.475292, 70.77675, 71.145375, 71.53425, 70.837958, 68.578459, 74.584333, 56.204959, 73.932167, 55.413625, 69.240167, 69.17925, 68.475625, 68.086458, 70.784458, 50.090084, 70.392, 68.501542, 68.781459, 69.988],
    "active_row_selection": [23.559041, 23.679041, 23.968583, 23.65, 23.195917, 24.024875, 23.939667, 23.806458, 23.588833, 23.65175, 22.974625, 21.436291, 22.118167, 23.713667, 22.728417, 22.117708, 22.950333, 21.576917, 21.251584, 22.371042]
  },
  "representative": {
    "initial_projection_mount": [59.417583, 126.136667, 34.251834, 53.301667, 68.998958, 52.307833, 49.456833, 48.974625, 47.523458, 50.500375, 52.603167, 48.368, 49.23275, 34.98375, 47.648166, 157.688834, 49.44425, 36.911125, 70.18075, 49.228792],
    "marker_update_5_percent": [24.467792, 42.62675, 23.364875, 22.7235, 22.939667, 22.91, 23.4135, 23.33425, 24.46775, 23.301417, 22.952, 23.44, 23.185209, 23.409416, 23.13375, 23.168208, 23.279791, 23.218708, 23.748125, 24.1065],
    "search_apply_clear": [59.441708, 58.075959, 71.705917, 73.772625, 73.70575, 72.658459, 74.626583, 193.801208, 72.425208, 71.9085, 74.331875, 73.209125, 72.478917, 71.411125, 72.279291, 56.218958, 72.586666, 71.467667, 73.866167, 56.869292],
    "active_row_selection": [23.348625, 23.255792, 24.175125, 23.757375, 43.788333, 23.530709, 21.800708, 24.035042, 23.840833, 23.797666, 22.365666, 22.807208, 23.885334, 23.126125, 23.290959, 22.489083, 24.128167, 24.528875, 23.461625, 44.303791]
  },
  "stress": {
    "initial_projection_mount": [110.127792, 87.589125, 54.33, 132.623333, 219.585667, 94.034625, 216.618166, 117.731375, 110.444167, 109.022875, 106.121375, 227.922625, 92.962, 116.430167, 108.986542, 107.590458, 242.42625, 109.396, 90.396166, 88.110583],
    "marker_update_5_percent": [28.383292, 48.485792, 52.150208, 28.622208, 33.762333, 49.62625, 51.344584, 29.737042, 31.686458, 29.515667, 28.150458, 50.8515, 28.738541, 29.266209, 27.213584, 28.066917, 51.514458, 28.663416, 28.379333, 28.604333],
    "search_apply_clear": [133.973041, 229.561792, 112.59, 121.410375, 128.476583, 147.770583, 98.323083, 103.946042, 253.510167, 112.306542, 113.427375, 90.6655, 116.618458, 261.347333, 104.851417, 103.898333, 109.3005, 90.407583, 258.748417, 110.444208],
    "active_row_selection": [26.882208, 26.017, 28.061375, 26.486791, 28.154583, 27.084, 32.401375, 27.791333, 26.462625, 26.526458, 26.603542, 26.458792, 25.89525, 26.566, 26.541792, 26.473208, 27.16975, 26.358125, 26.230667, 26.6305]
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
