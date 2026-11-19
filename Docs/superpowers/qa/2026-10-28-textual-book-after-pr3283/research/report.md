# Research Runs observation audit after PR #3283

Baseline and post-probe HEAD: `7884e11a496a677c8180cc5d67234a57b7d659b9`.
Only diagnostic files beneath this directory were authored. No product source,
tracked tests, tasks, Git state, ACLs, or permission guards were changed.
`source-pins-after.json` records the seven relevant files against HEAD: their
raw checkout hashes differ from Git's blobs because of CRLF, and every file
equals HEAD after CRLF-to-LF normalization. Git diff for those files is empty.

## Verified finding: stale native refresh replaces the current run/source owner

Suggested priority: **P2 / medium**. A native two-second refresh can restore an
old Local run after a real user selects another run or switches to Server.
This produces conflicting selection and detail, and the selected run property
also supplies subsequent run actions. No erroneous lifecycle mutation or live
remote request was exercised, so that consequence is a code-grounded risk.

Current code:

- `tldw_chatbook/UI/Research_Window.py:830-833`: mount starts the native interval.
- `:835-851`: callback checks current Local/nonterminal state, captures run ID,
  then awaits `controller.get_run`.
- `:856-857`: callback publishes the result without rechecking source or current
  selection after the await.
- `:621-622`, `:995-999`: the mounted ListView selection changes the selected run.
- `:449-455`: source switching changes source and clears selected state before
  loading the new catalog.
- `:625-626`, `:649-651`, `:825-828`: actions derive run ID from `selected_run`
  and route with the current source, rather than deriving it from the row.
- `UI/Research_Modules/research_controller.py:38-43`: the real controller awaits
  the scope service and likewise stores its returned selected run.
- `Research_Interop/research_scope_service.py:693-704`: real async detail routing
  reaches the real Local service through its single backend executor.

Controlled qualification uses production `ResearchScreen` and `ResearchWindow`,
production consolidated CSS, real `ResearchController`, real
`ResearchScopeService`, and a file-backed `LocalResearchService`. Two real local
records are created: A running and B draft. The ordinary mounted control presses
Enter on A and then B and observes B selected and rendered normally.

In the held-selection case, the unmodified native interval starts the real
`scope.get_run(A, mode=local)` SQLite read. A probe wrapper awaits that original
method and holds **only its completed return**. While held, a genuine Enter key
on the mounted B ListView row changes `selected_run` and detail to B. Releasing
the return inside the App lifetime yields:

| Point | Source | selected_run | ListView highlight | Detail |
| --- | --- | --- | --- | --- |
| Held A response | Local | A | A | LOCAL A ALPHA |
| Actual B key, before release | Local | B | B | LOCAL B BRAVO |
| Released response | Local | **A** | **B** | **LOCAL A ALPHA** |

In the held-source case, the real source Select is driven by Enter/Down/Enter
while the same A response is held. Before release the actual source is Server,
selection is cleared, and the catalog reports unavailable Server. Releasing
the Local response inside the App lifetime restores selected A while source
remains Server and the run list has no highlighted child. This is a real
cross-source publication error; it is not evidence of a remote fallback call.

The initially considered hypothesis that message-pump serialization would
prevent this is refuted. The installed Textual native timer invokes callbacks
from its own timer task (`textual/timer.py:189-191`), permitting the actual
mounted ListView and source-change events to complete during the hold.

## Independently verified adjacent defect: source switch leaves old detail

The ordinary control selects B, switches Local to unavailable Server, waits
for the actual handler to settle, and observes `source=server`,
`selected_run=None`, no highlighted row, and `Server research backend is
unavailable.` The detail still shows **LOCAL B BRAVO**, its draft status, phase,
control, and checkpoint. This requires no delayed response. `switch_source`
clears the data selection and payloads but never resets `#research-run-detail`.

This is suitable for the same bounded view-ownership task, or a separate small
P2 presentation task if the parent prefers splitting the two publication sites.
The delayed-response finding is stronger because it actually changes the action
target, whereas this ordinary defect leaves the action target absent.

## Existing task and ADR ownership screening

Used the parent's global full-body catalog (407 refs / 365 distinct trees /
8,482 task bodies / 56 worktrees / 184 local records), not filenames alone.
`catalog-hits.json` retains 15 matching task-body identities with paths, tree
provenance and relevant excerpts; `catalog-bodies.json` retains their complete
bodies. No local-only matching owner was found by this Research-specific
semantic screen. Closest current owners were read in full:

- **TASK-16486 (Done), Auto-refresh the selected research run in the window**:
  owns installing this interval, nonterminal/Local guards and payload retention.
  Its AC and immediate-call tests do not cover a selection/source change while
  the real return is in flight. Closest prior implementation, not an open owner.
- **TASK-21127 (Done), Research runs - per-op leaked connects and loop-side
  periodic reads/writes while a run is active**: owns real service thread
  offload, transactions and connection lifetime. It does not qualify UI owner
  publication after the async boundary. Earlier catalog bodies contain its
  In Progress and To Do history, not a separate current owner.
- **TASK-21507, Add Research shell authority and responsive Workspace
  foundation**: current file has contradictory duplicate Done/To Do metadata
  and duplicate AC blocks, so status cannot be inferred from a first YAML key.
  Full body explicitly owns qualified **Research Workspace** catalogs/overlays
  and preserves independent Runs state. Its stale Local/Server ABA checks are
  in `research_workspace`, not this separately routed `research` window.
- **TASK-21511 (To Do), Add Research Copy sharing and Deep Research return**:
  stale results concern qualified origin workspace import/return receipts.
  It does not own the existing run browser's timer or ordinary source switch.
- **TASK-19300 (In Progress), Make research runs durable, resumable jobs**:
  owns phase resume, scheduling, leases and completion delivery; no run-selection
  or source-switch UI outcome. Historical TASK-18060 matches are its documented
  renumbering rather than a new owner.
- **TASK-31795 (Done)** owns missing Runs layout CSS; **TASK-16483 (Done)** owns
  structured bundle/artifact rendering; neither covers current run ownership.
- **TASK-2306 (Done), Run selection populates the run detail region** was read
  in full because of its tempting title. It owns **Watchlists RunsPane**, not
  Research Runs. It provides a useful existing stale-result pattern, not scope
  for the current Research issue.
- **TASK-16814 (Done)** fixes engine-worker status dispatch, not the timer's
  selected-owner publication. The other catalog hits concern budget controls,
  initial engine wiring, eval/CI work, or historical versions of the above.

Relevant accepted decisions: ADR-033 keeps destination domain/view state with
its screen owner; ADR-068 makes this window the real run operator under route
`research`; ADR-070 owns execution leases rather than selected-view authority;
ADR-078 distinguishes the Workspace route `research_workspace` and the real
Runs route `research`, preserving separate screen owners. Its Workspace
late-result contract is a useful design precedent but must not be falsely
reported as already implemented or tested for Runs.

No new ADR appears necessary for a routine bug fix within these accepted
screen and service boundaries. No storage, service, or runtime owner needs to
change merely to enforce the selected view's lifetime.

## Suggested task outcomes

Title: **Keep Research Runs refresh tied to its current source and selection**

- A mounted real user selection of B remains selected, highlighted and rendered
  after a previously started refresh of A completes; run payload state remains
  associated with the selected run.
- A Local refresh completing after a Server switch cannot restore a Local
  selection/detail or retarget the next run action.
- Clearing selection for a source switch immediately presents the empty-run
  detail state, including unavailable/empty catalog outcomes; a later selected
  run populates the detail normally.
- The normal nonterminal Local refresh still updates current progress and
  preserves current payload state; terminal and Server selections retain their
  existing skip behavior.
- Targeted mounted cases hold the actual real scope/service return, prove the
  selection/source transition happened before release, observe final state and
  release every hold before leaving the App lifetime.

## Receipts, limits and retained unsuccessful attempt

Launcher: `.test-tmp/textual-37031-37034/run-targeted-external-profile.py`,
using the project Python, native private profile/TEMP, unchanged 180 s per-test
timeout and guards. No full suite was run. Reproduction selector:
`.test-tmp/textual-after3283/research/test_runs_observation.py`.

- `attempt1/`: original exact source, ordinary and held-source observations,
  three actual private child XML/logs and outer XML/log. **1 passed / 2 failed**
  in outer 159.17 s. Held selection waited for intended B after release and
  timed out without a final snapshot; held source failed the intended
  `selected is None` assertion with actual A. These original receipts remain.
- `attempt2/`: observation oracle records/asserts current defective outcomes
  rather than waiting for intended behavior. **3 passed** in outer 106.12 s.
  This means successful defect qualification, **not fixed product tests**.
  Actual child case durations: ordinary 11.700 s, held selection 13.359 s,
  held source 15.698 s. Outer XML SHA256:
  `a0e57bc72e7c1eb6af645a89670d98157170e0d45c1551751766c47c8753d14a`.
  Child XML hashes, exact source, observations, SVG captures and raw logs are
  copied under `attempt2/`; `manifest.json` records every retained byte hash.
- The actual Server-unavailable path was exercised; a live server or HTTP
  endpoint was not used. The Local engine/provider was not started. No actual
  cancel/pause write, remote-ID collision, source ABA, unmount result race,
  physical terminal, throughput claim, or performance budget was qualified.
- Pydantic deprecation and pytest-asyncio fixture-scope warnings remain in the
  raw logs. Optional sounddevice/WebRTC availability notices are setup output,
  not causes of either selection defect. An unprivileged read of the private
  run log was refused and was retried read-only with escalation. A first source
  hash helper used Windows backslashes for safe.directory and was refused;
  corrected slash spelling succeeded. Neither was product evidence or changed
  any global Git setting.

Book anchors (user-provided PDF extraction): physical pages 145-146 and 158
describe per-widget message queues and awaited handler behavior; that principle
requires checking whether a real operation is on that pump. Physical page 335
shows guarding background publication by producer lifetime. The finding applies
that ownership check to the existing timer rather than proposing a new shared
state framework. Recomposition examples at physical 179-180/189 were considered
but are not the cause of this run-selection corruption.
