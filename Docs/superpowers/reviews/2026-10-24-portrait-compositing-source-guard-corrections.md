# PR #3244 — required source-check corrections

The completed hosted run on `f4422372` reported two source-check failures in addition
to the receipt test failure: diagnostic inventory drift and worker-contract drift.
The earlier review missed those summary steps; the full summary was then inspected.
The receipt identity correction is recorded separately. No submitted reviews or
inline threads were present when these corrections began; skipped draft bot checks
are not review approval.

ADR required: no
ADR path: N/A — existing ADR-029/150/161/242/243 apply.
Reason: Repair existing UI lifetime guards and review existing diagnostic/worker
contracts without changing storage, provider or runtime boundaries.

## Lifetime corrections

Read all callers of the five flagged functions and the Personas message entry.
Held native crop/preview work, a held avatar mount and consent-time control removal
reproduce stale completion against an owner whose child controls were removed.
A fifth regression holds region Save and removes its validation label.

Capture controls before awaiting and verify their DOM attachment before publishing
rasters, transforms, avatars or sample speech. Textual 8.2.8 keeps `is_mounted`
true after removal; `is_attached` checks the live parent chain. An intermediate
mounted-only guard still published a raster; attachment checks pass that control.
Repeated preview requests return quietly when their target children are gone.
Busy/status/reset and region retirement tolerate missing controls during teardown.
A detached sample control returns before presentation allocation or TTS admission;
a second check after rendering prevents starting speech for a retired preview.
Source, authority, revision, destination consent and speech ownership checks remain.

The Save completion test fails with the original `_pressed` handler (NoMatches on
the removed validation label) and is rerun with its guarded handler. The other four
new cases fail before the production fixes and pass after them. A preliminary
all-module run was interrupted after the mounted-only guards failed; it is not
passing evidence.

## Reviewed worker census

W002 has no new unguarded post-await lookups; its committed census is unchanged.
W003 conservatively joins `obj.x()` by name and positional `type`/`getattr` arguments.
Each of the 35 added rows is the Cartesian product of five portrait wait sites and
seven unrelated dynamic roots: voice retained-attempt validation, Dreams dispatch,
local evaluation/quiz/study notification dispatch, MCP request dispatch and lifecycle
worker-event dispatch. Those roots do not import or call portrait controllers.

The real route is `PersonasScreen._handle_character_performance` →
`run_worker(controller.handle(message), exit_on_error=False)`, then `handle` →
`edit_regions`, `_review_pairs`, `_sample`, its Remove confirmation, or `_generate`
→ `generate_components`. All five wait-site functions execute inside that worker.
The existing region-modal and generation/sample UI tests exercise these paths;
the installed authoring journey opens and dismisses the production dialogs.

Only those 35 new false-positive rows are added, with individual third-column
review notes naming the actual worker entry and evidence. All prior rows and their
notes are preserved; W001/W002/W003 logic and thresholds are unchanged.

## Reviewed persistent diagnostics

The inventory changes from 694 to 696 owner files and 7,742 to 7,752 TASK-494 calls.
Read the statement-level report for all five changed owners before regeneration:
`UI/Persona_Modules/portrait_performance.py` (2), `portrait_regions.py` (1),
`Audio/streaming_sink.py` (5), `UI/Console_Modules/character.py` (1), and `message.py` (1).
All ten additions use fixed literal strings, without interpolated user content,
paths, URLs, provider payloads or exception text. Sink topology is unchanged.
The regenerated inventory records these reviewed statements; no diagnostic sink
or privacy exception is added.

## Qualification limit

TASK-36555 stays In Progress with live image-adapter qualification unchecked.
Configured SwarmUI does not support this app's reference-image path; zero live
image calls, no fallback and `live_qualified=false` remain the recorded result.
The fresh installed journey uses deterministic image/TTS/device fixtures. Earlier
actual Pocket TTS/CoreAudio evidence remains historical, without a new hardware
run or acoustic loopback claim.

## Additional verification findings

The checker/diagnostic run initially returned 437 passed, one skipped and one failed.
Its failing real-tree Console test still expected `on_key` to reach the old Send
helper directly. Current dev uses `send_acknowledgement`; the three relevant
Console methods are AST-identical to dev. Refresh that test to inject a valid
wait-for-dismiss push at the live `on_key` boundary, retaining the other two
Send-entry cases and the actual census/new-row assertions. Synchronous handlers
receive `push_screen(..., wait_for_dismiss=True)`; async handlers retain their
awaited push. No production checker rule changes. This legacy test file has
37 existing Ruff findings against 38 before the change; no new diagnostic.

The first broad UI rerun was interrupted while its close/drain case waited; the
isolated close case passes in 3.39s. It is rerun with the surrounding modules and
a bounded per-test diagnostic timeout, without competing diagnostic AST scans.
All 12 exact required-summary source commands pass before the next rebase.
The fresh wheel-installed authoring/Console/reopen/Remove journey passes in 44.07s.

The final affected UI run passes all 86 cases in 194.94s. All three live-keyboard
checker mutation cases pass in 19.55s. The fresh installed wheel matches every
one of the 53 production files changed from the pre-rebase dev source. Exact
commands, outputs hashes and red/green summaries are retained in
`../qa/portrait-compositing/2026-10-24/source-guard-corrections/`.

The complete worker-checker module then passes all 364 cases in 35.66s.

## Current-dev integration

GitHub stopped registering Actions runs on `1e336401` because newer dev made the
PR conflicting. Rebased onto `139333920d2a87cb011e4f3aa5f8319b59c44a5c`.
Resolved two successive conflicts in media-generation documentation by keeping
dev's new Image Gen worker-retirement section and the final compositing entry
point; retained both sets of testing lessons. No production code conflicted.
All four corrected portrait production modules are byte-identical to the
verified pre-rebase source. Consolidated the duplicate mounted/attachment lesson
with dev's independently recorded TASK-36696 incident, retaining both witnesses.

After rebase, all 12 required-summary source commands pass again. The new
wheel-installed authoring/Console journey passes in 42.98s and matches all 53
changed production files. Fresh API reads show no submitted reviews, inline
comments or review threads. The published head must still complete hosted CI;
this local evidence does not substitute for that required result.
