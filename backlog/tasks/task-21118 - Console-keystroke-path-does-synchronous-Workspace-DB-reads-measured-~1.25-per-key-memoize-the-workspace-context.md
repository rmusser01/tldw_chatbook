---
id: TASK-21118
title: >-
  Console keystroke path does synchronous Workspace-DB reads - measured ~1.25
  per key - memoize the workspace context
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 12:22'
labels:
  - performance
  - console
  - database
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21118).

Live counter: 20 printable keys in the configured composer -> 25 x `ensure_default_workspace` +
25 x `get_active_workspace` (LocalWorkspaceRegistryService), i.e. ~1.25 synchronous SQLite
round-trips per keystroke on the UI thread (chain: DraftChanged ->
`_build_console_control_state` -> `_current_console_workspace_context`). 62 us/call measured on
a warm fast SSD - the risk cases are cold page cache, slow disks, and the repair branch's
DELETE write. During staged live-work launches, `EvidenceBundle.from_payload` is additionally
re-parsed >=2x per keystroke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The workspace context is memoized on the screen and invalidated by workspace-change events (activation, registry mutation); the keystroke path performs zero DB round-trips (counter-probe verified)
- [x] #2 ensure_default_workspace's repair side-effects move to session-start/workspace-switch; the keystroke path is read-only
- [x] #3 The staged-launch evidence bundle is parsed once per launch and cached on the launch object
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline probe: mounted configured Console, wrap LocalWorkspaceRegistryService.ensure_default_workspace/get_active_workspace with counters, drive 20 printable keys, tee counts (expect ~25+25 on base 30c7e1fe9).
2. Red-first: commit the probe as a test asserting 0 registry calls per 20 keys (red on base) plus a staleness control (workspace switch mid-session must reflect in the next built state) and an EvidenceBundle.from_payload parse-count test.
3. Registry service: add an in-memory mutation_generation counter bumped by every workspace_records mutator (create/rename/archive/unarchive/set_active/clear_active/_restore_default). This is the invalidation subscription point: every activation/registry-mutation seam (Console switcher, browser row open, session switch, Settings 'Set active', Library create modal, archive flows) funnels through these mutators on the one app-level service.
4. ConsoleWorkspaceController: split _current_console_workspace_context into a memoized read-only _resolve_console_active_workspace_id (memo keyed on (service identity, mutation_generation); int-typed generation required so MagicMock doubles bypass the memo; None active floors to DEFAULT_WORKSPACE_ID in memory, matching ensure's returned id) + the staged-sources leg. Keystroke path stops calling ensure_default_workspace entirely (read-only AC2); repair side-effects remain at session-start (app-wiring ensure) and workspace-switch seams (_set_active_workspace_for_console_session global branch, _console_browser_workspace_records, archive_workspace).
5. Evidence bundle: cache the parse in evidence_bundle_from_launch on the launch object's __dict__ (frozen dataclass, no slots), keyed by payload-mapping identity so a payload change invalidates; all 6 parse consumers share the seam.
6. Verify compose with the task-15452 per-pass memo (scope memo keys untouched; new memo is cross-pass and lives on the controller).
7. Tests: new probe file + Tests/Workspaces + workspace/registry test files + Console keystroke/composer suites (test_console_draft_sync_equality_gate, test_console_composer_draft_changed, test_console_rail_search_debounce, test_console_native_chat_flow, test_console_workspace_controller, test_console_workspace_reconcile) + full --collect-only sweep; A/B reds against base.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped the three-part fix; all ACs verified by counter probes red on base 30c7e1fe9 and green after.

**Approach.**
1. Memo (AC1): `ConsoleWorkspaceController._resolve_console_active_workspace_id` (UI/Console_Modules/workspace.py) serves the per-keystroke active-workspace resolution from a per-screen memo `(service identity, mutation_generation, workspace_id)`. Invalidation is a generation counter on `LocalWorkspaceRegistryService` (`mutation_generation`), bumped by every workspace-record mutator: create_workspace, rename_workspace, archive_workspace, unarchive_workspace, set_active_workspace, clear_active_workspace, `_restore_default_workspace` (ensure_default_workspace bumps through those legs only when it changed something). Invalidation-event census: every seam that changes the active workspace - Console switcher `_switch_to`, browser-row open `_activate_console_workspace_for_browser_row`, session switch `_set_active_workspace_for_console_session`, Settings "Set active", Library create-modal, archive flows, boot ensure - funnels through those mutators on the one app-level service instance, so generation-compare == subscribing to all of them, including cross-screen mutations a screen-event subscription would miss. The generation must be a real int before anything is cached (MagicMock doubles' auto-attribute compares equal to itself forever - lesson TASK-21103); doubles without it stay on live reads. Composes with the task-15452 per-pass derivation memo: that memo dedupes provider-selection legs within one pass; the pass's single remaining context read is what this cross-pass memo serves.
2. Read-only keystroke path (AC2): the context read now calls `get_active_workspace` only, flooring a missing active workspace to DEFAULT_WORKSPACE_ID in memory (the id ensure returned for that state) without writing. The repair (`_delete_default_runtime_bindings`: probing SELECT + DELETE write) relocated to the switch seam - `set_active_workspace(DEFAULT_WORKSPACE_ID)` now performs it, covering ensure's own activate leg and every direct switch-to-Default; ensure's active-is-Default branch keeps its repair for the session-start seams (app wiring boot ensure, Console session-switch global branch, browser records, archive). ensure_default_workspace callers census: app.py `_wire_workspace_registry_services` (boot), workspace.py `_set_active_workspace_for_console_session` + `_console_browser_workspace_records`, registry_service.py archive_workspace - all retained; the keystroke caller is the only one converted to read-only.
3. Evidence bundle (AC3): `evidence_bundle_from_launch` (Chat/console_display_state.py) - the single seam all 6 staged-state consumers use - caches the parse on the launch object (`object.__setattr__` into the frozen dataclass's `__dict__`), keyed by identity of the payload's `evidence_bundle` mapping, so a replaced payload re-parses and a failed parse is not retried per key.

**Probe numbers** (mounted configured Console, Tests/UI/test_console_keystroke_workspace_reads.py, teed):
- 20 printable keys, before: 25 x ensure_default_workspace + 25 x get_active_workspace (exact match with the review's live counter); after: 0 + 0, and 0 WorkspaceDB connection/transaction entries of any kind.
- 3 keys with a staged launch, before: 11 x EvidenceBundle.from_payload; after: <=1.
- Staleness guard: a cross-screen `set_active_workspace` performed directly on the registry (no Console seam) reflects in the next keystroke's context read.

**Tests.** New: Tests/UI/test_console_keystroke_workspace_reads.py (8: probe, wired-counter control, staleness, 4 parse-cache units, mounted staged-launch parse gate); Tests/Workspaces/test_workspace_registry_service.py +3 (switch-seam repair pin, generation-bumps-on-every-mutator, generation-stable-across-reads). Updated: test_console_native_chat_flow.py `test_console_provider_selection_restores_default_workspace_when_none_active` -> `..._floors_missing_active_workspace_read_only` (it pinned the ensure-on-read write AC2 removes; now pins read-only floor + heal-at-seam). Runs, A/B'd vs base 30c7e1fe9: probe file 5-red->8-pass; Tests/Workspaces 372+40 passed (10 create-modal fails identical on base); Console UI batches 129+402+100 passed (6 scope-row fails + 1 workbench-contract fail + 16 errors identical on base); Chat suites 794 passed (3 tiktoken/network-guard teardown errors identical on base); display-state+settings 397 passed; consumers 67 passed (18 context-modal fails identical on base); --collect-only 56,586 collected, 5 collection errors identical on base.

**Deviations / deliberately not fixed:** direct SQL writes to the workspace DB bypassing the service (and other processes) do not bump the generation - documented out of scope, same as every in-process cache over this DB. The pre-existing reds above (create-modal clicks, scope-row clicks, context-modal, workbench-contract rail order, tiktoken downloads, 5 collection errors) are untouched.
<!-- SECTION:NOTES:END -->
