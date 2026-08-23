---
id: TASK-21112
title: >-
  Notes-sync runtime - gate the unconditional start, bound the legacy scan, and
  back off the 1Hz full-tree watcher
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 11:24'
labels:
  - performance
  - notes-sync
  - startup
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21112).

The lasting notes-sync runtime starts unconditionally at app.py:10297-10303
(`cutover_admitted=True` hardcoded at app.py:5986). Zero-profile boots still create the state
DB, run >=3 schema-censused transactions, and the first boot runs two unbounded SELECTs over
chachanotes.db (notes_sync_legacy.py:603-628). With >=1 active root, the watcher performs a
full recursive stat walk of every root every 1 second forever
(notes_sync_watcher.py:18,74-77; discovery bounds 10k entries - an over-bounds root pays the
full scan every tick before bailing). Library already falls back to `InertLastingSyncRuntime`
(library_screen.py:3212-3214).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `start()` is gated on actual configuration (non-empty root summaries, plus the legacy `notes.sync_directory` key for one-time migration); a zero-profile boot creates no notes-sync state DB and runs no legacy scans
- [x] #2 The watcher backs off when consecutive polls see no change (1 s -> 5-15 s with jitter, or native FS events with polling fallback); the interval is configurable
- [x] #3 The legacy first-boot SELECTs are bounded/paginated
- [x] #4 Existing notes-sync tests stay green; a probe with an active 5k-file root shows the reduced steady-state stat traffic
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: run Tests/Notes notes-sync suites + ProductionApp lifecycle + library sync UI tests on base 8e949873e (teed).\n2. Gate: add an injectable start_evidence callable to NotesSyncRuntimeOwner/build_notes_sync_runtime_owner; when it reports no configuration, _start_once defers inert (new 'not_configured' status) without touching the store; app.py wires it as legacy notes.sync_directory key presence OR state-DB file presence (Path.exists probe - never opens/creates the DB).\n3. Live bring-up: start(force=True) re-arms a deferred start; review_setup forces the start so activating the first root at runtime creates the machinery on demand; Library controller treats 'not_configured' as setup-available.\n4. Watcher backoff: PollingNotesSyncWatcher doubles its sleep on consecutive no-change polls up to a jittered max (default 1s -> ~5-15s), resets to base on any detected change; base+max configurable via [notes] keys read in config.py and passed through the builder.\n5. Legacy scan: LIMIT-bound the two first-boot SELECTs in notes_sync_legacy.py with an explicit overflow error instead of unbounded fetchall.\n6. Tests red-first: zero-profile boot creates no state DB; legacy-key gate starts; deferred owner unit contracts; watcher backoff shape with fake clock/sleep; legacy bound; then full Tests/Notes + touched suites green; A/B pre-existing reds vs base.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gated the lasting notes-sync runtime's boot start on actual configuration, added idle backoff to the polling watcher, and bounded the legacy first-boot SELECTs.

**Start gate (AC 1).** NotesSyncRuntimeOwner takes an optional start_evidence callable (None = start always, so every existing owner/test contract is unchanged). When it reports False, _start_once defers inert with the new not_configured/none projection before any store call: no state DB, no schema census, no legacy scan; shutdown on a deferred owner touches nothing. app.py wires the probe as legacy_sync_directory_configured(app_config) or state_db_path.exists(). Reading root summaries would itself CREATE the DB (store.transaction opens/creates on first use), so the probe is deliberately side-effect-free: key presence + Path.exists(), evaluated via asyncio.to_thread. Probe errors fail OPEN (a broken probe must never silently disable a configured user's sync). Trade-off: profiles carrying an empty state DB from the formerly-unconditional starts keep starting (cheap: marker read + empty inventories; no watcher without leases) - deleting user DBs was out of scope.

**Runtime activation.** start(force=True) re-arms exactly one full start after a deferral (concurrent-caller-safe re-arm loop); review_setup calls it first, so choosing "Keep a folder synced" live-starts the machinery - no restart required. LibraryNotesSyncController treats not_configured as setup-available (_SETUP_READY_STATUSES). With no roots, no other runtime entry point is reachable while deferred; root-targeted calls still raise the cutover error.

**Watcher backoff (AC 2).** PollingNotesSyncWatcher doubles its sleep after every no-change poll up to max_interval_seconds (default 10 s), jittered x uniform(0.5-1.5) -> the 5-15 s band; any detected change resets to the base interval (default 1 s). Emission eligibility still uses the base interval so a post-idle change is emitted immediately. Configurable via [notes] sync_watcher_interval_seconds / sync_watcher_max_interval_seconds (config.py get_notes_sync_watcher_intervals, validated, emitted in the config template) and forwarded through build_notes_sync_runtime_owner. FSEvents/watchdog deliberately not attempted (future work per task scope).

**Legacy scan bound (AC 3).** The two first-boot SELECTs in snapshot_legacy_notes_sync carry LIMIT (LEGACY_SNAPSHOT_EVIDENCE_LIMIT = 10,000, +1 overflow sentinel); exceeding it raises LegacyNotesSyncSnapshotError("legacy_snapshot_overflow") - refusing loudly instead of silently migrating a truncated (wrong) subset.

**Probe (AC 4).** Real 5,000-file root, production adapter scan (including its over-bounds bail path: 5k exceeds discovery max_files=1000, so each tick pays the walk then bails - exactly the finding): one scan ~17.8 ms; 60 virtual seconds of quiet steady state: 60 scans/min (old fixed 1 s) -> 8 scans/min (backoff), ~1067 -> ~142 ms/min stat traffic, 7.5x fewer scans.

**Tests.** Red-first per unit: 4 owner-gate contracts (cutover suite), 5 watcher backoff/validation, 3 legacy bounds, 2 controller availability, 2 app-level pins (zero-profile boot creates NO state DB through shutdown; legacy notes.sync_directory key still boots the one-time migration), plus config-helper, legacy-presence-semantics, and builder-forwarding tests. Full Tests/Notes: 2871 passed / 5 skipped / 0 failed. Updated 2 existing tests: the lasting-flow source pin (availability expression) and the mounted-migration-failure test (pre-creates the store so the gate admits the start it exercises). Full collect-only sweep: 56,552 collected; 5 collection errors (3 Confluence, 1 TTS chatterbox, 1 library_file_notes_workspace) reproduce identically on base.

**Pre-existing reds A/B'd vs base 8e949873e (identical both sides, not this change):** 15 failures across Tests/UI/test_library_shell.py + Tests/ProductionApp (incl. the surrogate-pattern meta-guard - AST comparison proves my edits add zero new violations); 7 failures in Tests/UI/test_library_notes_files_sync_journey.py; test_app_fences_console_then_drains_buddy_before_profile_teardown (same AttributeError on the skeletal fixture; not fixed by this change); preflight's production-diagnostic-inventory drift (FTS-backfill rows). Three extra failures appeared only while running the base and branch 24-minute suites CONCURRENTLY (ChatScreen compose race under load, one victim being a file that runs before mine); all 16 tests in those three files pass sequentially, and the lifecycle file passed 3/3 repeat runs.

**Files.** tldw_chatbook/Notes/notes_sync_runtime.py, notes_sync_watcher.py, notes_sync_legacy.py, tldw_chatbook/app.py, config.py, UI/Library_Modules/library_notes_sync_controller.py; tests in Tests/Notes/ (cutover/runtime/watcher/legacy), Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py, Tests/UI/Library_Modules/test_library_notes_sync_controller.py, Tests/UI/test_library_notes_lasting_sync_flow.py. Lessons entry added to backlog/docs/lessons-testing-evidence.md (a configured-probe that reads the store creates the store; recurs for TASK-21105).
<!-- SECTION:NOTES:END -->
