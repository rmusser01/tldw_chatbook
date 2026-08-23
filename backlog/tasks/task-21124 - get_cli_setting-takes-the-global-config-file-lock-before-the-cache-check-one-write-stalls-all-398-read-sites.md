---
id: TASK-21124
title: >-
  get_cli_setting takes the global config file lock before the cache check - one
  write stalls all 398 read sites
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 09:21'
labels:
  - performance
  - config
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21124).

Reads: `config.py:6057 -> :4976 -> :5107` acquires `_config_file_lock()` BEFORE the cache
short-circuit (which sits inside `_load_cli_config_bootstrap_unlocked:4872-4877`). Writes hold
that lock through 2 fsyncs (temp fd + parent dir, `Utils/private_paths.py:660,691`), 3 full
TOML parses, and a settings rebuild. With 398 `get_cli_setting` call sites - many on the event
loop - any concurrent config write (even correctly off-loop ones) stalls loop-side reads for
the whole write. Amplifiers verified: Logs filter chip = 2 rewrites/4 fsyncs per click
(UI/Logs_Window.py:273-276), theme switch (app.py:872), lab rail toggle, and a per-keystroke
writer (UI/Dictation_Window_Improved.py:602 -> dictation_service_lazy.py:1383).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cache-hit reads never take the file lock (double-checked fast path on the existing _CONFIG_GENERATION; writers already bump it)
- [x] #2 The write path is coalesced to one parse (verify re-parse dropped or debug-gated); the Logs-chip and dictation writers are debounced
- [x] #3 A two-thread probe (writer loop vs timed reader) shows reader p95 unaffected by concurrent writes; before/after numbers in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline the config test files (Tests/test_config_*.py, Tests/Utils config tests, Tests/UI/test_dictation_settings_debounce.py, Tests/UI/test_ux_batch7.py) with teed numbers on base 41a240ccd.
2. Red-first lock-counter test: cache-hit get_cli_setting must not call _config_file_lock (red against current always-lock behavior).
3. Fast path: double-checked generation-sandwich cache check in _load_cli_config_bootstrap before taking the file lock; document the writer-ordering invariant (cache reference swapped BEFORE _CONFIG_GENERATION bump, both under the file lock; GIL makes each reference read atomic; sandwich guards the cache/source pair). Preserve the existing no-copy mutable-dict return semantics exactly.
4. Write path to one parse: _write_raw_cli_config_unlocked returns its TASK-13157 verify-parse dict; _publish_runtime_config_unlocked(raw_config=...) installs the bootstrap cache from it (merge defaults + decrypt) instead of re-reading the file, and load_settings gains reload_bootstrap=False so the settings rebuild reuses the just-primed bootstrap cache. Verify guard kept at full strength (it becomes THE single parse). Fall back to the full locked reload if the raw install cannot decrypt.
5. Debounce hot writers: Logs filter chips -> one batched save_settings_to_cli_config write behind a task-15470-style debounce + worker + unmount flush; dictation service set_buffer_duration drops its duplicate synchronous save_setting_to_cli_config (the owning widget already persists the same key through its task-15470 debounce with unmount flush).
6. Tests: lock-counter fast-path test, two-thread torn/stale-read correctness test, informational timing probe (p95 printed, generous assert), Logs debounce + unmount-flush tests, dictation service no-sync-write test; update Tests/UI/test_ux_batch7.py saved-filter roundtrip to the batched API.
7. Full --collect-only sweep; A/B any red against base 41a240ccd; record probe numbers in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three-part fix, all on branch fix/task-21124-wave2 off dev 41a240ccd.

READ FAST PATH (config.py `_load_cli_config_bootstrap`): a warm cache hit
returns without touching `_config_file_lock()`. Invariant (documented at the
fast path): (1) each global read is a GIL-atomic reference load; (2) every
publication installs a brand-new dict (deepcopy+merge), never re-publishes an
old object, so the `_CONFIG_CACHE is cached_config` identity re-check proves
no install landed between the reads — the (cache, source) pair cannot be torn
across two installs; (3) writers store cache=None, source=None, ..., cache=new,
source=path under the lock; (4) `_CONFIG_GENERATION` bumps only AFTER the
cache swap (`_publish_runtime_config_unlocked`), so the AC's generation
sandwich (read gen, read cache/source, re-read gen) means a hit is the
currently-published config or a mid-flight writer's own fresh view — always
one complete never-mutated-in-place dict. The plain generation sandwich alone
is NOT sufficient for the two-variable read (a concurrent path-switch reload
mutates the pair without bumping) — the identity re-check is what closes it;
both checks are in place. The fast path returns the same shared mutable dict
the locked hit always returned (no copy semantics change). Misses and
force_reload serialize through the lock exactly as before.

WRITE PATH, 4 parses -> 2: before: RMW read (disk) + TASK-13157 verify
parse-back (string) + publish's bootstrap re-read (disk) + settings rebuild's
second bootstrap re-read (disk). Now `_write_raw_cli_config_unlocked` RETURNS
its verify parse-back and every writer hands it to
`_publish_runtime_config_unlocked(raw_config=...)`, which installs the
bootstrap cache from it (`_install_bootstrap_cache_from_raw`: merge defaults +
strict decrypt + cache/source/failure-flag stores identical to the bootstrap
success tail; decrypt failure falls back to the full locked reload, matching
historical failure modes) and rebuilds settings with the new
`load_settings(..., reload_bootstrap=False)` so the rebuild reuses the primed
cache. Remaining parses per write: the inherent RMW read + the verify parse.
Deviation from AC#2's parenthetical: the verify re-parse was NOT dropped or
debug-gated — it is now THE parse that feeds publish (publishing the
parsed-back view is strictly more faithful than the input mapping), so the
TASK-13157 "cannot leave behind a file its own next read cannot parse"
guarantee is preserved at full strength while the redundant parses are gone.
Atomic temp+fsync+rename write, interprocess flock, generation-bump-last
ordering, and all failure phases unchanged. All 8 writer sites wired
(apply_settings_mutation, replace_cli_config[_serialized], revisioned,
shutdown persist, enable/disable/change encryption).

HOT WRITERS: (1) Logs level chips (UI/Logs_Window.py) — a click used to run
two sequential synchronous `save_setting_to_cli_config` rewrites (4 fsyncs)
on the event loop, and on_unmount rewrote the file unconditionally on every
exit. Now: task-15470 debounce shape (0.6 s timer -> exclusive worker ->
`asyncio.to_thread` -> ONE batched `save_settings_to_cli_config`), baseline
compare against the last-persisted state (mount restore and click-away-and-
back write nothing), unmount stops the timer, awaits an in-flight worker, and
flushes only a real change. A `_filter_text` mirror backs the snapshot — the
teardown DOM query degrades to "" and clobbered the saved filter with an
unconditional flush (caught live by test_saved_filter_roundtrip during
development). (2) Dictation per-keystroke writer
(Audio/dictation_service_lazy.py `set_buffer_duration`) — deviation from the
AC's "debounce": the synchronous `save_setting_to_cli_config` there was a
DUPLICATE of the owning widget's already-debounced task-15470 snapshot (same
key, `Dictation_Window_Improved._settings_snapshot` includes
buffer_duration_ms, with unmount flush); the service write also fired on every
service init (write-on-open). Removed the service write instead of stacking a
second debounced writer on the same key; the keystroke path now has exactly
one debounced writer.

PROBE EVIDENCE (Tests/test_config_read_fastpath_task21124.py; base = clean
41a240ccd worktree, same venv): lock counter — 100 file-lock acquisitions per
100 warm-cache reads before, asserted exactly 0 after (the red-first test).
Write parse counter — 3 disk + 1 string parses per write before, asserted
1 + 1 after. Reader latency vs a 5-write burst (overlap enforced): solo
p50/p95 5.5/5.7 us; concurrent p50/p95 5.4/5.7 us both before AND after —
percentiles hide the stall class (few-but-huge samples; lesson recorded in
lessons-testing-evidence.md) — the honest stall signal: BASE max 18,210 us,
5 stalls >1 ms (one whole-write block per write, fsyncs included); FIXED max
3,681 us, 13 short stalls (only the invalidate->republish tail can block a
reader; the fsync phase no longer stalls anyone). AC#3's "p95 unaffected"
holds (5.7 us == 5.7 us), and the worst single event-loop stall dropped 5x.

TESTS: new Tests/test_config_read_fastpath_task21124.py (7),
Tests/UI/test_logs_filter_persist_debounce.py (4),
Tests/Audio/test_dictation_buffer_duration_persistence.py (2). Updated:
Tests/UI/test_ux_batch7.py (roundtrip -> batched API),
Tests/test_config_delete_settings.py (a `load_settings` stub gained the new
keyword). Green post-change: 216 core config (12 files incl. the api-key
precedence contract Tests/Utils/test_config_api_key_resolution.py), 22
writer/debounce, 858 write-path consumers (wizards/providers/TTS prefs/
profile isolation), 352 batch-2 consumers, full batch-B (TTS/UI/Audio).
Full --collect-only sweep: 55,389 collected / 29 errors on branch vs 55,376 /
29 on base — error lists byte-identical (numpy-missing optional-dep files +
the settings_screen->RAG_Search circular-import family), +13 = exactly the
new tests. Pre-existing reds A/B'd at base (identical failures):
settings_screen import family (provider_switch_atomic, context_memory,
qwencloud_api_mode, image_gen_panel, speech_tts_panel,
library_file_notes_workspace), tools_settings_window dangerous-backup-path
test, console_provider_gateway 4 errors + kimi_zai 1 failure + mlx_lm 4
failures, and the batch-B Audio set — branch 9 failed / 991 passed / 27
errors vs base 9 failed / 989 passed / 27 errors, failure lists
byte-identical (all in the OLD non-lazy test_dictation_service.py and
test_recording_vad_preroll.py; the +2 passes are this task's new tests).

DELIBERATELY NOT FIXED (out of scope, same family):
`update_privacy_settings` (dictation service) still does 3 sequential writes
per privacy toggle and 3 on service init — it persists a `save_history` key
the widget snapshot does NOT carry, so removing it needs its own task to
avoid orphaning that persistence. `set_encryption_password` clears the cache
without the file lock (pre-existing; benign for the fast path — readers just
miss). Logs filter TEXT remains unmount-persisted-only (unchanged scope,
now change-gated). The unused `Optional` import in Logs_Window.py predates
this task.

CRUX DEMO (isolated env): with a writer FROZEN mid-write while holding the
config write lock, `get_cli_setting` returned the cached value in 20.4 us;
on base the same read blocks for the writer's full hold. New-test flake
check: 3x repeats of the fastpath+logs files (11 tests) and the
dictation/roundtrip set (8 tests), all green.
<!-- SECTION:NOTES:END -->
