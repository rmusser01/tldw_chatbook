---
id: TASK-19190
title: Remove the orphaned play_current_audio pair (app wrapper + stts_events handler)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20'
labels:
  - dead-code
  - stts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third-wave burn-down residue (sibling of merged TASK-19043, which removed the
`export_current_audio` pair the same way). At dev `7877defba`,
`tldw_chatbook/app.py:11409` defines `async def play_current_audio` — it lazily
initializes the S/TT/S handler and awaits
`Event_Handlers/STTS_Events/stts_events.py:2767`'s `play_current_audio`
handler. A whole-tree grep (production + `Tests/`) finds exactly three hits:
the wrapper def, the wrapper's internal call (`app.py:11416`), and the handler
def. Zero callers of the wrapper anywhere; the handler's only caller is the
wrapper; zero test references. TASK-19043's reviewer independently confirmed
the orphan with a grep that included dynamic-dispatch shapes
(`getattr`/string-built names).

Use merged TASK-19043 as the template, including its security-coverage-map
discipline: before retiring anything, check whether the handler performs
validation that needs a live-path equivalent (here the handler only does a
path-existence check on `_current_playground_audio_path()` and notifies on
failure — no unique validation identified, and there are no tests to retire —
but the check must be recorded, not assumed). Two knock-ons the removal must
chase: (1) the handler contains one `logger.error` call, so the persistent
diagnostic inventory's `stts_events.py` row must be hand-edited in the same PR
(the exact playbook step both 19042 and 19043 initially missed — see the
2026-08-20 lesson in `backlog/docs/lessons-testing-evidence.md`); (2)
`_current_playground_audio_path` (`stts_events.py:2786`) has this handler as
its ONLY caller — decide its fate in the same PR (its underlying
`_current_playground_artifact`/`_current_audio_file` attributes have ~25 other
usages and stay).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The `play_current_audio` wrapper in `app.py` and the `play_current_audio` handler in `stts_events.py` no longer exist, and a whole-tree grep (including dynamic-dispatch shapes) finds no remaining reference.
- [x] #2 The security-coverage-map check from TASK-19043's playbook is performed and its outcome recorded in Implementation Notes: any validation the handler performed either has a live-path equivalent or is explicitly noted as not security-relevant.
- [x] #3 The persistent diagnostic inventory row for `stts_events.py` is hand-edited in the same PR to reflect the removed `logger.error`, and `scripts/check_persistent_diagnostic_inventory.py` does not regress further because of this change (it is already red on dev for unrelated drift — see TASK-19191).
- [x] #4 Any helper left caller-less by the removal (`_current_playground_audio_path`) is either removed with it or its retention justified; no new orphan is created.
- [x] #5 STTS-affected suites (`Tests/UI/test_stts_profile_library.py` and any suite importing the touched modules) pass.
<!-- AC:END -->

## Implementation Plan

1. Re-verify orphanhood at the branch base (`63901c30d`) with fresh whole-tree
   greps, including dynamic-dispatch shapes (string literals, getattr,
   `action_` names, concat fragments of `current_audio`).
2. Dead-graph walk both ends: census `_current_playground_audio_path`
   consumers (expected: only the handler → remove it too); census
   `play_audio_file` (the handler's function-local import target) for other
   live callers so its retention is justified; check `Path`/`logger` stay
   used in `stts_events.py`.
3. Coverage map: record whether the handler performs validation needing a
   live-path equivalent (expected: only a path-existence check, mirrored by
   `speech_playback_mixin._play_audio`'s captured-artifact + `exists()`
   checks on the real `#audio-play-btn`/`action_play_audio` path).
4. Baseline to files BEFORE deleting: `Tests/TTS/` (failure SET), the
   inventory gate suite `Tests/Architecture/test_persistent_diagnostic_
   inventory.py`, and `Tests/LLM_Calls/test_summarization_diagnostic_
   privacy.py` (its manifest-boundary hash pins the checked inventory, and is
   already red at base); plus a per-row rebuild-vs-committed drift census of
   `Docs/security/production-diagnostic-inventory.json`.
5. Delete the `app.py` wrapper (~:11409) and the `stts_events.py` handler
   (~:2767) plus the caller-less `_current_playground_audio_path` helper.
6. Hand-edit the inventory's `stts_events.py` row: re-derive call_count +
   diagnostic_digest from live code via the script's own `_scan_file`/
   `diagnostic_digest` helpers; adjust `summary.task_494_calls` by the same
   delta; verify internal invariants (len(owners)==owner_files; per-owner
   sums match summary buckets) with a printed check. Do NOT run `--write`
   (task-19191 owns dev's unrelated drift). Prove the residual
   rebuild-vs-committed drift is exactly the base drift minus the
   stts_events row.
7. Re-run the baselined suites + `Tests/UI/test_stts_profile_library.py`,
   repo-wide `--collect-only -q`, ruff on touched files, and a final
   whole-tree grep for the removed names.

## Implementation Notes

**Removed the orphaned pair durably** (base `63901c30d`, branch
`task/19190-burn`):

- `tldw_chatbook/app.py`: deleted the 9-line `play_current_audio` wrapper
  (was :11409). Re-verification at the base confirmed zero callers and no
  dynamic-dispatch shape: whole-tree greps for the name, for getattr/string
  shapes, for `action_` names, and for the `current_audio` concat fragment
  found only the two definitions, the wrapper's internal call, and backlog
  task records. The playground's real playback surface never touched this
  pair (`stts_screen.action_play_audio` -> `speech_playback_mixin.
  action_play_audio`/`_play_audio` via `#audio-play-btn`).
- `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`: deleted the
  `play_current_audio` handler (was :2767) AND
  `_current_playground_audio_path` (was :2786). Census at base: the helper
  had exactly two references in the tree — its definition and the call at
  :2769 inside the deleted handler — so with the handler gone it is
  caller-less and was removed with it (TASK-19043 had kept it precisely
  because this handler was then a live caller). Its underlying
  `_current_playground_artifact`/`_current_audio_file` attributes have many
  other consumers and stay. Dead-graph walk of the handler's other end:
  its function-locally imported `play_audio_file`
  (`Event_Handlers/TTS_Events/tts_events.py:4014`) retains live callers
  (tts_events itself :3546, plus the `TTS/audio_player.py` original used
  by watchlists) — no new orphan; `Path` (27 uses) and `logger` stay live
  in `stts_events.py`.

**Security coverage map (AC #2):** the handler performed no security-
relevant validation — only a path-existence check on the internally-tracked
playground artifact path with a warning notify (no user-supplied path, no
path/filename sanitization). The live playground playback path carries an
equivalent-or-stronger check on the real surface:
`speech_playback_mixin._play_audio` refuses with the identical "No audio
file to play" warning when no artifact is captured, and additionally checks
`audio_path.exists()` before playing. No tests drove the deleted handler
(zero test references at base), so nothing was retired or re-pointed.

**Inventory hand-edit (AC #3):** re-derived the
`Docs/security/production-diagnostic-inventory.json` row for
`stts_events.py` from live code via the script's own `_scan_file` +
`diagnostic_digest` helpers: call_count 30 -> 28, diagnostic_digest
`3f644cd30dd8e6a0fd5e` -> `e900ca7054c98c2d9160`, and
`summary.task_494_calls` 6974 -> 6972 (the -2 covers this task's deleted
`logger.error` plus TASK-19043's previously-missed decrement, which this
row-truthing necessarily absorbs — see the 2026-08-20 deletion-direction
lesson). Did NOT run `--write` (TASK-19191 owns dev's unrelated drift).
Verified: committed-file internal invariants all hold
(len(owners)=494==owner_files; sum492=1209, sum494=6972, sink_files=6 all
match summary); rebuild-vs-committed residual drift is exactly dev's
pre-existing drift minus the now-green stts_events row (base census: 33
drifted rows; after: 32; line-diff of the two censuses shows the
stts_events line as the sole change), so the check script stays red for
dev's pre-existing reasons only, with one fewer drifted row.

**Evidence** (identical commands both arms, venv python, module path
asserted into the worktree): `Tests/TTS/` before = 4086 passed / 16
skipped / 0 failed; after = 4086 / 16 / 0.
`Tests/Architecture/test_persistent_diagnostic_inventory.py` 10 failed /
55 passed both arms — failure SET diff empty (all 10 are the pre-existing
inventory-drift + task-15103 review-ledger reds tracked by TASK-19191).
`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` 3 failed / 254
passed both arms — failure SET identical (the manifest-boundary hash pins
were already red at base: the checked-inventory hash no longer matched the
ledger pin before this change). `Tests/UI/test_stts_profile_library.py`
163 passed. Repo-wide `--collect-only -q`: 52,113 collected, zero errors.
`ruff check` clean on both touched Python files. Final whole-tree grep for
the removed names: no hits outside backlog task records (this file and
TASK-19043's notes).
