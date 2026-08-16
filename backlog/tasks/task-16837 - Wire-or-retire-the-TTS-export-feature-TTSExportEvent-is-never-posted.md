---
id: TASK-16837
title: 'Wire or retire the TTS export feature (TTSExportEvent is never posted)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - dead-code
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-16199 review (PR #1676, its F1) established — and it still holds at dev
`ee741cf10` — that the TTS export path is dead code in the shipped app:

- `TTSExportEvent` (`Event_Handlers/TTS_Events/tts_events.py:430`) is **never constructed
  or posted anywhere in production** — repo-wide grep finds only the class definition,
  `handle_tts_export` (`:3694`), and `on_tts_export_event` (`:3850`), plus tests.
- Even if something posted it, it would not arrive: `TTSEventHandler` is **not a
  `MessagePump`** — it is a plain class instantiated at `app.py:11127` and driven by
  direct method calls, so Textual name-dispatch never reaches `on_tts_export_event`.

Meanwhile three merged tasks (15471, 16194-era claim work, 16199) have invested real
concurrency engineering in protecting exports that cannot currently happen. Decide
whether export gets a UI affordance (a Save/Export action on TTS-bearing messages posting
through a real dispatch path) or gets retired along with its claim machinery's
export-only half.

If wired, two known residuals from the same review become live and belong to the wiring
task: **F6** — the check-then-act window between the shutdown sweep's single up-front
union read (`:3928`) and its per-file deletes across awaits (demonstrated by probe: an
export claiming in that gap loses its source); and **F4** — the union's live half
(exports admitted after the cancel pass) is untested: mutating the union to
snapshot-only left all 26 tests green, so a refactor can delete that half for free.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An explicit wire-or-retire decision is recorded (owner call)
- [x] #2 If wired: a user-reachable affordance posts the export through a dispatch path that actually reaches the handler, and the F6 residual window is closed or bounded with evidence (N/A — decision is RETIRE; F6 dissolved with the deleted delete-vs-claim surface)
- [x] #3 If wired: a test pins the shutdown-sweep union's live half (F4) so it cannot be silently removed (N/A — decision is RETIRE; the union itself is deleted)
- [x] #4 If retired: the event, both handlers, and the export-only claim machinery are removed with reachability evidence, and the 15471/16199 tests are re-scoped accordingly
<!-- AC:END -->

## Implementation Plan

1. Re-verify F1 at HEAD (`ecbcd5cd8`): repo-wide grep for `TTSExportEvent` /
   `handle_tts_export` / `on_tts_export_event` — production references only inside
   `tts_events.py` itself, plus one test file.
2. Evidence for the wire side: hunt for a designed affordance (Console message actions,
   Speech surfaces, STTS windows, Docs/ADRs/backlog). Evidence for the retire side:
   enumerate every symbol the export half owns and its reference graph.
3. Decide per the evidence and the owner's standing rulings (stability over quick wins;
   no speculative UI for an affordance nothing designed).
4. Decision = RETIRE (see Implementation Notes for the chain). Execute:
   baseline `Tests/TTS` + `Tests/TTS_Events` to files; delete `TTSExportEvent`,
   `handle_tts_export`, `on_tts_export_event`, `_exporting_audio_refcounts`,
   `_TTS_EXPORT_CLEANUP_RETRY_SECONDS`; collapse `_cleanup_audio_file`'s claim-poll
   loop to a single lock hold; strip the sweep's claim snapshot/union/skip branches;
   delete the three export tests and their import; add a tombstone pin against zombie
   reintroduction; update the stale TTS-To-Do doc line.
5. Re-run the suites, `--collect-only` sweep, ruff on touched files; per-symbol
   dead-verdict table in the notes; state what remains of task-16836.

## Implementation Notes

**Decision: RETIRE** (applying the owner's standing rulings — stability over quick
wins; never build speculative UI for an affordance nothing designed). Evidence chain:

1. F1 re-verified at HEAD `ecbcd5cd8`: repo-wide grep finds `TTSExportEvent`
   constructed nowhere in production — only its definition, the two handlers on the
   same class, and `Tests/TTS/test_tts_improvements.py`.
2. Even a posted event could not arrive: `TTSEventHandler` is a plain class
   (`tts_events.py:577`, no `MessagePump` base), instantiated once (`app.py:11106`)
   and driven by direct method calls — name-dispatch to `on_tts_export_event` is
   impossible by construction.
3. No affordance was ever designed for a per-message export: the Console message
   surface offers exactly `speak`/`speak-stop` (`console_transcript.py:108-109`,
   `:3587`); no button stub, no ADR, no design doc. The only doc mention is the
   legacy 2025-07 `Docs/Features/TTS-To-Do.md` checklist claiming the (unwired)
   implementation as complete.
4. The export UX that WAS designed is live on a different, unrelated path: Speech
   playground `#audio-export-btn` → `_export_audio()` → `_handle_audio_export()`,
   self-contained in `UI/Speech/speech_playback_mixin.py` (review-corrected: the
   originally-cited `app.export_current_audio`/`STTSEventHandler.export_current_audio`
   chain is itself ORPHANED — no production caller; queued for filing). Either way it
   shares nothing with the retired machinery.
5. Design coherence: per-message cached audio is deliberately ephemeral — playback
   schedules a secure delete 5s after play starts, and the artifact discipline
   zeroes files in place. A per-message export affordance would race its own
   source's shredder; wiring it would also make F6/F2/F3/F4 (16199 review) live
   and mandatory. Retirement deletes that whole obligation.

**Dead-verdict table (all verdicts by repo-wide grep at HEAD, re-checked after
deletion — zero references remain outside the tombstone test's negative asserts):**

| Symbol | Was at | Production references | Verdict |
|---|---|---|---|
| `TTSExportEvent` | `tts_events.py:430` | never constructed/posted; only the two handlers' signatures | dead |
| `handle_tts_export` | `:3694` | sole caller `on_tts_export_event` | dead (transitively) |
| `on_tts_export_event` | `:3850` | no caller; name-dispatch impossible (plain class) | dead |
| `_exporting_audio_refcounts` | `:688` | only writer was `handle_tts_export`; readers (`_cleanup_audio_file` poll `:3800`, sweep `:3896/:3930`) always saw an empty dict | dead (reads were no-ops) |
| `_TTS_EXPORT_CLEANUP_RETRY_SECONDS` | `:82` | only the poll loop's sleep, unreachable once the poll collapses | dead |

**What shipped** (branch `task/16837-burn`, base dev `ecbcd5cd8`):

- `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`: deleted the event class,
  both handlers, the refcount dict + constant; collapsed `_cleanup_audio_file`'s
  claim-poll loop to a single lock hold (semantics identical with an always-empty
  claim dict); stripped the 16199 sweep's claim snapshot/union/skip branches. The
  sweep's deletion passes, retry bookkeeping, owner guards, and drain calls are
  untouched.
- `Tests/TTS/test_tts_improvements.py`: removed the three export tests (the 15471
  claim test, the 16199 sweep test, the naming/metadata test) and the import;
  added `test_per_message_export_surface_stays_retired` (tombstone — mutation-
  verified red against a reintroduced `handle_tts_export` stub) and
  `test_shutdown_sweep_still_deletes_cached_and_retry_artifacts` (pins the
  surviving sweep contract — mutation-verified red against a gutted retry pass).
  Both mutations reverted Edit-based; production file shasum-verified after every
  swap.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` +
  `Docs/security/production-diagnostic-inventory.json`: the deletion removed 7
  diagnostic calls from `tts_events.py` (55→48), so the pinned inventory was
  regenerated via the sanctioned `check_persistent_diagnostic_inventory.py
  --write` (regenerate-never-hand-merge), and the now-moot
  `"No audio file found to export"` row was dropped from the 15743 pin table.
  NOTE: the regeneration also absorbed a pre-existing dev drift
  (`console_turn_file_card.py` 8→9, one metadata-only warning from PR #1728 whose
  `--write` was skipped) — verified pre-existing by running the checker with the
  pristine base `tts_events.py` (still exit 1). The sink-topology gate is now
  GREEN where it was red on dev. `test_task_15743_final_rebase_diagnostics_are_
  metadata_only` remains red with exactly dev's single pre-existing
  `console_runtime.py` "captures exception details" item (proved by running the
  pristine test+production pair — identical single failure); not this task's to fix.
- `Docs/Features/TTS-To-Do.md`: corrected the two stale lines that claimed the
  per-message export as a completed feature.

**Evidence:** baseline `Tests/TTS` + `Tests/TTS_Events` = 7 failed / 4128 passed /
16 skipped; after = the identical 7 pre-existing failures / 4127 passed (−3
removed, +2 added) / 16 skipped. `--collect-only` over all 17 test files importing
`tts_events`: 427 collected, zero errors. Full run of the 7 non-TTS importer
files: only dev's pre-existing reds (mixin-identity test fails byte-identically
against the pristine base production file). `ruff check` clean on all touched
Python files.

**Consequence for task-16836** (claims keyed by path + `_discard_tts_artifact`
honouring them): **the task dissolves almost entirely.** Its AC #1/#2 protect
export claims that no longer exist — there are no claims to key by path, no claim
for `_discard_tts_artifact` to honour, and the F3 id→path join is deleted. The
only unconditionally-live residue of its motivating findings is F2's observation
that `_discard_tts_artifact` bypasses `_artifact_cleanup_retry`-style claim checks
— which is now vacuous, since the claim invariant itself was retired with the
export path. Recommend closing 16836 as dissolved by this task, or re-scoping it
to nothing more than a doc note.

**Review corrections (controller, post-review):** the live playground export chain is
`_export_audio()` → `_handle_audio_export()` in `speech_playback_mixin.py`, not the
orphaned `export_current_audio` pair originally cited (that pair is a new filing
candidate). The inventory-drift note's "console_turn_file_card 8→9" is actually two
rows (`console_transcript.py` 8→9 + `console_turn_file_card.py` new 0→2), both from
PR #1728's skipped regen. The 17-importer/427-collected figure was a grep-substring
artifact (STTS_Events matches TTS_Events): the AST-correct count is 15 files/357
tests, zero errors either way. Review also found stronger retire evidence: task-559
deliberately skipped wiring this handler over the 5s auto-delete race. Review
independently verified TASK-16836 DISSOLVED by this retirement.
