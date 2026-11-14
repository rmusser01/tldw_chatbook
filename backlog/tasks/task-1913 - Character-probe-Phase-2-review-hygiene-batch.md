---
id: TASK-1913
title: Character-probe Phase 2 review hygiene batch
status: Done
assignee: []
created_date: '2026-08-02 04:15'
labels:
  - evals
  - character-probe
  - hygiene
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small findings deferred during the character-probe Phase 2 authoring UI review
loop (`feat/character-probe-phase2-1691`). None is a defect a user would hit as a
trap; each is an inconsistency worth closing while the code is fresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `CharacterBenchEstimate` applies one honesty rule: it either prints a zero count or refuses to, consistently across a 0-card draft and a missing probe set
- [x] #2 The `character_bench` Delete branch is gated on a resolved row, matching the `bench` branch beside it
- [x] #3 `character_probe.storage` is imported at module scope in `evals_state.py`, matching every other consumer
- [x] #4 Probe-set selection does not depend on a `created_at DESC` tie at one-second granularity
- [x] #5 A target row carrying BOTH `prefix` and `system_prompt` is flagged as corrupt in the targets listing rather than silently degrading to an unsuffixed label
- [x] #6 `EvalsViewModel.character_benches()` and `probe_sets()` have direct unit tests
- [x] #7 `CardRow.__init__` calls `super().__init__()` before setting instance attributes, matching `Widgets/emoji_picker.py`
- [x] #8 `SnippetEditor._handle_import_file_selected` reads its file off the UI thread, matching the shared `LibraryRail._read_import_file_off_thread` seam its two siblings now use
<!-- AC:END -->

## Implementation Notes

Completed in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).
Premise re-verified at dev a3c47c6c72 before fixing: AC2 and AC8 had
already landed (AC2's gating is in `_compose_inspector_pane`'s
character branch, `character_bench_row is not None`, now shared with
the new Duplicate button; AC8's read runs on a bounded thread worker
with the parse/write/notify marshalled back). The audit's "AC1/AC5
landed" did NOT hold at re-verification -- both were still violated in
`inspector.py`/`character_bench_editor.py` and are fixed here. AC7 was
still violated exactly at card_picker.py:56-59, as the audit noted.

- **AC1**: the estimate now REFUSES consistently -- a bench with zero
  computable calls (no characters or no targets picked, the state every
  "+ New character bench" draft starts in) renders "Nothing to estimate
  yet ..." instead of "0 calls · ~00:00", matching the missing-probe-
  set refusal. Pinned by `test_a_zero_call_character_bench_refuses_to_
  print_an_estimate`.
- **AC3**: `character_probe.storage` (`is_character_bench`,
  `is_probe_set`) now imported at module scope in `evals_state.py`,
  matching `evals_screen.py`/`library_rail.py`'s module-scope imports;
  the three function-local imports are gone. (The Chat_Functions local
  import stays local -- that one is heavy by its own comment's
  rationale, and this AC names character_probe.storage only.)
- **AC4**: `EvalsDB.list_datasets` gains `ORDER BY created_at DESC,
  id DESC` -- `created_at` is `datetime('now')` at one-second
  granularity, so two same-second datasets left "the newest dataset"
  (the first row, which "+ New character bench" binds to) to SQLite's
  unspecified tie order. `id` is a uuid4 string: arbitrary but STABLE,
  which is the property the binding needs. Pinned by
  `test_probe_set_order_is_deterministic_across_a_created_at_tie`
  (forces the tie by pinning both rows' `created_at`, asserts the
  order repeats).
- **AC5**: a corrupt-steering row (`model_steering` raising -- BOTH
  `prefix` and `system_prompt`, a non-mapping config, a non-string
  field) renders "· corrupt steering config — unusable until its
  eval_models row is fixed (<error>)" in the character targets
  listing, never the silent unsuffixed label (which read as a healthy
  target that inexplicably fails to run). Pinned by
  `test_a_target_with_corrupt_steering_is_flagged_not_silently_bare`.
- **AC6**: new `Tests/UI/test_evals_view_model.py` -- 6 direct,
  mount-free unit tests for `character_benches()` /
  `character_bench_by_id()` / `probe_sets()` (classification, the
  classic-tasks subset invariant, the probe_sets/word_bench_datasets
  partition, and the AC4 tie determinism).
- **AC7**: `CardRow.__init__` now calls `super().__init__(...)` before
  setting `card_id`/`card_name`/`_selected`, matching
  `Widgets/emoji_picker.py`'s `EmojiButton` ordering
  (`emoji_picker.py:717-720`). Behaviour-neutral (the label is composed
  from the arguments), pinned by the existing card-picker suite
  passing unchanged.

Modified files: `tldw_chatbook/UI/Evals/card_picker.py`,
`tldw_chatbook/UI/Evals/evals_state.py`,
`tldw_chatbook/UI/Evals/character_bench_editor.py`,
`tldw_chatbook/UI/Evals/inspector.py`,
`tldw_chatbook/DB/Evals_DB.py`, `Tests/UI/test_evals_view_model.py`
(new), `Tests/UI/test_evals_character_bench_editor.py`.
