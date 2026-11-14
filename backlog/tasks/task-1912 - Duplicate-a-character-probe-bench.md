---
id: TASK-1912
title: Duplicate a character-probe bench
status: Done
assignee: []
created_date: '2026-08-02 04:15'
labels:
  - evals
  - character-probe
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Evals inspector offers Duplicate for a word bench but not for a character-probe
bench. `duplicate_bench`'s `BenchConfig`/`save_bench` round-trip rejects a
`CharacterProbeConfig`'s shape, so the affordance needs a `duplicate_character_bench`
counterpart rather than a widened gate.

Delete already works for both bench types, so there is no trap here — this is
missing convenience, not a dead end. Deferred from the character-probe Phase 2
authoring UI (`feat/character-probe-phase2-1691`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A character-probe bench can be duplicated from the same affordance a word bench uses
- [x] #2 The copy carries the probe set, character selection, targets, and sampler of the original
- [x] #3 The copy takes a unique name and does not disturb the original
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

- Engine: `duplicate_character_bench(db, task_id)` in
  `character_probe/storage.py` -- `load_character_bench` ->
  `replace(source, name=_unique_name(f"{source.name} copy"))` ->
  `save_character_bench`. A LOCAL `_unique_name` twin rather than an
  import of `word_bench.storage._unique_name`: that module imports
  `capture_client` (transitively `httpx`), and this package must never
  acquire distribution vocabulary in its import graph -- the same
  constraint `Evals/steering.py`'s module docstring already documents
  for `targets.py`.
- UI: the SAME `#evals-duplicate-bench` button now composes for a
  resolved character-bench selection (AC1), and
  `_on_duplicate_bench_pressed` dispatches by selection kind. The
  post-duplicate lookup/re-selection are kind-aware
  (`character_bench_by_id` / `kind="character_bench"`) -- `bench_by_id`
  only ever resolves word benches, so the word-bench path's lookups
  would have landed the copy on "could not be found".
- Tests: 3 engine cases in `test_bench_storage.py` (full-field carry
  incl. probe set/characters/targets/sampler per AC2; unique name +
  byte-identical original per AC3; missing source raises naming the
  id) and 2 UI cases in `test_evals_screen.py` (button present and
  enabled beside Delete; real-click copy -> copy selected, original
  undisturbed, 2 live rows, toast names the new bench). The pre-1912
  test that pinned "no Duplicate for character benches" was replaced
  by these -- it existed to stop Delete silently disappearing, which
  the new tests still pin.

Modified files: `tldw_chatbook/Evals/character_probe/storage.py`,
`tldw_chatbook/UI/Screens/evals_screen.py`,
`Tests/Evals/character_probe/test_bench_storage.py`,
`Tests/UI/test_evals_screen.py`.
