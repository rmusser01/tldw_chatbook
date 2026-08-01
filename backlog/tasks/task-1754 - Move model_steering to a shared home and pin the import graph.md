---
id: TASK-1754
title: Move model_steering to a shared home and pin the import graph
status: To Do
assignee: []
created_date: '2026-08-01 12:45'
labels:
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A target's steering lives inside its `eval_models` row's `config` JSON, and the app has exactly one
reader for it: `word_bench.storage.model_steering` (with its helper `_steering_field`). Both are
pure functions of a row dict — they touch nothing else in `word_bench` — but they live inside a
module whose own imports reach `capture_client`, `normalizer`, and `httpx`.

The character-probe engine reuses that reader deliberately (`character_probe/targets.py`), and that
reuse is correct: the alternative was tried and produced Critical C1 of the phase 1 whole-branch
review, in which a second, private steering reader looked for `row["system_prompt"]` — a key no
`eval_models` row has ever carried — so every real run silently dropped its target's steering while
a hand-built test fixture agreed with the bug. Duplicating a row reader is the failure mode, not the
fix.

The cost is that importing the character-probe package now drags word_bench's measurement stack in
transitively, which contradicts the letter of that package's own separation rule (phase 1 exit
criterion 3, since amended to say what is actually true). The intent of the rule holds — no
distribution vocabulary or concepts appear anywhere in the character-probe source or surface — but
the import graph does not.

The rule's in-repo guard is also weaker than the rule itself:
`Tests/Evals/character_probe/test_conversation_storage.py::test_character_probe_never_imports_the_word_bench_measurement_stack`
greps each module's SOURCE TEXT for forbidden tokens, so it cannot detect an import-graph violation
at all and passes on exactly this situation. A hygiene test that cannot fail on the thing it is
named after is a false assurance, and this repo has paid for false assurances before.

Moving the reader to a shared home that neither package's stack rides on lets both bench types keep
one definition of what a stored row means without either dragging the other's dependencies. It
complements TASK-1744's de-duplication theme: one function, one meaning, imported by everyone who
needs it, rather than a copy per caller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `model_steering` and its `_steering_field` helper live in a shared module that imports neither `word_bench`'s nor `character_probe`'s own modules, and both packages read a target's steering through it
- [ ] #2 No copy of the steering-reading logic remains in `word_bench.storage` or anywhere else — callers of the old name continue to work or are updated, with no second implementation
- [ ] #3 Importing `tldw_chatbook.Evals.character_probe` (any module of it) does not load `word_bench`'s capture client, normalizer, or `httpx`
- [ ] #4 The character-probe hygiene test asserts on the real import graph (e.g. inspecting `sys.modules` after a fresh package import in a subprocess) rather than on source tokens, and is proven to FAIL when a forbidden import is reintroduced
- [ ] #5 The word bench's own behaviour is unchanged: `Tests/Evals/word_bench/` passes untouched
- [ ] #6 Phase 1 exit criterion 3 in `Docs/superpowers/plans/2026-08-01-character-probe-phase1-engine.md` is restored to its original absolute wording once the import graph actually satisfies it
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Two Phase 2 caveats recorded here so they are not rediscovered the expensive way. Neither is this
task's work; both are consequences of phase 1's fail-loudly choices that a UI author will meet.

- **`_stored_int_field` is strict against JSON floats.** `load_character_bench` rejects a stored
  `512.0` as a non-integer `max_tokens`; the previous `int()` coercion silently truncated it (and
  accepted the string `"512"`). A form control or JSON payload that emits whole numbers as floats
  will make a bench fail to load. Coerce at the UI boundary — do not loosen the loader back.
- **`eval_models` has `UNIQUE(name, provider, model_id)`.** `targets.resolve_target` rejects a
  prefix-steered target and tells the user to use a chat-mode one instead; steering is immutable per
  row (there is no `update_model`), so that means creating a NEW row. A Phase 2 "duplicate this
  target" affordance must offer a **different name**, or `create_model` raises `ConflictError`.
<!-- SECTION:NOTES:END -->
