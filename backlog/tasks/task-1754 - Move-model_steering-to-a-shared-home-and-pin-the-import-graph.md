---
id: TASK-1754
title: Move model_steering to a shared home and pin the import graph
status: Done
assignee: []
created_date: '2026-08-01 12:45'
updated_date: '2026-08-01 22:34'
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
- [x] #1 `model_steering` and its `_steering_field` helper live in a shared module that imports neither `word_bench`'s nor `character_probe`'s own modules, and both packages read a target's steering through it
- [x] #2 No copy of the steering-reading logic remains in `word_bench.storage` or anywhere else — callers of the old name continue to work or are updated, with no second implementation
- [x] #3 Importing `tldw_chatbook.Evals.character_probe` (any module of it) does not load `word_bench`'s capture client, normalizer, or `httpx`
- [x] #4 The character-probe hygiene test asserts on the real import graph (e.g. inspecting `sys.modules` after a fresh package import in a subprocess) rather than on source tokens, and is proven to FAIL when a forbidden import is reintroduced
- [x] #5 The word bench's own behaviour is unchanged: `Tests/Evals/word_bench/` passes untouched
- [x] #6 Phase 1 exit criterion 3 in `Docs/superpowers/plans/2026-08-01-character-probe-phase1-engine.md` is restored to its original absolute wording once the import graph actually satisfies it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create tldw_chatbook/Evals/steering.py: pure `model_steering`/`_steering_field`, moved verbatim (docstrings intact, references updated) from word_bench/storage.py. Only stdlib imports (json, typing) -- no import of word_bench or character_probe modules.
2. word_bench/storage.py: delete the local definitions, import `model_steering` from `..steering` instead so `word_bench.storage.model_steering` keeps working for existing callers (bench_editor.py, sample_bench.py, tests) via re-export -- but importing storage.py still pulls capture_client for its own unrelated NEUTRAL_SAMPLER use, so this re-export is for BACKWARD COMPATIBILITY of the old import path only, never used by character_probe.
3. character_probe/targets.py: import `model_steering` directly from `..steering`, not from `..word_bench.storage`, so importing character_probe never touches word_bench.storage/capture_client/normalizer/httpx.
4. Verify with a throwaway sys.modules probe (pre- and post-move) that word_bench.capture_client/normalizer/httpx no longer load when importing character_probe.targets.
5. Rewrite the hygiene test in Tests/Evals/character_probe/test_conversation_storage.py: replace the source-text grep with a subprocess-based import-graph assertion -- launch a clean interpreter (cwd computed from Path(__file__).resolve() so it is cwd-independent), import every character_probe submodule, and assert word_bench.capture_client / word_bench.normalizer / httpx are absent from sys.modules. Confirm the new test FAILS against the pre-move code (temporarily revert targets.py's import) and PASSES after.
6. Update Docs/superpowers/plans/2026-08-01-character-probe-phase1-engine.md exit criterion 3: restore the original absolute wording and replace the amendment note with one stating the import graph now satisfies it, moved by task-1754, enforced by the strengthened test.
7. Run the targeted suites, update task ACs/status, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved `model_steering`/`_steering_field` (pure functions of an `eval_models` row mapping) out of
`word_bench.storage` into a new `tldw_chatbook/Evals/steering.py`, whose only imports are stdlib
(`json`, `typing`). `word_bench.storage` now does `from ..steering import model_steering as
model_steering` (explicit re-export) so every existing caller spelling it
`word_bench.storage.model_steering` (bench_editor.py, sample_bench.py, test_storage.py) keeps
working unchanged; `character_probe/targets.py` now imports `..steering` directly, never through
`word_bench.storage`, so importing character_probe no longer pulls capture_client/normalizer/httpx
in transitively. Verified interactively: a `sys.modules` probe importing `character_probe.targets`
shows word_bench/httpx entries before the move and none after.

Rewrote the hygiene test (`test_character_probe_never_imports_the_word_bench_measurement_stack`) to
launch a subprocess, walk-import every `character_probe` submodule in a clean interpreter, and
assert `word_bench.capture_client`/`word_bench.normalizer`/`httpx` are absent from `sys.modules`.
`cwd` is computed from the test file's own resolved path (`parents[3]`), not the ambient working
directory, fixing the old test's silent-pass-from-any-other-cwd bug. Proved the new test actually
tests something: temporarily reverted `targets.py`'s import back to `..word_bench.storage` and
confirmed the test FAILS with `LOADED:httpx,tldw_chatbook.Evals.word_bench.capture_client,
tldw_chatbook.Evals.word_bench.normalizer`, then restored the fix and confirmed it passes again.
Kept the old source-token grep as a secondary, cheap check (also fixed to use an absolute,
`__file__`-derived package path instead of a relative literal), not as a substitute for the graph
assertion.

Restored phase 1 exit criterion 3 in
`Docs/superpowers/plans/2026-08-01-character-probe-phase1-engine.md` to its original absolute
wording ("No module under `character_probe/` imports word_bench's capture client, normalizer, or
canary code") and replaced the amendment note explaining why it was only met in spirit with one
recording that TASK-1754 closed the gap in both letter and intent, and that the rewritten test now
enforces it.

Verification: `Tests/Evals/character_probe Tests/Evals/word_bench Tests/UI/test_evals_steering_e2e.py`
-- 368 passed. Also ran `Tests/UI/test_evals_bench_editor.py` (80 passed) as an extra sanity check
on the `word_bench.storage.model_steering` re-export path, since that suite exercises it end to end
through the UI.

Files changed: tldw_chatbook/Evals/steering.py (new), tldw_chatbook/Evals/word_bench/storage.py,
tldw_chatbook/Evals/character_probe/targets.py, Tests/Evals/character_probe/test_conversation_storage.py,
Docs/superpowers/plans/2026-08-01-character-probe-phase1-engine.md.
<!-- SECTION:NOTES:END -->

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
