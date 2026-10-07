---
id: TASK-17387
title: Kobold and TabbyAPI summarizers are generator functions
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-18 04:30'
labels:
  - llm-calls
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two local summarizers can never return a summary. Their streaming branches yield from the function body rather than from a nested generator, which makes the entire function a generator function: calling it returns a generator object and runs none of the body, on every path including the non-streaming one. No request reaches the server unless the caller happens to iterate the result.

The consequence for the deep-search pipeline is the same class of defect as the llama.cpp chain, one step worse. A caller that stores the result keeps a generator object where a summary belongs; the pipeline's failure detector inspects strings, so a generator passes it, and a generator is truthy, so the emptiness guard passes too. The object would be stored as a result's evidence content.

Fixing this is a contract change with governed-artifact fallout, which is why it is separate from the configuration fix that preceded it: nesting the streaming bodies re-attributes roughly twenty diagnostics from the owning function to the nested one, and the summarization diagnostic ledger keys every entry on its enclosing function name. The ledger and the contract tests that consume these functions as generators both need deliberate review, not a mechanical update.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A non-streaming call to either summarizer executes its body and returns a string
- [x] #2 A streaming call still returns an iterator that yields the same chunks it does today
- [x] #3 No caller can receive an object that the deep-search failure detector silently accepts as a summary
- [x] #4 The diagnostic ledger's re-attribution is reviewed and updated deliberately, with the reason recorded
- [x] #5 The existing contract tests are updated to the corrected contract, each with its reason, or shown to be unaffected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence gathered during task-17383 (2026-08-18), before any fix:

- An AST check over the module: `summarize_with_kobold` and
  `summarize_with_tabbyapi` are the only two `summarize_with_*` functions that
  are generators; the other seven are plain functions.
- The top-level yields are in the streaming branches and error paths --
  Kobold at the `if streaming:` body, TabbyAPI likewise plus its outer
  `except`, which yields for streaming and returns for non-streaming.
- Calling `summarize_with_tabbyapi(...)` directly with a stubbed transport
  returns `<generator object ...>`; the transport is never invoked.
- A prototype fix (nesting each streaming body in a `_stream_generator`, as
  `summarize_with_llama` already does) worked and left both functions
  non-generators, but broke 7 tests in
  `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`: the ledger keys
  entries as `(file, function, message, ordinal)`, so nesting re-attributes
  every diagnostic inside the moved body. That prototype was reverted rather
  than landed with a silently-rewritten ledger.

2026-10-04 close-out (wave5-g3 worktree, base = origin/dev `cddc89d3e7`): premise
verified STALE — the defect was fixed by a predecessor while this task sat
open, and this close changes no production or test code.

Fix landed as TASK-32805.3 (commit `78b9779c6c`, PR #2738, 2026-09-19, an
ancestor of this base), whose commit message already says "(closes the existing
task-17387)". That commit nests the streaming bodies exactly as the reverted
prototype did (`_kobold_stream` / `_tabby_stream`, and a one-shot
`_tabby_error_stream` for tabby's outer except), and the ledger
re-attribution this task required was reconciled deliberately in its own
commit `47bd46b20d` ("reconcile the summarization ledger and generator-shape
tests for task-32805.3/.4"), with the reason recorded in the ledger fixture
comment (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py:49-58`:
"the nine affected sites' qualnames and starting occurrences follow the
source ... so the canonical starting-projection digest changes").

Evidence at this base (all commands run in the worktree venv, Python 3.12.13):

- Ownership-scoped AST check (yields owned by the function's own body,
  excluding nested defs) over `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py`:
  all nine `summarize_with_*` functions report `generator_function=False`,
  including `summarize_with_kobold` (line 545) and `summarize_with_tabbyapi`
  (line 1046). Note: a naive `ast.walk` over the whole function reports True
  for all of them because it descends into the nested generator defs — the
  check must scope yield ownership.
- `python -m pytest Tests/LLM_Calls/test_kobold_tabby_config.py -q` ->
  13 passed (includes the return-type pins added by 78b9779c6c: non-streaming
  returns str "SUMMARY", streaming returns a non-string iterator -> ACs #1-#3).
- `python -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -q
  -k "starting_projection or ledger"` -> 9 passed (the re-attributed ledger
  digest is pinned and green -> AC #4).
- `python -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
  Tests/LLM_Calls/test_kobold_tabby_config.py -q` -> 3 failed, 266 passed.
  The 3 failures are PRE-EXISTING at base and unrelated to this task:
  `test_manifest_boundary_*` assert that the repo-wide generated manifest
  `Docs/security/production-diagnostic-inventory.json` matches a regenerated
  inventory ("checked inventory changed outside the two summarization owners").
  That is generated-manifest drift from later dev diagnostics commits (e.g.
  `410df59809` records File Notes diagnostics), the known class from
  lessons-backlog-hygiene ("A clean scoped rebase can still invalidate a
  repository-wide manifest"); no board task owns it as of this close — noted
  for the owner, not fixed here.

AC tick annotations: #1-#3 true since 78b9779c6c (2026-09-19), verified by
execution at this base; #4-#5 true since 47bd46b20d, verified by the passing
ledger-digest pins and the updated contract tests.

ADR check: ADR not required — no code or architectural change made in this
close; the fix's design (nested stream generators mirroring the llama/
oobabooga siblings) was already governed by TASK-32805 under the existing
summarization-transport decisions.
<!-- SECTION:NOTES:END -->
