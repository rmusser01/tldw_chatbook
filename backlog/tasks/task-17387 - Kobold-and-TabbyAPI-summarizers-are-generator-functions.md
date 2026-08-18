---
id: TASK-17387
title: Kobold and TabbyAPI summarizers are generator functions
status: To Do
assignee: []
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
- [ ] #1 A non-streaming call to either summarizer executes its body and returns a string
- [ ] #2 A streaming call still returns an iterator that yields the same chunks it does today
- [ ] #3 No caller can receive an object that the deep-search failure detector silently accepts as a summary
- [ ] #4 The diagnostic ledger's re-attribution is reviewed and updated deliberately, with the reason recorded
- [ ] #5 The existing contract tests are updated to the corrected contract, each with its reason, or shown to be unaffected
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
<!-- SECTION:NOTES:END -->
