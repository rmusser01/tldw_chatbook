# RAG query return and resize focus — TASK-32715

2026-09-17 UTC, `feat/component-pattern-library`, based on `71f7b33658`.

Shrinking Library from 170×48 to 80×24 moved focus from the RAG query or evidence
Open action to the Search/RAG rail row. The control stayed mounted; the
[initial trace](initial-focus-trace.json) rules out a stale query reference.
The resulting reverse-Tab journey walked the rail instead of returning through
the evidence/query controls. This resolves the ambiguity left by TASK-32714's
second native attempt.

The Notes responsive transition tried to restore a Notes semantic role for RAG
controls and fell back to the rail. Search/RAG now joins the existing exemption
for retained controls. Its panel reveals the currently focused descendant after
layout changes, using the same pattern as Import. A delayed reveal reads the
current focus, checks attachment/ancestry, and never restores an earlier target.
No bindings, styles, tokens, data or provider contracts change.

ADR required: no; this is a routine focus repair under existing
[ADR-031](../../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
and [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md).

## Automated evidence

- [Initial probe](initial-probe.txt): two wide-to-compact failures and six passing
  return journeys. Query identity remained stable throughout.
- [Regression red](regression-red.txt): eight focus-retention cases fail before
  the repair; four callback cases also fail because no resize reveal is queued.
- Eight repaired resize cases pass across both themes, query/evidence origins,
  compact/wide starts and width/height transitions. A deliberately held reveal
  initially also held Textual's unrelated focus-scrolling callbacks; the
  [intermediate result](intermediate-gate.txt) records that instrumentation
  failure. The gate now defers only the resize reveal, and all four
  [newer-focus cases](deferred-green.txt) pass.
- [Final targeted suite](targeted-tests.txt): **98 passed in 349.68s**. It covers
  query return, answer paging, result arrival, shared resize/focus gates, Prompt
  resize and Import resize, without exclusions.
- New test/runner pass Ruff lint/format; both changed production ranges pass
  formatting. [Lint comparison](lint-comparison.json) has no new diagnostics
  (205 existing in LibraryScreen; 5 existing in the RAG panel).
- Independent read-only review found no actionable issue in ownership, current
  focus checks, race coverage or native evidence boundaries.

```sh
.venv/bin/python -m pytest \
  Tests/UI/test_library_rag_query_return.py \
  Tests/UI/test_library_rag_answer_navigation.py \
  Tests/UI/test_library_rag_result_focus.py \
  Tests/UI/test_library_resize_focus_gates_t23025.py \
  Tests/UI/test_library_prompt_resize_focus.py \
  Tests/UI/test_library_ingest_resize_focus.py \
  -q --tb=short --show-capture=no
```

## Native evidence

The [result](result.json) passes all four journeys and sixteen resize transitions.
The [runner](native_check.py) uses real TldwCli, an owned tmux terminal,
private configuration/databases, real Media storage and workspace membership.
It adapts RAG requests to real local keyword retrieval and controls the provider
chat seam. The real answer service builds prompts and validates references.
Each independent cell starts with a programmatically focused/revealed query;
Enter, Tab, page keys, reverse Tab and text entry drive the measured journey.

The four journeys cover query and evidence focus in both themes through
80×24 → 170×48 → 170×24 → 170×48. The checks require current widget identity,
painted focus, preserved query selection/scope/mode/answer, editable query after
return, and exactly one retrieval/provider call per cell. No result counts are
injected. All eight captures were rendered and inspected: focused query borders,
carets and Open labels remain visible; the returned query accepts the appended
text in RAG Answer mode. SVG source trailing whitespace alone was normalized;
[hashes](capture-hashes.json) record the original and stored artifacts.

| Theme / origin | Retained focus at 80×24 | Editable query at 170×48 |
| --- | --- | --- |
| Dark / query | [Query](textual-dark-query-compact.svg) | [Edited](textual-dark-query-query-edited.svg) |
| Dark / evidence | [Open](textual-dark-evidence-compact.svg) | [Returned and edited](textual-dark-evidence-query-edited.svg) |
| Light / query | [Query](textual-light-query-compact.svg) | [Edited](textual-light-query-query-edited.svg) |
| Light / evidence | [Open](textual-light-evidence-compact.svg) | [Returned and edited](textual-light-evidence-query-edited.svg) |

[Lifecycle checks](lifecycle.json) confirm normal shutdown, shell exit 0 and
PID absence before closing the owned terminal. All ten private databases pass
read-only `quick_check`; no conversation messages were added. The seeded source
and three default profile files are unchanged. Logs contain a normal stop and
no error/critical/unhandled-exception lines; both faulthandler logs are empty.

Controlled replies do not qualify real-provider behavior, semantic
retrieval or factual grounding. The very narrow single-pane layout is outside
this bounded matrix. [Independent review](review.json) has no outstanding
findings. The [task-ID check](task-id-check.json) found no other owner across
310 refs and 27 worktrees. No full suite, push or merge was performed. Next
bounded review: Recent searches and replaying queries.
