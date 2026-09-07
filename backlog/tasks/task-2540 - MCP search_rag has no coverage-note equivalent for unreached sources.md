---
id: TASK-2540
title: MCP search_rag has no coverage-note equivalent for unreached sources
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - rag
  - honesty
dependencies: []
priority: low
---

## Description

The Library's evidence list shows a coverage note, `library_rag_coverage_note`
(`Library/library_rag_state.py:1537-1612`), naming which configured sources semantic
search never reached — sourced from a `diagnostics["semantic_scope_coverage"]` entry
the Library's retrieval service attaches to its result.

PR-T3 gave the MCP `search_rag` tool the Library's *all-weak* notice
(`library_rag_all_matches_weak`, called directly), but deliberately did not attempt
the coverage note: `SimplifiedRAGSearchService` (the service `MCP/tools.py`'s
`search_rag` calls) never produces a `semantic_scope_coverage` diagnostic at all —
there is no Library-shaped input to mirror, and inventing one from a different
service's plumbing was explicitly out of scope for that PR (it needs a new upstream
diagnostic, not a UI-side copy change).

Filed so the gap doesn't disappear once PR-T3 merges: an agent calling `search_rag`
today has no way to learn "your query only covered 2 of your 5 configured sources"
the way a Library user does.

## Acceptance Criteria

- [ ] `SimplifiedRAGSearchService` (or its caller) produces a
      `semantic_scope_coverage`-shaped diagnostic when a run's semantic scope does not
      cover every configured source.
- [ ] The MCP Test Tool result surfaces a coverage note mirroring
      `library_rag_coverage_note`'s register when that diagnostic is present.
- [ ] No coverage note — and no false claim of full coverage — appears when the
      diagnostic is absent (i.e. this task does not change behavior for any run that
      doesn't produce the new diagnostic).
- [ ] Additive tests cover both the diagnostic's emission and its surfacing in the
      Test Tool result.
