---
id: TASK-40129
title: Search and ask about local text files in Chatbook
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
type: feature
references:
  - Docs/superpowers/specs/2026-11-11-local-files-search-design.md
  - Docs/superpowers/plans/2026-11-11-local-files-search.md
---

## Description

Give people a dedicated native Local Files view for finding and asking about text files in a selected indexed Workspace folder.

## Acceptance Criteria

- [ ] Open Local Files from the command palette, select a local folder, and find an exact filename or words inside an eligible file without embeddings.
- [ ] Preview the matching lines and reveal the original file on supported hosts. A stale or excluded file cannot be disclosed through an old result.
- [ ] Ask a question and see citations with relative paths and line ranges. An unavailable or declined request sends no file evidence to an answer provider.
- [ ] Existing folder index controls, Library retrieval, and Console permissions retain their contracts.

## Implementation Plan

1. Add user-initiated search and bounded evidence validation to the existing project index.
2. Reuse the evidence and answer pipeline for file citations and validate the chosen destination before disclosure.
3. Add the native view, command palette action, and Settings handoff. Verify with targeted service, provider-boundary, and mounted tests.

- ADR required: yes
- ADR paths: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md and backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
- Reason: a new user-facing search authority and file citation source.
