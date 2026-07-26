---
id: TASK-694
title: >-
  Port the remaining legacy tools (rag_search, web_search, search_notes,
  code_audit) behind the built-in gate
status: To Do
assignee: []
created_date: '2026-07-26 06:30'
labels:
  - tools
  - agents
  - security
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-545's P2 acceptance criterion originally named eight tools to move into `BuiltinToolProvider`. The P2 design spec (`Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md`) deliberately narrowed that pass to five — `write_file`, `create_note`, `update_note` (ported and tagged `mutates`) plus `read_file`, `list_directory` (already registered, newly tagged `reads`) — and TASK-545's AC was rewritten to match what shipped.

That rewrite left `rag_search`, `web_search`, `search_notes`, and `code_audit` tracked by nothing. They remain on the legacy `ToolExecutor` (System A), which is reachable only from the deprecated chat path and is entirely ungated. This task exists so that scope narrowing does not silently become scope loss.

Note this is not purely a port: each tool needs a risk-tag decision. `search_notes` reads user notes and `rag_search` reads indexed user content, so both are candidates for the `reads` tag introduced in P2; `web_search` performs outbound network requests and has no existing tag that describes it. Whether the vocabulary needs a fourth tag is part of the work, not a precondition.

Sequencing: TASK-545's P3 decides System A's fate (port, gate in place, or delete `Tools/tool_executor.py`). If P3 removes System A, this task is how these four tools survive that removal rather than disappearing with it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `rag_search`, `web_search`, `search_notes`, and `code_audit` are reachable from the agent-runtime path (System B), behind default-off `[tools]` gates matching the established pattern
- [ ] Each carries a `risk_tags` value drawn from the built-in vocabulary, with the choice justified; if a new tag is needed it is added to `BUILTIN_HIGH_RISK_TAGS` only, leaving MCP's `HIGH_RISK_TAGS` unchanged
- [ ] Any tool whose tag makes it high-risk resolves to `ask` and is proven to prompt by a test using the real tool, not a synthetic double
- [ ] Each new tool name is added to `_SHADOWED_BUILTIN_NAMES`, with a gates-enabled test covering it (the drift guard builds a default-config provider and cannot see gated names)
- [ ] Default posture is unchanged: with no `[tools]` flags set, the built-in catalog is exactly what it was before this task
- [ ] TASK-545's P2 criterion is updated to record that the originally-named eight tools are now fully accounted for
<!-- AC:END -->
