---
id: TASK-26007
title: 'Tool catalog: fuzzy ranking and a visible deferred-tool listing'
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - agents
  - tools
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tool discovery misses paraphrase and hides what exists. Verified on origin/dev: Agents/tool_catalog.py:1408-1448 ranks by four tiers of substring match against a single query with a limit of 8 (Agents/agent_models.py:177), so "find files by name" does not match fs_glob; and when the catalog is deferred past the disclosure threshold (Agents/agent_models.py:174) the model sees only find_tools/load_tools (Agents/tool_catalog.py:238-249) with no enumeration, so it can conclude a capability is unavailable. Hermes ranks with BM25 plus stemming and embeds a name listing inside the search tool's own description so deferred tools are never invisible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A paraphrased query matches a tool whose name and description share no literal substring with it
- [ ] #2 The deferred-tool surface names the available tools (or their groups) so the model can never conclude a present capability is absent
- [ ] #3 The embedded listing is bounded and degrades to group names when the full list would be too large
- [ ] #4 Ranking remains deterministic for a given catalog and query - the same query returns the same order
- [ ] #5 Existing exact and prefix matches still rank above fuzzy matches
- [ ] #6 Tests assert a known paraphrase-to-tool mapping and that the listing appears when disclosure is deferred
<!-- AC:END -->
