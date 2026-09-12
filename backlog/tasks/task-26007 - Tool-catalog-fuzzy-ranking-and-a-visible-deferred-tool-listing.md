---
id: TASK-26007
title: 'Tool catalog: fuzzy ranking and a visible deferred-tool listing'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-09-01 18:25'
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
- [x] #1 A paraphrased query matches a tool whose name and description share no literal substring with it
- [x] #2 The deferred-tool surface names the available tools (or their groups) so the model can never conclude a present capability is absent
- [x] #3 The embedded listing is bounded and degrades to group names when the full list would be too large
- [x] #4 Ranking remains deterministic for a given catalog and query - the same query returns the same order
- [x] #5 Existing exact and prefix matches still rank above fuzzy matches
- [x] #6 Tests assert a known paraphrase-to-tool mapping and that the listing appears when disclosure is deferred
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: paraphrase mapping, determinism + tier ordering, bounded listing (names->groups->hard cap), deferred-plan embedding\n2. find(): tier-4 stemmed-token overlap (stopwords, light suffix stem, matched-count bias, ties on name/id)\n3. build_find_tools_schema: names <=700 chars, else prefix groups with counts, else hard slice; empty -> the plain schema object\n4. build_first_request_schema_plan wires the allowed catalog names into the discovery plan's find_tools
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Tier-4 fuzzy in ToolCatalogRegistry.find(): stemmed-token overlap (re token split, stopword drop, deterministic light suffix stem ing/es/ed/s with a 3-char stem floor) ranked strictly below the four substring tiers (AC#5 pinned: 'read' still puts fs_read first), more-matched-tokens-first then the same (name, id) tie-break — same query, same order (AC#4 pinned). 'find files by name' -> fs_glob with zero literal substring (AC#1/#6 pinned) and zero-overlap tools stay out. Deferred surface: build_find_tools_schema embeds a sorted name listing in find_tools' description; past 700 chars it degrades to prefix groups with counts (fs_* (7)…), then a hard slice — empty input returns the plain schema object (AC#3). build_first_request_schema_plan feeds the allowed catalog names into the discovery plan (AC#2; pinned by a 301-tool forced-deferral test asserting the group form). One legacy pin updated: find('leaked_name')==[] became 'suppressed entry never appears' — the fuzzy tier legitimately surfaces the SURVIVING same-token entry; suppression intent unchanged. Chose token-overlap over BM25: deterministic, dependency-free, and the catalog is ~dozens of rows — corpus statistics would be noise at this scale. 4 new tests; Tests/Agents exact baseline.
<!-- SECTION:NOTES:END -->
