---
id: task-1585
title: 'Settings hygiene batch: empty states, casing, repeated mode line'
status: To Do
assignee: []
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p3
dependencies: []
priority: low
---

## Description (the why)

Critique rescore P3 leftovers, batched: Theme's collapsed "Themes" tree
leaves a large blank region and Workspaces' center pane is nearly empty
(both need empty-state copy); "Runtime controls stay in MCP and ACP"
repeats verbatim on all 17 categories; snake_case leaf content
(rag_reranker (6), tech_pulse, trust_uninitialized) sits against Title Case
chrome; Internal Prompts center-aligns row labels and embeds a second
search idiom ("Search prompts…") beside the global "/" filter; Privacy's
"Provider env vars: 0 present / 19 missing / 19 configured" reads as
contradictory; Workspaces alone omits the Save/Revert buttons entirely
while other non-draft categories show them disabled. The five view-only
Domain Defaults placeholder pages are a KNOWN accepted WIP state (owner
review 2026-07-31) and are out of scope here.

## Acceptance Criteria (the what)

- [ ] Theme tree and Workspaces center pane have empty-state copy
- [ ] The mode-line disclaimer is not repeated verbatim on every category
      (category-specific or shown once)
- [ ] User-facing group headers and enum values render in consistent casing
      (raw config ids stay raw only where they name config keys)
- [ ] Internal Prompts rows left-align and its search idiom is reconciled
      with the global filter
- [ ] The provider env-var counts read unambiguously
- [ ] Non-draft categories are consistent about showing vs hiding the
      Save/Revert button pair
