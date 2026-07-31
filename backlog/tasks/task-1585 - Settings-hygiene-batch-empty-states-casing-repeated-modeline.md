---
id: TASK-1585
title: 'Settings hygiene batch: empty states, casing, repeated mode line'
status: Done
assignee:
  - '@claude'
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

- [x] Theme tree and Workspaces center pane have empty-state copy
- [x] The mode-line disclaimer is not repeated verbatim on every category
      (category-specific or shown once)
- [x] User-facing group headers and enum values render in consistent casing
      (raw config ids stay raw only where they name config keys)
- [x] Internal Prompts rows left-align and its search idiom is reconciled
      with the global filter
- [x] The provider env-var counts read unambiguously
- [x] Non-draft categories are consistent about showing vs hiding the
      Save/Revert button pair

## Implementation Plan (the how)

1. RED tests per item (pure helpers where possible).
2. Empty-state hints; Overview-only disclaimer via `_mode_line_text`;
   `subsystem_display_title` + `skill_trust_display` + `env_var_summary`
   helpers; left-align CSS; Save/Revert pair gated to guided categories.

## Implementation Notes

- Empty states: `#settings-theme-tree-hint` under the Themes tree and
  `#settings-workspace-card-hint` when no workspace is selected.
- Mode line: `_mode_line_text` keeps the MCP/ACP disclaimer on Overview
  only; every other category reads "Mode: <title>".
- Casing: `subsystem_display_title` (acronym map + special-case
  "websearch" → "Web search") for Internal Prompts group headers;
  `skill_trust_display` strips the raw `trust_` prefix on the Privacy
  rows. Splash gallery casing stays declined per task-1565's owner
  decision.
- Internal Prompts rows: `.internal-prompt-row { text-align: left; }` in
  `_agentic_terminal.tcss` (Buttons center labels by default) + bundle
  rebuild via build_css.py; the embedded search's placeholder already
  differs from the category filter and filters prompts, so beyond
  left-alignment no idiom change was needed.
- Env-var counts: shared `env_var_summary(present, missing, configured)`
  → "N of M referenced env vars are set (K unset)", used by both the
  Privacy rows and the inspector detail row; one test pin updated.
- Save/Revert pair now renders ONLY for GUIDED_SETTINGS_MUTATION_
  CATEGORIES (was: shown-disabled on read-only categories, omitted on
  five own-persistence ones with no stated rule) — completes the
  task-1580 footer gating at the widget level.

TDD RED-first throughout. Files: `settings_screen.py`,
`settings_privacy_security.py`, `settings_internal_prompts_panel.py`,
`settings_theme_editor.py`, `_agentic_terminal.tcss` (+ rebuilt bundle),
tests in `test_settings_configuration_hub.py`,
`test_settings_theme_editor.py`, `test_settings_privacy_security.py`.
