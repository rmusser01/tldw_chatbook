---
id: TASK-3222
title: Settings Tools screen does not show the web_deep_search gate
status: To Do
assignee: []
created_date: '2026-08-07 16:30'
labels:
  - web-tools
  - ux
  - settings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_deep_search's enable switch is the TOML key [tools] web_deep_search_enabled (double opt-in per the task-1356 spec), but Settings ▸ Tools derives its rows from _GATEABLE_BUILTINS — a different registry — so the toggle never appears in the UI. This was a recorded spec non-goal (the tool ships config-file-only on purpose), but a user who finds other tool switches in Settings will reasonably conclude this one does not exist. The final whole-branch review (2026-08-07) recommended a follow-up task rather than silence. Scope question for whoever picks this up: surface the [tools] boolean as a read-only or editable row, or generalize the Settings source so config-gated tools appear alongside _GATEABLE_BUILTINS. Note the restart-to-apply semantics (provider builds specs at construction) must stay visible in whatever UI ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings ▸ Tools shows the web_deep_search opt-in state (at minimum read-only with the config key named)
- [ ] #2 The restart-to-apply requirement is stated wherever the state is shown
- [ ] #3 Toggling (if editable) round-trips to [tools] web_deep_search_enabled in config.toml
<!-- AC:END -->
