---
id: TASK-24454
title: Provider readiness is recomputed on the composer keystroke path
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typing in the Console composer recomputes provider readiness on every keystroke.
`normalize_provider_config_key` runs 11,003 times across a 43-keystroke burst -- 256 calls per
keystroke -- reached via `_ensure_active_console_session_settings` ->
`build_console_settings_readiness` -> `get_provider_readiness`.
`_ensure_active_console_session_settings` itself runs roughly three times per keystroke.

The absolute cost is currently small (~30 ms across the burst), so this is not the headline
keystroke cost. It is filed because provider readiness is configuration-derived state that does
not change while a user types, and recomputing it per keystroke is work that should not be on
the input path at all -- it will scale with the provider/model catalogue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provider readiness is not recomputed as a side effect of an ordinary composer keystroke
- [ ] #2 Readiness still refreshes when the inputs that determine it actually change (provider, model, or credential configuration)
- [ ] #3 `normalize_provider_config_key` call count per keystroke is reduced to approximately zero in a live probe
- [ ] #4 Console readiness indicators continue to reflect configuration changes without requiring a restart
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
NOT IMPLEMENTED in the 2026-08-29 review pass.

Re-confirmed as real but low-value: `normalize_provider_config_key` runs 256 times per keystroke
via `_ensure_active_console_session_settings` -> `build_console_settings_readiness` ->
`get_provider_readiness`, and totals only ~30 ms across a 43-key burst. It is filed as a
correctness-of-layering issue (configuration-derived state does not change while a user types)
rather than a measurable win today, and it will matter more as the provider/model catalogue grows.

Deferred behind the composer guards in task-24453, which addressed the costs that actually
dominated the keystroke path.
<!-- SECTION:NOTES:END -->
