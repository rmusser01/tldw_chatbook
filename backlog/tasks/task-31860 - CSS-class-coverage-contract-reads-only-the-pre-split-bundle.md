---
id: TASK-31860
title: >-
  test_css_class_coverage_contract reads only the pre-split bundle — 157 styled tokens report as unstyled
status: To Do
assignee: []
created_date: '2026-09-05 23:30'
labels: [tests, css, infra]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the Inspect-rail critique burn-down (TASK-31662, ledgered close-out
commitment). `test_css_class_coverage_contract`'s `_styled_tokens()` reads only
`tldw_cli_modular.tcss`, but the TASK-25812 bundle split moved whole component
families (all `console-inspector-section-row*` rules among them) into per-screen
sheets like `screen_agentic_console.tcss`. Result: 157 genuinely-styled tokens
fail the contract at baseline, and every new styled class in a split sheet adds
another false red — the contract is unenforceable noise. The documented one-line
fix: point the helper at `APP_STYLESHEETS` (all sheets the app loads); a probe
during 31662 showed the failure list drops to 8 (all real).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The contract's styled-token source covers every stylesheet the app actually loads (post-split), not just the boot bundle
- [ ] #2 The residual genuinely-unstyled tokens (~8 at last probe) are fixed or individually allowlisted with reasons
- [ ] #3 The contract goes green at baseline and demonstrably reds on a genuinely unstyled new class (negative control)
<!-- AC:END -->
