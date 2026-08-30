---
id: TASK-24459
title: Pay the boot parsed CSS byte ratchet breach
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - boot
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`MAX_BOOT_PARSED_CSS_BYTES` is breached: boot-parsed CSS is 862,184 B against a limit of
860,000 B. Every one of those bytes is parsed before first paint.

Growth since the pinned snapshot: `tldw_cli_modular.tcss` +3,424 B (of which
`components/_agentic_terminal.tcss` is +2,156 B and `components/_forms.tcss` +881 B),
`widget_defaults_scoped.tcss` +3,399 B, `widget_defaults_self.tcss` +641 B, including two new
`ConsoleForkChatModal` segments totalling 2,770 B.

Per ADR-097 the constant must not be raised.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `test_boot_parsed_css_bytes_stay_within_budget` passes on a pristine checkout
- [ ] #2 `MAX_BOOT_PARSED_CSS_BYTES` is not raised
- [ ] #3 The bytes are shed by deferring CSS off the first-paint path or by removing redundant rules, not by moving the measurement
- [ ] #4 The CSS bundle regenerates from its sources with no drift
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
NOT IMPLEMENTED in the 2026-08-29 review pass. This is the one ratchet of the four still red:
862,184 B against an 860,000 B limit, so 2,184 B must be shed and the constant must not rise.

Because it is red, task-24461 deliberately EXCLUDED this guard when wiring the boot budgets into
`perf-guard.yml` -- including it would have failed every unrelated PR. It joins that step when
this task lands, and the workflow comment says so.

Growth since the pinned snapshot: `tldw_cli_modular.tcss` +3,424 B (of which
`components/_agentic_terminal.tcss` +2,156 and `components/_forms.tcss` +881),
`widget_defaults_scoped.tcss` +3,399, `widget_defaults_self.tcss` +641, including two new
`ConsoleForkChatModal` segments totalling 2,770 B.

Most tractable route is the `ConsoleForkChatModal` segments plus the `_agentic_terminal.tcss`
growth; it overlaps task-24451. A mechanical dead-selector sweep was attempted and abandoned --
the detection used `\w` in a POSIX ERE and reported every one of 609 ids as dead, which is
exactly the kind of false positive that would have deleted live CSS.
<!-- SECTION:NOTES:END -->
