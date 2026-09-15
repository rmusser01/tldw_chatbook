---
id: TASK-32644
title: >-
  dev is two over the ui-ready module ratchet, and it blinds every PR comparison
status: To Do
assignee: []
created_date: '2026-09-15 10:45'
labels:
  - performance
  - ci
  - adr-097
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`origin/dev` fails `test_ui_ready_module_census_stays_at_the_pinned_size`:
**977 modules resident at `_ui_ready` against ADR-097's
`MAX_TLDW_MODULES_AT_UI_READY = 975`**. Three modules arrived since the
snapshot was pinned and one was shed:

- `+ tldw_chatbook.Agents.approval_provenance`
- `+ tldw_chatbook.Notes.note_import_parsers`
- `+ tldw_chatbook.Widgets.select_values`
- `- tldw_chatbook.Widgets.Console.console_run_log_modal`

None of them look accidental: `approval_provenance` is imported at module
level by six `Agents/` providers and by `console_chat_controller`,
`select_values` by `settings_screen` and `console_model_popover`, and
`note_import_parsers` by `notes_sync_runtime` and
`library_notes_lasting_sync_state`. They are on the boot path because
features moved onto the boot path, not because someone forgot a deferral.

**The reason this is filed rather than lived with.** A ratchet that is red at
the base makes every PR's copy of it unreadable at a glance, and the standing
practice is to wave a shared red through. That went wrong on PR #2691, which
was red on both sides at **977 on dev and 980 on the branch** — three of the
six were the branch's own, and only reading the numbers caught it. Every PR
opened against this dev inherits the same trap.

ADR-097 is explicit that raising the constant is not one of the options. The
two that remain are an owner call, which is why this is a task and not a
drive-by fix: **(a)** defer or shed enough to get back to 975, or **(c)** an
explicit owner exception recorded in the ADR's exception ledger, naming these
three modules and why the boot path now legitimately needs them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `test_ui_ready_module_census_stays_at_the_pinned_size` passes on `origin/dev`, by deferring/shedding or by a recorded ADR-097 exception — not by raising the constant.
- [ ] #2 If the route is an exception, the ADR's ledger names all three modules individually and says what put each on the boot path, so the next breach is still legible.
- [ ] #3 `boot_budget_snapshots/ui_ready_modules.txt` matches whatever is decided, so the delta list a future PR sees is its own.
- [ ] #4 The three modules are re-derived at the time of the fix — the set will have moved again.
<!-- AC:END -->
