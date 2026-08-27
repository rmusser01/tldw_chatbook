---
id: TASK-23020
title: >-
  trajectory_export is back on the Chat first-paint leg - two guards RED on dev
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - startup
  - regression
  - dev-red
priority: high
---

## Description

`Chat.trajectory_export` is resident at `_ui_ready` again, breaking a guarantee that shipped ~24
hours earlier. Two guards are red on pristine dev, so every branch inherits them.

Three module-scope edges reach it from the Chat mount leg, each importing **one name**
(`TraceExportProfile`, a three-member `str, Enum`) that drags 1,463 LOC plus `Chat.trajectory`.
**Fixing one edge buys nothing** — all must break.

`chat_screen.py:52-57` carries a comment explicitly forbidding this; the change routed around it
through a file the comment does not name.

## Acceptance Criteria

- [ ] `Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size` passes
- [ ] `Tests/Packaging/test_rag_boot_import_closure.py::test_chat_screen_import_does_not_execute_the_deferred_packages` passes
- [ ] All three edges are broken, verified by an import tracer recording `(importer, imported)` — not by grep
- [ ] The export dialogs still work; a test drives them from the deferred state
- [ ] Neither guard is relaxed to accommodate the regression
- [ ] The guard names the offending edge well enough that the next person does not have to trace it

## Evidence

```
UI/Screens/chat_screen.py:448
  -> Widgets/Console/console_conversation_inspector.py:114
    -> Widgets/Console/console_exchange_export_dialog.py:22 -> Chat/trajectory_export.py
                                     :25 -> Widgets/Console/trace_export_dialog.py:16,17
    -> Chat/console_exchange_export.py:18 -> Chat/trajectory_export.py
```

Introduced by `c6218918d1` (#2126). Arrives on the **mount** leg, not the import leg, which is why
the import-weight guard stays green. Import self-time 1.67-1.98 ms; the cost is the contract, not
the milliseconds.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
