---
id: TASK-2542
title: _toast helper is duplicated between mcp_inspector.py and mcp_workbench.py
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - cleanup
dependencies: []
priority: low
---

## Description

`mcp_workbench.py:124` and `mcp_inspector.py:195` (the latter added by PR-T3 Task 3)
each define their own, byte-identical one-line helper:

```python
def _toast(text: str) -> str:
    return escape_markup(text)
```

The duplication is deliberate and correct as shipped, not an oversight: `mcp_
workbench.py` already imports FROM `mcp_inspector.py` (`_ORIGIN_SENTENCES`,
`MCPInspector`), so importing `_toast` the other way (`mcp_inspector.py` importing
from `mcp_workbench.py`) would create the exact import-cycle shape PR-T2 shipped a
real regression from. `mcp_inspector.py`'s own copy is commented to explain exactly
this.

Still, two copies of the same one-liner in the same module family can drift silently
(e.g. one gains a new escaping rule the other doesn't). A small shared leaf module —
with no import-direction conflict against either file — would let both sides use one
definition.

## Acceptance Criteria

- [ ] `_toast()`'s implementation exists in exactly one place: a small shared module
      with no dependents that would recreate an import-direction conflict with either
      `mcp_inspector.py` or `mcp_workbench.py`.
- [ ] Both `mcp_inspector.py` and `mcp_workbench.py` import it from that shared
      location instead of defining their own copy.
- [ ] No import cycle introduced — verified the same way PR-T3 Task 2 verified its own
      new cross-module import (both trigger orders, in fresh processes, plus a
      standalone narrow `--collect-only` sweep).
- [ ] All existing `_toast`-dependent tests keep passing unmodified (the function's
      behavior does not change, only its location).
