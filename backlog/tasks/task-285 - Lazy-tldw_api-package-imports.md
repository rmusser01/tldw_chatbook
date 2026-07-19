---
id: TASK-285
title: Lazy tldw_api package imports (~469ms of startup)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_api/__init__.py eagerly imports 54 schema files = 1,313 Pydantic models, ~469ms (31% of the ~1.5-2.0s app import), forced by app.py:353 plus 69 files importing names from the package. Fix: PEP 562 lazy __getattr__ re-exports (working, documented pattern in Local_Ingestion/__init__.py) and/or Server*Service modules importing their own schema submodules. Longer-term note: TldwCli.__init__ constructs ~30 Server*Service objects even in local-only mode. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 import tldw_chatbook.app no longer pays the full tldw_api package cost (measured importtime delta reported, expected >300ms)
- [ ] #2 All existing consumers keep working (full test suite green; no import cycles)
- [ ] #3 Remote-server mode still functions (schema names resolve on demand)
<!-- AC:END -->
