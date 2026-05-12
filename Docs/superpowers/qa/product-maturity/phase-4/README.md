# Product Maturity Phase 4 QA

Status: planned; child tasks defined

Phase 4 covers agent configuration and execution surfaces: Personas, Skills, MCP, ACP, Schedules, and Workflows. This phase is not verified until the running app has been walked through and actual screenshots are captured for every changed visible screen.

## Evidence Index

| Slice | Evidence | Status |
| --- | --- | --- |
| Phase 4 planning | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-planning.md` | planned |

## QA Rules

- Use actual running-app screenshots for UI approval. Do not use rendered SVGs, static mockups, or code-only layouts.
- Verify both first-use comprehension and power-user control paths where the screen supports repeated runs.
- Record focused regression commands, `git diff --check`, and any skipped or blocked workflow with a concrete reason.
- Keep blocked ACP/server-parity states honest until runtime or sync support exists.
