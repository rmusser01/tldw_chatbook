# Product Maturity Phase 4 QA

Status: TASK-11.1 through TASK-11.6 verified; implementation slices remain open

Phase 4 covers agent configuration and execution surfaces: Personas, Skills, MCP, ACP, Schedules, and Workflows. This phase is not verified until the running app has been walked through and actual screenshots are captured for every changed visible screen.

## Evidence Index

| Slice | Evidence | Status |
| --- | --- | --- |
| Phase 4.1 planning baseline | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-planning.md` | verified |
| Phase 4.2 Personas runtime launch | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md` | verified |
| Phase 4.3 Skills attach validation | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md` | verified |
| Phase 4.4 MCP source scope | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md` | verified |
| Phase 4.5 ACP runtime session | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-5-acp-runtime-session.md` | verified |
| Phase 4.6 Schedules and Workflows run control | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md` | verified |

## QA Rules

- Use actual running-app screenshots for UI approval. Do not use rendered SVGs, static mockups, or code-only layouts.
- Verify both first-use comprehension and power-user control paths where the screen supports repeated runs.
- Record focused regression commands, `git diff --check`, and any skipped or blocked workflow with a concrete reason.
- Keep blocked ACP/server-parity states honest until runtime or sync support exists.
