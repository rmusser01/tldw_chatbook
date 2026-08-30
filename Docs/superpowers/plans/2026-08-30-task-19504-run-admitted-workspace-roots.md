# TASK-19504 implementation plan

ADR required: yes
ADR path: `backlog/decisions/102-console-run-admitted-local-path-authority.md`
Reason: the task changes the Console's security authority, tool schemas, and the
disabled-session behavior explicitly retained by ADR-069.

Approved design:
`Docs/superpowers/specs/2026-08-30-task-19504-run-admitted-workspace-roots-design.md`

1. Add failing provider tests for zero, one, mixed, and multiple admitted roots,
   including exact alias schema and call-time revocation behavior.
2. Add the immutable admitted-root contract and minimally extend
   `LocalToolProvider` to route existing path specs and executors by alias.
3. Add failing controller tests for owning-workspace admission, default/bindingless
   schema removal, ADR-069 preservation, and active-workspace/config/CWD isolation.
4. Wire run admission through Console composition without changing standalone MCP
   or built-in sandbox authority.
5. Add Virtual CLI alias/revocation tests and route it through the same admitted
   read roots.
6. Add definition-hash and upgrade-copy coverage, then update the Console guide.
7. Run focused regression suites, changed-file Ruff/format/compile checks,
   diagnostic and diff checks, and final independent review.
