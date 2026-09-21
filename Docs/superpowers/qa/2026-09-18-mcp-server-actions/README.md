# MCP server-action ownership — TASK-32825 / PR2712

The current implementation and qualification are in
[current-dev/README.md](current-dev/README.md), rebuilt from merged PR2716 dev
`9cf5ba67b2396e67c16f3ad7157066b1f7bf740f`.

The saved PR2712 implementation and evidence remain available at immutable
commit [`2cd465c169582ff2a3c4bb0c6d18f4e16590176b`](https://github.com/rmusser01/tldw_chatbook/tree/2cd465c169582ff2a3c4bb0c6d18f4e16590176b/Docs/superpowers/qa/2026-09-18-mcp-server-actions).
Its historical native executable is retired; it used an obsolete repository-root
calculation, CLI admission and terminal warmup. Do not run that historical copy
as current qualification. The replacement uses shared private-profile/argument
validation, explicit module provenance and owned fixture cleanup.
