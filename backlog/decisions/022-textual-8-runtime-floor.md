# ADR-022: Bound the supported Textual runtime to version 8

Status: Accepted
Date: 2026-07-23
Related Task: [backlog/tasks/task-400 - Fix-MCP-navigation-crash-by-requiring-Textual-8.md](../tasks/task-400%20-%20Fix-MCP-navigation-crash-by-requiring-Textual-8.md)
Supersedes: N/A

## Decision

Chatbook supports Textual 8.x, expressed as `textual>=8.0.0,<9` in the
authoritative package metadata and mirrored in the development requirements
file. A focused CI lane installs exactly Textual 8.0.0 and mounts the MCP
workbench and tools mode so the declared minimum remains executable.

Support for a later Textual major version requires an explicit compatibility
review and dependency-bound update.

## Context

The package declared `textual>=3.3.0`, but production UI code uses
`Select.NULL`. Textual 3.3.0 exposes `Select.BLANK` instead, causing MCP
navigation to crash while composing the tools mode. Textual 8.0.0 introduced
the `Select.NULL` name as a breaking change.

The repository's normal environment runs Textual 8.2.7. An isolated
pre-implementation replay against exactly Textual 8.0.0 passed all 182 MCP
workbench and tools-mode tests, validating 8.0.0 as the minimum for the
reported path.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Support Textual 3–7 with a `Select.NULL` compatibility alias | Mutates third-party API state and expands the supported matrix for no product benefit. |
| Repair only the first MCP `Select.NULL` call site | Leaves numerous MCP and non-MCP uses exposed and keeps false package metadata. |
| Declare `textual>=8.0.0` without an upper bound | Resolver behavior would claim support for unreviewed future major versions, risking another breaking runtime upgrade. |
| Pin exactly Textual 8.2.7 everywhere | Prevents compatible bug-fix and feature releases within Textual 8 and is stricter than the verified API requirement. |

## Consequences

- Fresh installs and upgrades reject Textual releases older than 8.0.0 and
  Textual 9 or newer.
- Existing source checkouts must reinstall dependencies after pulling the
  changed metadata.
- Both `pyproject.toml` and `requirements.txt` carry the same supported range
  because CI and documented developer flows use both installation paths.
- CI pays for one focused minimum-version MCP test lane.
- Future Textual-major adoption is fail-closed until explicitly reviewed.

## Links

- Spec: Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md
- Plan: Docs/superpowers/plans/2026-07-23-mcp-textual-runtime-floor.md
