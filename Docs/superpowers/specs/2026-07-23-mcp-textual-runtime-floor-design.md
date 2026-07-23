# MCP Textual Runtime Floor — Design

**Status:** approved direction; written-spec review pending
**Date:** 2026-07-23
**Related task:** TASK-400

## Goal

Prevent the application from crashing when a user navigates to the MCP
destination after installing a Textual release that predates APIs already used
throughout the MCP UI.

## Verified failure

The package currently declares `textual>=3.3.0`, while the MCP workbench uses
`Select.NULL`. Textual 3.3.0 exposes the older `Select.BLANK` sentinel and raises
`AttributeError: type object 'Select' has no attribute 'NULL'` while composing
the MCP tools mode. Textual 8.0.0 introduced the `Select.NULL` name, and the MCP
screen mounts successfully on the repository's current Textual 8 runtime.

The same Textual 3.3.0 mount succeeds when `Select.NULL` is temporarily aliased
to `Select.BLANK`, isolating the crash to the falsely broad dependency contract
rather than MCP data, configuration, or navigation state.

## Decision

Raise the supported Textual runtime floor to 8.0.0.

- Change the authoritative dependency in `pyproject.toml` to
  `textual>=8.0.0`.
- Mirror the floor in the development-convenience `requirements.txt` so the two
  documented installation paths do not communicate different support ranges.
- Do not add a compatibility alias or conditional sentinel helper. The
  application already contains many `Select.NULL` references beyond the first
  crashing MCP compose path, so a local workaround would preserve a misleading
  package contract.
- Do not add an application startup guard. Standard dependency resolution is
  the enforcement boundary; duplicating it at runtime would add an unreachable
  error path for correctly installed packages.

Existing environments on Textual 3 through 7 must upgrade Textual before
installing this application version.

## Regression design

Add a focused packaging-contract test that parses the project's Textual
requirement and proves:

1. Textual 7.x does not satisfy it.
2. Textual 8.0.0 does satisfy it.

The test must fail against the current `textual>=3.3.0` declaration before the
metadata is changed. Existing mounted MCP tests remain the behavioral proof
that the supported runtime can compose and navigate the destination without
the reported `Select.NULL` crash.

## Alternatives considered

| Option | Reason rejected |
| --- | --- |
| Alias `Select.NULL` to `Select.BLANK` for Textual 3–7 | Mutates a third-party widget API globally and commits the project to testing a broad legacy range. |
| Replace only MCP `Select.NULL` references with a compatibility helper | Leaves non-MCP references exposed and spreads framework-version branching through UI code. |
| Catch the exception during MCP navigation | Hides a packaging defect and leaves users with an unusable destination. |

## Verification

1. Observe the new dependency-floor test fail before the metadata edit.
2. Update the two dependency declarations and rerun the test to green.
3. Run the focused MCP workbench and screen tests.
4. Run packaging/configuration tests that inspect `pyproject.toml`.
5. Run `git diff --check`.

## Scope boundaries

This repair does not redesign the MCP screen, change MCP service behavior,
modify persistence, or claim compatibility with every future Textual major
release. An upper bound can be introduced later only in response to an observed
incompatibility.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/022-textual-8-runtime-floor.md`

Reason: raising the minimum supported framework version changes the project's
dependency and runtime policy and deliberately ends support for Textual 3–7.
