# MCP Textual Runtime Floor — Design

**Status:** approved by the user on 2026-07-23; review amendments approved on 2026-07-23
**Date:** 2026-07-23
**Related task:** TASK-503

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

Support Textual 8.x, beginning at 8.0.0.

- Change the authoritative dependency in `pyproject.toml` to
  `textual>=8.0.0,<9`.
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
installing this application version. Textual 9 and later remain unsupported
until a deliberate compatibility review raises the upper bound.

## Regression design

Add a focused packaging-contract test that parses the Textual requirement from
both `pyproject.toml` and `requirements.txt` and proves:

1. Textual 7.x does not satisfy it.
2. Textual 8.0.0 does satisfy it.
3. Later Textual 8.x releases satisfy it.
4. Textual 9.x does not satisfy it.

The test must fail against the current `textual>=3.3.0` declaration before the
metadata is changed. Update the existing Phase 6 packaging assertion that
currently hard-codes `textual>=3.3.0`; leaving it stale would make the broader
suite fail after the dependency correction.

Add a focused CI lane that installs exactly Textual 8.0.0 and runs the packaging
contract plus the MCP workbench and tools-mode suites. This preserves the
minimum-version claim as later application code evolves instead of testing only
against the newest resolver-selected Textual release.

Existing mounted MCP tests remain the behavioral proof that the supported
runtime can compose and navigate the destination without the reported
`Select.NULL` crash. A pre-implementation review run against an isolated
Textual 8.0.0 install passed all 182 MCP workbench and tools-mode tests; two
pre-existing unawaited-coroutine warnings remain outside this dependency fix.

## Alternatives considered

| Option | Reason rejected |
| --- | --- |
| Alias `Select.NULL` to `Select.BLANK` for Textual 3–7 | Mutates a third-party widget API globally and commits the project to testing a broad legacy range. |
| Replace only MCP `Select.NULL` references with a compatibility helper | Leaves non-MCP references exposed and spreads framework-version branching through UI code. |
| Catch the exception during MCP navigation | Hides a packaging defect and leaves users with an unusable destination. |
| Leave the requirement open-ended as `textual>=8.0.0` | Claims compatibility with unreviewed future major versions and risks another resolver-installed breaking change. |

## Verification

1. Observe the new dependency-floor test fail before the metadata edit.
2. Update the two dependency declarations and rerun the test to green.
3. Update the existing packaging assertion and add an Unreleased changelog note.
4. Add and contract-test the minimum-version CI lane.
5. Run the focused MCP workbench and tools-mode tests on the normal runtime.
6. Run the same focused suites with exactly Textual 8.0.0.
7. Run packaging/configuration tests that inspect `pyproject.toml`.
8. Run `git diff --check`.

## Scope boundaries

This repair does not redesign the MCP screen, change MCP service behavior, or
modify persistence. Compatibility is deliberately bounded to Textual 8.x;
raising that major-version ceiling requires a separate compatibility review.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/022-textual-8-runtime-floor.md`

Reason: raising the minimum supported framework version changes the project's
dependency and runtime policy and deliberately ends support for Textual 3–7.
