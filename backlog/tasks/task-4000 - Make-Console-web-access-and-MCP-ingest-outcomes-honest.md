---
id: TASK-4000
title: Make Console web access and MCP ingest outcomes honest
status: Done
assignee: []
created_date: '2026-08-09 19:25'
updated_date: '2026-08-09 19:43'
labels: []
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure Console agents can discover and use the intended web research tools through an understandable registration path, and prevent the advertised MCP media-ingest tool from reporting a queued success when no ingestion work is performed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console users can enable the standard web_search, web_fetch, and web_crawl tools from the MCP management surface with clear restart and scope guidance.
- [x] #2 A Console agent is not offered an ingest_media tool that returns a fabricated queued success without submitting real work.
- [x] #3 The standalone and in-process MCP capability surfaces agree about whether ingest_media is available and never claim success for an unimplemented operation.
- [x] #4 Focused automated tests cover Console catalog composition, MCP capability inventory, and direct runtime behavior.
- [x] #5 User documentation explains the web-tool gate, permissions, restart boundary, and the supported route for persistent web ingestion.
- [x] #6 The PR leaves the Backlog duplicate-ID guard green by reconciling the pre-existing TASK-3401 collision exposed by the latest dev base.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the Console catalog path and the two MCP ingest handlers with focused regression tests.
2. Make the existing local-tools master gate explicitly disclose the standard web tools, scope, and restart requirement.
3. Remove the unimplemented ingest_media advertisement and dispatch path from both local MCP runtimes.
4. Document the distinction between ephemeral web research and persistent Library URL ingestion.
5. Run focused tests, lint the changed Python files, and review the final diff.

ADR required: no
ADR path: N/A
Reason: This is a routine truthfulness and discoverability bug fix that preserves the existing tool-provider and ingestion boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed and expanded the built-in local-tools gate copy so the MCP Servers surface, Permissions breadcrumb, and documentation explicitly identify web_search, web_fetch, and web_crawl, their Console-only scope, and the restart boundary. Removed the unimplemented ingest_media registration, runtime handler, and action mapping so neither local MCP surface can fabricate a queued result; persistent URL ingestion remains the Library Import workflow.

Added regression coverage for Console catalog composition, mounted Textual gate guidance, the AST-derived MCP manifest, and direct-runtime rejection. Focused changed-case tests passed (5), mounted UI tests passed (4), the complete gate suite passed (33), and Ruff passed all changed Python files. A broader related run passed 216 tests and exposed one pre-existing Windows HOME/expanduser test isolation failure unrelated to these changes.

ADR required: no. No provider, persistence, or service boundary changed. No new lessons entry was needed; the repository's existing testing-evidence guidance already covers fabricated success responses.

Post-review follow-up: rebased onto the latest dev, added the requested test docstrings, made breadcrumb gate reads single-pass and snapshot-consistent, and renumbered the smaller pre-existing Console rail task from TASK-3401 to TASK-14650 so the Backlog guard no longer collides with the established TASK-3401 video epic.

Post-review verification: 34 built-in gate tests, 108 MCP tests, 75 Console/provider tests (with the known Windows HOME isolation case deselected), and 4 mounted MCP UI tests passed. Ruff passed every changed Python file, and the CI-equivalent scan found no duplicate filename or frontmatter IDs across 1,658 Backlog task files.

The first post-rebase minimum-Textual CI run then exposed one additional stale Tools-mode inventory expectation for the retired ingest_media entry: 264 tests passed and that single assertion failed. The expectation was corrected and added to the focused verification set before the next push.
<!-- SECTION:NOTES:END -->
