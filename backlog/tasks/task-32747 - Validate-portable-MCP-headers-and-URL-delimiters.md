---
id: TASK-32747
title: Validate portable MCP headers and URL delimiters
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 18:24'
updated_date: '2026-09-17 18:37'
labels:
  - plugins
  - compatibility
dependencies:
  - TASK-32668
  - TASK-32680
documentation:
  - Docs/superpowers/plans/2026-09-15-managed-plugins-delivery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Agent Plugins 1.0.0 requires invalid remote MCP server configurations to be
identified during package inspection without disabling valid siblings. The
current inspector accepts several prohibited header and URL forms. Correct
those confirmed syntax gaps and document the boundary between portable
inspection, Chatbook host policies and actual runtime qualification.

Implements the accepted portable-core contract in backlog/decisions/162-managed-agent-plugins.md and Docs/superpowers/specs/2026-09-15-managed-plugins-design.md. Agent Plugins 1.0.0 reconciliation follows the user-supplied https://agent-plugins.org/ reference.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspection marks a remote MCP entry invalid for case-insensitive duplicate header names, forbidden ASCII controls or DEL in values, and leading/trailing SP or HTAB. Empty values and visible-bounded values with interior SP/HTAB stay valid and retain their exact text.
- [x] #2 Remote URLs containing userinfo or fragment delimiters are invalid even when those components are empty. Valid URLs with literal `@` in the path or an encoded `#` in a path/query remain valid when they satisfy existing host policy.
- [x] #3 Each invalid entry leaves a valid remote sibling and a valid portable skill inspectable. The repair preserves fixed discovery locations, unknown-field behavior, dependency guards and existing package safety boundaries.
- [x] #4 Authoring/qualification documentation cites the versioned portable contract and distinguishes stricter acquisition/HTTP-loopback policies, optional legacy SSE, and unqualified runtime/Unicode-to-wire behavior from parser evidence. The repair does not silently trim, expand or re-encode headers.
- [x] #5 Actual package-inspection RED/control tests and targeted native-inspection and package-file regressions pass, with changed-file static checks and an independent review. No full-suite or live external-server qualification is inferred.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/162-managed-agent-plugins.md
Reason: Direct correction of the accepted portable-core parser contract, with no storage, runtime, credential, permission or ownership change.

1. Read task-26-brief.md, the actual Backlog task, accepted plugin spec/ADR162, agent-plugins-standard-reconciliation.md and portable-parser-repair-preparation.md. Recheck current parser and actual inspect_package entry after H5; completed standards research is available and does not need repeating. Controller supplies BASE and frozen baseline paths before implementation.
2. Modify only tldw_chatbook/Plugins/schemas.py, Tests/Plugins/test_native_inspection.py and tldw_chatbook/Plugins/README.md. Tests/Plugins/fixtures/PROVENANCE.md may receive a factual dated contract-provenance amendment only if needed. Read-only regression Tests/Plugins/test_package_files.py and shared fixture/inspection/portable adapter remain unchanged. Escalate any additional edit path or contract uncertainty before code.
3. Establish genuine RED through inspect_package for case-insensitive duplicate headers, prohibited ASCII controls/DEL/edge SP or HTAB, and userinfo or fragment presence including empty delimiters. Each invalid server must preserve a valid remote sibling and portable skill. Add positive controls for empty header values, visible-bounded interior SP/HTAB, literal @ in a URL path and encoded # in path/query. Keep exact header strings unchanged and no variable expansion or runtime transport execution.
4. Apply minimal syntax validation. Do not invent a blanket JSON Unicode encoding restriction from RFC9110 obs-text octets; actual transport owners qualify encoding later. Preserve stricter existing loopback/acquisition gates, fixed discovery, unknown-field handling and dependency blockers. Update existing authoring/qualification documentation with primary-source links and precise optional legacy-SSE versus Streamable-HTTP-SSE/runtime evidence boundaries.
5. Format before final qualification. Run exactly Tests/Plugins/test_native_inspection.py and Tests/Plugins/test_package_files.py using the worktree interpreter/private profile, repository bootstrap, null keyring and network refusal. Capture exact interpreter/source/data provenance, actual RED/GREEN commands/output, before/after/current source/test/doc hashes, syntax/Ruff/Black baseline parity and whitespace. No full sweep, fixture refetch, external server or clean-resource claim inferred from warning absence.
6. Self-review and report to task-26-report.md with changed paths, exact commands/results, compatibility limits and M2/M3/M4 handoff. Controller verifies, commits exact paths, obtains independent spec/quality review, resolves findings, then uses CLI to check all ACs, add notes and mark Done. Do not mutate Git or Backlog yourself; no helper/reviewer subagents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected remote MCP inspection under the accepted Agent Plugins 1.0.0 contract (ADR162). Headers now reject case-insensitive duplicates, prohibited ASCII controls/DEL and edge SP/HTAB while preserving valid strings. URLs reject userinfo and fragment delimiters even when empty; valid path @ and encoded # remain allowed. Each invalid entry leaves valid remote servers and skills inspectable. Existing authoring documentation states the stricter HTTP/acquisition policy, legacy-SSE distinction and runtime qualification limits.

Changed Plugins/schemas.py, Tests/Plugins/test_native_inspection.py and Plugins/README.md. The delivery plan records this bounded follow-up between H5 and M1. Actual inspect_package RED reproduced11invalid cases with2passing controls; final targeted native-inspection/package-file run92passed in2.92s. Syntax, Ruff, Black(py312) and whitespace pass. Three final changed hashes,11unchanged qualification paths,14exactBASEsnapshots and private-runner/output hashes were verified; a later README wording clarification changed no runtime/test bytes. No full suite or external service execution.

Independent review approved spec and quality with no findings. Its unchanged-code evidence boundary is covered by the retained baseline hashes and existing regression cases. No new ADR was required; ADR162 remains governing. Arbitrary JSON Unicode header encoding and real transport behavior remain M2/M3/M4 qualification work. Implementation commit d1632600ab7068f625c06a88e1287f5fb60442fa. No new generalizable incident required a lessons entry.
<!-- SECTION:NOTES:END -->
