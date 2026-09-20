---
id: TASK-32881
title: Align connected sharing and Notes clients with server release contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-20 20:42'
updated_date: '2026-09-20 22:30'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2761'
  - 'https://github.com/rmusser01/tldw_server/pull/2972'
documentation:
  - backlog/decisions/174-server-sharing-release-contracts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair cross-repository Qodo findings 10,11,12 from tldw_server PR2761: canonical clone admission and receipt parsing, paginated shared sources, and version-aware note link deletion. Keep server authorization and concurrency protections intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clone requests send canonical names and stable request idempotency keys and parse durable operations.
- [x] #2 Shared source page envelopes parse and pagination remains accessible.
- [x] #3 Connected note link deletion supports dataset, expected-version and idempotency preconditions.
- [x] #4 Focused HTTP client and service regressions pass.
- [x] #5 Sharing panel retries uncertain clone admissions with the same retained key, including panel remounts; starting another logical clone is explicit.
- [x] #6 PR2763 review follow-up uses production Sharing wiring, shared UI validation, validated page queries, legacy identities, and advancing sparse pages with complete API documentation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/174-server-sharing-release-contracts.md. Reason: cross-module/server contract changes. Stage 1: verify authoritative server wire schemas and add failing real-httpx regressions for clone replay/receipt reads, source pages, and Notes deletion fences. Stage 2: update schemas/client and both Sharing families plus Notes connected wrappers, preserving legacy parsing/list convenience. Stage 3: focused tests, lint/format, scoped Bandit and self-review; record results and prepare reviewable changes without committing/pushing. Reference server approved Docs/superpowers/specs/2026-08-25-shared-workspace-clone-jobs-design.md.
Review found the actual SharingPanel action re-created a key through the service on user retry. Extend the bounded compatibility change with app-lifetime request-key retention per share/name and an explicit Start another clone action; exercise mounted panel -> real scope/service -> httpx timeout/replay. No CSS/token changes.
All three stages complete. Panel follow-up binds keys to configured server/base URL and stable authenticated-user authority plus normalized name/share; maximum 100 retained intents fails closed without eviction. Explicit reset stops event propagation and safely handles invalid IDs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-174 and documented the release contract in backlog/docs/server-sharing-release-client.md. Changes: tldw_api sharing schemas/client/exports; both Sharing and Sharing_Interop service/scope families; Notes connected scope/service deletion forwarding; SharingPanel app-lifetime clone identities and explicit new-copy action; HTTP and mounted-panel regressions; existing Sharing provider tests now use the standard private_profile_test isolation helper. Added the admission-lifecycle incident to lessons-testing-evidence.md. No server auth or concurrency checks weakened. Canonical clone body/header and stable caller keys, full durable receipts/local polling, source pages/legacy aliases and pagination, and exact selected Notes version/dataset/key/reason are preserved.

Verification: final focused command using Chatbook Python3.12 venv and PYTHONPATH=cwd ran Tests/tldw_api/test_sharing_release_contracts.py, test_sharing_client.py, test_notes_workspace_client.py, Tests/Sharing, Tests/Notes/test_server_notes_workspace_service.py, and test_notes_scope_service.py: 143 passed (4.61s), /tmp/qodo-chatbook-final2-tests.log. Red tests captured pre-fix schema/transport and panel replay/scope failures; mounted panel now 2 passed and independently rerun by qodo_transport with no remaining findings (/tmp/qodo-chatbook-independent-review.md). Bandit on all ten touched production Python files: zero findings, zero parse errors (/tmp/qodo-chatbook-bandit-final.json). New test files and changed sharing schema/test module Ruff clean; changed code ranges formatted and new/schema files format --check clean. Exact HEAD-vs-current Ruff comparison across touched tracked Python files: baseline 870/current 868 diagnostics, zero new diagnostics (/tmp/qodo-chatbook-static-comparison.json); existing broad legacy lint debt remains, not falsely reported as a clean whole-scope lint. git diff --check passes.

Verification limits: three pre-existing full ToolsSettings-window sharing tests fail in app_factory/load_settings before panel construction with raw_source_selection_changed. Applying private_profile_test did not fix that broad harness; those unrelated UI test edits were reverted. The real focused mounted-panel tests pass independently. Retained panel keys survive remount/account/server switches but are intentionally in-memory, not process-restart durable; external callers retain request keys/receipts for that lifecycle. No commit/push performed; parent owns publication.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR2763 Qodo follow-up plan: reproduce production panel wiring and null legacy operation identity; centralize UI clone validation and schema-validate source page parameters using exact server bounds; preserve advancing sparse pages and progress guard; expand Google docstrings. Amend ADR174, run focused HTTP/mounted/validator tests, scoped Ruff and Bandit, self-review, then commit/push existing PR after passing checks. Server state filter is free text length1..64, not a closed enum.

PR2763 review follow-up completed: panel consumes production sharing_scope_service/Sharing_Interop with compatibility fallback; shared Utils validator normalizes clone identities; Pydantic source query model enforces actual server bounds (state remains free text); legacy null-operation responses retain real job record IDs; sparse advancing pages continue while stalled cursors fail. Google Args/Returns/Raises sections documented across affected public APIs and handlers; ADR174 and compatibility docs updated. Production wiring tests reproduced failure before repair; 12 behavior cases red, then 165 focused tests passed (4.96s), /tmp/chatbook2763-final.log. Shared validator11 cases pass. Scoped Bandit all11production Python files has0 findings0parseerrors (/tmp/chatbook2763-bandit.json). Ruff baseline876/current876/zero new diagnostics (/tmp/chatbook2763-lint.json); newtests andschema clean, fourfile formatcheck clean, changedranges formatted, diffcheckclean. Separate self-review checked production attribute ownership, actual server free-text state semantics, no transport on invalid input, nonadvancing cursor safety, caller key lifetime, and unchanged Notes version fences. No full suite run. Preparing reviewed commit/push to existing PR2763; no merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved Qodo10/11/12 client compatibility with canonical clone admissions and durable operation receipts, paginated source envelopes and legacy parsing, and version-aware Notes deletion. Added real HTTP and mounted-panel regressions; 143 focused tests passed, scoped Bandit clean, no new Ruff diagnostics versus baseline. ADR-174 and compatibility documentation record ownership, limits, and intentional post-revocation receipt semantics. Independent review closed all findings; publication remains with parent.

PR2763 follow-up closes all eight Qodo findings with production service wiring, shared validation, legacy identity fallback, validated pagination, sparse-page compatibility, and API documentation; 165 focused tests pass and scoped Bandit is clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
