---
id: TASK-24653
title: Network TLS trust policy (corp DPI)
status: Done
assignee: []
created_date: '2026-08-29 22:51'
updated_date: '2026-08-30 00:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Corporate TLS-inspection networks break every HTTPS call because no transport consults the OS trust store and no setting expresses trust. Add one global ternary [network] ssl_verify with additive custom-CA semantics, plumbed through a shared helper and every in-scope outbound client.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Config [network] ssl_verify ternary normalizes leniently and fails safe to verification-on
- [x] #2 tls_trust helper tests cover coercion, additive contexts, merged bundle, and factories
- [x] #3 Shared httpx seams, requests long tail, aiohttp, websockets, and the OpenAI-SDK site honor the policy
- [x] #4 tldw_api client exposes an ssl_verify constructor param and the app bootstrap passes the policy
- [x] #5 F9 Network category saves valid values, rejects bad paths, and warns when verification is relaxed
- [x] #6 ADR-079 records the decision and rejected alternatives
<!-- AC:END -->

## Implementation Plan

Followed Docs/superpowers/plans/2026-08-29-network-tls-trust-policy.md (Tasks 1-10, in order; Task 10 was the final sweep + close-out). Decision record: backlog/decisions/079-network-tls-trust-policy.md.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per Docs/superpowers/plans/2026-08-29-network-tls-trust-policy.md (Tasks 1-10 complete). Helper: tldw_chatbook/Utils/tls_trust.py (normalization + additive contexts + merged CA bundle + httpx/requests factories + warn-once + metrics). Adoption: shared httpx seams (console gateway/TTS/image-gen/provider catalog/evals capture/local-server discovery/web tools), tldw_api client ssl_verify ctor param wired from bootstrap (ssl_verify=httpx_verify()), requests long tail (LLM_API_Calls/hosted_chat/qwencloud/Local_Summarization_Lib/Summarization_General_Lib/WebSearch_APIs), aiohttp (Article_Scraper crawler 2x, swarmui_client), websockets (realtime transport; ws:// never passes ssl, wss:// uses a CERT_NONE context per the websockets>=14 contract amendment), OpenAI SDK (OCR_Backends http_client=build_httpx_client). UI: F9 Network category (verify/off/custom-CA select + CA path + relaxed-verification warnings). ADR: backlog/decisions/079-network-tls-trust-policy.md. Spec: Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md.

Verification (Task 10 close-out): targeted sweep of the seven touched test files = 53 passed, 0 failed; completeness greps confirm every executable requests.Session() site is wired (LLM_API_Calls 14 after excluding one comment match, hosted_chat 1, qwencloud 1, Local_Summarization_Lib 15, Summarization_General_Lib 16, WebSearch_APIs 3), crawler 2 + swarmui 1 aiohttp seams, realtime transport 1, OCR 1, and every shared httpx seam file consults tls_trust (bootstrap via httpx_verify() into the tldw_api ssl_verify param); ruff clean over all touched source files.

Lessons decision: no lessons entry — nothing generalized beyond this feature.
<!-- SECTION:NOTES:END -->

## Diagnostic inventory review record (PR #2223, 2026-08-30)

`Utils/tls_trust.py` became a TASK-494 diagnostic owner (6 calls) and one
path-privacy candidate (`tls_verify_setting` error interpolating
`str(path)` — the user's own `[network] ssl_verify` config value).
Reviewed per the guard's procedure (`--statements` output read in full):
no statement interpolates secrets, conversation content, or content-derived
URLs; the only interpolations are the user's own config path/type names,
present for actionable remedy text, emitted at error level on
misconfiguration. Accepted; manifest regenerated with
`check_persistent_diagnostic_inventory.py --write`. The three
`test_summarization_diagnostic_privacy.py` failures are pre-existing on
dev (fixture pins an older projection; failed at this branch's base before
any edit) and remain with the diagnostic-inventory owner.

## Renumbering provenance

This task previously held id TASK-21513, colliding with the older
"Daily Reports surface and demo" task (created 2026-08-29 22:08; this task
arrived 2026-08-29 22:51). Per the owner rule decided 2026-08-21 in
TASK-19601 (**older id keeps it; the younger task renumbers with a
provenance note, regardless of Done status**), it renumbered to TASK-24653
(verified free: dev's task-id ceiling was 24652). Citations to TASK-21513
in already-merged commit messages or docs written before this renumber
refer to THIS task; the other TASK-21513 holder is
"backlog/tasks/task-21513 - Daily-Reports-surface-and-demo.md".
