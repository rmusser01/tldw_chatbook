---
id: TASK-367
title: Make endpoint URL entry forgiving - normalize v1 and completion-path variants
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three contradictory URL shapes in one flow: (1) selecting llama.cpp prefills 'http://localhost:8080/completion'; (2) Discover models with a working server at bare-origin http://127.0.0.1:9099 was refused with 'This endpoint is not OpenAI-compatible for v1 discovery. Configure a /v1 endpoint to discover models.' — meaning the app's own prefilled '/completion' default would also fail discovery; (3) with a realistic typo 'ttp://127.0.0.1:9099/v1' (dropped h), the app returned the SAME '/v1' message even though the URL already ends in /v1 — the real problem (invalid scheme) is never surfaced, and no inline validation flags the malformed URL. Only after typing the exact 'http://.../v1' form did discovery succeed (~0.8s, j1-24).

**Repro:** Provider=llama.cpp (note /completion default) -> Endpoint=http://127.0.0.1:9099 -> Discover models -> '/v1' refusal; set Endpoint='ttp://127.0.0.1:9099/v1' -> Discover -> same '/v1' refusal; fix to http://127.0.0.1:9099/v1 -> succeeds.

**Verifier note:** Code-confirmed on a post-ledger surface (Settings discovery shipped 1fd5f5f0c, 2026-07-11, tasks 188/191-console). _discovery_status_from_error flattens every unsupported_endpoint kind into the single '/v1' copy (settings_screen.py:4958 → MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY:218), discarding the client's distinct messages; supports_openai_compatible_model_discovery (openai_compatible_model_discovery.py:224-249) returns False both for a malformed scheme (parse→None, e.g. 'ttp://…/v1') and for llama.cpp's native /completion path — so the app's own prefilled default fails its own discovery and a scheme typo is misdiagnosed as a /v1-path problem. No inline URL validation on the field. Nothing in the ledger covers discovery UX. P2: first-run guidance actively contradicts itself, but manual model entry and the Console-modal discover path remain.

**Source:** Console UX expert review 2026-07-20 (finding j1-endpoint-v1-guessing; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-12-after-provider-switch-settle.png`, `j1-21-discover-good-result.png`, `j1-23-endpoint-check.png`, `j1-24-discover-success.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prefill a default that works with the app's own features
- [ ] #2 Auto-append or auto-probe /v1 instead of demanding the user reformat
- [ ] #3 Distinguish 'malformed URL' from 'missing /v1 path'
- [ ] #4 Validate the URL inline on blur
<!-- AC:END -->
