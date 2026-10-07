---
id: TASK-33507
title: Add Cloudflare Workers AI (REST API) as an engine preset
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:28'
updated_date: '2026-09-29 20:03'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cloudflare recommends its api.cloudflare.com REST endpoint for new integrations; it serves Workers AI models (and third-party models on unified billing) over OpenAI Chat Completions with a per-account URL and a Cloudflare API token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user who sets their account URL and API token can chat with seeded Workers AI models through the engine
- [x] #2 An optional AI Gateway id is sent as the documented header when set, and nothing is sent when it is unset
- [x] #3 The Settings guide documents setup, and why the gateway.ai.cloudflare.com compat endpoint is not used
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. CLOUDFLARE record: user-supplied account URL (full .../ai/v1), Bearer API token, seeded @cf models (no models route), cf-aig-gateway-id <- gateway_id via config_headers.
2. Readiness base URL required with a Cloudflare example.
3. Hand lists, config, tests, Settings guide incl. why not the gateway.ai.cloudflare.com compat endpoint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CLOUDFLARE record (key cloudflare): Workers AI REST API (developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility), Bearer CLOUDFLARE_API_TOKEN (Cloudflare's own env convention), default_base_url None -- the account id is in the path, so the user sets the full .../accounts/<id>/ai/v1 URL (no suffix; readiness "Missing account URL" with that example). No models route (GET 405), so ten verified @cf/... models ship seeded. cf-aig-gateway-id <- [api_settings.cloudflare] gateway_id via config_headers. Reasoning_content private (proprietary). The gateway.ai.cloudflare.com compat endpoint is deliberately not used (two credential headers) -- documented in the registry and Settings guide.
<!-- SECTION:NOTES:END -->
