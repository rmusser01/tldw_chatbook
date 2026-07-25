---
id: TASK-498
title: >-
  Adopt image-generation egress/SSRF policy (Utils/egress.py)
status: To Do
assignee: []
created_date: '2026-07-22 11:32'
updated_date: '2026-07-24 16:45'
labels:
  - image-generation
  - security
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RESCOPED 2026-07-24: dev now ships its own SSRF policy — `Utils/egress.py` (PR #822: per-hop redirect revalidation, private/link-local/metadata blocking, cross-origin cred-strip, byte caps) — so this task is no longer a port from tldw_server. Instead, ADOPT `Utils/egress.py` inside `tldw_chatbook/Image_Generation/http_client.py`, replacing the Phase-1 light guard (`_validate_egress_or_raise` scheme-check placeholder). The original intent stands: image URLs RETURNED by remote backends (OpenRouter/Novita/ModelStudio) must be validated before fetch, while user-configured local backend base_urls (127.0.0.1 SwarmUI, local sd.cpp) keep working. Also wire ModelStudio's dead `allowlist` local (built but only partially enforced) through the adopted policy, and cover `fetch_json`'s manual redirect loop + `image_format_utils.fetch_image_bytes`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `Image_Generation/http_client.py`'s egress guard delegates to `Utils/egress.py` policy (no parallel/duplicate SSRF logic): API-returned image URLs are blocked for private/link-local/loopback/metadata ranges; user-configured backend `base_url`s (e.g. `http://127.0.0.1:7801`) continue to work.
- [ ] Every hop of `fetch_json`'s manual redirect loop and `fetch_image_bytes` re-validates through the adopted policy.
- [ ] ModelStudio's host allowlist (aliyuncs + base host) is enforced via the adopted policy; the dead `allowlist` local in `modelstudio_image_adapter._is_allowed_remote_image_url` is wired or removed.
- [ ] Unit tests with SSRF payloads (private IPs, metadata IP, scheme abuse, redirect-to-private) pass and local-backend generation does not regress.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
