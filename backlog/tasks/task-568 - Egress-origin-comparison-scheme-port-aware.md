---
id: TASK-568
title: >-
  Make egress same-origin comparison scheme/port-aware app-wide
status: To Do
assignee: []
created_date: '2026-07-25 06:00'
updated_date: '2026-07-25 06:00'
labels:
  - security
  - egress
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Utils/egress.py`'s credential-stripping decision on redirect hops (`guarded_fetch_httpx` → `_hop_headers`) compares origins by HOSTNAME only (`host_of`), ignoring scheme and port. A same-host HTTPS→HTTP downgrade redirect, or a redirect to a different port on the same host, therefore keeps `Authorization`/`Cookie` headers. The Image_Generation redirect loops (PR #862, task-498) deliberately reuse the same primitive to stay policy-consistent, so they inherit the same weakness — flagged by Qodo on #862 and declined there precisely because the right fix is central, not a local fork. Note the app is internally inconsistent: SwarmUI's same-origin image-URL gate already compares scheme+host+port.

Upgrading `host_of`-based same-origin decisions to scheme+host+port must be done centrally so every consumer (egress helpers, Web_Scraping, Subscriptions, Image_Generation) changes together, with the local-backend trust flows (trusted_origins) re-verified — `origin_set` already produces scheme://host:port origins, so the comparison upgrade must not break trusted-origin matching for operator-configured base_urls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Same-origin decisions that gate credential forwarding on redirect hops compare scheme + host + port (with default-port normalization: 443 for https, 80 for http) in `Utils/egress.py` and every consumer that reuses the primitive (including `Image_Generation/http_client.py` and `adapters/image_format_utils.py`).
- [ ] A same-host HTTPS→HTTP downgrade redirect and a same-host different-port redirect both strip credentials; a genuinely same-origin redirect (same scheme, host, effective port) still carries them — each pinned by test.
- [ ] Operator-configured local backends (trusted_origins flows) keep working, including through their own same-origin redirects — regression-tested.
- [ ] All existing egress/SSRF suites pass (`Tests/Utils/test_egress.py`, `Tests/Image_Generation/`, subscriptions egress wiring).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
