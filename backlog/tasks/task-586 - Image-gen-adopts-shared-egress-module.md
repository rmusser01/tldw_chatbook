---
id: TASK-586
title: Image-gen adopts shared egress module (Utils/egress.py)
status: Done
assignee: []
created_date: '2026-07-23 12:00'
labels: [image-generation, security, followup]
dependencies: [task-328, task-498]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-328 shipped `Utils/egress.py` (policy + guarded helpers with per-hop redirect re-validation). TASK-498's port should now REUSE it: `Image_Generation/http_client.py`'s light `_validate_egress_or_raise` and `fetch_json` manual-hop loop can delegate to `evaluate_url_policy`/`guarded_fetch_httpx`, with API-returned image URLs evaluated as content-derived (`trusted_origins=frozenset()`) and user-configured backend `base_url` hosts as trusted origins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] `fetch_image_bytes` and `fetch_json` validate via the shared egress module
- [x] API-returned image URLs are evaluated with no trusted origins (fail-closed for private/metadata ranges)
- [x] User-configured backend `base_url` hosts are marked as trusted origins
- [x] Local backends (127.0.0.1 SwarmUI, local sd.cpp) continue to work
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Image_Generation/http_client.py delegates to the shared module: 'Egress is enforced by Utils/egress.py (the app-wide SSRF policy, task-498)' — _validate_egress_or_raise calls egress.check_url_or_raise, API-returned URLs are enforced with no trusted origins, and configured backend base_url hosts are passed as trusted_origins (http_client.py:1-9, 44-68; landed a4ac6b5a7e, hardened faad737d1d).
