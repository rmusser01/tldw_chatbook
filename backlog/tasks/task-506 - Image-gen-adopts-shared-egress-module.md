---
id: TASK-506
title: Image-gen adopts shared egress module (Utils/egress.py)
status: To Do
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
- [ ] `fetch_image_bytes` and `fetch_json` validate via the shared egress module
- [ ] API-returned image URLs are evaluated with no trusted origins (fail-closed for private/metadata ranges)
- [ ] User-configured backend `base_url` hosts are marked as trusted origins
- [ ] Local backends (127.0.0.1 SwarmUI, local sd.cpp) continue to work
<!-- AC:END -->
