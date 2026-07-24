---
id: TASK-498
title: >-
  Port image-generation egress/SSRF protections from tldw_server
status: To Do
assignee: []
created_date: '2026-07-22 11:32'
updated_date: '2026-07-22 11:32'
labels:
  - image-generation
  - security
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to the image-generation multi-provider foundation (Phase 1 of the in-chat image-gen program; see `docs/superpowers/specs/` image-generation design). The server's `Image_Generation` package we port includes egress/SSRF guards that tldw_chatbook currently lacks app-wide: `_validate_egress_or_raise` / `_resolve_redirect_url` in `http_client`, and `evaluate_url_policy` from `Security/egress`. Phase 1 deliberately ships only a **light** guard (reject non-`http(s)` schemes; keep each adapter's built-in per-backend host checks — SwarmUI same-origin, ModelStudio `aliyuncs` allowlist) and stays permissive for user-configured `base_url`s so local backends (127.0.0.1 SwarmUI, local sd.cpp) keep working.

That light guard does NOT protect against SSRF via **API-returned image URLs** (OpenRouter / Novita / ModelStudio can return arbitrary `http(s)` URLs that the adapters then fetch via `fetch_image_bytes`). Port the server's real egress protections so those fetches are validated (block private/link-local/metadata ranges for URLs the app did not originate, DNS-rebinding-aware where feasible), while still allowing user-configured local backend base URLs. This is the security hardening the light guard is a placeholder for, and it doubles as the first real SSRF protection in the app.

TASK-328 has now shipped a shared egress module at `tldw_chatbook/Utils/egress.py` that can be reused here (see TASK-506 for the adoption follow-up).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Image URLs returned by remote backends (OpenRouter/Novita/ModelStudio) are validated before fetch: private, link-local, loopback, and cloud-metadata (169.254.169.254) ranges are blocked for app-non-originated URLs.
- [ ] User-configured backend `base_url`s (e.g. `http://127.0.0.1:7801` SwarmUI, local paths) continue to work — the guard distinguishes user-configured endpoints from API-returned URLs.
- [ ] Non-`http(s)` schemes and redirect chains beyond the configured max are rejected.
- [ ] Guard is unit-tested with SSRF payloads (private IPs, metadata IP, scheme abuse, redirect-to-private) and does not regress local-backend generation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
