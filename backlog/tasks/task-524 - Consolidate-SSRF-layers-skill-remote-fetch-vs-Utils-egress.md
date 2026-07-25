---
id: TASK-524
title: >-
  Consolidate SSRF layers: skill_remote_fetch fetcher vs Utils/egress guarded fetch
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - security
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #822 introduced Utils/egress.guarded_fetch_httpx_async (shared SSRF policy + guarded fetch) while PR #831 shipped skill_remote_fetch.fetch_zip_bytes with its own, deeper SSRF layer (per-hop resolve-and-reject incl. mixed A/AAAA, https-only downgrade check, GitHub-family-scoped auth stripping, streamed cap, wall-clock deadline). The codebase now has two independent SSRF implementations that can drift. Evaluate consolidating: either fetch_zip_bytes adopts the shared helper (must not lose per-hop revalidation or family-scoped auth) or the shared host-allow predicate is extracted so both layers use one address-classification source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared host/address-classification predicate is used by both egress.py and skill_remote_fetch.py (or a documented decision records why they stay separate).
- [ ] #2 No remote-fetch security property is weakened: per-hop revalidation, auth scoping, size cap, and total deadline retain their existing regression tests green.
- [ ] #3 Any behavioral delta between the two layers (e.g. address categories rejected) is reconciled and tested.
<!-- AC:END -->
