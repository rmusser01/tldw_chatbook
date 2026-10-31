---
id: TASK-609
title: >-
  Consolidate SSRF layers: skill_remote_fetch fetcher vs Utils/egress guarded fetch
status: Done
assignee: [rmusser01]
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
- [x] #1 One shared host/address-classification predicate is used by both egress.py and skill_remote_fetch.py (or a documented decision records why they stay separate).
- [x] #2 No remote-fetch security property is weakened: per-hop revalidation, auth scoping, size cap, and total deadline retain their existing regression tests green.
- [x] #3 Any behavioral delta between the two layers (e.g. address categories rejected) is reconciled and tested.
<!-- AC:END -->

## Implementation Plan

1. Build the empirical delta matrix between the two layers' per-address verdicts (skill layer's six predicates + the task-610 not-is_global floor vs egress `_classify_ip == "public"`) over public/private/loopback/link-local/unspecified/multicast/CGNAT/documentation/reserved/special-registry/NAT64/6to4/v4-mapped/metadata addresses, on Python 3.12.
2. Extract one shared public predicate `address_is_fetchable(ip_str)` in Utils/egress (classify-public AND not reserved/unspecified/loopback/link-local; fail closed on unparseable) and route BOTH layers through it: egress `_post_resolution`/`is_public_http_url` and `skill_remote_fetch._assert_host_allowed`. The two layers keep their deliberately different surrounding policy (trusted_origins + config allowlist + http/https vs https-only per-hop + no bypasses) — only the address classification is shared.
3. Reconcile the behavioral delta found in step 1 in favor of the stricter layer, with tests on both layers (expected: the NAT64 well-known prefix 64:ff9b::/96 is is_global yet is_reserved — and can embed loopback/private IPv4 — so the reserved rejection is kept and egress tightens to match the skill layer).
4. Regression: the four property suites (per-hop revalidation, GitHub-family auth scoping, stream cap, wall-clock deadline) plus both files' full suites stay green; TASK-610's CGNAT tests must still pass through the shared predicate.
5. ADR: no existing egress/SSRF ADR exists (checked backlog/decisions/); this is a cross-module security-interface consolidation, so record it as a new ADR (next free number 206) linking both layers and naming Petdex/network.py as an existing `_classify_ip` consumer that can adopt the shared predicate separately.

ADR required: yes
ADR path: backlog/decisions/206-shared-ssrf-address-classification.md
Reason: Cross-module interface decision (Skills_Interop consuming a Utils security predicate as the single classification source of truth) and a security-policy reconciliation (one address category changes verdict in egress).

## Implementation Notes

- **Approach** — consolidated on ONE shared predicate rather than one fetch implementation, per the AC's first branch. New public `Utils.egress.address_is_fetchable(ip_str)`: fetchable iff `_classify_ip == "public"` (metadata/multicast/`is_global` with IPv4-mapped normalization) AND none of reserved/unspecified/loopback/link-local; fail closed on unparseable. Consumers: egress `_post_resolution` (the whole `evaluate_url_policy`/`check_url_or_raise`/guarded-fetch pipeline), egress `is_public_http_url`, and `skill_remote_fetch._assert_host_allowed` (module-level import, no cycle). The layers keep their deliberately different surrounding policy (egress: trusted_origins + `[web_security]` allowlist + http/https; skill: https-only per hop, no bypasses) — only address classification is shared. TASK-610's CGNAT fix survives inside the shared predicate (re-verified).
- **Delta reconciliation (AC#3)** — the empirical delta matrix (37 addresses, every IANA special-registry category, Python 3.12.11) found exactly ONE disagreement: NAT64 well-known prefix `64:ff9b::/96` — `is_global` True yet `is_reserved` True — allowed by egress's old classification, rejected by the skill layer's six-predicate chain. Reconciled to the STRICT side: the reserved floor is explicit in the shared predicate, so both layers reject it (egress tightens; `64:ff9b::7f00:1` embeds `127.0.0.1`, so the loose verdict was a rebinding-shaped hole). Tested on both layers.
- **Evidence (red -> green)**:
  - Red: `test_egress.py -k "address_is_fetchable or nat64"` -> `2 failed` (no such predicate; NAT64 allowed by egress); `test_skill_remote_fetch.py -k shares_the_egress` -> `1 failed` (ImportError).
  - Green: full `Tests/Utils/test_egress.py` + `Tests/Skills/test_skill_remote_fetch.py` + allowlist + census -> `203 passed`, including the four AC#2 property tests run by name (`test_redirect_hop_revalidated_and_capped`, `test_auth_scoped_to_github_family`, `test_stream_cap_aborts`, `test_fetch_total_deadline_aborts_slow_drip`) plus both CGNAT pins -> `10 passed`.
  - No regressions: all 14 egress-policy consumer suites (`run_webhooks`, `http_client`, `ingest_preflight_egress`, `video_egress_guard`, `stream_resolve`, `remote_huggingface`, `stream_fetch`, `exposure_paths`, `deep_search_pipeline`, `web_fetch_wiring`, `settings_probe_egress`, `github_api_client`, `download_caps_wiring`, `sitemap_crawl`) run twice — with this change and with both files reverted to HEAD — producing BYTE-IDENTICAL failure lists (70 failed / 258 passed both sides; pre-existing: TASK-32628 admission signature in non-enrolled files, missing optional `playwright` dep, and the known crawler red). Zero new failures.
- **Tests added**: `test_address_is_fetchable_shared_floor` + `test_nat64_well_known_prefix_blocked_everywhere` (Tests/Utils/test_egress.py, incl. a cross-layer call into `_assert_host_allowed`); `test_host_allow_shares_the_egress_address_predicate` + `test_nat64_host_rejected_per_hop` (Tests/Skills/test_skill_remote_fetch.py).
- **Files changed**: `tldw_chatbook/Utils/egress.py`, `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`, `Tests/Utils/test_egress.py`, `Tests/Skills/test_skill_remote_fetch.py`, `backlog/decisions/206-shared-ssrf-address-classification.md` (+ README index row), this task file.
- **ADR**: ADR-206 created (see plan); it also records the rejected alternative of making `fetch_zip_bytes` adopt `guarded_fetch_httpx_async` wholesale (would lose per-hop re-resolution semantics and mix incompatible auth-scoping models), and names `Petdex/network.py`/`Tools/web_tool_impls` as existing `_classify_ip` consumers that can adopt the shared predicate as a follow-up (out of scope here).
- **PR #2993 review amendment (owner decision, 2026-10-03)**: the reconciliation above refused the whole NAT64 prefix; on a DNS64/NAT64 network every IPv4-only host resolves into it, so every guarded fetch to such a host was refused. The shared predicate now classifies the IPv4 the address embeds (`_effective_ip`): `64:ff9b::7f00:1` is still refused, a public embedding is fetchable. `test_nat64_well_known_prefix_blocked_everywhere` became `test_nat64_well_known_prefix_gets_its_embedded_ipv4_verdict`; ADR-206 amended.
- Note: the decisions README index was already stale past ADR-187 (files 188-205 exist without rows — pre-existing, left as-is); only the ADR-206 row was added.
