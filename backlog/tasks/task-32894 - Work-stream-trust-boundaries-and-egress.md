---
id: TASK-32894
title: "Work stream: trust boundaries and egress"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six boundary defects: an endpoint probe that reaches the network without the SSRF check its sibling
branch already applies four lines away, a watchlist that seeds `trusted_origins` from the discovered
`<loc>` rather than the subscription's own source, a Confluence auth branch whose predicate makes the
check unreachable (with a test that is green for the wrong reason), eight credentialed calls that follow
redirects with no byte cap, two TTS backends that bypass `resolve_provider_api_key`, and four XML
parsers still on the stdlib parser while `defusedxml` is already a base dependency.

ADR-012's 2026-09-19 amendment is cited, not re-decided.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The endpoint probe applies one SSRF check on every branch
- [ ] #2 No credentialed request follows a redirect without an explicit allow and a byte cap
- [ ] #3 The four XML parsers use `defusedxml` and their `_KNOWN_UNHARDENED` rows are dropped in the same commit
- [ ] #4 The Confluence false-green test asserts the behaviour it claims to
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-security` as `f25f707e6e` (24 files, +1188/-115). Not pushed. All six were real.

**The open severity question is answered: NO, and the P1 rating stands.** The review left one item
explicitly unverified, with promotion to P0 hinging on it: can any **non-restore** path set
`api_settings.<provider>.base_url`? Swept every writer of that key and of `custom_endpoints.*`:

- `Chatbooks/` -- the `ContentType` enum is CONVERSATION/NOTE/CHARACTER/MEDIA/EMBEDDING/PROMPT/EVALUATION/
  KEPT_BRIEFING. There is no config content type. (Independently re-checked: `grep base_url Chatbooks/` -> 0.)
- `Sync_Interop/` -- adapters are chat/media/notes/notes_organization/workspaces/source_cache; no config
  surface at all. (Independently re-checked: `grep api_settings Sync_Interop/` -> 0.)
- `Backup_Recovery/` -- the only external-input writer, via the guarded `_write_raw_cli_config_unlocked`.

Every other writer is a user-driven UI surface. **Restore remains the sole non-user vector**, so the
prerequisite compromise the P1 rating assumes still holds. This closes the corresponding "Left UNVERIFIED"
row in `report.md`.

Funnel points chosen (one per item, not per caller): `probe_settings_endpoint` after endpoint resolution;
`URLMonitor._fetch_url_content` reading a new `trusted_source` key with the sitemap arm threading the
subscription's configured source; `make_request` with `params` folded into the URL so the predicate becomes
true for the shapes production actually uses; one module-local `_credentialed_search_request` for all eight
calls; `config.get_api_key` (the ADR-012-amendment accessor) in both TTS backends; a `defusedxml` swap with
the four `_KNOWN_UNHARDENED` rows dropped in the same commit.

Three things found along the way:

1. **Item 3 had a second half.** The guarded Confluence branch never took `self._session_lock` -- harmless
   while unreachable, a real hazard once the guard is live, since `scrape_many` runs these in
   `asyncio.to_thread`. Measured **6 concurrent session entries** without the lock. The existing concurrency
   test also had to move from stubbing `session.request` to `session.send`, or it measured a method the
   shipped path no longer calls.
2. **`Local_Ingestion/XML_Ingestion.py` is dead at dev.** `from tldw_chatbook.DB.Client_Media_DB_v2 import
   add_media_to_database` raises `ImportError`; that name no longer exists. Independently confirmed. So the
   XXE hardening there is hygiene, not a live exposure -- and the module belongs in the deletion stream
   (TASK-32899), which does not currently name it.
3. The new sitemap-refusal handlers log the exception **type only**, deliberately: `EntitiesForbidden`'s
   message carries the entity name a hostile document chose, and that sink is persistent.

Gates: `preflight.sh` exit 0 (diagnostic-inventory drift read row-by-row before regenerating, per the
artifact's own rule); size ratchet exactly the 5 baseline reds by name; baseline diffs name-set compared --
`Tests/Web_Scraping` 133=133, watchlist/monitor 55=55, TTS backends 18=18, zero new failures.

Merge status: conflicts with current `origin/dev` in `Docs/security/production-diagnostic-inventory.json`
only -- a derived artifact, to be regenerated rather than hand-merged.

Follow-up filed as TASK-32907: two `Tests/TTS/test_alltalk_backend.py` parametrisations issue real outbound
requests to `.example` hosts and eat a DNS timeout. Confirmed pre-existing on clean dev.
<!-- SECTION:NOTES:END -->
