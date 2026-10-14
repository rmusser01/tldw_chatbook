---
id: TASK-34926
title: Users cannot mark an older or imported watch source as their own entry
status: Done
created_date: 2026-10-12 23:18
labels:
- watchlists
- security
- decision
priority: medium
updated_date: 2026-10-14 00:48
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since the URL-provenance rework (ADR-207, amended 2026-10-12) a watch source counts as user-entered only when created or edited through the app's own entry paths. Every source a user made before the rework, and every source from an OPML or bundle import, now reads as unknown and is refused when its address is private, so intranet and NAS feeds that worked before stop. Watchlists has no way to edit a source's address, so the only remedies are the allowed-hosts setting or deleting and re-creating the source. This needs an owner decision: an explicit mark-as-mine action, or documenting allowed-hosts as the way. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded
- [x] #2 A user whose private feed stopped after the upgrade can find in the app or the guide how to make it work again
- [x] #3 Nothing marks a source as user-entered without an explicit user action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Owner decision 2026-10-13: document [web_security] allowed_hosts as the way; no mark-as-mine action.
2. Trace who marks a source user-entered (Watchlists New source and Add several only; update_source keeps the stored mark; OPML/bundle imports, Dreams and agent writers are unknown) and how a refused fetch reaches Watchlists (watchlist_failure POLICY_BLOCKED copy).
3. Docs/User_Guide/watchlists.md: a short section on feeds at a private address -- why an older or imported one stopped, how to list its host in [web_security] allowed_hosts, that a listed host is admitted for every fetch, that a source typed in with New source or Add several needs no allow-list.
4. Docs/User_Guide/dreams.md: correct the sentences saying a user's own source is fetched exactly as before, private address or not; point to the watchlists section.
5. config.py [web_security] comment: drop the claim that every configured feed source may be private; say what does and what allowed_hosts does.
6. Check overlap with TASK-34925 and list what remains; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision 2026-10-13: document [web_security] allowed_hosts as the way; no mark-as-mine action.

- Docs/User_Guide/watchlists.md: new section Feeds at a private address -- only a source typed in with New source or Add several may be at a private address and needs no setup; a source from an earlier version or an OPML/bundle import does not count and now fails with Blocked by network safety policy; Watchlists cannot change a source's address, so list the bare host in [web_security] allowed_hosts and restart; a listed host is admitted for every fetch the app makes (feed links and cloud metadata included), so list only trusted hosts.
- Docs/User_Guide/dreams.md: the Tracking paragraph no longer says a user's own source is fetched exactly as before, private address or not; it says Dreams does not change how Watchlists fetches it, a private address is fetched only for a source typed in there, and an older or imported one needs allowed_hosts, linking to the new section. The Switching-off bullet said fetched exactly as before; it now says on its own schedule.
- config.py [web_security] comment (default config template): no longer says every configured feed source may be private; names what counts as typed in, what never does, and that allowed_hosts admits a host (exact name, no port) for every fetch, metadata included.
- Restart (review of #3101, R4): the guide keeps "restart the app". Probed 2026-10-13 on this branch with a temporary test (not committed) through the scratch wrapper: a config selected with Tests/Backup_Recovery/config_test_support.select_config_source, egress bound to it, the real 1 s throttle, DNS faked to 192.168.1.20; after allowed_hosts was edited to list the host, five policy checks 1.2 s apart over about 5 s all still refused it, for an in-place save (same inode) and for a replace-by-rename save, with the edit made at once or 1.2 s after the first read; load_cli_config_and_ensure_existence also still returned the old list; after the caches were cleared (what a restart does) the host was admitted. This matches TASK-34943 (an edit outside the app is not picked up while it runs).
- AC 3, traced: UrlProvenance.USER_ENTERED is set only by the Watchlists New source and Add several handlers (watchlists_collections_screen.py _create_source / _create_bulk_sources); update_source keeps the stored mark unless an app-owned caller passes the enum, and no UI does; the subscriptions v2->v3 migration keeps only an explicit user_entered; a serialized user_entered string is refused (Tests/Subscriptions/test_source_authority_boundary.py). Nothing was changed there.
- Test: Tests/Subscriptions/test_url_fetch_source_provenance.py::test_an_older_private_source_runs_once_its_host_is_allowed pins the remedy through the real egress policy (bare host fetched; host with port or a wildcard refused, nothing sent). Seen red with the allow-list check disabled in egress.py (host case failed), then restored.
- In the app: a refused Watchlists check shows The source was blocked by the network safety policy. / Next: Choose a public HTTP(S) source allowed by the network safety policy. (Subscriptions/watchlist_failure.py POLICY_BLOCKED); it does not name allowed_hosts. Left unchanged by the owner's scope (docs only).
- Overlap: TASK-34925 AC 2 (the guide says how a user's own private sources are treated) is covered by this change; its AC 1 items remain (track_service.backfill_egress_marks docstring: an unmarked restored source fetched as a trusted origin, and the boot-path cost reason; the startup-reconcile docstring; the guide's restore paragraph).
- Verified on branch fix/task-34922-34926-dreams-holds-and-legacy-sources off dev 729f99d0ed, 2026-10-13.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
