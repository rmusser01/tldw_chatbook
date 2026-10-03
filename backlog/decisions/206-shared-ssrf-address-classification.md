# ADR-206: One shared SSRF address-classification predicate

**Status:** Accepted
**Date:** 2026-09-30
**Task:** backlog/tasks/task-609 - Consolidate-SSRF-layers-skill-remote-fetch-vs-Utils-egress.md

## Context

Two independent SSRF implementations shipped a month apart: PR #822's
`Utils/egress.guarded_fetch_httpx_async` (shared egress policy +
guarded fetch) and PR #831's `Skills_Interop/skill_remote_fetch.fetch_zip_bytes`
(per-hop resolve-and-reject incl. mixed A/AAAA, https-only downgrade check,
GitHub-family-scoped auth stripping, streamed 30 MB cap, wall-clock deadline).
Each carried its own per-address verdict:

- egress `_classify_ip`: metadata / multicast / `is_global`-based
  public-vs-private, with IPv4-mapped normalization.
- skill `_assert_host_allowed`: six named stdlib predicates (private,
  loopback, link-local, reserved, multicast, unspecified) plus — after
  task-610 — a `not is_global` floor for RFC 6598 CGNAT space.

Two hand-rolled predicate chains over the same question can drift silently.
The task-609 delta matrix (Python 3.12, every address category from the
IANA special registries) found exactly one disagreement: the NAT64
well-known prefix `64:ff9b::/96` is `is_global` **and** `is_reserved`, so
egress classified it "public" while the skill layer rejected it — and that
prefix embeds IPv4 (`64:ff9b::7f00:1` *is* `127.0.0.1`), making the loose
verdict a DNS-rebinding-shaped hole.

## Decision

One public predicate in `Utils/egress` is the single source of truth for
"may this address be fetched": `address_is_fetchable(ip_str) -> bool`.
An address is fetchable iff it classifies `"public"` under `_classify_ip`
(not a cloud-metadata endpoint, not multicast, globally reachable,
IPv4-mapped normalized) **and** is in none of the stdlib non-global
categories `is_global` alone misses: reserved, unspecified, loopback,
link-local. Unparseable input fails closed (`False`).

Consumers:

- `Utils/egress._post_resolution` (the `evaluate_url_policy` /
  `check_url_or_raise` pipeline behind every `guarded_fetch_*` helper),
- `Utils/egress.is_public_http_url` (the strict trust-free pre-fetch guard),
- `Skills_Interop/skill_remote_fetch._assert_host_allowed` (per-hop
  revalidation in `fetch_zip_bytes`).

The layers keep their deliberately different policy AROUND the predicate:
egress adds `trusted_origins`, the `[web_security]` allowlist/kill switch,
and allows http+https; the skill layer stays stricter — https-only per hop
(redirect downgrades reject), no bypasses of any kind. Only the per-address
classification is shared.

The one behavioral delta is reconciled toward the stricter layer: egress now
rejects `64:ff9b::/96` (as reason `"private"`), matching the skill layer.
This is a tightening of egress, not a weakening of either side.

`Tools/web_tool_impls._is_public_ip` already documents itself as matching
egress's classification, and `Petdex/network.py` imports `_classify_ip`
directly; both can adopt `address_is_fetchable` in place — left as
follow-ups since neither was in task-609's scope.

## Considered and rejected

- **`fetch_zip_bytes` adopts `guarded_fetch_httpx_async` wholesale**: the
  skill layer's guarantees are strictly deeper (per-hop re-resolution rather
  than re-validation of the URL's current resolution target, https-only
  downgrade rejection, GitHub-family auth scoping, wall-clock deadline over
  the whole exchange). Adopting the shared helper would lose per-hop
  revalidation semantics or force the generic helper to grow skill-specific
  policy; the auth-scoping models also differ on purpose (strict-origin
  allowlist vs GitHub-host-family).
- **Keep two predicates, document them**: rejected — the delta matrix proved
  the drift was already real (NAT64), and every future category fix (like
  task-610's CGNAT) would have to be landed twice.
- **Predicate = `_classify_ip == "public"` alone**: rejected — it admits
  `64:ff9b::/96` (see Context); the reserved/unspecified/loopback/link-local
  floor stays explicit so the taxonomy never depends on stdlib
  `is_global`/`is_reserved` complementarity quirks.

## Consequences

- Adding or changing a rejected address category is a one-line change in one
  function (`address_is_fetchable` / `_classify_ip`), and the category
  matrices in `Tests/Utils/test_egress.py` and
  `Tests/Skills/test_skill_remote_fetch.py` pin both layers to the same
  verdict in both directions.
- egress tightens: hosts resolving into `64:ff9b::/96` are now blocked with
  the standard `EgressBlockedError` remedy (allowed_hosts escape hatch
  applies as for any private range).
- `ipaddress` behavior is load-bearing (CGNAT is `is_private=False` only on
  Python ≥3.12.4; the task-610 floor and the shared predicate are tested
  against the actual runtime). A future stdlib change to `is_global` /
  `is_reserved` semantics shows up as a category-matrix test failure, not a
  silent policy change.
