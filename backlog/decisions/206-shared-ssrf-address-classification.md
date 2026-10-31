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
(not a cloud-metadata endpoint, not multicast, globally reachable) **and**
is in none of the stdlib non-global categories `is_global` alone misses:
reserved, unspecified, loopback, link-local. Both halves are computed on
the address a connection actually reaches (`_effective_ip`): an IPv4-mapped
(`::ffff:a.b.c.d`) or NAT64 well-known-prefix (`64:ff9b::a.b.c.d`, RFC 6052)
address gets the verdict of the IPv4 address it embeds. Unparseable input
fails closed (`False`).

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

The one behavioral delta is reconciled by classifying what the prefix
embeds: `64:ff9b::7f00:1` is `127.0.0.1` and is rejected (reason
`"private"`; an embedded metadata address is `"metadata"`), while
`64:ff9b::5db8:d822` is `93.184.216.34` and is fetchable. For egress this
closes the hole; for the skill layer it relaxes a blanket refusal to a
precise one.

Amended in PR #2993 review (owner decision, 2026-10-03). The first cut of
this ADR rejected the whole prefix to match the skill layer. On an
IPv6-only network with DNS64, every IPv4-only host resolves into
`64:ff9b::/96`, so that verdict refused every guarded fetch to such a host
there — web fetch, article ingest, media download — with a per-host
`allowed_hosts` entry as the only remedy. RFC 6052 forbids the well-known
prefix from representing non-global IPv4 addresses, so refusing exactly the
non-global embeddings keeps the security property the blanket refusal was
after.

`Tools/web_tool_impls._is_public_ip` already documents itself as matching
egress's classification, and `Petdex/network.py` imports `_classify_ip`
directly; both can adopt `address_is_fetchable` in place — left as
follow-ups since neither was in task-609's scope. Petdex inherits the
embedded-IPv4 verdict through `_classify_ip`; `_is_public_ip` keeps its own
predicate chain but unwraps addresses through the same `_effective_ip`, so
all four consumers agree on NAT64 and IPv4-mapped addresses.

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
- **Predicate = `_classify_ip == "public"` alone**: rejected — the
  reserved/unspecified/loopback/link-local floor stays explicit so the
  taxonomy never depends on stdlib `is_global`/`is_reserved`
  complementarity quirks (the NAT64 disagreement in Context was one).
- **Reject `64:ff9b::/96` wholesale** (this ADR's first cut): rejected in
  PR #2993 review — it breaks every guarded fetch to an IPv4-only host on a
  DNS64/NAT64 network; see the amendment under Decision.

## Consequences

- Adding or changing a rejected address category is a one-line change in one
  function (`address_is_fetchable` / `_classify_ip`), and the category
  matrices in `Tests/Utils/test_egress.py` and
  `Tests/Skills/test_skill_remote_fetch.py` pin both layers to the same
  verdict in both directions.
- egress tightens: a host resolving to a `64:ff9b::/96` address that embeds
  a non-global IPv4 address is now blocked with the standard
  `EgressBlockedError` remedy (allowed_hosts escape hatch applies as for
  any private range). One embedding a public IPv4 address is fetchable, as
  it was before this ADR. Network-specific NAT64 prefixes (RFC 6052 §2.2)
  are not recognisable from the address alone and are classified as
  ordinary IPv6, unchanged.
- `ipaddress` behavior is load-bearing (CGNAT is `is_private=False` only on
  Python ≥3.12.4; the task-610 floor and the shared predicate are tested
  against the actual runtime). A future stdlib change to `is_global` /
  `is_reserved` semantics shows up as a category-matrix test failure, not a
  silent policy change.
