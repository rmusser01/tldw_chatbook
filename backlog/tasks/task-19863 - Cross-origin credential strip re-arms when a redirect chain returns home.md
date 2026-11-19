---
id: TASK-19863
title: >-
  Cross-origin credential strip re-arms when a redirect chain returns home
status: Done
assignee:
  - '@t1-chatbook'
created_date: '2026-08-22'
labels:
  - security
  - credentials
  - egress
priority: low
dependencies:
  - TASK-19733
---

## Description

Source: residual finding from **TASK-19733**'s reviewer, recorded in that
task's notes and left unfiled. Re-verified at `3605bd52d`.

`Utils/egress.py` strips credential-bearing headers once a redirect takes a
request off its original origin. The decision is made per hop by
`same_origin(url, current)` — and both arguments are wrong for the job:
`url` is the **original** request URL, held constant for the whole chain, so
each hop is compared against the starting point rather than against the
previous hop. All four guarded-fetch loops do this
(`egress.py:753`, `:833`, `:905`, `:975`).

The consequence is that the strip is not sticky. A chain of
`feed.example → evil.example → feed.example` strips on hop 2 and then
**re-attaches** the credential on hop 3, because hop 3's URL is same-origin
with the original. The attacker does not receive the credential directly, but
they choose the path and query string of the request that carries it, so they
control which endpoint on the credential's own origin gets called with the
user's key.

The Fetch specification removes a credential **permanently** once it has been
stripped, precisely so that a redirect chain cannot launder its way back into
an authenticated state. This implementation does not.

Severity is genuinely low — the recipient is always the credential's own origin,
so this is not a disclosure — but the shape is wrong, and the cost of an
attacker-chosen authenticated request against the legitimate origin is not
zero.

A second, smaller residual from the same review belongs here: on the
built-request layer, `Content-Type` is the only exempted header that can carry
arbitrary caller-supplied text across an origin boundary. TASK-19733's Qodo
round already narrowed it to hops that actually carry a body; what remains is
that the *value* is caller-controlled text on a cross-origin hop, which is
worth a deliberate decision rather than an inherited exemption.

## Acceptance Criteria

- [x] Once a credential header has been stripped on any hop of a redirect
      chain, it is not re-attached on any later hop, regardless of that hop's
      origin
- [x] A test drives an `A → B → A` redirect chain and asserts the credential is
      absent on the third request, and is mutation-checked (restoring the
      compare-against-original behaviour makes it red)
- [x] Same-origin chains that never leave the original origin still carry their
      credentials on every hop (no regression for the ordinary case)
- [x] All four guarded-fetch loops in `Utils/egress.py` share one stickiness
      rule rather than each implementing it, so the next loop added inherits it
- [x] The `Content-Type` cross-origin exemption is either justified in a
      comment with the reason it is safe, or narrowed — the decision is
      recorded either way

## Implementation Plan

1. Add one shared stickiness rule — `_StickyOriginPolicy` — next to
   `same_origin`: the first hop that leaves the original origin permanently
   ends credential forwarding for the rest of the chain (Fetch-spec
   behaviour), with the A→B→A incident documented on the class.
2. Replace the per-hop `same_origin(url, current)` comparison in all four
   guarded-fetch loops (`guarded_fetch_httpx`, `guarded_fetch_httpx_async`,
   `guarded_fetch_requests`, `guarded_fetch_aiohttp`) with the policy.
3. Tests: unit tests of the policy; A→B→A chain tests per transport
   (httpx sync headers + async headers + async `auth=` argument, requests
   adapter, aiohttp fake session); a three-hop same-origin regression test;
   and a structural lock that all four loops route through the policy and
   none regresses to the per-hop comparison.
4. Mutation check: restore `same_origin(url, current)` in all four loops and
   confirm the A→B→A tests go red.
5. Record the Content-Type decision status.

## Implementation Notes

- `_StickyOriginPolicy` (Utils/egress.py, beside `same_origin`) is the one
  rule: `hop_carries_credentials(current)` returns True only while the chain
  has never left home; the first cross-origin hop flips it permanently. All
  four loops instantiate it once per chain and ask it per hop. The aiohttp
  session-credential refusal and the httpx `auth=` suppression both key off
  the same sticky answer, so a chain that returns home stays refused /
  unauthenticated.
- Tests: `Tests/Utils/test_egress_redirect_stickiness.py` (10 tests, green;
  `bootstrap_profile` marker with a comment pointing at the same admission
  signature as the egress suites on Tests/conftest.py's keep-bootstrap list).
  The A→B→A chain is driven per transport with synthetic sentinels and
  in-process transports (httpx.MockTransport / requests adapter / fake
  aiohttp session). Mutation check performed: restoring
  `same_origin(url, current)` in all four loops turns the five A→B→A tests
  plus the shared-rule lock red; the sticky code passes all 10. No
  regressions: `test_egress.py` + `test_egress_cross_origin_header_allowlist.py`
  + `test_egress_adoption_census.py` + the new file = 165 passed.
- Content-Type decision (AC#5): recorded, unchanged by this task. The
  exemption is conditional, not absolute: `_BODY_DESCRIBING_HEADERS`
  (egress.py) allows `content-type` across an origin boundary ONLY on a hop
  that carries a body (inferred from the built request's framing headers by
  `strip_cross_origin_request_headers`), and `_hop_headers`' docstring
  records that every guarded-fetch helper issues a bodyless GET, so a
  caller-supplied Content-Type describes nothing and never crosses. That
  justification -- with the PR #1942/Qodo history of how it was narrowed
  from an unconditional transport exemption -- is the comment the AC asks
  for; no further narrowing is warranted while all helpers are bodyless.
- Severity note preserved: the recipient of the re-armed credential was
  always the credential's own origin; the fix closes attacker-controlled
  path/query selection on authenticated requests, not a disclosure.
- ADR required: no -- security-hardening fix inside one module's existing
  helper contract; no new boundary, storage, or interface.
- Modified/added files: `tldw_chatbook/Utils/egress.py`,
  `Tests/Utils/test_egress_redirect_stickiness.py` (new).

## Notes

Filed as low deliberately. Reporting this as a credential leak would
misrepresent the evidence: the credential never reaches the attacker's origin.
The finding is that the guard's rule does not match the specification it is
implementing, which is the kind of drift that becomes a real leak the next time
someone extends the loop.
