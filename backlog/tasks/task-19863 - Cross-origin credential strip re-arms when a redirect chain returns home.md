---
id: TASK-19863
title: >-
  Cross-origin credential strip re-arms when a redirect chain returns home
status: To Do
assignee: []
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

- [ ] Once a credential header has been stripped on any hop of a redirect
      chain, it is not re-attached on any later hop, regardless of that hop's
      origin
- [ ] A test drives an `A → B → A` redirect chain and asserts the credential is
      absent on the third request, and is mutation-checked (restoring the
      compare-against-original behaviour makes it red)
- [ ] Same-origin chains that never leave the original origin still carry their
      credentials on every hop (no regression for the ordinary case)
- [ ] All four guarded-fetch loops in `Utils/egress.py` share one stickiness
      rule rather than each implementing it, so the next loop added inherits it
- [ ] The `Content-Type` cross-origin exemption is either justified in a
      comment with the reason it is safe, or narrowed — the decision is
      recorded either way

## Notes

Filed as low deliberately. Reporting this as a credential leak would
misrepresent the evidence: the credential never reaches the attacker's origin.
The finding is that the guard's rule does not match the specification it is
implementing, which is the kind of drift that becomes a real leak the next time
someone extends the loop.
