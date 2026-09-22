---
id: TASK-32907
title: Two TTS tests issue real network calls and stall the suite
status: To Do
assignee: []
created_date: '2026-09-22 00:15'
labels:
  - tier2-review
  - review-testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while gating the tier-2 security stream. Two parametrisations of one test attempt a **real outbound
request** and pay a DNS timeout for it:

```
Tests/TTS/test_alltalk_backend.py::test_alltalk_uses_immutable_leased_default_for_actual_request
    [app_tts0-https://leased-default.example:9443]
    [app_tts1-https://leased-runtime.example:9444]
```

`.example` is a reserved, deliberately non-resolvable TLD (RFC 2606), so the request cannot succeed; it can
only time out. On this machine the pair costs ~21 s wall clock and both then fail on
`assert provider_id == "alltalk"` -- so the test does not even get to assert what it means to.

**Confirmed pre-existing on a clean `origin/dev` worktree**, not introduced by any tier-2 change.

The cost is environment-dependent and can be far worse than 21 s: the agent running the security stream
reported `Tests/TTS/` and `Tests/Subscriptions/`+`Tests/Utils/` each stalling with no output until killed at
**17 and 20 minutes**, which is consistent with a resolver that blackholes rather than refuses. That is why
those directories could not be run whole, and why that stream had to fall back to diffing individual files.

## A second, distinct stall mode — reported but NOT reproduced

Separately from the DNS timeouts above, the agent implementing TASK-32892 reported a whole-suite sweep
**wedging at 97% on a TTS lock test**, which it killed. That is a different failure mode from a DNS timeout
(a deadlock does not resolve on its own), and if real it is the more serious of the two.

**I could not reproduce it in a bounded run and am not asserting it.** An attempt to isolate it with
`pytest Tests/TTS/ --timeout=20 --timeout-method=thread` exceeded a 600 s budget without printing a summary
line, so the result is inconclusive — not evidence of a hang, and not evidence against one. Anyone picking
this up should start by reproducing it rather than trusting this paragraph.

Note `--timeout-method=thread` cannot interrupt a thread blocked on a lock; `--timeout-method=signal` can,
and is the right instrument if a deadlock is suspected.

## Fix

Stub the transport. The repo already has the idiom in several places (`httpx.MockTransport`, and
`Tests/Utils/test_settings_probe_egress.py` added in TASK-32894 uses a DNS/config stub for exactly this
reason). A test whose name ends `_for_actual_request` should assert the request was *formed* correctly
against a stub, not that it reaches a host that does not exist.

## Deliberately NOT claimed

A broader sweep was inconclusive and should not be treated as a finding: 401 test files reference
`.example`/`.invalid`/`.test` hosts, but the overwhelming majority use them as **inert fixture data** (a URL
string passed to a parser or stored in config), never as a request target. Mechanically separating "mentions
a fake host" from "connects to a fake host" is not something grep can do, so the scope here is the two
parametrisations that were actually observed connecting, plus whatever the timing investigation below turns
up. Do not open a 401-file cleanup on the strength of that grep.

Source: tier-2 code review 2026-09-21, found while gating TASK-32894.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two parametrisations assert against a stubbed transport and make no outbound request
- [ ] #2 They pass, rather than failing on `assert provider_id == "alltalk"` after a timeout
- [ ] #3 `Tests/TTS/` runs to completion in a normal working time, and the actual cost of `Tests/Subscriptions/` + `Tests/Utils/` is measured rather than assumed
- [ ] #4 If other tests are found connecting to unroutable hosts, they are fixed the same way -- identified by observation, not by grepping for hostnames
<!-- AC:END -->
