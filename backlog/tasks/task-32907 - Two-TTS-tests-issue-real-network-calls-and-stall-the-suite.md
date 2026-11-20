---
id: TASK-32907
title: Two TTS tests issue real network calls and stall the suite
status: Done
assignee:
  - '@claude'
created_date: '2026-09-22 00:15'
updated_date: '2026-11-20 17:24'
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
- [x] #1 The two parametrisations assert against a stubbed transport and make no outbound request
- [x] #2 They pass, rather than failing on `assert provider_id == "alltalk"` after a timeout
- [x] #3 `Tests/TTS/` runs to completion in a normal working time, and the actual cost of `Tests/Subscriptions/` + `Tests/Utils/` is measured rather than assumed
- [x] #4 If other tests are found connecting to unroutable hosts, they are fixed the same way -- identified by observation, not by grepping for hostnames
<!-- AC:END -->

## Implementation Notes

Closed by TASK-42027 on 2026-11-20, at dev d85e90bc9e, with no code change. Every run below went through the scratch-profile wrapper with the network guard that, since TASK-41708, refuses and reports any `socket.getaddrinfo` of a non-local name.

- **AC #1.** `test_alltalk_uses_immutable_leased_default_for_actual_request` has replaced the backend's client with `httpx.AsyncClient(transport=httpx.MockTransport(handle))` and asserted `requested == [f"{configured}/v1/audio/speech"]` since 8b0048751d (2026-08-13), which added the test, before this task was filed. Both nodes recorded no lookup. Negative control: with the `MockTransport` line removed (temporary edit, reverted), both failed, the guard naming `socket.getaddrinfo -> b'leased-default.example':9443` and `...leased-runtime.example:9444` at teardown.
- **AC #2.** Under a temporary `@private_profile_test` (removed): 2 passed in 4.6 s (2.0 s and 1.9 s). Plain, on this machine, both fail after 10.0 s each (21.6 s for the pair, the figures this task reported) with `RecoveryRequired: raw_source_selection_changed`, raised in `AllTalkTTSBackend.__init__` -> `build_httpx_async_client` -> config read. That is the local recovery trap, not a timeout. `assert provider_id == "alltalk"` appears in that traceback only as the line above the `>` marker in `manager_factory`'s frame; it passed. Where the 10 s goes was not traced (it is not the legacy host's 10 s shutdown drain: lowering that to 1 s left both at 10.0 s).
- **AC #3.** The description reports two separate stalls: `Tests/TTS` killed at 17 minutes, and `Tests/Subscriptions` + `Tests/Utils` killed at 20 minutes, both with no output. Both were measured here; neither reproduced as a stall.
  - `pytest Tests/TTS -n 4 --timeout-method=signal`: completed in 563 s wall (load average 20 to 33), 5,168 passed, 272 failed, 99 errors, 21 skipped; no test reached the 300 s timeout and the guard recorded no egress. Most failures are the local trap (239 `raw_source_selection_changed`, 21 `storage_scope_not_enrolled`). Neither the 17-minute stall nor the "wedged at 97% on a TTS lock test" mode appeared.
  - `pytest Tests/Subscriptions Tests/Utils -n 4 --timeout-method=signal`: completed in 1,368 s wall (22:40 in pytest), 2,979 passed, 312 failed, 10 errors, 22 skipped, no egress. 1,200 s of that is four tests in `Tests/Subscriptions/test_briefing_cast.py` that each hit the 300 s timeout: `test_a_concurrent_cast_for_the_same_briefing_is_refused`, `test_row_scoped_exclude_sweeps_a_same_briefing_zombie_while_sparing_the_live_row` and `test_row_scoped_exclude_still_sweeps_a_same_briefing_zombie_once_the_id_lands` start `generate_script(...)` as a task, and `test_a_concurrent_kept_cast_for_the_same_kept_briefing_is_refused` starts `generate_script_from_text(...)`; each then does `await entered.wait()` with no bound (lines 865, 929, 1072, 1472). Locally the task dies on `RecoveryRequired` before it calls the chat stub, so the event never fires (log: "Task exception was never retrieved ... RecoveryRequired('raw_source_selection_changed')"). Plain with `--timeout=30`: 4 failed on the timeout; with the file temporarily marked `bootstrap_profile` (reverted): 4 passed in 7.7 s and the whole file ran in 12.7 s (65 passed, 2 failed). The four ran back to back on one worker, so without them the two directories take about 3 minutes (1,360 s less 1,200 s).
  - This does not explain the reported 20-minute stall. The 300 s timeout was already configured when this task was filed (`timeout = 300` in `pyproject.toml` at d168c49fa5, the last dev commit before it), and in this run each timed-out test printed its `F` as it ended, about every 5 minutes, so the run was never silent for 20 minutes. The four waits only show what such a stall would cost if those tests hung.
  - While diagnosing I sent SIGABRT to xdist worker gw2 of the `Tests/Subscriptions Tests/Utils` run above. Its last test directory was created at 08:41:17 local time (`Tests/Utils/test_workspace_name_validation.py`); it created none after that, showed 0% CPU every time I looked from 08:44 until I aborted it at about 08:58, and `sample` showed its main thread waiting on a lock and a reader thread blocked on a pipe read, which is how an xdist worker waits for its next test. It was idle, not stalled: worker gw0 held the four timed-out tests. xdist reported no crashed test.
- **AC #4.** TASK-41708 swept all 5,816 test files under the guard by observation, not by host-name grep, and fixed every test it saw resolve a name (listed in its notes). Both runs above recorded 0 egress attempts.
- **Left over, not in this task's criteria:** the four unbounded `entered.wait()` calls turn any early failure of the code under test into a 300 s hang; proposed as a follow-up by TASK-42027.
