---
id: TASK-38965
title: "The provider TLS handshake-stall test waits out a 30 s handler sleep, so the TLS failure suite takes about 39 s"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- tests
- ci
- performance
priority: low
references:
- Tests/Chat/test_provider_tls_failures.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38717
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-38717 branch (the state after it merges), 2026-11-05. TASK-38717's notes list it as a follow-up.

**Where the 30 s goes.** In `Tests/Chat/test_provider_tls_failures.py`, `_never_answer` (`:242-249`) sleeps 30 s after its read (`:245`). `test_real_tls_stalls_are_timeouts_not_plain_http[handshake-stall]` (`:266-282`) probes inside `async with server` (`:276`), and leaving that block waits for the handler to finish. The probe itself times out after 0.5 s.

**Measured through the scratch-profile wrapper, at load 12-17, with all 13 tests passing:**

| Run | Time |
|---|---|
| handshake-stall alone | 32.55 s call |
| Whole file, serial, with the lane's command (`--timeout=180 --tb=short`) | 38.87 s |
| Whole file, `-n 4` | 35.57 s |

The handshake-stall node accounts for 32.5 s of each run; the next slowest test takes 2.6 s.

**Why it matters now.** After TASK-38717 merges, this file runs whole in the serial "Run Dreams and Guardian boundary contracts" step (`derived-artifacts.yml:1113` on that branch), so every PR pays the sleep. TASK-38717 measured the file at 42.8-45.7 s at load 43-59.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The whole file passes in 12 s or less, serial, with the lane's command, in 3 of 3 runs with the load recorded (38.9 s on dev)
- [ ] #2 Both parametrized cases still assert timeout, and each still fails when the probe misclassifies a stall, shown under a temporary edit that makes the stall answer as plain HTTP
<!-- AC:END -->
