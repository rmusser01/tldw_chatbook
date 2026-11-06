---
id: TASK-38965
title: "The provider TLS handshake-stall test waits out a 30 s handler sleep, so the TLS failure suite takes about 39 s"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: '2026-11-06 06:13'
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
- [x] #1 The whole file passes in 12 s or less, serial, with the lane's command, in 3 of 3 runs with the load recorded (38.9 s on dev)
- [x] #2 Both parametrized cases still assert timeout, and each still fails when the probe misclassifies a stall, shown under a temporary edit that makes the stall answer as plain HTTP
<!-- AC:END -->

## Implementation Plan

1. `_never_answer` reads until the client hangs up instead of sleeping 30 s after its read. It still never writes, so both stalls are unchanged, and leaving `async with server` no longer waits out a sleep. A 30 s ceiling stays as a backstop.
2. Time the whole file serially with the lane's command (`--timeout=180 --tb=short`) 3 times, recording the load.
3. Negative control: a temporary edit makes the stall answer as plain HTTP; both parametrized cases must fail on their assertion. Revert it.

## Implementation Notes

The TLS failure suite's handshake-stall case no longer waits out a 30 s handler sleep: the file runs in about 10 s serially.
- `_never_answer` (`Tests/Chat/test_provider_tls_failures.py`) read once, then slept 30 s. Leaving `async with server` waits until every connection is closed (asyncio 3.12 `Server.wait_closed`), not for handlers. Over plain TCP, `StreamReaderProtocol.eof_received` returns True, so the connection the client closed stayed half-open until the handler's `writer.close()`, which ran only after the sleep. Over TLS the client's EOF closes the transport, which is why tls-read-timeout was already fast (corrected in review r10e-rev1, which measured both on Python 3.12.11). It now reads until the client hangs up and still never writes, so both stalls are the same as before. The probe's ConnectTimeout or ReadTimeout closes its connection, and the plain-HTTP confirmation closes its client, so each handler ends within the probe's timeouts. A 30 s `asyncio.timeout` stays as a backstop for a client that never hangs up. `OSError` covers a reset or a TLS error on that read.
- AC#1, the whole file with the lane's command (run.sh, `pytest -p no:xdist Tests/Chat/test_provider_tls_failures.py --timeout=180 --tb=short`, 13 passed each time): 10.18 s, 10.34 s and 10.61 s at load average 23-25 on 18 CPUs. The same command at base 40.02 s at load 26, with handshake-stall at 32.55 s; after the change that case takes 2.86 s (at load 42).
- AC#2, negative control: a temporary edit made the stall answer `HTTP/1.1 200 OK` after its first read. Both parametrized cases failed on their assertion: handshake-stall with `assert 'tls_plain_http' == 'timeout'`, tls-read-timeout with `assert 'invalid_payload' == 'timeout'`. Reverted, green again. The assertions themselves are unchanged.
- `Tests/CI/test_ci_queue_pressure_contract.py`: the comment beside the lane pin records the new time; the targets are unchanged.

Files: `Tests/Chat/test_provider_tls_failures.py`, `Tests/CI/test_ci_queue_pressure_contract.py` (comment only).
