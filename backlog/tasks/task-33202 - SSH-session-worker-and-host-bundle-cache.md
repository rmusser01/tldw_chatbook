---
id: TASK-33202
title: SSH session worker and host bundle cache
status: Done
assignee:
  - '@robert'
created_date: '2026-09-27 12:00'
updated_date: '2026-09-27 23:45'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33009
  - TASK-33021
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A warm `fs_*` call on an SSH workspace binding cost ~0.7 s on a LAN host, almost all of it round trips (a new ssh channel plus the shipped bundle per call). Make a warm call cost about one network round trip by reusing one per-run session per binding, and make the first call of a run cheap with a host-side bundle cache, without weakening any ADR-181 guarantee. Design: `Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md`; decision: ADR-181 "Amendment 2026-09-27: session worker and bundle cache".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Within one Console run, calls on one SSH binding reuse a single session; each operation still runs in a fresh host process with root pin, denylist, exclusions and the two-tier watchdog
- [x] #2 The host caches the worker only in a private `$XDG_RUNTIME_DIR/tldw-worker/`, sha256-verified on every load, and caches nothing when that directory is missing or not private
- [x] #3 A retargeted destination starts no session and marks the binding BLOCKED; a recreated root goes STALE_IDENTITY and re-captures inside the live session
- [x] #4 Session failures keep the ADR-181 status rules: a live-session queue timeout, a laptop-ended session and a clean host idle-exit never flip a binding to BLOCKED
- [x] #5 `[console_ssh] session_worker = false` restores pure one-shot calls; `session_idle_s` and `bundle_cache` are configurable
- [x] #6 Live UAT against a real host passes, with warm-call median at or below the spike echo floor + 15 ms
- [x] #7 ADR-181 amendment, user guide, CHANGELOG updated; preflight passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach (9 plan tasks, branch `feat/ssh-session-worker`): spike (framed echo over the ControlMaster: GO, echo median 6.5-8.1 ms, fork+waitpid ~0 ms); stdlib frame codec; host fork-server `serve` loop; stage-1 loader with the runtime-dir cache; bundle `serve_session` entry + loopback harness + Python 3.10 CI run; laptop `RemoteSessionWorker`; session registry + executor routing + failure policy + `[console_ssh]` keys; one session per binding per run, closed at run end and app exit; ADR amendment, docs, live UAT.
- Rulings that changed behaviour vs the spec: R7 per-child output cap = MAX_RESPONSE_BYTES + 64 KiB; R8 unadmitted timeout on a live session = OP_TIMEOUT; R9 laptop-ended session = REMOTE_OP_FAILED, natural death classified by the real ssh exit code; R10 clean exit 0 is benign and restart-free, laptop reaps at idle_s while the host idles at idle_s + grace; R11 sessions closed by a decorator on `_run_agent_reply` (every exit path); R12 session key unique per run (message id + uuid) and closed keys tombstoned (last 1024) so stragglers go one-shot.
- Live UAT (`Tests/Tools/test_remote_session_live.py`, opt-in `live_ssh`, `TLDW_LIVE_SSH_HOST=ml-user@192.168.5.84`, 2026-09-27, LAN Wi-Fi, Debian 13 / Python 3.13): 8/8 checks pass. Cold miss start 498 ms (first call 700 ms incl. ping + op); cache-hit start min/median/max 110 / 192 / 422 ms over 5 starts; warm `fs_read` median 13.3 ms, p90 28.5 ms (n=420 back-to-back); same-window ICMP ping median 9.4 ms, p90 83.5 ms. Target warm median <= 7.51 + 15 = 22.51 ms: met. Second run minutes later on a noisier link (ping median 32.4 ms, p90 123.9 ms): warm median 14.8 ms, p90 127.3 ms (n=317); cold miss 591 ms; cache-hit starts 175 / 367 / 422 ms; 8/8 pass, no host leftovers besides the one 0600 cache file.
- Not met as written: the spec's "cache-hit cold start <= ~3 RTT + 40 ms". Start time is dominated by opening the ssh channel over the ControlMaster, which alone measured 110-390 ms (`ssh true` over the master) on this link; the loader/cache adds little. Paid once per run per binding.
- Findings while measuring: (1) pacing calls 0.1-0.2 s apart raises the warm median to ~60-75 ms, all of it before the admitted marker arrives (Wi-Fi power-save wake-up, not the protocol); back-to-back calls are ~20 ms. (2) admitted marker -> terminal frame is a steady ~8 ms against a ~4 ms host op, possibly Nagle between the host's two writes (sshd leaves Nagle on for non-pty sessions); not confirmed. (3) ext4 reuses a freed inode at once, so `rm -rf root && mkdir root` recreates the SAME (dev, ino) and the pin cannot detect it (existing ADR-181 identity rule, not new); the UAT builds the replacement before removing the original.
- Also fixed: `Tests/Tools/test_remote_watchdog.py::test_exit_75_is_reserved_to_the_watchdog` failed since the serve loop landed; the pin now allows the fork child's single `os._exit(<worker return code>)`.
- Files (task 9): `backlog/decisions/181-ssh-remote-workspace-bindings.md`, `Docs/User_Guide/console/context-and-rag.md`, `CHANGELOG.md`, `pyproject.toml` (marker), `Tests/Tools/test_remote_session_live.py`, `Tests/Tools/test_remote_watchdog.py`.
<!-- SECTION:NOTES:END -->
