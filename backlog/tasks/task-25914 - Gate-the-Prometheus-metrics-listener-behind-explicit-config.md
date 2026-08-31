---
id: TASK-25914
title: Gate the Prometheus metrics listener behind explicit config
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:11'
updated_date: '2026-08-31 16:25'
labels:
  - ops
  - security
  - defect
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Installing an optional extra silently opens a network listener. Verified on origin/dev: app.py:16900 calls init_metrics_server(port=int(os.environ.get("METRICS_PORT", "8000"))) unconditionally during boot, and Metrics/metrics.py:245-262 starts prometheus_client.start_http_server(port) whenever PROMETHEUS_AVAILABLE is true - the only gate is whether the dependency is importable. prometheus_client ships in the dev and debugging extras (pyproject.toml:312,315), so anyone who installs either gets an unauthenticated HTTP listener bound at boot with no setting to decline and nothing in the UI saying it happened. Found while verifying an ops-area claim during the 2026-08-31 parity pass. Dependency presence is not consent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The metrics listener starts only when a config setting explicitly enables it; the setting defaults to off
- [x] #2 With the dependency installed and the setting off, no socket is bound - verified by a test or a documented ss/lsof check
- [x] #3 The bind address is configurable and defaults to loopback rather than all interfaces
- [x] #4 When the listener does start, the fact and the port are stated in the log at a level the user will see
- [x] #5 Existing metric collection is unaffected when the listener is off - counters and histograms still record
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. A defect fix that adds one config gate; it removes a network surface rather than adding one, and introduces no new architectural seam.

1. Put the gate inside init_metrics_server, not at the app.py call site, so any future caller inherits it (one choke point, per the fix-once-where-callers-route-through rule).
2. Resolve enabled/port/bind_address through a small indirection so the config import stays lazy and tests need no config file on disk.
3. Default bind_address to loopback -- prometheus_client's own default is 0.0.0.0, which must not be inherited.
4. Keep METRICS_PORT working as a port override but NOT as an enabler (AC#1 says config enables it).
5. Add a [metrics] block to CONFIG_TOML_CONTENT so the setting is discoverable.
6. Update the three existing startup-outcome tests, which pin the old contract, preserving each invariant they protect.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gated the Prometheus listener on `[metrics] enabled`, defaulting off, and moved port/bind-address resolution into `Metrics/metrics.py` so the gate sits at the choke point rather than the call site.

**Approach.** The gate lives in `init_metrics_server`, not in `app.py`, so any future caller inherits it. `_metrics_server_config()` resolves the three settings through `_get_cli_setting`, a thin indirection that keeps the `config` import lazy (this module is imported early) and lets tests exercise resolution with no config file on disk. The bind address defaults to `127.0.0.1`; `prometheus_client.start_http_server` defaults to `0.0.0.0`, which is exactly the behaviour that made this a defect.

**Behaviour changes worth knowing.**
- `app.py` no longer reads `METRICS_PORT` with a `"8000"` fallback. That fallback meant the env default silently overrode a configured port. The variable still overrides the port, now resolved in one place, but it no longer *enables* the listener.
- The unimplemented 2026-08-12 launch-diagnostics plan proposed "environment port alone opts in". That conflicts with AC#1, which requires a config setting, so the AC was followed and the divergence is recorded here rather than decided silently. If env-as-consent is wanted, it is a one-line change and an owner call.
- When metrics are disabled the function logs at DEBUG, not INFO: a feature the user never turned on should not narrate itself at startup. A new test pins that.

**Existing tests updated, not bypassed.** Three tests in `test_startup_metric_outcomes.py` pinned the old contract and failed. Each protects a real invariant that was preserved: dependency-unavailable still returns False with an honest message (now reachable only when enabled), a start failure still propagates with no success log, and an explicit port is still honoured. Their stubs took `(port)` only and were widened for the `addr` keyword.

**Verification.** 22 tests pass in `Tests/Metrics/`; 268 pass across `Tests/Metrics/`, `Tests/App/`, `Tests/Utils/test_metrics_logger.py`. Three collection errors under `Tests/Terminal/` are baseline (`No module named 'pyte'`), confirmed by re-collecting with the changes stashed.

Because `prometheus_client` is not in the project venv, the socket-level criteria were verified against the real dependency in an isolated venv on `PYTHONPATH`:
- AC#2: disabled -> returns False, nothing listening on the port. Enabled -> returns True and the port accepts a connection.
- AC#3: `lsof -nP -iTCP:8758 -sTCP:LISTEN` reported `TCP 127.0.0.1:8758 (LISTEN)` - loopback, not `*:8758`.
- AC#5: with the listener off, a counter recorded 3.0 and a histogram count 1.0 in the real registry. The in-suite test only proves the call path stays reachable, since the metric classes are no-op stand-ins without the dependency; its docstring says so.

A trap worth repeating: the first socket check ran a script from the scratchpad and silently imported `tldw_chatbook` from the **main checkout** via the editable install, because `sys.path[0]` is the script's directory. It reported a passing-looking result for code that was never under test. Verification scripts must put the worktree on `PYTHONPATH` and assert `module.__file__` before trusting anything.

**Files:** `tldw_chatbook/Metrics/metrics.py` (gate, config resolution, imports), `tldw_chatbook/app.py` (call site), `tldw_chatbook/config.py` (`[metrics]` block in `CONFIG_TOML_CONTENT`), `Tests/Metrics/test_metrics_server_gate.py` (new, 8 tests), `Tests/Metrics/test_startup_metric_outcomes.py` (3 updated, 1 added).
<!-- SECTION:NOTES:END -->
