---
id: TASK-25914
title: Gate the Prometheus metrics listener behind explicit config
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:11'
updated_date: '2026-08-31 16:52'
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

**Approach.** The gate lives in `init_metrics_server`, not in `app.py`, so any future caller inherits it, and it is checked *before* `PROMETHEUS_AVAILABLE` so a missing dependency can never mask a broken gate. `_metrics_server_config()` resolves the three settings through `_get_cli_setting`, a thin indirection that keeps the `config` import lazy (this module is imported early) and lets tests exercise resolution with no config file on disk. Bind address defaults to `127.0.0.1`; `prometheus_client.start_http_server` defaults to `0.0.0.0`, which is what made this a defect rather than a preference.

**Every value fails closed.** Coercion goes through the shared `config.coerce_bool_setting` / `coerce_int_setting` helpers rather than builtins. This matters: `bool("false")` is `True` in Python, so a user quoting the boolean out of YAML or env habit would have meant *off* and got an unauthenticated listener - the same fail-open shape this task exists to remove, relocated from dependency-presence to value coercion. Caught in review, not by me. A non-string or empty `bind_address` also falls back to loopback, because `str(0)` is `"0"` and `getaddrinfo` resolves that to `0.0.0.0`.

**Behaviour changes worth knowing.**
- Default port is now 9090 (Prometheus convention). 8000 collided with `[web_server] port`; both bound localhost:8000 and whichever started second would fail.
- A non-loopback bind logs at WARNING, not INFO, naming the address - exposing an unauthenticated endpoint to the network should be hard to miss.
- `app.py` no longer reads `METRICS_PORT` with a `"8000"` fallback. `METRICS_PORT` still overrides the port, resolved in one place, and a junk value now falls back to the *configured* port rather than the built-in default. It does not enable the listener. (Correction to an earlier draft of these notes: that fallback could not previously have overridden "a configured port", because `[metrics]` did not exist before this change. Removing it is still right; the justification was retroactive.)
- The unimplemented 2026-08-12 launch-diagnostics plan proposed "environment port alone opts in". That conflicts with AC#1, so the AC was followed and the divergence recorded rather than decided silently. Making env-as-consent is a one-line change and an owner call.
- When metrics are disabled the function logs at DEBUG: a feature the user never enabled should not narrate itself at startup.

**Existing tests updated, not bypassed.** Three tests in `test_startup_metric_outcomes.py` pinned the old contract. Each protects a real invariant that was preserved - dependency-unavailable returns False with an honest message, a start failure propagates with no success log, an explicit port is honoured - and their stubs were widened for the `addr` keyword. The rewrites are strictly stronger: `calls == [8123]` became `calls == [(8123, "127.0.0.1")]`, pinning the loopback default at the call boundary.

**Verification.** 295 tests pass across `Tests/Metrics/`, `Tests/App/`, `Tests/Utils/test_metrics_logger.py` and `Tests/Packaging/test_config_import_closure.py`. Three collection errors under `Tests/Terminal/` are baseline (`No module named 'pyte'`), confirmed by re-collecting with the changes stashed.

`prometheus_client` is not in the project venv, so the socket-level criteria were verified against the real dependency in an isolated venv on `PYTHONPATH`:
- AC#2: disabled -> returns False, nothing listening. Enabled -> returns True, port accepts a connection.
- AC#3: `lsof -nP -iTCP:8758 -sTCP:LISTEN` reported `TCP 127.0.0.1:8758 (LISTEN)` - loopback, not `*:8758`.
- AC#5: with the listener off, a counter recorded 3.0 and a histogram count 1.0 in the real registry.
- Post-review: `enabled = "false"` returns False and binds nothing, against the real dependency.

**Review round.** A code review found one Important fail-open (the `bool()` coercion above) plus eight Minor items; all were fixed rather than deferred, since they were the same class of defect as the one being closed: unusable `bind_address`/`port` values now fall back instead of reaching the socket layer, junk `METRICS_PORT` no longer discards a configured port, non-loopback binds warn, the port default moved off the `[web_server]` collision, the unused `addr` parameter was dropped, and stale references in the sample comment, `Docs/Development/Metrics/STARTUP_METRICS_SUMMARY.md` and a dead `test_startup_init_hygiene.py` fixture entry were cleaned up. The review also showed two tests were asserting `default == default` through a stubbed resolver, so they pinned module constants rather than shipped behaviour; a template-parse test and an out-of-process test that reads a real config file through the real loader were added. That second one matters because every other test here stubs the resolver and would stay green if the `get_cli_setting` lookup shape silently stopped resolving - a failure this repo has had before (TASK-1771's dotted-section trap).

A trap worth repeating: the first socket check ran a script from the scratchpad and silently imported `tldw_chatbook` from the **main checkout** via the editable install, because `sys.path[0]` is the script's directory. It reported a passing-looking result for code that was never under test. Verification scripts must put the worktree on `PYTHONPATH` and assert `module.__file__` before trusting anything.

**Files:** `tldw_chatbook/Metrics/metrics.py`, `tldw_chatbook/app.py`, `tldw_chatbook/config.py`, `Docs/Development/Metrics/STARTUP_METRICS_SUMMARY.md`, `Tests/Metrics/test_metrics_server_gate.py` (new), `Tests/Metrics/test_startup_metric_outcomes.py`, `Tests/App/test_startup_init_hygiene.py`.
<!-- SECTION:NOTES:END -->
