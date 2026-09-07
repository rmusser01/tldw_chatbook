---
id: TASK-19160
title: >-
  One failing realtime test crashes both Core CI jobs via raw LogRecords in
  xdist reports
status: Done
assignee: []
created_date: '2026-08-20'
labels: [ci, testing, triage]
dependencies: []
priority: high
---

## Description (the why)

Both Core Tests jobs on dev die with an xdist INTERNALERROR —
`execnet.gateway_base.DumpError: can't serialize <class
'websockets.asyncio.server.ServerConnection'>` — whenever a test in
`Tests/LLM_Calls/test_openai_realtime_session.py` fails on the runner. The
crash aborts the whole job's reporting, so every other suite's result in
that job is garbage. This cluster is NOT in TASK-18610's pass-2 inventory;
it postdates it (the realtime suite is new).

**The mechanism, reproduced deterministically:**

1. pytest-json-report's `_capture_log` stores **raw `logging.LogRecord`
   objects** per test phase and attaches them to the report
   (`report._json_report_extra`).
2. The websockets library logs through a `LoggerAdapter` whose extra
   carries the **live connection object**, so any captured websockets
   record embeds a `ServerConnection`/`ClientConnection`.
3. pytest-xdist serializes `report.__dict__` over execnet, which supports
   only basic types → `DumpError` → the worker dies mid-test → the
   controller hits `assert not crashitem` → INTERNALERROR.
4. Precondition: log level permitting websockets records to reach the
   handler. Locally the default WARNING filters them; in CI an earlier
   test in the same `loadscope` worker leaves DEBUG on.

Repro: force `test_connect_sends_session_update_and_fires_ready` to fail +
`--log-level=DEBUG` + `-n 2 --dist loadscope --json-report` = the exact CI
DumpError. Note the existing in-file guard (`_transport_safe_error`)
covers only the handler's own exception path — the log-capture path walks
around it entirely.

## Acceptance Criteria (the what)

- [x] The crash is reproduced locally with a named mechanism, not inferred
      from CI logs
- [x] Every `--json-report` activation in `.github/workflows/test.yml`
      carries `--json-report-omit log` (core, UI shards, full suite,
      nightly — 4 sites)
- [x] The fix is verified under the exact crash conditions: same forced
      failure + DEBUG + xdist + json-report now reports a clean `1 failed`
      instead of an INTERNALERROR
- [x] Nothing consumed the omitted section:
      `.github/scripts/generate_test_summary.py` reads only
      `tests[].outcome` and `call.longrepr`
- [x] Pinned in the CI contract suite
      (`test_every_json_report_invocation_omits_log_capture`), with an
      inertness backstop (≥4 activations must be found) —
      mutation-verified: removing one omit flag reds the pin

## Implementation Notes

One workflow change (4 lines) plus one contract pin. The alternative —
filtering websockets' loggers in the realtime test module — was rejected
because it fixes one library's records while leaving the class open: ANY
test whose captured logs carry a non-serializable object kills its whole
worker. Omitting log capture closes the class; the log section had no
consumer.

The repo-side guard already present in the test file
(`_transport_safe_error`) was necessary but insufficient: it sanitizes
exceptions the scripted handler catches, but the leak was in json-report's
log capture, a path that never touches the handler's error field.

**Files:** `.github/workflows/test.yml`,
`Tests/CI/test_github_actions_test_workflow.py`.
No production source changed. Realtime suite 36 passed; CI contract 15
passed.
