---
id: TASK-25914
title: Gate the Prometheus metrics listener behind explicit config
status: To Do
assignee: []
created_date: '2026-08-31 15:11'
updated_date: '2026-08-31 15:11'
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
- [ ] #1 The metrics listener starts only when a config setting explicitly enables it; the setting defaults to off
- [ ] #2 With the dependency installed and the setting off, no socket is bound - verified by a test or a documented ss/lsof check
- [ ] #3 The bind address is configurable and defaults to loopback rather than all interfaces
- [ ] #4 When the listener does start, the fact and the port are stated in the log at a level the user will see
- [ ] #5 Existing metric collection is unaffected when the listener is off - counters and histograms still record
<!-- AC:END -->
