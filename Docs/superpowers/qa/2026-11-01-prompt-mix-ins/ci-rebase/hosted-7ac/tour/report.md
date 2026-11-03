# 7ac Settings destination timing investigation

Head: `7ac214599d4969a4d1a57db8bef41ff02eb433c9`.
Comparison: `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

## Verdict

No repeatable Settings regression or production fix candidate established. The exact local destination-tour test passed once, with F4 Settings at 1.5164 seconds against the unchanged 10.0-second ceiling. The original hosted failure remains real: its rounded elapsed was 10.1 seconds. A single same-head hosted job rerun is appropriate diagnostic evidence; it does not erase the initial failure or establish a general performance guarantee. Parent started that rerun independently. Its result is not yet part of this report.

No repository code, test, threshold, startup policy, quiet period, or readiness assertion was changed. The only current repository working-tree change belongs to the parent's reopened Backlog task.

## Path and source comparison

The Settings screen, app composition, navigation/lifecycle, shell shortcuts, route registry, bundled CSS, and exact tour test are byte-identical to f095 (eight SHA-256 comparisons in `source-proof.json`). Only two production files changed between the heads: first-send module inventory and web-tool footer lazy loading (`production-diff.patch`). The tour sets `PYTEST_CURRENT_TEST`; `_screen_preimport_enabled` therefore disables background screen/first-send preimport absent an explicit environment override. The fixture supplies no override. The changed footer delegate is invoked at the end of deep search, not by the Settings route.

F4 is the existing shell-destination binding, which calls `action_shell_destination('settings')`, posts `NavigateToScreen`, and enters the existing admitted navigation worker/transition lock. The route resolves `SettingsScreen`, constructs a new instance, restores the existing view snapshot, and switches/mounts it. Settings is not an installed reusable route. Its mount schedules the existing prompt count, subscription readiness, theme subscription, responsive layout, sync rows, provider picker, and contextual refresh work. Ctrl+9 earlier in the same tour resolves the MCP compatibility alias to Settings. Nothing in these paths changed since f095.

The measured ceiling includes `Pilot.press`, arrival, and six existing settle passes (each sleep plus Pilot.pause), so the CI assertion alone does not locate a slow callback. It proves arrival occurred, followed by an over-budget total. No CI per-phase profiler data was available; no cause such as CPU contention is asserted as proven.

## Evidence

- f095 hosted destination tour: PASS; its latency group was 13 passed in 218.13s. That job later failed unrelated stale contract nodes.
- 7ac first hosted attempt: destination tour FAIL at F4 Settings 10.1s; CSS tour and seeded Library passed; latency group 1 failed / 12 passed in 237.50s.
- Current local exact node: 1 passed in 47.03s. All eight destinations arrived. F4: 1.0291s key handling, 0.4873s original arrival/settle helper, 1.5164s combined. Ctrl+9 Settings: 1.5417s combined. Maximum local route total: Console 6.5682s.

Command (single targeted node, no suite expansion):

```sh
PYTHONPATH=/private/tmp/mixins-7ac-tour:$PWD PYTEST_PLUGINS=tour_observer /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Performance/test_ui_latency_guardrails.py::test_destination_tour_stays_under_switch_budgets -q --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixins-7ac-tour/pytest --junitxml=/private/tmp/mixins-7ac-tour/result.xml
```

The private-profile subprocess remains the test's original mechanism. The observer records monotonic timestamps around the original Pilot.press and original wait helper, forwarding exact arguments/results without altering readiness, timeouts, waits, or assertions. It buffers observations and writes only at session completion. This is a local instrumented timing observation, not a substitute for the same-head hosted rerun and not an exact cross-machine benchmark. No heavyweight profiler was added after the failure did not reproduce. After completion, the core agent reported a potentially overlapping TldwCli-constructor/backup subprocess test (no mounted UI); therefore these local timings are diagnostic observations, not isolated performance qualification. The exact local node still passed its original assertion. No rerun is added merely to replace that qualified evidence.

All local test processes stopped. Owned artifacts are explicitly enumerated and hashed in `artifact-files.json`; the two original CI logs are external input artifacts, not modified files.
