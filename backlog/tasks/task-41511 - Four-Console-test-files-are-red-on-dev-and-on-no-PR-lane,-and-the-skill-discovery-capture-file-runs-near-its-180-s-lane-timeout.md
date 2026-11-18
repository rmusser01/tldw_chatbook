---
id: TASK-41511
title: "Four Console test files are red on dev and on no PR lane, and the skill-discovery capture file runs near its 180 s lane timeout"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:31
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Performance/test_console_skill_discovery_genuine_capture.py
- Tests/Chat/test_console_compaction_hold_cardless_sends.py
- Tests/Chat/test_polling_static_update_idempotence.py
- Tests/Chat/test_console_staged_context.py
dependencies:
- TASK-41140
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 15 by the console, console-review and testinfra streams:
- `Tests/Performance/test_console_skill_discovery_genuine_capture.py` (9 cases, about 100 s at `-n 4`, slowest 155.8 s against the 180 s lane timeout) is named in no workflow or census (checked at the stack head); TASK-41140 repaired its three red cases but did not lane it;
- `test_console_compaction_hold_cardless_sends.py` has 4 red nodes at base (TASK-33621.4 area);
- `test_polling_static_update_idempotence.py` has 2 red nodes at base (TASK-40315 area);
- `test_console_staged_context.py::test_staged_context_summary_normalizes_launch_text` expects `&amp;`, but TASK-32802.4 removed the escaping; it is red and in no census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each file passes under a plain run
- [x] #2 Each is on a PR lane (or the notes say why not), with its measured time
- [x] #3 The capture file's slowest case is under 120 s at the lane's flags, or its timeout is raised with evidence
<!-- AC:END -->

## Implementation Plan

1. Reproduce each file's red nodes and find the cause (harness drift vs product).
2. Fix the tests (harness, waits, stale expectations) so each file passes plainly; product changes only if a red node is a product bug.
3. Lane each file (UI census tail for Tests/UI, a whole-file target for the Performance file) with its measured time under the lane's flags.
4. Measure the capture file's slowest case at the lane's flags; bring it under 120 s or raise its timeout with evidence.

## Implementation Notes

The three Console files are under `Tests/UI/`, not `Tests/Chat/` as the description says. At base, plain `-n 4`: 7 failed, 22 passed across them, the counts the description gives.

- **`test_console_staged_context.py::test_staged_context_summary_normalizes_launch_text`.** TASK-32802.4 removed the HTML escaping on purpose: every sink renders literally, so `R&D` showed as `R&amp;D`. The test now asserts the summary is exactly `Q&A <notes> (Library Search/RAG, staged)`. Red at base (`'&amp;' in 'Q&A <notes> ...'`), green after.
- **`test_polling_static_update_idempotence.py::test_buddy_conversation_poll_keeps_unchanged_text[True|False]`.** The test called `refresh_projection()`, whose default `force=True` has repainted every Static since 077e5bfa09, by design: direct calls such as mount, resume and send always render. The poll's own call is `refresh_projection(force=False)` (`_on_poll_tick`). The test now drives that and still checks that a changed notice repaints only the notice. Red at base ("Unchanged conversation repaints"), green after.
- **`test_console_compaction_hold_cardless_sends.py`, 4 nodes.** Three harness faults in the shared helpers of `test_console_compaction_hold_flow.py`, plus one race in the queued test:
  1. `_ask_console` watched only `runtime.accept_turn`. A visible Send is admitted through `accept_received_intent`, so `_send` returned before the turn ran, and the hold loop read "no provider call" as a hold. It now watches both.
  2. `_held_console` built the app on a factory-made data folder, so the MCP store stayed unbound. With the agent runtime on (the default), every Send then stopped in `Agents/native_source_storage.py`'s `_AgentSources.__init__` with `RecoveryRequired('mcp_source_not_installed')`. Runtime-exception spy (reverted) on the first send of the Buddy case: 2 of 2 runs. It now uses the profile's own data folder, as the app does. The product side is a proposed follow-up (TASK-40714's "Send without a bound MCP store" holds only with the agent runtime off).
  3. The first reply's first-send onboarding write (`console.onboarding.first_send_completed`) landed while the next Send was in flight. That Send read "Draft or chat changed; Send again." (notices captured in a temporary probe). The harness now saves onboarding as done with its other settings.
  4. The queued follow-up reaches the queue from its receipt's own task (`queue_prompt` in `console_received_dispatch`), not before Send returns. The queued test now waits for it, up to 10 s.

  With these, the cardless file passed 4 of 4 in two `-n 4` runs (load 20-46) and in the serial lane measurement; the queued node alone passed 3 of 3. As a side effect `test_console_compaction_hold_flow.py` went from 8 red to 1 (a stale tooltip copy assertion, proposed follow-up).
- **`Tests/Performance/test_console_skill_discovery_genuine_capture.py`.** It was green at base. Its slow case was load: serially with the lane's flags (`--timeout=300 --tb=short`) at load 22-36, 9 passed in 103 s, slowest case 13.6 s (`[plain_closing]`). **AC#3, met**: under 120 s, timeout unchanged.

**Lanes (AC#2).**
- The three UI files are whole entries at the UI census tail, timed serially with the UI lane's flags at load 10-12: staged context 17 passed in 3.1 s, Buddy poll 8 in 4.2 s, cardless holds 4 in 92.3 s.
- The capture file (bootstrap_profile) runs whole in `console-p0-gate`'s admission-sensitive step, beside TASK-41508's trace files. The whole step serially, as CI runs it: 453 passed in 1464 s at load 15-27. `recovered-suites`, where its sibling stock-owned nodes run, measured 11.8-15.9 min of its 20 on CI.

Files: `Tests/UI/test_console_staged_context.py`, `Tests/UI/test_polling_static_update_idempotence.py`, `Tests/UI/test_console_compaction_hold_cardless_sends.py`, `Tests/UI/test_console_compaction_hold_flow.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_response_rules_critical_gate.py`.
