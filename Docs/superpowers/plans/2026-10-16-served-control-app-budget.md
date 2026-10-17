# Served Control App Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task by task. The user has already authorized fixing and merging the size failures.

**Goal:** Restore the app composition root to its existing size ceiling after newly merged served-relay wiring, preserving the control lifecycle and relay decisions.

**Architecture:** Keep the five original App methods and their signatures. Delegate only their bodies to a deferred, stateless app_served module; App still owns every controller, task, publisher and client. Runtime asyncio lookups consume the original App module namespace at the original execution points, including the late completion callback.

**Tech Stack:** Python 3.12+, Textual 8.x, existing asyncio, pytest and source-governance tools; no new dependency.

**Spec:** TASK-34406.7 acceptance criteria; backlog/decisions/236-served-desktop-approval-relay.md and the existing app module ratchet.

**ADR required:** no. **ADR path:** N/A (existing ADR-097/236). This is mechanical delegation inside the same App ownership boundary, not a new service, permission, transport, storage or runtime contract.

## Global Constraints

- Never raise an existing module ceiling. The current app ceiling is 5712; the incoming source is 5794. Pin the new helper at its actual line count and lower the app pin if its resulting slack requires it.
- No UI styling, timer, deadline, authority, relay protocol or native-admission changes. Native runs preserve original hook identity; served decisions retain the existing armed-round checks and call-through behavior.
- Keep imports at their original demand point. No application/optional package imports from the helper at module scope; asyncio/typing are standard library only.
- Use targeted tests in the already reviewed admitted private-profile launcher; no full local sweep. Preserve and attribute inherited failures.
- PR3139 remains armed and untouched. Publish this follow-up only after its actual merge, rebasing onto then-current dev. Independent review replaces billing-blocked Qodo under existing user authorization; required CI and resolved threads remain mandatory.
- The saved unprofiled and diagnostic packets remain bound to their original heads; this refactor establishes no new latency result.

### Task 1: Delegate served control wiring without changing its owners

**Files:**
- Create: tldw_chatbook/app_served.py
- Modify: tldw_chatbook/app.py, Tests/Architecture/test_module_size_ratchet.py, Tests/Architecture/test_app_extracted_patch_targets.py, .github/workflows/derived-artifacts.yml
- Test: Tests/UI/test_app_startup_delegation.py, Tests/Web_Server/test_canvas_control_spawn.py, Tests/Web_Server/test_served_relay_publisher.py, Tests/Performance/test_app_import_weight.py
- Update: Docs/Architecture/app-shell.md
- Update only if generated owner changes: Docs/security/production-diagnostic-inventory.json

**Interfaces:** App callers and signatures stay _start_served_canvas_control(self), _served_relay_run_states(self), served_relay_wrap_pending_approval(self, original), _served_relay_decide(self, round_id, decisions), and async _stop_served_canvas_control(self). The helper takes the same self and arguments; start/stop additionally take keyword-only app_globals, the issuing App module dictionary. No helper state or exported service is introduced.

- [x] **Step 1: Reproduce the original failure.** The retained exact5845 admitted control ran the real parameterized App ratchet and failed with 5794 > 5712 (+82), not a setup/import error. Keep that RED packet.

- [x] **Step 2: Extend existing behavior controls before moving production.** In test_app_startup_delegation.py, exercise original App methods on finite doubles: native hook identity and served wrapper arguments; current runtime/run-state failure behavior; unknown/settled/active decision rounds and copied decisions; attach-before-task-start, late asyncio cancellation lookup and self-owned warning logger; reset/cancel/join/publisher-close/client-close ordering. These behavior controls should pass on the original bodies; the existing size test is the causal RED for the defect. A late cancellation probe must replace the App module alias after callback registration, so capturing that dependency too early fails the control.

```python
# The original callback must use the CURRENT App module cancellation type.
app_module.TldwCli._start_served_canvas_control(app)
monkeypatch.setattr(app_module, "asyncio", late_asyncio)
completion_callback(cancelled_task)
assert warnings == []
```

Run those controls through the private-profile launcher. Expected: original behavior controls pass; retained App size RED still fails before implementation.

- [x] **Step 3: Move exactly the five bodies.** Copy the existing bodies and docstrings, preserving relative imports, self-owned fields/logger, callback name and all try/except/return/await order. Only these four runtime global reads change in the helper (including the eagerly evaluated callback annotation):

| Existing lookup | Helper lookup at the same execution point |
| --- | --- |
| `asyncio.create_task` | `app_globals["asyncio"].create_task` |
| Eager `asyncio.Task[None]` callback annotation | `app_globals["asyncio"].Task[None]` |
| Late `asyncio.CancelledError` callback handler | `app_globals["asyncio"].CancelledError` |
| `asyncio.gather` during stop | `app_globals["asyncio"].gather` |

The original App has no future-annotations import: preserve the nested callback annotation as an eager App-namespace lookup. The helper must not add future annotations; its only module dependency is typing.Any. The App wrappers are:

```python
def _start_served_canvas_control(self) -> None:
    from .app_served import _start_served_canvas_control
    return _start_served_canvas_control(self, app_globals=globals())

def _served_relay_run_states(self):
    from .app_served import _served_relay_run_states
    return _served_relay_run_states(self)

def served_relay_wrap_pending_approval(self, original):
    from .app_served import _served_relay_wrap_pending_approval
    return _served_relay_wrap_pending_approval(self, original)

def _served_relay_decide(self, round_id, decisions):
    from .app_served import _served_relay_decide
    return _served_relay_decide(self, round_id, decisions)

async def _stop_served_canvas_control(self) -> None:
    from .app_served import _stop_served_canvas_control
    return await _stop_served_canvas_control(self, app_globals=globals())
```

Add concise wrapper docstrings. Compare the helper body ASTs with source5845 after replacing only the four runtime asyncio lookups and helper function name for the public wrapper. Expected: executable body ASTs equal after restoring those four lookups; cleaned docstrings equal (raw multiline indentation may change when moving out of the class); original App signatures and the rest of its AST unchanged.

- [x] **Step 4: Govern the relocation.** Measure both files with len(read_text().splitlines()), pin the helper and never raise the App ceiling. Add app_served to EXTRACTED_MODULES. Run the complete patch-target scanner; only add a specific documented test exemption if the scanner cannot express the deliberately App-issued live namespace route. The current required lane runs three Settings ratchets but omits App ratchets (confirmed on5845). Add the six existing ceiling/slack node selectors for app.py, app_startup.py and app_served.py to its existing fast PR contract; do not alter job permissions, deadlines or other targets. Run those exact selectors locally before publication. Run the diagnostic generator in check mode, then regenerate only the moved log owner if required; verify call/sink counts unchanged against this branch base.

- [x] **Step 5: Verify the connected behavior.** Through the admitted private-profile control run the App/helper ceiling and slack nodes, complete test_app_startup_delegation.py and test_app_extracted_patch_targets.py, original canvas-control-spawn and served-relay-publisher files, and the relevant existing import tests. Run ruff check/format on touched files, comparing inherited findings with exact base when necessary. Expected: new controls and owned checks pass, no introduced lint/format finding; inherited failures must be retained and reproduced or attributed, never called green. Ruling after exact5845 RED: the original parent/child Canvas node retargets the already admitted config; retain its collection-time private profile via the existing per-node bootstrap_profile marker. The test still uses actual gateway/control broker/client authentication and shutdown. Use set_app_global in new task-retirement controls, with no patch-scanner exemption.

- [ ] **Step 6: Commit, independently review, and merge sequentially.** Update the task notes/criteria with actual evidence. Commit the isolated change, create the final review package and dispatch one fresh whole-branch reviewer. Resolve material findings with causal controls. After3139 is actually merged, rebase onto current dev; publish the ready PR with its exact reviewed scope, attach it, check current threads/queue mode, arm auto-merge under the existing independent-review exception and verify actual required checks plus actual merge. Keep the under-one-second parent task open.

## Review Focus

Native and served identity/arguments; no controller or publisher cache; duplicate-start attachment order; delayed completion/cancellation and late App asyncio replacement; current run/round lookup and settled refusal; exact task retirement order; partial App doubles; original optional import failure behavior; helper import weight; unchanged module ceilings and diagnostic call/sink counts. Do not treat source movement or the historical profiling packets as latency improvement evidence.

### Task 2: Keep recovery synchronization behind the real mount boundary

**Trigger:** required Console P0 run38983800930, job117250474298:236 passed/1 failed. The hook-review journey failed because transcript synchronization queried a Trace recovery action before the nested action container had mounted; the original selector and exception are retained in served-app-console-p0-failure-r1.log. PR3141 auto-merge is disabled.

**Files:** Tests/UI/test_console_trace_call_recovery.py; tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py; Tests/CI/test_ci_queue_pressure_contract.py (the existing independent exact target pin for the six planned workflow selectors).

**Interfaces:** keep the two callout classes, builders and sync_recovery signatures. The owning transcript region defers its original fresh builders/queries until mounted; its own on_mount queues a current-state sync through existing call_later (Textual sets is_mounted after the Mount handler returns). No new timer, cache, changed wait or CSS.

**ADR required:** no. **ADR path:** N/A; existing ADR-094/126 apply. Routine partial-mount lifecycle repair in the same view owner; no new storage, provider, permission or runtime interface.

- [x] Step1: add real mount-order controls before production. Hold only the existing nested action-container mount hook, with original mounted controls and finite Event/finally retirement. Confirm title exists while buttons do not, publish a changed or cleared current builder state, call the original region sync, release mounting, then verify automatic current-state replay and actual keyboard callback identity. Cover trace and continuation. Mounted missing children must still raise, rather than being silently swallowed. Expected: original code reaches actual NoMatches at the pending buttons (not an import/setup failure).
- [x] Step2: only in ProviderContinuationTranscriptRegion, return from sync_recovery before is_mounted; on_mount schedules self.sync_recovery with call_later so it executes after the real flag and reads current builders. Keep every original mounted query, state projection and action callback unchanged. Expected: new controls pass; guard-only without mount replay must fail the late-state assertion.
- [ ] Step3: admitted targeted recovery controls, original failed hook-review node and its complete file, provider continuation/trace recovery/temporary capture/compaction/refusal flows, applicable token/bounds checks; no broad local sweep or altered waits. Compare inherited source/lint honestly. Verify original5-method extraction unchanged. Expected: no introduced failure and no new diagnostic calls or CSS delta.
- [ ] Step4: update notes, commit and independently review final whole branch; rebase onto then-current dev, publish with force-with-lease only if necessary, check current threads, re-arm queue and verify actual required CI/merge. Keep prior failed packet and all profiling packets at their original heads.

**Additional Review Focus:** actual partially mounted nested controls, current changed/cleared state replay without polling, Textual Mount flag timing, post-mount missing-control errors, live callback identity, no new authority/cache/timer/deadline/style and unchanged direct callout behavior.

CI companion ruling before correction: the completed fast lane also fails its two exact-invocation controls because Task1 added six workflow selectors without updating FAST_LANE_TARGETS. Preserve hosted and local RED, then add those same six explicit literals at the corresponding place in the independent pin; keep every other target, order, timeout and negative control unchanged. This enforces existing AC4, no new behavior or weakened CI.

Connected-fixture ruling before marker change: connected R1 is135P1F213.20s; the original real-screen recovery test fails raw_source_selection_changed before UI assertions. Exact d8ea200 production baseline with matching admitted App-import setup reproduces1P1F3.06s. Mark only that existing real-config node bootstrap_profile; retain both failed packets and all guards/waits. Tests/UI/test_console_provider_continuation_recovery.py and Docs/Architecture/console.md join Task2 scope.

Final connected-fixture ruling: the admission marker reaches a second failure (immediate callout.display assertion); that same assertion also fails on exact d8ea200 production with identical admission. Restore the original provider test file completely rather than change its waits, state or fixture to qualify this release. The complete connected run remains135P1pre-existing-local-failure, and all4new real mount/keyboard cases plus the exact failed hosted journey and70CI contracts pass. Final qualified task command selects every other provider-recovery function explicitly; the full required UI lanes still execute the unchanged legacy node. This is permitted by AC3 inherited-failure attribution, not a claim that the legacy node passed.

Task2 Ruling: final connected qualification is189P2F287.15s, both compaction-card display assertions; exact d8 production recheck2P69.29s, unchanged candidate recheck1P1F54.23s. Do not call these inherited or exclude them. Inspect the existing transcript gate: recovery synchronization sits after the unchanged-token return, so a held preparation without a message/run token change can be skipped. Add deterministic real mounted-region controls for fresh and cleared trace/continuation state while the expensive message pipeline remains gated, then move only the existing recovery sync ahead of that gate if RED confirms it. Cost is current recovery builders on each transcript sync; no new timer, forced message rebuild, cache, TTL or authority change. ChatScreen, its gate test and architecture join the existing recovery-lifecycle scope before implementation; existing ADR094/126 and TASK34958 transcript contract applies, no new ADR.

Gate causal2F2.62s -> complete gate/compaction/trace21P53.24s, including all3original compaction actions. Only the same existing recovery statement moves before the gate; restored whole-screen AST equal and24390 lines unchanged (base already21over the old24369 ceiling; no budget raised or new line growth). All original gates/forced cadence remain. Retain finalR2189P2F and baseline/candidate rechecks honestly. Final task run adds the complete gate file to the unchanged full connected selection; no compaction exclusions.

Task2 Ruling: finalR3 has196P1F348.14s; every compaction action and new gate/mount case passes. The sole failure is WorkerCancelled from unrelated exclusive console-sync work at host.workers.wait_for_complete(), not the recovery worker. Narrow only that test wait to its existing trace-call-recovery group, keeping the10s deadline, real Enter action, provider/row/tombstone/reply/card assertions and recovery-worker errors. No product cancellation change or exception suppression; no test exclusions. Tests/UI/test_console_surface_refusal_recovery.py joins scope before edit. Cost if wrong is premature action observation, prevented by unchanged end-state assertions and waiting any still-active recovery worker.

Owned-wait correction: all3real retry/send-without/cancel actions pass82.26s; original10s recovery-worker deadline and every provider/persistence/tombstone/reply/card assertion retained. Prior full197-case run has196passes and only the unrelated-global-worker wait failure; keep that failure packet. Final complete197-case cohort will rerun on the committed test correction. Both packets guards0/0, before/after source maps equal;0introduced lint and formatting clean in the corrected test.
