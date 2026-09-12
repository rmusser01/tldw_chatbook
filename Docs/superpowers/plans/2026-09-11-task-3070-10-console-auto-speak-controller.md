# TASK-3070.10 Console auto-speak ownership implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Move the five reviewed auto-speak entry points into HandsFree ownership while preserving destination resolution, consent, resume/retry, queue and control presentation behavior.

**Architecture:** Extend the existing ConsoleHandsFreeController with two moved methods and three command targets. Keep the decorated Textual handlers as bounded Screen delegates. A named callback defined in screen wiring performs the existing control-bar projection; the controller never queries the DOM. The existing ConsoleAutoSpeakCoordinator remains the consent/queue/lifecycle owner.

**Tech Stack:** Python 3.11+, Textual 8, pytest, Ruff.

**Spec:** `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`, auto-speak boundary/inventory; `DESIGN.md` section 7.

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md (existing)
Reason: Direct implementation of the approved Wave 6 controller contract, preserving existing speech/session policy. No new storage, provider, security or runtime boundary.

## Global Constraints

- A region widget owns pixels; a controller owns non-DOM state and behaviour.
- Controllers do not use `query_one`; screen/region code passes data or named operations through explicit callables.
- Dependencies are named, keyword-only constructor arguments wired as late-binding callables in `UI/Console_Modules/wiring.py`.
- Cross-controller traffic uses those named callables, never a controller reaching through the screen to a sibling controller.
- Every state name that was a plain assignable screen attribute retains read/write proxy compatibility, whether or not the current caller scan observes an assignment.
- Existing worker group names, cancellation ownership, persistence ordering, and remount/shutdown behaviour are preserved.
- Preserve resolver absence/exception semantics exactly: absent handler factory/resolver returns None; factory exceptions propagate; resolver exceptions return None; cancellation propagates.
- The three decorated Screen entry points stop the event and call HandsFree once, with complete definition spans of at most five physical lines excluding decorators.
- Preserve boolean values, retry default False, notification text, consent/destination fences and all existing coordinator state. No provider calls, new logs or persisted data.
- Work in the authorized shared checkout using immutable before-images; no staging, commits, branch switching, rebase, full-suite runs, network or host cleanup. Recheck scoped hashes before closeout.
- Never raise a ratchet. Current starting Screen is 17,209 lines / 584 methods versus caps 17,169 / 584; preserve unrelated changes. Lower each cap only if the final measured dimension earns a lower value, and report any remaining excess.
- Existing unrelated HandsFree `_sync_hands_free_switch` DOM access is recorded separately; do not widen this task into its redesign. Update only ownership prose made inaccurate by this extraction.
- Parent TASK-3070 and final rebased-wave closeout TASK-3070.11 remain open.

### Task 1: Extract auto-speak routing and presentation boundary

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/hands_free.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`, `tldw_chatbook/UI/Screens/chat_screen.py`.
- Create: `Tests/UI/test_console_auto_speak_controller.py` (plain fakes, no mount).
- Modify: `Tests/UI/test_console_controller_wiring.py`, `Tests/UI/test_console_auto_speak_wiring.py`, `Tests/Architecture/test_console_wave6_inventory.py`, `Tests/Architecture/test_screen_size_ratchet.py` as required by the boundary.
- Existing relevant tests: `Tests/UI/test_console_hands_free_wiring.py`, `Tests/Chat/test_console_auto_speak.py`, auto-speak control widget tests located by caller scan.
- Audit only affected diagnostic inventory rows in `Docs/security/production-diagnostic-inventory.json`; this move contains no logging calls, so no row update is expected.
- Parent updates this plan, task notes and `backlog/docs/agent-orchestration-review-2026-09-07.md`.

**Interfaces:**

Retain the exact destination resolver body under its original name in HandsFree, importing CharacterRef from its defining module. Reuse the existing stable app_instance dependency. Add four required keyword-only callbacks and document them:

```python
request_auto_speak_enabled: Callable[[bool], None]
request_auto_speak_resume: Callable[[], None]
request_auto_speak_retry: Callable[[], None]
sync_auto_speak_controls: Callable[[bool, bool, bool], None]
```

Store callbacks under corresponding private `_..._fn` names. HandsFree's `_sync_console_auto_speak_controls(enabled, paused, retry_available=False)` forwards the three values unchanged. The three controller targets are `on_console_auto_speak_changed(enabled: bool)`, `on_console_auto_speak_resume_requested()` and `on_console_auto_speak_retry_requested()`. Each calls its corresponding command dependency once. No event object or widget crosses that controller boundary.

Screen handlers keep their existing event types/decorators and `event.stop()` before the following target:

```python
self._hands_free.on_console_auto_speak_changed(event.enabled)
self._hands_free.on_console_auto_speak_resume_requested()
self._hands_free.on_console_auto_speak_retry_requested()
```

Define `_sync_console_auto_speak_controls` as a local callback in `build_console_controllers`, before HandsFree construction, containing the original query/QueryError/sync_auto_speak projection with `screen` in place of `self`. This follows the existing named local presentation callback in that wiring function. Inject the local callable; its screen query resolves at call time. Inject the three coordinator commands as lambdas that resolve `screen._console_auto_speak` at call time, since the coordinator is constructed later. Repoint the coordinator's destination/control callbacks to `screen._hands_free` at call time. Do not move coordinator mount/unmount or scheduling, and do not introduce compatibility aliases for the two M names on Screen.

- [x] **Step 1: Characterize the current checkout.** Snapshot all touched files before editing and save baseline hashes. Record exact Screen counts, relevant policy ASTs and diagnostic call atoms. Run the existing auto-speak wiring, HandsFree wiring, controller wiring and pure auto-speak modules plus the two size-ratchet cases; retain commands/JUnit. Attribute failures using those before-images, not the dirty-tree label.

- [x] **Step 2: Establish failing ownership and behavior probes.** Add a completed-owner test using the existing inventory helpers:

```python
group = WAVE6_GROUPS["auto_speak"]
screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)
assert not (group.moved & screen_methods.keys())
assert group.moved | group.delegates <= target_methods.keys()
```

Require exact HandsFree calls, event stop ordering, decorators and five-line spans for all three delegates; reject query_one or sibling reach-through in the five moved/controller targets. Add plain-fake resolver cases for absent/noncallable factory, absent/noncallable resolver, exact CharacterRef identity/arguments, await ordering, factory exception propagation, resolver exception containment and cancellation propagation. Call-recording callback tests require untouched values/default False, one command call, and command exception propagation. Run the new assertions RED before production movement.

- [x] **Step 3: Move and wire the boundary.** Apply the interfaces above, preserving original resolver and widget projection bodies. Repoint real callers/patch handles to their defining owner. Required dependencies must be supplied explicitly; do not add default no-ops to hide stale constructor fixtures.

- [x] **Step 4: Prove actual wiring and integration.** Add tests that replace coordinator and HandsFree targets after wiring, exercise the real Screen event handlers, and observe event.stop before commands. Query the control bar through the injected projection, with a replaced query function and an absent bar; verify default retry False and all three flags. Add a small mounted real Screen/ConsoleControlBar test emitting the three real events and observing the wired commands and rendered enabled/paused/retry controls, with explicit settlement. Run the no-mount module, existing auto-speak/HandsFree/controller wiring, pure decision table, control-bar tests, complete Wave6 inventory and ratchet. Mutation-check each of the three Screen delegations and a resolver argument/exception distinction against the behavioral oracles, then restore exact bytes. Keep stateful coordinator queue/consent tests unchanged.

- [x] **Step 5: Verify and close only this child.** Compare resolver/projection ASTs after documented substitutions. Conserve diagnostic atoms and compare only affected inventory rows, disclosing unrelated global drift. Compare Ruff at identical filenames against before-images; format owned ranges, compile touched files, and run scoped diff checks. Measure final Screen counts and only lower earned caps. Self-review the before-image diff, obtain independent task and final integration reviews, address findings, then update all three AC/notes/status through Backlog CLI and this plan/public issue ledger. Retain uncommitted evidence; do not claim final parent/rebased-wave completion.

## Completion evidence

TASK-3070.10 is Done. Five HandsFree targets and four explicit callbacks now own
the reviewed boundary. Screen retains stop-first decorated delegates with spans
3/5/5. Resolver AST is exact; projection body matches after self-to-screen
substitution. All 94 diagnostic calls/sinks are conserved and the inventory was
not edited. The existing coordinator's consent, queue and lifecycle are unchanged.

The final targeted matrix reports 229 passes and one characterized size failure:
Screen 17,171 lines / 582 methods versus caps 17,169 / 582, improving the baseline
40-line excess to two. All six mutations are killed. Task review found only a
Minor formatter issue; the correction preserves all three affected module ASTs
and passes 17 focused cases. Scoped re-review and final integration review approved
with no remaining findings. Lint introduces no new findings (153 inherited
findings become 151), owned formatting/compile/diff checks pass, and inherited
whole-file formatting debt remains disclosed.

Global diagnostic drift is identical to baseline and confined to two Settings
owners. The older HandsFree Switch DOM access remains separately recorded.
No full-suite, provider/network, staging/commit or host cleanup was performed.
Parent TASK-3070 remains open for final rebased-wave closeout. Existing ADR-033
and approved Wave 6 / DESIGN.md section 7 apply; no new ADR was required.
Evidence remains in
`.superpowers/sdd/2026-09-11-task-3070-10-console-auto-speak-controller/`.
