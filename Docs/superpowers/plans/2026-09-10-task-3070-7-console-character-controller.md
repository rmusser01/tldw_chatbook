# TASK-3070.7 Console character controller implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete the existing character controller boundary and restore the Console size gate by moving the remaining character selection and identity policy out of ChatScreen.

**Architecture:** Extend the existing ConsoleCharacterController, whose avatar refresh state and compatibility descriptors are already implemented. Wire non-DOM dependencies explicitly and lazily; keep picker modal presentation, worker admission, and rail rendering on the screen. Preserve session, prompt, visual identity, and notification behavior.

**Tech Stack:** Python 3.11+, Textual 8, pytest, SQLite, Ruff.

**Spec:** `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`, character boundary and source-inspected inventory; `DESIGN.md` section 7.

ADR required: no
ADR path: N/A
Reason: Direct implementation of the existing approved controller boundary; no new service, runtime, security, storage, or UX contract.

## Global Constraints

- A region widget owns pixels; a controller owns non-DOM state and behaviour.
- Dependencies are named, keyword-only constructor arguments wired as late-binding callables in `UI/Console_Modules/wiring.py`.
- Cross-controller traffic uses those named callables, never a controller reaching through the screen to a sibling controller.
- Existing worker group names, cancellation ownership, persistence ordering, and remount/shutdown behaviour are preserved.
- No attachment bytes, prompts, paths, message/session IDs, signed URLs, provider payloads, or exception messages may be added to persistent diagnostics.
- Every character M method must be absent from ChatScreen. Existing avatar refresh and four compatibility descriptors stay controller-owned; the deleted expression-fetch path stays deleted.
- The picker callback remains a presentation/worker boundary, preserving `exclusive=True` and `group="console-character-pick"`. Preserve all notification strings and best-effort error containment.
- Continue in the existing authorized shared checkout. Snapshot starting files and review this task's before-image diff; do not stage, commit, reset, switch branches, or overwrite unrelated changes.
- Run targeted checks only. Do not run the full suite, real providers, network fetches, or host-resource cleanup. Do not claim final rebased-dev completion of parent TASK-3070.

### Task 1: Finish character selection and identity ownership

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/character.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`, `tldw_chatbook/UI/Screens/chat_screen.py`.
- Add: `Tests/UI/test_console_character_controller.py` (plain fakes; no mounted Textual).
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`, `Tests/Architecture/test_screen_size_ratchet.py`.
- Repoint existing tests that directly call or patch moved private methods to their real owner, preserving all behavioral assertions. Search the exact six names below and `ConsoleCharacterController` in `Tests/` before movement.
- Update only the affected diagnostic inventory rows after conservation checks. Parent owns task notes and consolidated orchestration ledger.

**Interfaces:**
- Move `_console_character_picker_options`, `_current_console_rail_conversation_id`, `_current_console_rail_character_id`, `_current_console_rail_character_name`, `_fetch_character_card_for_avatar`, `_apply_console_character_choice_async` from ChatScreen to the existing character controller. Preserve parameter and result contracts.
- Use named late-bound accessors for the DB, active native session, persisted conversation fallback, lazy chat store, provider readiness config, default session settings, session character swap, notifications, temporary-chip sync, and native UI sync. Read current character name directly from the controller's owned identity method instead of a screen round trip.
- Keep avatar config's current app-config source separate from selection's provider-readiness config source; they currently have distinct call sites.
- Existing screen picker uses `await asyncio.to_thread(self._character._console_character_picker_options)` and current-character lookup from that owner. Worker callback calls `self._character._apply_console_character_choice_async(choice)` with the existing group/exclusive settings. Other screen, wiring and test consumers target the controller directly.
- Import prompt seed, character ID/display helpers and choice/option value objects from their defining modules. No screen imports or re-export aliases in the controller.

- [x] **Step 1: Characterize and pin the starting tree.** Preserve before-images and current hashes, then run the existing character prompt seed/avatar/picker and narrow architecture tests. Record the known line/method gate failure separately. Read the actual six method bodies before editing: preserve the 500-card cap, 200-character description projection, native-only character identity, and missing-card refusal.

- [x] **Step 2: Add a failing ownership regression and isolated controller behavior tests.** Add a completed character-family AST assertion patterned after the existing completed retrieval-family assertion:

```python
group = WAVE6_GROUPS["character"]
screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)
assert not (group.moved & screen_methods.keys())
assert group.moved <= target_methods.keys()
```

Run it before production movement and preserve the expected failure naming the six remaining screen-owned methods. Plain-fake controller tests must cover DB absent/error and filtered picker values; live native identity versus fallback; missing card; new-session prompt and roleplay-template seed before switch; temporary-chip sync; successful swap versus refused swap; awaited UI sync followed by avatar refresh. Assert observable data and call ordering outside fail-soft handlers. Preserve real SQLite/mounted integration tests for the actual store and pixels.

- [x] **Step 3: Move the six methods and rewire real consumers.** Preserve each policy body and its error branches while replacing screen access with the named dependencies above. Keep controller dependencies free of DOM and sibling handles. Update existing controller constructor fixtures and the exact-dependency AST test to the final explicit signature. Do not create fake Screen aliases to preserve stale tests or remove behavior assertions.

- [x] **Step 4: Verify behavior and earned architecture reduction.** Run the new isolated module and complete affected character prompt/avatar/picker/controller-wiring suites, then real character handoff/composer/rail/scope consumers identified by moved-method references. Run the Wave 6 inventory, screen-size ratchet, persistent diagnostic inventory and focused CSS/decomposition gates. Prove the new completed-owner oracle detects a moved method returning to Screen and the callback integration detects losing the worker dispatch. Compare Ruff findings with before-images, format changed ranges only, compile touched Python, and verify diagnostic-call conservation with only justified per-file inventory changes. Set the Console ratchet down to exact final measured line/method counts; never raise a budget. Save commands, output and hashes in the task report.

- [x] **Step 5: Review and close this child.** Self-review the task's before-image diff, then obtain independent spec/code review and final scoped review before marking all four child AC checked and Done via the Backlog CLI. Record the earned counts and targeted evidence in the consolidated ledger. Parent TASK-3070 stays open because first-chat, auto-speak and final rebased-wave closeout remain separate tasks.

### Verification amendment: native history fixture

The affected fleet-history navigation test fails in both the original-file overlay and extracted working tree because it binds its real bridge only after native startup recovery has run without a run database. Recovery records `history_unavailable`, and the fixture therefore tests the failure state while asserting the normal four-row preview. A scratch pre-mount binding restores both size variants without production changes.

TASK-3070.7 AC #4 now requires successful native recovery in these affected mounted checks. Bind the existing real bridge with `ensure_console_runtime(app).set_agent_bridge(bridge)` before mounting this test's host, then positively await/assert `console._ensure_console_chat_controller().fleet_wake.wait_for_recovery()` after setup. Preserve all original preview, paging, pointer/keyboard selection and drilldown assertions. Keep the repair local to this test; do not suppress the production recovery error, alter its safety policy, or broadly change other fixtures. Record the before-overlay RED and mounted GREEN in the report. Completion now checks all four AC.

## Completion evidence

TASK-3070.7 is complete. Task and final integration reviews approved with no
Critical/Important findings. Four stale ownership comments were corrected and
passed scoped re-review; their executable ASTs are unchanged. The final ratchet
is 17,483 lines / 588 methods. Targeted verification has 323 distinct passing
latest outcomes plus the separate 49-case avatar run. Existing static debt,
unrelated global diagnostic inventory drift, and test infrastructure warnings
remain disclosed in the task notes and orchestration ledger. No new ADR required;
this implements the approved Wave 6 / DESIGN.md section 7 boundary. Parent
TASK-3070 remains open. Exact diffs, commands, review reports and hashes are retained
in `.superpowers/sdd/2026-09-10-task-3070-7-console-character-controller/`.
