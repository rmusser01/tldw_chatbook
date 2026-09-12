# TASK-3070.9 Console first-chat controller implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete the first-chat handoff ownership boundary while preserving exact claims, generation fences, rollback, retry, privacy, and focus restoration.

**Architecture:** Extend the existing ConsoleSessionController with the eight reviewed first-chat methods. Reuse its live store and UI synchronization dependencies, add explicit named control/focus operations, and keep widget operations on ChatScreen. The wizard and mount/resume callers address the real policy owner.

**Tech Stack:** Python 3.11+, Textual 8, pytest, SQLite, Ruff.

**Spec:** `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`, first-chat boundary and inventory; `DESIGN.md` section 7.

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md (existing)
Reason: Direct implementation of the approved Wave 6 controller boundary. Revisioned handoff ownership and memory-only retry remain governed by ADR-033; no new runtime, storage, security, or service contract.

## Global Constraints

- A region widget owns pixels; a controller owns non-DOM state and behaviour.
- Dependencies are named, keyword-only constructor arguments wired as late-binding callables in `UI/Console_Modules/wiring.py`.
- Cross-controller traffic uses those named callables, never a controller reaching through the screen to a sibling controller.
- Existing worker group names, cancellation ownership, persistence ordering, and remount/shutdown behaviour are preserved.
- Every state name that was a plain assignable screen attribute retains read/write proxy compatibility.
- No attachment bytes, prompts, paths, message/session IDs, signed URLs, provider payloads, or exception messages may be added to persistent diagnostics.
- All eight first-chat M methods must leave ChatScreen. Screen callbacks may read/restore focus and project control values; no renamed policy blobs or compatibility method aliases.
- Preserve the exact pending-handoff claim/revision/generation gates, notification strings, no-overwrite behavior, release fallback, retry semantics, and acknowledgement ordering.
- Continue in the authorized shared checkout; preserve before-images and unrelated edits. No staging, commits, branch switching, rebase, network, provider calls, full-suite run, or host-resource cleanup.
- Keep auto-speak and parent rebased-wave closeout separate. This task never raises a ratchet or claims parent completion.

### Task 1: Move first-chat handoff policy into Session

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`, `tldw_chatbook/UI/Screens/chat_screen.py`.
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` at its eligibility lookup only.
- Create: `Tests/UI/test_console_first_chat_controller.py`, using plain fakes and the real in-memory PendingHandoffStore/ConsoleChatStore where valuable; no Textual mount.
- Modify: `Tests/UI/test_console_session_settings.py`, `Tests/Wizards/test_first_run_setup_wizard.py`, existing Session constructor fixtures, `Tests/Architecture/test_console_wave6_inventory.py`, and `Tests/Architecture/test_screen_size_ratchet.py`.
- Update only affected rows in `Docs/security/production-diagnostic-inventory.json` after conserving call atoms. Parent owns task/plan/public ledger documentation.

**Interfaces:**
- Move `_first_chat_defaults_match`, `_current_first_chat_defaults`, `eligible_console_first_chat_session_id`, `_release_first_chat_claim`, `_log_first_chat_handoff_exception`, `_resync_console_after_first_chat_rollback`, `_resync_mounted_console_after_first_chat_rollback`, and `consume_pending_console_first_chat_intent` into ConsoleSessionController.
- Reuse existing `app_instance`, current/lazy store accessors, native UI sync, core state sync, settings summary, and control bar sync dependencies. Keep framework `is_mounted`/`run_worker` live-read like existing Session services.
- Add named keyword-only dependencies for reading and setting the `(provider, model)` control values, capturing an opaque focus token, and restoring that token. Wire them through screen callbacks. The controller must not inspect Widget fields or call focus/query methods. Screen callbacks contain only presentation/scalar access, including the mounted-widget check.
- Move `_first_chat_handoff_notified_revision` default into Session; preserve assignable Screen compatibility through the existing `_ControllerState` descriptor without changing the 31-name historical inventory. Add a separate phase-safe regression for this post-baseline field.
- Wizard reads `eligible_console_first_chat_session_id` from the available screen's Session owner, preserving current reverse-stack traversal and absence fallback. Mount/resume invoke the Session consumer directly. Existing private tests patch/call the real owner; imports target defining modules.
- Rollback still restores scalars first, synchronizes mounted projections, schedules group `console-first-chat-rollback` with `exit_on_error=False`, awaits native UI sync, rechecks mount, then restores still-mounted focus. Use `functools.partial` of the bound async method for its captured token so worker rejection does not allocate an orphan coroutine; never use a synchronous coroutine-returning lambda.

- [x] **Step 1: Characterize and pin the current tree.** Save before-images/hashes before editing each existing file. Record exact Screen line/method counts and diagnostic-call atoms. Run `Tests/UI/test_console_session_settings.py -k first_chat`, wizard first-chat/future-target/generation-advance selections, Session controller and wiring tests, and the narrow ratchet. Save command output and JUnit. Read all eight current method bodies and exact-name callers before movement. Diagnose failures against the task before-images rather than calling them pre-existing by assumption.

- [x] **Step 2: Prove the missing ownership boundary.** Add a completed-owner AST test following the character family:

```python
group = WAVE6_GROUPS["first_chat"]
screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)
assert not (group.moved & screen_methods.keys())
assert group.moved <= target_methods.keys()
```

Run it before moving production code and retain the expected eight-name failure. Add isolated cases for no-store eligibility, untouched versus edited/global versus workspace sessions, stale-generation release, exact-claim replacement, created and refreshed rollback, acknowledgement exception/retry, warning deduplication and metadata-only logs. Use existing handoff/store assertions as fixtures, preserve data snapshots, and prove observable ordering outside contained callbacks. Add opaque-focus ordering/unmount checks and worker rejection without an unawaited coroutine. Run the isolated tests RED on the missing boundary before implementation.

- [x] **Step 3: Move policy and rewire callers.** Move the eight policy bodies with only named-dependency substitutions and the documented presentation split. Initialize the controller's notification-revision state and add its compatibility descriptor. Rewire wizard, mount/resume and test monkeypatch targets. Preserve all behavioral assertions. Update immediately affected ownership prose. Do not add default no-op dependencies merely to keep stale constructor fixtures passing.

- [x] **Step 4: Verify the extracted behavior.** Run the isolated module, complete first-chat session-settings selection, wizard staging/claim integration selection, Session controller/wiring suites, and relevant existing handoff-store tests. Run Wave 6 inventory, Screen ratchet and relevant decomposition/CSS gates. Mutate one config-generation fence and one exact-claim fence to prove existing behavioral cases reject the bypass; prove mount/resume dispatch and completed-owner oracles detect removal. Retain mounted rollback/focus assertions after actual settlement, not fixed sleeps. Compare policy ASTs after documented substitutions and account for every changed statement.

- [x] **Step 5: Validate and review this child.** Run persistent diagnostic inventory and compare only affected rows, conserving call atoms and disclosing unrelated drift. Compare Ruff findings against before-images, format owned ranges only, compile touched Python, and run scoped diff checks. Lower ratchet ceilings to exact earned Screen counts. Self-review the before-image diff, obtain independent task and final scoped reviews, then update task notes/all three AC/status via Backlog CLI and public ledger. Parent remains open. Preserve uncommitted evidence rather than deleting the only copies.

## Completion evidence

All eight first-chat policy methods now belong to Session. The reviewed extraction
reduced ChatScreen from 17,483 lines / 588 methods to 17,169 / 584 and earned
matching lower ratchets. Its 135 distinct targeted cases passed; config-generation,
exact-claim and mount-dispatch mutations were killed. Eight policy AST comparisons
and conservation of all 159 diagnostic-call atoms verify the intended boundary.
No new Ruff findings were introduced; compile and scoped formatting/diff checks
passed. Task and final integration reviews approved with no Critical/Important
findings. The final annotation-only cleanup also passed scoped re-review.

During final review, concurrent TASK-32309 Character rail handlers added 40 Screen
lines. The current shared checkout is 17,209 lines / 584 methods; a fresh ratchet
run reports one failure / one pass against the unchanged 17,169 / 584 caps.
Before-image comparison isolates the external addition, which is preserved.
The historical 135-case result is not a claim that the current shared checkout
passes every architecture gate. Unrelated global diagnostic-inventory drift and
existing whole-file static debt also remain; only the task-owned rows/ranges were
updated. No full suite or provider validation was run.

ADR required: no. Existing ADR-033 and the approved Wave 6 / DESIGN.md section 7
apply. TASK-3070.9 is complete; auto-speak and parent rebased-wave closeout remain
open. Detailed evidence is retained in
`.superpowers/sdd/2026-09-10-task-3070-9-console-first-chat-controller/`.
