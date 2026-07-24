# Console Conversation Branching — Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a user edit a past **user** message and re-send it, forking a new branch from that point — the "Edit & resend" half of the branching foundation (Phase A shipped assistant regenerate-branching).

**Architecture:** Reuse Phase A's tree entirely. Editing-and-resending a user message U with new text T calls the existing role-parameterized `store.create_sibling(U.id, role=USER, content=T)` (a sibling of U under U's parent; U's old subtree is preserved off-path), then streams an assistant reply under the new user node via the existing streaming path. The `<`/`>` swipe + `n/m` counter are already role-generic (Phase A Task 7 gates on `sibling_count > 1`), so user-message branches navigate with no new UI. The only genuinely new surface is the edit modal's **"Edit & resend"** affordance and a controller `edit_and_resend_message` method modeled on `regenerate_message`.

**Tech Stack:** Python ≥3.11, Textual, SQLite (ChaChaNotes), pytest.

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-22-console-conversation-branching-foundation-design.md` (Phase B section + decision **D1**: in-place "Edit" stays; branching is an *explicit* "Edit & resend"). Do not change assistant regenerate (Phase A) or start Phase C (agent markers).
- **In-place edit must remain unchanged** for the plain "Save" path (`update_message_content`). "Edit & resend" is an additional, explicit choice — never the default.
- **Reuse, don't reinvent:** `store.create_sibling(anchor_message_id, *, role, content="", persist=False)` already exists and is role-parameterized. `regenerate_message` (assistant) is the structural template. Do NOT add a parallel branching path.
- **Persist correctly:** pass `persist=self.store.persistence is not None` to `create_sibling` and to any appended assistant node (the Phase A Task-6 lesson — an unpersisted branch vanishes on resume).
- **Failure semantics** match regenerate: a failed/empty resend leaves a retryable `failed` assistant node on the new branch; the old branch is reachable by swiping back.
- **Tests:** real in-memory SQLite / real store+controller (existing harnesses), not mocks. Run via `./.venv/bin/python -m pytest`. Baseline: `Tests/Chat/test_anthropic_native_tools.py` and `test_chat_functions.py` carry pre-existing unrelated failures — ignore.
- **Style/commits:** match surrounding code; commit after each task; end commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- **Modify** `tldw_chatbook/Widgets/Console/console_edit_message_modal.py` — widen the dismiss contract; add an "Edit & resend" button shown only for user messages.
- **Modify** `tldw_chatbook/Chat/console_chat_controller.py` — add `edit_and_resend_message(message_id, new_content)`.
- **Modify** `tldw_chatbook/UI/Screens/chat_screen.py` — `_open_console_message_edit_modal` passes `can_resend`; `_apply_edit` branches in-place-save vs edit-&-resend (worker, like regenerate).
- **Add tests** under `Tests/Chat/` and `Tests/integration/`.

Two facts (verified on this base):
- Edit modal is `ModalScreen[str | None]`; Save dismisses the text, Cancel dismisses `None`; it has a `_EditMessageTextArea` stale-key guard (TASK-360) — preserve it.
- `_apply_edit` (chat_screen.py ~:12972) currently does `store.update_message_content(message_id, result)` in place. `regenerate_message` (controller ~:1243) does `create_sibling(role=ASSISTANT)` → `_provider_messages_for_session(before_message_id=...)` → `_stream_assistant_response(variant_mode=False)`.

---

### Task 1: Widen the edit modal + add "Edit & resend" (user messages only)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py`
- Test: `Tests/Chat/test_console_edit_message_modal.py`

**Interfaces:**
- Produces: a result type `ConsoleEditResult` (frozen dataclass: `text: str`, `resend: bool`) and `ConsoleEditMessageModal(ModalScreen[ConsoleEditResult | None])` taking a new kwarg `can_resend: bool = False`. Cancel/Escape → `None`. "Save" → `ConsoleEditResult(text, resend=False)`. "Edit & resend" (only rendered when `can_resend`) → `ConsoleEditResult(text, resend=True)`. Blank text still blocked inline on both actions.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chat/test_console_edit_message_modal.py
from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal, ConsoleEditResult,
)

def test_edit_result_dataclass_shape():
    r = ConsoleEditResult(text="hi", resend=True)
    assert (r.text, r.resend) == ("hi", True)

def test_modal_accepts_can_resend_kwarg():
    # construction only (no mount) — the resend button is gated on can_resend
    m = ConsoleEditMessageModal(content="orig", can_resend=True)
    assert m._can_resend is True
    m2 = ConsoleEditMessageModal(content="orig")
    assert m2._can_resend is False
```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/python -m pytest Tests/Chat/test_console_edit_message_modal.py -v` → FAIL (`ConsoleEditResult` undefined / no `can_resend`).

- [ ] **Step 3: Implement.** Add the frozen dataclass; change the generic type to `ConsoleEditResult | None`; add `can_resend` to `__init__` (store `self._can_resend`); in `compose`, when `self._can_resend`, yield a third button `Button("Edit & resend", id="console-edit-message-resend", variant="primary")` (make plain "Save" `variant="default"` when resend is present, per surrounding style); update the context Static copy to mention the resend option only when `can_resend`. Add `@on(Button.Pressed, "#console-edit-message-resend")` mirroring `_save`'s blank-check but dismissing `ConsoleEditResult(text, resend=True)`; change `_save` to dismiss `ConsoleEditResult(text, resend=False)`; `action_dismiss`/`_cancel` dismiss `None`. Keep the `_EditMessageTextArea` stale-key guard untouched, and give the new button width in DEFAULT_CSS.

- [ ] **Step 4: Run to verify it passes** (+ add a mounted `run_test` pilot assertion if the existing modal tests use one — mirror them).

- [ ] **Step 5: Commit** — `feat(console): edit modal gains an "Edit & resend" option`.

---

### Task 2: Controller `edit_and_resend_message`

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_edit_resend.py`

**Interfaces:**
- Consumes: `store.create_sibling`, `_provider_messages_for_session`, `_stream_assistant_response`, the existing skill/dictionary/world-info/prefill transforms used by `regenerate_message`.
- Produces: `async edit_and_resend_message(message_id: str, new_content: str) -> ConsoleSubmitResult`.

Behavior (mirror `regenerate_message`, but for a USER anchor and creating BOTH a user sibling and its reply):
1. Active-run rejection gate; resolve session; block if `message_id` is not a USER message ("Only your messages can be edited and re-sent.").
2. `_validated_draft(new_content)`-style non-blank guard; block on empty.
3. `_set_run_state(VALIDATING…)`; `resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())`; block if not ready.
4. `new_user = self.store.create_sibling(message_id, role=ConsoleMessageRole.USER, content=new_content, persist=self.store.persistence is not None)` — new_user becomes the active leaf; U's old subtree is off-path.
5. `assistant = self.store.append_message(session_id, role=ConsoleMessageRole.ASSISTANT, content="", persist=self.store.persistence is not None)` — parented at new_user (the active leaf), becomes the new active leaf.
6. `provider_messages = self._provider_messages_for_session(session_id, before_message_id=assistant.id)` — the active path up to but excluding the empty assistant = history + the edited user prompt.
7. Apply the same `_apply_skill_substitution` / `_apply_chat_dictionaries` / `_apply_world_info` / `_pinned_prefill_for_session` steps `regenerate_message` applies (block on skill refusal).
8. `return await self._stream_assistant_response(resolution=resolution, provider_messages=provider_messages, assistant_message_id=assistant.id, variant_mode=False, prefill=prefill)`.

> Confirm ordering against `regenerate_message`: create the tree nodes only AFTER the block checks (Phase A Task-6 lesson — a blocked resend must leave no orphan sibling). If a blocked path is reached after `create_sibling`, that is acceptable only if it matches regenerate's own post-fork behavior; prefer gating before the fork.

- [ ] **Step 1: Write the failing test** (controller-level, reuse the fake provider/gateway harness from `Tests/Chat/test_console_regenerate_branching.py` / `test_console_chat_controller.py`):

```
# contract:
# after edit_and_resend_message(u1.id, "edited"):
#   - u1's parent now has TWO user children (u1 and the new sibling)
#   - the new user sibling's content == "edited", and it is on the active path
#   - an assistant reply node exists under the new user sibling with the streamed text
#   - u1 (and any old tail under it) still exist off the active path
#   - a non-USER message id is rejected (blocked)
```

- [ ] **Step 2: Run RED.**
- [ ] **Step 3: Implement** per the behavior above.
- [ ] **Step 4: Run GREEN** + `./.venv/bin/python -m pytest Tests/Chat/ -k "controller or regenerate or edit_resend or sibling" -q`.
- [ ] **Step 5: Commit** — `feat(console): edit_and_resend forks a user-message branch`.

---

### Task 3: Screen wiring — branch in-place-save vs edit-&-resend

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_open_console_message_edit_modal` ~:12966, `_apply_edit` ~:12972)
- Test: extend an existing chat_screen console test or add `Tests/UI/test_console_edit_resend_wiring.py`

**Interfaces:**
- Consumes: `ConsoleEditResult`, `controller.edit_and_resend_message`, the existing worker pattern used for regenerate (`run_worker(..., exclusive=True, group="console-run")`).

- [ ] **Step 1: Write the failing test** — assert that dismissing the modal with `resend=True` routes to `edit_and_resend_message` (spy/fake controller) and with `resend=False` routes to `store.update_message_content` (existing in-place path), and that the modal is opened with `can_resend=True` only for a USER message.

- [ ] **Step 2: Run RED.**

- [ ] **Step 3: Implement.**
- `_open_console_message_edit_modal(message_id, content)`: fetch the message; pass `can_resend = (message.role is ConsoleMessageRole.USER)` into `ConsoleEditMessageModal(content=content, can_resend=...)`.
- `_apply_edit(result: ConsoleEditResult | None)`: `if result is None: return`. `if not result.resend:` → existing `store.update_message_content(message_id, result.text)` path (unchanged, incl. its ValueError/KeyError handling + notify). `else:` gate on `controller.run_state.is_send_allowed` (notify `CONSOLE_RUN_ALREADY_RUNNING_COPY` if not), then `self.run_worker(self._edit_resend_console_message(controller, message_id, result.text), exclusive=True, group="console-run")`. Add a small `_edit_resend_console_message` worker mirroring `_regenerate_console_message` (start the transcript sync timer, `await controller.edit_and_resend_message(...)`, notify on `not accepted`, `await self._sync_native_console_chat_ui()`).

- [ ] **Step 4: Run GREEN** + console UI/chat_screen console suites.
- [ ] **Step 5: Commit** — `feat(console): wire Edit & resend into the transcript edit flow`.

---

### Task 4: User-message sibling navigation (verify + close any assistant-only gap)

The `<`/`>` + counter already gate on `sibling_count > 1` (Phase A Task 7), so a user message with siblings should show them. This task verifies the user-row path end-to-end and fixes any place that assumed siblings only exist on assistant rows.

**Files:**
- Test: `Tests/Chat/test_console_user_sibling_nav.py`
- Modify (only if a gap is found): `tldw_chatbook/Chat/console_message_actions.py` / `tldw_chatbook/Widgets/Console/console_transcript.py` / the screen's `_select_console_message_variant`.

- [ ] **Step 1: Write the test** — build two user siblings via `store.create_sibling(role=USER)`; assert `siblings_at` reports count 2 for the user node; assert `ConsoleMessageActionService.available_actions(user_msg_with_sibling_count_2)` includes `variant-previous`/`variant-next`; assert the counter renders on a user row; assert `_select_console_message_variant` moves the active leaf across the user siblings.
- [ ] **Step 2: Run.** If it passes with no production change, record that (the role-generic gate already covers it). If it fails, fix the specific assistant-only assumption (Step 3), then re-run.
- [ ] **Step 3 (conditional): Implement the fix** for whatever assistant-only assumption surfaced.
- [ ] **Step 4: Commit** — `test(console): user-message sibling navigation` (or `fix(console): …` if a gap was closed).

---

### Task 5: End-to-end + regression

**Files:**
- Test: `Tests/integration/test_console_edit_resend_e2e.py`

- [ ] **Step 1: Write the E2E** over real DB + store + controller (fake provider): send U1→A1; **edit U1 & resend** with new text → assert two user siblings under U1's parent, the new one active with its own assistant reply, U1's old subtree preserved off-path; swipe the user row `<`→ old branch (U1 + A1), `>` → new branch; persist→drop→resume → the active branch is restored (active-leaf pointer honored); in-place "Save" on a user message still edits in place (no new sibling).
- [ ] **Step 2: Run + regression sweep** — `./.venv/bin/python -m pytest Tests/Chat/ Tests/UI/test_console_native_transcript.py Tests/integration/ -q`; call out that non-passing tests are the known pre-existing baseline files. Fix any integration bug surfaced (document prominently).
- [ ] **Step 3: Commit** — `test(console): end-to-end edit-&-resend user branching`.

---

## Self-Review

**Spec coverage (Phase B):** widen edit modal + "Edit & resend" (Task 1); controller resend that forks a user sibling + reply (Task 2); screen routing keeping in-place edit as default (Task 3); user-row swipe/counter (Task 4, mostly pre-existing); e2e incl. resume (Task 5). In-place edit unchanged for plain Save (Tasks 1/3). ✅

**Placeholder scan:** Task 2/3 give the exact method behavior + interfaces + test contracts rather than full pasted bodies, because they are structural mirrors of the quoted `regenerate_message`/`_apply_edit`/`submit_draft` — the tests are the precise contract. Tasks 1/4/5 carry concrete code/contracts. No TBDs.

**Type consistency:** `ConsoleEditResult(text: str, resend: bool)` used across Tasks 1/3; `edit_and_resend_message(message_id, new_content) -> ConsoleSubmitResult` across Tasks 2/3/5; `create_sibling(role=USER, persist=…)` per Global Constraints.

**Known carry-forward (from Phase A, not Phase B's job):** the deferred-marker (task-498) and regenerate-failure-eviction (task-499) items also apply to a failed edit-&-resend (same node model) — no new work here, but Task 2's failure semantics should match regenerate so those follow-ups cover both.
