# Parked-Payload Re-Key (PR0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-key the three Console interrupt bridges' retained-payload maps from `session_id` to `round_id` with per-session FIFO, so concurrent same-session rounds stop clobbering each other.

**Architecture:** All three bridges (MCP approvals, skill-install confirm, skill-script confirm) retain the payload their mounted card is re-derived from in a `dict[session_id, payload]` — one slot per session, last-armed-wins. Arming a second round for the same session overwrites the first's payload, so the first round's card vanishes and it hangs undecidable until its own timeout. This plan re-keys all three maps to `dict[round_id, payload]` and adds shared park/head/unpark helpers that pick the session's oldest-armed round as the mounted head. All three maps are already guarded by the single `_approval_state_lock`, so one helper set serves all three.

**Tech Stack:** Python 3.11+, Textual 8.2.8, pytest, stdlib `threading` only. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-08-19-console-user-interaction-design.md` (§3 defect, §4 PR0)

## Global Constraints

- Run tests with the checked-out venv only: `.venv/bin/python -m pytest`. The venv is uv-managed and ships no `pip`.
- All three retained-payload maps are guarded by `self._approval_state_lock`. Never introduce a second lock for them, and never hold that lock across a `call_from_thread`.
- Any decision about what the mounted card should show MUST be computed on the UI thread, inside the callable passed to `call_from_thread` — never from a worker-thread snapshot. This is the invariant three prior fix rounds converged on.
- FIFO head = oldest-armed round for the session. Python dicts preserve insertion order; do not add a separate ordering structure.
- Insertion order is arm order only because every write goes through `_park_round_payload`. Never assign into these maps directly.
- No behavior change is visible to users in this PR. Card content, timeouts, badges, and decisions are all unchanged for the single-round case.
- **Locate code by SYMBOL, never by line number.** This file changes fast; every `:NNNN` anchor in an earlier draft of this plan was stale within a day. Each step below names the enclosing method and quotes the code block to replace. Use `grep -n` on the quoted text.
- This work is **task-15661**, already filed. It is cited in `remount_pending_approval_for_active_session`'s docstring and in the test that pins the defect. Do not open a new backlog task.
- Branch from `origin/dev`. `origin/main` is over 10,000 commits behind and is not the trunk.
- **Grep the WHOLE REPO for each map you re-key, not just the controller.** These maps are private but not local: a consumer outside `console_chat_controller.py` read `_parked_approval_payloads` by session key (`ChatScreen._current_park_round_ids`), and a second in-file caller of a deleted guard would have raised `AttributeError` on every cancellation. Before you finish a bridge, run `grep -rn "_parked_<bridge>_payloads" --include=*.py .` and account for every hit, production and test.
- **Never read `tldw_chatbook/Chat/console_chat_controller.py` whole.** It is 12,482 lines / 608KB, and a single full read is large enough to kill an agent — it already killed one attempt at Task 2. Read it only in bounded windows (~120 lines) with explicit offset/limit, and locate sites with `grep -n` on the quoted code rather than by browsing.

---

### Task 1: Failing tests for the same-session defect

**Files:**
- Create: `Tests/UI/test_console_parked_payload_rekey.py`

**Interfaces:**
- Consumes: `ConsoleChatController.request_mcp_approvals(pending, *, session_id)`, `.resolve_pending_approval(decisions, *, round_id)`, `MCPPendingCall`.
- Produces: the `FakeApp` / `_wait_until` / `_arm` harness that Tasks 3, 4, and 5 reuse.

- [ ] **Step 1: Write the failing tests**

```python
"""PR0: concurrent same-session interrupt rounds must not clobber each other.

Mirrors ``Tests/UI/test_skill_install_concurrent_confirms.py`` (TASK-910) but
targets the half that task left unfixed: the RETAINED PAYLOAD each bridge
re-derives its mounted card from is keyed by ``session_id``, not ``round_id``,
so arming a second round for the same session overwrites the first's payload.
The code names this itself in ``request_mcp_approvals``' ``finally`` block:
"per-round payload storage is a larger change out of scope here".
"""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class FakeApp:
    """``call_from_thread`` stand-in: invokes the callback immediately."""

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def controller():
    """A controller with a fake UI wired and one ACTIVE session.

    ``mounted`` records every payload the approval card was told to show,
    including the ``None`` clears, so a test can assert on what the user
    would actually be looking at after each transition.
    """
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_approval = ctrl.mounted.append
    ctrl.mcp_approval_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    return ctrl


def _call(name):
    return MCPPendingCall(
        llm_name=name,
        server_key="agent:builtin",
        tool_name=name,
        server_label="builtin",
        arguments={},
        reason="ask",
    )


def _arm(ctrl, name, session_id, results, key):
    def worker():
        results[key] = ctrl.request_mcp_approvals(
            [_call(name)], session_id=session_id
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def _round_ids(ctrl):
    return list(ctrl._pending_approval_rounds)


def _mounted_round(ctrl):
    """The round id of the card currently shown, or None if cleared.

    The approvals payload names its id `round_id`; both skill bridges name
    theirs `request_id`. Every payload already carries one of the two, so
    no production payload needs a duplicate field for this helper.
    """
    payload = ctrl.mounted[-1] if ctrl.mounted else None
    if not payload:
        return None
    return payload.get("round_id") or payload.get("request_id")


def test_arming_a_second_same_session_round_does_not_evict_the_first_card(
    controller,
):
    """The head round keeps the card; a later sibling waits its turn."""
    results = {}
    first = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]
    assert _wait_until(lambda: _mounted_round(controller) == round_1)

    second = _arm(controller, "beta", controller.session_a, results, "beta")
    assert _wait_until(lambda: len(_round_ids(controller)) == 2)
    time.sleep(0.1)  # let any errant mount land before asserting

    assert _mounted_round(controller) == round_1, (
        "arming a second same-session round must not evict the first's card"
    )

    for round_id in _round_ids(controller):
        controller.resolve_pending_approval({"alpha": "approve_once", "beta": "approve_once"}, round_id=round_id)
    first.join(timeout=5)
    second.join(timeout=5)


def test_the_queued_round_mounts_when_the_head_resolves(controller):
    """FIFO: resolving the head promotes the next same-session round."""
    results = {}
    first = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]

    second = _arm(controller, "beta", controller.session_a, results, "beta")
    assert _wait_until(lambda: len(_round_ids(controller)) == 2)
    round_2 = [r for r in _round_ids(controller) if r != round_1][0]

    # Pre-condition: the head still owns the card. Without this, the test
    # cannot tell FIFO promotion from the arm-time clobber it exists to
    # catch -- round_2 would already be mounted and the post-assert below
    # would pass for the wrong reason, on both sides of the fix.
    time.sleep(0.1)
    assert _mounted_round(controller) == round_1

    controller.resolve_pending_approval({"alpha": "approve_once"}, round_id=round_1)
    first.join(timeout=5)

    assert _wait_until(lambda: _mounted_round(controller) == round_2), (
        "the queued round must mount once the head resolves"
    )

    controller.resolve_pending_approval({"beta": "approve_once"}, round_id=round_2)
    second.join(timeout=5)


def test_last_round_teardown_clears_the_card(controller):
    """With no rounds left for the session, the card clears."""
    results = {}
    only = _arm(controller, "alpha", controller.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(controller)) == 1)
    round_1 = _round_ids(controller)[0]

    controller.resolve_pending_approval({"alpha": "approve_once"}, round_id=round_1)
    only.join(timeout=5)

    assert _wait_until(lambda: _mounted_round(controller) is None), (
        "the card must clear once the session has no armed rounds left"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py -v`

Expected: BOTH `test_arming_a_second_same_session_round_does_not_evict_the_first_card` and `test_the_queued_round_mounts_when_the_head_resolves` FAIL, each on the assertion that the head still owns the card — the second arm overwrites `_parked_approval_payloads[session_id]` and marshals its own payload, so `_mounted_round` is `round_2`. `test_last_round_teardown_clears_the_card` may already PASS; that is expected and it guards against regressing the single-round case.

A failure that is an import, fixture, or API error is NOT valid RED — fix the harness until each failure is a real assertion failure.

- [ ] **Step 3: Commit the failing tests**

```bash
git add Tests/UI/test_console_parked_payload_rekey.py
git commit -m "test(console): pin same-session interrupt round clobbering (PR0)"
```

---

### Task 2: Shared park/head/unpark helpers and the approvals bridge

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — the `_parked_approval_payloads` declaration; `request_mcp_approvals` (arming + teardown `finally`); `_clear_pending_approval_if_round_is_current` (deleted whole); `_revoke_tool_approval_rounds` (its pop); and FOUR re-derive sites: the three inline `self._parked_approval_payloads.get(...)` blocks in `new_session`, `switch_session`, and `close_session`, plus the public `remount_pending_approval_for_active_session`
- Modify: `Tests/UI/test_console_headless_approval.py` — `test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts`
- Test: `Tests/UI/test_console_parked_payload_rekey.py`

**Interfaces:**
- Produces, for Tasks 3 and 4:
  - `_park_round_payload(self, store: dict[str, dict], round_id: str, payload: dict) -> bool` — stores the payload and returns whether this round is now its session's head.
  - `_head_round_payload(self, store: dict[str, dict], session_id: str) -> dict | None` — the session's oldest-armed retained payload.
  - `_unpark_round_payload(self, store: dict[str, dict], round_id: str) -> None`
  - `_remount_head(self, store: dict[str, dict], setter, session_id: str) -> None` — worker-thread-safe UI re-derive.
  - Every payload dict MUST carry both `"round_id"` and `"session_id"` keys.

- [ ] **Step 1: Add the helpers**

Add these four methods to `ConsoleChatController`, immediately after the `_marshal_pending_approval` method:

```python
    # -- PR0: per-round retained payloads ------------------------------
    #
    # All three bridges' retained-payload maps are keyed by ROUND id and
    # guarded by `_approval_state_lock`. The mounted card is always the
    # session's FIFO HEAD -- its oldest-armed round. Dict insertion order
    # is arm order, which is why every write goes through
    # `_park_round_payload` and nothing assigns into these maps directly.
    #
    # This replaces the pre-PR0 single-slot-per-session maps, whose
    # last-armed-wins semantics let a second same-session round overwrite
    # the first's payload and strand it until timeout.

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds the lock."""
        for payload in store.values():
            if payload.get("session_id") == session_id:
                return payload
        return None

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head.

        A round that is NOT the head must not mount -- an older sibling is
        still holding the card.
        """
        session_id = payload.get("session_id")
        with self._approval_state_lock:
            store[round_id] = payload
            head = self._head_round_payload_locked(store, session_id)
        return head is payload

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show."""
        with self._approval_state_lock:
            return self._head_round_payload_locked(store, session_id)

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Drop one round's retained payload. Idempotent."""
        with self._approval_state_lock:
            store.pop(round_id, None)

    def _remount_head(
        self,
        store: dict[str, dict[str, Any]],
        setter: Callable[[dict[str, Any] | None], None] | None,
        session_id: str | None,
    ) -> None:
        """WORKER THREAD: enqueue a head re-derive onto the UI thread.

        Replaces the pre-PR0 two-part TOCTOU guard. That guard existed
        because CLEARING the card was order-dependent -- whether to clear
        depended on which sibling resolved first, and a worker-thread
        snapshot of that answer could be stale by the time the UI thread
        ran it. Re-deriving the head is order-INDEPENDENT: it is a pure
        function of current state. The race-proofing principle is
        unchanged -- the decision still runs inside the callable on the UI
        thread, never from a snapshot -- but the decision itself is now one
        lookup instead of an identity check plus a sibling check.
        """
        if self.app is None or setter is None or session_id is None:
            return

        def _apply() -> None:
            if session_id != (self.store.active_session_id or ""):
                return
            setter(self._head_round_payload(store, session_id))

        self.app.call_from_thread(_apply)
```

- [ ] **Step 2: Re-key the declaration**

Find the `self._parked_approval_payloads: dict[str, dict[str, Any]] = {}` assignment in `__init__` and replace it and its preceding `#:` comment block with:

```python
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `round_id`. `switch_session` and every teardown re-derive the
        #: mounted card from this map's FIFO head for the session, so a
        #: second same-session round no longer evicts an older sibling's
        #: card. Every payload carries its own `round_id` and `session_id`.
        self._parked_approval_payloads: dict[str, dict[str, Any]] = {}
```

- [ ] **Step 3: Re-key arming**

In `request_mcp_approvals`, find the retained-payload write — a `with self._approval_state_lock:` block whose body is `self._parked_approval_payloads[session_id] = payload` — and replace the whole block with:

```python
            is_head = self._park_round_payload(
                self._parked_approval_payloads, round_id, payload
            )
```

Then, in the `try:` immediately below, change the `else:` arm that calls `self._marshal_pending_approval(payload)` to an `elif is_head:` arm:

```python
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_approval(payload)
```

`is_head` is `False` for the legacy `session_id is None` callers, which never park and never queue; keep their existing unconditional mount by initialising `is_head = True` before the `if session_id is not None:` block.

- [ ] **Step 4: Re-key teardown**

In `request_mcp_approvals`' `finally`, replace everything from the `with self._approval_state_lock:` that pops `_pending_approval_rounds` through the `self._clear_pending_approval_if_round_is_current(round_id, session_id)` call — including the `still_armed_same_session` computation and its conditional `self._parked_approval_payloads.pop(session_id, None)` — with:

```python
            with self._approval_state_lock:
                self._pending_approval_rounds.pop(round_id, None)
            self._unpark_round_payload(self._parked_approval_payloads, round_id)
            if session_id is not None:
                self.discard_pending_round(session_id, round_id)
            try:
                self._remount_head(
                    self._parked_approval_payloads,
                    self.set_pending_approval,
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal approval remount during teardown"
                )
```

- [ ] **Step 5: Delete the superseded guard and fix the revocation pop**

Delete the whole `_clear_pending_approval_if_round_is_current` method. Its two-part identity-plus-sibling check is subsumed by `_remount_head`.

Then in `_revoke_tool_approval_rounds`, replace the loop that pops the payload map — the `for _round_id, session_id in revoked:` block containing `self._parked_approval_payloads.pop(session_id, None)` guarded by a `not any(... state.get("session_id") == session_id ...)` test — with a straight per-round unpark:

```python
            for round_id_to_drop, session_id in revoked:
                self._unpark_round_payload(
                    self._parked_approval_payloads, round_id_to_drop
                )
```

The "is this the last armed round for the session" test existed only because the slot was shared. With per-round storage each revoked round drops exactly its own payload, so a still-armed sibling keeps its own.
**Lock discipline — this bit the Task 2 implementer.** `self._approval_state_lock` is a plain non-reentrant `threading.Lock`, and `_unpark_round_payload` acquires it internally. The replacement loop must therefore sit OUTSIDE any enclosing `with self._approval_state_lock:` block. Placed inside one it self-deadlocks on every run cancellation. Check the indentation of the code you are replacing before you paste.

- [ ] **Step 6: Re-key the three re-derive call sites**

There are THREE inline sites, one each in `new_session`, `switch_session`, and `close_session`. Replace each

```python
            with self._approval_state_lock:
                parked_payload = self._parked_approval_payloads.get(session_id)
            self.set_pending_approval(parked_payload)
```

with (the local variable differs per site: `session.id` in `new_session`, `session_id` in `switch_session`, `new_active_id` in `close_session`):

```python
            self.set_pending_approval(
                self._head_round_payload(self._parked_approval_payloads, session_id)
            )
```

These three sites already run on the UI thread, so they call `_head_round_payload` directly rather than `_remount_head`.

There is a FOURTH, public re-derive site: `remount_pending_approval_for_active_session`. Replace its `with self._approval_state_lock:` block (the one computing `still_armed` and then `payload`) with a head lookup, and delete the now-false "Known limitation" paragraph from its docstring:

```python
        payload = self._head_round_payload(
            self._parked_approval_payloads, session_id
        )
        if payload is None:
            return False
        self.set_pending_approval(payload)
        return True
```

The `still_armed` test is redundant now: a round's payload is unparked in its own teardown, so a payload present in the map belongs to a live round.

- [ ] **Step 7: Run the new tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py -v`
Expected: all three PASS.

- [ ] **Step 8: Rewrite the test that pins the defect**

`Tests/UI/test_console_headless_approval.py::test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts` deliberately asserts the BROKEN behaviour. Its own docstring says it is "pinned here so the limitation is a measured fact with a failing test the day someone fixes it, rather than folklore in a comment." Today is that day — this test MUST fail after Steps 2-6, and that failure is success.

Do not delete it: its `_detached_rig()` / `_leave()` harness covers the headless detached-attach path nothing else exercises. Rewrite it in place:

- Rename it to `test_two_headless_rounds_each_mount_in_turn`.
- Replace the docstring with a statement of the fixed contract, citing task-15661 as fixed rather than pinned.
- Keep the setup verbatim through the `assert len(app.notifications) == 2` line — both rounds arming and both announcing is unchanged behaviour.
- Replace the assertions that only one card can mount with: `remount_pending_approval_for_active_session()` mounts round A (the FIFO head, armed first); after round A resolves, a second call mounts round B.

Also update `remount_pending_approval_for_active_session`'s docstring reference to this test name if it names it.

- [ ] **Step 9: Run the existing approval suites for regressions**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_task_cards_sync.py Tests/UI/test_console_parallel_runs.py Tests/UI/test_console_headless_approval.py -v`
Expected: all PASS, including the rewritten headless test. Any OTHER failure is a real regression — the pre-PR0 behaviour those pin is what PR0 must preserve for the single-round case.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_parked_payload_rekey.py
git commit -m "fix(console): re-key approval retained payloads by round with FIFO head (task-15661)"
```

---

### Task 3: Skill-install bridge

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — the `_parked_skill_install_payloads` declaration; `request_skill_install_confirm` (arming + teardown `finally`); `_clear_pending_skill_install_if_round_is_current` (deleted whole); `_remount_parked_skill_install`
- Test: `Tests/UI/test_console_parked_payload_rekey.py`

**Interfaces:**
- Consumes: `_park_round_payload`, `_head_round_payload`, `_unpark_round_payload`, `_remount_head` from Task 2.
- Note: the skill-install payload built inside `request_skill_install_confirm` already carries both `"request_id"` and `"session_id"`. The helpers key the STORE by round id (passed as an argument) and look up the head by the payload's `"session_id"`, so no production payload needs a new field.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_parked_payload_rekey.py`:

```python
def _arm_install(ctrl, url, session_id, results, key):
    def worker():
        results[key] = ctrl.request_skill_install_confirm(url, session_id=session_id)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def install_controller():
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_skill_install = ctrl.mounted.append
    ctrl.skill_install_confirm_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    return ctrl


def test_install_second_same_session_round_does_not_evict_the_first(
    install_controller,
):
    ctrl = install_controller
    results = {}
    first = _arm_install(ctrl, "https://x/one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 1)
    round_1 = ctrl.pending_skill_install_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_1)

    second = _arm_install(ctrl, "https://x/two", ctrl.session_a, results, "two")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 2)
    time.sleep(0.1)

    assert _mounted_round(ctrl) == round_1, (
        "a second same-session install confirm must not evict the first's card"
    )

    ctrl.resolve_pending_skill_install(True, request_id=round_1)
    first.join(timeout=5)
    round_2 = ctrl.pending_skill_install_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_2)

    ctrl.resolve_pending_skill_install(True, request_id=round_2)
    second.join(timeout=5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py::test_install_second_same_session_round_does_not_evict_the_first -v`
Expected: FAIL — the second arm overwrites `_parked_skill_install_payloads[session_id]` and marshals its own payload.

- [ ] **Step 3: Re-key declaration, arming, teardown, and re-derive**

Find the `self._parked_skill_install_payloads: dict[str, dict[str, Any]] = {}` assignment in `__init__` and replace it and its preceding `#:` comment block with:

```python
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `request_id`. The mounted card is the session's FIFO head, so a
        #: second same-session confirm no longer evicts an older sibling.
        self._parked_skill_install_payloads: dict[str, dict[str, Any]] = {}
```

In `request_skill_install_confirm`, replace the `with self._approval_state_lock:` block whose body is `self._parked_skill_install_payloads[session_id] = payload` with:

```python
            is_head = self._park_round_payload(
                self._parked_skill_install_payloads, request_id, payload
            )
```

and make the mount conditional, exactly as Task 2 Step 3 did:

```python
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_skill_install(payload)
```

In the `finally`, delete the `still_armed_same_session` computation and its conditional pop, and replace with:

```python
            with self._pending_skill_install_lock:
                self._pending_skill_install_rounds.pop(request_id, None)
            self._unpark_round_payload(
                self._parked_skill_install_payloads, request_id
            )
            if session_id is not None:
                self.discard_pending_round(session_id, request_id)
            try:
                self._remount_head(
                    self._parked_skill_install_payloads,
                    self.set_pending_skill_install,
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal skill-install remount during teardown"
                )
```

Note the round registry keeps its own `_pending_skill_install_lock`; only the payload map moves to the shared helpers under `_approval_state_lock`.

Delete the whole `_clear_pending_skill_install_if_round_is_current` method — this bridge has its own copy of the same order-dependent guard, subsumed by `_remount_head` exactly as the approvals one was.

Finally, rewrite `_remount_parked_skill_install`'s body to use the head:

```python
        if self.set_pending_skill_install is None:
            return
        self.set_pending_skill_install(
            self._head_round_payload(self._parked_skill_install_payloads, session_id)
        )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py Tests/UI/test_skill_install_concurrent_confirms.py Tests/UI/test_console_skill_install_confirm.py -v`

Expected: your new tests PASS and every pre-existing TASK-910 test PASSES, **except** these two, which already fail on `origin/dev` before any PR0 change and are NOT yours to fix:

- `test_skill_install_concurrent_confirms.py::test_bare_shutdown_flag_alone_denies_a_real_session_round_within_one_poll_interval`
- `test_skill_install_concurrent_confirms.py::test_shutdown_flag_alone_denies_both_unregistered_sessions_rounds_and_cleans_accounting`

The baseline across the suites this plan touches is **4 known pre-existing failures**, all verified at the base commit in a detached worktree before any PR0 code change. The other two are in `Tests/UI/test_console_parallel_runs.py`:

- `test_navigating_away_with_busy_fleet_confirms_and_records_teardown`
- `test_navigation_guard_survives_stay_then_renavigate_then_leave_by_coordinates`

Any failure OUTSIDE these four is a real regression. Do not "fix" the two known ones; if your change makes them pass, say so in your report rather than assuming it is unrelated luck.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_parked_payload_rekey.py
git commit -m "fix(console): re-key skill-install retained payloads by round"
```

---

### Task 4: Skill-script bridge

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — the `_parked_skill_script_payloads` declaration; `request_skill_script_confirm` (arming + teardown `finally`); `_clear_pending_skill_script_if_round_is_current` (deleted whole); `_revoke_skill_script_rounds` (its pop); `_remount_parked_skill_script`
- Test: `Tests/UI/test_console_parked_payload_rekey.py`

**Interfaces:**
- Consumes: `_park_round_payload`, `_head_round_payload`, `_unpark_round_payload`, `_remount_head` from Task 2.
- Exact API for this bridge, verified against the source — it differs from skill-install in two ways that will break a copy-paste:
  - `request_skill_script_confirm(self, payload: dict[str, Any], *, session_id: str | None = None) -> dict[str, bool]` — the first argument is a **dict**, not a string.
  - `resolve_pending_skill_script(self, allow: bool, remember: bool, request_id: str | None = None) -> None` — **three** parameters, and `request_id` is positional-or-keyword, not keyword-only.
  - `pending_skill_script_ids() -> list[str]` returns armed ids in insertion order.
- The card payload is `card_payload` (a copy of the caller's dict plus `timeout_seconds`, `request_id`, `session_id`). It already carries the ids the helpers need.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_parked_payload_rekey.py`:

```python
def _arm_script(ctrl, script, session_id, results, key):
    def worker():
        results[key] = ctrl.request_skill_script_confirm(
            {"skill": "demo", "script": script}, session_id=session_id
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def script_controller():
    store = ConsoleChatStore()
    ctrl = ConsoleChatController(store=store, provider_gateway=object())
    ctrl.app = FakeApp()
    ctrl.mounted = []
    ctrl.set_pending_skill_script = ctrl.mounted.append
    ctrl.skill_script_confirm_timeout_seconds = lambda: 30.0
    ctrl.session_a = store.create_session(title="A").id
    store.switch_session(ctrl.session_a)
    return ctrl


def test_script_second_same_session_round_does_not_evict_the_first(
    script_controller,
):
    ctrl = script_controller
    results = {}
    first = _arm_script(ctrl, "echo one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_script_ids()) == 1)
    round_1 = ctrl.pending_skill_script_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_1)

    second = _arm_script(ctrl, "echo two", ctrl.session_a, results, "two")
    assert _wait_until(lambda: len(ctrl.pending_skill_script_ids()) == 2)
    time.sleep(0.1)

    assert _mounted_round(ctrl) == round_1, (
        "a second same-session script confirm must not evict the first's card"
    )

    # resolve_pending_skill_script takes (allow, remember, request_id).
    ctrl.resolve_pending_skill_script(True, False, round_1)
    first.join(timeout=5)

    round_2 = ctrl.pending_skill_script_ids()[0]
    assert _wait_until(lambda: _mounted_round(ctrl) == round_2), (
        "the queued script confirm must mount once the head resolves"
    )

    ctrl.resolve_pending_skill_script(True, False, round_2)
    second.join(timeout=5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py::test_script_second_same_session_round_does_not_evict_the_first -v`
Expected: FAIL — the second arm overwrites `_parked_skill_script_payloads[session_id]` and marshals its own `card_payload`, so `_mounted_round` is `round_2`.

- [ ] **Step 3: Re-key the declaration**

Find the `self._parked_skill_script_payloads: dict[str, dict[str, Any]] = {}` assignment in `__init__` and replace it and its preceding `#:` comment block with:

```python
        #: PR0: retained payload per ROUND (was per session), keyed by
        #: `request_id`. The mounted card is the session's FIFO head, so a
        #: second same-session confirm no longer evicts an older sibling.
        self._parked_skill_script_payloads: dict[str, dict[str, Any]] = {}
```

- [ ] **Step 4: Re-key arming**

In `request_skill_script_confirm`, replace the `with self._approval_state_lock:` block whose body is `self._parked_skill_script_payloads[session_id] = card_payload` with:

```python
            is_head = self._park_round_payload(
                self._parked_skill_script_payloads, request_id, card_payload
            )
```

Initialise `is_head = True` immediately before the enclosing `if session_id is not None:` block, so the legacy no-session callers keep their unconditional mount. Then make the mount conditional:

```python
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_skill_script(card_payload)
```

- [ ] **Step 5: Re-key teardown**

In the same method's `finally`, delete the `still_armed_same_session` computation and its conditional pop of `_parked_skill_script_payloads`, leaving:

```python
            with self._pending_skill_script_lock:
                self._pending_skill_script_rounds.pop(request_id, None)
            self._unpark_round_payload(
                self._parked_skill_script_payloads, request_id
            )
            if session_id is not None:
                self.discard_pending_round(session_id, request_id)
            try:
                self._remount_head(
                    self._parked_skill_script_payloads,
                    self.set_pending_skill_script,
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal skill-script remount during teardown"
                )
```

The round registry keeps its own `_pending_skill_script_lock`; only the payload map moves to the shared helpers under `_approval_state_lock`.

- [ ] **Step 6: Delete the guard, fix the revocation pop, and re-key the re-derive helper**

Delete the whole `_clear_pending_skill_script_if_round_is_current` method — the third copy of the same order-dependent guard.

In `_revoke_skill_script_rounds`, replace the conditional `self._parked_skill_script_payloads.pop(session_id, None)` and its "last armed for this session" guard with a per-round unpark, exactly as Task 2 Step 5 did for `_revoke_tool_approval_rounds`.

**The same lock trap applies here.** `_unpark_round_payload` acquires the non-reentrant `_approval_state_lock` internally, so this loop must sit OUTSIDE any enclosing `with self._approval_state_lock:` block or it self-deadlocks on every run cancellation. Task 2's implementer hit exactly this and had to move the loop out of the lock:

```python
            for round_id_to_drop, session_id in revoked:
                self._unpark_round_payload(
                    self._parked_skill_script_payloads, round_id_to_drop
                )
```

Then replace the body of `_remount_parked_skill_script` with:

```python
        if self.set_pending_skill_script is None:
            return
        self.set_pending_skill_script(
            self._head_round_payload(self._parked_skill_script_payloads, session_id)
        )
```

It already runs on the UI thread, so it calls `_head_round_payload` directly rather than `_remount_head`.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py Tests/Chat/test_skill_script_concurrent_confirms.py -v`
Expected: all PASS, including every pre-existing task-581 test.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_parked_payload_rekey.py
git commit -m "fix(console): re-key skill-script retained payloads by round"
```

---

### Task 5: Cross-bridge coverage and cleanup

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (orphaned `#:` comment before the `run_state` property)
- Test: `Tests/UI/test_console_parked_payload_rekey.py`

**Interfaces:**
- Consumes: every fixture and helper from Tasks 1, 3, and 4.

- [ ] **Step 1: Write the cross-bridge test**

Append to `Tests/UI/test_console_parked_payload_rekey.py`:

```python
def test_bridges_do_not_share_a_head(controller):
    """Each bridge keeps its own FIFO head for the same session.

    An approval round and a skill-install round armed for one session are
    independent surfaces -- neither may evict or promote the other.
    """
    ctrl = controller
    approvals_mounted = ctrl.mounted
    install_mounted = []
    ctrl.set_pending_skill_install = install_mounted.append
    ctrl.skill_install_confirm_timeout_seconds = lambda: 30.0

    results = {}
    approval = _arm(ctrl, "alpha", ctrl.session_a, results, "alpha")
    assert _wait_until(lambda: len(_round_ids(ctrl)) == 1)
    approval_round = _round_ids(ctrl)[0]

    install = _arm_install(ctrl, "https://x/one", ctrl.session_a, results, "one")
    assert _wait_until(lambda: len(ctrl.pending_skill_install_ids()) == 1)
    install_round = ctrl.pending_skill_install_ids()[0]

    # The approvals payload names its id `round_id`; the install payload
    # names its own `request_id`. Neither bridge renames the other's.
    assert approvals_mounted[-1]["round_id"] == approval_round
    assert install_mounted[-1]["request_id"] == install_round

    ctrl.resolve_pending_approval({"alpha": "approve_once"}, round_id=approval_round)
    approval.join(timeout=5)

    assert _wait_until(lambda: approvals_mounted[-1] is None)
    assert install_mounted[-1] is not None, (
        "resolving an approval round must not clear the install card"
    )

    ctrl.resolve_pending_skill_install(True, request_id=install_round)
    install.join(timeout=5)
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_parked_payload_rekey.py -v`
Expected: all PASS. If this one fails, a bridge is reading another bridge's map — check the store argument at each `_park_round_payload` / `_remount_head` call site.

- [ ] **Step 3: Delete the orphaned comment**

In `__init__`, immediately before the `run_state` property, there is a dangling `#:` comment block describing "The currently-armed round's unique id" — it documents a field that no longer exists (the next statement is the `run_state` property, not an assignment). Locate it with `grep -n "currently-armed round's unique id"` and delete the comment block.

- [ ] **Step 4: Run the full Console suite**

Run: `.venv/bin/python -m pytest Tests/UI/ Tests/Chat/ -q`
Expected: no new failures versus the pre-PR0 baseline. Record the baseline first on a clean checkout if you do not already have it — this repo carries known pre-existing failures, and a plain "N failed" number is not evidence by itself.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_parked_payload_rekey.py
git commit -m "test(console): cross-bridge head independence; drop orphaned comment"
```

---

## Done criteria

- All three bridges key retained payloads by round id. `grep -n "_parked_.*_payloads\[session_id\]"` returns nothing, and no `.pop(session_id` remains against any of the three maps.
- All THREE order-dependent guards are deleted: `_clear_pending_approval_if_round_is_current`, `_clear_pending_skill_install_if_round_is_current`, `_clear_pending_skill_script_if_round_is_current`.
- All FOUR approvals re-derive sites use the head, including the public `remount_pending_approval_for_active_session`, whose "Known limitation" docstring paragraph is gone.
- Both revocation paths (`_revoke_tool_approval_rounds`, `_revoke_skill_script_rounds`) unpark per round.
- `test_two_headless_rounds_share_one_payload_slot_and_only_one_mounts` is rewritten to assert per-round mounting.
- `Tests/UI/test_console_parked_payload_rekey.py` passes, and every pre-existing suite named in Tasks 2-4 still passes.
- No user-visible change for the single-round case.
