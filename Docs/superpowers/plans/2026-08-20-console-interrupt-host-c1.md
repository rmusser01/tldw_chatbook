# Console Interrupt Host C1 (Spine Extraction) Implementation Plan

> **task-31384 (2026-09-05):** recovered from the closed PR #1903 and executed against the five-kind controller (approvals, skill-install, skill-script, worktree-merge, ask_user question). Deltas from this document: `KIND_SETTER_ATTRS` carries `worktree_merge`; `run_round` gains an `on_teardown` hook for the approvals bridge's post-C1 "finishing phase" retention; the revocation sweep moves into the host (`revoke_for_run`) parameterised by each kind's closed-decision stamp; the activation-site re-derives collapse into `remount_for_session` with a per-kind post-remount hook (approvals fire the permission summary). Line numbers below are from the August tree and are stale -- anchor on the quoted code. **Review round (opus bridge-by-bridge diff review):** the payload layer is four module-level functions over `(lock, store)` (`park_round_payload`, `head_round_payload`, `session_round_payloads`, `unpark_round_payload`) so the controller's store-based shims and the host share one implementation and bare test doubles keep working; the per-call post-remount hook became a per-kind `after_remount` registry every remount path applies (the teardown promotion had lost the permission summary); `run_round` gained `on_outcome` (runs after the wait, before teardown -- decision snapshot, audit rows and the question marker land there, as they did before the host) and `announce_detached` returns True when it announced; the four non-question bridges register their state before their timeout config read, as before; `resolve()` was dropped (no production caller).


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the interrupt round lifecycle (arm → park/mount → poll → resolve/timeout/revoke → teardown re-derive) into `Chat/console_interrupt_rounds.py` and migrate all three bridges onto it, byte-identical to users.

**Architecture:** One `InterruptRoundHost` holding a single lock, per-kind registries and payload maps (legacy attribute names aliased to the same dict objects), the five PR0 helpers moved verbatim, and one generic `run_round` loop parameterized by per-kind wiring. The controller's three bridge methods keep their exact names/signatures/returns and shrink to payload construction + host call + bridge-specific legs.

**Tech Stack:** Python 3.11+, stdlib `threading` only, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-08-20-console-interrupt-host-design.md` (read it first; this plan argues from it)

## Global Constraints

- Run tests with the main checkout's interpreter FROM the worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`. The worktree has no `.venv`. Verify once that `import tldw_chatbook` resolves inside the worktree before trusting any run.
- **Never read `tldw_chatbook/Chat/console_chat_controller.py` whole** (~12.5k lines / 600KB — a full read killed a PR0 subagent). Bounded windows (~120 lines) with explicit offset/limit; locate by `grep -n` on quoted code. Locate code by SYMBOL, never by stored line number.
- **Never use `git stash`** (shared stack, other sessions' entries). **Never start background test runs and wait on them** (two PR0 implementers stalled that way). Foreground, focused suites.
- **Parity oracle:** the existing interrupt suites must pass UNCHANGED. Any edit to an existing test file in C1 invalidates the byte-parity claim and must be reported as a deviation, never silently made.
- The single host lock is NON-REENTRANT and aliased behind three historical names. Never call a host method that takes the lock from inside a `with <any of the three lock names>:` block — that self-deadlocked a PR0 implementer.
- The six legacy dict names and three lock names MUST remain live controller attributes (aliases): 11 test files and production `ChatScreen._current_park_round_ids` read them. Aliases are assignments of the SAME objects, never copies.
- Host access to `.app`, `.store`, and the per-kind setters must be LATE-BOUND (read through the controller reference at call time) — `.app` and setters are assigned at screen attach, after construction, and UI tests swap controller doubles.
- Bridge public API frozen: `request_mcp_approvals(pending, *, session_id=None) -> dict[str, str]`; `request_skill_install_confirm(url, *, session_id=None) -> bool`; `request_skill_script_confirm(payload: dict, *, session_id=None) -> dict[str, bool]`; the three `resolve_pending_*` signatures; `revoke_approval_rounds_for_run(run_id) -> int`; `pending_skill_install_ids()` / `pending_skill_script_ids()`; `remount_pending_approval_for_active_session() -> bool`.
- Known-failing baseline is established by Task 1 at the current tip, in a detached worktree — do not reuse PR0's counts. Any failure outside Task 1's recorded set is a regression.

---

### Task 1: Baseline and shutdown-failure characterization

**Files:**
- Create: none in the repo. Findings go to your report file only.

**Interfaces:**
- Produces: the authoritative failing-test baseline for every later task, and a written explanation of WHY each shutdown-flag test fails at this tip.

- [ ] **Step 1: Create a detached baseline worktree at the branch's merge base**

```bash
BASE=$(git merge-base origin/dev HEAD)
git worktree add --detach /tmp/c1-baseline "$BASE"
```

- [ ] **Step 2: Run the interrupt battery there**

```bash
cd /tmp/c1-baseline && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_console_parked_payload_rekey.py Tests/UI/test_console_mcp_approval.py \
  Tests/UI/test_console_headless_approval.py Tests/UI/test_skill_install_concurrent_confirms.py \
  Tests/UI/test_console_skill_install_confirm.py Tests/UI/test_console_parallel_runs.py \
  Tests/Chat/test_skill_script_concurrent_confirms.py Tests/Chat/test_console_viewless_hooks.py \
  Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_console_fleet_wake_safety.py \
  Tests/Chat/test_console_run_markers.py Tests/UI/test_probe_headless_approval_behaviour.py \
  Tests/UI/test_chat_task_cards_sync.py -q
```

Record the exact failing test list. (At PR0 time it was 4: two `test_skill_install_concurrent_confirms` shutdown-flag tests, two `test_console_parallel_runs` navigation tests — re-verify, do not assume.)

- [ ] **Step 3: Characterize the shutdown failures**

For each failing shutdown-flag test, run it alone with `-x -l` and read its assertion plus the code path it exercises (`_is_session_cancelled`'s `_shutdown_requested` branch). Write down, in your report: the assertion that fails, the observed vs expected value, and your best statement of the mechanism. C1 rewrites the poll loop these tests exercise; without this record, a C1-introduced shutdown regression is indistinguishable from the pre-existing redness. Do NOT fix them.

- [ ] **Step 4: Clean up and report**

```bash
git worktree remove /tmp/c1-baseline --force
```

Report the baseline list + characterization. No commit (nothing changed).

---

### Task 2: Host module — storage, lock, aliases, helpers moved

**Files:**
- Create: `tldw_chatbook/Chat/console_interrupt_rounds.py`
- Create: `Tests/Chat/test_console_interrupt_rounds.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — `__init__` (replace the six dict + three lock declarations with host construction + aliases) and the five helper methods (bodies become delegations)

**Interfaces:**
- Consumes: nothing from earlier tasks except Task 1's baseline.
- Produces, for Tasks 3–5 (exact):
  - `InterruptRoundHost(seams)` where `seams` is the controller (read late-bound).
  - `host.lock: threading.Lock` — THE lock.
  - `host.registries: dict[str, dict[str, dict]]` and `host.payloads: dict[str, dict[str, dict]]`, keys `"approval" | "skill_install" | "skill_script" | "question"`.
  - `host.park_round_payload(kind, round_id, payload) -> bool`
  - `host.head_round_payload(kind, session_id) -> dict | None` (returns the remaining-time SNAPSHOT — behavior moved verbatim)
  - `host.session_round_payloads(kind, session_id) -> list[dict]`
  - `host.unpark_round_payload(kind, round_id) -> None`
  - `host.remount_head(kind, session_id: str | None) -> None` (None = resolve active at callback; setter looked up late-bound by kind)
  - `KIND_SETTER_ATTRS: dict[str, str]` mapping kind → controller setter attribute name.

- [ ] **Step 1: Write the failing host unit tests**

Create `Tests/Chat/test_console_interrupt_rounds.py`:

```python
"""InterruptRoundHost unit tests -- no ConsoleChatController anywhere.

Sub-project C1 (spec: 2026-08-20-console-interrupt-host-design.md): the
host must be testable against a minimal seams double, which is exactly
the surface it is allowed to touch on the controller.
"""

from __future__ import annotations

import time

import pytest

from tldw_chatbook.Chat.console_interrupt_rounds import (
    KIND_SETTER_ATTRS,
    InterruptRoundHost,
)


class FakeStore:
    def __init__(self) -> None:
        self.active_session_id = "sess-A"


class FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class FakeSeams:
    """The exact controller surface the host may touch, and nothing more."""

    def __init__(self) -> None:
        self.app = FakeApp()
        self.store = FakeStore()
        self.mounted: dict[str, list] = {k: [] for k in KIND_SETTER_ATTRS}
        for kind, attr in KIND_SETTER_ATTRS.items():
            setattr(self, attr, self.mounted[kind].append)


@pytest.fixture
def host():
    return InterruptRoundHost(FakeSeams())


def _payload(round_id, session_id="sess-A", **extra):
    return {"round_id": round_id, "session_id": session_id, **extra}


def test_park_returns_head_only_for_the_oldest_round(host):
    assert host.park_round_payload("approval", "r1", _payload("r1")) is True
    assert host.park_round_payload("approval", "r2", _payload("r2")) is False


def test_kinds_do_not_share_heads(host):
    host.park_round_payload("approval", "r1", _payload("r1"))
    assert host.park_round_payload("skill_install", "r2", _payload("r2")) is True


def test_unpark_promotes_the_next_round(host):
    host.park_round_payload("approval", "r1", _payload("r1"))
    host.park_round_payload("approval", "r2", _payload("r2"))
    host.unpark_round_payload("approval", "r1")
    head = host.head_round_payload("approval", "sess-A")
    assert head is not None and head["round_id"] == "r2"


def test_head_returns_remaining_time_snapshot_without_mutating_the_stored_payload(host):
    stored = _payload(
        "r1", timeout_seconds=30.0, deadline_monotonic=time.monotonic() + 5.0
    )
    host.park_round_payload("approval", "r1", stored)
    head = host.head_round_payload("approval", "sess-A")
    assert 0.0 < head["timeout_seconds"] <= 5.0
    assert stored["timeout_seconds"] == 30.0


def test_remount_head_none_session_resolves_the_active_session(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1", session_id="sess-A"))
    seams.store.active_session_id = "sess-A"
    host.remount_head("approval", None)
    assert seams.mounted["approval"][-1]["round_id"] == "r1"
    seams.store.active_session_id = "sess-B"
    host.remount_head("approval", None)
    assert seams.mounted["approval"][-1] is None


def test_remount_head_mismatched_session_is_a_no_op(host):
    seams = host._seams
    host.park_round_payload("approval", "r1", _payload("r1", session_id="sess-B"))
    host.remount_head("approval", "sess-B")  # active is sess-A
    assert seams.mounted["approval"] == []


def test_missing_setter_attr_is_a_safe_no_op(host):
    seams = host._seams
    delattr(seams, KIND_SETTER_ATTRS["question"])
    host.park_round_payload("question", "q1", _payload("q1"))
    host.remount_head("question", "sess-A")  # must not raise
```

- [ ] **Step 2: Run them to verify they fail on import**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_interrupt_rounds.py -q`
Expected: collection error — `console_interrupt_rounds` does not exist.

- [ ] **Step 3: Write the host module**

Create `tldw_chatbook/Chat/console_interrupt_rounds.py`. The five helper bodies are MOVES from `ConsoleChatController` (grep each `def _park_round_payload` etc. and carry the logic and its comments; the code below is the moved logic re-homed — semantics must stay identical):

```python
"""Generic Console interrupt-round host (sub-project C1, task program spec
2026-08-20-console-interrupt-host-design.md).

One lifecycle, ONE lock, per-kind storage for the Console's blocking
interrupt rounds (MCP approvals, skill-install confirms, skill-script
confirms; "question" reserved for sub-project A).

Locking: ``lock`` is a plain NON-REENTRANT ``threading.Lock``. The
controller aliases its three historical lock names to this one object, so
nesting any two of them -- or calling a host method that takes the lock
from inside a ``with`` on any of the names -- self-deadlocks immediately.
Nothing nests today (verified at C1 design time, tests included); keep it
that way.

Seams: the host holds the CONTROLLER and reads ``.app``, ``.store``, and
the per-kind setter attributes late-bound, at call time -- they are
assigned at screen attach, after construction, and UI tests swap in
controller doubles. The full surface the host may touch is exactly what
``Tests/Chat/test_console_interrupt_rounds.py``'s ``FakeSeams`` provides.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: Kind -> the controller attribute holding that kind's UI setter. The
#: setters are attach-time assignments and may be absent entirely
#: (headless, or a kind not yet wired -- "question" until sub-project A);
#: every read goes through ``getattr(..., None)`` and treats None as
#: "no UI, no-op".
KIND_SETTER_ATTRS: dict[str, str] = {
    "approval": "set_pending_approval",
    "skill_install": "set_pending_skill_install",
    "skill_script": "set_pending_skill_script",
    "question": "set_pending_question",
}


class InterruptRoundHost:
    """Own the registries, payload maps, and FIFO-head render contract."""

    POLL_SECONDS = 1.0

    def __init__(self, seams: Any) -> None:
        self._seams = seams
        self.lock = threading.Lock()
        self.registries: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        self.payloads: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }

    # -- setter / app access (always late-bound) -----------------------

    def _setter(self, kind: str):
        return getattr(self._seams, KIND_SETTER_ATTRS[kind], None)

    def _active_session_id(self) -> str:
        store = getattr(self._seams, "store", None)
        return (getattr(store, "active_session_id", None) or "") if store else ""

    # -- payload layer (moved verbatim from ConsoleChatController) -----

    @staticmethod
    def _head_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds ``lock``."""
        for payload in store.values():
            if payload.get("session_id") == session_id:
                return payload
        return None

    def park_round_payload(
        self, kind: str, round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        session_id = payload.get("session_id")
        store = self.payloads[kind]
        with self.lock:
            store[round_id] = payload
            head = self._head_locked(store, session_id)
        return head is payload

    def head_round_payload(
        self, kind: str, session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show.

        Carries the PR #1836 remaining-time snapshot behavior verbatim:
        a payload with a live ``deadline_monotonic`` is returned as a
        shallow copy whose ``timeout_seconds`` is the remaining window;
        the retained payload is never mutated. The ``head is payload``
        identity check in ``park_round_payload`` goes through
        ``_head_locked`` and is unaffected.
        """
        store = self.payloads[kind]
        with self.lock:
            payload = self._head_locked(store, session_id)
        if payload is None:
            return None
        deadline = payload.get("deadline_monotonic")
        if not deadline:
            return payload
        snapshot = dict(payload)
        snapshot["timeout_seconds"] = max(0.0, deadline - time.monotonic())
        return snapshot

    def session_round_payloads(
        self, kind: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``kind`` retains for ``session_id``, arm order."""
        store = self.payloads[kind]
        with self.lock:
            return [
                payload
                for payload in store.values()
                if payload.get("session_id") == session_id
            ]

    def unpark_round_payload(self, kind: str, round_id: str) -> None:
        """Drop one round's retained payload. Idempotent."""
        with self.lock:
            self.payloads[kind].pop(round_id, None)

    def remount_head(self, kind: str, session_id: str | None) -> None:
        """Enqueue a head re-derive onto the UI thread (worker-safe).

        The decision -- WHICH payload, and whether the session is still
        the one being viewed -- is computed INSIDE the callable, on the
        UI thread, never from a worker-thread snapshot: the invariant
        three pre-PR0 fix rounds converged on.

        ``session_id=None`` means "the session being VIEWED when the
        callback runs" (legacy no-session rounds mount unconditionally,
        so their card can sit over any session by teardown time).
        """
        app = getattr(self._seams, "app", None)
        if app is None or self._setter(kind) is None:
            return

        def _apply() -> None:
            setter = self._setter(kind)
            if setter is None:
                return
            if session_id is None:
                setter(self.head_round_payload(kind, self._active_session_id()))
                return
            if session_id != self._active_session_id():
                return
            setter(self.head_round_payload(kind, session_id))

        app.call_from_thread(_apply)
```

- [ ] **Step 4: Run the host unit tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_interrupt_rounds.py -q`
Expected: all PASS.

- [ ] **Step 5: Wire the controller — construction, aliases, delegating helpers**

In `ConsoleChatController.__init__`, locate the NINE declarations (grep each name): `_approval_state_lock`, `_pending_approval_rounds`, `_parked_approval_payloads`, `_pending_skill_install_lock`, `_pending_skill_install_rounds`, `_parked_skill_install_payloads`, `_pending_skill_script_lock`, `_pending_skill_script_rounds`, `_parked_skill_script_payloads`. Replace each ASSIGNMENT with an alias into one host constructed first (keep every existing `#:` comment block, updating only what the alias changes):

```python
        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        #: C1: the interrupt-round spine. One lifecycle, ONE lock,
        #: per-kind storage. The nine historical attribute names below
        #: are ALIASES onto the host's own objects -- 11 test files and
        #: `ChatScreen._current_park_round_ids` read them by name, and
        #: none is ever reassigned outside __init__ (verified), so the
        #: aliases cannot go stale.
        self._interrupt_host = InterruptRoundHost(self)
        self._approval_state_lock = self._interrupt_host.lock
        self._pending_skill_install_lock = self._interrupt_host.lock
        self._pending_skill_script_lock = self._interrupt_host.lock
        self._pending_approval_rounds = self._interrupt_host.registries["approval"]
        self._pending_skill_install_rounds = self._interrupt_host.registries["skill_install"]
        self._pending_skill_script_rounds = self._interrupt_host.registries["skill_script"]
        self._parked_approval_payloads = self._interrupt_host.payloads["approval"]
        self._parked_skill_install_payloads = self._interrupt_host.payloads["skill_install"]
        self._parked_skill_script_payloads = self._interrupt_host.payloads["skill_script"]
```

Keep the declarations' positions (the lock alias must exist before anything reads it; putting the whole block where `_approval_state_lock` is declared today is correct — the two skill-lock declarations later in `__init__` are then deleted).

Then convert the five helper methods to delegations, keeping their exact names and signatures (external callers exist in `chat_screen.py` and tests). The store-dict first argument maps to a kind via identity:

```python
    def _kind_for_store(self, store: dict) -> str:
        for kind, mapped in self._interrupt_host.payloads.items():
            if mapped is store:
                return kind
        raise ValueError("unknown payload store")

    def _park_round_payload(self, store, round_id, payload):
        return self._interrupt_host.park_round_payload(
            self._kind_for_store(store), round_id, payload
        )

    def _head_round_payload(self, store, session_id):
        return self._interrupt_host.head_round_payload(
            self._kind_for_store(store), session_id
        )

    def _session_round_payloads(self, store, session_id):
        return self._interrupt_host.session_round_payloads(
            self._kind_for_store(store), session_id
        )

    def _unpark_round_payload(self, store, round_id):
        self._interrupt_host.unpark_round_payload(
            self._kind_for_store(store), round_id
        )

    def _remount_head(self, store, setter, session_id):
        # `setter` is ignored: the host looks the setter up late-bound by
        # kind, which is what every production caller passed anyway.
        # Kept in the signature for call-site compatibility until Task 5
        # collapses the call sites.
        self._interrupt_host.remount_head(self._kind_for_store(store), session_id)

    @staticmethod
    def _head_round_payload_locked(store, session_id):
        # Caller-holds-lock variant, still used by in-file readers.
        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost
        return InterruptRoundHost._head_locked(store, session_id)
```

Delete the five original helper bodies. NOTE the one semantic hazard: production `_remount_head` callers pass three positional args `(store, setter, session_id)` — the delegation must keep that arity.

- [ ] **Step 6: Run the parity battery**

Run the Task 1 battery command (from THIS worktree, not the baseline one).
Expected: identical pass/fail set to Task 1's baseline. Any new failure is yours.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_interrupt_rounds.py Tests/Chat/test_console_interrupt_rounds.py tldw_chatbook/Chat/console_chat_controller.py
git commit -m "refactor(console): extract interrupt-round storage and payload layer to InterruptRoundHost (C1)"
```

---

### Task 3: Generic run_round; migrate the approvals bridge

**Files:**
- Modify: `tldw_chatbook/Chat/console_interrupt_rounds.py` (add `run_round`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`request_mcp_approvals` body)
- Test: `Tests/Chat/test_console_interrupt_rounds.py` (add loop tests)

**Interfaces:**
- Consumes: Task 2's host exactly as produced.
- Produces, for Task 4:

```python
    def run_round(
        self,
        kind: str,
        round_id: str,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        session_id: str | None,
        owning_session_id: str,
        deadline: float | None,
        is_parked: bool,
        announce_detached: Callable[[], None] | None = None,
        human_wait_run_id: str | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
        check_revoked: bool = True,
    ) -> str:  # "decided" | "timeout" | "cancelled" | "revoked"
```

  `state` MUST contain `"event"` (threading.Event) and `"session_id"`; it is the registry entry. `run_round` registers it, parks the payload, runs the mount/park/announce branch, waits on the poll loop, and performs the full teardown (registry pop, unpark, `discard_pending_round`, `remount_head` with the legacy-None rule). The wrapper maps the returned resolution to its public return value.

- [ ] **Step 1: Write the failing loop tests**

Append to `Tests/Chat/test_console_interrupt_rounds.py` (extend `FakeSeams` first):

```python
import threading


class FakeSeamsFull(FakeSeams):
    """Adds the probe/badge surface run_round touches."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False
        self.badges: list[tuple[str, str, str]] = []
        self.park_pending_approval = None

    def _is_session_cancelled(self, session_id, *, cancel_event=None, visit_event=None):
        return self.cancelled

    def add_pending_round(self, session_id, round_id):
        self.badges.append(("add", session_id, round_id))

    def discard_pending_round(self, session_id, round_id):
        self.badges.append(("discard", session_id, round_id))


def test_run_round_decided_when_the_event_is_set():
    host = InterruptRoundHost(FakeSeamsFull())
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()  # pre-resolved: loop exits immediately
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert outcome == "decided"
    assert host.registries["approval"] == {}
    assert host.payloads["approval"] == {}


def test_run_round_times_out_and_calls_on_timeout():
    host = InterruptRoundHost(FakeSeamsFull())
    fired = []
    state = {"event": threading.Event(), "session_id": "sess-A"}
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=time.monotonic() - 1.0, is_parked=False,
        on_timeout=lambda: fired.append("t"),
    )
    assert outcome == "timeout" and fired == ["t"]


def test_run_round_cancelled_calls_on_cancelled():
    seams = FakeSeamsFull()
    seams.cancelled = True
    host = InterruptRoundHost(seams)
    fired = []
    state = {"event": threading.Event(), "session_id": "sess-A"}
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
        on_cancelled=lambda: fired.append("c"),
    )
    assert outcome == "cancelled" and fired == ["c"]


def test_run_round_revoked_wins_over_decided():
    host = InterruptRoundHost(FakeSeamsFull())
    state = {"event": threading.Event(), "session_id": "sess-A", "revoked": True}
    state["event"].set()
    outcome = host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert outcome == "revoked"


def test_run_round_teardown_promotes_the_queued_sibling():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    host.park_round_payload("approval", "r0", _payload("r0"))  # will be head
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert seams.mounted["approval"][-1]["round_id"] == "r0"


def test_run_round_badge_add_and_discard_bracket_the_wait():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    state = {"event": threading.Event(), "session_id": "sess-A"}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1"), state,
        session_id="sess-A", owning_session_id="sess-A",
        deadline=None, is_parked=False,
    )
    assert seams.badges == [("add", "sess-A", "r1"), ("discard", "sess-A", "r1")]


def test_run_round_legacy_none_session_skips_badge_and_park():
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    state = {"event": threading.Event(), "session_id": ""}
    state["event"].set()
    host.run_round(
        "approval", "r1", _payload("r1", session_id=""), state,
        session_id=None, owning_session_id="",
        deadline=None, is_parked=False,
    )
    assert seams.badges == []
    assert host.payloads["approval"] == {}
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_interrupt_rounds.py -q`
Expected: the new tests fail with `AttributeError: ... run_round`.

- [ ] **Step 3: Implement run_round**

Add to `InterruptRoundHost`, with these module-top imports: `from collections.abc import Callable`, `from contextlib import nullcontext`, and `from tldw_chatbook.Agents.human_input_wait import use_human_input_wait`:

```python
    def run_round(
        self,
        kind: str,
        round_id: str,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        session_id: str | None,
        owning_session_id: str,
        deadline: float | None,
        is_parked: bool,
        announce_detached: Callable[[], None] | None = None,
        human_wait_run_id: str | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
        check_revoked: bool = True,
    ) -> str:
        """One blocking interrupt round, registration through teardown.

        WORKER THREAD. Reproduces the (converged) bridge lifecycle:
        register -> badge -> park -> announce/park-toast/mount -> poll ->
        teardown (pop, unpark, badge-discard, head re-derive). The
        per-bridge deltas ride the hooks: ``announce_detached`` is the
        MCP detached-view leg; ``on_cancelled``/``on_timeout`` let the
        approvals wrapper stamp its decisions box and audit-log;
        ``human_wait_run_id`` wraps the wait in ``use_human_input_wait``
        (the script bridge passes None -- it is dispatched in-loop and
        never hosted by a per-call wrapper); ``check_revoked`` is False
        for skill-install, which is never swept.
        """
        event: threading.Event = state["event"]
        with self.lock:
            self.registries[kind][round_id] = state
        is_head = True
        if session_id is not None:
            add = getattr(self._seams, "add_pending_round", None)
            if add is not None:
                add(session_id, round_id)
            is_head = self.park_round_payload(kind, round_id, payload)
        try:
            app = getattr(self._seams, "app", None)
            park_toast = getattr(self._seams, "park_pending_approval", None)
            if announce_detached is not None:
                announce_detached()
            elif is_parked:
                if app is not None and park_toast is not None:
                    app.call_from_thread(park_toast, session_id)
            elif is_head:
                setter = self._setter(kind)
                if app is not None and setter is not None:
                    app.call_from_thread(setter, payload)
            outcome = "decided"
            wait_cm = (
                use_human_input_wait(human_wait_run_id)
                if human_wait_run_id is not None
                else nullcontext()
            )
            with wait_cm:
                while not event.wait(self.POLL_SECONDS):
                    if self._seams._is_session_cancelled(
                        session_id,
                        cancel_event=state.get("cancel_event"),
                        visit_event=state.get("visit_event"),
                    ):
                        if on_cancelled is not None:
                            on_cancelled()
                        outcome = "cancelled"
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        if on_timeout is not None:
                            on_timeout()
                        outcome = "timeout"
                        break
            if check_revoked:
                with self.lock:
                    if bool(state.get("revoked")):
                        outcome = "revoked"
            return outcome
        finally:
            with self.lock:
                self.registries[kind].pop(round_id, None)
            self.unpark_round_payload(kind, round_id)
            if session_id is not None:
                discard = getattr(self._seams, "discard_pending_round", None)
                if discard is not None:
                    discard(session_id, round_id)
            try:
                self.remount_head(
                    kind, owning_session_id if session_id is not None else None
                )
            except Exception:  # noqa: BLE001 -- teardown must never raise
                pass
```

Semantic notes the implementer must honor (each is pinned by an existing controller test):
- `announce_detached` REPLACES the park/mount branch when provided and firing — the wrapper passes it only when `self._approval_view_is_detached()` is true, i.e. the wrapper decides, the host just runs the chosen branch. Pass `announce_detached=lambda: self._announce_detached_approval(owning_session_id)` **only when detached**, else `None`.
- `state` carries `cancel_event` / `visit_event` — the wrapper puts its `_bind_round_cancel_signal` / `_bind_visit_cancel_signal` results there.
- The teardown swallow mirrors today's `except Exception: logger...debug` — keep a debug log line if `loguru` is imported in the module; a bare pass is acceptable only if the module stays logger-free.

- [ ] **Step 4: Run the host tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_interrupt_rounds.py -q`
Expected: all PASS.

- [ ] **Step 5: Migrate `request_mcp_approvals`**

In the bridge method, everything from the registry-write (`self._pending_approval_rounds[round_id] = round_state` under lock) through the `finally` block's end is replaced by ONE `run_round` call plus the wrapper's own resolution mapping. The wrapper keeps: pending-collection, `round_id = str(uuid4())`, timeout/deadline resolution, `round_state` construction (ADD `"cancel_event": round_cancel_event, "visit_event": visit_cancel_event` to it), payload build, `is_parked` computation, and after the call: the revoked/decided mapping to the `decisions` dict, exactly today's post-wait code. The hooks:

```python
        def _on_cancelled() -> None:
            cancelled_names = [
                name for name in unique_names if name not in decisions
            ]
            for name in unique_names:
                decisions.setdefault(name, "deny")
            self._record_cancelled_approval_decisions(
                cancelled_names, call_by_name
            )

        def _on_timeout() -> None:
            for name in unique_names:
                decisions.setdefault(name, "timeout")

        outcome = self._interrupt_host.run_round(
            "approval", round_id, payload, round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            announce_detached=(
                (lambda: self._announce_detached_approval(owning_session_id))
                if self._approval_view_is_detached()
                else None
            ),
            human_wait_run_id=owning_run_id,
            on_cancelled=_on_cancelled,
            on_timeout=_on_timeout,
        )
        if outcome == "revoked":
            return {name: "deny" for name in unique_names}
        return dict(decisions)
```

Match the revoked return against today's exact code (grep `was_revoked` in the bridge) — reproduce ITS mapping, not this sketch, if they differ.

- [ ] **Step 6: Run the parity battery**

Same command; expected: identical to Task 1's baseline.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_interrupt_rounds.py Tests/Chat/test_console_interrupt_rounds.py tldw_chatbook/Chat/console_chat_controller.py
git commit -m "refactor(console): generic run_round; approvals bridge rides the host (C1)"
```

---

### Task 4: Migrate the skill-install and skill-script bridges

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`request_skill_install_confirm`, `request_skill_script_confirm` bodies)
- Test: parity battery only (no new tests — the concurrency suites for both bridges already exist)

**Interfaces:**
- Consumes: `run_round` exactly as Task 3 produced it.

- [ ] **Step 1: Migrate `request_skill_install_confirm`**

Wrapper keeps: request_id mint, cancel binds, `owning_run_id`, registry-entry construction (ADD `"cancel_event"`/`"visit_event"` keys), timeout/deadline, payload build, `is_parked`. Replace registration-through-finally with:

```python
        outcome = self._interrupt_host.run_round(
            "skill_install", request_id, payload, install_round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=owning_run_id,
            check_revoked=False,  # install is never swept (primary-agent only)
        )
        return bool(decision.get("allow", False))
```

(`outcome` is deliberately unused beyond documentation — today's install code returns the decision box regardless of why the wait ended. Name the variable `_outcome` if the linter complains.)

- [ ] **Step 2: Migrate `request_skill_script_confirm`**

Same shape, with this bridge's two deltas. **CORRECTION (Task 3 review):** this plan originally claimed the script bridge passes no `human_wait_run_id`; that was true when drafted but dev moved — the current method wraps its wait in `use_human_input_wait(str(script_round_state.get("run_id") or ""))` (grep it). Pass `human_wait_run_id=str(script_round_state.get("run_id") or "")`. The remaining true delta is the revoked mapping:

```python
        outcome = self._interrupt_host.run_round(
            "skill_script", request_id, card_payload, script_round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=str(script_round_state.get("run_id") or ""),
        )
        if outcome == "revoked":
            return {"allow": False, "remember": False}
        return {
            "allow": bool(decision.get("allow", False)),
            "remember": bool(decision.get("remember", False)),
        }
```

Match the tail against today's exact post-wait code (grep `was_revoked` in this method) and reproduce it.

- [ ] **Step 3: Run the parity battery**

Expected: identical to Task 1's baseline. Pay particular attention to the shutdown-flag tests: they must fail in exactly the way Task 1 characterized — a changed failure mode is a finding even though the tests were already red.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py
git commit -m "refactor(console): skill bridges ride the host's run_round (C1)"
```

---

### Task 5: Consolidate resolve and remount; collapse the re-derive sites

**Deliberate C1 deferral — the revoke sweep stays in the controller.** The spec's §3.1 lists `revoke_for_run` as a host method; C1 does not move it. `revoke_approval_rounds_for_run` and its two private sweeps operate entirely through the aliased registries, the host's single lock, and `unpark_round_payload`/`remount_head` — so they work unchanged, and moving them buys no behavior until a FOURTH kind needs sweeping. Sub-project A's question rounds will (a cancelled run must fail its question round closed), so the sweep moves then. The spec carries a matching note; do not "complete" this migration on your own initiative.

**Files:**
- Modify: `tldw_chatbook/Chat/console_interrupt_rounds.py` (add `resolve`, `remount_for_session`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (three `resolve_pending_*` bodies; `new_session`/`switch_session`/`close_session` re-derive blocks; `remount_pending_approval_for_active_session`)
- Test: `Tests/Chat/test_console_interrupt_rounds.py` (resolve tests)

**Interfaces:**
- Consumes: everything prior.
- Produces: `host.resolve(kind, round_id, mutate: Callable[[dict], None]) -> bool` and `host.remount_for_session(session_id: str) -> None`.

- [ ] **Step 1: Write the failing resolve tests**

```python
def test_resolve_fails_closed_on_none_and_unknown_ids(host):
    assert host.resolve("approval", None, lambda s: None) is False
    assert host.resolve("approval", "ghost", lambda s: None) is False


def test_resolve_mutates_the_snapshotted_state_and_sets_the_event():
    host = InterruptRoundHost(FakeSeamsFull())
    event = threading.Event()
    state = {"event": event, "session_id": "sess-A", "decision": {}}
    with host.lock:
        host.registries["approval"]["r1"] = state
    assert host.resolve(
        "approval", "r1", lambda s: s["decision"].update({"allow": True})
    ) is True
    assert event.is_set() and state["decision"] == {"allow": True}
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
    def resolve(
        self, kind: str, round_id: str | None, mutate: Callable[[dict[str, Any]], None]
    ) -> bool:
        """Apply a user decision to one armed round. UI THREAD.

        Fail-closed (the TASK-913 contract, now in one place): ``None``
        and unknown ids resolve nothing and return False. The state is
        SNAPSHOTTED under the lock and mutated outside it -- the worker's
        ``finally`` pops the entry concurrently, and acting on the
        snapshot is what keeps a stale click harmless.
        """
        if round_id is None:
            return False
        with self.lock:
            state = self.registries[kind].get(round_id)
        if state is None:
            return False
        mutate(state)
        state["event"].set()
        return True

    def remount_for_session(self, session_id: str) -> None:
        """UI THREAD: push every kind's head for ``session_id`` in one call."""
        for kind in KIND_SETTER_ATTRS:
            setter = self._setter(kind)
            if setter is not None:
                setter(self.head_round_payload(kind, session_id))
```

- [ ] **Step 3: Delegate the three `resolve_pending_*` bodies**

Each keeps its signature, docstring, and any bridge-specific validation, and ends in one `host.resolve` call. Example for install (adapt each from its current body — the mutate closure carries exactly what the current body writes into the state):

```python
        self._interrupt_host.resolve(
            "skill_install",
            request_id,
            lambda state: state["decision"].update({"allow": bool(allow)}),
        )
```

For approvals, the mutate merges `decisions` into the state's shared box exactly as the current body does (grep the body; it snapshots then `update`s). Preserve any logging/telemetry lines around it.

- [ ] **Step 4: Collapse the four re-derive sites**

In `new_session`, `switch_session`, and `close_session`, replace each site's THREE per-kind blocks (the `set_pending_approval(self._head_round_payload(...))` call plus `_remount_parked_skill_install(...)` plus `_remount_parked_skill_script(...)`) with one:

```python
        self._interrupt_host.remount_for_session(session_id)
```

(local variable name differs per site — `session.id` / `session_id` / `new_active_id`). Then rewrite `remount_pending_approval_for_active_session` to use `head_round_payload` via the host (keep its bool return: True when a payload mounted), and delete `_remount_parked_skill_install` / `_remount_parked_skill_script` if no caller remains — grep first; docstring mentions do not count.

- [ ] **Step 5: Run the parity battery + the wider Chat/UI interrupt set, foreground**

The Task 1 battery, expected identical to baseline. Then `Tests/UI/test_chat_task_cards_sync.py` and `Tests/UI/test_destination_shells.py` if not already in the battery.

- [ ] **Step 6: Grep-audit the done criteria**

```bash
grep -n "threading.Lock()" tldw_chatbook/Chat/console_chat_controller.py   # the three round locks must be gone (others may remain)
grep -rn "_clear_pending" tldw_chatbook/Chat/console_chat_controller.py    # still absent
grep -c "use_human_input_wait" tldw_chatbook/Chat/console_chat_controller.py  # 1 (import) or callers outside the bridges only
```

Account for every hit in your report.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_interrupt_rounds.py Tests/Chat/test_console_interrupt_rounds.py tldw_chatbook/Chat/console_chat_controller.py
git commit -m "refactor(console): host owns resolve and remount; re-derive sites collapse (C1)"
```

---

## Done criteria

- `console_interrupt_rounds.py` exists with host unit tests that never import `ConsoleChatController`.
- The nine legacy names are live aliases; `grep -rn "threading.Lock()"` in the controller shows the three round locks gone.
- All three bridges' bodies contain one `run_round` call each; no inlined poll loop remains in the controller.
- The parity battery matches Task 1's baseline exactly, including the characterized failure mode of the shutdown tests.
- No existing test file was modified.
