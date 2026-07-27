# Parallel Agents Across Workspaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the two-PR train from
`Docs/superpowers/specs/2026-07-26-parallel-agents-across-workspaces-design.md`:
PR1 makes Console runs per-session (capped globally, Settings-adjustable),
PR2 adds the fleet UX (markers, parked approvals, toasts).

**Architecture:** `ConsoleRunState` becomes a per-session map inside
`ConsoleChatController` behind an **active-session facade property** named
`run_state` — the ~16 existing read sites in `chat_screen.py` keep working
untouched for the viewed session, while run lifecycle WRITES gain an
explicit `session_id` so background completions never stamp the viewed
session. Workers move to per-session exclusive groups; the send gate
becomes per-session + global cap. PR2 layers marker state, parked
approval badges, and toasts on top.

**Tech Stack:** Python 3.11+, Textual, pytest (venv-only:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`
from the repo/worktree root, FOREGROUND only — background pytest runs never
report back).

## Global Constraints

- Branches: PR1 `feat/console-per-session-runs`, PR2
  `feat/console-agent-fleet-ux` stacked on PR1; both off `origin/dev`;
  merged as one train.
- Concurrency rule: one in-flight run per session; global cap
  `[console] max_parallel_runs`, integer minimum 1, **default 3**, no
  upper bound.
- Lowering the cap NEVER kills in-flight runs (count drains naturally).
- No hidden send queue: cap-exceeded sends are refused with the toast in
  Task 2 (exact copy there).
- The run-bound folder-roots behavior from PRs #943/#944 is untouched;
  Task 6 pins it under real concurrency.
- Approvals never auto-resolve; a parked background approval blocks only
  its own session's run.
- Toast-once contracts: one toast per approval card, one per run
  completion/failure.
- Copy strings given in tasks are verbatim requirements.
- Line numbers in this plan are hints only — locate every site by grep,
  never by number (this file was written against dev @ 5d1229eba).
- The plan's implementer-facing rules: work only in the designated
  worktree; foreground pytest in one blocking call; never touch
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook`.

---

## PR1 — per-session runs + cap

### Task 1: per-session run-state map with active-session facade

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (init ~line 596:
  `self.run_state = ConsoleRunState()` and `self.run_state_history`;
  `_set_run_state` ~753; `_clear_terminal_run_state` ~893;
  `switch_session` ~991)
- Test: `Tests/Chat/test_console_run_state_per_session.py` (new)

**Interfaces:**
- Consumes: existing `ConsoleRunState` (frozen dataclass,
  `console_chat_models.py:153` — `status`, `visible_copy`,
  `is_send_allowed`, `is_stop_allowed`), `self.store.active_session_id`,
  `ConsoleRunStatus` enum.
- Produces (later tasks rely on these exact names):
  - `controller.run_state` — property returning the ACTIVE session's
    state (a fresh `ConsoleRunState()` when none recorded). Read-only
    facade; assignment raises `AttributeError` (frozen out so stray
    writers are caught loudly).
  - `controller.run_state_for(session_id: str) -> ConsoleRunState`
  - `controller._set_run_state(state, *, session_id: str | None = None)`
    — `None` means the active session (existing call sites keep working).
  - `controller._clear_terminal_run_state(session_id: str | None = None)`
  - `controller.run_states() -> dict[str, ConsoleRunState]` — snapshot
    copy for fleet displays (PR2) and the cap count (Task 2).
  - `controller.in_flight_run_count() -> int` — sessions whose state has
    `is_send_allowed == False`.
  - Per-session history: `controller.run_state_history_for(session_id)`;
    the legacy `run_state_history` attribute becomes a property for the
    active session.

- [ ] **Step 1: Write the failing tests** (new file; build the controller
  the way `Tests/Chat/test_console_chat_controller*.py` files do — copy the
  minimal fixture idiom from the smallest existing controller test file,
  found via `ls Tests/Chat/ | grep controller`):

```python
"""Per-session Console run state (parallel-agents spec §2)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus


def test_run_states_are_isolated_per_session(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"),
        session_id=session_a,
    )

    assert controller.run_state_for(session_a).status is ConsoleRunStatus.STREAMING
    assert controller.run_state_for(session_b).is_send_allowed
    assert controller.in_flight_run_count() == 1


def test_facade_property_tracks_active_session(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    controller.store.set_active_session(session_a)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"),
        session_id=session_a,
    )

    assert controller.run_state.status is ConsoleRunStatus.STREAMING
    controller.store.set_active_session(session_b)
    assert controller.run_state.is_send_allowed  # B is idle

    with pytest.raises(AttributeError):
        controller.run_state = ConsoleRunState()  # facade is read-only


def test_terminal_clear_is_session_scoped(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run B"), session_id=session_b
    )

    controller._clear_terminal_run_state(session_id=session_a)

    assert controller.run_state_for(session_a).status is ConsoleRunStatus.IDLE
    assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING
```

  Write the `controller_with_two_sessions` fixture in the same file using
  the real store + controller construction idiom you copied (two sessions
  via the store's `new_session`, returning `(controller, id_a, id_b)`).
  Check the store's real method name for activating a session
  (`switch_session` on the controller or `set_active_session` on the
  store — grep and use what exists; adjust the test to the code, never
  the reverse).

- [ ] **Step 2: Run to verify failure**

Run: `.venv-path python -m pytest "Tests/Chat/test_console_run_state_per_session.py" -q`
Expected: FAIL — no `run_state_for` attribute.

- [ ] **Step 3: Implement.** In the controller `__init__`, replace the two
  attributes:

```python
        self._run_states: dict[str, ConsoleRunState] = {}
        self._run_state_histories: dict[str, list[ConsoleRunStatus]] = {}
```

  Facade + accessors (place near the old attribute site):

```python
    @property
    def run_state(self) -> ConsoleRunState:
        """The ACTIVE session's run state (parallel-agents spec §2).

        Read-only facade: the ~16 pre-existing read sites in chat_screen
        keep their semantics ("the viewed session's run"), while writes go
        through _set_run_state with an explicit owning session id.
        """
        return self.run_state_for(self.store.active_session_id or "")

    def run_state_for(self, session_id: str) -> ConsoleRunState:
        return self._run_states.get(session_id) or ConsoleRunState()

    def run_states(self) -> dict[str, ConsoleRunState]:
        return dict(self._run_states)

    def in_flight_run_count(self) -> int:
        return sum(
            1 for state in self._run_states.values() if not state.is_send_allowed
        )

    @property
    def run_state_history(self) -> list[ConsoleRunStatus]:
        return self.run_state_history_for(self.store.active_session_id or "")

    def run_state_history_for(self, session_id: str) -> list[ConsoleRunStatus]:
        return self._run_state_histories.setdefault(
            session_id, [ConsoleRunStatus.IDLE]
        )
```

  Rework `_set_run_state` to accept `*, session_id: str | None = None`,
  resolving `None` to `self.store.active_session_id`, writing into both
  maps. Rework `_clear_terminal_run_state(session_id=None)` the same way
  (only clears when that session's state is terminal — preserve the
  existing terminal-check logic). `switch_session` keeps calling
  `_clear_terminal_run_state()` (now clearing the PREVIOUS active
  session — check the call's position relative to the active-session
  swap and pass the correct explicit id so the semantic "clear the
  session you are leaving if terminal" is preserved; state the choice in
  a comment). Grep every `self.run_state =` assignment in the controller
  (~9 sites) and convert to `_set_run_state(...)` calls with the correct
  session id — sites inside run/completion callbacks must use the run's
  OWNING session id, which those code paths already have in scope as
  `session_id`/`owner_id` locals (grep the enclosing function; if a site
  truly has no session in scope, use `None` and add a comment saying it
  is an active-session UI path).

- [ ] **Step 4: Run to verify pass**

Run: `.venv-path python -m pytest "Tests/Chat/test_console_run_state_per_session.py" "Tests/Chat/" -q -k "run_state or controller"`
Expected: PASS (new file + existing controller suites).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_run_state_per_session.py
git commit -m "feat(console): per-session run state behind an active-session facade"
```

### Task 2: per-session send gate + global cap + copy retirement

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  (`CONSOLE_RUN_ALREADY_RUNNING_COPY` ~373 and its 3 use sites ~11285,
  ~12245, ~12259 — grep the constant), `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_run_state_per_session.py` (append) +
  the chat_screen gate's existing tests (grep
  `CONSOLE_RUN_ALREADY_RUNNING_COPY` in Tests/ and update pins)

**Interfaces:**
- Consumes: Task 1's `in_flight_run_count`, `run_state_for`.
- Produces:
  - `controller.max_parallel_runs` — property reading
    `[console] max_parallel_runs` via the controller's existing config
    seam (grep how the controller reads `[console]` values — mirror it;
    if the controller has no config seam, read via
    `tldw_chatbook.config.get_cli_setting("console", "max_parallel_runs", 3)`),
    coerced `int`, clamped to `>= 1`.
  - `controller.send_refusal_copy(session_id) -> str | None` — `None`
    when the send is allowed; the per-session copy
    `"A run is already running in this tab."` when THAT session is busy;
    otherwise the cap copy when `in_flight_run_count() >= max_parallel_runs`:
    `f"{count} agents already running ({titles}). Wait for one to finish or interrupt it."`
    where `titles` = first 3 busy sessions' titles joined with ", ",
    plus `f" and {k} more"` when more than 3.
  - chat_screen: the three `CONSOLE_RUN_ALREADY_RUNNING_COPY` sites call
    `send_refusal_copy` and toast its result; the module constant is
    deleted.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_send_refusal_is_per_session_and_capped(controller_with_two_sessions, monkeypatch):
    controller, session_a, session_b = controller_with_two_sessions
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda self: 1)
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"), session_id=session_a
    )

    assert controller.send_refusal_copy(session_a) == (
        "A run is already running in this tab."
    )
    refusal = controller.send_refusal_copy(session_b)
    assert refusal is not None and "1 agents already running" in refusal
    assert "Wait for one to finish or interrupt it." in refusal


def test_cap_default_and_floor(controller_with_two_sessions, monkeypatch):
    controller, _, _ = controller_with_two_sessions
    import tldw_chatbook.Chat.console_chat_controller as ccc
    monkeypatch.setattr(
        ccc, "get_cli_setting", lambda *a, **k: 0, raising=False
    )
    assert controller.max_parallel_runs == 1  # floor
    monkeypatch.setattr(
        ccc, "get_cli_setting", lambda *a, **k: None, raising=False
    )
    assert controller.max_parallel_runs == 3  # default


def test_lowering_cap_never_kills_running(controller_with_two_sessions, monkeypatch):
    controller, session_a, session_b = controller_with_two_sessions
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "B"), session_id=session_b
    )
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda self: 1)
    )
    # Both stay streaming; only NEW sends are refused.
    assert controller.run_state_for(session_a).status is ConsoleRunStatus.STREAMING
    assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING
    assert controller.in_flight_run_count() == 2
```

  (Adjust the config monkeypatch target to the real import the
  implementation uses — the test must patch where the name is LOOKED UP.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv-path python -m pytest "Tests/Chat/test_console_run_state_per_session.py" -q`
Expected: FAIL — no `send_refusal_copy`.

- [ ] **Step 3: Implement** `max_parallel_runs` + `send_refusal_copy` on
  the controller:

```python
    @property
    def max_parallel_runs(self) -> int:
        """User-adjustable global cap (spec §4); floor 1, default 3."""
        raw = get_cli_setting("console", "max_parallel_runs", 3)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 3
        if raw is None:
            value = 3
        return max(1, value)

    def send_refusal_copy(self, session_id: str) -> str | None:
        """Why a send to ``session_id`` must be refused, or None if allowed."""
        if not self.run_state_for(session_id).is_send_allowed:
            return "A run is already running in this tab."
        busy = [
            sid
            for sid, state in self._run_states.items()
            if not state.is_send_allowed
        ]
        if len(busy) < self.max_parallel_runs:
            return None
        titles = []
        for sid in busy[:3]:
            session = next(
                (s for s in self.store.sessions() if s.id == sid), None
            )
            titles.append(session.title if session is not None else sid)
        suffix = f" and {len(busy) - 3} more" if len(busy) > 3 else ""
        return (
            f"{len(busy)} agents already running "
            f"({', '.join(titles)}{suffix}). "
            "Wait for one to finish or interrupt it."
        )
```

  Then in chat_screen: grep `CONSOLE_RUN_ALREADY_RUNNING_COPY` — at each
  of the 3 use sites, replace the `is_send_allowed` check + constant
  toast with `refusal = controller.send_refusal_copy(<target session id>)`
  / `if refusal: notify(refusal, severity="warning"); return`. The target
  session id at those sites is the active session (they are the composer
  send/retry paths) — pass `controller.store.active_session_id`. Delete
  the constant; update any test pinning it (grep Tests/) to the new
  per-session copy.

- [ ] **Step 4: Run to verify pass**

Run: `.venv-path python -m pytest "Tests/Chat/test_console_run_state_per_session.py" -q` then the pinned-copy suites you touched.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/
git commit -m "feat(console): per-session send gate with user-adjustable global cap"
```

### Task 3: per-session worker groups + scoped interrupt

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — all 7
  `group="console-run"` sites (grep `group="console-run"`)
- Test: `Tests/UI/test_console_parallel_runs.py` (new, mounted
  ConsoleHarness — import idiom from
  `Tests/UI/test_console_new_workspace.py`)

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: every run worker dispatched as
  `group=f"console-run-{session_id}", exclusive=True` where `session_id`
  is the TARGET session at dispatch time (the active session in the
  composer paths; the owning session in retry/regenerate paths — those
  handlers already resolve a message/session, grep the enclosing
  functions). Interrupt/stop paths cancel only their session's group
  (grep how the stop action cancels workers today — if it uses
  `workers.cancel_group` or the run worker handle, scope it by the same
  session-derived group name).

- [ ] **Step 1: Write the failing test:**

```python
"""Two sessions run concurrently; interrupt is session-scoped (spec §2)."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus


@pytest.mark.asyncio
async def test_two_sessions_run_concurrently_and_interrupt_is_scoped() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_a = store.active_session_id
        session_b = controller.new_session().id  # check real API by grep

        release_a = asyncio.Event()
        release_b = asyncio.Event()

        async def fake_run(session_id, release):
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
                session_id=session_id,
            )
            await release.wait()
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
                session_id=session_id,
            )

        console.run_worker(
            fake_run(session_a, release_a),
            exclusive=True,
            group=f"console-run-{session_a}",
        )
        console.run_worker(
            fake_run(session_b, release_b),
            exclusive=True,
            group=f"console-run-{session_b}",
        )
        await pilot.pause(0.2)
        assert controller.in_flight_run_count() == 2  # truly concurrent

        # Cancelling A's group leaves B running.
        console.workers.cancel_group(console, f"console-run-{session_a}")
        release_b.set()
        await pilot.pause(0.3)
        assert controller.run_state_for(session_b).status is ConsoleRunStatus.COMPLETED
```

  (Verify `new_session` and `workers.cancel_group` signatures by grep and
  adjust the TEST to reality.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv-path python -m pytest "Tests/UI/test_console_parallel_runs.py" -q`
Expected: FAIL only if the harness idioms are wrong OR — after Task 1-2 —
this test may pass at the worker level; the REAL red for this task is
production sites still using the shared group. Verify with:
`grep -c 'group="console-run"' tldw_chatbook/UI/Screens/chat_screen.py`
Expected before implementation: 7.

- [ ] **Step 3: Implement.** At each of the 7 sites, derive the target
  session id already in scope (the send path computes it before
  dispatch; retry/regenerate paths resolve it from the message/action
  context — grep the enclosing function for `session_id`/`active_session_id`)
  and change to `group=f"console-run-{target_session_id}"`. Then grep any
  stop/interrupt handler that cancels the `"console-run"` group and scope
  it identically. After the change:
  `grep -c 'group="console-run"' ...` must be 0 (the literal), and
  `grep -c 'group=f"console-run-' ...` must be 7 (+ any cancel sites).

- [ ] **Step 4: Run to verify pass**

Run: `.venv-path python -m pytest "Tests/UI/test_console_parallel_runs.py" "Tests/UI/test_console_native_chat_flow.py" -q`
Expected: PASS (native flow is the big regression net for send paths; ~4 min).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_parallel_runs.py
git commit -m "feat(console): per-session run worker groups with scoped interrupt"
```

### Task 4: background-write audit (store-first discipline)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` and/or
  `tldw_chatbook/Chat/console_chat_controller.py` (sites found by the
  audit)
- Test: `Tests/UI/test_console_parallel_runs.py` (append)

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: every streaming-delta, tool-marker, generation-card, and
  run-status view write is gated on
  `writing_session_id == store.active_session_id`; store writes are
  unconditional. No new names — this is a discipline pass.

- [ ] **Step 1: Run the audit.** Grep the streaming/tool paths for view
  writes: candidates are the callbacks that append/patch transcript
  widgets (`grep -n "call_from_thread\|_append_.*message\|update_stream\|stream_delta\|tool_marker" tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_chat_controller.py`).
  For each hit, answer: does this mutate a VIEW object (widget/transcript)
  without checking the owning session? List every such site in the
  commit message. (The store-tree writes are already per-session — leave
  them.)

- [ ] **Step 2: Write the failing test** (append; drive a fake background
  run against a NON-active session using the same seam the audit found —
  the test must fail if any audited site writes the viewed transcript):

```python
@pytest.mark.asyncio
async def test_background_run_never_mutates_viewed_transcript() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.set_active_session(viewed)  # keep viewing the first session

        before = console.query_one("#console-native-transcript").render()
        # Drive the audited append path for the BACKGROUND session directly
        # (use the real method the audit gated, e.g. the assistant-delta
        # apply seam, with session_id=background).
        console._apply_console_stream_delta(background, "SHOULD-NOT-APPEAR")
        await pilot.pause(0.2)
        after = console.query_one("#console-native-transcript").render()
        assert "SHOULD-NOT-APPEAR" not in str(after)
        assert str(before) == str(after)
```

  The seam name `_apply_console_stream_delta` is illustrative — use the
  REAL method the audit identifies, and assert on the real transcript
  widget id (grep `console-native-transcript` for the actual id; adjust
  the test to the code).

- [ ] **Step 3: Implement the gates** at every audited site:
  `if session_id != self.store.active_session_id: return` (view portion
  only — store writes stay). Where a callback lacks the session id, thread
  it from the dispatch site (the run knows its session from Task 3).

- [ ] **Step 4: Run to verify pass**

Run: `.venv-path python -m pytest "Tests/UI/test_console_parallel_runs.py" "Tests/UI/test_console_native_chat_flow.py" -q`
Expected: PASS.

- [ ] **Step 5: Commit** (list the audited sites in the body)

```bash
git add -A tldw_chatbook/ Tests/UI/test_console_parallel_runs.py
git commit -m "fix(console): gate view writes on viewed session (background-run audit)"
```

### Task 5: Settings row for the cap (Console Behavior)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (Console Behavior
  card + staging + guidance — mirror an existing numeric `[console]` row:
  grep `_console_behavior_loaded_values` ~2597 and the card renderer)
- Test: `Tests/UI/test_settings_configuration_hub.py` (append)

**Interfaces:**
- Consumes: the guided Console Behavior draft machinery
  (`GUIDED_SETTINGS_MUTATION_CATEGORIES` already contains
  CONSOLE_BEHAVIOR; staging via the card's existing `_stage_*` pattern).
- Produces: input id `settings-console-max-parallel-runs`, config key
  `console.max_parallel_runs`, validation integer `>= 1` (non-numeric and
  `< 1` rejected inline with the category's standard validation copy).
  Focused-field guidance copy VERBATIM:
  purpose `"How many agent runs may be in flight at once, across all tabs."`;
  consequences `"Each concurrent run holds a provider generation, its own tool activity, and memory for its transcript. Local providers (llama.cpp) typically serialize or slow under concurrent generations; high values can exhaust provider slots, rate limits, or RAM. Raise it as far as you like - the app enforces no ceiling."`;
  saved-as `console.max_parallel_runs`; applies-to-new-sends note
  `"Applies to new sends on save; running agents are never stopped by lowering it."`

- [ ] **Step 1: Write the failing test** (append to the hub suite,
  mirroring an existing Console Behavior save test found via
  `grep -n "console_behavior" Tests/UI/test_settings_configuration_hub.py`):

```python
@pytest.mark.asyncio
async def test_console_behavior_exposes_max_parallel_runs(monkeypatch):
    saved = {}
    # mirror the existing console-behavior save test's monkeypatch of the
    # settings persistence seam, capturing section values into `saved`.
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        field = screen.query_one("#settings-console-max-parallel-runs", Input)
        assert field.value == "3"  # default surfaced

        field.value = "0"
        # stage + save via the category's existing save action; assert the
        # inline validation copy appears and nothing was persisted.
        screen.action_settings_save_category()
        await pilot.pause(0.3)
        assert "must be an integer of at least 1" in _visible_text(screen)

        field.value = "12"
        screen.action_settings_save_category()
        await pilot.pause(0.3)
        assert saved.get("console", {}).get("max_parallel_runs") == 12
```

  (Adapt harness/save/staging idioms to the neighboring Console Behavior
  tests — they are the source of truth for how staging and the save
  action are driven; the validation copy
  `"must be an integer of at least 1"` is the verbatim requirement.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv-path python -m pytest "Tests/UI/test_settings_configuration_hub.py" -q -k max_parallel`
Expected: FAIL — no such input id.

- [ ] **Step 3: Implement**: loaded-values entry (default `3`), card row
  (label `"Max parallel agent runs"`, `Input` with the id above,
  `settings-compact-input` class), staging on `Input.Changed` via the
  card's existing pattern, validation in the category's validator
  (integer, `>= 1`, inline copy
  `"Max parallel agent runs must be an integer of at least 1."`),
  save-section wiring so it persists under `[console]`, and the
  focused-field guidance rows with the four verbatim strings from
  **Interfaces**.

- [ ] **Step 4: Run to verify pass**

Run: `.venv-path python -m pytest "Tests/UI/test_settings_configuration_hub.py" -q`
Expected: PASS (full hub).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py
git commit -m "feat(settings): user-adjustable max parallel agent runs (Console Behavior)"
```

### Task 6: workspace-isolation concurrency pin + PR1 assembly

**Files:**
- Test: `Tests/Agents/test_builtin_provider_workspace_binding.py` (append)
- No production changes expected.

**Interfaces:**
- Consumes: `BuiltinToolProvider(gate=..., workspace_id=...)`,
  `workspace_file_roots.run_workspace/current_run_workspace_id` (merged
  #943), `_ProbeTool`/`_OpenGate` idioms already in the test file.

- [ ] **Step 1: Write the failing-or-green pin** (this is a regression
  pin; it should pass immediately — its value is catching future breaks):

```python
def test_concurrent_providers_keep_distinct_workspace_bindings() -> None:
    """Spec §7: two overlapping runs resolve different roots (ContextVar
    isolation across interleaved invokes)."""
    import threading

    results: dict[str, str | None] = {}

    class _EchoTool:
        name = "probe_workspace"
        description = "echo bound workspace"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            import asyncio
            from tldw_chatbook.Tools import workspace_file_roots as wfr
            await asyncio.sleep(0.05)  # force overlap window
            return {"workspace": wfr.current_run_workspace_id()}

    def run(workspace_id: str) -> None:
        provider = BuiltinToolProvider(gate=_OpenGate(), workspace_id=workspace_id)
        provider._tools["probe_workspace"] = _EchoTool()
        result = provider.invoke("builtin:probe_workspace", {})
        results[workspace_id] = result.content

    threads = [
        threading.Thread(target=run, args=("ws-alpha",)),
        threading.Thread(target=run, args=("ws-beta",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert '"workspace": "ws-alpha"' in (results["ws-alpha"] or "")
    assert '"workspace": "ws-beta"' in (results["ws-beta"] or "")
```

- [ ] **Step 2: Run it** — Expected: PASS (ContextVars are thread-local;
  if it fails, STOP: that is a real isolation bug — report BLOCKED).

- [ ] **Step 3: PR1 full verification**

Run: `.venv-path python -m pytest "Tests/Chat/" "Tests/UI/test_console_parallel_runs.py" "Tests/UI/test_settings_configuration_hub.py" "Tests/Agents/" -q`
then: `.venv-path python -m pytest "Tests/UI/test_console_native_chat_flow.py" -q`
Expected: PASS both (theme-editor mount flake under load is a documented
baseline exception; native flow must be 201 passed).

- [ ] **Step 4: Commit** (no push — controller handles PR creation after
  the final review + smoke)

```bash
git add Tests/Agents/test_builtin_provider_workspace_binding.py
git commit -m "test(agents): pin workspace-binding isolation under concurrent runs"
```

---

## PR2 — fleet UX
*(branch `feat/console-agent-fleet-ux` off PR1's branch)*

### Task 7: run-marker state + clear-on-visit

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (marker enum),
  `tldw_chatbook/Chat/console_chat_controller.py` (marker map)
- Test: `Tests/Chat/test_console_run_markers.py` (new)

**Interfaces:**
- Consumes: Task 1's per-session states; run completion paths.
- Produces:
  - `class ConsoleRunMarker(str, Enum)` in `console_chat_models.py`:
    `NONE = "none"`, `RUNNING = "running"`,
    `NEEDS_APPROVAL = "needs-approval"`, `FINISHED_OK = "finished-ok"`,
    `FINISHED_FAILED = "finished-failed"`.
  - Controller: `run_marker_for(session_id) -> ConsoleRunMarker` derived
    live: `RUNNING` when in-flight; `NEEDS_APPROVAL` when the session's
    run has a pending approval (Task 9 supplies
    `set_pending_approval(session_id, bool)` — Task 7 stores the flag:
    `self._pending_approvals: set[str]`); `FINISHED_OK`/`FINISHED_FAILED`
    when the session's terminal state is COMPLETED/FAILED AND the session
    is in `self._unvisited_outcomes: dict[str, ConsoleRunMarker]`
    (stamped by the terminal `_set_run_state`); `NONE` otherwise.
  - `mark_session_visited(session_id)` — clears the unvisited outcome and
    pending-approval flag; called from `switch_session`.
  - `fleet_summary_counts() -> tuple[int, int]` — (other running, other
    pending-approval) relative to the active session.

- [ ] **Step 1: Write the failing tests:**

```python
"""Run markers: running/needs-approval/finished-unvisited (spec §6)."""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
)


def test_marker_lifecycle(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.RUNNING

    controller.set_pending_approval(session_a, True)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    controller.set_pending_approval(session_a, False)

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.FINISHED_OK

    controller.mark_session_visited(session_a)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE


def test_failed_marker_and_fleet_counts(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    controller.store.set_active_session(session_a)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=session_b
    )
    running, pending = controller.fleet_summary_counts()
    assert (running, pending) == (1, 0)

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, "boom"), session_id=session_b
    )
    assert controller.run_marker_for(session_b) is ConsoleRunMarker.FINISHED_FAILED
```

  (Reuse the Task 1 fixture via import or a conftest move — put
  `controller_with_two_sessions` into `Tests/Chat/conftest.py` if both
  files need it.)

- [ ] **Step 2: Run to verify failure** — Expected: FAIL, no
  `ConsoleRunMarker`.

- [ ] **Step 3: Implement** per Interfaces (marker derivation lives in
  the controller; `_set_run_state` stamps `_unvisited_outcomes` on
  COMPLETED/FAILED terminal transitions for NON-active sessions only —
  the viewed session's outcome is seen live and never becomes
  "unvisited"; `switch_session` calls `mark_session_visited(new_id)`).

- [ ] **Step 4: Run to verify pass** — the new file + Task 1's file.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/
git commit -m "feat(console): per-session run markers with clear-on-visit"
```

### Task 8: tab + sidebar marker rendering and fleet summary line

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (tab-bar label build —
  grep the session-tab label builder; Agent rail section body),
  `tldw_chatbook/Workspaces/conversation_browser_state.py` +
  `tldw_chatbook/Widgets/Console/console_workspace_context.py`
  (browser-row marker glyph)
- Test: `Tests/UI/test_console_parallel_runs.py` (append)

**Interfaces:**
- Consumes: Task 7's `run_marker_for`, `fleet_summary_counts`.
- Produces: glyph map (verbatim): RUNNING `"●"`, NEEDS_APPROVAL `"◆"`,
  FINISHED_OK `"✓"`, FINISHED_FAILED `"✗"`, NONE `""` — exposed as
  `CONSOLE_RUN_MARKER_GLYPHS: dict[ConsoleRunMarker, str]` in
  `console_chat_models.py`; tab labels prefix the glyph; browser input
  rows gain `run_marker: str = ""` (threaded like `openable` was in
  TASK-717 — input row → normalize → display row → row label); Agent
  rail renders `f"{running} other agents running, {pending} waiting for approval."`
  only when `running + pending > 0`, id
  `console-agent-fleet-summary`.

- [ ] **Step 1: Write the failing tests** (append; mounted):

```python
@pytest.mark.asyncio
async def test_tab_and_sidebar_show_run_markers_and_fleet_line() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        background = controller.new_session().id
        controller.store.set_active_session(controller.store.sessions()[0].id)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"),
            session_id=background,
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)

        text = _visible_text(console)
        assert "●" in text  # running glyph on tab/row
        assert "1 other agents running, 0 waiting for approval." in text
```

  (Import `_visible_text` from the workspace-context rail test module as
  other files do; adjust session-creation idioms to the real API.)

- [ ] **Step 2: Run to verify failure** — Expected: FAIL, no glyph/line.

- [ ] **Step 3: Implement** per Interfaces: glyph dict; tab label builder
  prefixes `CONSOLE_RUN_MARKER_GLYPHS[marker]`; browser rows thread
  `run_marker` exactly like TASK-717 threaded `openable`
  (input dataclass field → `_normalize_input_row` → `_to_browser_row` →
  row label prefix in the tray); chat_screen populates it from
  `controller.run_marker_for(...)` when building browser input rows;
  Agent rail body appends the fleet Static when counts are non-zero.

- [ ] **Step 4: Run to verify pass** — the parallel-runs file +
  `Tests/UI/test_console_workspace_context_rail.py` +
  `Tests/UI/test_console_rail_sections.py`.

- [ ] **Step 5: Commit**

```bash
git add -A tldw_chatbook/ Tests/UI/test_console_parallel_runs.py
git commit -m "feat(console): fleet markers on tabs and sidebar + Agent rail summary"
```

### Task 9: parked background approvals

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
  (`build_tool_review_hook` ~line 279 — grep it) and the chat_screen
  approval-card mount path (grep `Approval required` / the card widget's
  mount site)
- Test: `Tests/UI/test_console_parallel_runs.py` (append)

**Interfaces:**
- Consumes: Task 7's `set_pending_approval`; the review hook's existing
  await-decision flow; Task 3's session-scoped runs.
- Produces: when the hook raises a card for a session that is NOT the
  active session: no card mounts; `set_pending_approval(session_id, True)`;
  ONE toast `f"Agent in {session_title} ({workspace_name}) needs approval."`
  (workspace name resolved via the session's workspace_id through the
  registry; fall back to the raw id). Visiting the session
  (`switch_session`) mounts the pending card through the existing mount
  path and the flag clears when the decision is submitted (approve or
  deny), not merely on visit — `set_pending_approval(sid, False)` hooks
  the decision-submit path. Card state derives from the run's pending
  review state, so switching away and back re-mounts it.

- [ ] **Step 1: Write the failing test** (append):

```python
@pytest.mark.asyncio
async def test_background_approval_parks_with_badge_and_single_toast() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.set_active_session(viewed)

        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        # Drive the review hook's park path for the background session via
        # the seam the implementation adds (grep its real name after
        # implementing; the test asserts observable behavior):
        console._park_console_approval(background)
        await pilot.pause(0.3)

        assert not console.query("#console-approval-card")  # not mounted
        approval_toasts = [n for n in notifications if "needs approval" in n]
        assert len(approval_toasts) == 1
        from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
        assert (
            controller.run_marker_for(background)
            is ConsoleRunMarker.NEEDS_APPROVAL
        )

        # Visiting mounts the card.
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)
        assert console.query("#console-approval-card")
```

  (`_park_console_approval` and `#console-approval-card` are the
  interface names this task produces — if the existing card widget has a
  different stable id, use the REAL id in both code and test; grep the
  approval-card compose site first and record the actual id in the commit
  message.)

- [ ] **Step 2: Run to verify failure** — Expected: FAIL, no park seam.

- [ ] **Step 3: Implement** per Interfaces. The review hook already knows
  its run's session (Task 3 threads it); the park branch replaces the
  unconditional card mount with: active-session check → mount as today
  OR park (flag + toast-once guard per card + no mount). `switch_session`
  checks for a parked card on the newly-viewed session and mounts it.
  Decision submit clears the flag via `set_pending_approval(sid, False)`.

- [ ] **Step 4: Run to verify pass** — the parallel-runs file + the
  approval-related suites (grep Tests/ for the review-hook tests:
  `grep -rln "build_tool_review_hook\|Approve all" Tests/ | head` and run
  those files).

- [ ] **Step 5: Commit**

```bash
git add -A tldw_chatbook/ Tests/UI/test_console_parallel_runs.py
git commit -m "feat(console): park background approvals with badge and single toast"
```

### Task 10: completion toasts + PR2 assembly

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (terminal
  `_set_run_state` transitions) or chat_screen's completion callbacks —
  wherever Task 1 routed terminal states; put the toast beside the
  `_unvisited_outcomes` stamp so it fires exactly once per run
- Test: `Tests/UI/test_console_parallel_runs.py` (append)

**Interfaces:**
- Consumes: Task 7's stamp point.
- Produces: on a NON-active session's terminal transition, ONE toast:
  `f"Agent in {session_title} ({workspace_name}) finished."` or
  `"... failed."` (same workspace-name resolution as Task 9). Active
  session: no toast (the user is watching).

- [ ] **Step 1: Write the failing test** (append):

```python
@pytest.mark.asyncio
async def test_background_completion_fires_single_toast() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.set_active_session(viewed)
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"), session_id=background
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        finished = [n for n in notifications if "finished" in n]
        assert len(finished) == 1

        # Re-setting the same terminal state must not double-toast.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        assert len([n for n in notifications if "finished" in n]) == 1
```

  (The controller needs a notify seam for toasts — grep how the
  controller currently surfaces notifications to the app; if toasts are
  screen-level today, put the toast in the screen's completion callback
  instead and drive that seam in the test. Adjust test to reality.)

- [ ] **Step 2: Run to verify failure** — Expected: FAIL, no toast.

- [ ] **Step 3: Implement** with a once-guard (toast only on the
  transition INTO the terminal state, i.e. when the previous status for
  that session was non-terminal).

- [ ] **Step 4: PR2 full verification**

Run: `.venv-path python -m pytest "Tests/Chat/" "Tests/UI/test_console_parallel_runs.py" "Tests/UI/test_console_workspace_context_rail.py" "Tests/UI/test_console_rail_sections.py" "Tests/UI/test_settings_configuration_hub.py" -q`
then `.venv-path python -m pytest "Tests/UI/test_console_native_chat_flow.py" -q`
Expected: PASS (native flow 201).

- [ ] **Step 5: Commit**

```bash
git add -A tldw_chatbook/ Tests/UI/test_console_parallel_runs.py
git commit -m "feat(console): background-run completion toasts (once per run)"
```

### Task 11: live smoke (non-negotiable before merge)

**Files:** none (verification only; controller-driven).

- [ ] **Step 1:** Launch per the folder-roots smoke recipe: scratch
  profile (`TLDW_CONFIG_PATH`, fresh `users_name`), llama.cpp at :9099,
  tmux socket `wssmoke`, `[tools] read_file_enabled = true`, from the PR2
  worktree. Create TWO workspaces in Settings ▸ Workspaces, bind a
  distinct temp folder to each (rw not required).
- [ ] **Step 2:** Open a tab in each workspace (switch active workspace,
  Ctrl+T). Start an agent turn in BOTH tabs (read_file instructions
  targeting each tab's own bound folder). Verify: both run concurrently
  (Agent rail fleet line "1 other agents running…" while viewing either),
  no cap refusal at default 3.
- [ ] **Step 3:** While viewing tab A, let tab B's run hit its approval
  card: verify NO card over tab A, `◆` on tab B + its sidebar row, one
  toast. Visit tab B, Approve all THEN Submit (row-level clicks do not
  stamp). Verify each run's read succeeds only inside its OWN workspace's
  folder (transcripts show the two distinct markers/contents).
- [ ] **Step 4:** Let both complete; verify one completion toast each and
  ✓ markers that clear on visit. Capture pane dumps of the fleet line,
  the parked badge, and both transcripts into the session scratchpad.
- [ ] **Step 5:** Teardown: kill the tmux server, delete the scratch
  profile dir.

## Self-review notes (already applied)

- Spec coverage: §2→Tasks 1-3, §3→Task 4, §4→Tasks 2+5, §5→Task 9,
  §6→Tasks 7-8+10, §7→per-task tests + Tasks 6 and 11, §8→PR boundaries,
  §9 honored (no queues, no per-workspace caps, no approval-semantics
  changes).
- Name consistency: `run_state_for`, `run_states`, `in_flight_run_count`,
  `send_refusal_copy`, `max_parallel_runs`, `ConsoleRunMarker`,
  `CONSOLE_RUN_MARKER_GLYPHS`, `run_marker_for`, `set_pending_approval`,
  `mark_session_visited`, `fleet_summary_counts`, group
  `f"console-run-{session_id}"` — used identically across tasks.
- Deliberate verify-at-implementation flags: store's activate-session
  API name (T1), config lookup target for monkeypatch (T2), retry-path
  session resolution (T3), audit seam + transcript widget id (T4),
  Console Behavior staging idiom (T5), approval-card real id + hook
  session threading (T9), controller notify seam (T10).
