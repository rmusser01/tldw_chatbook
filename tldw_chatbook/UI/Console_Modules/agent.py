"""Console agent controller.

Extracted out of `ChatScreen` (wave-4 console decomposition, task 3): the
agent runtime's screen-side cluster -- the lazily-built `ConsoleAgentBridge`
handle, the Agent rail section's text derivation, the sub-agent drill-in
cycle, the "View full log" target/probe/loader, the fleet auto-open
override, the conversation-browser's `[N Sub-Agents]` badge-count cache, and
the resume-time TOOL-marker re-derivation.

The binding rule this file follows is stated canonically in
`ConsoleDictationController.__init__`'s docstring (`dictation.py`); the
construction itself lives in `Console_Modules/wiring.py` (wave-4 task 1),
not in `ChatScreen.__init__`.

Moved from `ChatScreen`, thirteen methods byte-for-byte plus one split:

- `_ensure_console_agent_bridge`, `_console_agent_section_lines`,
  `_console_agent_fleet_summary_line`, `_console_agent_full_log_run_id`,
  `_console_agent_full_log_available`, `_open_console_agent_run_log_viewer`,
  `_load_console_agent_run_log`, `_show_console_agent_run_log_modal`,
  `_toggle_console_agent_drilldown_from_subagents_click`,
  `_apply_fleet_agent_section_auto_open`,
  `_console_subagent_counts_refresh_needed`,
  `_console_subagent_counts_for_rows`, `_inject_resume_agent_markers`.
- `_console_agent_section_payload` -- the non-DOM half of
  `_sync_console_agent_section`. See that method's own docstring below.

Four bodies are NOT byte-for-byte, each for a reason the binding rule
already names:

1. `_load_console_agent_run_log` lost its `@work(thread=True)` decorator.
   Textual's decorator asserts `isinstance(self, DOMNode)` (`textual/
   _work_decorator.py`), and a controller is deliberately not one, so the
   dispatch moved to its caller as an explicit `run_worker(..., thread=
   True, group="default", name="_load_console_agent_run_log")`. Those are
   the exact values `@work(thread=True)` itself passes -- the pre-move
   worker group name is preserved, per the binding rule.
2. ...and its `self.app.call_from_thread(...)` became `self.call_from_
   thread(...)`, reaching the framework service through this controller's
   own property rather than through a whole `App` handle.
3. `_show_console_agent_run_log_modal`'s `self.app.push_screen(...)`
   became `self.push_screen(...)`, same reason (and the same shape
   `ConsoleMessageController` already uses).
4. `_console_subagent_counts_refresh_needed` imports its two module-level
   constants (`CONSOLE_ACTIVE_RUN_STATUSES`, `CONSOLE_SUBAGENT_COUNTS_
   CACHE_TTL_SECONDS`) inside the method body. They still live in
   `chat_screen.py` -- `Tests/UI/test_console_native_chat_flow.py` imports
   the first from there -- and a module-level import here would be a cycle
   that fails for real: `chat_screen.py` imports `Console_Modules.wiring`
   at line 89, which imports this module, and both constants are defined
   at line 587+, so a cold `import chat_screen` would find them missing.

**Zero DOM.** No `query_one`/`query`/`mount` traffic reaches through
`screen` here, matching every other controller. That is what keeps
`_sync_console_agent_section` on `ChatScreen`: its second half is nine
`query_one` calls against the mounted rail. Only its first half -- deriving
the seven-field payload, the part that reads the bridge, the run store, the
rail state and the full-log probe -- moved, as `_console_agent_section_
payload`. The screen's own method is now the equality guard plus the DOM
writes, and `_console_agent_section_last` (that guard's memo, and nothing
else's) stayed with it.

`ChatScreen` keeps seven one-line delegations, under their ORIGINAL private
names, because each is reached from outside this cluster:
`_ensure_console_agent_bridge` (five screen-level call sites -- the chat
controller's construction and core-state sync, the conversation browser,
the Change Review opener -- plus three test sites that replace it on the
screen instance), `_console_agent_section_lines`,
`_console_agent_fleet_summary_line`, `_toggle_console_agent_drilldown_from_
subagents_click`, `_open_console_agent_run_log_viewer` and `_console_
subagent_counts_for_rows` (screen-level callers and/or direct test drives),
and `_inject_resume_agent_markers` (`ConsoleWorkspaceController` reaches it
by name through the wiring, and two test files drive it directly). The
other seven moved with no residue.

One narrowing worth stating plainly: a test that replaces `screen.
_ensure_console_agent_bridge` on the instance still steers every SCREEN-side
consumer (they call the delegation), but no longer steers this controller's
own internal bridge lookups, which call this class's method directly. Every
such site in the suite today drives screen-owned code
(`Tests/Chat/test_change_turn_tracking.py`, `Tests/Chat/test_console_agent_
swap.py`), so nothing regressed; a future test that needs to stub the
bridge for the rail itself should set `screen._console_agent_bridge`
instead, which is what `Tests/UI/test_console_agent_rail.py` already does
(and which still works, via the read-write proxy on the screen).

`_console_agent_runtime_enabled` is deliberately NOT part of this cluster,
despite matching on name. It is a four-line `[console] agent_runtime`
config read whose only two callers are the screen's own chat-controller
construction and core-state sync; it touches no agent state, and its
identical twin two lines below it in `chat_screen.py`
(`_console_native_tool_calls_enabled`, same shape, same consumers) is not
name-matched and would have been split away from it. See the task-3 report
for the full per-method verdict table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Iterable, TYPE_CHECKING

import time

from loguru import logger
from textual.message_pump import NoActiveAppError

from ...Widgets.Console.console_run_log_modal import ConsoleRunLogModal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...Chat.console_chat_models import ConsoleChatMessage
    from ...Chat.console_rail_state import ConsoleRailState
    from ..Screens.chat_screen import ChatScreen

__all__ = ["ConsoleAgentController"]


class ConsoleAgentController:
    """Owns the Console agent runtime's screen-side state and behaviour.

    See this module's docstring for the map of what moved, what stayed, and
    why.
    """

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        chat_store_accessor: Callable[[], Any],
        provider_gateway_accessor: Callable[[], Any],
        native_tool_calls_enabled_accessor: Callable[[], Any],
        current_rail_conversation_id: Callable[[], str | None],
        current_rail_state_accessor: Callable[[], "ConsoleRailState"],
        chat_controller_accessor: Callable[[], Any],
        sync_native_console_chat_ui_accessor: Callable[[], Any],
    ) -> None:
        """Wire the agent controller. Built in `Console_Modules/wiring.py`.

        The binding rule is stated canonically in
        `ConsoleDictationController.__init__`'s docstring. As it applies
        here:

        1. **Framework services** (`run_worker`, `push_screen`,
           `call_from_thread`) are live-read from the screen via `@property`
           below, never snapshotted.
        2. **`app_instance`** is the one justified snapshot: it never
           changes identity over this controller's life, and
           `_ensure_console_agent_bridge` only ever `getattr`s off it.
        3. Every **app-level dependency** is a named keyword-only callable,
           wired at the call site as a late-binding lambda -- never a bound
           method, which would freeze the screen's CURRENT method and stop
           observing a later `monkeypatch.setattr` on the instance. Two of
           these are patched by name in the pre-existing suite
           (`_current_console_rail_conversation_id` and
           `_current_console_rail_state`, both in
           `Tests/UI/test_console_agent_rail.py`), so this is load-bearing,
           not ceremony.

        Args:
            screen: The Console screen this controller belongs to. Held for
                the framework-service properties only; no DOM is reached
                through it.
            app_instance: The `TldwCli` app. Snapshot (see above).
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`,
                called to hand the bridge its store.
            provider_gateway_accessor:
                `ChatScreen._ensure_console_provider_gateway`, same.
            native_tool_calls_enabled_accessor: Returns the screen's
                `_console_native_tool_calls_enabled` **method object** --
                the bridge takes it as a callable and calls it later, so
                this accessor must not call it here.
            current_rail_conversation_id: Returns the conversation id the
                rail is currently showing, or `None`.
            current_rail_state_accessor: `ChatScreen._current_console_rail_
                state`, for the Agent section's own open/collapsed state.
            chat_controller_accessor: Reads the screen's
                `_console_chat_controller` attribute (`getattr`-safe: the
                fleet line is reachable on a screen that has never built
                one).
            sync_native_console_chat_ui_accessor: Returns the screen's
                `_sync_native_console_chat_ui` **method object**, handed
                straight to `run_worker` by the drill-in toggle.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._chat_store_accessor = chat_store_accessor
        self._provider_gateway_accessor = provider_gateway_accessor
        self._native_tool_calls_enabled_accessor = native_tool_calls_enabled_accessor
        self._current_rail_conversation_id = current_rail_conversation_id
        self._current_rail_state_accessor = current_rail_state_accessor
        self._chat_controller_accessor = chat_controller_accessor
        self._sync_native_console_chat_ui_accessor = (
            sync_native_console_chat_ui_accessor
        )

        # -- Moved state (was `ChatScreen.__init__`) ------------------------
        #: The lazily-built `ConsoleAgentBridge`, or `None` once resolved to
        #: "no agent runtime". Read-write through the screen's proxy of the
        #: same name (`Tests/UI/test_console_agent_rail.py` replaces it on
        #: the screen 10 times).
        self._console_agent_bridge: Any | None = None
        #: The sub-agent run currently drilled into, and the conversation
        #: that drill-in is scoped to. Both proxied read-write on the screen.
        self._console_agent_drilldown_run_id: str | None = None
        self._console_agent_drilldown_conversation_id: str | None = None
        #: `_console_agent_full_log_available`'s probe cache -- the resolved
        #: target run id and the answer for it. No reader outside this
        #: cluster, so no screen proxy.
        self._console_agent_full_log_cache_run_id: str | None = None
        self._console_agent_full_log_cache_available: bool = False
        #: The batched `[N Sub-Agents]` badge-count cache and its two
        #: invalidation keys. Also cluster-private.
        self._console_subagent_counts_cache: Dict[str, int] = {}
        self._console_subagent_counts_cache_row_ids: frozenset = frozenset()
        self._console_subagent_counts_cache_at: float = 0.0
        #: TASK-915: sticky suppression of the fleet force-open for the rest
        #: of THIS busy window. Written by the screen's own `_toggle_console_
        #: rail_section`, so it is proxied read-write there.
        self._agent_section_user_dismissed_while_busy = False

    # -- Framework services (live-read via `@property`) --------------------

    @property
    def run_worker(self) -> Any:
        """`Screen.run_worker`, bound. See `__init__`'s docstring for why
        this is a property rather than a value snapshotted once."""
        return self._screen.run_worker

    @property
    def push_screen(self) -> Any:
        """`Screen.app.push_screen`, bound. See `__init__`'s docstring."""
        return self._screen.app.push_screen

    @property
    def call_from_thread(self) -> Any:
        """`Screen.app.call_from_thread`, bound. See `__init__`'s docstring.

        Used by `_load_console_agent_run_log`, which runs on a real thread
        and must hop back to the UI thread to push the modal.
        """
        return self._screen.app.call_from_thread

    # -- Named constructor dependencies -------------------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method/attribute used --
    # which is what lets the method bodies further down be byte-for-byte
    # copies of the pre-extraction source.

    @property
    def _ensure_console_chat_store(self) -> Any:
        """`ChatScreen._ensure_console_chat_store`, reached by name."""
        return self._chat_store_accessor

    @property
    def _ensure_console_provider_gateway(self) -> Any:
        """`ChatScreen._ensure_console_provider_gateway`, reached by name."""
        return self._provider_gateway_accessor

    @property
    def _console_native_tool_calls_enabled(self) -> Any:
        """The screen's `_console_native_tool_calls_enabled` method object.

        Handed to `ConsoleAgentBridge` as a callable it stores and calls
        later, so this returns the method rather than its result -- the
        same "bare-attribute READ, not a call" shape
        `ConsolePromptsController` documents for
        `open_console_provider_recovery_accessor`.
        """
        return self._native_tool_calls_enabled_accessor()

    @property
    def _current_console_rail_conversation_id(self) -> Any:
        """`ChatScreen._current_console_rail_conversation_id`, by name."""
        return self._current_rail_conversation_id

    @property
    def _current_console_rail_state(self) -> Any:
        """`ChatScreen._current_console_rail_state`, by name."""
        return self._current_rail_state_accessor

    @property
    def _console_chat_controller(self) -> Any:
        """The screen's current `ConsoleChatController`, or `None`."""
        return self._chat_controller_accessor()

    @property
    def _sync_native_console_chat_ui(self) -> Any:
        """The screen's `_sync_native_console_chat_ui` method object.

        Handed straight to `run_worker` (not called here) by the drill-in
        toggle, exactly as the pre-move body did.
        """
        return self._sync_native_console_chat_ui_accessor()

    # -- Moved methods -------------------------------------------------------

    def _ensure_console_agent_bridge(self) -> Any:
        """Return the native Console agent bridge, creating it lazily.

        Returns ``None`` (no agent runtime) when there is no durable
        ChaChaNotes DB to key the sibling ``AgentRunsDB`` file off of (e.g. an
        in-memory test harness) -- callers use the provider-direct Console
        stream in that case regardless of the config gate.
        """
        if self._console_agent_bridge is not None:
            return self._console_agent_bridge
        db = getattr(self.app_instance, "chachanotes_db", None)
        db_path = getattr(db, "db_path", None) if db is not None else None
        if not db_path or str(db_path) == ":memory:":
            self._console_agent_bridge = None
            return None
        from pathlib import Path

        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
        # TASK-1971 (Agent Change Review): the tracker is None when git is
        # absent -- the bridge then skips tracking entirely, and runs behave
        # exactly as before the feature existed (spec gating decision).
        from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

        change_tracker = ChangeTurnTracker()
        self._console_agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=self._ensure_console_chat_store(),
            provider_gateway=self._ensure_console_provider_gateway(),
            skills_service=getattr(self.app_instance, "skills_scope_service", None),
            native_tools_enabled=self._console_native_tool_calls_enabled,
            change_tracker=change_tracker if change_tracker.available else None,
        )
        return self._console_agent_bridge

    def _console_agent_section_lines(self) -> tuple[str, str, str]:
        """Return the Agent rail's (status, steps, sub-agents) line text.

        Reads the live in-memory run snapshot (or, when drilled into one
        sub-agent, that run's durable record) via the Console agent bridge --
        the same bridge whose ``AgentRunsDB`` backs resume re-derivation, so
        this always reflects the latest known state without any extra event
        plumbing (the 0.2s Console poll re-calls this on every tick).

        Finding B: none of this text is escaped -- every string returned
        here is rendered into a ``markup=False`` Static (see the compose
        block below), so escaping would be a second guard stacked on top
        of ``markup=False`` and would render literal backslashes (e.g.
        ``fetch [docs]`` -> ``fetch \\[docs]``). Contrast with the
        conversation-browser badge label (``format_console_conversation_
        row_label``), which renders through ``Text.from_markup`` and must
        stay escaped.

        Finding C: a drill-in is scoped to the conversation active when
        the user drilled in. Every call here re-checks that scope --
        catching any switch path that doesn't itself clear the drill-down
        -- and falls back to the overview on a mismatch rather than show
        a foreign conversation's sub-agent detail.

        Gate Finding 2 (agent-runtime live gate): the top-level overview
        line used to read only ``bridge.live_snapshot`` -- an in-memory,
        per-process cache that starts empty every new bridge instance, so
        it showed "Agent: idle" for a resumed conversation right after an
        app restart even though the drill-in and the conversation-row
        badge both correctly re-derived from ``AgentRunsDB``. An idle live
        snapshot now falls back to ``bridge.historical_snapshot`` (cached
        by the bridge itself, so this does not add a DB hit per 0.2s poll
        tick) -- a live/in-process run always reports non-"idle" and keeps
        precedence over the fallback.
        """
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        if bridge is None:
            return ("Agent: unavailable", "", "")
        if conversation_id != self._console_agent_drilldown_conversation_id:
            # The active conversation/session changed since the drill-in
            # (tab switch, Ctrl+K switcher, saved-conversation resume,
            # workspace switch, ...) -- this self-heals even for a switch
            # path that doesn't explicitly clear the drill-down itself.
            self._console_agent_drilldown_run_id = None
            self._console_agent_drilldown_conversation_id = conversation_id
        drill = self._console_agent_drilldown_run_id
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None and record.get("conversation_id") == conversation_id:
                # Finding A (review round 2): this used to slice the raw
                # `summary`/`result` field to a hardcoded 80 characters,
                # bypassing `_summarize_persisted_step` entirely -- so a
                # drilled-in sub-agent's step text neither respected the
                # user-configurable display cap (TASK-870) nor got the
                # word-boundary truncation affordance every other render
                # path shares. Route through the same helper the top-level
                # overview's persisted/resumed steps already use (see
                # `ConsoleAgentBridge._derive_historical_snapshot`).
                from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

                steps = "\n".join(
                    f"{s.get('kind')}: "
                    f"{ConsoleAgentBridge._summarize_persisted_step(s)}"
                    for s in record.get("steps", [])
                )
                return (
                    f"Sub-agent · {record.get('status')} (Back)",
                    steps,
                    str(record.get("task") or ""),
                )
            # The drilled-into run vanished, or (defensive re-check) its
            # recorded conversation_id no longer matches the one now
            # active -- fall back to the live snapshot instead of showing
            # a stale/foreign drill-in view.
            self._console_agent_drilldown_run_id = None
        snapshot = bridge.live_snapshot(conversation_id)
        if snapshot.status == "idle":
            # Finding 2 (Plan-B agent-runtime gate): this bridge instance
            # has never run this conversation in-process -- most likely a
            # resumed conversation right after an app restart, since
            # ``live_snapshot`` is an in-memory-only, per-process cache
            # that starts empty every new instance. Fall back to
            # AgentRunsDB so the summary reflects history immediately
            # instead of showing "Agent: idle" until the next live run.
            # A live run already in progress/finished in this process
            # always reports a non-"idle" status above and keeps
            # precedence -- this fallback is only ever consulted when
            # there is nothing live to show. ``getattr`` tolerates a bare
            # test double that only implements ``live_snapshot``.
            historical = getattr(bridge, "historical_snapshot", None)
            if historical is not None:
                snapshot = historical(conversation_id)
        status = f"Agent: {snapshot.status}"
        if snapshot.status == "running":
            status = f"Agent: running · step {snapshot.step}"
        # Finding A (review round 2): `s.text` is already truncated to the
        # user-configurable cap by whichever bridge helper built this
        # snapshot (`_summarize` for a live step, `_summarize_persisted_
        # step` for a resumed/historical one) -- re-slicing to a hardcoded
        # 80 here silently overrode any configured value above 80 with no
        # visible effect, defeating the whole point of that setting.
        steps = "\n".join(f"· {s.text}" for s in snapshot.steps)
        glyphs = {
            "done": "✓",
            "running": "●",
            "stuck": "⚠",
            "error": "✗",
            "cancelled": "✗",
        }
        subagents = "\n".join(
            f"{glyphs.get(s.status, '●')} {s.text[:60]}" for s in snapshot.subagents
        )
        return (status, steps, subagents)

    def _console_agent_fleet_summary_line(self) -> str:
        """Return the Agent rail's fleet summary line (parallel-agents spec §6).

        Sourced from ``ConsoleChatController.fleet_summary_counts`` (other
        running / other pending-approval sessions, relative to the active
        one). Copy is VERBATIM per spec §6 -- no singular/plural grammar
        handling, so ``"1 other agents running, ..."`` is intentional, not a
        bug. Returns ``""`` when both counts are zero; the caller hides the
        fleet Static in that case (absent, not present-but-blank) so it
        never crowds the rail with an empty line.
        """
        controller = getattr(self, "_console_chat_controller", None)
        if controller is None:
            return ""
        running, pending = controller.fleet_summary_counts()
        if running + pending <= 0:
            return ""
        return f"{running} other agents running, {pending} waiting for approval."

    def _console_agent_full_log_run_id(self) -> str | None:
        """Return the run id the "View full log" affordance should target.

        TASK-870: mirrors ``_console_agent_section_lines``'s own
        drill-vs-overview precedence -- the drilled-into sub-agent run
        (when drilled in and still valid for the active conversation, the
        same check that method uses), else the conversation's latest
        primary run, which is what the top-level overview is summarizing.

        Returns:
            The relevant run id, or ``None`` when there is nothing to
            target -- no bridge, no active conversation, a stale drill-in
            left over from a conversation switch, or a conversation that
            has never run an agent. Callers must still confirm
            ``ConsoleAgentBridge.run_log_available`` before showing the
            affordance for whatever id this returns -- a valid run id does
            not imply a log was ever written for it.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return None
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return None
        drill = self._console_agent_drilldown_run_id
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None and record.get("conversation_id") == conversation_id:
                return drill
            return None
        # getattr tolerates a bare test double that only implements the
        # older bridge surface (subagent_run/subagent_runs/live_snapshot) --
        # same idiom _console_agent_section_lines already uses for
        # historical_snapshot, immediately below this method in this file.
        latest_primary_run_id = getattr(bridge, "latest_primary_run_id", None)
        if latest_primary_run_id is None:
            return None
        return latest_primary_run_id(conversation_id)

    def _console_agent_full_log_available(self, *, allow_probe: bool = True) -> bool:
        """Whether the "View full log" affordance should be shown right now.

        TASK-870 (AC#6/#7): ``True`` only when ``_console_agent_full_log_
        run_id`` resolves to a run AND that run actually has an on-disk log
        -- absent (button hidden) for every other case, including a bridge
        or filesystem lookup that raises, so a resolution failure can never
        surface as a dangling or erroring button.

        Finding D (review round 2): the underlying check costs a SQLite
        lookup (``_console_agent_full_log_run_id``, to resolve the target
        run id) plus a filesystem probe (``bridge.run_log_available``, to
        confirm a log directory/segment exists -- for a drilled-in
        sub-agent this can also mean parsing its primary's whole log, see
        finding B) -- paying that unconditionally on every 0.2s rail tick
        is real, avoidable I/O for a value that is overwhelmingly the same
        tick to tick. The filesystem/DB probe result is cached keyed by
        the resolved run id and only redone when that id changes; when
        ``allow_probe`` is ``False`` (the periodic sync passes this while
        the Agent section is collapsed -- see
        ``_sync_console_agent_section``), this returns the last cached
        answer WITHOUT even resolving the current run id, so a collapsed
        section's steady-state tick touches neither disk nor the DB.

        Args:
            allow_probe: Whether a cache miss may fall through to the
                SQLite/filesystem lookup. Callers that need a fresh,
                authoritative answer (the one-shot compose-time render,
                and the "open the viewer" press-time re-check) should
                leave this ``True``; the periodic rail sync passes
                ``section_open`` here.

        Returns:
            Whether the affordance should be visible for the current
            target run -- possibly a stale cached value when
            ``allow_probe`` is ``False`` and the target has since changed
            unobserved (self-corrects the moment the section reopens).
        """
        if not allow_probe:
            return self._console_agent_full_log_cache_available
        run_id = self._console_agent_full_log_run_id()
        if run_id == self._console_agent_full_log_cache_run_id:
            return self._console_agent_full_log_cache_available
        if not run_id:
            available = False
        else:
            bridge = self._ensure_console_agent_bridge()
            if bridge is None:
                available = False
            else:
                try:
                    available = bool(bridge.run_log_available(run_id))
                except Exception:
                    logger.opt(exception=True).warning(
                        "console agent rail: run_log_available check failed for "
                        f"run_id={run_id}; hiding the View full log affordance"
                    )
                    available = False
        self._console_agent_full_log_cache_run_id = run_id
        self._console_agent_full_log_cache_available = available
        return available

    def _open_console_agent_run_log_viewer(self) -> None:
        """Kick off loading the full run log for whatever "View full log" targets.

        TASK-870 (AC#6): re-resolves the target run id at press time
        (rather than trusting a value cached from the last 0.2s sync) so a
        drill-in change between sync ticks can never open the wrong run's
        log. No-ops quietly if there is no current target.

        Finding C (review round 2): the actual filesystem read + record
        parse + formatting now happens off the UI thread (see
        ``_load_console_agent_run_log``) -- a run's segments can total
        many megabytes (4MB per segment, no cap on segment count), and
        doing that synchronously on the Textual event loop could freeze
        the whole app for the duration of the read.
        """
        run_id = self._console_agent_full_log_run_id()
        if not run_id:
            return
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return
        # Was `self._load_console_agent_run_log(bridge, run_id)` while
        # that method carried `@work(thread=True)`. Same worker, same
        # group ("default") and name -- see the loader's own docstring.
        self.run_worker(
            partial(self._load_console_agent_run_log, bridge, run_id),
            thread=True,
            group="default",
            name="_load_console_agent_run_log",
        )

    def _load_console_agent_run_log(self, bridge: Any, run_id: str) -> None:
        """Load, filter, and format one run's full log off the UI thread.

        Finding C: the worker half of ``_open_console_agent_run_log_
        viewer`` -- everything here is filesystem/CPU work (no widget
        access), so it is safe to run in a real thread. The modal is only
        ever pushed back on the UI thread, via ``call_from_thread``.

        Wave-4 task 3: this carried ``@work(thread=True)`` before the move.
        Textual's decorator asserts its target is a ``DOMNode`` and a
        controller is not one, so the caller now dispatches it through
        ``run_worker`` with the same thread/group/name ``@work`` supplied.

        Args:
            bridge: The already-resolved Console agent bridge (resolved on
                the UI thread by the caller -- lazy bridge construction
                touches ``self.app_instance``/config and should not run
                off-thread).
            run_id: The run id to load, as resolved by the caller at press
                time.
        """
        try:
            if not bridge.run_log_available(run_id):
                return
            log_text = bridge.load_run_log_text(run_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"console agent rail: failed to load run log for run_id={run_id}"
            )
            return
        if not log_text:
            return
        self.call_from_thread(
            self._show_console_agent_run_log_modal, run_id, log_text
        )

    def _show_console_agent_run_log_modal(self, run_id: str, log_text: str) -> None:
        """Push the full-log modal. UI-thread only -- see ``_load_console_agent_run_log``.

        Args:
            run_id: The run id the loaded text belongs to.
            log_text: The fully rendered, untruncated log text.
        """
        self.push_screen(ConsoleRunLogModal(run_id=run_id, log_text=log_text))

    def _console_agent_section_payload(
        self,
    ) -> tuple[str, str, str, str, bool, bool, bool]:
        """Derive everything the mounted Agent rail should be showing.

        TASK-251: equality-guarded against the last successfully-applied
        payload -- the 0.2s tick called this unconditionally, forcing three
        ``Static.update()`` calls plus a style write per tick even when
        nothing agent-related had changed.

        Fix round 2 (parallel-agents spec §6 live-smoke finding): also
        tracks and applies the Agent section's own open/collapsed state
        (header chevron + body ``display``). Compose-time already applies
        ``_apply_fleet_agent_section_auto_open`` once at mount, but that is
        a one-shot snapshot -- a background session's run starting or
        ending *after* mount (the overwhelmingly common case) must reopen
        or release the section on this same periodic sync, or the fleet
        line would only ever be reachable for whichever fleet state
        happened to exist at the moment the screen was first composed.

        TASK-870: also tracks the "View full log" affordance's visibility
        (``_console_agent_full_log_available``) -- present only while a run
        log actually exists for whatever run the section is currently
        showing (AC#6/#7).

        Wave-4 task 3: this is the non-DOM half of the screen's
        ``_sync_console_agent_section``, which kept the equality guard
        itself (``_console_agent_section_last``, compared against what this
        returns) and the nine ``query_one`` writes -- controllers own no
        DOM. Every paragraph above still describes what is derived here;
        only the "applies"/"tracks" half of each is now the screen's. The
        ordering constraint the third paragraph records -- probe the
        full-log affordance only once ``section_open`` is known -- lives
        here too, which is why the derivation moved as one piece rather
        than as seven accessors the screen would have had to sequence.

        Returns:
            ``(status_line, steps_text, subagents_text, fleet_line,
            back_visible, section_open, full_log_visible)`` -- the exact
            tuple the screen compares against its last applied payload.
        """
        status_line, steps_text, subagents_text = self._console_agent_section_lines()
        fleet_line = self._console_agent_fleet_summary_line()
        back_visible = bool(self._console_agent_drilldown_run_id)
        try:
            section_open = self._current_console_rail_state().agent_open
        except (AttributeError, NoActiveAppError):
            # A bare/unmounted screen (several tests construct
            # `ChatScreen(app)` directly with no active Textual app
            # context -- e.g. `test_console_provider_selection_carries_
            # active_session_system_prompt`) has no real rail width to
            # derive responsive state from: `_console_rail_available_
            # columns` reads `self.size`, which raises here rather than
            # returning `None` (`Screen.size` needs `self.app`). Same
            # guard idiom `_provider_readiness_app_config` already uses
            # for this exact failure mode. Fall back to the persisted
            # default (collapsed) -- the Statics-only updates below are
            # still worth applying even when the section's own open state
            # cannot be derived.
            section_open = False
        # Finding D: never probe disk/DB for the full-log affordance while
        # the section is collapsed -- `section_open` must be known first.
        full_log_visible = self._console_agent_full_log_available(
            allow_probe=section_open
        )
        return (
            status_line,
            steps_text,
            subagents_text,
            fleet_line,
            back_visible,
            section_open,
            full_log_visible,
        )

    def _toggle_console_agent_drilldown_from_subagents_click(self) -> None:
        """Step the drill-in through this conversation's sub-agent runs.

        Finding D: a conversation can have more than one sub-agent run,
        but the combined ``subagents`` rail line only ever opened
        ``runs[0]`` no matter how many times it was clicked, leaving every
        other sub-agent unreachable. Repeated clicks now cycle through
        ``runs[0], runs[1], ..., runs[n-1]`` (newest first, matching
        ``AgentRunsDB.list_runs``' order) and then back to the overview,
        rather than adding a new per-row widget for what is usually a
        small N. The dedicated Back button always returns to the overview
        directly, regardless of where the cycle currently is.
        """
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        runs = bridge.subagent_runs(conversation_id) if bridge is not None else []
        run_ids = [run.get("id") for run in runs]
        current = self._console_agent_drilldown_run_id
        if not run_ids:
            next_run_id = None
        elif current in run_ids:
            next_index = run_ids.index(current) + 1
            next_run_id = run_ids[next_index] if next_index < len(run_ids) else None
        else:
            next_run_id = run_ids[0]
        self._console_agent_drilldown_run_id = next_run_id
        self._console_agent_drilldown_conversation_id = conversation_id
        self.run_worker(
            self._sync_native_console_chat_ui,
            exclusive=True,
            group="console-sync",
        )

    def _apply_fleet_agent_section_auto_open(
        self, rail_state: ConsoleRailState
    ) -> ConsoleRailState:
        """Force the Agent rail section open while the fleet has anything to report.

        Parallel-agents spec §6, fix round 2 (live-smoke finding): the Agent
        section's persisted preference defaults collapsed (``agent_open=
        False``, see ``ConsoleRailPreferences``) and nothing previously
        reopened it, so its BODY -- the status/step/sub-agent detail for
        the viewed session's own run -- stayed ``display: none`` even
        while another session was running or parked on an approval.
        Scrolling the rail only ever reached the still-collapsed header;
        that detail was unreachable regardless of scroll position.

        TASK-1140 (UAT F1, fix round 1): ``#console-agent-fleet-summary``
        itself no longer lives inside this section's body -- it is now a
        pinned, non-scrolling sibling of the rail's own header (see the
        compose block a few hundred lines up), painted unconditionally
        whenever the fleet has anything to report, independent of this
        section's open/collapsed state. This method's force-open still
        matters for the Agent section's OWN contextual detail (status/
        steps/sub-agents for whichever conversation is viewed), just not
        for fleet-line visibility anymore.

        Mirrors ``_apply_pending_launch_inspector_auto_open``: an ephemeral
        override applied to the RENDERED rail state only, never written back
        to the persisted preference, so a user's own explicit collapse still
        takes effect the moment the fleet goes quiet (``fleet_summary_
        counts()`` returns to ``(0, 0)``). Uses ``_console_agent_fleet_
        summary_line()`` -- the exact same non-empty-string signal the
        pinned fleet Static's own ``display`` toggles on -- so "must render
        the section open" and "has a line to show" can never disagree.

        TASK-915: fix round 3 (live-smoke finding). Round 2's force was a
        one-shot-per-rendered-state override, but the 0.2s sync tick
        recomputes it every time the agent-section payload changes (e.g. a
        second background run starting/finishing) -- so a manual collapse
        while the fleet was busy held only until the NEXT such change, then
        got silently re-forced open. `_toggle_console_rail_section` now sets
        `_agent_section_user_dismissed_while_busy` when the user closes this
        section with a non-empty fleet line; honoured here so the force
        stays suppressed for the rest of THIS busy window. Still never
        touches the persisted preference -- only the transient flag and the
        returned dataclass change.
        """
        fleet_line = self._console_agent_fleet_summary_line()
        if not fleet_line:
            # Fleet is quiet: release any sticky dismissal so the NEXT busy
            # window auto-opens again (TASK-915 AC2).
            self._agent_section_user_dismissed_while_busy = False
            return rail_state
        if rail_state.agent_open:
            return rail_state
        if self._agent_section_user_dismissed_while_busy:
            return rail_state
        return replace(rail_state, agent_open=True)

    def _console_subagent_counts_refresh_needed(self, row_ids: frozenset) -> bool:
        """Decide whether the sub-agent badge-count cache needs a DB round trip.

        Finding A: refreshing on every 0.2s poll tick would re-issue the
        batched count query up to 5x/second even when nothing sub-agent
        related changed. This gates the refresh to three cheap-to-check
        conditions instead of refreshing unconditionally:

        1. The visible conversation row set changed (a rebuild) -- new
           rows may need counts we have never cached.
        2. A run is actively streaming/validating/retrying for this
           screen -- a just-spawned sub-agent's count should show up
           promptly rather than wait out the full TTL.
        3. The cache has aged past ``CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS``
           -- a fallback bound covering counts that changed from a
           different Console session/tab or a resumed run, where neither
           of the above two signals fires on this screen.

        Args:
            row_ids: The conversation ids of the currently visible browser
                rows (deduplicated, blanks excluded).

        Returns:
            ``True`` when the cache should be rebuilt from the DB.
        """
        # Imported in-body, not at module scope: both constants still
        # live in `chat_screen.py`, which imports this module's package at
        # import time and defines them well after that -- see this module's
        # docstring.
        from ..Screens.chat_screen import (
            CONSOLE_ACTIVE_RUN_STATUSES,
            CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS,
        )

        if row_ids != self._console_subagent_counts_cache_row_ids:
            return True
        controller = self._console_chat_controller
        if (
            controller is not None
            and controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
        ):
            return True
        age = time.monotonic() - self._console_subagent_counts_cache_at
        return age >= CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS

    def _console_subagent_counts_for_rows(
        self,
        bridge: Any | None,
        rows: Iterable[Any],
    ) -> Dict[str, int]:
        """Return ``conversation_id -> sub-agent count`` for browser rows.

        Finding A: previously called ``bridge.subagent_count(cid)`` once
        per row (a fresh sqlite connection per call) on every poll tick --
        up to ~75 queries/tick. Replaced with one batched
        ``bridge.subagent_counts(...)`` call, gated by
        ``_console_subagent_counts_refresh_needed`` so it isn't reissued
        unconditionally every tick either.

        Args:
            bridge: The Console agent bridge, or ``None`` when the agent
                runtime is unavailable (e.g. in-memory test harness).
            rows: The conversation-browser input rows currently visible.

        Returns:
            Mapping of ``conversation_id -> count``; conversations with
            zero sub-agent runs are simply absent (see
            ``AgentRunsDB.count_subagents_by_conversation``).
        """
        if bridge is None:
            return {}
        row_ids = frozenset(
            cid for row in rows if (cid := getattr(row, "conversation_id", None))
        )
        if self._console_subagent_counts_refresh_needed(row_ids):
            self._console_subagent_counts_cache = (
                bridge.subagent_counts(list(row_ids)) if row_ids else {}
            )
            self._console_subagent_counts_cache_row_ids = row_ids
            self._console_subagent_counts_cache_at = time.monotonic()
        return self._console_subagent_counts_cache

    def _inject_resume_agent_markers(
        self,
        messages: list[ConsoleChatMessage],
        conversation_id: str,
    ) -> list[ConsoleChatMessage]:
        """Re-derive and interleave TOOL markers from ``AgentRunsDB`` on resume.

        Plan-B final-review Medium-1: the rail already re-derives from
        ``AgentRunsDB`` on resume (``_console_agent_section_lines`` ->
        ``bridge.historical_snapshot``, and the ``[N Sub-Agents]`` badge);
        the inline transcript TOOL markers did not, since
        ``_console_messages_from_conversation_tree`` only ever reads
        persisted ChaChaNotes rows, where markers never land
        (``ConsoleAgentBridge._append_marker`` uses ``persist=False`` so
        agent activity survives a restart without being written into the
        conversation itself).

        Task 3: ``resume_marker_messages`` now pairs each run's block with
        the ``assistant_message_id`` of the reply it produced, and
        ``inject_resume_agent_markers`` anchors placement to that id
        against ``messages``'s own ``persisted_message_id``s -- a run
        whose reply isn't on the active path (edited/regenerated onto
        another branch) has its block hidden rather than misattributed to
        a different reply; a legacy/null-id run keeps the prior ordinal
        placement. ``messages`` is the caller's already-active-path
        transcript (``_console_messages_from_conversation_tree`` walks the
        active thread only), so this composes correctly with branching
        without any extra filtering here. See ``inject_resume_agent_
        markers`` for the full placement/idempotency contract, and
        ``resume_marker_messages`` for how each run's marker block and
        anchor id are derived.

        Returns ``messages`` unchanged when there is no durable agent
        bridge available (e.g. an in-memory test harness, matching
        ``_ensure_console_agent_bridge``'s own fallback).
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return messages
        from tldw_chatbook.Chat.console_agent_bridge import inject_resume_agent_markers

        # bridge.resume_marker_messages returns the (anchor_id, block) pairs
        # inject_resume_agent_markers now expects directly -- no reshaping
        # needed here, just passed straight through.
        return inject_resume_agent_markers(
            messages, bridge.resume_marker_messages(conversation_id)
        )
