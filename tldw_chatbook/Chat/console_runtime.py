"""App-owned Console runtime holder (task-15860, headless wake — Task 1).

**This module changes WHO constructs the Console runtime objects, and
nothing else.** It is the "pure ownership move" the owner made a staging
condition for design A (`.superpowers/sdd/2026-08-14-headless-wake/
DECISIONS.md`, owner answer (3)): separately reviewable and separately
revertable from the semantics work that follows it.

## What it owns

One `ConsoleRuntime` per app owns the four objects `ChatScreen` used to
build for itself:

- the `ConsoleChatStore` (with its `ChatPersistenceService`),
- the `ConsoleProviderGateway`,
- the `ConsoleAgentBridge` and the sibling `AgentRunsDB` file it is keyed
  off, together with the `register_fleet_attention` fan-out registration
  that `FleetDrainFanout.register`'s contract requires to sit next to
  bridge construction (the wake coordinator's own registration travels
  inside `ConsoleChatController.__init__`, so it moves with the
  controller),
- the `ConsoleChatController`.

Each is built lazily, on first `ensure_*` call, from parameters the
calling view supplies — the same parameters, in the same order, the
screen's own `_ensure_*` methods passed before this module existed.

## What it deliberately does NOT change (Task 1's whole discipline)

- **Lifetime.** `ChatScreen.on_unmount` still records the fleet-teardown
  notice and still `await`s `controller.shutdown()`, and then the runtime
  is DISPOSED (`dispose_console_runtime`). A second Console visit
  therefore still gets a brand-new store/gateway/bridge/controller —
  exactly as today. Making the runtime *survive* teardown is Task 2, and
  `Tests/Chat/test_console_runtime_ownership.py` pins today's behaviour so
  that change lands as a visible diff to a test rather than a diff to
  nothing.
- **Hook binding.** The controller and store are still handed
  screen-bound callables (`on_scope_flushed`, the dictionary/world-info
  appliers, the wake probes, …). Task 0's P3 found all five slots a
  viewless wake turn touches still pointing at a DEAD screen; rebinding
  them is Task 4, not this task.
- **The `_shutdown_requested` gate** in `ConsoleFleetWakeCoordinator.
  _attempt` — Task 0's P2 proved it is the single line refusing a headless
  wake — is untouched here.

## Why an app attribute rather than a global

`app.console_runtime` follows `app.console_image_edit_operations`
(`app.py`, Phase 1 of `TldwCli.__init__`): constructed on the app, and
re-created lazily by the screen when a test's app object never had one
(`ChatScreen._h3_image_edit_registry` is the shape copied here). Screens
are never cached (`app.py` `_create_navigation_screen`), so anything that
must outlive a navigation cannot live on one.

It deliberately does NOT join `_shutdown_app_owned_lifecycles`, unlike the
image-edit registry: that hook runs *before* Textual closes screen state,
and `dispose()` is a reference drop with no durable work to settle, so
disposing there would only reorder Console's quit path for no gain. The
unmount that Textual performs at exit already disposes it. When Task 2
gives the runtime real cross-navigation lifetime, an exit-time settle
becomes worth adding — and it should be added deliberately, with the quit
ordering re-verified, not inherited from this task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

#: The app attribute this module's helpers read and write. Named once so a
#: test can assert on the protocol rather than on a string literal.
CONSOLE_RUNTIME_ATTR = "console_runtime"

__all__ = [
    "CONSOLE_RUNTIME_ATTR",
    "ConsoleRuntime",
    "dispose_console_runtime",
    "ensure_console_runtime",
]


class ConsoleRuntime:
    """The app-owned holder for one Console runtime.

    Every `ensure_*` method is idempotent: it builds its object on the
    first call and returns the cached instance afterwards, ignoring the
    parameters on subsequent calls. That is the same laziness
    `ChatScreen._ensure_*` had — the parameters were only ever read at
    construction there too.

    Attributes are exposed through non-constructing read-only properties
    so a caller can ask "has this been built yet?" without building it,
    which several `ChatScreen` call sites do (`store = self.
    _console_chat_store` followed by an `is None` early return).
    """

    def __init__(self, app: Any) -> None:
        """Bind the holder to one app object.

        Args:
            app: The `TldwCli` app (or, in tests, whatever object plays
                that role for the screen under test). Read for
                `chachanotes_db`, `citation_trace_repository`,
                `workspace_registry_service` and the
                `console_provider_gateway_factory` test seam — never
                mutated.
        """
        self._app = app
        self._chat_store: Any | None = None
        self._provider_gateway: Any | None = None
        self._agent_bridge: Any | None = None
        self._chat_controller: Any | None = None
        #: The view (a `ChatScreen`) that claimed this runtime, or `None`
        #: while unclaimed. **Task 1's lifetime-preservation device, and
        #: nothing more** -- it is NOT the attach/detach seam Task 2 builds.
        #:
        #: Two `ChatScreen`s are briefly alive at once whenever a navigation
        #: lands back on Console: `_complete_screen_navigation` constructs
        #: and `restore_state`s the incoming screen BEFORE `switch_screen`
        #: unmounts the outgoing one (`app.py`), and `restore_state` reaches
        #: `_ensure_console_chat_store`. Without a claim the incoming screen
        #: would adopt the outgoing screen's controller and then watch
        #: `on_unmount` shut it down underneath it. Today each screen builds
        #: its own runtime, so a claim by a different view means "build a
        #: fresh one" -- which is exactly what `ensure_console_runtime` does.
        self.view: Any | None = None
        #: Bumped by every `dispose()`. Task 1's new-runtime-per-visit pin
        #: reads it; Task 2 will make it stop moving on a navigation.
        self.generation: int = 0

    # -- non-constructing accessors ---------------------------------------

    @property
    def app(self) -> Any:
        """The app this runtime is bound to."""
        return self._app

    @property
    def chat_store(self) -> "ConsoleChatStore | None":
        """The built store, or `None` if nothing has asked for one yet."""
        return self._chat_store

    @property
    def provider_gateway(self) -> Any | None:
        """The built provider gateway, or `None`."""
        return self._provider_gateway

    @property
    def agent_bridge(self) -> Any | None:
        """The built agent bridge, or `None` (also `None` when resolved to
        "no agent runtime" — see `ensure_agent_bridge`)."""
        return self._agent_bridge

    @property
    def chat_controller(self) -> "ConsoleChatController | None":
        """The built chat controller, or `None`."""
        return self._chat_controller

    # -- construction ------------------------------------------------------

    def ensure_chat_store(
        self,
        *,
        workspace_context: Any | None = None,
        on_scope_flushed: Callable[..., Any] | None = None,
    ) -> "ConsoleChatStore":
        """Return the Console chat store, creating it lazily.

        Moved verbatim from `ChatScreen._ensure_console_chat_store`: the
        durable `ChatPersistenceService` is attached only when the app has
        a ChaChaNotes DB, and the citation repository is dropped when it
        belongs to a different DB than the one being persisted to.

        Args:
            workspace_context: The view's current
                `ConsoleWorkspaceContext`. Read at construction only.
            on_scope_flushed: The view's session-scope-flushed callback.
                Read at construction only.

        Returns:
            ConsoleChatStore: The runtime's store.
        """
        if self._chat_store is not None:
            return self._chat_store
        from tldw_chatbook.Chat.chat_persistence_service import (
            ChatPersistenceService,
        )
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

        persistence = None
        db = getattr(self._app, "chachanotes_db", None)
        if db is not None:
            citation_repository = getattr(
                self._app,
                "citation_trace_repository",
                None,
            )
            if (
                citation_repository is not None
                and getattr(citation_repository, "db", None) is not db
            ):
                citation_repository = None
            persistence = ChatPersistenceService(
                db,
                workspace_registry=getattr(
                    self._app,
                    "workspace_registry_service",
                    None,
                ),
                citation_repository=citation_repository,
            )
        self._chat_store = ConsoleChatStore(
            persistence=persistence,
            workspace_context=workspace_context,
            on_scope_flushed=on_scope_flushed,
        )
        return self._chat_store

    def ensure_provider_gateway(
        self,
        *,
        config_provider: Callable[[], Any] | None = None,
    ) -> Any:
        """Return the Console provider gateway, creating it lazily.

        Moved verbatim from `ChatScreen._ensure_console_provider_gateway`,
        including the `console_provider_gateway_factory` test-injection
        seam read off the app.

        Args:
            config_provider: Fresh-config source handed to a
                real gateway; the gateway re-resolves readiness at send
                time and must see Settings saves made after boot. Ignored
                when the app supplies a factory.

        Returns:
            Any: The runtime's provider gateway.
        """
        if self._provider_gateway is not None:
            return self._provider_gateway
        factory = getattr(self._app, "console_provider_gateway_factory", None)
        if callable(factory):
            self._provider_gateway = factory()
        else:
            from tldw_chatbook.Chat.console_provider_gateway import (
                ConsoleProviderGateway,
            )

            self._provider_gateway = ConsoleProviderGateway(
                config_provider=config_provider,
            )
        return self._provider_gateway

    def ensure_agent_bridge(
        self,
        *,
        store_factory: Callable[[], Any],
        provider_gateway_factory: Callable[[], Any],
        skills_service: Any | None = None,
        native_tools_enabled_factory: Callable[[], Any] | None = None,
    ) -> Any:
        """Return the Console agent bridge, creating it lazily.

        Moved verbatim from
        `ConsoleAgentController._ensure_console_agent_bridge`, ordering
        included: the durable-DB probe runs FIRST and returns `None` (no
        agent runtime) before the store or the gateway is touched, so an
        in-memory harness still builds neither.

        Every screen-supplied dependency arrives as a callable, precisely
        to preserve that ordering — nothing on the view is read until the
        probe has passed.

        Args:
            store_factory: Returns the chat store the bridge should use.
                Called only past the durable-DB probe.
            provider_gateway_factory: Returns the provider gateway. Same.
            skills_service: The app's skills scope service, or `None`. A
                plain value: it is a `getattr` on the APP, which the probe
                has already touched.
            native_tools_enabled_factory: Returns the callable the bridge
                stores and calls later to read the `[console]
                native_tool_calls` gate. A factory, not that callable, so
                the view is not read on the no-agent-runtime path.

        Returns:
            Any: The `ConsoleAgentBridge`, or `None` when there is no
            durable ChaChaNotes DB to key the sibling `AgentRunsDB` off.
        """
        if self._agent_bridge is not None:
            return self._agent_bridge
        db = getattr(self._app, "chachanotes_db", None)
        db_path = getattr(db, "db_path", None) if db is not None else None
        if not db_path or str(db_path) == ":memory:":
            self._agent_bridge = None
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
        self._agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=store_factory(),
            provider_gateway=provider_gateway_factory(),
            skills_service=skills_service,
            native_tools_enabled=(
                native_tools_enabled_factory()
                if native_tools_enabled_factory is not None
                else None
            ),
            change_tracker=change_tracker if change_tracker.available else None,
        )
        # PR3a-2 Task 4: the survivor-completion attention consumer (durable
        # unseen mark + app-wide toast + deep link), registered NEXT TO
        # bridge construction per `FleetDrainFanout.register`'s contract.
        # Captures the APP object only -- never a screen -- because the
        # bridge (and its registered consumers) outlives the screen whenever
        # a survivor is still running at teardown.
        from tldw_chatbook.Chat.console_fleet_attention import (
            register_fleet_attention,
        )

        register_fleet_attention(self._agent_bridge, self._app)
        return self._agent_bridge

    def ensure_chat_controller(self, **kwargs: Any) -> "ConsoleChatController":
        """Return the Console chat controller, creating it lazily.

        The keyword arguments are `ConsoleChatController.__init__`'s, passed
        straight through from the view exactly as
        `ChatScreen._ensure_console_chat_controller` passed them. They are
        read at construction only; the screen keeps re-applying its
        selection state and its UI hooks after this returns, as before.

        Returns:
            ConsoleChatController: The runtime's controller.
        """
        if self._chat_controller is not None:
            return self._chat_controller
        from tldw_chatbook.Chat.console_chat_controller import (
            ConsoleChatController,
        )

        self._chat_controller = ConsoleChatController(**kwargs)
        return self._chat_controller

    # -- teardown ----------------------------------------------------------

    def dispose(self) -> None:
        """Drop every built object so the next `ensure_*` builds a fresh one.

        Reference-drop ONLY. It does not shut the controller down and does
        not close the gateway: `ChatScreen.on_unmount` still owns both of
        those steps in Task 1, in the order it always ran them
        (`fleet_teardown_split` snapshot -> `await controller.shutdown()`
        -> `await gateway.aclose()`), and this is called after them.
        """
        self._chat_controller = None
        self._agent_bridge = None
        self._provider_gateway = None
        self._chat_store = None
        self.view = None
        self.generation += 1


def _attach(app: Any, runtime: ConsoleRuntime | None) -> None:
    """Write `runtime` (or `None`) onto the app's runtime attribute."""
    if app is None:
        return
    try:
        setattr(app, CONSOLE_RUNTIME_ATTR, runtime)
    except Exception:  # noqa: BLE001 - a read-only app double is not an error
        logger.debug(
            "Console runtime: could not write the app's %s attribute.",
            CONSOLE_RUNTIME_ATTR,
        )


def ensure_console_runtime(app: Any, *, view: Any | None = None) -> ConsoleRuntime:
    """Return `app`'s Console runtime, creating and attaching one if needed.

    Mirrors `ChatScreen._h3_image_edit_registry`: the app normally builds
    this in `__init__`, but a test app object (or one whose runtime was
    disposed at the last unmount) has none, and the screen must still be
    able to run.

    Args:
        app: The app object to read/attach the runtime on. `None` is
            tolerated — a detached runtime is returned rather than raising,
            because the screen's own accessors are reached from bare
            `ChatScreen.__new__` fixtures.
        view: The `ChatScreen` asking. A runtime already claimed by a
            DIFFERENT view is replaced with a fresh one rather than shared,
            which is what keeps Task 1's per-screen lifetime exact — see
            `ConsoleRuntime.view`.

    Returns:
        ConsoleRuntime: The runtime `view` should use.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if isinstance(runtime, ConsoleRuntime) and (
        view is None or runtime.view is None or runtime.view is view
    ):
        if view is not None:
            runtime.view = view
        return runtime
    runtime = ConsoleRuntime(app)
    runtime.view = view
    _attach(app, runtime)
    return runtime


def dispose_console_runtime(app: Any, *, view: Any | None = None) -> None:
    """Dispose `app`'s Console runtime and detach it.

    Task 1 semantics: the next Console mount builds a brand-new runtime,
    exactly as every navigation built a brand-new store/controller before
    this module existed. **This call is what Task 2 removes.**

    Args:
        app: The app object holding the runtime. A missing or foreign
            attribute is a no-op.
        view: The `ChatScreen` unmounting. When the attached runtime was
            claimed by a different, still-live view (the overlapping-screens
            window described on `ConsoleRuntime.view`), this is a no-op:
            a dying screen must not tear down the runtime its successor is
            already using.
    """
    runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
    if not isinstance(runtime, ConsoleRuntime):
        return
    if view is not None and runtime.view is not None and runtime.view is not view:
        return
    runtime.dispose()
    _attach(app, None)
