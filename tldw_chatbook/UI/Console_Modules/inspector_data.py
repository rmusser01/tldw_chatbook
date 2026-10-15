"""Disposable Next Send factories for the current Console Inspector."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from textual.css.query import QueryError

from ...Chat.console_chat_models import console_pending_round_copy_for
from ...Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from ...Chat.console_display_state import ConsoleInspectorState
from ...Chat.console_runtime import ConsoleRuntime
from ...Widgets.Console.console_run_inspector import ConsoleRunInspector
from ...Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
)

if TYPE_CHECKING:
    from ...Chat.console_chat_models import ConsoleContextSnapshot
    from ...Widgets.Console.console_next_send_selection import (
        ConsoleNextSendSelectionResult,
    )


def build_next_send_factories(
    self, controller: Any, session_id: str
) -> tuple[
    Callable[[], Awaitable[ConsoleContextSnapshot | ConsoleNextSendSelectionResult]],
    Callable[[], int | None],
    int | None,
    bool,
]:
    """Build the Next Send tab's snapshot/estimate factories (task-18300).

    Shared by BOTH ``ConsoleConversationInspector`` entry points (the
    cost chip and Ctrl+Shift+P) -- the two push the SAME modal
    instance, and the user can switch to the Next Send tab regardless
    of which tab it opened on, so a caller that skipped this would
    leave the tab showing nothing. Building the closures themselves is
    cheap: the Context view calls ``snapshot_factory`` only when opened
    or explicitly refreshed. Usage and exchange views do not prepare
    hidden next-send content.

    ``session_id`` is threaded in rather than re-read from the store
    because the composer only reflects the ACTIVE session; see
    ``_captured_draft``.
    """
    from ...Widgets.Console.console_next_send_selection import (
        ConsoleNextSendSelectionResult,
    )
    from ..Screens import chat_screen as owner

    def _captured_draft() -> str:
        if controller.store.active_session_id == session_id:
            try:
                return self.query_one(
                    "#console-native-composer", owner.ConsoleComposerBar
                ).draft_text()
            except (owner.NoMatches, owner.QueryError):
                pass
        session = next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        )
        return session.draft if session is not None else ""

    def _captured_attachments() -> tuple[owner.MessageAttachment, ...]:
        pending = controller.store.pending_attachments(session_id)
        return tuple(
            owner.MessageAttachment(
                data=attachment.data,
                mime_type=attachment.mime_type or "image/png",
                display_name=attachment.display_name,
                position=index,
            )
            for index, attachment in enumerate(pending)
        )

    def _session_workspace() -> str | None:
        session = next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        )
        return session.workspace_id if session is not None else None

    async def _factory() -> ConsoleContextSnapshot | ConsoleNextSendSelectionResult:
        current_draft = _captured_draft()
        current_attachments = _captured_attachments()
        current_staged_sources = tuple(
            controller.store.workspace_context.allowed_sources
        )
        captured_turn_context = controller.resolve_turn_execution_context(session_id)
        canvas_current = controller.capture_personal_context_preview_guard(session_id)
        owner = (
            controller.store.active_session_id,
            _session_workspace(),
            controller.store.workspace_context.active_workspace_id,
            controller.store.conversation_context_epoch(session_id),
            controller._lifecycle_revision_for(session_id),
            captured_turn_context,
        )
        captured_selection: list[tuple[Any, Any, Any]] = []
        snapshot = await controller.build_context_snapshot(
            draft=current_draft,
            attachments=current_attachments,
            staged_sources=current_staged_sources,
            session_id=session_id,
            profile_selection_sink=lambda builder, request, explanation: (
                captured_selection.append((builder, request, explanation))
            ),
        )
        # The preview planner owns exactly one profile build. If that
        # contract changes, no callback can safely be paired with this
        # snapshot by guessing which one was final.
        if len(captured_selection) != 1:
            return snapshot

        def inputs_current() -> bool:
            try:
                return (
                    canvas_current()
                    and owner
                    == (
                        controller.store.active_session_id,
                        _session_workspace(),
                        controller.store.workspace_context.active_workspace_id,
                        controller.store.conversation_context_epoch(session_id),
                        controller._lifecycle_revision_for(session_id),
                        controller.resolve_turn_execution_context(session_id),
                    )
                    and current_draft == _captured_draft()
                    and current_attachments == _captured_attachments()
                    and current_staged_sources
                    == tuple(controller.store.workspace_context.allowed_sources)
                )
            except Exception:  # noqa: BLE001 - stale inspection fails closed.
                return False

        builder, request, explanation = captured_selection[0]

        async def selection_current() -> bool:
            if not inputs_current():
                return False
            if not await controller.personal_context_selection_current(
                builder,
                request,
                explanation,
                provider_selection=captured_turn_context.provider_selection,
            ):
                return False
            return inputs_current()

        return ConsoleNextSendSelectionResult(
            snapshot,
            explanation if inputs_current() else None,
            selection_current,
        )

    def _estimate_factory() -> int | None:
        return self._estimate_tokens({"draft": _captured_draft()})

    token_estimate = _estimate_factory()
    in_progress = controller.run_state.status in owner.CONSOLE_ACTIVE_RUN_STATUSES
    return _factory, _estimate_factory, token_estimate, in_progress


def _pending_method_metadata(function):
    return (
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
    )


def _pending_method_current(record):
    function, code, namespace, defaults, keywords, items, closure, cells = record
    return (
        function.__code__ is code
        and function.__globals__ is namespace
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is keywords
        and len(function.__kwdefaults__ or {}) == len(items)
        and all(
            (function.__kwdefaults__ or {}).get(key) is value for key, value in items
        )
        and function.__closure__ is closure
        and all(cell.cell_contents is value for cell, value in cells)
    )




def bind_pending_screen_sources(screen_class, torn_down):
    """Called at ChatScreen definition completion, before any first-use display."""
    global ChatScreen, _console_screen_is_torn_down
    global _CONSOLE_PENDING_SCREEN_READERS
    global _CONSOLE_PENDING_SCREEN_PROPERTIES
    global _CONSOLE_PENDING_RUNTIME_METHOD
    global _CONSOLE_PENDING_MODEL_READER
    global _CONSOLE_PENDING_PARTIAL_MODEL_READER
    global _CONSOLE_PENDING_FULL_INPUT_TYPE
    global _CONSOLE_PENDING_MODEL_TABLES_ORIGINAL
    global _CONSOLE_PENDING_STATE_CLASS_ORIGINAL
    global _CONSOLE_PENDING_MODEL_ABSENT_FIELD
    global _CONSOLE_PENDING_MODULES_ORIGINAL
    ChatScreen = screen_class
    _console_screen_is_torn_down = torn_down
    _CONSOLE_PENDING_SCREEN_READERS = tuple(
        (
            name,
            getattr(ChatScreen, name),
            _pending_method_metadata(getattr(ChatScreen, name)),
        )
        for name in ("_console_pending_approval_count", "_build_console_inspector_state")
    )
    _CONSOLE_PENDING_SCREEN_PROPERTIES = tuple(
        (
            name,
            inspect.getattr_static(ChatScreen, name),
            _pending_method_metadata(inspect.getattr_static(ChatScreen, name).fget),
        )
        for name in ("_console_chat_controller", "_console_chat_store")
    )
    _CONSOLE_PENDING_RUNTIME_METHOD = _pending_method_metadata(ChatScreen._console_runtime)
    _CONSOLE_PENDING_MODEL_READER = _pending_method_metadata(
        ConsoleInspectorState.with_pending_facts
    )
    _CONSOLE_PENDING_PARTIAL_MODEL_READER = _pending_method_metadata(
        ConsoleInspectorState.from_pending_facts.__func__
    )
    _CONSOLE_PENDING_FULL_INPUT_TYPE = _CONSOLE_PENDING_MODEL_READER[2][
        "_CONSOLE_PENDING_MODEL_SLOTS"
    ][1][0]
    _CONSOLE_PENDING_MODEL_TABLES_ORIGINAL = tuple(
        (name, _CONSOLE_PENDING_MODEL_READER[2][name])
        for name in (
            "_CONSOLE_PENDING_MODEL_SOURCES",
            "_CONSOLE_PENDING_MODEL_SLOTS",
            "_CONSOLE_PENDING_MODEL_CLASSES",
            "_CONSOLE_PENDING_MODEL_CONSTRUCTION",
        )
    )
    _CONSOLE_PENDING_STATE_CLASS_ORIGINAL = ConsoleInspectorState
    _CONSOLE_PENDING_MODEL_ABSENT_FIELD = _CONSOLE_PENDING_MODEL_READER[2][
        "_CONSOLE_PENDING_MODEL_ABSENT_FIELD"
    ]
    _CONSOLE_PENDING_MODULES_ORIGINAL = (
        (
            "tldw_chatbook.Chat.console_display_state",
            sys.modules.get("tldw_chatbook.Chat.console_display_state"),
            _CONSOLE_PENDING_MODEL_READER[2],
        ),
        (
            "tldw_chatbook.Chat.console_chat_models",
            sys.modules.get("tldw_chatbook.Chat.console_chat_models"),
            console_pending_round_copy_for.__globals__,
        ),
        (
            "tldw_chatbook.Chat.console_runtime",
            sys.modules.get("tldw_chatbook.Chat.console_runtime"),
            ConsoleRuntime.chat_controller.fget.__globals__,
        ),
    )



def _console_pending_display_owner(screen):
    try:
        return _capture_console_pending_display_owner(screen)
    except Exception:  # noqa: BLE001 - optional malformed display facts retain the original full-sync fallback.
        # Optional source/owner metadata never authorizes entry or crashes a
        # cold refresh. The original full config path and retry stay intact.
        return None


def _capture_console_pending_display_owner(screen):
    from types import FunctionType, ModuleType

    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_chat_controller as controller_module
    from tldw_chatbook.Chat import console_chat_models as models_module
    from tldw_chatbook.Chat import console_display_state as display_module
    from tldw_chatbook.Chat import console_interrupt_rounds as host_module
    from tldw_chatbook.Chat import console_runtime as runtime_module

    modules = _CONSOLE_PENDING_MODULES_ORIGINAL
    if type(modules) is not tuple or len(modules) != 3:
        return None
    if any(
        type(entry) is not tuple
        or len(entry) != 3
        or type(entry[0]) is not str  # exact stock type prevents foreign dispatch
        or type(entry[1]) is not ModuleType
        or type(entry[2]) is not dict  # exact stock type prevents foreign dispatch
        for entry in modules
    ):
        return None
    for imported, (name, original, namespace) in zip(
        (display_module, models_module, runtime_module), modules
    ):
        if (
            type(imported) is not ModuleType
            or imported is not original
            or sys.modules.get(name) is not original
            or vars(imported) is not namespace
        ):
            return None

    # Tables are definition-time references, not injectable iterable authority.
    tables = _CONSOLE_PENDING_MODEL_TABLES_ORIGINAL
    if type(tables) is not tuple or len(tables) != 4:
        return None
    if any(
        type(entry) is not tuple
        or len(entry) != 2
        or type(entry[0]) is not str  # exact stock type prevents foreign dispatch
        or type(entry[1]) is not tuple
        for entry in tables
    ):
        return None
    if any(vars(display_module).get(name) is not original for name, original in tables):
        return None
    sources, slots, classes, construction = (entry[1] for entry in tables)
    if (
        ConsoleInspectorState is not _CONSOLE_PENDING_STATE_CLASS_ORIGINAL
        or vars(display_module).get("ConsoleInspectorState")
        is not _CONSOLE_PENDING_STATE_CLASS_ORIGINAL
        or vars(display_module).get("_CONSOLE_PENDING_MODEL_ABSENT_FIELD")
        is not _CONSOLE_PENDING_MODEL_ABSENT_FIELD
        or any(type(entry) is not tuple or len(entry) != 8 for entry in sources)
        or any(
            type(entry) is not tuple
            or len(entry) != 3
            or type(entry[0]) is not type
            or type(entry[1]) is not str  # exact stock type prevents foreign dispatch
            for entry in slots
        )
        or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str  # exact stock type prevents foreign dispatch
            or type(entry[1]) is not type
            for entry in classes
        )
        or len(construction) != 3
        or any(
            type(entry) is not tuple
            or len(entry) != 3
            or type(entry[0]) is not type
            or type(entry[1]) is not tuple
            or type(entry[2]) is not tuple
            for entry in construction
        )
    ):
        return None
    if any(
        type(slot) is not tuple or len(slot) != 2 or type(slot[0]) is not str  # exact stock type prevents foreign dispatch
        for _, _, fields in construction
        for slot in fields
    ):
        return None
    if any(
        owner.__bases__ is not bases
        or any(
            inspect.getattr_static(owner, name, _CONSOLE_PENDING_MODEL_ABSENT_FIELD)
            is not original
            for name, original in fields
        )
        for owner, bases, fields in construction
    ):
        return None

    if (
        type(screen) is not ChatScreen
        or inspect.getattr_static(ChatScreen, "__getattribute__")
        is not object.__getattribute__
        or _console_screen_is_torn_down(screen)
        or getattr(screen, "_pruning", False)
    ):
        return None
    if any(
        inspect.getattr_static(screen, name) is not original
        or not _pending_method_current(record)
        for name, original, record in _CONSOLE_PENDING_SCREEN_READERS
    ):
        return None
    if any(
        inspect.getattr_static(screen, name) is not original
        or not _pending_method_current(record)
        for name, original, record in _CONSOLE_PENDING_SCREEN_PROPERTIES
    ):
        return None
    if (
        inspect.getattr_static(screen, "_console_runtime")
        is not _CONSOLE_PENDING_RUNTIME_METHOD[0]
        or not _pending_method_current(_CONSOLE_PENDING_RUNTIME_METHOD)
        or not _pending_method_current(_CONSOLE_PENDING_MODEL_READER)
        or not _pending_method_current(_CONSOLE_PENDING_PARTIAL_MODEL_READER)
    ):
        return None
    if (
        models_module._CONSOLE_PENDING_COPY_SOURCE[0]
        is not console_pending_round_copy_for
        or models_module._CONSOLE_PENDING_COPY_SOURCE[2] is not vars(models_module)
        or not _pending_method_current(models_module._CONSOLE_PENDING_COPY_SOURCE)
        or models_module.console_pending_round_copy
        is not models_module._CONSOLE_PENDING_COPY_KIND_SOURCE[0]
        or not _pending_method_current(models_module._CONSOLE_PENDING_COPY_KIND_SOURCE)
        or any(
            vars(display_module).get(name) is not original
            or type(original) is not type
            or inspect.getattr_static(original, "__getattribute__")
            is not object.__getattribute__
            for name, original in display_module._CONSOLE_PENDING_MODEL_CLASSES
        )
        or any(
            record[2] is not vars(display_module) or not _pending_method_current(record)
            for record in display_module._CONSOLE_PENDING_MODEL_SOURCES
        )
        or any(
            inspect.getattr_static(owner, name) is not original
            for owner, name, original in display_module._CONSOLE_PENDING_MODEL_SLOTS
        )
        or display_module._clean
        is not display_module._CONSOLE_PENDING_MODEL_SOURCES[2][0]
        or display_module.coerce_non_negative_int
        is not display_module._CONSOLE_PENDING_MODEL_SOURCES[3][0]
    ):
        return None
    fields = vars(screen)
    runtime, app = fields.get("_console_runtime_ref"), fields.get("app_instance")
    if (
        type(runtime) is not ConsoleRuntime
        or inspect.getattr_static(ConsoleRuntime, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(ConsoleChatStore, "__getattribute__")
        is not object.__getattribute__
    ):
        return None
    runtime_class, properties = runtime_module._CONSOLE_PENDING_RUNTIME_PROPERTIES
    if runtime_class is not ConsoleRuntime:
        return None
    for name, descriptor, function, *metadata in properties:
        if (
            inspect.getattr_static(runtime, name) is not descriptor
            or descriptor.fget is not function
            or function.__globals__ is not vars(runtime_module)
            or not _pending_method_current((function, *metadata))
        ):
            return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    import threading

    thread = threading.current_thread()
    state = vars(runtime)
    controller, store = state.get("_chat_controller"), state.get("_chat_store")
    if (
        state.get("_app") is not app
        or state.get("_disposed")
        or state.get("view") is not screen
        or state.get("_reconciled_view") is not screen
        or state.get("_attached_generation") is None
        or state.get("_attached_generation")
        != fields.get("_console_runtime_attachment_generation")
        or fields.get("_console_runtime_attachment_retired")
        or type(controller) is not controller_module.ConsoleChatController
        or type(store) is not ConsoleChatStore
        or vars(controller).get("store") is not store
    ):
        return None
    for cls, instance, definitions, namespace in (
        (
            controller_module.ConsoleChatController,
            controller,
            controller_module._CONSOLE_PENDING_FACTS_READERS,
            vars(controller_module),
        ),
        (
            host_module.InterruptRoundHost,
            vars(controller).get("_interrupt_host"),
            host_module._CONSOLE_PENDING_FACTS_READERS,
            vars(host_module),
        ),
    ):
        if (
            type(instance) is not cls
            or inspect.getattr_static(cls, "__getattribute__")
            is not object.__getattribute__
        ):
            return None
        for name, function, *metadata in definitions:
            if (
                function.__globals__ is not namespace
                or inspect.getattr_static(instance, name) is not function
                or not _pending_method_current((function, *metadata))
            ):
                return None
    # These exact stock readers dereference instance-only owner fields. A
    # new data descriptor must not redirect them past the captured receiver.
    for cls, names in (
        (ConsoleRuntime, ("_chat_controller", "_chat_store")),
        (
            controller_module.ConsoleChatController,
            ("_interrupt_host", "_pending_approvals", "_pending_round_kinds"),
        ),
        (ConsoleChatStore, ("active_session_id",)),
        (ChatScreen, ("_task_resume_state",)),
        (
            host_module.InterruptRoundHost,
            (
                "read_controller__pending_approvals",
                "read_controller__pending_round_kinds",
            ),
        ),
    ):
        if any(inspect.getattr_static(cls, name, None) is not None for name in names):
            return None
    host = vars(controller)["_interrupt_host"]
    for (
        attribute,
        code,
    ) in controller_module._CONSOLE_PENDING_FACTS_CALLBACK_CODES.items():
        callback = vars(host).get("read_controller_" + attribute)
        if (
            type(callback) is not FunctionType
            or callback.__code__ is not code
            or callback.__globals__ is not vars(controller_module)
            or callback.__defaults__ is not None
            or callback.__kwdefaults__ is not None
            or callback.__code__.co_freevars != ("self",)
            or callback.__closure__[0].cell_contents is not controller
        ):
            return None
    identity = config._CONSOLE_PENDING_FACTS_IDENTITY_SOURCE
    function, code, namespace, defaults, keywords, closure = identity
    if (
        config.current_config_identity is not function
        or function.__code__ is not code
        or function.__globals__ is not namespace
        or namespace is not vars(config)
        or function.__defaults__ is not defaults
        or function.__kwdefaults__ is not keywords
        or function.__closure__ is not closure
    ):
        return None
    current = function()
    session_id = vars(store).get("active_session_id")
    sessions = vars(store).get("_sessions")
    if type(sessions) not in (dict,):
        return None
    session = sessions.get(session_id)
    if (
        type(session) is not ConsoleChatSession
        or inspect.getattr_static(ConsoleChatSession, "__getattribute__")
        is not object.__getattribute__
    ):
        return None
    return (
        screen,
        runtime,
        app,
        controller,
        store,
        session,
        host,
        state.get("_attached_generation"),
        current,
        getattr(session, "incarnation_id", None),
        getattr(session, "workspace_id", None),
        getattr(session, "persisted_conversation_id", None),
        getattr(session, "conversation_binding_revision", None),
        (
            session.settings_revision,
            session.generation_settings_revision,
            session.context_policy_revision,
            session.identity_revision,
        ),
        loop,
        thread,
        vars(app).get("chachanotes_db"),
        vars(store).get("persistence"),
        vars(vars(store).get("persistence")).get("db")
        if vars(store).get("persistence") is not None
        else None,
        state.get("generation"),
        vars(app).get("app_config"),
    )


def _same_console_pending_owner(left, right, *, prior_base=False):
    if (
        type(left) is not tuple
        or type(right) is not tuple
        or len(left) != 21
        or len(right) != 21
    ):
        return False
    if any(
        type(value[8]) is not tuple
        or len(value[8]) != 2
        or type(value[8][0]) not in (int,)
        or type(value[8][1]) not in (str,)
        for value in (left, right)
    ):
        return False
    if any(
        left[index] is not right[index] for index in (*range(7), 14, 15, 16, 17, 18, 20)
    ):
        return False
    if prior_base:
        return (
            left[7] == right[7]
            and left[8][1] == right[8][1]
            and left[9:14] == right[9:14]
            and left[19] == right[19]
        )
    return left[7:14] == right[7:14] and left[19] == right[19]


def _sync_console_pending_display(screen):
    try:
        owner = _console_pending_display_owner(screen)
        if owner is None:
            return False
        base = vars(screen).get("_console_pending_display_base")
        if (
            owner is None
            or type(base) is not tuple
            or len(base) != 2
            or type(base[1]) is not ConsoleInspectorState
        ):
            return False
        same_base = _same_console_pending_owner(base[0], owner, prior_base=True)
        # First persistence keeps the exact Session/host/binding incarnation.
        # Its complete predecessor is a provenance receipt, never copied content.
        first_persistence = (
            not same_base
            and type(base[0]) is tuple
            and len(base[0]) == 21
            and vars(base[1]).get("pending_only") is False
            and type(vars(base[1]).get("_pending_inputs"))
            is _CONSOLE_PENDING_FULL_INPUT_TYPE
            and type(vars(vars(base[1])["_pending_inputs"]).get("live_work_title"))  # exact stock type prevents foreign dispatch
            is str
            and base[0][11] is None
            and type(owner[11]) is str  # exact stock type prevents foreign dispatch
            and bool(owner[11])
            and _same_console_pending_owner(
                (*base[0][:11], owner[11], *base[0][12:]), owner, prior_base=True
            )
        )
        if not same_base and not first_persistence:
            return False
        controller, _, session = owner[3:6]

        def current() -> bool:
            if _same_console_pending_owner(
                owner, _console_pending_display_owner(screen)
            ):
                return True
            screen._request_console_control_bar_sync(delayed=True)
            return False

        count = _CONSOLE_PENDING_SCREEN_READERS[0][1](screen)
        if not current():
            return False
        kinds = controller.pending_round_kinds(session.id)
        if not current():
            return False
        if first_persistence and (
            type(kinds) is not frozenset
            or any(type(kind) is not str for kind in kinds)  # exact stock type prevents foreign dispatch
            or not kinds
            <= {
                "approval",
                "question",
                "skill_install",
                "skill_script",
                "worktree_merge",
                "chat_create",
            }
        ):
            return False
        copy = console_pending_round_copy_for(controller, session.id) if kinds else ""
        if not current():
            return False
        state = (
            _CONSOLE_PENDING_PARTIAL_MODEL_READER[0](ConsoleInspectorState, count, copy)
            if first_persistence
            else _CONSOLE_PENDING_MODEL_READER[0](base[1], count, copy)
        )
        if state is None or not current():
            return False
        # Each original reader owns its non-reentrant lock. A final count/kind
        # reread detects intervening facts without taking an outer host lock.
        if count != _CONSOLE_PENDING_SCREEN_READERS[0][1](screen) or not current():
            screen._request_console_control_bar_sync(delayed=True)
            return False
        if kinds != controller.pending_round_kinds(session.id) or not current():
            screen._request_console_control_bar_sync(delayed=True)
            return False
        try:
            inspector = screen.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
            authority = screen.query_one(
                "#console-send-authority-summary", ConsoleSendAuthoritySummary
            )
        except QueryError:
            return False
        if not current():
            return False
        inspector.sync_state(state)
        if not current():
            return False
        authority.sync_state(state)
        if not current():
            return False
        if first_persistence and (
            count != _CONSOLE_PENDING_SCREEN_READERS[0][1](screen)
            or not current()
            or kinds != controller.pending_round_kinds(session.id)
            or not current()
        ):
            screen._request_console_control_bar_sync(delayed=True)
            return False
        return current()
    except Exception:  # noqa: BLE001 - optional malformed display facts retain the original full-sync fallback.
        # This optional display fragment never replaces the original full
        # body or its errors. Malformed optional facts safely keep the retry.
        return False
