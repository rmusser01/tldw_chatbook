"""TldwCli destination launchers, handoffs and Personal Context launchers.

Moved from ``TldwCli`` in ``app.py`` (TASK-33011): the ``open_*`` destination
launchers, typed handoff staging and Home controls (cluster D), Roleplay-to-
Console character-conversation activation (K2), and the Personal Context
interview and first-link launchers (K3). Each function takes the app as its
first parameter; the bodies are unchanged apart from ``self`` -> ``app``.

``TldwCli`` keeps a same-named stub for every function here, which imports
this module on first call (``app._destinations``). Nothing imports it at module
scope, so it stays off the boot path and out of the ADR-097 UI-ready census.
A test that patches a module-level name one of these bodies reads must patch
it HERE, not on ``tldw_chatbook.app``.
"""

# ADR-126: importing ``tldw_chatbook.app`` first runs its recovery fence
# (``admit_startup``) before any runtime import below.
from tldw_chatbook.app import TldwCli  # noqa: I001 -- the fence must import first

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from textual.css.query import QueryError

from tldw_chatbook.ACP_Interop.runtime_session import ACPRuntimeSessionState
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    resolve_console_live_work_primary_action,
)
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_MODE,
    TAB_ACP,
    TAB_ARTIFACTS,
    TAB_CHAT,
    TAB_LIBRARY,
    TAB_RESEARCH_WORKSPACE,
    TAB_STUDY,
    TAB_WATCHLISTS_COLLECTIONS,
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_RUN_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
)
from tldw_chatbook.Home.active_work_adapter import (
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
    UnavailableHomeActiveWorkAdapter,
)
from tldw_chatbook.Prompt_Management.prompt_variables import PromptVariableApplication
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffValueError,
)
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext
from tldw_chatbook.Utils.input_validation import escape_markup


# --- Destination launchers, typed handoffs and Home controls (cluster D) ---


def open_study_screen(
    app,
    scope_context: Optional[StudyScopeContext] = None,
    *,
    initial_section: Optional[str] = None,
    origin: Optional[str] = None,
) -> None:
    """Stage Study handoffs and navigate to the Study screen.

    Args:
        scope_context: Scoped study context to apply, or None to clear
            any pending scope.
        initial_section: Study section to land on, or None to clear any
            pending section.
        origin: Where the user is coming FROM (``STUDY_ORIGINS``:
            "home" or "library"), threaded to StudyScreen so its
            breadcrumb and Escape target name the actual origin
            (task-4011). None clears the channel and StudyScreen falls
            back to its historical Library default (task-2854's one
            considered origin).
    """
    if scope_context is None:
        app.pending_handoffs.clear_pending(HandoffChannel.STUDY_SCOPE)
    elif not app._stage_handoff(
        HandoffChannel.STUDY_SCOPE,
        scope_context,
        recovery="Study scope could not be opened. Try again.",
    ):
        return

    if initial_section is None:
        app.pending_handoffs.clear_pending(HandoffChannel.STUDY_INITIAL_SECTION)
    elif not app._stage_handoff(
        HandoffChannel.STUDY_INITIAL_SECTION,
        initial_section,
        recovery="Study section could not be opened. Try again.",
    ):
        return

    if origin is None:
        app.pending_handoffs.clear_pending(HandoffChannel.STUDY_ORIGIN)
    elif not app._stage_handoff(
        HandoffChannel.STUDY_ORIGIN,
        origin,
        recovery="Study could not be opened. Try again.",
    ):
        return
    app.post_message(NavigateToScreen(TAB_STUDY))


def open_notes_workspace(
    app,
    workspace_id: str,
    subview: Any = None,
) -> None:
    """Return to Library's Notes list after leaving it for another screen.

    The standalone Notes tab's per-workspace scope has no equivalent in
    Library, which browses notes as a flat list -- this always re-opens
    the shared Library Notes list rather than any workspace-scoped view.

    Args:
        workspace_id: The retired Notes tab's workspace identifier.
            Accepted for backward compatibility with existing callers
            (e.g. Study's "back to workspace" action) but no longer
            applied.
        subview: The retired Notes tab's workspace subview. Accepted for
            backward compatibility; no longer applied.
    """
    app.post_message(
        NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_MODE: "notes"})
    )


def open_conversation_archive(
    app, query: str = "", archive_scope: str = "archived"
) -> None:
    """Open Library conversation search with an explicit archive scope.

    Args:
        query: Initial title/message search text; empty lists the scope.
        archive_scope: "active", "archived" (default), or "all" saved chats.
    """
    app.post_message(
        NavigateToScreen(
            TAB_LIBRARY,
            {
                LIBRARY_NAV_CONTEXT_MODE: "conversations",
                "conversation_archive_scope": archive_scope,
                "conversation_query": query,
            },
        )
    )


def resume_console_conversation(app, conversation_id: str) -> None:
    """Review restoration scope and resume the original local conversation."""
    from .UI.Console_Modules.archive import request_conversation_resume

    app.run_worker(
        request_conversation_resume(app, conversation_id),
        name="resume-saved-conversation",
        group="resume-saved-conversation",
        exclusive=True,
    )


def open_chat_with_handoff(
    app,
    payload: ChatHandoffPayload,
    *,
    action_label: str = "Use in Chat",
) -> None:
    """Stage a handoff payload for Chat and navigate there.

    Args:
        payload: The handoff payload to stage as pending Chat context.
        action_label: The calling surface's own action label (e.g. "Use
            in Chat" for the legacy MediaWindow_v2/search_rag_window
            surfaces, "Use in Console" for Library). Currently unused
            inside this method -- it previously fed the retired
            chat-tabs gate's blocked notify (task-577 U5, which removed
            the gate so handoffs proceed unconditionally); kept for
            caller-signature compatibility.
    """
    if not app._stage_handoff(
        HandoffChannel.CHAT,
        payload,
        recovery="Chat context could not be staged. Try again.",
    ):
        return
    app.post_message(NavigateToScreen(TAB_CHAT))


def stage_console_prompt_insert(
    app,
    application: PromptVariableApplication,
) -> None:
    """Stage a guarded Prompt application and then navigate to Console.

    The typed, memory-only application carries the final selected lanes
    plus destination/session/staleness guards. Console remains the only
    owner allowed to settle the claim and mutate its active draft.

    Args:
        application: Validated Prompt application to stage.
    """
    if not app._stage_handoff(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        application,
        recovery="Console prompt could not be staged. Review it and try again.",
    ):
        return
    app.post_message(NavigateToScreen(TAB_CHAT))


def open_console_for_live_work(
    app,
    *,
    source: str,
    title: str,
    payload: dict | None = None,
    status: str | None = None,
    recovery: str | None = None,
    action_label: str | None = None,
) -> None:
    """Open Console for live work launched from another destination."""
    if not app._stage_handoff(
        HandoffChannel.CONSOLE_LIVE_WORK,
        {
            "source": source,
            "title": title,
            "payload": payload,
            "status": status,
            "recovery": recovery,
            "action_label": action_label,
        },
        recovery="Console live work could not be staged. Try again.",
    ):
        return
    app.post_message(NavigateToScreen(TAB_CHAT))


def _stage_handoff(
    app,
    channel: HandoffChannel,
    value: Any,
    *,
    recovery: str,
) -> bool:
    """Stage one typed handoff without exposing its value in recovery."""
    try:
        app.pending_handoffs.stage(channel, value)
    except HandoffValueError:
        app.notify(recovery, severity="warning")
        return False
    return True


def get_acp_runtime_session_state(app) -> ACPRuntimeSessionState:
    """Return current ACP runtime/session state for ACP and Console surfaces."""
    explicit_state = getattr(app, "acp_runtime_session_state", None)
    normalized_state = ACPRuntimeSessionState.from_any(explicit_state)
    if normalized_state.runtime_configured:
        return normalized_state
    manager = getattr(app, "acp_runtime_process_manager", None)
    snapshot = getattr(manager, "snapshot", None)
    if callable(snapshot):
        return ACPRuntimeSessionState.from_any(snapshot())
    return normalized_state


def open_console_live_work_primary_action(app, launch: Any) -> bool:
    """Follow through on a supported Console live-work status-card action."""
    normalized_launch = ConsoleLiveWorkLaunch.from_pending(launch)
    if normalized_launch is None:
        app.notify(
            "Console action is unavailable for this live-work item.",
            severity="warning",
        )
        return False

    action = resolve_console_live_work_primary_action(normalized_launch)
    if action is None:
        app.notify(
            "Console action is unavailable for this live-work item.",
            severity="warning",
        )
        return False

    if action.target_route == TAB_WATCHLISTS_COLLECTIONS:
        app.post_message(
            NavigateToScreen(
                TAB_WATCHLISTS_COLLECTIONS,
                app._watchlists_run_navigation_context(action.target_id),
            )
        )
        return True

    if action.target_route == TAB_ARTIFACTS:
        if not app._stage_handoff(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            action.target_id,
            recovery="Console action target could not be opened. Try again.",
        ):
            return False
        app.post_message(NavigateToScreen(TAB_ARTIFACTS))
        return True

    if action.target_route == TAB_ACP:
        if not app._stage_handoff(
            HandoffChannel.ACP_SESSION_TARGET,
            action.target_id,
            recovery="Console action target could not be opened. Try again.",
        ):
            return False
        app.post_message(NavigateToScreen(TAB_ACP))
        return True

    app.notify("Console action route is not available yet.", severity="warning")
    return False


def _handle_home_control_action(
    app,
    action: HomeControlAction,
    *,
    target_id: str | None = None,
    target_route: str | None = None,
) -> HomeControlResult:
    adapter = getattr(
        app, "home_active_work_adapter", UnavailableHomeActiveWorkAdapter()
    )
    if target_id is None and target_route is None:
        result = adapter.handle_control(action)
    else:
        result = adapter.handle_control(
            action,
            target_id=target_id,
            target_route=target_route,
        )
    # B3 (task-282): approve/reject/pause/resume/retry can change the
    # watchlist-run/notification state the adapter's short-TTL cache
    # holds -- invalidate so the next Home read is not stale for up to
    # the TTL window. Defensive getattr: the honest-unavailable adapter
    # and test doubles don't implement this hook.
    invalidate_cache = getattr(adapter, "invalidate_active_work_cache", None)
    if callable(invalidate_cache):
        invalidate_cache()
    app.notify(result.message, severity=result.severity)
    return result


def approve_active_home_item(
    app, *, target_id: str | None = None
) -> HomeControlResult:
    """Approve the active Home item through the configured adapter."""
    return app._handle_home_control_action(
        HomeControlAction.APPROVE, target_id=target_id
    )


def reject_active_home_item(
    app, *, target_id: str | None = None
) -> HomeControlResult:
    """Reject the active Home item through the configured adapter."""
    return app._handle_home_control_action(
        HomeControlAction.REJECT, target_id=target_id
    )


def pause_active_home_item(
    app, *, target_id: str | None = None
) -> HomeControlResult:
    """Pause the active Home item through the configured adapter."""
    return app._handle_home_control_action(
        HomeControlAction.PAUSE, target_id=target_id
    )


def resume_active_home_item(
    app, *, target_id: str | None = None
) -> HomeControlResult:
    """Resume the active Home item through the configured adapter."""
    return app._handle_home_control_action(
        HomeControlAction.RESUME, target_id=target_id
    )


def retry_active_home_item(
    app, *, target_id: str | None = None
) -> HomeControlResult:
    """Retry the active Home item through the configured adapter.

    Library ingest targets (``local:ingest:<job_id>``) use the ingest
    retry seam instead of the generic Home adapter. Ordinary jobs retain
    synchronous registry requeueing; Research-owned jobs schedule their
    durable catalog-stage retry and report Research Workspace recovery.
    Non-ingest targets are unaffected and still route through the adapter.
    """
    if target_id is not None and str(target_id).startswith("local:ingest:"):
        job_id = str(target_id)[len("local:ingest:") :]
        source = app.library_ingest_jobs.get_job(job_id)
        operation_id = str(
            getattr(source, "research_source_operation_id", "") or ""
        ).strip()
        research_retry_requested = bool(
            source is not None
            and operation_id
            and app._schedule_research_source_catalog_retry(
                source,
                operation_id=operation_id,
                notify_unavailable=False,
            )
        )
        requeued = None if operation_id else app.retry_library_ingest_job(job_id)
        if research_retry_requested:
            basename = escape_markup(
                Path(str(source.source_path)).name or str(source.source_path)
            )
            result = HomeControlResult(
                action=HomeControlAction.RETRY,
                status=HomeControlResultStatus.HANDLED,
                message=f"Research source retry requested for {basename}.",
                recovery_route=TAB_RESEARCH_WORKSPACE,
                target_id=target_id,
                target_route=TAB_RESEARCH_WORKSPACE,
            )
        elif operation_id:
            result = HomeControlResult(
                action=HomeControlAction.RETRY,
                status=HomeControlResultStatus.UNAVAILABLE,
                message=app._RESEARCH_SOURCE_RETRY_UNAVAILABLE_COPY,
                severity="warning",
                recovery_route=TAB_RESEARCH_WORKSPACE,
                target_id=target_id,
                target_route=TAB_RESEARCH_WORKSPACE,
            )
        elif requeued is None:
            # Unknown job id, or the job is no longer FAILED (e.g. it
            # was already retried/finished by the time the button was
            # pressed) -- ``requeue`` is a documented no-op in that case.
            result = HomeControlResult(
                action=HomeControlAction.RETRY,
                status=HomeControlResultStatus.UNAVAILABLE,
                message="This import job can no longer be retried.",
                severity="warning",
                recovery_route="library",
                target_id=target_id,
            )
        else:
            # The basename is a user-controlled filename (arbitrary
            # source path picked in the Library ingest form) that flows
            # straight into a Home toast, which parses Rich markup --
            # same hazard class as the open-details title fix. Escape
            # defensively.
            basename = escape_markup(
                Path(str(requeued.source_path)).name or str(requeued.source_path)
            )
            result = HomeControlResult(
                action=HomeControlAction.RETRY,
                status=HomeControlResultStatus.HANDLED,
                message=f"Retry queued for {basename}.",
                recovery_route="library",
                target_id=f"local:ingest:{requeued.job_id}",
                target_route="library",
            )
        app.notify(result.message, severity=result.severity)
        return result
    return app._handle_home_control_action(
        HomeControlAction.RETRY, target_id=target_id
    )


def open_home_flashcards_review(app) -> None:
    """Open the Study screen directly on the flashcards review surface.

    task-4011: this is the one entry into Study that does NOT come from
    Library's staging canvas, so it declares its origin -- StudyScreen's
    breadcrumb reads "Home ▸ Study" and Escape returns to Home instead
    of a Library canvas the user never visited.
    """
    app.open_study_screen(initial_section="flashcards", origin="home")


def open_active_home_item_details(
    app,
    *,
    target_id: str | None = None,
    target_route: str = TAB_CHAT,
) -> HomeControlResult:
    """Open active Home item details through the configured adapter."""
    result = app._handle_home_control_action(
        HomeControlAction.OPEN_DETAILS,
        target_id=target_id,
        target_route=target_route,
    )
    if result.status is HomeControlResultStatus.HANDLED and result.target_route:
        if result.target_route in {
            "subscriptions",
            TAB_WATCHLISTS_COLLECTIONS,
        }:
            app.post_message(
                NavigateToScreen(
                    TAB_WATCHLISTS_COLLECTIONS,
                    app._watchlists_run_navigation_context(
                        result.target_id or target_id
                    ),
                )
            )
        elif result.target_route == "library" and str(
            result.target_id or target_id or ""
        ).startswith("local:ingest:"):
            # Home's ingest-jobs Running/Needs Attention rows one-hop
            # back to the Library ingest canvas via the nav-context
            # contract instead of a bare route (mirrors the
            # subscriptions staging special-case above). Navigation
            # always composes a fresh Library screen, so the deep link
            # lands on a cleanly mounted, repainted ingest canvas.
            app.post_message(
                NavigateToScreen("library", {LIBRARY_NAV_CONTEXT_INGEST: True})
            )
        else:
            app.post_message(NavigateToScreen(result.target_route))
    return result


def _watchlists_run_navigation_context(
    target_id: str | None,
) -> dict[str, object]:
    """Build the destination-owned context for a Watchlists run deep link."""
    context: dict[str, object] = {
        WATCHLISTS_NAV_CONTEXT_SECTION: WATCHLISTS_SECTION_RUNS
    }
    if target_id:
        target_id_text = str(target_id)
        context[WATCHLISTS_NAV_CONTEXT_RUN_ID] = target_id_text
        backend = target_id_text.partition(":watchlist_run:")[0]
        if backend in {"local", "server"}:
            context[WATCHLISTS_NAV_CONTEXT_BACKEND] = backend
    return context


def open_active_home_item_in_console(
    app,
    *,
    target_id: str | None = None,
    target_route: str = TAB_CHAT,
) -> HomeControlResult:
    """Open active Home item in Console only when the adapter supplies launch context."""
    result = app._handle_home_control_action(
        HomeControlAction.OPEN_IN_CONSOLE,
        target_id=target_id,
        target_route=target_route,
    )
    if (
        result.status is HomeControlResultStatus.HANDLED
        and result.console_launch is not None
    ):
        launch_kwargs = {
            "source": result.console_launch.source,
            "title": result.console_launch.title,
            "payload": dict(result.console_launch.payload or {}),
        }
        if result.console_launch.status is not None:
            launch_kwargs["status"] = result.console_launch.status
        if result.console_launch.recovery is not None:
            launch_kwargs["recovery"] = result.console_launch.recovery
        if result.console_launch.action_label is not None:
            launch_kwargs["action_label"] = result.console_launch.action_label
        app.open_console_for_live_work(**launch_kwargs)
    return result


# --- Roleplay-to-Console character-conversation activation (cluster K2) ---


async def activate_character_conversation_from_roleplay(
    app,
    request: object,
    cancellation: asyncio.Event,
    phase_changed: Callable[[str], None],
) -> object:
    """Own an exact Roleplay-to-Console activation without losing its caller.

    The reusable Console workspace performs cancellable preflight while
    Roleplay remains the current screen. Once that check settles, the
    operation enters its non-cancellable finishing phase and mounts Console
    for the final atomic revalidation/hydration.  Failure restores the exact
    mounted Roleplay caller; only ``OPENED`` replaces it.
    """

    from tldw_chatbook.Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
        ConsoleActivationResultKind,
        ConsoleConversationActivationResult,
    )

    if not isinstance(request, CharacterConversationActivationRequest):
        raise TypeError("request must be a character activation request")
    cancelled_result = ConsoleConversationActivationResult(
        ConsoleActivationResultKind.CANCELLED_PRECOMMIT,
        request.target,
        False,
    )
    if cancellation.is_set():
        return cancelled_result
    runtime = getattr(app, "console_runtime", None)
    lane = getattr(runtime, "character_conversation_activation_lock", None)
    if not isinstance(lane, asyncio.Lock):
        raise RuntimeError("Console activation lane is unavailable")  # noqa: TRY004 - unavailable runtime ownership, not a public input type check
    admission: asyncio.Task[bool] | None = None
    cancelled: asyncio.Task[bool] | None = None
    preflight: asyncio.Task[Any] | None = None
    lane_acquired = False
    commit_started = False
    candidate = None
    try:
        # These children are owned from creation: cancellation of this
        # outer worker must never leave an orphan that later acquires the
        # app-lifetime lane.
        admission = asyncio.create_task(lane.acquire())
        cancelled = asyncio.create_task(cancellation.wait())
        done, _pending = await asyncio.wait(
            {admission, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancelled in done and cancellation.is_set():
            return cancelled_result
        lane_acquired = bool(await admission)
        if cancellation.is_set():
            return cancelled_result

        _, _, screen_class = app._resolve_screen_navigation_target(TAB_CHAT)
        if screen_class is None:
            return ConsoleConversationActivationResult(
                ConsoleActivationResultKind.FAILED, request.target, False
            )
        runtime_identity = app._current_runtime_identity()
        candidate = app._reusable_navigation_screen(TAB_CHAT, runtime_identity)
        if candidate is None:
            candidate = app._create_navigation_screen(TAB_CHAT, screen_class)
        preflight = asyncio.create_task(
            candidate._workspace.preflight_character_conversation_activation(
                request
            )
        )
        if cancelled is not None and not cancelled.done():
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
        cancelled = asyncio.create_task(cancellation.wait())
        done, _pending = await asyncio.wait(
            {preflight, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancelled in done and cancellation.is_set():
            preflight.cancel()
            await asyncio.gather(preflight, return_exceptions=True)
            return cancelled_result
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        preflight_result = await preflight
        if preflight_result is not None:
            return ConsoleConversationActivationResult(
                preflight_result, request.target, False
            )
        if cancellation.is_set():
            return cancelled_result

        # The app-wide lane and final revalidation are both owned here.
        # Publishing Finishing is the atomic commit acknowledgement; only
        # after it is visible does the surviving Console touch the stack.
        phase_changed("finishing")
        commit_started = True
        caller = app.screen
        post_commit = asyncio.create_task(
            TldwCli._complete_character_conversation_post_commit(
                app, candidate, caller, request, runtime_identity
            ),
            name="character_conversation_post_commit",
        )
        return await TldwCli._await_character_conversation_post_commit(
            post_commit
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - admission failures return typed outcomes and drain owned tasks
        logger.opt(exception=True).warning(
            "Roleplay character-conversation activation failed"
        )
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED,
            request.target,
            commit_started,
        )
    finally:
        children = tuple(
            task
            for task in (admission, cancelled, preflight)
            if task is not None
        )
        for child in children:
            if not child.done():
                child.cancel()
        child_results = (
            await asyncio.gather(*children, return_exceptions=True)
            if children
            else ()
        )
        if not lane_acquired and admission is not None:
            admission_index = children.index(admission)
            lane_acquired = child_results[admission_index] is True
        if lane_acquired:
            lane.release()


async def _await_character_conversation_post_commit(
    operation: asyncio.Task[Any],
) -> Any:
    """Delay caller cancellation until app-owned commit work settles."""

    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(operation)
        except asyncio.CancelledError as error:
            if operation.cancelled():
                raise
            cancellation = cancellation or error
            continue
        break
    if cancellation is not None:
        raise cancellation
    return result


async def _complete_character_conversation_post_commit(
    app,
    candidate: Any,
    caller: Any,
    request: Any,
    runtime_identity: Any,
) -> Any:
    """Mount, hydrate, transfer, or roll back one committed activation."""

    from tldw_chatbook.Chat.console_conversation_activation import (
        ConsoleActivationResultKind,
        ConsoleConversationActivationResult,
    )

    transferred = False

    async def finalize_visible() -> None:
        nonlocal transferred
        await TldwCli._transfer_pushed_console_to_content(
            app, candidate, caller
        )
        transferred = True

    # A cached Console resumes instead of mounting. Its ordinary registry
    # reconciliation must not compete with this exact target's transaction.
    prior_resume_gate = getattr(
        candidate, "_resume_navigation_startup_in_progress", False
    )
    retire_resume = getattr(candidate, "_retire_resume_navigation_startup", None)
    if callable(retire_resume) and await retire_resume():
        prior_resume_gate = False
    candidate._resume_navigation_startup_in_progress = True
    try:
        await app.push_screen(candidate)
        result = (
            await candidate._workspace.activate_character_conversation_after_commit(
                request,
                finalize_visible=finalize_visible,
            )
        )
        if result.kind is ConsoleActivationResultKind.OPENED and transferred:
            if not app.is_screen_installed(candidate):
                app._retain_reusable_navigation_screen(
                    TAB_CHAT, runtime_identity, candidate
                )
            app.current_tab = TAB_CHAT
            return result
        if getattr(app, "screen", None) is candidate:
            await app.pop_screen()
        if result.kind is ConsoleActivationResultKind.OPENED:
            return ConsoleConversationActivationResult(
                ConsoleActivationResultKind.FAILED, request.target, True
            )
        return result
    except Exception:  # noqa: BLE001 - restore caller after any committed mount or transfer failure
        logger.bind(
            operation_id=id(request), candidate_token=id(candidate),
            caller_token=id(caller), stage="commit_screen",
        ).opt(exception=True).warning(
            "Committed Roleplay character-conversation activation failed"
        )
        if getattr(app, "screen", None) is candidate:
            try:
                await app.pop_screen()
            except Exception:  # noqa: BLE001 - report caller restoration without losing typed outcome
                logger.bind(
                    operation_id=id(request), candidate_token=id(candidate),
                    caller_token=id(caller), stage="restore_caller",
                ).opt(exception=True).error(
                    "Could not restore Roleplay after Console mount failure"
                )
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, True
        )
    finally:
        candidate._resume_navigation_startup_in_progress = prior_resume_gate


async def _transfer_pushed_console_to_content(
    app,
    candidate: Any,
    caller: Any,
) -> None:
    """Promote the proved pushed Console while replacing its Roleplay caller."""

    stack = app._screen_stack
    if (
        len(stack) < 2
        or stack[-1] is not candidate
        or stack[-2] is not caller
    ):
        raise RuntimeError("Console activation stack ownership changed")

    original_stack = tuple(stack)
    candidate_callbacks = tuple(candidate._result_callbacks)
    caller_callbacks = tuple(caller._result_callbacks)

    # The proved candidate is already current, mounted, and owns its push
    # callback. Remove only the caller beneath it; unlike switch_screen,
    # this never exposes a state where Textual has popped both screens.
    try:
        stack.pop(-2)
        caller._pop_result_callback()
        await app._remove_promoted_screen_caller(caller)
    except BaseException:
        # Restore exact membership/callback ownership even if the removal
        # seam failed after changing the current stack. A normal pop then
        # resumes Roleplay and unmounts the attempt-owned candidate.
        stack[:] = original_stack
        candidate._result_callbacks[:] = candidate_callbacks
        caller._result_callbacks[:] = caller_callbacks
        try:
            if not caller.is_running:
                restored_caller, await_mount = app._get_screen(caller)
                if restored_caller is not caller:
                    raise RuntimeError("Roleplay screen identity changed during restore")
                await await_mount
            await app.pop_screen()
        except BaseException:  # noqa: BLE001 - fallback restores ownership even when cleanup is cancelled
            logger.bind(
                candidate_token=id(candidate), caller_token=id(caller),
                stage="restore_caller",
            ).opt(exception=True).error(
                "Could not atomically restore Roleplay after Console promotion"
            )
            stack[:] = original_stack[:-1]
            caller._result_callbacks[:] = caller_callbacks
            candidate._result_callbacks[:] = candidate_callbacks[:-1]
            try:
                if not caller.is_running:
                    restored_caller, await_mount = app._get_screen(caller)
                    if restored_caller is not caller:
                        raise RuntimeError(
                            "Roleplay screen identity changed during fallback restore"
                        )
                    await await_mount
                if (
                    candidate.is_running
                    and candidate.parent is app
                    and not app.is_screen_installed(candidate)
                ):
                    await candidate.remove()
            except BaseException:  # noqa: BLE001 - final attempt-owned cleanup must not replace original failure
                logger.bind(
                    candidate_token=id(candidate), caller_token=id(caller),
                    stage="cleanup_candidate",
                ).opt(exception=True).error(
                    "Could not clean failed Console promotion candidate"
                )
        raise


async def _remove_promoted_screen_caller(app, caller: Any) -> None:
    """Unmount the former content screen after its stack slot is removed."""

    await caller.remove()


if TYPE_CHECKING:
    from tldw_chatbook.Personal_Context.interview_launch import (
        ProfileInterviewLaunchRequest,
    )
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewScreen,
    )


# --- Personal Context interview and first-link launchers (cluster K3) ---


def prepare_personal_context_interview_request(
    app,
    *,
    kind: str,
    mode: str = "fixed",
    scope_id: str | None = None,
    local_workspace_id: str | None = None,
    workspace_label: str = "",
    source: str | None = None,
) -> "ProfileInterviewLaunchRequest":
    """Resolve one canonical scope without touching workspace ownership.

    Args:
        app: The running TldwCli.
        kind: ``"personal"`` (the global scope) or ``"workspace"``.
        mode: ``"fixed"`` or ``"adaptive"`` interview.
        scope_id: An existing scope to interview; it must match ``kind``.
        local_workspace_id: The workspace to bind (``kind="workspace"``
            without ``scope_id``); its scope is created if missing.
        workspace_label: Label for a newly created workspace scope.
        source: Where the launch came from: ``None``, ``"setup"``,
            ``"workspace"`` or ``"settings"``.

    Returns:
        The launch request naming the resolved scope.

    Raises:
        ValueError: For an unknown kind, mode or source; when Personal
            Context is locked or removed; when ``scope_id`` does not match
            ``kind``; when the global scope is missing; or when a workspace
            interview has no local workspace.
    """

    from tldw_profile_core import ScopeKind

    from .Personal_Context.interview_launch import ProfileInterviewLaunchRequest

    if kind not in {"personal", "workspace"}:
        raise ValueError("Unknown Personal Context interview kind.")
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("Unknown Personal Context interview mode.")
    if source not in {None, "setup", "workspace", "settings"}:
        raise ValueError("Unknown Personal Context interview source.")
    service = app.get_personal_context_service(retry_locked=True)
    status = service.status()
    if status.state.value == "absent":
        service.create_profile()
    elif status.state.value in {"locked", "removed"}:
        raise ValueError("Personal Context is unavailable.")
    scopes = service.list_scopes()
    if scope_id is not None:
        scope = next((item for item in scopes if item.scope_id == scope_id), None)
        expected = ScopeKind.GLOBAL if kind == "personal" else ScopeKind.WORKSPACE
        if scope is None or scope.kind is not expected:
            raise ValueError("Personal Context scope does not match interview.")
    elif kind == "personal":
        scope = next(
            (item for item in scopes if item.kind is ScopeKind.GLOBAL), None
        )
        if scope is None:
            raise ValueError("Global Personal Context scope is unavailable.")
    else:
        local_workspace_id = str(local_workspace_id or "").strip()
        if not local_workspace_id:
            raise ValueError("Workspace interview requires a local workspace.")
        bindings = service.list_workspace_bindings()
        scope = next(
            (
                item
                for item in scopes
                if item.kind is ScopeKind.WORKSPACE
                and bindings.get(item.scope_id, {}).get("local_workspace_id")
                == local_workspace_id
            ),
            None,
        )
        if scope is None:
            scope = service.create_workspace_scope(
                local_workspace_id,
                workspace_label or "Workspace",
            )
    return ProfileInterviewLaunchRequest(
        kind=kind,
        scope_id=scope.scope_id,
        local_workspace_id=local_workspace_id,
        mode=mode,
        source=source,
    )


def build_personal_context_interview_screen(
    app: TldwCli, request: "ProfileInterviewLaunchRequest"
) -> "ProfileInterviewScreen":
    """Build a fresh profile interview screen for one resolved request.

    Args:
        app: The running TldwCli.
        request: A request from ``prepare_personal_context_interview_request``.

    Returns:
        The new, not yet pushed, interview screen.
    """

    from .Personal_Context.interview_launch import build_profile_interview_screen

    return build_profile_interview_screen(app, request)


def launch_personal_context_interview(
    app,
    kind: str,
    scope_id: str,
    mode: str = "fixed",
) -> None:
    """Settings re-interview seam over the shared post-commit launcher."""

    from .Personal_Context.interview_launch import (
        launch_profile_interview_after_commit,
    )

    request = app.prepare_personal_context_interview_request(
        kind=kind,
        scope_id=scope_id,
        mode=mode,
        source="settings",
    )
    launch_profile_interview_after_commit(
        app,
        request,
        lambda: TldwCli._reload_personal_context_settings_panel(app),
    )


def _reload_personal_context_settings_panel(app) -> None:
    """Reload the mounted My Profile panel after a re-interview returns."""

    from .Widgets.Settings_Widgets.personal_context_panel import (
        PersonalContextSettingsPanel,
    )

    try:
        panel = app.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
    except QueryError:
        return
    panel.load_records(retry_locked=True)


def launch_personal_context_link(app) -> None:
    """Open the reviewed home-server link flow from canonical Settings."""

    app.run_worker(
        app._run_personal_context_link(),
        group="personal-context-first-link",
        exclusive=True,
        exit_on_error=False,
    )


async def _run_personal_context_link(app) -> None:
    """Plan, review, and apply one content-safe first-link attempt."""

    import platform

    from .Personal_Context.link_key_custody import (
        KeyringPersonalContextLinkKeyCustodian,
        KeyringPersonalContextWrappingKeyProvider,
    )
    from .Personal_Context.link_service import (
        PersonalContextLinkAttentionRequired,
        PersonalContextLinkService,
    )
    from .Personal_Context.key_protector import ProfileLockedError
    from .Personal_Context.paths import get_personal_context_db_path
    from .Personal_Context.repository import (
        release_first_link_freeze_for_recovery,
    )
    from .Widgets.Settings_Widgets.personal_context_link_modal import (
        PersonalContextLinkModal,
    )

    scope = app._server_notification_event_scope()
    server_profile_id = scope.get("server_profile_id")
    if not server_profile_id:
        app.notify(
            "Choose and authenticate a home server before linking your profile.",
            severity="warning",
        )
        return
    try:
        wrapping_provider = KeyringPersonalContextWrappingKeyProvider()
        key_custodian = KeyringPersonalContextLinkKeyCustodian()
        existing = app.sync_state_repository.get_personal_context_link_state(
            server_profile_id=str(server_profile_id),
            authenticated_principal_id=scope.get("authenticated_principal_id"),
        )
        recovered_apply = False
        if existing is not None and existing["state"] == "applying":
            binding = PersonalContextLinkService._key_binding(existing)
            try:
                staged_integrity_key = key_custodian.load(**binding)
                from .Personal_Context.bootstrap import (
                    bootstrap_personal_context_service,
                )

                recovered_service = bootstrap_personal_context_service(
                    recovery_integrity_key=staged_integrity_key,
                    expected_recovery_profile_id=str(existing["profile_id"]),
                )
            except (ProfileLockedError, ValueError):
                recovered_service = None
            if (
                recovered_service is not None
                and recovered_service.status().state.value == "ready"
            ):
                app._personal_context_service = recovered_service
                recovered_apply = True
        coordinator = PersonalContextLinkService(
            personal_context_service=app.get_personal_context_service(
                retry_locked=True
            ),
            server_sync_service=app.server_sync_service,
            state_repository=app.sync_state_repository,
            wrapping_key_provider=wrapping_provider,
            key_custodian=key_custodian,
            freeze_release_fallback=lambda plan_id: (
                release_first_link_freeze_for_recovery(
                    get_personal_context_db_path(), plan_id=plan_id
                )
            ),
            local_first_sync_service=app.local_first_sync_service,
            server_profile_id=str(server_profile_id),
            authenticated_principal_id=scope.get("authenticated_principal_id"),
            display_name=platform.node() or "Chatbook",
        )
        if existing is not None and existing["state"] == "complete":
            await coordinator.resume()
            # task-33081: this restore reads the link storage key from
            # the OS keyring (D-Bus SecretService on Linux) -- run it
            # off the UI event loop.
            await asyncio.to_thread(
                app._load_personal_context_sync_runtime,
                server_profile_id=str(server_profile_id),
                authenticated_principal_id=scope.get("authenticated_principal_id"),
            )
            app.notify("Profile is already linked. Sync is ready.")
            app._reload_personal_context_settings_panel()
            return
        if recovered_apply:
            await coordinator.resume_after_local_activation(
                rebaseline_version=(
                    app.get_personal_context_service()
                    .first_link_rebaseline_version()
                )
            )
            app.notify("Profile link completed.")
            app._reload_personal_context_settings_panel()
            return
        if existing is not None and existing["state"] == "applying":
            active_reader = getattr(
                coordinator,
                "authenticated_committed_rebaseline_version",
                None,
            )
            active_version = active_reader() if callable(active_reader) else None
            if active_version is not None:
                await coordinator.resume_after_local_activation(
                    rebaseline_version=active_version
                )
                app.notify("Profile link completed.")
                app._reload_personal_context_settings_panel()
                return
            if not coordinator.abandon_uncommitted_apply():
                mark_attention = getattr(
                    coordinator, "mark_ambiguous_apply_attention", None
                )
                if callable(mark_attention):
                    mark_attention()
                raise ProfileLockedError(
                    "Interrupted Personal Context link recovery is pending."
                )
        if existing is not None and existing["state"] in {
            "local_rebaseline_complete",
            "reconciling",
        }:
            await coordinator.resume()
            app.notify("Profile link completed.")
            app._reload_personal_context_settings_panel()
            return
        while True:
            manifest = app.get_personal_context_service().get_manifest()
            try:
                plan = await coordinator.plan(
                    expected_purge_generation=manifest.purge_generation
                )
            except PersonalContextLinkAttentionRequired as exc:
                attention_result = await app.push_screen_wait(
                    PersonalContextLinkModal.for_bootstrap_attention(
                        exc.attention,
                        retry_callback=True,
                    )
                )
                if attention_result is not None and attention_result.retry:
                    continue
                return
            result = await app.push_screen_wait(
                PersonalContextLinkModal(plan, retry_callback=True)
            )
            if result is None:
                coordinator.cancel(plan.plan_id)
                return
            if result.retry:
                coordinator.cancel(plan.plan_id)
                continue
            await coordinator.apply(result.plan_id, result.decisions)
            break
    except Exception:
        app.notify(
            "Profile linking needs attention. No profile content was shown; retry from Settings.",
            severity="error",
        )
        return
    app.notify("Profile linked to the home server.")
    app._reload_personal_context_settings_panel()
