"""Console settings submission durability and app-lifetime default recovery."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
import asyncio
from functools import partial
from loguru import logger
from ...Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsSurface,
    ConsoleSettingsSubmission,
)
from ...Chat.console_settings_durability import ConsoleSettingsDurabilityOwner
from ...Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultMutationOutcome,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultRuntimePublicationClaim,
    ConsoleDefaultSavePhase,
    abort_console_default_runtime_publication,
    apply_console_default_intent,
    build_console_default_intent,
    complete_console_default_runtime_publication,
    next_console_default_intent_generation,
    prepare_console_default_intent_reservation,
    prepare_console_default_runtime_publication,
    publish_console_default_runtime_if_current,
    refresh_console_runtime_after_saved_default,
    reserve_console_default_intent_generation,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    unsaved_console_endpoint_warning,
)
from ...Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleRoleplayProjectionPersistencePlan,
    ConsoleRoleplayProjectionPersistenceResult,
    ConsoleSettingsPolicyFailureLabel,
)
from ...Chat.provider_readiness import provider_config_key
from ...Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    normalize_chat_display_name,
)


if TYPE_CHECKING:
    from ...Widgets.Console.console_settings_modal import ConsoleSettingsResult


logger = logger.bind(module="ChatScreen")
_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS = 8
#: Maximum default wait for the screen-owned roleplay drain during unmount.
#: The immutable writer may outlive this deadline, but never the screen-bound
#: coordinator or its store/session presentation state.
CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS = 0.25


def _consume_console_roleplay_writer_completion(
    task: asyncio.Task[ConsoleRoleplayProjectionPersistenceResult | None],
    *,
    session_id: str,
    generation: int,
) -> None:
    """Consume an app-owned serializer result without retaining its screen."""
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.exception(
            "App-owned Console roleplay projection writer failed "
            "(session_id={}, generation={}).",
            session_id,
            generation,
        )


def _release_console_roleplay_transition_after_writer(
    future: asyncio.Future[ConsoleRoleplayProjectionPersistenceResult | None],
    *,
    store: ConsoleChatStore,
    plan: ConsoleRoleplayProjectionPersistencePlan,
) -> None:
    """Schedule fallback release after the owner's acceptance callback can run."""

    try:
        future.get_loop().call_soon(store.abandon_roleplay_projection_plan, plan)
    except RuntimeError:
        store.abandon_roleplay_projection_plan(plan)


def _consume_console_roleplay_repair_for_current_screen(
    app_instance: Any,
) -> None:
    """Ask the app's current Console owner to consume a repair marker."""
    try:
        current_screen = app_instance.screen
    except Exception:  # noqa: BLE001 - lifecycle repair is best-effort
        return
    consume = getattr(
        current_screen,
        "_consume_pending_console_roleplay_repair",
        None,
    )
    if callable(consume):
        consume()


def _conversation_settings_modal_module():
    """Load the Conversation Settings modal only when its workflow starts."""
    from ...Widgets.Console import console_settings_modal

    return console_settings_modal


class ConsoleSettingsDurabilityController:
    """Own console settings submission durability and app-lifetime default recovery.

    Dependencies, including the app owner, resolve through named callables
    at use time. The controller owns no DOM or screen handle.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        app_accessor: Callable[[], Any],
        is_mounted_accessor: Callable[[], bool],
        current_console_chat_store_accessor: Callable[[], ConsoleChatStore | None],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _provider_readiness_app_config: Callable[..., Any],
        _sync_console_identity_surfaces: Callable[..., Any],
        _sync_console_settings_recovery_surfaces: Callable[..., Any],
        _sync_native_console_chat_ui: Callable[..., Any],
        run_worker: Callable[..., Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._app_accessor = app_accessor
        self._is_mounted_accessor = is_mounted_accessor
        self._current_console_chat_store_accessor = current_console_chat_store_accessor
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._provider_readiness_app_config = _provider_readiness_app_config
        self._sync_console_identity_surfaces = _sync_console_identity_surfaces
        self._sync_console_settings_recovery_surfaces = (
            _sync_console_settings_recovery_surfaces
        )
        self._sync_native_console_chat_ui = _sync_native_console_chat_ui
        self.run_worker = run_worker
        self._console_settings_coordinated_submission_ids = deque(maxlen=64)
        self._last_console_roleplay_refresh_key: tuple[str, str] | None = None
        self._console_roleplay_persistence_task: asyncio.Task[None] | None = None
        self._console_roleplay_writer_task: (
            asyncio.Task[ConsoleRoleplayProjectionPersistenceResult | None] | None
        ) = None
        self._console_roleplay_active_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None
        self._console_roleplay_pending_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None
        self._console_roleplay_drain_scheduled = False
        self._console_roleplay_tearing_down = False
        self._console_roleplay_repair_generation = 0
        self._console_roleplay_repair_inflight_generation = 0
        self._console_roleplay_repair_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    @property
    def app(self) -> Any:
        return self._app_accessor()

    @property
    def is_mounted(self) -> bool:
        return self._is_mounted_accessor()

    @property
    def _console_chat_store(self) -> ConsoleChatStore | None:
        return self._current_console_chat_store_accessor()

    def _global_chat_display_name(self) -> str:
        """Return the live in-memory global chat label without touching disk."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        chat_defaults = (
            app_config.get("chat_defaults", {})
            if isinstance(app_config, Mapping)
            else {}
        )
        raw_value = (
            chat_defaults.get("user_display_name", "User")
            if isinstance(chat_defaults, Mapping)
            else "User"
        )
        try:
            return (
                normalize_chat_display_name(raw_value, blank_means_none=False) or "User"
            )
        except ChatDisplayNameError:
            return "User"

    def _apply_console_settings_result(
        self,
        result: "ConsoleSettingsResult | ConsoleSessionSettings | None",
        *,
        origin_session_id: str | None = None,
        origin_system_prompt: str | None = None,
        origin_pinned_prefill: str | None = None,
    ) -> None:
        """Apply provider settings and the separately owned chat-name override."""
        modal_contract = _conversation_settings_modal_module()
        if not isinstance(
            result,
            (modal_contract.ConsoleSettingsResult, ConsoleSessionSettings),
        ):
            return
        settings = (
            result.settings
            if isinstance(result, modal_contract.ConsoleSettingsResult)
            else result
        )
        store = self._ensure_console_chat_store()
        session_id = origin_session_id or store.active_session_id
        if session_id is None:
            return
        try:
            current_settings = store.session_settings(session_id)
        except KeyError:
            return
        current_system_prompt = (
            origin_system_prompt
            if origin_session_id is not None
            else (
                current_settings.system_prompt if current_settings is not None else None
            )
        )
        current_pinned_prefill = (
            origin_pinned_prefill
            if origin_session_id is not None
            else (
                current_settings.pinned_prefill
                if current_settings is not None
                else None
            )
        )
        store.replace_session_settings(
            session_id,
            replace(
                settings,
                source="user",
                system_prompt=current_system_prompt,
                pinned_prefill=current_pinned_prefill,
            ),
        )
        if isinstance(result, modal_contract.ConsoleSettingsResult):
            if result.context_policy_overrides is not None:
                _session, policy_persisted = store.set_session_context_policy_overrides(
                    session_id,
                    result.context_policy_overrides,
                )
                if not policy_persisted:
                    self.app_instance.notify(
                        "Context policy applied in memory but could not be saved.",
                        severity="warning",
                    )
            if result.thinking_history_policy is not None:
                _session, thinking_persisted = (
                    store.set_session_thinking_history_policy(
                        session_id,
                        result.thinking_history_policy,
                    )
                )
                if not thinking_persisted:
                    self.app_instance.notify(
                        "Thinking history policy applied in memory but could not be saved.",
                        severity="warning",
                    )
            _session, persisted = store.set_session_user_display_name_override(
                session_id,
                result.user_display_name_override,
                global_default=self._global_chat_display_name(),
            )
            self._last_console_roleplay_refresh_key = (
                session_id,
                self._global_chat_display_name(),
            )
            if not persisted:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
        if store.active_session_id == session_id:
            self._sync_console_identity_surfaces()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
        self.app_instance.notify("Console settings saved.", severity="success")
        # task-16473: a session endpoint with no persisted backing works for
        # this run (llama.cpp readiness even reports "Ready") and then
        # silently evaporates on restart -- the exact trap behind the
        # "re-enter my llama.cpp IP:Port every boot" report.
        endpoint_warning = unsaved_console_endpoint_warning(
            settings,
            app_config=self._provider_readiness_app_config(),
        )
        if endpoint_warning:
            self.app_instance.notify(endpoint_warning, severity="warning")

    async def _refresh_console_roleplay_projections(
        self,
        plan: ConsoleRoleplayProjectionPersistencePlan,
    ) -> None:
        """Persist one current immutable plan without off-thread store access."""
        store = self._ensure_console_chat_store()
        if not store.is_roleplay_projection_plan_current(plan):
            store.abandon_roleplay_projection_plan(plan)
            if self._console_roleplay_repair_plan is plan:
                self._console_roleplay_repair_plan = None
                self._console_roleplay_repair_inflight_generation = 0
            return
        owner = self._console_settings_durability_owner()
        admission = owner.try_acquire()
        if admission is None:
            store.abandon_roleplay_projection_plan(plan)
            if self._console_roleplay_repair_plan is plan:
                self._console_roleplay_repair_plan = None
                self._console_roleplay_repair_inflight_generation = 0
            return
        persistence_task = owner.launch(
            admission,
            store.persist_roleplay_projection_plan_serialized(plan),
            name=(f"console-roleplay-{plan.session_id}-{plan.generation}"),
        )
        persistence_task.add_done_callback(
            partial(
                _consume_console_roleplay_writer_completion,
                session_id=plan.session_id,
                generation=plan.generation,
            )
        )
        persistence_task.add_done_callback(
            partial(
                _release_console_roleplay_transition_after_writer,
                store=store,
                plan=plan,
            )
        )
        self._console_roleplay_writer_task = persistence_task
        try:
            result = await asyncio.shield(persistence_task)
        except asyncio.CancelledError:
            # Unmount abandons only the screen waiter; the app owner keeps and
            # drains the durable task. Mounted cancellation continues to wait
            # so the latest coalesced refresh is not lost.
            if self._console_roleplay_tearing_down:
                raise
            while not persistence_task.done():
                try:
                    await asyncio.shield(persistence_task)
                except asyncio.CancelledError:
                    continue
            result = persistence_task.result()
        finally:
            if self._console_roleplay_writer_task is persistence_task:
                self._console_roleplay_writer_task = None
        if result is None:
            if self._console_roleplay_repair_plan is plan:
                self._console_roleplay_repair_plan = None
                self._console_roleplay_repair_inflight_generation = 0
            return
        accepted = store.accept_roleplay_projection_persistence_result(result)
        if self._console_roleplay_repair_plan is plan:
            repair_generation = self._console_roleplay_repair_inflight_generation
            self._console_roleplay_repair_plan = None
            self._console_roleplay_repair_inflight_generation = 0
            if accepted and result.persisted and repair_generation > 0:
                self._console_roleplay_repair_generation = repair_generation
                self.app_instance._console_roleplay_repair_consumed_generation = max(
                    repair_generation,
                    int(
                        getattr(
                            self.app_instance,
                            "_console_roleplay_repair_consumed_generation",
                            0,
                        )
                        or 0
                    ),
                )
        if accepted and not result.persisted:
            self.app_instance.notify(
                "Your chat name is active, but updated character templates may not "
                "survive reopening.",
                severity="warning",
            )
        if accepted and store.active_session_id == plan.session_id:
            self._sync_console_identity_surfaces()

    async def _drain_console_roleplay_persistence(self) -> None:
        """Drain one active and one replaceable latest projection plan."""
        plan: ConsoleRoleplayProjectionPersistencePlan | None = None
        try:
            while self._console_roleplay_pending_plan is not None:
                plan = self._console_roleplay_pending_plan
                self._console_roleplay_pending_plan = None
                self._console_roleplay_active_plan = plan
                await self._refresh_console_roleplay_projections(plan)
                self._console_roleplay_active_plan = None
                plan = None
        except asyncio.CancelledError:
            if (
                self._console_roleplay_pending_plan is None
                and plan is not None
                and self._ensure_console_chat_store().is_roleplay_projection_plan_current(
                    plan
                )
            ):
                self._console_roleplay_pending_plan = plan
            raise
        finally:
            self._console_roleplay_active_plan = None
            self._console_roleplay_drain_scheduled = False

    async def _await_console_roleplay_persistence_task(
        self, task: asyncio.Task[None]
    ) -> None:
        """Let Textual cancel its waiter without cancelling the durable queue."""
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            return

    def _finish_console_roleplay_persistence_task(
        self, task: asyncio.Task[None]
    ) -> None:
        """Release a drained task and consume any unexpected exception."""
        if self._console_roleplay_persistence_task is task:
            self._console_roleplay_persistence_task = None
        if task.cancelled():
            error = None
        else:
            error = task.exception()
            if error is not None:
                failed_plan = (
                    self._console_roleplay_active_plan
                    or self._console_roleplay_pending_plan
                )
                logger.error(
                    "Console roleplay projection persistence task failed "
                    "(session_id={}, generation={}, task_name={}): {!r}",
                    failed_plan.session_id if failed_plan is not None else "unknown",
                    failed_plan.generation if failed_plan is not None else 0,
                    task.get_name(),
                    error,
                )
        if (
            self._console_roleplay_pending_plan is not None
            and self.is_mounted
            and not self._console_roleplay_tearing_down
        ):
            self._start_console_roleplay_persistence_drain()

    def _console_roleplay_unmount_timeout_seconds(self) -> float:
        """Return the bounded configurable drain deadline for this screen."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        raw_timeout = (
            console_config.get(
                "roleplay_refresh_teardown_timeout_seconds",
                CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS,
            )
            if isinstance(console_config, dict)
            else CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        if not 0.01 <= timeout <= 5.0:
            return CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        return timeout

    def _publish_console_roleplay_repair_marker(self) -> None:
        """Publish the latest desired identity on the app, not this screen."""
        generation = (
            int(
                getattr(self.app_instance, "_console_roleplay_repair_generation", 0)
                or 0
            )
            + 1
        )
        self.app_instance._console_roleplay_repair_generation = generation
        self.app_instance._console_roleplay_repair_global_name = (
            self._global_chat_display_name()
        )
        # Textual resumes the uncovered screen before awaiting this screen's
        # unmount hook. A marker published only after the teardown deadline
        # therefore misses that screen's normal ``on_screen_resume`` probe.
        # The loop callback resolves the app's current screen only after pop
        # completes and captures neither this retiring screen nor its store.
        asyncio.get_running_loop().call_later(
            0.1,
            _consume_console_roleplay_repair_for_current_screen,
            self.app,
        )

    async def _teardown_console_roleplay_persistence(self) -> None:
        """Bound screen teardown while app-owned immutable durability continues."""
        self._console_roleplay_tearing_down = True
        task = self._console_roleplay_persistence_task
        if task is None or task.done():
            if (
                self._console_roleplay_pending_plan is not None
                or self._console_roleplay_active_plan is not None
                or (
                    self._console_roleplay_writer_task is not None
                    and not self._console_roleplay_writer_task.done()
                )
            ):
                self._publish_console_roleplay_repair_marker()
            pending_plan = self._console_roleplay_pending_plan
            if pending_plan is not None and self._console_chat_store is not None:
                self._console_chat_store.abandon_roleplay_projection_plan(pending_plan)
            self._console_roleplay_persistence_task = None
            self._console_roleplay_active_plan = None
            self._console_roleplay_pending_plan = None
            self._console_roleplay_writer_task = None
            self._console_roleplay_drain_scheduled = False
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._console_roleplay_unmount_timeout_seconds(),
            )
        except TimeoutError:
            self._publish_console_roleplay_repair_marker()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            self._publish_console_roleplay_repair_marker()
            failed_plan = (
                self._console_roleplay_active_plan
                or self._console_roleplay_pending_plan
            )
            logger.exception(
                "Console roleplay projection drain failed during teardown "
                "(session_id={}, generation={}, task_name={}).",
                failed_plan.session_id if failed_plan is not None else "unknown",
                failed_plan.generation if failed_plan is not None else 0,
                task.get_name(),
            )
        finally:
            pending_plan = self._console_roleplay_pending_plan
            if pending_plan is not None and self._console_chat_store is not None:
                self._console_chat_store.abandon_roleplay_projection_plan(pending_plan)
            self._console_roleplay_persistence_task = None
            self._console_roleplay_active_plan = None
            self._console_roleplay_pending_plan = None
            self._console_roleplay_writer_task = None
            self._console_roleplay_drain_scheduled = False

    def _start_console_roleplay_persistence_drain(self) -> None:
        """Start the sole retained persistence drain when work is pending."""
        if self._console_roleplay_pending_plan is None:
            return
        task = self._console_roleplay_persistence_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._console_roleplay_drain_scheduled:
                return
            self._console_roleplay_drain_scheduled = True
            self.run_worker(
                self._drain_console_roleplay_persistence(),
                exclusive=False,
                group="console-roleplay-refresh",
            )
            return
        self._console_roleplay_drain_scheduled = True
        task = loop.create_task(self._drain_console_roleplay_persistence())
        self._console_roleplay_persistence_task = task
        task.add_done_callback(self._finish_console_roleplay_persistence_task)
        self.run_worker(
            partial(self._await_console_roleplay_persistence_task, task),
            exclusive=False,
            group="console-roleplay-refresh",
        )

    def _dispatch_active_console_roleplay_refresh(
        self,
        *,
        force_persistence: bool = False,
        repair_generation: int = 0,
    ) -> bool:
        """Coalesce refresh writes by active session and effective global name."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return False
        self._start_console_roleplay_persistence_drain()
        global_user_display_name = self._global_chat_display_name()
        refresh_key = (store.active_session_id, global_user_display_name)
        if (
            not force_persistence
            and refresh_key == self._last_console_roleplay_refresh_key
        ):
            return False
        self._last_console_roleplay_refresh_key = refresh_key
        try:
            plan = store.prepare_session_roleplay_projection_refresh(
                refresh_key[0],
                global_default=global_user_display_name,
                force_persistence=force_persistence,
            )
        except KeyError:
            return False
        self._sync_console_identity_surfaces()
        if plan is not None:
            if repair_generation > 0:
                self._console_roleplay_repair_plan = plan
                self._console_roleplay_repair_inflight_generation = repair_generation
            previous_plan = self._console_roleplay_pending_plan
            self._console_roleplay_pending_plan = plan
            if previous_plan is not None and previous_plan is not plan:
                store.abandon_roleplay_projection_plan(previous_plan)
            self._start_console_roleplay_persistence_drain()
        return plan is not None if force_persistence else True

    def _consume_pending_console_roleplay_repair(self) -> bool:
        """Force-persist the latest source projection after abandoned teardown."""
        generation = int(
            getattr(self.app_instance, "_console_roleplay_repair_generation", 0) or 0
        )
        app_consumed = int(
            getattr(
                self.app_instance,
                "_console_roleplay_repair_consumed_generation",
                0,
            )
            or 0
        )
        if generation <= max(self._console_roleplay_repair_generation, app_consumed):
            return False
        if self._console_roleplay_repair_inflight_generation >= generation:
            return False
        self._ensure_console_chat_store()
        self._last_console_roleplay_refresh_key = None
        dispatched = self._dispatch_active_console_roleplay_refresh(
            force_persistence=True,
            repair_generation=generation,
        )
        return dispatched

    def _console_default_durability_state(self) -> ConsoleDefaultDurabilityState:
        """Return the single app-lifetime default recovery holder."""

        state = getattr(
            self.app_instance,
            "console_default_durability_state",
            None,
        )
        if not isinstance(state, ConsoleDefaultDurabilityState):
            state = ConsoleDefaultDurabilityState()
            self.app_instance.console_default_durability_state = state
        if (
            type(
                getattr(self.app_instance, "console_new_chat_default_generation", None)
            )
            is not int
        ):
            self.app_instance.console_new_chat_default_generation = 0
        return state

    def _console_default_readiness(
        self,
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness:
        """Resolve future-chat readiness through the target default chain."""

        app_config = self._provider_readiness_app_config()
        settings = build_target_default_console_session_settings(
            app_config,
            provider,
            model,
        )
        return build_console_settings_readiness(settings, app_config=app_config)

    def _commit_console_settings_submission_live(
        self,
        submission: ConsoleSettingsSubmission,
    ):
        """Revalidate/rebase and commit one exact-origin submission live."""

        owner = self._console_settings_durability_owner()
        admission = owner.try_acquire()
        if admission is None:
            raise ValueError("Application is closing; nothing applied.")
        controller = self._ensure_console_chat_controller()
        try:
            exposed_fields = frozenset(
                field.name for field in submission.draft.field_drafts
            )
            rebased = controller.rebase_console_settings_draft(
                submission.draft,
                provider=submission.draft.settings.provider,
                model=submission.draft.settings.model,
                app_config=self._provider_readiness_app_config(),
                exposed_fields=exposed_fields,
            )
            if submission.surface is ConsoleSettingsSurface.QUICK_POPOVER:
                # Rebasing restores the config-owned endpoint draft. Quick
                # settings may use that endpoint live, but must never turn it
                # into a default-persistence intent.
                rebased = replace(
                    rebased,
                    model_drafts=tuple(
                        replace(model_draft, endpoint_draft=None)
                        for model_draft in rebased.model_drafts
                    ),
                    endpoint_draft=None,
                )
            live_commit = (
                self._ensure_console_chat_store().commit_console_settings_live(
                    replace(submission, draft=rebased)
                )
            )
        except BaseException:
            owner.release(admission)
            raise
        return replace(live_commit, durability_admission=admission)

    def _console_settings_durability_owner(self) -> ConsoleSettingsDurabilityOwner:
        """Return the app-owned settings admission and task registry."""

        app_instance = self.app_instance
        owner = getattr(app_instance, "console_settings_durability_owner", None)
        if not isinstance(owner, ConsoleSettingsDurabilityOwner):
            owner = ConsoleSettingsDurabilityOwner()
            app_instance.console_settings_durability_owner = owner
            app_instance.console_settings_durability_tasks = owner.tasks
        return owner

    def _reserve_console_default_intent(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Synchronously reserve an intent for non-production callers/tests."""

        if submission.action is ConsoleSettingsAction.APPLY_TO_CHAT:
            raise ValueError("Apply to chat does not create a default intent")
        state = self._console_default_durability_state()
        generation = next_console_default_intent_generation(
            state.newest_intent_generation
        )
        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            intent = build_console_default_intent(
                generation=generation,
                action=submission.action,
                provider_config_key=provider_config_key(
                    submission.draft.settings.provider
                ),
                literal_model_id=str(submission.draft.settings.model or ""),
                field_drafts=submission.draft.field_drafts,
                field_mask=submission.default_field_mask,
                endpoint=submission.draft.endpoint_draft,
            )
            if reserve_console_default_intent_generation(
                intent,
                pending_runtime_publisher=(
                    self._accept_console_default_runtime_publication
                ),
            ):
                break
            generation = next_console_default_intent_generation(generation)
        else:
            raise RuntimeError("Console default reservation changed repeatedly")
        self.app_instance.console_default_durability_state = (
            ConsoleDefaultDurabilityState(newest_intent_generation=generation)
        )
        return intent

    async def _reserve_console_default_intent_off_event_loop(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Serialize one reservation with every app-level claim publication."""

        if submission.action is ConsoleSettingsAction.APPLY_TO_CHAT:
            raise ValueError("Apply to chat does not create a default intent")
        async with self._console_default_operation_lock():
            return await self._reserve_console_default_intent_locked(
                submission,
            )

    def _console_default_operation_lock(self) -> asyncio.Lock:
        """Return the one app-lifetime serializer for claim UI operations."""

        app_instance = self.app_instance
        operation_lock = getattr(
            app_instance,
            "console_default_operation_lock",
            None,
        )
        if not isinstance(operation_lock, asyncio.Lock):
            operation_lock = asyncio.Lock()
            app_instance.console_default_operation_lock = operation_lock
        return operation_lock

    async def _reserve_console_default_intent_locked(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Reserve while the caller owns the non-reentrant operation lock."""

        app_instance = self.app_instance
        state = self._console_default_durability_state()
        generation = await asyncio.to_thread(
            next_console_default_intent_generation,
            state.newest_intent_generation,
        )

        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            intent = build_console_default_intent(
                generation=generation,
                action=submission.action,
                provider_config_key=provider_config_key(
                    submission.draft.settings.provider
                ),
                literal_model_id=str(submission.draft.settings.model or ""),
                field_drafts=submission.draft.field_drafts,
                field_mask=submission.default_field_mask,
                endpoint=submission.draft.endpoint_draft,
            )
            (
                preparation,
                cancelled,
            ) = await self._run_console_default_worker_settled(
                prepare_console_default_intent_reservation,
                intent,
            )
            if preparation.reserved:
                app_instance.console_default_durability_state = (
                    ConsoleDefaultDurabilityState(newest_intent_generation=generation)
                )
                if cancelled:
                    raise asyncio.CancelledError
                return intent
            claim = preparation.predecessor_claim
            if claim is None:
                if cancelled:
                    raise asyncio.CancelledError
                generation = await asyncio.to_thread(
                    next_console_default_intent_generation,
                    generation,
                )
                continue
            if cancelled:
                await self._run_console_default_worker_settled(
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise asyncio.CancelledError
            try:
                published = self._accept_console_default_runtime_publication(
                    claim.intent_generation,
                    claim.action,
                    claim.settings_view,
                )
            except Exception:
                published = False
            if not published:
                await self._run_console_default_worker_settled(
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise RuntimeError("Pending default publication was rejected")
            (
                completed,
                cancelled,
            ) = await self._run_console_default_worker_settled(
                complete_console_default_runtime_publication,
                claim,
                successor_intent=intent,
            )
            if completed:
                app_instance.console_default_durability_state = (
                    ConsoleDefaultDurabilityState(newest_intent_generation=generation)
                )
                if cancelled:
                    raise asyncio.CancelledError
                return intent
            if cancelled:
                raise asyncio.CancelledError
            generation = await asyncio.to_thread(
                next_console_default_intent_generation,
                generation,
            )
        raise RuntimeError("Console default reservation changed repeatedly")

    async def _run_console_default_worker_settled(
        self,
        callback: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> tuple[object, bool]:
        """Await a mutating worker to completion before exposing cancellation."""

        worker = asyncio.create_task(
            asyncio.to_thread(partial(callback, *args, **kwargs))
        )
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True

    async def _publish_console_default_outcome_off_event_loop(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Serialize one publication with reservation and recovery claims."""

        async with self._console_default_operation_lock():
            return await self._publish_console_default_outcome_locked(
                intent,
                outcome,
            )

    async def _publish_console_default_outcome_locked(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Publish while the caller owns the non-reentrant operation lock."""

        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            claim, cancelled = await self._run_console_default_worker_settled(
                prepare_console_default_runtime_publication,
                intent,
                outcome,
            )
            if claim is None:
                if cancelled:
                    raise asyncio.CancelledError
                return False
            if not isinstance(claim, ConsoleDefaultRuntimePublicationClaim):
                raise RuntimeError("Default runtime publication claim is invalid")
            if cancelled:
                await self._run_console_default_worker_settled(
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise asyncio.CancelledError
            try:
                published = self._accept_console_default_runtime_publication(
                    claim.intent_generation,
                    claim.action,
                    claim.settings_view,
                )
            except Exception:
                published = False
            if not published:
                await self._run_console_default_worker_settled(
                    abort_console_default_runtime_publication,
                    claim,
                )
                return False
            completed, cancelled = await self._run_console_default_worker_settled(
                complete_console_default_runtime_publication,
                claim,
            )
            if completed:
                if cancelled:
                    raise asyncio.CancelledError
                return True
            if cancelled:
                raise asyncio.CancelledError
        raise RuntimeError("Default runtime publication changed repeatedly")

    def _publish_console_default_outcome(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Publish a fresh runtime mapping once for the newest intent."""

        return publish_console_default_runtime_if_current(
            intent,
            outcome,
            lambda settings_view: self._accept_console_default_runtime_publication(
                intent.generation,
                intent.action,
                settings_view,
            ),
        )

    def _accept_console_default_runtime_publication(
        self,
        intent_generation: int,
        action: ConsoleSettingsAction,
        settings_view: Mapping[str, object],
    ) -> bool:
        """Install one app view while the defaults service fences reservations."""

        state = self._console_default_durability_state()
        if intent_generation != state.newest_intent_generation:
            return False
        try:
            self.app_instance.app_config = settings_view
        except Exception:
            return False
        if state.runtime_published_intent_generation == intent_generation:
            return True
        next_state, accepted = state.accept_runtime_publication(intent_generation)
        if not accepted:
            return False
        self.app_instance.console_default_durability_state = next_state
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            self.app_instance.console_new_chat_default_generation += 1
        return True

    def _launch_console_settings_durability_task(
        self,
        committed: ConsoleSettingsCommittedSubmission,
        default_intent: ConsoleDefaultMutationIntent | None,
    ) -> asyncio.Task[None] | None:
        """Launch post-close durability under the application lifetime."""

        owner = self._console_settings_durability_owner()
        admission = committed.live_commit.durability_admission
        if admission is None:
            admission = owner.try_acquire()
        if admission is None:
            logger.warning(
                "Console settings durability rejected after shutdown admission closed"
            )
            return None
        task = owner.launch(
            admission,
            self._coordinate_console_settings_submission(
                committed,
                default_intent,
            ),
            name=f"console-settings-{committed.submission.submission_id}",
        )

        def report_failure(completed: asyncio.Task[None]) -> None:
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.opt(exception=error).error(
                    "Console settings app-owned durability task failed"
                )

        task.add_done_callback(report_failure)
        return task

    def _dispatch_console_settings_submission(self, result: object) -> None:
        """Refresh live UI and launch durability exactly once per submission."""

        if not isinstance(result, ConsoleSettingsCommittedSubmission):
            return
        owner = self._console_settings_durability_owner()
        admission = result.live_commit.durability_admission
        if admission is None:
            admission = owner.try_acquire()
            if admission is None:
                return
            result = replace(
                result,
                live_commit=replace(
                    result.live_commit,
                    durability_admission=admission,
                ),
            )
        submission_id = result.submission.submission_id
        coordinated = getattr(
            self,
            "_console_settings_coordinated_submission_ids",
            None,
        )
        if not isinstance(coordinated, deque):
            coordinated = deque(maxlen=64)
            self._console_settings_coordinated_submission_ids = coordinated
        if submission_id in coordinated:
            if admission is not None:
                owner.release(admission)
            return
        coordinated.append(submission_id)

        try:
            task = self._launch_console_settings_durability_task(result, None)
        except BaseException:
            owner.release(admission)
            raise
        if task is None:
            return
        store = self._ensure_console_chat_store()
        if store.active_session_id == result.live_commit.session_id:
            self._sync_console_identity_surfaces()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
        self.app_instance.notify("This chat updated", severity="success")

    async def _coordinate_console_settings_submission(
        self,
        committed: ConsoleSettingsCommittedSubmission,
        default_intent: ConsoleDefaultMutationIntent | None,
    ) -> None:
        """Publish independent conversation and default durability outcomes."""

        store = self._ensure_console_chat_store()
        submission = committed.submission
        full_settings_submission = (
            submission.surface is ConsoleSettingsSurface.FULL_SETTINGS
        )
        policy_failure_label = (
            ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS
            if full_settings_submission
            else ConsoleSettingsPolicyFailureLabel.COMPACTION
        )
        display_name_plan: ConsoleRoleplayProjectionPersistencePlan | None = None
        display_name_prepare_failed = False
        if full_settings_submission:
            try:
                _session, display_name_plan = (
                    store.prepare_session_user_display_name_override_for_commit(
                        committed.live_commit,
                        submission.user_display_name_override,
                        global_default=self._global_chat_display_name(),
                    )
                )
            except Exception:
                logger.exception(
                    "Console settings display-name preparation failed (submission_id={})",
                    submission.submission_id,
                )
                display_name_prepare_failed = True

        async def persist_display_name() -> None:
            if not full_settings_submission:
                return
            if display_name_prepare_failed:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
                return
            if display_name_plan is None:
                return
            try:
                result = await store.persist_roleplay_projection_plan_serialized(
                    display_name_plan,
                )
            except Exception:
                logger.exception(
                    "Console settings display-name persistence failed (submission_id={})",
                    submission.submission_id,
                )
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
                return
            if result is None:
                return
            accepted = store.accept_roleplay_projection_persistence_result(result)
            if not accepted:
                return
            if store.active_session_id == display_name_plan.session_id:
                self._sync_console_identity_surfaces()
            if not result.persisted:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )

        async def persist_conversation() -> None:
            try:
                await store.persist_console_settings_commit_serialized(
                    committed.live_commit,
                    policy_failure_label=policy_failure_label,
                )
            except Exception as exc:
                from .settings_diagnostics import log_settings_failure

                log_settings_failure(
                    "conversation_persist",
                    exc,
                    session_id=committed.live_commit.session_id,
                    submission_id=submission.submission_id,
                )
            finally:
                self._sync_console_settings_recovery_surfaces()

        async def persist_default() -> None:
            intent = default_intent
            if (
                intent is None
                and submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
            ):
                return
            if intent is None:
                try:
                    intent = await self._reserve_console_default_intent_off_event_loop(
                        submission
                    )
                except Exception as exc:
                    from .settings_diagnostics import log_settings_failure

                    log_settings_failure(
                        "default_reserve",
                        exc,
                        session_id=committed.live_commit.session_id,
                        submission_id=submission.submission_id,
                    )
                    self._sync_console_settings_recovery_surfaces()
                    recovery = self._console_default_durability_state()
                    recovery_copy = (
                        "the previous default recovery remains available."
                        if recovery.recovery_intent is not None
                        else "try this default action again."
                    )
                    self.app_instance.notify(
                        "Default not saved for "
                        f"{provider_config_key(submission.draft.settings.provider)}/"
                        f"{submission.draft.settings.model}; {recovery_copy}",
                        severity="warning",
                    )
                    return
            try:
                outcome = await asyncio.to_thread(
                    apply_console_default_intent,
                    intent,
                )
            except Exception as exc:
                from .settings_diagnostics import log_settings_failure

                log_settings_failure(
                    "default_apply",
                    exc,
                    session_id=committed.live_commit.session_id,
                    submission_id=submission.submission_id,
                    generation=intent.generation,
                )
                self._record_console_default_failure(
                    intent,
                    ConsoleDefaultSavePhase.BEFORE_REPLACE,
                )
                return
            try:
                published = await self._publish_console_default_outcome_off_event_loop(
                    intent,
                    outcome,
                )
            except Exception as exc:
                from .settings_diagnostics import log_settings_failure

                log_settings_failure(
                    "default_publish",
                    exc,
                    session_id=committed.live_commit.session_id,
                    submission_id=submission.submission_id,
                    generation=intent.generation,
                )
                published = False
            if published:
                scope = (
                    "Eligible new-chat default saved"
                    if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
                    else "Model profile default saved"
                )
                self.app_instance.notify(
                    f"{scope}: {intent.provider_config_key}/{intent.literal_model_id}",
                    severity="success",
                )
            elif outcome.failure_phase is not None:
                self._record_console_default_failure(
                    intent,
                    outcome.failure_phase,
                )
            elif outcome.runtime_published and outcome.settings_view is not None:
                self._record_console_default_failure(
                    intent,
                    ConsoleDefaultSavePhase.CACHE_PUBLICATION,
                )

        display_name_task = asyncio.create_task(persist_display_name())
        if display_name_plan is not None:
            display_name_task.add_done_callback(
                lambda _task: store.abandon_roleplay_projection_plan(display_name_plan)
            )
        await asyncio.gather(
            persist_conversation(),
            persist_default(),
            display_name_task,
        )

    def _record_console_default_failure(
        self,
        intent: ConsoleDefaultMutationIntent,
        phase: ConsoleDefaultSavePhase,
    ) -> None:
        """Retain only a current app-global recovery record."""

        state = self._console_default_durability_state()
        if state.newest_intent_generation != intent.generation:
            return
        self.app_instance.console_default_durability_state = (
            ConsoleDefaultDurabilityState(
                newest_intent_generation=intent.generation,
                recovery_intent=intent,
                failure_phase=phase,
                runtime_published_intent_generation=(
                    state.runtime_published_intent_generation
                ),
            )
        )
        self._sync_console_settings_recovery_surfaces()

    async def _handle_console_default_recovery(
        self,
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        """Admit and execute one generation-bound app-global recovery."""

        state = self._console_default_durability_state()
        if not isinstance(request, ConsoleDefaultRecoveryRequest):
            return state
        intent = state.recovery_intent
        if (
            intent is None
            or request.intent_generation != state.newest_intent_generation
        ):
            return state
        allowed_actions = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: {
                ConsoleDefaultRecoveryAction.RETRY_SAVE,
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            },
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: {
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            },
        }
        if request.action not in allowed_actions.get(state.failure_phase, set()):
            return state
        owner = self._console_settings_durability_owner()
        admission = owner.try_acquire()
        if admission is None:
            return state
        task = owner.launch(
            admission,
            self._run_console_default_recovery(request),
            name=f"console-default-recovery-{request.intent_generation}",
        )
        return await asyncio.shield(task)

    async def _run_console_default_recovery(
        self,
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        """Run one admitted recovery under generation/phase single-flight."""

        state = self._console_default_durability_state()
        if not isinstance(request, ConsoleDefaultRecoveryRequest):
            return state
        intent = state.recovery_intent
        if (
            intent is None
            or request.intent_generation != state.newest_intent_generation
        ):
            return state
        allowed_actions = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: {
                ConsoleDefaultRecoveryAction.RETRY_SAVE,
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            },
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: {
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            },
        }
        failure_phase = state.failure_phase
        if request.action not in allowed_actions.get(failure_phase, set()):
            return state
        inflight = getattr(
            self.app_instance,
            "console_default_recovery_inflight",
            None,
        )
        if not isinstance(inflight, set):
            inflight = set()
            self.app_instance.console_default_recovery_inflight = inflight
        assert isinstance(failure_phase, ConsoleDefaultSavePhase)
        flight_key = (
            request.intent_generation,
            failure_phase.value,
        )
        if flight_key in inflight:
            return state
        inflight.add(flight_key)
        try:
            if request.action in {
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            }:
                state = ConsoleDefaultDurabilityState(
                    newest_intent_generation=state.newest_intent_generation,
                    runtime_published_intent_generation=(
                        state.runtime_published_intent_generation
                    ),
                )
                self.app_instance.console_default_durability_state = state
                self._sync_console_settings_recovery_surfaces()
                return state
            if (
                request.action is ConsoleDefaultRecoveryAction.RETRY_SAVE
                and failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
            ):
                outcome = await asyncio.to_thread(
                    apply_console_default_intent,
                    intent,
                )
            elif (
                request.action is ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
                and failure_phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION
            ):
                refresh = await asyncio.to_thread(
                    refresh_console_runtime_after_saved_default
                )
                outcome = ConsoleDefaultMutationOutcome(
                    intent_generation=intent.generation,
                    file_replaced=True,
                    runtime_published=refresh.published,
                    settings_view=refresh.settings_view,
                    failure_phase=(
                        None
                        if refresh.published
                        else ConsoleDefaultSavePhase.CACHE_PUBLICATION
                    ),
                )
            else:
                return state
        except Exception:
            logger.exception("Console default recovery failed")
            current = self._console_default_durability_state()
            if (
                current.recovery_intent == intent
                and current.failure_phase is failure_phase
            ):
                self._record_console_default_failure(intent, failure_phase)
            return self._console_default_durability_state()
        finally:
            inflight.discard(flight_key)
        current = self._console_default_durability_state()
        if (
            current.recovery_intent != intent
            or current.failure_phase is not failure_phase
        ):
            return current
        try:
            published = await self._publish_console_default_outcome_off_event_loop(
                intent,
                outcome,
            )
        except Exception:
            logger.exception("Console default recovery publication failed")
            published = False
        if not published:
            phase = outcome.failure_phase or ConsoleDefaultSavePhase.CACHE_PUBLICATION
            self._record_console_default_failure(intent, phase)
        self._sync_console_settings_recovery_surfaces()
        return self._console_default_durability_state()
