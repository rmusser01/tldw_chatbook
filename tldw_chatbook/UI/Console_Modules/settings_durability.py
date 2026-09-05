"""Console settings submission durability and app-lifetime default recovery."""

from __future__ import annotations

from typing import Any
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
    ConsoleSettingsReadiness,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
)
from ...Chat.console_chat_store import (
    ConsoleRoleplayProjectionPersistencePlan,
    ConsoleSettingsPolicyFailureLabel,
)
from ...Chat.provider_readiness import provider_config_key


logger = logger.bind(module="ChatScreen")
_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS = 8


class ConsoleSettingsDurabilityController:
    """Own console settings submission durability and app-lifetime default recovery.

    Dependencies, including the app owner, resolve through named callables
    at use time. The controller owns no DOM or screen handle.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _global_chat_display_name: Callable[..., Any],
        _provider_readiness_app_config: Callable[..., Any],
        _sync_console_identity_surfaces: Callable[..., Any],
        _sync_console_settings_recovery_surfaces: Callable[..., Any],
        _sync_native_console_chat_ui: Callable[..., Any],
        run_worker: Callable[..., Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._global_chat_display_name = _global_chat_display_name
        self._provider_readiness_app_config = _provider_readiness_app_config
        self._sync_console_identity_surfaces = _sync_console_identity_surfaces
        self._sync_console_settings_recovery_surfaces = (
            _sync_console_settings_recovery_surfaces
        )
        self._sync_native_console_chat_ui = _sync_native_console_chat_ui
        self.run_worker = run_worker
        self._console_settings_coordinated_submission_ids = deque(maxlen=64)

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

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
            except Exception:
                logger.exception("Console settings conversation persistence failed")
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
                except Exception:
                    logger.exception("Console default reservation failed")
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
            except Exception:
                logger.exception("Console default persistence failed")
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
            except Exception:
                logger.exception("Console default runtime publication failed")
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

        await asyncio.gather(
            persist_conversation(),
            persist_default(),
            persist_display_name(),
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
