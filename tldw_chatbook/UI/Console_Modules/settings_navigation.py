"""Conversation settings modal transfer and credential-return ownership."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from dataclasses import replace
import asyncio
from loguru import logger
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffClaim,
    PendingHandoffStore,
)
from ..Navigation.conversation_settings_navigation import (
    ConsoleSettingsReturnTarget,
    ConversationSettingsReturnIntent,
    ProviderSettingsNavigationTarget,
)
from ...Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsTransfer,
)
from ...Chat.console_provider_gateway import AuxiliaryCompletionRequest
from ...Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from ...Chat.provider_test_evidence import (
    ConsoleGenerationTestRequest,
    ProviderGenerationProbeResult,
)
from ...Constants import TAB_SETTINGS


if TYPE_CHECKING:
    from ...Chat.console_session_settings import ConsoleSessionSettings
    from ...Chat.console_context_policy import ConsoleContextPolicyOverrides
    from ...Widgets.Console.console_settings_modal import (
        ConsoleSettingsCredentialRequest,
        ConsoleSettingsDraftSnapshot,
        ConsoleSettingsModal,
    )


def _conversation_settings_modal_module():
    """Load the settings modal only when this workflow starts."""
    from ...Widgets.Console import console_settings_modal

    return console_settings_modal


logger = logger.bind(module="ChatScreen")


class ConsoleSettingsNavigationController:
    """Own conversation settings modal transfer and credential-return ownership.

    Dependencies, including the app owner, resolve through named callables
    at use time. The controller owns no DOM or screen handle.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _build_console_provider_selection_for_settings: Callable[..., Any],
        _commit_console_settings_submission_live: Callable[..., Any],
        _console_context_control_state_for_session: Callable[..., Any],
        _console_default_durability_state: Callable[..., Any],
        _console_default_readiness: Callable[..., Any],
        _console_run_active: Callable[..., Any],
        _console_settings_context_estimate_for_session: Callable[..., Any],
        _dispatch_console_settings_submission: Callable[..., Any],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _ensure_console_provider_gateway: Callable[..., Any],
        _global_chat_display_name: Callable[..., Any],
        _handle_console_default_recovery: Callable[..., Any],
        _mount_conversation_settings_return_status: Callable[..., Any],
        _owns_console_screen_stack: Callable[..., Any],
        _provider_readiness_app_config: Callable[..., Any],
        _providers_models_for_console_settings: Callable[..., Any],
        _sync_native_console_chat_ui: Callable[..., Any],
        _test_console_connection: Callable[..., Any],
        call_after_refresh: Callable[..., Any],
        notify: Callable[..., Any],
        pop_screen: Callable[..., Any],
        post_message: Callable[..., Any],
        push_screen: Callable[..., Any],
        run_worker: Callable[..., Any],
        is_mounted_accessor: Callable[[], Any],
        screen_stack_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._build_console_provider_selection_for_settings = (
            _build_console_provider_selection_for_settings
        )
        self._commit_console_settings_submission_live = (
            _commit_console_settings_submission_live
        )
        self._console_context_control_state_for_session = (
            _console_context_control_state_for_session
        )
        self._console_default_durability_state = _console_default_durability_state
        self._console_default_readiness = _console_default_readiness
        self._console_run_active = _console_run_active
        self._console_settings_context_estimate_for_session = (
            _console_settings_context_estimate_for_session
        )
        self._dispatch_console_settings_submission = (
            _dispatch_console_settings_submission
        )
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._ensure_console_provider_gateway = _ensure_console_provider_gateway
        self._global_chat_display_name = _global_chat_display_name
        self._handle_console_default_recovery = _handle_console_default_recovery
        self._mount_conversation_settings_return_status = (
            _mount_conversation_settings_return_status
        )
        self._owns_console_screen_stack = _owns_console_screen_stack
        self._provider_readiness_app_config = _provider_readiness_app_config
        self._providers_models_for_console_settings = (
            _providers_models_for_console_settings
        )
        self._sync_native_console_chat_ui = _sync_native_console_chat_ui
        self._test_console_connection = _test_console_connection
        self.call_after_refresh = call_after_refresh
        self.notify = notify
        self.pop_screen = pop_screen
        self.post_message = post_message
        self.push_screen = push_screen
        self.run_worker = run_worker
        self.is_mounted_accessor = is_mounted_accessor
        self.screen_stack_accessor = screen_stack_accessor
        self._suspended_conversation_settings = None
        self._suspended_conversation_settings_token = None
        self._next_suspended_conversation_settings_token = 0
        self._pending_conversation_settings_return_claim = None
        self._pending_conversation_settings_return_target = None
        self._conversation_settings_return_restore_in_progress = False

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    @property
    def is_mounted(self) -> Any:
        return self.is_mounted_accessor()

    @property
    def screen_stack(self) -> Any:
        return self.screen_stack_accessor()

    async def _test_console_generation(
        self,
        session_id: str,
        request: ConsoleGenerationTestRequest,
    ) -> ProviderGenerationProbeResult:
        """Run one isolated, bounded completion against a validated modal draft."""
        if type(request) is not ConsoleGenerationTestRequest:
            return ProviderGenerationProbeResult("failed", "bad_request")
        try:
            selection = self._build_console_provider_selection_for_settings(
                session_id, request.settings
            )
            gateway = self._ensure_console_provider_gateway()
            async with asyncio.timeout(20.0):
                resolution = await gateway.resolve_for_send(selection)
                if not resolution.ready:
                    return ProviderGenerationProbeResult("failed", "bad_request")
                test_resolution = replace(
                    resolution,
                    streaming=False,
                    reasoning_effort=None,
                    reasoning_summary=None,
                    verbosity=None,
                    thinking_effort=None,
                    thinking_budget_tokens=None,
                    request_timeout=15.0,
                    request_retries=0,
                    request_retry_delay=0.0,
                )
                auxiliary_request = AuxiliaryCompletionRequest(
                    resolution=test_resolution,
                    messages=(
                        {"role": "user", "content": "Reply with one short token."},
                    ),
                    response_format=None,
                    max_output_tokens=1,
                )
                await gateway.complete_auxiliary(auxiliary_request)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError):
            return ProviderGenerationProbeResult("failed", "timeout")
        except ChatAuthenticationError:
            return ProviderGenerationProbeResult("failed", "authentication")
        except ChatRateLimitError:
            return ProviderGenerationProbeResult("failed", "rate_limit")
        except (ChatBadRequestError, ChatConfigurationError, ValueError, TypeError):
            return ProviderGenerationProbeResult("failed", "bad_request")
        except (ConnectionError, OSError):
            return ProviderGenerationProbeResult("failed", "connection_error")
        except ChatProviderError as exc:
            category = {
                400: "bad_request",
                401: "authentication",
                403: "authentication",
                408: "timeout",
                504: "timeout",
                429: "rate_limit",
                503: "connection_error",
            }.get(exc.status_code, "provider_error")
            return ProviderGenerationProbeResult("failed", category)
        except Exception:
            return ProviderGenerationProbeResult("failed", "provider_error")
        return ProviderGenerationProbeResult("succeeded")

    async def _open_console_settings(
        self,
        *,
        focus_model: bool = False,
        focus_context: bool = False,
        transfer: ConsoleSettingsTransfer | None = None,
        suspended_draft: "ConsoleSettingsDraftSnapshot | None" = None,
        _pre_push_guard: Callable[[], bool] | None = None,
        _suspended_owner_token: int | None = None,
        _on_transfer_committed: Callable[[], bool] | None = None,
    ) -> bool:
        """Open Console session settings for the active native session."""
        controller = self._ensure_console_chat_controller()
        store = self._ensure_console_chat_store()
        if transfer is None:
            session_id = store.active_session_id
            if session_id is None:
                return False
            origin = store.capture_console_settings_origin(session_id)
            settings = (
                suspended_draft.settings
                if suspended_draft is not None
                else store.session_settings(session_id)
            )
            if settings is None:
                return False
            initial_draft = self._console_settings_initial_draft(
                settings,
                suspended_draft.context_policy_overrides
                if suspended_draft is not None
                else store.session_context_policy_overrides(session_id),
                exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
            )
        else:
            origin = transfer.origin
            session_id = origin.session_id
            settings = transfer.draft.settings
            initial_draft = transfer.draft
        try:
            display_name = store.session_user_display_name_override(session_id)
        except KeyError:
            return False
        active_provider = settings.provider
        active_model = settings.model
        if suspended_draft is not None:
            raw_provider = suspended_draft.raw_values.get("console-settings-provider")
            if type(raw_provider) is str:
                active_provider = raw_provider
            active_model = suspended_draft.provider_model_drafts.get(
                active_provider,
                settings.model if active_provider == settings.provider else None,
            )
        effective_thinking_policy = (
            await controller.effective_thinking_history_policy_for_session(session_id)
        )
        context_estimate = self._console_settings_context_estimate_for_session(
            session_id,
            settings=settings,
        )
        context_state = self._console_context_control_state_for_session(
            session_id,
            estimate=context_estimate,
            settings=settings,
            thinking_history_effective_policy=effective_thinking_policy,
        )
        providers_models = await self._providers_models_for_console_settings(
            active_provider,
            current_model=active_model,
        )
        active_run = self._console_run_active()

        modal_contract = _conversation_settings_modal_module()
        modal = modal_contract.ConsoleSettingsModal(
            settings=settings,
            origin=origin,
            initial_draft=initial_draft,
            transfer=transfer,
            user_display_name_override=display_name,
            global_user_display_name=self._global_chat_display_name(),
            app_config=self._provider_readiness_app_config(),
            providers_models=providers_models,
            context_estimate=context_estimate,
            context_state=context_state,
            can_save=(
                controller.run_state_for(session_id).is_send_allowed and not active_run
            ),
            active_run=active_run,
            focus_model=focus_model,
            focus_context=focus_context,
            reset_current_memory=lambda: controller.reset_active_context_memory(
                session_id
            ),
            undo_current_memory_reset=controller.undo_context_memory_reset,
            reset_all_memories=lambda: controller.reset_all_context_memories(
                session_id
            ),
            compact_now=lambda: controller.compact_context_now(session_id),
            draft_rebaser=controller.rebase_console_settings_draft,
            live_committer=self._commit_console_settings_submission_live,
            default_readiness_resolver=self._console_default_readiness,
            default_durability_state=self._console_default_durability_state(),
            default_recovery_handler=self._handle_console_default_recovery,
            suspended_draft=suspended_draft,
            connection_tester=self._test_console_connection,
            generation_tester=lambda request: self._test_console_generation(
                session_id, request
            ),
        )

        transfer_revoked = False

        def apply_origin_result(result) -> None:
            if transfer_revoked:
                return
            if isinstance(result, modal_contract.ConsoleSettingsCredentialRequest):
                self._stage_console_settings_credential_request(
                    result,
                    session_id=session_id,
                )
                return
            self._dispatch_console_settings_submission(result)

        transfer_outcome: bool | None = None

        def report_transfer_committed() -> bool:
            """Commit exact modal ownership once without retaining its draft."""

            nonlocal transfer_outcome, transfer_revoked
            if transfer_outcome is not None:
                return transfer_outcome
            try:
                transfer_outcome = (
                    True
                    if _on_transfer_committed is None
                    else _on_transfer_committed() is True
                )
            except Exception:
                logger.error("Unable to commit Conversation settings modal transfer")
                transfer_outcome = False
            if not transfer_outcome:
                # The source snapshot remains authoritative. The covered-modal
                # cancellation path cannot safely pop through a newer overlay,
                # so revoke this tentative modal's result and draft ownership.
                transfer_revoked = True
                modal._suspended_draft = None
                modal.disabled = True
            return transfer_outcome

        if _pre_push_guard is not None and not _pre_push_guard():
            return False
        try:
            await self.push_screen(modal, callback=apply_origin_result)
        except asyncio.CancelledError:
            removed = await self._unwind_failed_console_settings_modal(modal)
            if (
                not removed
                and self._console_settings_modal_is_on_stack(modal)
                and suspended_draft is not None
                and _suspended_owner_token is not None
                and getattr(self, "_suspended_conversation_settings", None)
                is suspended_draft
                and getattr(self, "_suspended_conversation_settings_token", None)
                == _suspended_owner_token
            ):
                committed = report_transfer_committed()
                if committed and (
                    getattr(self, "_suspended_conversation_settings", None)
                    is suspended_draft
                    and getattr(
                        self,
                        "_suspended_conversation_settings_token",
                        None,
                    )
                    == _suspended_owner_token
                ):
                    self._suspended_conversation_settings = None
                    self._suspended_conversation_settings_token = None
            raise
        except Exception:
            removed = await self._unwind_failed_console_settings_modal(modal)
            if not removed and self._console_settings_modal_is_on_stack(modal):
                # A concurrently covered modal still owns the only live
                # draft. Report ownership so a suspended snapshot is not
                # retained as a second owner.
                if report_transfer_committed():
                    return True
                return False
            return False
        if report_transfer_committed():
            return True
        await self._unwind_failed_console_settings_modal(modal)
        return False

    def _console_settings_modal_is_on_stack(
        self,
        modal: "ConsoleSettingsModal",
    ) -> bool:
        """Return whether the exact pushed modal remains anywhere on the stack."""
        try:
            return any(screen is modal for screen in self.screen_stack)
        except Exception:
            return False

    async def _unwind_failed_console_settings_modal(
        self,
        modal: "ConsoleSettingsModal",
    ) -> bool:
        """Pop only the exact failed modal when it still owns the stack top."""
        try:
            stack = self.screen_stack
            if not stack or stack[-1] is not modal:
                return not any(screen is modal for screen in stack)
            pop_result = self.pop_screen()
        except Exception:
            return not self._console_settings_modal_is_on_stack(modal)
        try:
            await pop_result
        except asyncio.CancelledError:
            # Preserve the original cancellation from the failed mount; the
            # synchronous stack re-check below remains the ownership truth.
            pass
        except Exception:
            # Textual removes the exact top synchronously before its returned
            # AwaitComplete performs unmount work, so ownership may already
            # be repaired even when that later work reports an error.
            pass
        return not self._console_settings_modal_is_on_stack(modal)

    def _stage_console_settings_credential_request(
        self,
        request: "ConsoleSettingsCredentialRequest",
        *,
        session_id: str,
    ) -> None:
        """Suspend a draft and navigate to canonical API-key configuration.

        The raw modal state stays solely in this screen's native snapshot.
        The return handoff and Settings route contain only typed restoration
        coordinates, never prompt, prefill, endpoint, or credential content.
        """
        store = self._ensure_console_chat_store()
        try:
            settings_revision = store.session_settings_revision(session_id)
        except KeyError:
            return
        # Validate provider/model before touching the single-slot return
        # channel. The final target only substitutes the positive opaque
        # handoff revision, so it cannot broaden this validated route.
        ProviderSettingsNavigationTarget(
            category="providers-models",
            provider=request.provider,
            model=request.model,
            field="api_key",
            return_revision=1,
        )
        intent = ConversationSettingsReturnIntent(
            session_id=session_id,
            settings_revision=settings_revision,
            active_view=request.snapshot.active_view,
            focus_control_id=request.snapshot.focus_control_id,
        )
        handoff_revision = self.app_instance.pending_handoffs.stage(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN,
            intent,
        )
        target = ProviderSettingsNavigationTarget(
            category="providers-models",
            provider=request.provider,
            model=request.model,
            field="api_key",
            return_revision=handoff_revision,
        )
        request_token = (
            getattr(self, "_next_suspended_conversation_settings_token", 0) + 1
        )
        self._next_suspended_conversation_settings_token = request_token
        self._suspended_conversation_settings = request.snapshot
        self._suspended_conversation_settings_token = request_token

        def settle_navigation(succeeded: bool) -> None:
            """Repair the source only when the existing route did not leave it."""
            if succeeded:
                return
            discarded = self.app_instance.pending_handoffs.discard_pending_exact(
                HandoffChannel.CONVERSATION_SETTINGS_RETURN,
                handoff_revision,
                intent,
            )
            if not discarded:
                return
            if (
                getattr(self, "_suspended_conversation_settings_token", None)
                != request_token
            ):
                return
            if self._owns_console_screen_stack():
                self.run_worker(
                    self._reopen_suspended_console_settings(
                        request_token,
                        session_id=session_id,
                        settings_revision=settings_revision,
                    ),
                    exclusive=False,
                )

        navigation = NavigateToScreen(
            TAB_SETTINGS,
            target.to_context(),
            on_completion=settle_navigation,
        )
        if self.post_message(navigation) is False:
            navigation.report_completion(False)

    async def _reopen_suspended_console_settings(
        self,
        request_token: int,
        *,
        session_id: str,
        settings_revision: int,
        _on_transfer_committed: Callable[[], bool] | None = None,
    ) -> bool:
        """Transfer the exact retained draft to a restored modal when safe."""
        modal_contract = _conversation_settings_modal_module()
        snapshot = getattr(self, "_suspended_conversation_settings", None)
        store = self._ensure_console_chat_store()

        def may_push() -> bool:
            """Revalidate exact source, token, snapshot, and session ownership."""
            if (
                not isinstance(snapshot, modal_contract.ConsoleSettingsDraftSnapshot)
                or getattr(self, "_suspended_conversation_settings", None)
                is not snapshot
                or getattr(self, "_suspended_conversation_settings_token", None)
                != request_token
                or store.active_session_id != session_id
                or not self._owns_console_screen_stack()
            ):
                return False
            try:
                return store.session_settings_revision(session_id) == settings_revision
            except KeyError:
                return False

        if not may_push():
            return False
        try:
            reopened = await self._open_console_settings(
                suspended_draft=snapshot,
                _pre_push_guard=may_push,
                _suspended_owner_token=request_token,
                _on_transfer_committed=_on_transfer_committed,
            )
        except Exception:
            return False
        if (
            reopened
            and getattr(self, "_suspended_conversation_settings", None) is snapshot
            and getattr(self, "_suspended_conversation_settings_token", None)
            == request_token
        ):
            self._suspended_conversation_settings = None
            self._suspended_conversation_settings_token = None
        return reopened

    def _claim_conversation_settings_return(
        self,
        handoffs: PendingHandoffStore,
        target: ConsoleSettingsReturnTarget,
    ) -> bool:
        """Claim only the exact retained retry coordinate."""

        revision_status = handoffs.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN,
            target.return_revision,
        )
        if revision_status == "in_flight":
            return False
        if revision_status == "superseded":
            if self._clear_conversation_settings_return_target(target):
                self._notify_conversation_settings_return(
                    "This return was superseded by a newer request. "
                    "Open Conversation settings again."
                )
            return False
        if revision_status == "settled":
            self._discard_conversation_settings_return(
                target,
                "Conversation settings return is no longer available. "
                "Open Conversation settings again.",
            )
            return False
        claim = handoffs.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
        if claim is None:
            return False
        if (
            claim.revision != target.return_revision
            or type(claim.value) is not ConversationSettingsReturnIntent
        ):
            handoffs.release(claim)
            if self._clear_conversation_settings_return_target(target):
                self._notify_conversation_settings_return(
                    "This return was superseded by a newer request. "
                    "Open Conversation settings again."
                )
            return False
        intent = claim.value
        if (
            target.session_id != intent.session_id
            or target.settings_revision != intent.settings_revision
            or target.active_view != intent.active_view
            or target.focus_control_id != intent.focus_control_id
        ):
            self._reject_conversation_settings_return(
                handoffs,
                claim,
                target,
                "Conversation settings return was stale. The draft was not restored.",
            )
            return False
        rejection_copy = self._conversation_settings_return_rejection_copy(target)
        if rejection_copy is not None:
            self._reject_conversation_settings_return(
                handoffs,
                claim,
                target,
                rejection_copy,
            )
            return False
        self._pending_conversation_settings_return_claim = claim
        return True

    def _conversation_settings_return_rejection_copy(
        self,
        target: ConsoleSettingsReturnTarget,
    ) -> str | None:
        """Return fixed terminal copy, or ``None`` when restoration is safe."""

        store = self._ensure_console_chat_store()
        try:
            if store.session_is_ephemeral(target.session_id):
                return (
                    "The temporary conversation is no longer available. Its "
                    "Conversation settings draft was not restored."
                )
            if (
                store.session_settings_revision(target.session_id)
                != target.settings_revision
            ):
                return (
                    "Conversation settings changed while credentials were open. "
                    "The earlier draft was not restored."
                )
        except KeyError:
            return (
                "The original conversation closed. Its Conversation settings draft "
                "was not restored."
            )
        snapshot = getattr(self, "_suspended_conversation_settings", None)
        token = getattr(self, "_suspended_conversation_settings_token", None)
        if not (
            isinstance(
                snapshot,
                _conversation_settings_modal_module().ConsoleSettingsDraftSnapshot,
            )
            and type(token) is int
            and token > 0
            and snapshot.active_view == target.active_view
            and snapshot.focus_control_id == target.focus_control_id
        ):
            return "Conversation settings return was stale. The draft was not restored."
        return None

    def _notify_conversation_settings_return(self, message: str) -> None:
        """Show only allowlisted return recovery text."""

        try:
            self.notify(message, severity="warning")
        except Exception:
            logger.debug("Unable to show Conversation settings recovery notice")

    def _clear_conversation_settings_return_target(
        self,
        target: ConsoleSettingsReturnTarget,
        *,
        discard_snapshot: bool = False,
    ) -> bool:
        """Clear only the exact screen-local return captured by one consumer."""

        if self._pending_conversation_settings_return_target is not target:
            return False
        self._pending_conversation_settings_return_target = None
        if discard_snapshot:
            self._suspended_conversation_settings = None
            self._suspended_conversation_settings_token = None
        return True

    def _discard_conversation_settings_return(
        self,
        target: ConsoleSettingsReturnTarget,
        message: str,
    ) -> None:
        """Clear obsolete private state after a terminal unclaimed route."""

        if self._clear_conversation_settings_return_target(
            target,
            discard_snapshot=True,
        ):
            self._notify_conversation_settings_return(message)

    def _reject_conversation_settings_return(
        self,
        handoffs: PendingHandoffStore,
        claim: HandoffClaim[ConversationSettingsReturnIntent],
        target: ConsoleSettingsReturnTarget,
        message: str,
    ) -> None:
        """Settle one terminal rejection and discard its obsolete private draft."""

        handoffs.acknowledge(claim)
        if self._pending_conversation_settings_return_claim is claim:
            self._pending_conversation_settings_return_claim = None
        if self._clear_conversation_settings_return_target(
            target,
            discard_snapshot=True,
        ):
            self._notify_conversation_settings_return(message)

    def _consume_pending_conversation_settings_return(self) -> None:
        """Schedule the mounted restore once, retaining transient failures."""

        if (
            self._conversation_settings_return_restore_in_progress
            or self._pending_conversation_settings_return_target is None
            or not self.is_mounted
            or not self._owns_console_screen_stack()
        ):
            return
        self._conversation_settings_return_restore_in_progress = True
        self.run_worker(
            self._restore_claimed_conversation_settings_return(),
            exclusive=False,
            group="conversation-settings-return",
        )

    async def _restore_claimed_conversation_settings_return(self) -> None:
        """Restore the exact suspended draft, then acknowledge its handoff."""

        claim: HandoffClaim[ConversationSettingsReturnIntent] | None = None
        target = self._pending_conversation_settings_return_target
        handoffs = getattr(self.app_instance, "pending_handoffs", None)
        prior_active_session_id: str | None = None
        switched_session = False
        selected_active_session_epoch: int | None = None
        transfer_committed = False

        def commit_transfer() -> bool:
            """Commit exact handoff ownership once at the modal transfer edge."""

            nonlocal claim, transfer_committed
            if transfer_committed:
                return True
            transferred_claim = claim
            if transferred_claim is None or not isinstance(
                handoffs,
                PendingHandoffStore,
            ):
                return False
            try:
                settled = handoffs.settle_transferred_claim(transferred_claim)
            except Exception:
                logger.error(
                    "Conversation settings return atomic transfer settlement "
                    "raised for revision {}",
                    transferred_claim.revision,
                )
                return False
            if not settled:
                logger.error(
                    "Conversation settings return atomic transfer settlement "
                    "rejected revision {}",
                    transferred_claim.revision,
                )
                return False
            transfer_committed = True
            if self._pending_conversation_settings_return_claim is transferred_claim:
                self._pending_conversation_settings_return_claim = None
            self._clear_conversation_settings_return_target(target)
            claim = None
            return True

        async def restore_prior_session_if_still_owned() -> None:
            """Undo only this worker's exact active-session transition."""

            if (
                transfer_committed
                or not switched_session
                or prior_active_session_id is None
                or selected_active_session_epoch is None
            ):
                return
            store = self._ensure_console_chat_store()
            if (
                store.active_session_id != target.session_id
                or store.active_session_epoch() != selected_active_session_epoch
            ):
                return
            store.switch_session(prior_active_session_id)
            await self._sync_native_console_chat_ui()

        try:
            if target is None or not isinstance(handoffs, PendingHandoffStore):
                return
            if not self.is_mounted or not self._owns_console_screen_stack():
                return
            if not self._claim_conversation_settings_return(handoffs, target):
                return
            claim = self._pending_conversation_settings_return_claim
            if claim is None:
                return
            if not handoffs.is_current_claim(claim):
                handoffs.release(claim)
                if self._clear_conversation_settings_return_target(target):
                    self._notify_conversation_settings_return(
                        "This return was superseded by a newer request. "
                        "Open Conversation settings again."
                    )
                return
            rejection_copy = self._conversation_settings_return_rejection_copy(target)
            if rejection_copy is not None:
                self._reject_conversation_settings_return(
                    handoffs,
                    claim,
                    target,
                    rejection_copy,
                )
                return
            snapshot = self._suspended_conversation_settings
            token = self._suspended_conversation_settings_token
            if (
                not isinstance(
                    snapshot,
                    _conversation_settings_modal_module().ConsoleSettingsDraftSnapshot,
                )
                or type(token) is not int
            ):
                self._reject_conversation_settings_return(
                    handoffs,
                    claim,
                    target,
                    "Conversation settings return was stale. The draft was not restored.",
                )
                return
            store = self._ensure_console_chat_store()
            prior_active_session_id = store.active_session_id
            if store.active_session_id != target.session_id:
                try:
                    store.switch_session(target.session_id)
                except KeyError:
                    self._reject_conversation_settings_return(
                        handoffs,
                        claim,
                        target,
                        "The original conversation closed. Its Conversation settings "
                        "draft was not restored.",
                    )
                    return
                switched_session = True
                selected_active_session_epoch = store.active_session_epoch()
                await self._sync_native_console_chat_ui()
            restored = await self._reopen_suspended_console_settings(
                token,
                session_id=target.session_id,
                settings_revision=target.settings_revision,
                _on_transfer_committed=commit_transfer,
            )
            if not restored:
                handoffs.release(claim)
                await restore_prior_session_if_still_owned()
                return
            # The restored modal now owns A's only private draft. Settle A at
            # that transfer boundary; status copy below is optional UI work
            # and cannot make an already-owned draft retryable again.
            commit_transfer()
            try:
                await self._mount_conversation_settings_return_status(
                    target,
                    snapshot,
                )
            except Exception:
                logger.debug(
                    "Unable to mount Conversation settings return status",
                    exc_info=True,
                )
        except asyncio.CancelledError:
            if isinstance(handoffs, PendingHandoffStore) and claim is not None:
                handoffs.release(claim)
            try:
                await restore_prior_session_if_still_owned()
            except Exception:
                logger.debug("Unable to restore the prior Console session")
            raise
        except Exception:
            logger.debug(
                "Conversation settings return restore failed transiently",
                exc_info=True,
            )
            if isinstance(handoffs, PendingHandoffStore) and claim is not None:
                handoffs.release(claim)
            try:
                await restore_prior_session_if_still_owned()
            except Exception:
                logger.debug("Unable to restore the prior Console session")
        finally:
            if isinstance(handoffs, PendingHandoffStore) and claim is not None:
                handoffs.release(claim)
            if self._pending_conversation_settings_return_claim is claim:
                self._pending_conversation_settings_return_claim = None
            self._conversation_settings_return_restore_in_progress = False
            replacement = self._pending_conversation_settings_return_target
            prior_return_is_terminal = replacement is target
            if (
                isinstance(handoffs, PendingHandoffStore)
                and target is not None
                and replacement is not target
            ):
                try:
                    prior_return_is_terminal = handoffs.exact_revision_status(
                        HandoffChannel.CONVERSATION_SETTINGS_RETURN,
                        target.return_revision,
                    ) in ("settled", "superseded")
                except Exception:
                    logger.debug(
                        "Unable to confirm prior Conversation settings return settlement"
                    )
            replacement_is_pending = False
            if (
                prior_return_is_terminal
                and isinstance(handoffs, PendingHandoffStore)
                and replacement is not None
            ):
                try:
                    replacement_is_pending = (
                        handoffs.exact_revision_status(
                            HandoffChannel.CONVERSATION_SETTINGS_RETURN,
                            replacement.return_revision,
                        )
                        == "pending"
                    )
                except Exception:
                    logger.debug(
                        "Unable to inspect replacement Conversation settings return"
                    )
            if (
                replacement_is_pending
                and replacement is not target
                and self.is_mounted
                and self._owns_console_screen_stack()
            ):
                self.call_after_refresh(
                    self._consume_pending_conversation_settings_return
                )

    def _release_claimed_conversation_settings_return(self) -> None:
        """Release this screen's exact claim before its mounted lifetime ends."""

        claim = self._pending_conversation_settings_return_claim
        handoffs = getattr(self.app_instance, "pending_handoffs", None)
        if isinstance(handoffs, PendingHandoffStore) and claim is not None:
            handoffs.release(claim)
        if self._pending_conversation_settings_return_claim is claim:
            self._pending_conversation_settings_return_claim = None

    @staticmethod
    def _console_settings_initial_draft(
        settings: ConsoleSessionSettings,
        context_policy: ConsoleContextPolicyOverrides,
        *,
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState:
        """Build one process-local transaction from an exact live snapshot."""

        return ConsoleSettingsDraftState(
            settings=settings,
            context_policy_overrides=context_policy,
            field_drafts=tuple(
                ConsoleSettingsFieldDraft(
                    name=name,
                    effective_value=getattr(settings, name),
                    profile_override=getattr(settings, name),
                    provenance=ConsoleSettingsFieldProvenance.INHERITED,
                    dirty=False,
                )
                for name in sorted(exposed_fields)
            ),
            model_drafts=(),
            endpoint_draft=None,
        )
