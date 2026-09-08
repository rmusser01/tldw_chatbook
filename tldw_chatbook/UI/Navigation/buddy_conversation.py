"""Exact-target Buddy commands; accepted work belongs to the application runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding


@dataclass
class _VoiceCapture:
    binding: BuddyBinding
    session: Any
    draft: str
    speech: Any
    status: str = "starting"


class BuddyConversationCoordinator:
    """Keep private drafts and submit commands without selecting Console sessions."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._runtime = getattr(app, "console_runtime", None)
        self._profile = self._profile_identity()
        self.drafts: dict[BuddyBinding, str] = {}
        self.notices: dict[BuddyBinding, str] = {}
        self.submitting: set[BuddyBinding] = set()
        self.restoring: set[str] = set()
        self._voices: dict[object, _VoiceCapture] = {}
        self.voice_factory: Any = None
        self.voice_availability: Any = None

    @property
    def controller(self) -> Any:
        if self._profile_identity() != self._profile:
            return None
        return getattr(
            getattr(self.app, "console_runtime", None), "chat_controller", None
        )

    def _profile_identity(self) -> tuple[object, ...]:
        return tuple(
            getattr(self.app, name, None)
            for name in (
                "console_runtime",
                "chachanotes_db",
                "local_chat_conversation_service",
                "chat_conversation_scope_service",
            )
        )

    def resolve(
        self, binding: BuddyBinding, *, verify_record: bool = False
    ) -> Any | None:
        runtime = getattr(self.app, "console_runtime", None)
        store = getattr(runtime, "chat_store", None)
        if (
            self._profile_identity() != self._profile
            or store is None
            or runtime._disposed
        ):
            return None
        session = binding.resolve_session(store.sessions())
        if session is not None and session.persisted_conversation_id in self.restoring:
            return None
        if (
            session is not None
            and verify_record
            and session.persisted_conversation_id
            and not self._saved_record_available(session.persisted_conversation_id)
        ):
            return None
        return session

    def _saved_record_available(self, conversation_id: str) -> bool:
        service = getattr(self.app, "local_chat_conversation_service", None)
        if service is None:
            return False
        try:
            record = service.get_conversation_metadata(conversation_id)
            return bool(
                record
                and not record.get("deleted")
                and record.get("runtime_backend", "local") == "local"
            )
        except Exception:  # noqa: BLE001 - unavailable identity never becomes another target
            return False

    def can_open_console(self, binding: BuddyBinding) -> bool:
        if self.resolve(binding) is not None:
            return True
        runtime = getattr(self.app, "console_runtime", None)
        store = getattr(runtime, "chat_store", None)
        return bool(
            self._profile_identity() == self._profile
            and runtime is not None
            and not runtime._disposed
            and binding.conversation_id
            and not binding.ephemeral
            and (
                store is None
                or not any(row.id == binding.target_id for row in store.sessions())
            )
        )

    def prepare(self, binding: BuddyBinding) -> None:
        """Retain a conversation draft and load an explicitly requested saved row."""
        session = self.resolve(binding)
        if session is not None and self.controller is not None:
            self.controller._interrupt_host.retain_decision_target(session.id)
        for previous in tuple(self.drafts):
            previous_session = self.resolve(previous)
            same_conversation = (
                binding.conversation_id is not None
                and binding.conversation_id
                == (
                    previous_session.persisted_conversation_id
                    if previous_session is not None
                    else previous.conversation_id
                )
            )
            if (
                session is not None and previous_session is session
            ) or same_conversation:
                if binding not in self.drafts:
                    self.drafts[binding] = self.drafts[previous]
                if previous != binding:
                    self.drafts.pop(previous, None)
                break
        if session is None and self.can_open_console(binding):
            target = binding.conversation_id
            if target not in self.restoring:
                self.restoring.add(target)
                self.notices[binding] = "Loading the saved conversation…"
                self.app.run_worker(
                    self._restore_saved(binding),
                    group=f"buddy-restore:{target}",
                    exclusive=False,
                    exit_on_error=False,
                )

    async def _restore_saved(self, binding: BuddyBinding) -> None:
        from tldw_chatbook.Chat.console_conversation_hydration import (
            hydrate_console_generation_settings,
            hydrate_console_session,
            load_console_conversation_tree,
        )

        target = binding.conversation_id
        runtime = self._runtime
        store = None
        restored = None
        try:
            if not await asyncio.to_thread(self._saved_record_available, target):
                raise ValueError("This saved conversation is missing or unavailable.")
            if self._profile_identity() != self._profile or runtime._disposed:
                return
            if self.controller is None:
                # Explicit interaction may construct execution services. Passive
                # inbox inspection never reaches the existing launch bootstrap.
                from tldw_chatbook.Chat.console_launch_wake import (
                    _ensure_launch_runtime,
                )

                if _ensure_launch_runtime(self.app) is None:
                    raise ValueError("Console interaction could not start.")
            store = runtime.chat_store
            tree = await load_console_conversation_tree(self.app, target)
            if not tree or not await asyncio.to_thread(
                self._saved_record_available, target
            ):
                raise ValueError("This saved conversation is missing or unavailable.")
            if self.controller is None or runtime._disposed:
                return
            conversation = tree.get("conversation", {})
            if (
                conversation.get("id") != target
                or conversation.get("runtime_backend", "local") != "local"
            ):
                raise ValueError(
                    "This saved conversation is not available for local interaction."
                )
            # Recheck after I/O. An existing or repurposed slot owns its own state.
            matches = [
                row
                for row in store.sessions()
                if row.id == binding.target_id
                or row.persisted_conversation_id == target
            ]
            if matches:
                matched = binding.resolve_session(matches)
                if matched is None:
                    raise ValueError(
                        "This conversation binding changed. Choose it again."
                    )
                self.controller._interrupt_host.retain_decision_target(matched.id)
                self.notices[binding] = ""
                return
            hydration = hydrate_console_generation_settings(
                getattr(self.app, "app_config", {}) or {},
                conversation,
            )
            restored = await hydrate_console_session(
                app=self.app,
                store=store,
                conversation_id=target,
                tree=tree,
                settings=hydration.settings,
                generation_durable_snapshot=hydration.durable_snapshot,
                generation_metadata_status=hydration.metadata_status,
                target_scope_type="global"
                if not conversation.get("workspace_id")
                else None,
                activate=False,
            )
            record_available = await asyncio.to_thread(
                self._saved_record_available, target
            )
            if self.controller is None or runtime._disposed or not record_available:
                raise ValueError("This saved conversation is no longer available.")
            if binding.resolve_session(store.sessions()) is not restored:
                raise ValueError("This conversation binding changed. Choose it again.")
            self.notices[binding] = ""
            self.controller._interrupt_host.retain_decision_target(restored.id)
        except Exception:  # noqa: BLE001 - retain a bounded unavailable state
            if restored is not None:
                store.rollback_restored_session(
                    restored.id,
                    expected_session=restored,
                    prior_active_session_id=store.active_session_id,
                )
            self.notices[binding] = (
                "This saved conversation could not be loaded. Open Console to retry."
            )
        finally:
            self.restoring.discard(target)

    def decision_payloads(self, binding: BuddyBinding) -> dict[str, dict[str, Any]]:
        session = self.resolve(binding)
        controller = self.controller
        if session is None or controller is None:
            return {}
        host = controller._interrupt_host
        return {
            kind: dict(payload)
            for kind in host.payloads
            if (payload := host.head_round_payload(kind, session.id)) is not None
        }

    def show_decisions(self, owner: object, binding: BuddyBinding | None) -> None:
        controller = self.controller
        if controller is not None:
            session = self.resolve(binding) if binding is not None else None
            controller._interrupt_host.set_decision_view(
                owner,
                session.id if session else None,
                kinds=("approval", "question", "skill_install", "skill_script"),
            )

    def request_send(self, binding: BuddyBinding, text: str) -> None:
        """Capture a target/text and schedule on App, never on the modal DOM owner."""
        if binding in self.submitting or not text.strip():
            return
        self.drafts[binding] = text
        self.submitting.add(binding)
        self.app.run_worker(
            self._submit(binding, text),
            group=f"buddy-reply:{binding.target_id}",
            exclusive=False,
            exit_on_error=False,
        )

    async def _submit(self, binding: BuddyBinding, text: str) -> None:
        try:
            session = await asyncio.to_thread(self.resolve, binding, verify_record=True)
            controller = self.controller
            if (
                session is None
                or controller is None
                or self.resolve(binding) is not session
            ):
                raise ValueError(
                    "This conversation is unavailable. Open Console to review it."
                )
            if not controller.run_state_for(session.id).is_send_allowed:
                raise ValueError(
                    "This conversation is busy. Wait for its current turn or answer its decision."
                )
            # The normal send path keeps all admission, capture and tool checks.
            result = await controller.submit_draft(
                text,
                session_id=session.id,
                preserve_composer=True,
            )
            self.notices[binding] = result.visible_copy or ""
            if result.accepted and self.drafts.get(binding) == text:
                self.drafts[binding] = ""
        except ValueError as exc:
            self.notices[binding] = str(exc)
        except Exception:  # noqa: BLE001 - report a bounded UI failure without provider data
            self.notices[binding] = (
                "The reply could not finish. Open Console to review its state."
            )
        finally:
            self.submitting.discard(binding)

    def resolve_decision(
        self, binding: BuddyBinding, kind: str, round_id: str | None, decision: Any
    ) -> bool:
        """Only the bound session's current FIFO card can reach its existing resolver."""
        session = self.resolve(binding, verify_record=True)
        controller = self.controller
        if (
            session is None
            or controller is None
            or round_id is None
            or kind not in {"approval", "question", "skill_install", "skill_script"}
        ):
            return False
        host = controller._interrupt_host
        payload = host.head_round_payload(kind, session.id)
        if (
            payload is None
            or (payload.get("round_id") or payload.get("request_id")) != round_id
        ):
            return False
        with host.lock:
            state = host.registries[kind].get(round_id)
            if (
                state is None
                or state.get("session_id") != session.id
                or state.get("revoked")
            ):
                return False
        if kind == "approval":
            controller.resolve_pending_approval(decision, round_id=round_id)
        elif kind == "question":
            controller.resolve_pending_question(decision, request_id=round_id)
        elif kind == "skill_install":
            controller.resolve_pending_skill_install(decision, request_id=round_id)
        elif kind == "skill_script":
            controller.resolve_pending_skill_script(*decision, request_id=round_id)
        else:
            return False
        return True

    def voice_status(self, owner: object) -> str:
        capture = self._voices.get(owner)
        return capture.status if capture is not None else "idle"

    def request_voice(
        self, owner: object, binding: BuddyBinding, *, allowed: bool
    ) -> None:
        target = self.resolve(binding, verify_record=True)
        if not allowed or target is None or self.controller is None:
            return
        capture = self._voices.get(owner)
        if capture is not None:
            if capture.status == "recording":
                capture.status = "transcribing"
                self.app.run_worker(self._finish_voice(owner, capture), exclusive=False)
            return
        if not self.controller.run_state_for(target.id).is_send_allowed:
            return
        if self._voices:
            self.notices[binding] = (
                "Another Buddy microphone capture is active. Finish it first."
            )
            return
        from tldw_chatbook.Chat.console_voice_input import probe
        from tldw_chatbook.UI.Console_Modules.dictation import (
            ConsoleStreamingDictationSession,
        )

        availability = (self.voice_availability or probe)()
        if not availability.ok:
            self.notices[binding] = (
                f"{availability.reason} {availability.remedy}".strip()
            )
            return
        session = (self.voice_factory or ConsoleStreamingDictationSession)(
            on_event=lambda _session, _event: None,
        )
        from .buddy_speech import ensure_buddy_speech

        speech = ensure_buddy_speech(self.app)
        capture = _VoiceCapture(binding, session, self.drafts.get(binding, ""), speech)
        self._voices[owner] = capture
        speech.set_input_active(owner, True)
        self.app.run_worker(self._start_voice(owner, capture), exclusive=False)

    async def _start_voice(self, owner: object, capture: _VoiceCapture) -> None:
        try:
            await capture.speech.wait_silent()
            if (
                self._voices.get(owner) is not capture
                or self.resolve(capture.binding) is None
            ):
                self.close_voice(owner)
                return
            await asyncio.to_thread(
                capture.session.start,
                on_buffer_limit=lambda: self.app.call_later(
                    self.request_voice, owner, capture.binding, allowed=True
                ),
            )
            if (
                self._voices.get(owner) is not capture
                or self.resolve(capture.binding) is None
            ):
                capture.session.discard()
                return
            capture.status = "recording"
        except Exception:  # noqa: BLE001 - audio errors can contain device/profile details
            if self._voices.get(owner) is capture:
                self.notices[capture.binding] = (
                    "Microphone capture could not start. Check Console voice settings."
                )
                self.close_voice(owner)

    async def _finish_voice(self, owner: object, capture: _VoiceCapture) -> None:
        try:
            text = await asyncio.to_thread(capture.session.stop_and_transcribe)
            target = await asyncio.to_thread(
                self.resolve, capture.binding, verify_record=True
            )
            if (
                self._voices.get(owner) is not capture
                or target is None
                or self.resolve(capture.binding) is not target
            ):
                return
            if self.drafts.get(capture.binding, "") != capture.draft:
                self.notices[capture.binding] = (
                    "The draft changed during dictation. Recording discarded."
                )
            else:
                self.drafts[capture.binding] = " ".join(
                    part for part in (capture.draft, text) if part
                )
                self.notices[capture.binding] = "Review your dictated reply, then Send."
        except Exception:  # noqa: BLE001 - local speech failures must not submit text
            if self._voices.get(owner) is capture:
                self.notices[capture.binding] = (
                    "Dictation could not finish. Retry or type your reply."
                )
        finally:
            if self._voices.get(owner) is capture:
                self.close_voice(owner)

    def close_voice(self, owner: object) -> None:
        capture = self._voices.pop(owner, None)
        if capture is not None:
            try:
                capture.session.discard()
            finally:
                capture.speech.set_input_active(owner, False)

    def open_console(self, binding: BuddyBinding) -> bool:
        if not self.can_open_console(binding):
            return False
        self.app.run_worker(
            self._open_bound(binding),
            group="buddy-open-console",
            exclusive=False,
        )
        return True

    async def _console_target(self, binding: BuddyBinding) -> str | None:
        session = await asyncio.to_thread(self.resolve, binding, verify_record=True)
        if session is not None and self.resolve(binding) is session:
            return f"native:{session.id}"
        if binding.conversation_id and self.can_open_console(binding):
            available = await asyncio.to_thread(
                self._saved_record_available, binding.conversation_id
            )
            if available and self.can_open_console(binding):
                return binding.conversation_id
        return None

    async def _open_bound(self, binding: BuddyBinding) -> None:
        if await self._console_target(binding) is None:
            self.app.notify(
                "The bound conversation is unavailable.", severity="warning"
            )
            return
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        def navigated(success: bool) -> None:
            if not success or self.controller is None:
                return
            screen = self.app.screen
            workspace = getattr(screen, "_workspace", None)
            if workspace is not None:
                self.app.run_worker(
                    self._activate_bound(binding, workspace),
                    group="buddy-open-console",
                    exclusive=False,
                )

        self.app.post_message(NavigateToScreen("chat", on_completion=navigated))

    async def _activate_bound(self, binding: BuddyBinding, workspace: Any) -> None:
        target = await self._console_target(binding)
        if target is None:
            self.app.notify(
                "The bound conversation is unavailable.", severity="warning"
            )
            return
        if (
            self.controller is None
            or getattr(self.app.screen, "_workspace", None) is not workspace
        ):
            return
        # The existing handoff owns view/composer synchronization and failure rollback.
        await workspace.open_console_workspace_conversation(target)


def open_buddy_conversation(
    app: Any, binding: BuddyBinding, *, allow_voice: bool = True
) -> Any:
    """Open exact local conversation interaction, preserving the underlying screen."""
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_conversation_modal import (
        BuddyConversationModal,
    )

    coordinator = getattr(app, "buddy_conversation_coordinator", None)
    if coordinator is None or coordinator._profile != coordinator._profile_identity():
        coordinator = BuddyConversationCoordinator(app)
        app.buddy_conversation_coordinator = coordinator
    modal = BuddyConversationModal(coordinator, binding, allow_voice=allow_voice)
    coordinator.prepare(binding)
    app.push_screen(modal)
    return modal
