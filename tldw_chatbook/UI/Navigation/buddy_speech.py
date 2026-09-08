"""App-owned Buddy speech using existing trusted TTS and destination consent."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Persona_Buddy.speech import BuddySpeechItem, BuddySpeechQueue


@dataclass(frozen=True, slots=True)
class _SpeechIdentity:
    session: Any
    assistant_kind: str | None
    character_ref: Any


class BuddySpeechCoordinator:
    """Read the selected scope without selecting, acknowledging or answering it."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.binding: BuddyBinding | None = None
        self.enabled = False
        self.queue = BuddySpeechQueue(self._play)
        self.notice = ""
        self._epoch = 0
        self._profile: tuple[object, ...] = ()
        self._poller: asyncio.Task | None = None
        self._consent_task: asyncio.Task | None = None
        self._consents: set[str] = set()
        self._pending_consent: tuple[BuddySpeechItem, Any] | None = None
        self._closed = False

    @property
    def needs_consent(self) -> bool:
        return self._pending_consent is not None

    def _profile_identity(self) -> tuple[object, ...]:
        return (
            getattr(self.app, "console_runtime", None),
            getattr(self.app, "chachanotes_db", None),
            getattr(self.app, "local_character_persona_service", None),
        )

    def configure(self, binding: BuddyBinding | None, enabled: bool) -> None:
        """Apply the committed selection synchronously; work remains app-owned."""
        profile = self._profile_identity()
        enabled = enabled is True and binding is not None and not self._closed
        if (
            binding != self.binding
            or enabled != self.enabled
            or profile != self._profile
        ):
            self._epoch += 1
            if self._consent_task is not None:
                self._consent_task.cancel()
            self.queue.reset()
            self._consents.clear()
            self._pending_consent = None
            self.notice = ""
            self.binding, self.enabled, self._profile = binding, enabled, profile
        if not enabled:
            if self._poller is not None:
                self._poller.cancel()
            if self._consent_task is not None:
                self._consent_task.cancel()
        elif self._poller is None:
            self._poller = asyncio.create_task(self._poll())

    async def _poll(self) -> None:
        try:
            while self.enabled and not self._closed:
                if self._profile_identity() != self._profile:
                    self.configure(self.binding, False)
                    self.notice = (
                        "The active profile changed. Enable Buddy speech again."
                    )
                    break
                try:
                    await self.refresh()
                except Exception:  # noqa: BLE001 - preserve execution, expose no private errors
                    self.notice = "Buddy speech is unavailable. Review the conversation in Console."
                await asyncio.sleep(1.0)
        finally:
            self._poller = None
            # A quick disable/re-enable waits for the previous owner to unwind.
            if self.enabled and not self._closed:
                self._poller = asyncio.create_task(self._poll())

    def _session_stamp(self, session: Any) -> tuple[object, ...]:
        persona_version = None
        if session.assistant_kind == "persona":
            service = getattr(self.app, "local_character_persona_service", None)
            record = (
                service.get_persona_profile(session.assistant_id) if service else None
            )
            if (
                not isinstance(record, Mapping)
                or record.get("deleted")
                or record.get("is_active", True) is not True
            ):
                raise ValueError("Persona unavailable")
            persona_version = record.get("version")
        return (
            session.assistant_kind,
            session.assistant_id,
            session.assistant_authority_id,
            session.identity_revision,
            persona_version,
        )

    def _context(self, store: Any, session: Any) -> Any:
        raw = getattr(self.app, "app_config", {})
        defaults = raw.get("chat_defaults", {}) if isinstance(raw, Mapping) else {}
        return store.presentation_context(session.id, defaults.get("user_display_name"))

    async def refresh(self) -> None:
        """Queue only completed responses/current questions; never streaming progress."""
        self.queue.revalidate()
        if not self.enabled or self.binding is None or self.queue.state.muted:
            return
        epoch, binding, profile = self._epoch, self.binding, self._profile
        runtime = getattr(self.app, "console_runtime", None)
        store = getattr(runtime, "chat_store", None)
        controller = getattr(runtime, "chat_controller", None)
        if store is None or controller is None or not runtime.alive:
            return
        candidates: list[tuple[Any, str | None, str | None]] = []
        if binding.kind == "workspace":
            from .buddy_workspace import BuddyWorkspaceCoordinator

            _, entries = await BuddyWorkspaceCoordinator(self.app, binding).snapshot()
            if epoch != self._epoch or profile != self._profile_identity():
                return
            receipt_service = getattr(runtime, "activity_receipts", None)
            receipts = (
                {r.activity_id: r for r in receipt_service.unseen_snapshot()}
                if receipt_service
                else {}
            )
            for entry in entries:
                session = entry.binding.resolve_session(store.sessions())
                if session is None or not binding.includes(session):
                    continue  # Never restore/activate a saved conversation to speak it.
                if entry.group == "needs_you":
                    candidates.append((session, None, None))
                elif entry.group == "results":
                    for receipt_id in entry.receipt_ids:
                        receipt = receipts.get(receipt_id)
                        if receipt is not None and receipt.assistant_message_id:
                            candidates.append(
                                (session, receipt.assistant_message_id, receipt_id)
                            )
        else:
            session = binding.resolve_session(store.sessions())
            if session is not None:
                candidates.append((session, None, None))
                from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

                latest = next(
                    (
                        m
                        for m in reversed(store.messages_for_session(session.id))
                        if m.role is ConsoleMessageRole.ASSISTANT
                        and m.status == "complete"
                    ),
                    None,
                )
                if latest is not None:
                    candidates.append((session, latest.id, None))
        for session, message_id, receipt_id in candidates:
            try:
                self._capture(
                    store,
                    controller,
                    session,
                    message_id,
                    epoch,
                    binding,
                    profile,
                    receipt_id,
                )
            except (ValueError, KeyError):
                continue  # Missing, edited, incomplete or unverifiable sources stay silent.

    def _capture(
        self,
        store,
        controller,
        session,
        message_id,
        epoch,
        binding,
        profile,
        receipt_id=None,
    ) -> None:
        exact = BuddyBinding.for_session(session)
        stamp = self._session_stamp(session)
        title = session.title
        identity = _SpeechIdentity(
            session, session.assistant_kind, session.character_ref()
        )

        def owner_current() -> bool:
            return (
                self.enabled
                and epoch == self._epoch
                and profile == self._profile_identity()
                and exact.resolve_session(store.sessions()) is session
                and binding.includes(session)
                and session.title == title
                and self._session_stamp(session) == stamp
                and (
                    receipt_id is None
                    or any(
                        receipt.activity_id == receipt_id
                        for receipt in profile[0].activity_receipts.unseen_snapshot()
                    )
                )
            )

        if message_id is not None:
            snapshot = store.issue_tts_message_speech_snapshot(
                message_id,
                owner_session_id=session.id,
                presentation_context=self._context(store, session),
            )

            def current() -> bool:
                return (
                    owner_current()
                    and store.validate_tts_message_speech_snapshot(
                        snapshot,
                        owner_session_id=session.id,
                        presentation_context=self._context(store, session),
                    )
                    == snapshot.raw_content
                )

            self.queue.enqueue(
                BuddySpeechItem(
                    f"message:{session.id}:{message_id}",
                    title,
                    snapshot.raw_content,
                    False,
                    current,
                    identity,
                )
            )
            return
        host = controller._interrupt_host
        for kind in host.payloads:
            payload = host.head_round_payload(kind, session.id)
            if not payload:
                continue
            round_id = payload.get("round_id") or payload.get("request_id")
            if not round_id:
                continue
            text = self._decision_text(kind, payload)

            def current(kind=kind, round_id=round_id, text=text) -> bool:
                with host.lock:
                    state = host.registries[kind].get(round_id)
                    live = bool(
                        state
                        and not state.get("revoked")
                        and not state["event"].is_set()
                    )
                head = host.head_round_payload(kind, session.id)
                return bool(
                    live
                    and owner_current()
                    and head
                    and (head.get("round_id") or head.get("request_id")) == round_id
                    and self._decision_text(kind, head) == text
                )

            self.queue.enqueue(
                BuddySpeechItem(
                    f"decision:{kind}:{round_id}",
                    title,
                    text,
                    True,
                    current,
                    identity,
                )
            )

    @staticmethod
    def _decision_text(kind: str, payload: Mapping) -> str:
        if kind == "question":
            questions = payload.get("questions") or ()
            return (
                "Your response is needed. "
                + " ".join(
                    str(q.get("question") or "")
                    for q in questions
                    if isinstance(q, Mapping)
                )[:2000]
            )
        return "Approval is needed. Open the Buddy or Console to review the request."

    async def _play(self, item: BuddySpeechItem, current) -> bool:
        identity = item.context
        if not isinstance(identity, _SpeechIdentity) or not current():
            return False
        ensure = getattr(self.app, "_ensure_tts_handler", None)
        if ensure is None:
            self.notice = "Configure speech in Settings before enabling Buddy speech."
            return False
        handler = await ensure()
        if handler is None or not current():
            return False
        destination = await handler.resolve_console_speech_destination(
            identity.assistant_kind, identity.character_ref
        )
        if destination is None or not current():
            return False
        consent = getattr(identity.session, "speech_preferences", None)
        if (
            destination.fingerprint not in self._consents
            and getattr(consent, "consent_destination", None) != destination.fingerprint
        ):
            self._pending_consent = (item, destination)
            self.notice = "Confirm the speech destination in Buddy speech controls."
            return False
        self.notice = ""
        return await handler.speak_guarded_utterance(
            item.named_text,
            assistant_kind=identity.assistant_kind,
            character_ref=identity.character_ref,
            expected_destination_fingerprint=destination.fingerprint,
            validator=current,
        )

    def request_consent(self) -> None:
        """Open consent only after an explicit user action, never from polling."""
        if self._pending_consent is not None and self._consent_task is None:
            self._consent_task = asyncio.create_task(self._confirm_destination())

    def set_input_active(self, owner: object, active: bool) -> None:
        """Accept an explicit microphone presentation hold without starting input."""
        self.queue.set_input_active(owner, active)

    async def wait_silent(self) -> None:
        """Wait for owned output to stop before a caller starts its microphone."""
        await self.queue.wait_idle()

    async def _confirm_destination(self) -> None:
        from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
            AutoSpeakConsentModal,
        )

        pending = self._pending_consent
        epoch = self._epoch
        modal = None
        try:
            if pending is None:
                return
            item, destination = pending
            if not BuddySpeechQueue._valid(item):
                self._pending_consent = None
                return
            result = asyncio.get_running_loop().create_future()

            def decided(value):
                if not result.done():
                    result.set_result(value is True)

            modal = AutoSpeakConsentModal(
                destination.provider_label,
                destination.sanitized_destination,
                destination.charges_may_apply,
                scope_label="the selected Buddy conversation or workspace",
            )
            self.app.push_screen(modal, decided)
            if (
                not await result
                or epoch != self._epoch
                or not BuddySpeechQueue._valid(item)
            ):
                return
            identity = item.context
            handler = await self.app._ensure_tts_handler()
            if (
                handler is None
                or epoch != self._epoch
                or not BuddySpeechQueue._valid(item)
            ):
                return
            current = await handler.resolve_console_speech_destination(
                identity.assistant_kind, identity.character_ref
            )
            if (
                epoch != self._epoch
                or not BuddySpeechQueue._valid(item)
                or current is None
                or current.fingerprint != destination.fingerprint
            ):
                self.notice = (
                    "The speech destination changed. Confirm its current settings."
                )
                return
            self._consents.add(destination.fingerprint)
            self._pending_consent = None
            self.notice = ""
            self.queue.reset()
            await self.refresh()
        except Exception:  # noqa: BLE001 - configuration failures expose no private detail
            self.notice = "Buddy speech is unavailable. Review speech settings."
        finally:
            if modal is not None and getattr(self.app, "screen", None) is modal:
                modal.dismiss(False)
            self._consent_task = None

    async def aclose(self) -> None:
        """Actual app shutdown only; normal navigation never calls this."""
        self._closed = True
        self.configure(None, False)
        tasks = [
            task for task in (self._poller, self._consent_task) if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.queue.aclose()


def ensure_buddy_speech(app: Any) -> BuddySpeechCoordinator:
    """Return the single app-owned queue without starting synthesis or polling."""
    coordinator = getattr(app, "buddy_speech_coordinator", None)
    if coordinator is None:
        coordinator = BuddySpeechCoordinator(app)
        app.buddy_speech_coordinator = coordinator
    return coordinator
