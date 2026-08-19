"""Destination consent and lifecycle coordination for Console auto-speak."""

from __future__ import annotations

import unicodedata
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_auto_speak import (
    AutoSpeakContext,
    AutoSpeakDisposition,
    decide_auto_speak,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        ConsoleTTSDestination,
    )
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

_FALLBACK_DESTINATION_COPY = "Configured TTS destination"


def sanitize_auto_speak_destination(value: object) -> str:
    """Render only a normalized HTTP(S) authority, never URL secrets or paths."""
    if type(value) is not str:
        return _FALLBACK_DESTINATION_COPY
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return _FALLBACK_DESTINATION_COPY
    if scheme not in {"http", "https"} or not hostname:
        return _FALLBACK_DESTINATION_COPY
    try:
        rendered_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return _FALLBACK_DESTINATION_COPY
    if ":" in rendered_host and not rendered_host.startswith("["):
        rendered_host = f"[{rendered_host}]"
    port_suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{rendered_host}{port_suffix}"


def _safe_label(value: object, *, fallback: str) -> str:
    text = value if type(value) is str else fallback
    projected: list[str] = []
    for character in text[:80]:
        category = unicodedata.category(character)
        if character in {"\r", "\n", "\t"}:
            projected.append(" ")
        elif category in {"Cc", "Cf", "Cs"}:
            projected.append("?")
        else:
            projected.append(character)
    normalized = " ".join("".join(projected).split())
    return normalized or fallback


class AutoSpeakConsentModal(SafeModalDismissMixin, ModalScreen[bool]):
    """Confirm automatic speech for one sanitized effective destination."""

    DEFAULT_CSS = """
    AutoSpeakConsentModal {
        align: center middle;
    }

    #console-auto-speak-consent-modal {
        width: 68;
        max-width: 95%;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-auto-speak-consent-actions {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "request_safe_cancel", "Cancel", show=False)
    ]
    SAFE_MODAL_CONTENT = "#console-auto-speak-consent-modal"

    def __init__(
        self,
        provider_label: str,
        sanitized_destination: str,
        charges_may_apply: bool,
    ) -> None:
        super().__init__()
        self.provider_label = _safe_label(provider_label, fallback="TTS provider")
        self.sanitized_destination = sanitize_auto_speak_destination(
            sanitized_destination
        )
        self.charges_may_apply = charges_may_apply is True

    def __repr__(self) -> str:
        return (
            "AutoSpeakConsentModal("
            f"provider_label={self.provider_label!r}, "
            f"sanitized_destination={self.sanitized_destination!r}, "
            f"charges_may_apply={self.charges_may_apply!r})"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="console-auto-speak-consent-modal"):
            yield Static("Speak new replies automatically?", markup=False)
            yield Static(
                f"Provider: {self.provider_label}",
                id="console-auto-speak-consent-provider",
                markup=False,
            )
            yield Static(
                f"Destination: {self.sanitized_destination}",
                id="console-auto-speak-consent-destination",
                markup=False,
            )
            if self.charges_may_apply:
                yield Static(
                    "Provider charges may apply.",
                    id="console-auto-speak-consent-charges",
                    markup=False,
                )
            yield Static(
                "Only new replies in this conversation will be spoken.",
                markup=False,
            )
            with Horizontal(id="console-auto-speak-consent-actions"):
                yield Button("Cancel", id="console-auto-speak-consent-cancel")
                yield Button(
                    "Enable",
                    id="console-auto-speak-consent-confirm",
                    variant="primary",
                )

    @on(Button.Pressed, "#console-auto-speak-consent-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(False)

    @on(Button.Pressed, "#console-auto-speak-consent-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)


class ConsoleAutoSpeakCoordinator:
    """Coordinate opt-in state, destination consent, and trusted dispatch."""

    def __init__(
        self,
        *,
        store_accessor: Callable[[], ConsoleChatStore],
        resolve_destination: Callable[[str | None, object | None], Awaitable[ConsoleTTSDestination | None]],
        issue_message_speech: Callable[
            [str, Callable[[bool], None], str | None, bool], Awaitable[bool]
        ],
        open_consent: Callable[
            [AutoSpeakConsentModal, Callable[[bool], None]], None
        ],
        hands_free_active: Callable[[], bool],
        sync_controls: Callable[[bool, bool, bool], None],
        notify: Callable[[str, str], None],
        schedule: Callable[[Coroutine[Any, Any, Any]], None],
    ) -> None:
        self._store_accessor = store_accessor
        self._resolve_destination_fn = resolve_destination
        self._issue_message_speech = issue_message_speech
        self._open_consent_fn = open_consent
        self._hands_free_active = hands_free_active
        self._sync_controls_fn = sync_controls
        self._notify = notify
        self._schedule = schedule
        self._mounted = False
        self._unsubscribe: Callable[[], None] | None = None
        self._modal_generation = 0
        self._modal_open = False
        self._modal_callback_consumed = False
        self._operation_generation = 0
        self._enable_operation_generation: int | None = None
        self._next_dispatch_generation = 1
        self._inflight_dispatches: dict[tuple[str, str], tuple[int, int]] = {}
        self._pending_completions: dict[
            tuple[str, str], tuple[int, int, int]
        ] = {}
        self._observed_completion_generations: dict[tuple[str, str], int] = {}
        self.failed_message_ids: dict[str, str] = {}

    def mount(self) -> None:
        """Subscribe exactly once for this screen lifecycle."""
        if self._mounted:
            return
        self._mounted = True
        self._unsubscribe = self._store_accessor().subscribe_message_completed(
            self._on_message_completed
        )
        self.sync_controls()

    def unmount(self) -> None:
        """Tombstone callbacks and release the store observer."""
        if not self._mounted:
            return
        self._mounted = False
        self._operation_generation += 1
        self._enable_operation_generation = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._modal_generation += 1
        self._modal_open = False
        self._modal_callback_consumed = True

    def _schedule_work(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        on_rejected: Callable[[], None] | None = None,
    ) -> bool:
        try:
            self._schedule(coroutine)
        except Exception:  # noqa: BLE001 - screen teardown rejects work
            coroutine.close()
            if on_rejected is not None:
                on_rejected()
            return False
        return True

    def _purge_failed_owners(self) -> None:
        try:
            live_session_ids = {
                session.id for session in self._store_accessor().sessions()
            }
        except Exception:  # noqa: BLE001 - lifecycle access fails closed
            return
        self.failed_message_ids = {
            session_id: message_id
            for session_id, message_id in self.failed_message_ids.items()
            if session_id in live_session_ids
        }

    def sync_controls(self) -> None:
        """Reflect only the active conversation's durable preference state."""
        session = self._active_session()
        self._purge_failed_owners()
        preferences = getattr(session, "speech_preferences", None)
        enabled = bool(getattr(preferences, "auto_speak", False))
        paused = bool(getattr(preferences, "paused", False))
        retry_available = False
        failed_message_id = (
            self.failed_message_ids.get(session.id) if session is not None else None
        )
        if enabled and paused and session is not None and failed_message_id:
            try:
                retry_available = (
                    self._store_accessor().session_id_for_message(
                        failed_message_id
                    )
                    == session.id
                )
            except (KeyError, ValueError):
                retry_available = False
        self._sync_controls_fn(enabled, paused, retry_available)

    def request_enabled(self, enabled: bool) -> None:
        """Apply a user toggle request without optimistic UI claims."""
        if not self._mounted or type(enabled) is not bool:
            return
        if enabled:
            if self._modal_open or self._enable_operation_generation is not None:
                self.sync_controls()
                return
            session = self._active_session()
            if session is None:
                self.sync_controls()
                return
            generation = self._reserve_operation()
            self._enable_operation_generation = generation
            active_epoch = self._store_accessor().active_session_epoch()

            def rejected() -> None:
                if self._enable_operation_generation == generation:
                    self._enable_operation_generation = None
                if self._mounted:
                    self.sync_controls()

            self._schedule_work(
                self._change_enabled(session.id, generation, active_epoch),
                on_rejected=rejected,
            )
            return
        generation = self._reserve_operation()
        self._enable_operation_generation = None
        active_epoch = self._store_accessor().active_session_epoch()
        self._schedule_work(self._disable(generation, active_epoch))

    def request_resume(self) -> None:
        """Persistently resume auto-speak after a speech failure."""
        if not self._mounted:
            return
        session = self._active_session()
        if session is None:
            return
        active_epoch = self._store_accessor().active_session_epoch()
        self._schedule_work(self._resume(session.id, active_epoch))

    def request_retry(self) -> None:
        """Retry only the failed reply while leaving auto-speak paused."""
        if not self._mounted:
            return
        session = self._active_session()
        if session is None:
            return
        generation = self._reserve_operation()
        active_epoch = self._store_accessor().active_session_epoch()
        self._schedule_work(self._retry(session.id, generation, active_epoch))

    def _on_message_completed(self, token: tuple[str, str]) -> None:
        if not self._mounted:
            return
        if (
            type(token) is not tuple
            or len(token) != 2
            or not all(type(value) is str and value for value in token)
        ):
            return
        try:
            completion_generation = (
                self._store_accessor().message_completion_generation(token[1])
            )
        except KeyError:
            return
        if (
            completion_generation
            <= self._observed_completion_generations.get(token, 0)
        ):
            return
        self._observed_completion_generations[token] = completion_generation
        if self._modal_open:
            return
        generation = self._operation_generation
        active_epoch = self._store_accessor().active_session_epoch()
        self._schedule_work(
            self._handle_completion(
                token,
                generation,
                completion_generation,
                active_epoch,
            )
        )

    def _reserve_operation(self) -> int:
        self._operation_generation += 1
        return self._operation_generation

    def _operation_is_current(
        self,
        generation: int,
        session_id: str,
        active_epoch: int,
    ) -> bool:
        if not self._mounted or generation != self._operation_generation:
            return False
        try:
            if self._store_accessor().active_session_epoch() != active_epoch:
                return False
        except Exception:  # noqa: BLE001 - lifecycle access fails closed
            return False
        active = self._active_session()
        return active is not None and active.id == session_id

    def _completion_is_current(
        self,
        token: tuple[str, str],
        completion_generation: int,
    ) -> bool:
        try:
            return (
                self._store_accessor().message_completion_generation(token[1])
                == completion_generation
            )
        except KeyError:
            return False

    def _active_session(self):
        try:
            store = self._store_accessor()
            active_id = store.active_session_id
            return next(
                (session for session in store.sessions() if session.id == active_id),
                None,
            )
        except Exception:  # noqa: BLE001 - screen lifecycle access fails closed
            return None

    async def _current_destination(
        self,
        session,
        generation: int,
        active_epoch: int,
    ) -> ConsoleTTSDestination | None:
        if session is None or not self._operation_is_current(
            generation, session.id, active_epoch
        ):
            return None
        try:
            destination = await self._resolve_destination_fn(
                getattr(session, "assistant_kind", None),
                session.character_ref(),
            )
        except Exception:  # noqa: BLE001 - provider resolution fails closed
            return None
        if not self._operation_is_current(generation, session.id, active_epoch):
            return None
        # TASK-18605: imported HERE, not at module scope. This is the only
        # runtime use of the symbol in this file -- every other reference is
        # an annotation, and `from __future__ import annotations` keeps
        # those as strings. The module-scope import cost 224 ms on the
        # Console screen's import path, because
        # `Event_Handlers/TTS_Events/tts_events.py` pulls in
        # `Audio/streaming_sink.py` at ITS module scope. By the time this
        # runs the user has asked to speak a reply, so the TTS stack is
        # about to load anyway.
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            ConsoleTTSDestination,
        )

        if type(destination) is not ConsoleTTSDestination:
            return None
        return destination

    async def _disable(self, generation: int, active_epoch: int) -> None:
        session = self._active_session()
        if session is None or not self._operation_is_current(
            generation, session.id, active_epoch
        ):
            return
        _session, persisted = self._store_accessor().set_auto_speak(
            session.id, False
        )
        if self._operation_is_current(generation, session.id, active_epoch):
            if not persisted:
                self._notify(
                    "Speak replies could not be changed. Try again.",
                    "error",
                )
            self.sync_controls()

    async def _change_enabled(
        self,
        session_id: str,
        generation: int,
        active_epoch: int,
    ) -> None:
        try:
            if not self._operation_is_current(
                generation, session_id, active_epoch
            ):
                return
            session = self._active_session()
            destination = await self._current_destination(
                session, generation, active_epoch
            )
            if not self._operation_is_current(
                generation, session_id, active_epoch
            ):
                return
            if destination is None:
                self._notify(
                    "A TTS destination is not available. Review Speech settings.",
                    "warning",
                )
                self.sync_controls()
                return
            self._show_enable_consent(
                session_id,
                destination,
                generation,
                active_epoch,
            )
        finally:
            if self._enable_operation_generation == generation:
                self._enable_operation_generation = None

    def _begin_modal(self) -> int:
        self._modal_generation += 1
        self._modal_open = True
        self._modal_callback_consumed = False
        return self._modal_generation

    def _modal_callback(
        self,
        generation: int,
        finish: Callable[[bool], Coroutine[Any, Any, Any]],
    ) -> Callable[[bool], None]:
        def callback(accepted: bool) -> None:
            if (
                not self._mounted
                or not self._modal_open
                or generation != self._modal_generation
                or self._modal_callback_consumed
            ):
                return
            self._modal_callback_consumed = True
            coroutine = finish(accepted is True)

            def rejected() -> None:
                self._finish_modal(generation)
                if self._mounted:
                    self.sync_controls()

            self._schedule_work(coroutine, on_rejected=rejected)

        return callback

    def _finish_modal(self, generation: int) -> None:
        if generation != self._modal_generation:
            return
        self._modal_open = False
        self._modal_callback_consumed = True

    def _open_modal(
        self,
        modal: AutoSpeakConsentModal,
        callback: Callable[[bool], None],
        *,
        modal_generation: int,
        operation_generation: int,
        active_epoch: int,
        session_id: str,
    ) -> None:
        try:
            self._open_consent_fn(modal, callback)
        except Exception:  # noqa: BLE001 - teardown can reject a modal post
            self._finish_modal(modal_generation)
            if self._operation_is_current(
                operation_generation, session_id, active_epoch
            ):
                self._notify(
                    "Speech confirmation could not be opened. Try again.",
                    "error",
                )
                self.sync_controls()

    def _show_enable_consent(
        self,
        session_id: str,
        destination: ConsoleTTSDestination,
        operation_generation: int,
        active_epoch: int,
    ) -> None:
        generation = self._begin_modal()

        async def finish(accepted: bool) -> None:
            try:
                if not accepted or not self._operation_is_current(
                    operation_generation, session_id, active_epoch
                ):
                    return
                store = self._store_accessor()
                active = self._active_session()
                if active is None or active.id != session_id:
                    return
                _session, persisted = store.confirm_auto_speak_destination(
                    session_id,
                    destination.fingerprint,
                )
                if not persisted:
                    self._notify(
                        "Destination consent could not be saved. Try again.",
                        "error",
                    )
                    return
                current = await self._current_destination(
                    active, operation_generation, active_epoch
                )
                if (
                    current is None
                    or current.fingerprint != destination.fingerprint
                    or not self._operation_is_current(
                        operation_generation, session_id, active_epoch
                    )
                ):
                    if self._operation_is_current(
                        operation_generation, session_id, active_epoch
                    ):
                        self._notify(
                            "The TTS destination changed. Turn on Speak replies again.",
                            "warning",
                        )
                    return
                _session, persisted = store.set_auto_speak(session_id, True)
                if (
                    not persisted
                    and self._operation_is_current(
                        operation_generation, session_id, active_epoch
                    )
                ):
                    self._notify(
                        "Speak replies could not be enabled. Try again.",
                        "error",
                    )
            finally:
                self._finish_modal(generation)
                if self._mounted:
                    self.sync_controls()

        modal = AutoSpeakConsentModal(
            destination.provider_label,
            destination.sanitized_destination,
            destination.charges_may_apply,
        )
        self._open_modal(
            modal,
            self._modal_callback(generation, finish),
            modal_generation=generation,
            operation_generation=operation_generation,
            active_epoch=active_epoch,
            session_id=session_id,
        )

    async def _handle_completion(
        self,
        token: tuple[str, str],
        operation_generation: int,
        completion_generation: int,
        active_epoch: int,
    ) -> None:
        if self._modal_open:
            return
        disposition, destination, preference_epoch = await self._disposition(
            token, operation_generation, active_epoch
        )
        if (
            not self._operation_is_current(
                operation_generation, token[0], active_epoch
            )
            or not self._completion_is_current(token, completion_generation)
        ):
            return
        if disposition is AutoSpeakDisposition.SPEAK:
            await self._dispatch(
                token,
                operation_generation,
                completion_generation,
                active_epoch,
                destination=destination,
                expected_preference_epoch=preference_epoch,
                revalidated=True,
            )
            return
        if (
            disposition is AutoSpeakDisposition.NEEDS_CONSENT
            and destination is not None
            and not self._modal_open
        ):
            self._show_completion_consent(
                token,
                destination,
                operation_generation,
                completion_generation,
                active_epoch,
            )

    async def _disposition(
        self,
        token: tuple[str, str],
        operation_generation: int,
        active_epoch: int,
    ) -> tuple[AutoSpeakDisposition, ConsoleTTSDestination | None, int | None]:
        store = self._store_accessor()
        session_id, message_id = token
        active = self._active_session()
        if active is None or active.id != session_id:
            return AutoSpeakDisposition.BACKGROUND, None, None
        try:
            message = store.get_message(message_id)
            preference_epoch = store.speech_preference_epoch(session_id)
        except KeyError:
            return AutoSpeakDisposition.INELIGIBLE, None, None
        preferences = active.speech_preferences
        if preferences.auto_speak is not True:
            return AutoSpeakDisposition.DISABLED, None, preference_epoch
        if preferences.paused is True:
            return AutoSpeakDisposition.PAUSED, None, preference_epoch
        if self._hands_free_active() is not False:
            return AutoSpeakDisposition.HANDSFREE_OWNS, None, preference_epoch
        destination = await self._current_destination(
            active, operation_generation, active_epoch
        )
        if not self._operation_is_current(
            operation_generation, session_id, active_epoch
        ):
            return AutoSpeakDisposition.BACKGROUND, None, None
        if self._hands_free_active() is not False:
            return AutoSpeakDisposition.HANDSFREE_OWNS, None, None
        try:
            current_epoch = store.speech_preference_epoch(session_id)
        except KeyError:
            return AutoSpeakDisposition.INELIGIBLE, None, None
        if current_epoch != preference_epoch:
            return AutoSpeakDisposition.INELIGIBLE, None, current_epoch
        active = self._active_session()
        if active is None or active.id != session_id:
            return AutoSpeakDisposition.BACKGROUND, None, current_epoch
        preferences = active.speech_preferences
        fingerprint = destination.fingerprint if destination is not None else ""
        disposition = decide_auto_speak(
            message,
            session_id=session_id,
            context=AutoSpeakContext(
                preferences=preferences,
                destination_fingerprint=fingerprint,
                active_session_id=store.active_session_id or "",
                hands_free_active=self._hands_free_active(),
            ),
        )
        return disposition, destination, current_epoch

    def _show_completion_consent(
        self,
        token: tuple[str, str],
        destination: ConsoleTTSDestination,
        operation_generation: int,
        completion_generation: int,
        active_epoch: int,
    ) -> None:
        generation = self._begin_modal()

        async def finish(accepted: bool) -> None:
            try:
                if (
                    not accepted
                    or not self._operation_is_current(
                        operation_generation, token[0], active_epoch
                    )
                    or not self._completion_is_current(
                        token, completion_generation
                    )
                ):
                    return
                store = self._store_accessor()
                session_id, _message_id = token
                active = self._active_session()
                if active is None or active.id != session_id:
                    return
                _session, persisted = store.confirm_auto_speak_destination(
                    session_id,
                    destination.fingerprint,
                )
                if not persisted:
                    self._notify(
                        "Destination consent could not be saved. Try again.",
                        "error",
                    )
                    return
                disposition, current, preference_epoch = await self._disposition(
                    token, operation_generation, active_epoch
                )
                if (
                    disposition is AutoSpeakDisposition.SPEAK
                    and current is not None
                    and current.fingerprint == destination.fingerprint
                    and self._operation_is_current(
                        operation_generation, session_id, active_epoch
                    )
                ):
                    await self._dispatch(
                        token,
                        operation_generation,
                        completion_generation,
                        active_epoch,
                        destination=current,
                        expected_preference_epoch=preference_epoch,
                        revalidated=True,
                    )
            finally:
                self._finish_modal(generation)
                if self._mounted:
                    self.sync_controls()

        modal = AutoSpeakConsentModal(
            destination.provider_label,
            destination.sanitized_destination,
            destination.charges_may_apply,
        )
        self._open_modal(
            modal,
            self._modal_callback(generation, finish),
            modal_generation=generation,
            operation_generation=operation_generation,
            active_epoch=active_epoch,
            session_id=token[0],
        )

    async def _dispatch(
        self,
        token: tuple[str, str],
        operation_generation: int,
        completion_generation: int,
        active_epoch: int,
        *,
        destination: ConsoleTTSDestination | None = None,
        expected_preference_epoch: int | None = None,
        revalidated: bool = False,
    ) -> None:
        if (
            not self._operation_is_current(
                operation_generation, token[0], active_epoch
            )
            or not self._completion_is_current(token, completion_generation)
        ):
            return
        inflight = self._inflight_dispatches.get(token)
        if inflight is not None:
            if completion_generation > inflight[0]:
                self._pending_completions[token] = (
                    completion_generation,
                    operation_generation,
                    active_epoch,
                )
            return
        if not revalidated:
            (
                disposition,
                destination,
                expected_preference_epoch,
            ) = await self._disposition(token, operation_generation, active_epoch)
            if (
                disposition is not AutoSpeakDisposition.SPEAK
                or not self._operation_is_current(
                    operation_generation, token[0], active_epoch
                )
                or not self._completion_is_current(
                    token, completion_generation
                )
            ):
                return
        if destination is None:
            return
        store = self._store_accessor()
        try:
            preference_epoch = store.speech_preference_epoch(token[0])
            active = self._active_session()
            message = store.get_message(token[1])
        except KeyError:
            return
        if (
            expected_preference_epoch is None
            or preference_epoch != expected_preference_epoch
            or active is None
            or active.id != token[0]
            or self._hands_free_active() is not False
            or decide_auto_speak(
                message,
                session_id=token[0],
                context=AutoSpeakContext(
                    preferences=active.speech_preferences,
                    destination_fingerprint=destination.fingerprint,
                    active_session_id=store.active_session_id or "",
                    hands_free_active=self._hands_free_active(),
                ),
            )
            is not AutoSpeakDisposition.SPEAK
        ):
            return
        dispatch_generation = self._next_dispatch_generation
        self._next_dispatch_generation += 1
        self._inflight_dispatches[token] = (
            completion_generation,
            dispatch_generation,
        )

        def on_outcome(ok: bool) -> None:
            if self._inflight_dispatches.get(token) != (
                completion_generation,
                dispatch_generation,
            ):
                return
            self._inflight_dispatches.pop(token, None)
            if ok is not True:
                self._pending_completions.pop(token, None)
                self._pause_after_failure(token, preference_epoch)
                return
            pending = self._pending_completions.pop(token, None)
            if pending is None:
                return
            pending_completion, pending_operation, pending_active_epoch = pending
            self._schedule_work(
                self._handle_completion(
                    token,
                    pending_operation,
                    pending_completion,
                    pending_active_epoch,
                )
            )

        try:
            issued = await self._issue_message_speech(
                token[1],
                on_outcome,
                destination.fingerprint,
                False,
            )
        except Exception:  # noqa: BLE001 - dispatch failures become paused state
            issued = False
        if not issued:
            on_outcome(False)

    def _pause_after_failure(
        self,
        token: tuple[str, str],
        preference_epoch: int,
    ) -> None:
        store = self._store_accessor()
        session_id, message_id = token
        try:
            session = next(
                session
                for session in store.sessions()
                if session.id == session_id
            )
            if store.speech_preference_epoch(session_id) != preference_epoch:
                return
            preferences = session.speech_preferences
            if preferences.auto_speak is not True or preferences.paused is True:
                return
            _session, persisted = store.pause_auto_speak(session_id)
        except (KeyError, StopIteration):
            return
        if not persisted:
            if self._mounted and self._active_session() is session:
                self._notify(
                    "Automatic speech failed and its paused state could not be saved.",
                    "error",
                )
                self.sync_controls()
            return
        self.failed_message_ids[session_id] = message_id
        if self._mounted and self._active_session() is session:
            self._notify(
                "Automatic speech paused. Use Retry speech for this reply or "
                "Resume auto-speak for future replies.",
                "warning",
            )
            self.sync_controls()

    async def _retry(
        self,
        session_id: str,
        operation_generation: int,
        active_epoch: int,
    ) -> None:
        if not self._operation_is_current(
            operation_generation, session_id, active_epoch
        ):
            return
        store = self._store_accessor()
        session = self._active_session()
        self._purge_failed_owners()
        message_id = self.failed_message_ids.get(session_id)
        if session is None or not message_id:
            return
        preferences = session.speech_preferences
        if preferences.auto_speak is not True or preferences.paused is not True:
            return
        try:
            preference_epoch = store.speech_preference_epoch(session_id)
            if store.session_id_for_message(message_id) != session_id:
                return
            store.get_message(message_id)
        except (KeyError, ValueError):
            return
        if self._hands_free_active() is not False:
            return
        destination = await self._current_destination(
            session, operation_generation, active_epoch
        )
        if not self._operation_is_current(
            operation_generation, session_id, active_epoch
        ):
            return
        if destination is None or self._hands_free_active() is not False:
            return
        self._purge_failed_owners()
        current = self._active_session()
        try:
            ownership_is_current = (
                current is not None
                and current.id == session_id
                and store.speech_preference_epoch(session_id) == preference_epoch
                and store.session_id_for_message(message_id) == session_id
                and self.failed_message_ids.get(session_id) == message_id
            )
        except (KeyError, ValueError):
            return
        if not ownership_is_current or current is None:
            return
        current_preferences = current.speech_preferences
        if (
            current_preferences.auto_speak is not True
            or current_preferences.paused is not True
            or current_preferences.consent_destination
            != destination.fingerprint
        ):
            return

        settled = False

        def failed_retry_is_current() -> bool:
            if not self._operation_is_current(
                operation_generation, session_id, active_epoch
            ):
                return False
            try:
                active = self._active_session()
                return (
                    active is not None
                    and active.id == session_id
                    and store.speech_preference_epoch(session_id)
                    == preference_epoch
                    and self.failed_message_ids.get(session_id) == message_id
                    and active.speech_preferences.auto_speak is True
                    and active.speech_preferences.paused is True
                )
            except KeyError:
                return False

        def on_outcome(ok: bool) -> None:
            nonlocal settled
            if settled:
                return
            settled = True
            if ok is not True and failed_retry_is_current():
                self._notify(
                    "Retry speech failed. Automatic speech remains paused.",
                    "warning",
                )
                self.sync_controls()

        try:
            issued = await self._issue_message_speech(
                message_id,
                on_outcome,
                destination.fingerprint,
                True,
            )
        except Exception:  # noqa: BLE001 - retry remains safely paused
            issued = False
        if not issued:
            on_outcome(False)

    async def _resume(self, session_id: str, active_epoch: int) -> None:
        if not self._mounted:
            return
        session = self._active_session()
        if (
            session is None
            or session.id != session_id
            or self._store_accessor().active_session_epoch() != active_epoch
        ):
            return
        _session, persisted = self._store_accessor().resume_auto_speak(session.id)
        if persisted:
            self.failed_message_ids.pop(session.id, None)
        else:
            self._notify(
                "Automatic speech could not be resumed. Try again.",
                "error",
            )
        self.sync_controls()
