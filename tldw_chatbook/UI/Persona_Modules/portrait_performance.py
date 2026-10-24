"""Source-bound local portrait review and publication for the Personas editor."""

from __future__ import annotations

import asyncio
import threading
import weakref
from dataclasses import dataclass
from typing import Any

from loguru import logger
from textual.widgets import Select

from ...Character_Chat.portrait_performance import (
    MAX_FRAME_BYTES,
    PerformanceConflict,
    capture_performance_source,
    normalize_performance_frame,
)
from ...Character_Chat.portrait_performance_authoring import PortraitPerformanceDraft
from ...Chat.character_expression_playback import expression_motion_enabled
from ...DB.base_db import operation_owned_connection
from ...DB.ChaChaNotes_DB import CharactersRAGDB
from ...Utils.path_validation import validate_path_simple
from ...Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
    PersonasVisualIdentityPackWidget,
)
from ...Widgets.Persona_Widgets.portrait_performance_editor import (
    PortraitPerformanceEditor,
)


@dataclass(frozen=True, slots=True)
class _Authority:
    db: CharactersRAGDB
    scope: Any
    local: Any
    editor: Any
    identity: tuple
    character_id: int


class PortraitPerformanceController:
    """Own one disposable draft; revalidate the actual editor/profile around I/O."""

    def __init__(self, screen: Any, drain: Any) -> None:
        self._screen = weakref.ref(screen)
        self._drain = drain
        self.draft: PortraitPerformanceDraft | None = None
        self._authority: _Authority | None = None
        self._source_request = "idle"
        self._local_authority = ""
        self._busy = False
        self._saving = False
        self._epoch = 0
        self._task: asyncio.Task | None = None
        self._generation_event: threading.Event | None = None
        self._speech_task: asyncio.Task | None = None
        self._presentation = None

    def _capture(self) -> _Authority:
        screen = self._screen()
        editor = screen._editor_or_none() if screen is not None else None
        db = getattr(screen.app_instance, "chachanotes_db", None) if screen else None
        scope = (
            getattr(screen.app_instance, "character_persona_scope_service", None)
            if screen
            else None
        )
        local = getattr(scope, "local_service", None) if scope is not None else None
        character_id = editor.expression_character_id() if editor else None
        if (
            screen is None
            or not screen.is_mounted
            or not screen._local_character_actions_allowed()
            or not screen._character_editor_is_active()
            or screen.state.active_mode != "characters"
            or screen.state.selected_entity_kind != "character"
            or character_id is None
            or str(screen.state.selected_entity_id) != str(character_id)
            or type(db) is not CharactersRAGDB
            or (scope is not None and (local is None or local.db is not db))
        ):
            raise ValueError("Open a saved local character to prepare its performance.")
        if editor.has_unsaved_attachment():
            raise ValueError(
                "Save changed portrait artwork first. Text edits will not be saved by Prepare."
            )
        if screen._visual_identity_has_unsaved_authoring() or any(
            browser.has_staged_changes
            for browser in editor.query(PersonasVisualIdentityPackWidget)
        ):
            raise ValueError("Apply or discard reaction artwork changes first.")
        if screen._character_save_inflight:
            raise ValueError("Wait for character Save to finish.")
        identity = (
            id(db),
            id(scope),
            id(local),
            id(editor),
            screen._character_editor_session_token(),
            editor.visual_identity_session_token,
            character_id,
        )
        return _Authority(db, scope, local, editor, identity, character_id)

    def _current(self, authority: _Authority) -> bool:
        try:
            return self._capture().identity == authority.identity
        except (AttributeError, ValueError):
            return False

    def _view(self) -> PortraitPerformanceEditor | None:
        screen = self._screen()
        editor = screen._editor_or_none() if screen is not None else None
        return editor.query_one(PortraitPerformanceEditor) if editor else None

    def _check_context(self) -> None:
        if self._authority is not None and not self._current(self._authority):
            self.discard()

    def discard(self) -> None:
        """Fence in-flight work before dropping any draft or preview bytes."""
        self._epoch += 1
        if self._generation_event is not None:
            self._generation_event.set()
        if self._speech_task is not None:
            self._speech_task.cancel()
        self._presentation = None
        if self.draft is not None:
            self.draft.cancel()
        self.draft = None
        self._authority = None
        view = self._view()
        if view is not None:
            view.reset()

    async def _native(self, authority: _Authority, function, *args):
        def work():
            with operation_owned_connection(authority.db):
                return function(*args)

        return await self._drain(work, task_name="personas-portrait-performance")

    def _source(self, authority: _Authority, requested: str):
        source, data = capture_performance_source(
            authority.db,
            character_id=authority.character_id,
            requested_state=requested,
            manual_expression_key=None
            if requested in {"idle", "thinking", "speaking", "error"}
            else requested,
        )
        return source, data, authority.db.get_local_authority_id()

    def _verify_source(
        self,
        authority: _Authority,
        draft: PortraitPerformanceDraft,
        requested: str,
        local_authority: str,
    ) -> None:
        source, _, local = self._source(authority, requested)
        if source != draft.source or local != local_authority:
            raise PerformanceConflict("The saved artwork changed. Prepare it again.")

    async def handle(self, message, *, path: str | None = None) -> bool:
        """Run one explicit action; duplicate work never cancels an admitted write."""
        if message.action == "sample" and self._speech_task is not None:
            task = self._speech_task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return False
        if message.action == "cancel":
            if not self._saving:
                speech = self._speech_task
                self.discard()
                if speech is not None:
                    await asyncio.gather(speech, return_exceptions=True)
            return False
        if self._busy:
            return False
        screen = self._screen()
        view = self._view()
        if screen is None or view is None:
            return False
        focus = screen.focused
        if focus is None or focus.id != f"portrait-performance-{message.action}":
            focus = None
        try:
            authority = self._capture()
        except ValueError as error:
            view.status(str(error))
            return False
        if message.action != "prepare" and (
            self.draft is None
            or self._authority is None
            or not self._current(self._authority)
        ):
            self.discard()
            view.status("Prepare saved artwork again.")
            return False
        if message.action == "prepare":
            self.discard()
            self._authority = authority
        self._busy = True
        self._saving = message.action in {"apply", "remove"}
        self._task = asyncio.current_task()
        epoch = self._epoch
        view.set_busy(True, saving=self._saving)
        cancellation = None
        published = False
        try:
            if message.action == "prepare":
                requested = message.source or "idle"

                def prepare():
                    source, data, local = self._source(authority, requested)
                    record = authority.db.get_character_portrait_performance(
                        character_id=authority.character_id,
                        expression_key=source.expression_key,
                    )
                    return source, data, local, record

                outcome = await self._native(authority, prepare)
                cancellation = outcome.cancellation
                if outcome.error:
                    raise outcome.error
                if (
                    not outcome.completed
                    or cancellation
                    or epoch != self._epoch
                    or not self._current(authority)
                ):
                    return False
                source, data, self._local_authority, record = outcome.value
                self.draft = PortraitPerformanceDraft(
                    source, data, expected_revision=record.revision if record else 0
                )
                if record is not None and record.active and record.source == source:
                    for frame in record.frames:
                        self.draft.stage(frame)
                self._source_request = requested
                view.applied = bool(
                    record and record.active and record.source == source
                )
                view.status(
                    "Saved source captured. Import six matching stills, then review and Apply."
                )
            else:
                draft = self.draft
                requested, local = self._source_request, self._local_authority
                if message.action in {"generate", "regenerate"}:
                    from ...Character_Chat.portrait_performance import FRAME_ROLES

                    return await self._generate(
                        authority,
                        draft,
                        FRAME_ROLES
                        if message.action == "generate"
                        else (message.role,),
                        epoch,
                    )
                if message.action == "sample":
                    return await self._sample(authority, draft, epoch)
                if message.action == "import":
                    if path is None:
                        from ...Widgets.enhanced_file_picker import (
                            EnhancedFileOpen,
                            Filters,
                        )

                        path = await screen.app.push_screen_wait(
                            EnhancedFileOpen(
                                title="Import portrait pose",
                                context="portrait_performance_import",
                                filters=Filters(
                                    (
                                        "Still images",
                                        lambda p: (
                                            p.suffix.lower()
                                            in {".png", ".jpg", ".jpeg", ".webp"}
                                        ),
                                    )
                                ),
                            )
                        )
                    if not path:
                        return False
                    if epoch != self._epoch or not self._current(authority):
                        return False

                    def read():
                        candidate = validate_path_simple(str(path), require_exists=True)
                        if not candidate.is_file() or candidate.suffix.lower() not in {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                        }:
                            raise ValueError("Choose a still PNG, JPEG or WebP image.")
                        with candidate.open("rb") as stream:
                            data = stream.read(MAX_FRAME_BYTES + 1)
                        frame = normalize_performance_frame(message.role, data)
                        self._verify_source(authority, draft, requested, local)
                        return frame

                    outcome = await self._native(authority, read)
                    cancellation = outcome.cancellation
                    if outcome.error:
                        raise outcome.error
                    if (
                        not outcome.completed
                        or cancellation
                        or epoch != self._epoch
                        or not self._current(authority)
                    ):
                        return False
                    draft.stage(outcome.value)
                    view.status(
                        f"{message.role.replace('_', ' ')} ready. All six poses must have matching dimensions."
                    )
                elif message.action == "clear":
                    draft.remove(message.role)
                elif message.action in {"apply", "remove"}:
                    frames = draft.frames() if message.action == "apply" else None

                    def publish():
                        self._verify_source(authority, draft, requested, local)
                        if frames is None:
                            return authority.db.remove_character_portrait_performance(
                                draft.source, expected_revision=draft.expected_revision
                            )
                        return authority.db.apply_character_portrait_performance(
                            draft.source,
                            frames,
                            expected_revision=draft.expected_revision,
                        )

                    outcome = await self._native(authority, publish)
                    cancellation = outcome.cancellation
                    if outcome.error:
                        raise outcome.error
                    if outcome.completed:
                        published = True
                        await self._invalidate(authority.character_id)
                    if (
                        not outcome.completed
                        or cancellation
                        or epoch != self._epoch
                        or not self._current(authority)
                    ):
                        return False
                    draft.expected_revision = outcome.value
                    view.applied = frames is not None
                    if frames is None:
                        self.discard()
                        view.status("Performance removed. Ordinary artwork restored.")
                        return True
                    view.status(
                        "Performance applied. Saved only in this local profile."
                    )
            if (
                self.draft is not None
                and epoch == self._epoch
                and self._current(authority)
            ):
                await view.show_draft(
                    self.draft,
                    lambda: epoch == self._epoch and self._current(authority),
                    motion=expression_motion_enabled(
                        getattr(screen.app_instance, "app_config", {}) or {},
                        react=True,
                        manual=True,
                    ),
                )
            return True
        except (ValueError, OSError) as error:
            if epoch == self._epoch:
                if isinstance(error, PerformanceConflict):
                    self.discard()
                view.status(
                    "The artwork changed; Prepare again."
                    if isinstance(error, PerformanceConflict)
                    else str(error)
                    .replace("portrait_performance_", "")
                    .replace("_", " ")
                )
            return False
        except Exception:  # noqa: BLE001 -- optional authoring/invalidation must preserve the saved artwork
            logger.warning(
                "Portrait performance operation failed (category=authoring_failed)."
            )
            if epoch == self._epoch:
                view.status(
                    "Performance saved; preview unavailable. Prepare again to review it."
                    if published
                    else "Performance could not be saved or prepared. Current artwork kept; retry explicitly."
                )
            return published
        finally:
            self._busy = self._saving = False
            self._task = None
            view.set_busy(False)

            def restore_focus():
                if (
                    focus is not None
                    and focus.is_mounted
                    and not focus.disabled
                    and screen.is_current
                    and epoch == self._epoch
                    and self._current(authority)
                ):
                    focus.focus()

            if focus is not None:
                screen.call_after_refresh(restore_focus)
            if cancellation is not None:
                raise cancellation

    async def _invalidate(self, character_id: int) -> None:
        screen = self._screen()
        for mounted in tuple(getattr(screen.app, "screen_stack", ())) if screen else ():
            invalidate = getattr(
                getattr(mounted, "_session", None),
                "invalidate_visual_identity_actor",
                None,
            )
            if callable(invalidate):
                try:
                    await invalidate("character", str(character_id))
                except Exception:  # noqa: BLE001 -- optional authoring/invalidation must preserve the saved artwork
                    logger.warning(
                        "Portrait performance Console invalidation failed (category=cache_invalidation_failed)."
                    )

    async def _source_current(self, authority, draft, epoch) -> bool:
        if epoch != self._epoch or not self._current(authority):
            return False
        outcome = await self._native(
            authority,
            self._verify_source,
            authority,
            draft,
            self._source_request,
            self._local_authority,
        )
        if outcome.error:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation
        return outcome.completed and epoch == self._epoch and self._current(authority)

    async def _generate(self, authority, draft, roles, epoch) -> bool:
        from ...Character_Chat.expression_generation import compose_performance_prompt
        from ...Character_Chat.portrait_performance import FRAME_ROLES
        from ...Character_Chat.portrait_performance_authoring import (
            performance_reference,
        )
        from ...Image_Generation.adapter_registry import get_registry
        from ...Image_Generation.adapters.base import ImageGenResult
        from ...Image_Generation.capabilities import (
            resolve_backend_reference_image_capability,
        )
        from ...Image_Generation.config import get_image_generation_config
        from ...Image_Generation.worker import build_request, run_generation
        from ...Utils.input_validation import escape_markup
        from ...Widgets.confirmation_dialog import ConfirmationDialog
        from ...Widgets.Console.console_auto_speak_consent import (
            sanitize_auto_speak_destination,
        )

        screen, view = self._screen(), self._view()
        if not roles or any(role not in FRAME_ROLES for role in roles):
            raise ValueError("Choose a portrait pose.")
        cfg = get_image_generation_config()
        backend = cfg.default_backend
        if (
            not backend
            or backend not in cfg.enabled_backends
            or not resolve_backend_reference_image_capability(
                backend, config=cfg
            ).supported
        ):
            view.status(
                "Choose an enabled reference-image backend in Settings. Imports remain available."
            )
            return False
        warning = screen._image_generation_backend_warning()
        if warning:
            view.status(warning + " Imports remain available.")
            return False
        registry = get_registry()
        if registry.config is not cfg:
            view.status(
                "Image settings changed. Reload the configured image engine before generating."
            )
            return False
        model = getattr(cfg, f"{backend}_image_default_model", None)
        destination = sanitize_auto_speak_destination(
            getattr(cfg, f"{backend}_image_base_url", None)
        )
        message = (
            f"Send {len(roles)} image-edit requests to {backend} / {model or 'configured workflow/default'} ({destination}). "
            f"Use saved character {authority.character_id}, expression {draft.source.expression_key}, as the reference. "
            "Charges may apply. Successful draft slots will be replaced. Uncertain requests are never retried automatically; review before Apply."
        )
        confirmed = await screen.app.push_screen_wait(
            ConfirmationDialog(
                title="Generate portrait poses",
                message=escape_markup(message),
                confirm_label=f"Generate {len(roles)} frames",
                cancel_label="Cancel",
            )
        )
        if confirmed is not True or not await self._source_current(
            authority, draft, epoch
        ):
            return False
        outcome = await self._drain(
            performance_reference,
            draft.source_bytes,
            draft.source.source_sha256,
            task_name="portrait-performance-reference",
        )
        if outcome.error:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if not outcome.completed or not await self._source_current(
            authority, draft, epoch
        ):
            return False
        reference = outcome.value
        event = self._generation_event = threading.Event()
        failed = []
        try:
            for role in roles:
                if event.is_set() or not await self._source_current(
                    authority, draft, epoch
                ):
                    return False
                if (
                    get_image_generation_config() is not cfg
                    or get_registry() is not registry
                ):
                    view.status(
                        "Image settings changed. Remaining requests cancelled; review partial results."
                    )
                    return False
                view.status(
                    f"Generating {role.replace('_', ' ')} via {backend}; {len(roles)} explicit requests, no automatic retry."
                )
                request = build_request(
                    backend=backend,
                    model=model,
                    prompt=compose_performance_prompt(
                        role=role, expression_key=draft.source.expression_key
                    ),
                    width=reference.width,
                    height=reference.height,
                    image_format="png",
                    reference_image=reference,
                    cancel_event=event,
                )
                try:
                    outcome = await self._drain(
                        run_generation,
                        request,
                        task_name="portrait-performance-generate",
                    )
                    if outcome.cancellation is not None:
                        event.set()
                        raise outcome.cancellation
                    if event.is_set() or not await self._source_current(
                        authority, draft, epoch
                    ):
                        return False
                    if outcome.error:
                        raise outcome.error
                    result = outcome.value
                    if (
                        not outcome.completed
                        or type(result) is not ImageGenResult
                        or type(result.content) is not bytes
                        or result.bytes_len != len(result.content)
                        or result.content_type
                        not in {"image/png", "image/jpeg", "image/webp"}
                    ):
                        raise ValueError("Invalid generated still.")
                    normalized = await self._drain(
                        normalize_performance_frame,
                        role,
                        result.content,
                        task_name="portrait-performance-normalize",
                    )
                    if normalized.error:
                        raise normalized.error
                    if normalized.cancellation is not None:
                        event.set()
                        raise normalized.cancellation
                    if event.is_set() or not await self._source_current(
                        authority, draft, epoch
                    ):
                        return False
                    draft.stage(normalized.value)
                except PerformanceConflict:
                    raise
                except Exception:  # noqa: BLE001 -- paid/uncertain outcomes require an explicit slot retry
                    failed.append(role)
            try:
                await view.show_draft(
                    draft,
                    lambda: epoch == self._epoch and self._current(authority),
                    motion=expression_motion_enabled(
                        getattr(screen.app_instance, "app_config", {}) or {},
                        react=True,
                        manual=True,
                    ),
                )
            except Exception:  # noqa: BLE001 -- a failed preview must not suggest another paid request
                view.status(
                    "Generated slots retained; preview unavailable. Review them before sending further requests."
                )
                return True
            if failed:
                view.status(
                    f"Review partial results. Failed or uncertain: {', '.join(failed)}. Retry only a selected slot explicitly; charges may already apply."
                )
            else:
                view.status(
                    "Generated poses ready for review. Nothing activates until Apply."
                )
            return True
        finally:
            event.set()
            if self._generation_event is event:
                self._generation_event = None

    async def _sample(self, authority, draft, epoch) -> bool:
        from uuid import uuid4

        from ...Audio.speech_presentation import (
            SpeechPresentation,
            SpeechPresentationOwner,
        )
        from ...TTS.profile_types import CharacterRef
        from ...Utils.input_validation import escape_markup
        from ...Widgets.confirmation_dialog import ConfirmationDialog

        screen, view = self._screen(), self._view()
        draft.frames()
        ensure = getattr(screen.app_instance, "_ensure_tts_handler", None)
        if not callable(ensure):
            view.status(
                "Configure TTS in Settings to hear a sample. Imports, generation and Apply remain available."
            )
            return False
        handler = await ensure()
        if not await self._source_current(authority, draft, epoch):
            return False
        if handler is None:
            view.status(
                "TTS is unavailable. Review poses offline or configure speech in Settings."
            )
            return False
        character = CharacterRef(
            source="local",
            authority_id=self._local_authority,
            character_id=str(authority.character_id),
        )
        destination = await handler.resolve_console_speech_destination(
            "character", character
        )
        if not await self._source_current(authority, draft, epoch):
            return False
        if destination is None:
            view.status("No usable speech destination. Review poses offline.")
            return False
        confirmed = await screen.app.push_screen_wait(
            ConfirmationDialog(
                title="Play portrait speech sample",
                message=escape_markup(
                    f"Send the fixed preview text to {destination.provider_label} ({destination.sanitized_destination}) using this character's voice. {'Charges may apply. ' if destination.charges_may_apply else ''}This sample creates no message or automatic speech preference."
                ),
                confirm_label="Play sample",
                cancel_label="Cancel",
            )
        )
        if confirmed is not True or not await self._source_current(
            authority, draft, epoch
        ):
            return False
        current = await handler.resolve_console_speech_destination(
            "character", character
        )
        if not await self._source_current(authority, draft, epoch):
            return False
        if current is None or current.fingerprint != destination.fingerprint:
            view.status(
                "The speech destination changed. Request and confirm a new sample."
            )
            return False
        token = uuid4().hex
        self._presentation = SpeechPresentation(
            SpeechPresentationOwner(
                token,
                object(),
                f"portrait-draft-{epoch}",
                epoch,
                authority.character_id,
                None,
                epoch,
            )
        )
        view.query_one("#portrait-performance-preview-mode", Select).value = "speech"
        await view.show_draft(
            draft,
            lambda: epoch == self._epoch and self._current(authority),
            motion=expression_motion_enabled(
                getattr(screen.app_instance, "app_config", {}) or {},
                react=True,
                manual=True,
            ),
            read_speech=self._presentation.read,
        )
        task = self._speech_task = asyncio.create_task(
            handler.speak_guarded_utterance(
                "Hello. This is a portrait preview. Notice the pauses.",
                assistant_kind="character",
                character_ref=character,
                expected_destination_fingerprint=destination.fingerprint,
                validator=lambda: (
                    epoch == self._epoch
                    and self._current(authority)
                    and self.draft is draft
                ),
                presentation=self._presentation,
            )
        )
        view.sample_running = True
        view.set_busy(True)
        try:
            played = await task
            if epoch == self._epoch and self._current(authority):
                view.status(
                    "Speech sample finished. Review poses before Apply."
                    if played
                    else "Speech sample unavailable or interrupted. Offline review remains available."
                )
            return played
        finally:
            if self._speech_task is task:
                self._speech_task = None
            view.sample_running = False

    async def close(self) -> None:
        """Drain any actual native callback before retiring this screen owner."""
        self.discard()
        if self._task is not None and self._task is not asyncio.current_task():
            await asyncio.gather(self._task, return_exceptions=True)
