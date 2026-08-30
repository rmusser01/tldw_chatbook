"""Controller-owned Console image-generation and H3 lifecycle policy."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import threading
from typing import Any

from loguru import logger

from ...Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from ...Chat.console_chat_fork import (
    ConsoleForkImageSelectionFence,
    fingerprint_console_fork_selected_image,
)
from ...Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from ...Chat.console_command_grammar import CommandParse
from ...Chat.console_generate_image import (
    GenerationRefusal,
    LLMContextOptions,
    PreparedGeneration,
    clamp_initial_batch,
    generation_content_marker,
    parse_generate_image_args,
    prepare_generation_request,
    run_generation_batch,
)
from ...Chat.console_image_edit_operations import (
    ImageEditCompletion,
    ImageEditFailureNotice,
    ImageEditOperationRegistry,
)
from ...Chat.console_image_view import (
    ConsoleImageRowSpec,
    extract_image_urls,
    next_view_mode,
    resolve_render_remote_images,
)
from ...Widgets.Console.console_generation_card import ConsoleGenerationCardSpec
from ...Image_Generation.config import get_image_generation_config
from ...Image_Generation.listing import list_image_models_for_catalog

REMOTE_IMAGE_SCAN_WINDOW = 20
REMOTE_IMAGE_MAX_BYTES = 8 * 1024 * 1024
REMOTE_IMAGE_FETCH_ATTEMPT_LIMIT = 256


class ConsoleImageController:
    """Own non-DOM image-generation and H3 lifecycle behavior."""

    _CONSOLE_PENDING_STASH_ATTR = "_console_pending_attachment_stash"
    _H3_FAILURE_GUIDANCE_COPY = frozenset(
        {
            "ComfyUI image edits require one valid in-memory PNG, JPEG, or WebP image.",
            "The source image could not be uploaded. Please try again.",
            "The image-edit operation did not complete. Please try again.",
            "The edited image could not be saved locally. The source remains staged.",
        }
    )

    def __init__(
        self,
        screen: Any,
        *,
        app_instance: Any,
        ensure_console_image_view: Callable[[], tuple[Any, Any]],
        recent_console_image_messages: Callable[[Any], list[Any]],
        console_image_default_mode: Callable[[], str | None],
        console_generation_browse: Callable[[], dict[str, int]],
        sync_native_console_chat_ui: Callable[[], Any],
        ensure_console_chat_store: Callable[[], Any],
        build_console_provider_selection: Callable[[], Any],
        ensure_console_provider_gateway: Callable[[], Any],
        console_image_preparing: Callable[[], set[str] | None],
        current_console_chat_store: Callable[[], ConsoleChatStore | None],
        console_composer_or_none: Callable[[], Any | None],
        console_visible_draft_session_id: Callable[[], str | None],
        append_native_console_system_message: Callable[..., Any],
        request_console_control_bar_sync: Callable[[], None],
        default_console_session_settings: Callable[[], Any],
        clear_console_composer_draft: Callable[[], None],
    ) -> None:
        self._screen = screen
        self.app_instance = app_instance
        self._ensure_console_image_view_fn = ensure_console_image_view
        self._recent_console_image_messages_fn = recent_console_image_messages
        self._console_image_default_mode_fn = console_image_default_mode
        self._console_generation_browse_fn = console_generation_browse
        self._sync_native_console_chat_ui_fn = sync_native_console_chat_ui
        self._ensure_console_chat_store_fn = ensure_console_chat_store
        self._build_console_provider_selection_fn = build_console_provider_selection
        self._ensure_console_provider_gateway_fn = ensure_console_provider_gateway
        self._console_image_preparing_fn = console_image_preparing
        self._current_console_chat_store_fn = current_console_chat_store
        self._console_composer_or_none_fn = console_composer_or_none
        self._console_visible_draft_session_id_fn = console_visible_draft_session_id
        self._append_native_console_system_message_fn = (
            append_native_console_system_message
        )
        self._request_console_control_bar_sync_fn = request_console_control_bar_sync
        self._default_console_session_settings_fn = default_console_session_settings
        self._clear_console_composer_draft_fn = clear_console_composer_draft
        self._imagegen_inflight_sessions: set[str] = set()
        self._imagegen_inflight_message_ids: set[str] = set()
        self._console_h3_ui_generations: dict[str, str] = {}
        self._remote_image_fetch_attempts: OrderedDict[str, None] = OrderedDict()
        self._fork_image_browse_revisions: dict[str, int] = {}

    def _ensure_console_image_view(self) -> tuple[Any, Any]:
        return self._ensure_console_image_view_fn()

    def _recent_console_image_messages(self, messages: Any) -> list[Any]:
        return self._recent_console_image_messages_fn(messages)

    @property
    def _console_image_default_mode(self) -> str | None:
        return self._console_image_default_mode_fn()

    def _console_generation_browse(self) -> dict[str, int]:
        return self._console_generation_browse_fn()

    def _bump_fork_image_browse_revision(self, message_id: str) -> None:
        self._fork_image_browse_revisions[message_id] = (
            self._fork_image_browse_revisions.get(message_id, 0) + 1
        )

    def capture_console_fork_image_selections(
        self,
        messages: Sequence[ConsoleChatMessage],
    ) -> tuple[ConsoleForkImageSelectionFence, ...]:
        """Capture selected generated-image facts for one fork prefix."""

        browse = self._console_generation_browse()
        selections: list[ConsoleForkImageSelectionFence] = []
        for message in messages:
            metadata = message.generation_metadata
            if not metadata:
                continue
            if len(metadata) != len(message.attachments):
                raise ValueError("Fork generated image metadata is unavailable.")
            position = browse.get(message.id, 0)
            if type(position) is not int or not 0 <= position < len(metadata):
                raise ValueError("Fork generated image selection is unavailable.")
            selections.append(
                ConsoleForkImageSelectionFence(
                    native_message_id=message.id,
                    selected_position=position,
                    browse_revision=self._fork_image_browse_revisions.get(
                        message.id, 0
                    ),
                    attachment_meta_fingerprint=(
                        fingerprint_console_fork_selected_image(
                            message.attachments[position],
                            metadata[position],
                        )
                    ),
                )
            )
        return tuple(selections)

    def validate_console_fork_image_selections(
        self,
        messages: Sequence[ConsoleChatMessage],
        expected: Sequence[ConsoleForkImageSelectionFence],
    ) -> bool:
        """Return whether current generated-image choices exactly match a capture."""

        try:
            return self.capture_console_fork_image_selections(messages) == tuple(
                expected
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def invalidate_console_fork_image_selections(
        self,
        message_ids: Sequence[str],
    ) -> None:
        """Invalidate and clean browse state for removed messages or subtrees."""

        browse = self._console_generation_browse()
        for message_id in dict.fromkeys(message_ids):
            browse.pop(message_id, None)
            self._bump_fork_image_browse_revision(message_id)

    async def _sync_native_console_chat_ui(self) -> None:
        await self._sync_native_console_chat_ui_fn()

    def _ensure_console_chat_store(self) -> Any:
        return self._ensure_console_chat_store_fn()

    @property
    def _console_chat_store(self) -> ConsoleChatStore | None:
        return self._current_console_chat_store_fn()

    def _console_composer_or_none(self) -> Any | None:
        return self._console_composer_or_none_fn()

    @property
    def _console_visible_draft_session_id(self) -> str | None:
        return self._console_visible_draft_session_id_fn()

    async def _append_native_console_system_message(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return await self._append_native_console_system_message_fn(*args, **kwargs)

    def _request_console_control_bar_sync(self) -> None:
        self._request_console_control_bar_sync_fn()

    def _build_console_image_specs(
        self, messages: Any
    ) -> dict[str, ConsoleImageRowSpec]:
        """Build image-row payloads for prepared, visible image messages."""
        state, cache = self._ensure_console_image_view()
        default_mode = self._console_image_default_mode
        specs: dict[str, ConsoleImageRowSpec] = {}
        for message in self._recent_console_image_messages(messages):
            mode = state.mode_for(message.id, default=default_mode)
            if mode == "hidden":
                continue
            pil = cache.get_pil(message.id)
            if pil is None:
                continue
            specs[message.id] = ConsoleImageRowSpec(
                message_id=message.id,
                mode=mode,
                pixels=cache.get_pixels(message.id) if mode == "pixels" else None,
                pil=pil if mode == "graphics" else None,
            )
        self._extend_specs_with_remote_images(messages, specs, state, cache)
        return specs

    def _extend_specs_with_remote_images(
        self,
        messages: Any,
        specs: dict[str, ConsoleImageRowSpec],
        state: Any,
        cache: Any,
    ) -> None:
        """Add egress-hardened remote image rows when explicitly enabled."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        if not resolve_render_remote_images(app_config):
            return
        default_mode = self._console_image_default_mode
        for message in messages[-REMOTE_IMAGE_SCAN_WINDOW:]:
            if message.role is not ConsoleMessageRole.ASSISTANT:
                continue
            if message.id in specs or message.status == "failed":
                continue
            urls = extract_image_urls(message.content or "", limit=1)
            if not urls:
                continue
            url = urls[0]
            key = f"remote:{url}"
            pil = cache.get_pil(key)
            if pil is not None:
                mode = state.mode_for(message.id, default=default_mode)
                if mode != "hidden":
                    specs[message.id] = ConsoleImageRowSpec(
                        message_id=message.id,
                        mode=mode,
                        pixels=cache.get_pixels(key) if mode == "pixels" else None,
                        pil=pil if mode == "graphics" else None,
                    )
                continue
            if cache.is_failed(key) or url in self._remote_image_fetch_attempts:
                continue
            self._remote_image_fetch_attempts[url] = None
            self._remote_image_fetch_attempts.move_to_end(url)
            if (
                len(self._remote_image_fetch_attempts)
                > REMOTE_IMAGE_FETCH_ATTEMPT_LIMIT
            ):
                self._remote_image_fetch_attempts.popitem(last=False)
            self._screen.run_worker(
                self._fetch_remote_transcript_image(url, key),
                group="console-remote-image-fetch",
                exclusive=False,
                exit_on_error=False,
            )

    async def _fetch_remote_transcript_image(self, url: str, cache_key: str) -> None:
        """Fetch and cache one linked image without raising into its worker."""
        try:
            from ...Image_Generation.adapters.image_format_utils import (
                fetch_image_bytes,
            )

            data, content_type = await asyncio.to_thread(
                fetch_image_bytes,
                url,
                timeout=20,
                max_bytes=REMOTE_IMAGE_MAX_BYTES,
            )
            if content_type and not str(content_type).split(";")[
                0
            ].strip().lower().startswith("image/"):
                return
            _state, cache = self._ensure_console_image_view()
            prepared = await asyncio.to_thread(cache.prepare, cache_key, data)
            if prepared and self._screen.is_mounted:
                await self._sync_native_console_chat_ui()
        except Exception as exc:
            logger.warning(
                "remote transcript image fetch failed (exception_type={})",
                type(exc).__name__,
            )

    def _build_generation_card_specs(
        self, messages: Any
    ) -> dict[str, ConsoleGenerationCardSpec]:
        """Build generation-card payloads for every visible generation message."""
        state, cache = self._ensure_console_image_view()
        default_mode = self._console_image_default_mode
        browse = self._console_generation_browse()
        specs: dict[str, ConsoleGenerationCardSpec] = {}
        for message in messages:
            metadata = getattr(message, "generation_metadata", ())
            if not metadata:
                continue
            mode = state.mode_for(message.id, default=default_mode)
            browsed_index = browse.get(message.id, 0)
            if not 0 <= browsed_index < len(metadata):
                browsed_index = 0
            cache_key = f"{message.id}:{browsed_index}"
            specs[message.id] = ConsoleGenerationCardSpec(
                message_id=message.id,
                browsed_index=browsed_index,
                variant_count=len(metadata),
                meta=metadata[browsed_index],
                mode=mode,
                pixels=cache.get_pixels(cache_key) if mode == "pixels" else None,
                pil=cache.get_pil(cache_key) if mode == "graphics" else None,
            )
        return specs

    def _pending_console_generation_card_images(
        self,
        messages: Any,
        card_specs: Mapping[str, ConsoleGenerationCardSpec],
    ) -> list[tuple[str, bytes]]:
        """Return uncached browsed generation variants that still need decoding."""
        _state, cache = self._ensure_console_image_view()
        by_id = {message.id: message for message in messages}
        pending: list[tuple[str, bytes]] = []
        for message_id, spec in card_specs.items():
            if spec.mode == "hidden":
                continue
            message = by_id.get(message_id)
            attachments = getattr(message, "attachments", ()) or () if message else ()
            if not 0 <= spec.browsed_index < len(attachments):
                continue
            data = attachments[spec.browsed_index].data
            cache_key = f"{message_id}:{spec.browsed_index}"
            if (
                data is not None
                and cache.get_pil(cache_key) is None
                and not cache.is_failed(cache_key)
            ):
                pending.append((cache_key, data))
        return pending

    def _handle_console_toggle_image_view(self, message_id: str) -> None:
        """Cycle one message's inline-image view mode."""
        state, _cache = self._ensure_console_image_view()
        default_mode = self._console_image_default_mode
        current = state.mode_for(message_id, default=default_mode)
        state.set_mode(message_id, next_view_mode(current), default=default_mode)

    def _console_imagegen_inflight_sessions(self) -> set[str]:
        """Return sessions with an ordinary generation batch in flight."""
        return self._imagegen_inflight_sessions

    def _console_imagegen_inflight_message_ids(self) -> set[str]:
        """Return messages with a regenerate append in flight."""
        return self._imagegen_inflight_message_ids

    def _console_generate_image_conversation_pairs(
        self, store: Any, session_id: str
    ) -> list[tuple[str, str]]:
        """Return completed non-empty turns for conversation prompt composition."""
        return [
            (
                message.role.value
                if hasattr(message.role, "value")
                else str(message.role),
                message.content,
            )
            for message in store.messages_for_session(session_id)
            if message.status == "complete"
            and message.content
            and message.content.strip()
        ]

    async def _console_generate_image_llm_context_options(
        self, cfg: Any
    ) -> LLMContextOptions:
        """Resolve the active chat provider for conversation prompt composition."""
        if not cfg.context_llm_enabled:
            return LLMContextOptions(
                enabled=False,
                turns=cfg.context_llm_turns,
                timeout_seconds=cfg.context_llm_timeout_seconds,
                provider_ready=False,
            )
        try:
            selection = self._build_console_provider_selection_fn()
            gateway = self._ensure_console_provider_gateway_fn()
            resolution = await gateway.resolve_for_send(selection)
        except Exception as exc:  # noqa: BLE001 - context composition is optional
            logger.debug(
                "generate-image provider resolution for LLM context failed (exception_type={})",
                type(exc).__name__,
            )
            return LLMContextOptions(
                enabled=True,
                turns=cfg.context_llm_turns,
                timeout_seconds=cfg.context_llm_timeout_seconds,
                provider_ready=False,
            )
        return LLMContextOptions(
            enabled=True,
            turns=cfg.context_llm_turns,
            timeout_seconds=cfg.context_llm_timeout_seconds,
            provider_ready=resolution.ready,
            api_endpoint=resolution.execution_key or None,
            model=resolution.model,
            api_key=resolution.api_key,
        )

    async def _regenerate_console_generation_variant(self, message_id: str) -> None:
        """Append one random-seed variant to an existing generation message."""
        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.", severity="warning"
            )
            return
        if not message.generation_metadata:
            return
        base_meta = message.generation_metadata[0]
        if base_meta.params.get("operation") == "edit":
            self.app_instance.notify(
                "Image edits cannot be regenerated. Restage the source image and run "
                "/generate-image :comfyui again.",
                severity="warning",
            )
            return
        inflight = self._console_imagegen_inflight_message_ids()
        if message_id in inflight:
            self.app_instance.notify(
                "An image generation is already running for this message.",
                severity="warning",
            )
            return
        cfg = get_image_generation_config()
        if len(message.generation_metadata) >= cfg.max_variants_per_message:
            self.app_instance.notify(
                f"Already at the maximum of {cfg.max_variants_per_message} "
                "variants for this message.",
                severity="warning",
            )
            return
        inflight.add(message_id)
        try:
            batch = await asyncio.to_thread(
                run_generation_batch,
                backend=base_meta.backend,
                prompt=base_meta.prompt,
                negative_prompt=base_meta.negative_prompt or None,
                seed=-1,
                count=1,
                style_name=base_meta.style,
            )
            if not batch.successes:
                detail = "; ".join(batch.errors) or "unknown error"
                self.app_instance.notify(
                    f"Image regeneration failed: {detail}", severity="error"
                )
                return
            data, mime_type, meta = batch.successes[0]
            session_id = store.session_id_for_message(message_id)
            position = store.append_generation_variant(
                session_id,
                message_id,
                data=data,
                mime_type=mime_type,
                meta=meta,
                persist=True,
            )
            self._console_generation_browse()[message_id] = position
            self._bump_fork_image_browse_revision(message_id)
            await self._sync_native_console_chat_ui()
        finally:
            inflight.discard(message_id)

    def _select_console_generation_variant(
        self, message: ConsoleChatMessage, *, direction: str
    ) -> None:
        """Move a generation message's ephemeral browsed index when in bounds."""
        variant_count = len(message.generation_metadata)
        if variant_count <= 1:
            return
        browse = self._console_generation_browse()
        current = browse.get(message.id, 0)
        if not 0 <= current < variant_count:
            current = 0
        if direction == "variant-previous":
            target = current - 1
        elif direction == "variant-next":
            target = current + 1
        else:
            return
        if 0 <= target < variant_count:
            browse[message.id] = target
            self._bump_fork_image_browse_revision(message.id)

    def _keep_console_generation_variant(self, message: ConsoleChatMessage) -> None:
        """Promote the browsed variant to canonical and evict stale renders."""
        if not message.generation_metadata:
            return
        browse = self._console_generation_browse()
        browsed_index = browse.get(message.id, 0)
        if not 0 < browsed_index < len(message.generation_metadata):
            return
        store = self._ensure_console_chat_store()
        session_id = store.session_id_for_message(message.id)
        variant_count = len(message.generation_metadata)
        store.keep_generation_variant(
            session_id, message.id, position=browsed_index, persist=True
        )
        browse[message.id] = 0
        self._bump_fork_image_browse_revision(message.id)
        _state, cache = self._ensure_console_image_view()
        stale_keys = [f"{message.id}:{index}" for index in range(variant_count)]
        cache.evict_session(stale_keys)
        preparing = self._console_image_preparing_fn()
        if preparing is not None:
            preparing.difference_update(stale_keys)

    def _h3_image_edit_registry(self) -> ImageEditOperationRegistry:
        """Return the app-owned H3 operation registry."""
        registry = getattr(self.app_instance, "console_image_edit_operations", None)
        if not isinstance(registry, ImageEditOperationRegistry):
            registry = ImageEditOperationRegistry()
            self.app_instance.console_image_edit_operations = registry
        return registry

    @staticmethod
    def _h3_reference_snapshot(pending: Any) -> tuple[str, bytes, str]:
        """Snapshot immutable source identity and bytes without decoding."""
        from ...Image_Generation.request_validation import (
            IMAGE_GEN_REFERENCE_MAX_BYTES,
        )

        if getattr(pending, "file_type", None) != "image":
            raise ValueError("source_type")
        attachment_id = getattr(pending, "attachment_id", None)
        if type(attachment_id) is not str or not attachment_id:
            raise ValueError("source_identity")
        data = getattr(pending, "data", None)
        if type(data) is not bytes or not data:
            raise ValueError("source_content")
        if len(data) > IMAGE_GEN_REFERENCE_MAX_BYTES:
            raise ValueError("source_size")
        mime_type = str(getattr(pending, "mime_type", "") or "").lower()
        return attachment_id, data, mime_type

    @staticmethod
    def _h3_reference_from_snapshot(snapshot: tuple[str, bytes, str]):
        """Read source header metadata without duplicating canonical policy."""
        from io import BytesIO

        from PIL import Image as PILImage

        from ...Image_Generation.capabilities import ResolvedReferenceImage

        attachment_id, data, mime_type = snapshot
        try:
            with PILImage.open(BytesIO(data)) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError("source_header") from exc
        return ResolvedReferenceImage(
            file_id=attachment_id,
            filename=None,
            mime_type=mime_type,
            width=width,
            height=height,
            bytes_len=len(data),
            content=data,
            temp_path=None,
        )

    def _filter_h3_attachment_from_app_stash(
        self, session_id: str, attachment_id: str
    ) -> None:
        """Drop only the initiating source from the app's remount stash."""
        app = getattr(self, "app_instance", None)
        if app is None:
            return
        stash = getattr(app, self._CONSOLE_PENDING_STASH_ATTR, None)
        if not isinstance(stash, dict):
            return
        pendings = stash.get(session_id)
        if not isinstance(pendings, (list, tuple)):
            return
        filtered = tuple(
            pending
            for pending in pendings
            if getattr(pending, "attachment_id", None) != attachment_id
        )
        if filtered:
            stash[session_id] = filtered
        else:
            stash.pop(session_id, None)

    def _h3_origin_screen_is_live(self, generation: str) -> bool:
        """Return whether this exact screen/generation may update live UI."""
        current = getattr(
            self.app_instance, "_console_h3_image_edit_screen", self._screen
        )
        terminal = getattr(self._screen, "_console_h3_terminal_generations", set())
        return current is self._screen and generation not in terminal

    def _cleanup_h3_completion_in_store(
        self,
        store: ConsoleChatStore,
        completion: ImageEditCompletion,
        *,
        clear_visible_composer: bool,
    ) -> bool:
        """Hydrate durable success, then perform exact identity cleanup."""
        session = next(
            (
                candidate
                for candidate in store.sessions()
                if candidate.id == completion.session_id
            ),
            None,
        )
        if session is None:
            return False
        try:
            message = store.merge_persisted_generation_message(
                completion.session_id, completion.message_id
            )
        except Exception:  # noqa: BLE001 - keep cleanup pending for later retry
            return False
        if message is None:
            return False

        try:
            store.consume_pending_attachment(
                completion.session_id, completion.attachment_id
            )
            remaining = store.pending_attachments(completion.session_id)
            if any(
                pending.attachment_id == completion.attachment_id
                for pending in remaining
            ):
                return False
            if store.session_draft(completion.session_id) == completion.captured_draft:
                store.set_session_draft(completion.session_id, "")
        except Exception as exc:  # noqa: BLE001 - committed success is retained
            logger.bind(
                component="image_edit",
                phase="persistence",
                error_type=type(exc).__name__,
            ).error("Console image edit cleanup failed")
            return False

        self._filter_h3_attachment_from_app_stash(
            completion.session_id, completion.attachment_id
        )
        if clear_visible_composer:
            try:
                composer = self._console_composer_or_none()
            except Exception:  # noqa: BLE001 - mounted UI cleanup is retryable
                composer = None
            try:
                if (
                    composer is not None
                    and store.active_session_id == completion.session_id
                    and self._console_visible_draft_session_id == completion.session_id
                    and composer.draft_text() == completion.captured_draft
                ):
                    composer.clear_draft()
            except Exception:  # noqa: BLE001 - retain completion for a later screen
                return False
        return True

    def _reconcile_h3_image_edit_completions(
        self, store: ConsoleChatStore | None = None
    ) -> None:
        """Adopt byte-free durable H3 outcomes into a current Console store."""
        registry = self._h3_image_edit_registry()
        store = store or self._console_chat_store
        if store is None:
            return
        live_session_ids = {session.id for session in store.sessions()}
        for completion in registry.completions():
            if completion.session_id not in live_session_ids:
                continue
            if self._cleanup_h3_completion_in_store(
                store, completion, clear_visible_composer=True
            ):
                registry.ack_completion(completion.session_id, completion.generation)
        for notice in registry.failure_notices():
            if notice.session_id not in live_session_ids:
                continue
            if self._merge_h3_failure_notice_in_store(store, notice):
                registry.ack_failure_notice(notice.session_id, notice.generation)

    def _merge_h3_failure_notice_in_store(
        self,
        store: ConsoleChatStore,
        notice: ImageEditFailureNotice,
    ) -> bool:
        """Idempotently hydrate one exact privacy-safe durable system row."""
        session = next(
            (
                candidate
                for candidate in store.sessions()
                if candidate.id == notice.session_id
            ),
            None,
        )
        if session is None:
            return False
        try:
            existing = next(
                (
                    message
                    for message in store.messages_for_session(notice.session_id)
                    if message.persisted_message_id == notice.message_id
                ),
                None,
            )
        except KeyError:
            return False
        if existing is not None:
            return (
                existing.role is ConsoleMessageRole.SYSTEM
                and existing.content in self._H3_FAILURE_GUIDANCE_COPY
            )
        try:
            recovered = store.merge_persisted_system_message(
                notice.session_id,
                notice.message_id,
                allowed_content=frozenset(self._H3_FAILURE_GUIDANCE_COPY),
            )
        except Exception:  # noqa: BLE001 - retain notice for a later retry
            return False
        return recovered is not None

    async def _settle_current_h3_outcome(
        self, session_id: str, generation: str
    ) -> None:
        """Settle one terminal H3 outcome on its current adopted screen."""
        if getattr(
            self.app_instance, "_console_h3_image_edit_screen", None
        ) is not self._screen or generation in getattr(
            self._screen, "_console_h3_terminal_generations", set()
        ):
            return
        store = self._console_chat_store
        if store is None or store.active_session_id != session_id:
            return
        ui_generation = getattr(self, "_console_h3_ui_generations", {}).get(session_id)
        if ui_generation is not None and ui_generation != generation:
            return
        if self._h3_image_edit_registry().active(session_id) is not None:
            return
        if not any(session.id == session_id for session in store.sessions()):
            return

        self._reconcile_h3_image_edit_completions(store)
        try:
            await self._sync_native_console_chat_ui()
        finally:
            self._request_console_control_bar_sync()

    def _schedule_current_h3_settlement(self, session_id: str, generation: str) -> None:
        """Schedule terminal settlement only on the currently mounted Console."""
        from functools import partial

        current = getattr(self.app_instance, "_console_h3_image_edit_screen", None)
        if current is None or not getattr(current, "_is_mounted", False):
            return
        schedule = getattr(current, "call_after_refresh", None)
        if callable(schedule):
            schedule(
                partial(
                    current._image._settle_current_h3_outcome,
                    session_id,
                    generation,
                )
            )

    async def _append_h3_image_edit_error(
        self,
        *,
        session_id: str,
        generation: str,
        phase: str,
        error_type: str,
        copy: str,
    ) -> None:
        """Append safe failure copy without touching a terminal screen's UI."""
        logger.bind(component="image_edit", phase=phase, error_type=error_type).error(
            "Console image edit failed"
        )
        store = self._console_chat_store
        if store is None:
            return
        persisted_message_id: str | None = None
        try:
            message = store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=copy,
                persist=True,
            )
            if (
                type(message.persisted_message_id) is str
                and message.persisted_message_id
            ):
                persisted_message_id = message.persisted_message_id
        except KeyError:
            return
        except Exception as exc:  # noqa: BLE001 - preserve the primary failure
            logger.bind(
                component="image_edit",
                phase="failure_guidance_persistence",
                error_type=type(exc).__name__,
            ).error("Console image edit failure guidance persistence failed")
            try:
                guidance_present = any(
                    message.role is ConsoleMessageRole.SYSTEM
                    and message.content == copy
                    for message in store.messages_for_session(session_id)
                )
                if not guidance_present:
                    store.append_message(
                        session_id,
                        role=ConsoleMessageRole.SYSTEM,
                        content=copy,
                    )
            except Exception:  # noqa: BLE001 - best-effort in-memory fallback
                return
        if persisted_message_id is not None:
            notice = ImageEditFailureNotice(
                session_id=session_id,
                generation=generation,
                message_id=persisted_message_id,
            )
            if self._h3_image_edit_registry().publish_failure_notice(notice):
                if self._h3_origin_screen_is_live(generation):
                    self._reconcile_h3_image_edit_completions(store)
        ui_generations = getattr(self, "_console_h3_ui_generations", {})
        if ui_generations.get(
            session_id
        ) == generation and self._h3_origin_screen_is_live(generation):
            await self._sync_native_console_chat_ui()

    async def _run_h3_image_edit_command(
        self,
        *,
        args: Any,
        cfg: Any,
        store: ConsoleChatStore,
        session: ConsoleChatSession,
    ) -> None:
        """Validate, own, persist, and reconcile one ComfyUI H3 edit."""
        from functools import partial

        from ...Image_Generation import worker as image_worker
        from ...Image_Generation.exceptions import (
            ComfyUIImageEditError,
            ImageGenerationCancelled,
        )

        if args.style is not None:
            await self._append_native_console_system_message(
                "ComfyUI image edits do not support style tokens.",
                session_id=session.id,
            )
            return
        instruction = args.prompt
        if not instruction.strip():
            await self._append_native_console_system_message(
                "ComfyUI image edits require a non-empty instruction.",
                session_id=session.id,
            )
            return
        pendings = store.pending_attachments(session.id)
        if len(pendings) != 1:
            await self._append_native_console_system_message(
                "ComfyUI image edits require exactly one staged image.",
                session_id=session.id,
            )
            return
        pending = pendings[0]
        try:
            snapshot = self._h3_reference_snapshot(pending)
        except (TypeError, ValueError):
            await self._append_native_console_system_message(
                "ComfyUI image edits require one valid in-memory PNG, JPEG, or WebP image.",
                session_id=session.id,
            )
            return
        composer = self._console_composer_or_none()
        captured_draft = (
            composer.draft_text()
            if composer is not None
            else store.session_draft(session.id)
        )
        cancel_event = threading.Event()
        registry = self._h3_image_edit_registry()
        sampler = getattr(cfg, "comfyui_image_default_sampler", None)
        build_request = partial(image_worker.build_request, sampler=sampler)

        async def _owned(generation: str) -> None:
            def _prepare_and_run():
                reference = self._h3_reference_from_snapshot(snapshot)
                if cancel_event.is_set():
                    raise ImageGenerationCancelled()
                return run_generation_batch(
                    backend="comfyui",
                    prompt=instruction,
                    negative_prompt=None,
                    seed=getattr(cfg, "comfyui_image_default_seed", None),
                    count=1,
                    style_name=None,
                    width=None,
                    height=None,
                    steps=getattr(cfg, "comfyui_image_default_steps", None),
                    cfg_scale=None,
                    reference_image=reference,
                    cancel_event=cancel_event,
                    build=build_request,
                )

            try:
                batch = await asyncio.to_thread(_prepare_and_run)
            except ImageGenerationCancelled:
                return
            except (TypeError, ValueError) as exc:
                await self._append_h3_image_edit_error(
                    session_id=session.id,
                    generation=generation,
                    phase="source_validation",
                    error_type=type(exc).__name__,
                    copy=(
                        "ComfyUI image edits require one valid in-memory PNG, "
                        "JPEG, or WebP image."
                    ),
                )
                return
            except ComfyUIImageEditError as exc:
                await self._append_h3_image_edit_error(
                    session_id=session.id,
                    generation=generation,
                    phase=exc.phase,
                    error_type=type(exc).__name__,
                    copy=str(exc),
                )
                return
            except Exception as exc:  # noqa: BLE001 - normalized below
                await self._append_h3_image_edit_error(
                    session_id=session.id,
                    generation=generation,
                    phase="history_polling",
                    error_type=type(exc).__name__,
                    copy="The image-edit operation did not complete. Please try again.",
                )
                return
            if not batch.successes:
                await self._append_h3_image_edit_error(
                    session_id=session.id,
                    generation=generation,
                    phase="history_polling",
                    error_type="ImageGenerationError",
                    copy="The image-edit operation did not complete. Please try again.",
                )
                return

            before_ids = {
                message.id for message in store.messages_for_session(session.id)
            }
            try:
                message = store.append_generation_message(
                    session.id,
                    content=generation_content_marker(instruction),
                    variants=batch.successes,
                    persist=True,
                )
                persisted_message_id = message.persisted_message_id
                if not persisted_message_id:
                    raise RuntimeError("durable image message missing")
            except Exception as exc:  # noqa: BLE001 - normalized below
                for candidate in store.messages_for_session(session.id):
                    if candidate.id not in before_ids:
                        try:
                            store.delete_message(candidate.id)
                        except (KeyError, RuntimeError, ValueError):
                            pass
                await self._append_h3_image_edit_error(
                    session_id=session.id,
                    generation=generation,
                    phase="persistence",
                    error_type=type(exc).__name__,
                    copy=(
                        "The edited image could not be saved locally. "
                        "The source remains staged."
                    ),
                )
                return

            completion = ImageEditCompletion(
                session_id=session.id,
                generation=generation,
                message_id=persisted_message_id,
                attachment_id=pending.attachment_id,
                captured_draft=captured_draft,
            )
            registry.publish_completion(completion)
            self._filter_h3_attachment_from_app_stash(session.id, pending.attachment_id)
            cleaned = self._cleanup_h3_completion_in_store(
                store,
                completion,
                clear_visible_composer=self._h3_origin_screen_is_live(generation),
            )
            if cleaned and self._h3_origin_screen_is_live(generation):
                registry.ack_completion(session.id, generation)

        operation = registry.start(
            session_id=session.id,
            attachment_id=pending.attachment_id,
            captured_draft=captured_draft,
            cancel_event=cancel_event,
            runner=_owned,
            on_settled=lambda generation: self._schedule_current_h3_settlement(
                session.id, generation
            ),
        )
        if operation is None:
            await self._append_native_console_system_message(
                "An image edit is already running for this session.",
                session_id=session.id,
            )
            return
        ui_generations = getattr(self, "_console_h3_ui_generations", None)
        if ui_generations is None:
            ui_generations = {}
            self._console_h3_ui_generations = ui_generations
        ui_generations[session.id] = operation.generation
        if self._h3_origin_screen_is_live(operation.generation):
            self._request_console_control_bar_sync()

    async def _console_command_generate_image(self, parse: CommandParse) -> None:
        """Resolve and run one `/generate-image` batch.

        Grammar: ``[:backend] [@style] <prompt>`` (tokens in any order), or
        no prompt at all to generate from the session's conversation
        context. `prepare_generation_request` (`Chat/console_generate_image.py`)
        owns all of that decision logic -- style-token resolution, prompt
        composition against a template, and the conversation-context
        fallback (optionally LLM-composed, Task-559 AC1) -- so it stays
        independently unit-testable; this handler just executes its result.

        Refusals (a `GenerationRefusal` from `prepare_generation_request`,
        no resolvable/configured backend, or a batch already running for
        this session) leave the composer draft untouched, mirroring
        `/system`'s no-system-part behavior, and never touch the store.
        Once a batch is actually going to run the draft is cleared up front
        (this IS the "successful dispatch" point — not "successful
        generation"): the blocking batch loop (`run_generation_batch`) then
        runs off the UI loop via `asyncio.to_thread`, exactly like
        `_prep_console_images`. On the zero-success path the original draft
        is restored so the user can edit and retry -- and the same restore
        happens if `run_generation_batch` RAISES outright instead of
        returning a zero-success `BatchResult` (an `except Exception`
        around the whole batch call/append/sync sequence; the failure is
        reported as a system message either way). One or more successes
        append a single generation message via
        `ConsoleChatStore.append_generation_message` with a trailing
        partial status line when some variants failed. The in-flight guard
        is always cleared in a `finally`, so a crashed/cancelled batch
        never wedges the session against further `/generate-image`
        commands.

        `prepare_generation_request` itself now also runs via
        `asyncio.to_thread` (not called directly on the UI loop, unlike
        before this AC): on the no-prompt path it may attempt an LLM call
        (`compose_llm_context_prompt`) to compose a richer prompt from
        conversation context, which is blocking network I/O and must never
        run on the event loop -- exactly the same offloading rule
        `run_generation_batch` already follows below. The provider identity
        for that call is resolved first, on the loop, via
        `_console_generate_image_llm_context_options` -- the same cheap
        resolution a normal Console send does at this point (config/env
        for most providers; llama.cpp additionally does its own bounded
        ``/health`` reachability probe there, same as a normal send).
        """
        args = parse_generate_image_args(parse.args)
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings_fn(),
        )
        cfg = get_image_generation_config()
        backend = args.backend or cfg.default_backend
        if backend == "comfyui":
            catalog = list_image_models_for_catalog()
            entry = next(
                (item for item in catalog if item.get("name") == backend), None
            )
            if entry is None or not entry.get("is_configured"):
                await self._append_native_console_system_message(
                    "Image backend 'comfyui' is not enabled/configured. "
                    "Check [image_generation] settings.",
                    session_id=session.id,
                )
                return
            await self._run_h3_image_edit_command(
                args=args, cfg=cfg, store=store, session=session
            )
            return
        conversation_pairs = self._console_generate_image_conversation_pairs(
            store, session.id
        )
        llm_context: LLMContextOptions | None = None
        if not args.prompt.strip():
            llm_context = await self._console_generate_image_llm_context_options(cfg)
        prepared: PreparedGeneration | GenerationRefusal = await asyncio.to_thread(
            prepare_generation_request, args, conversation_pairs, llm_context
        )
        if isinstance(prepared, GenerationRefusal):
            # Task 4 (background-write audit): every append below threads
            # `session_id=session.id` explicitly -- this handler already
            # spans real await gaps (the two `asyncio.to_thread` calls
            # above/below), and `session` is this batch's owning session,
            # captured once at the top. Re-resolving "active" implicitly
            # (the old behavior) would misattribute the outcome to whatever
            # session the user switched to while the batch was running.
            await self._append_native_console_system_message(
                prepared.reason, session_id=session.id
            )
            return
        if not backend:
            await self._append_native_console_system_message(
                "No image generation backend configured. Set "
                "[image_generation].default_backend, or use "
                "/generate-image :backend <prompt>.",
                session_id=session.id,
            )
            return
        catalog = list_image_models_for_catalog()
        entry = next((item for item in catalog if item.get("name") == backend), None)
        if entry is None or not entry.get("is_configured"):
            await self._append_native_console_system_message(
                f"Image backend '{backend}' is not enabled/configured. "
                "Check [image_generation] settings.",
                session_id=session.id,
            )
            return
        inflight = self._console_imagegen_inflight_sessions()
        if session.id in inflight:
            await self._append_native_console_system_message(
                "An image generation is already running for this session.",
                session_id=session.id,
            )
            return
        inflight.add(session.id)
        # Capture draft before clearing so we can restore it on zero-success.
        composer = self._console_composer_or_none()
        saved_draft = composer.draft_text() if composer is not None else ""
        self._clear_console_composer_draft_fn()
        try:
            count = clamp_initial_batch(cfg.default_batch, cfg.max_variants_per_message)
            batch = await asyncio.to_thread(
                run_generation_batch,
                backend=backend,
                prompt=prepared.prompt,
                negative_prompt=prepared.negative_prompt,
                seed=None,
                count=count,
                style_name=prepared.style_name,
                width=prepared.width,
                height=prepared.height,
                steps=prepared.steps,
                cfg_scale=prepared.cfg_scale,
            )
            if not batch.successes:
                # Restore the saved draft so the user can edit and retry.
                if composer is not None and saved_draft:
                    composer.clear_draft()
                    composer.insert_text_as_paste(saved_draft)
                detail = "; ".join(batch.errors) or "unknown error"
                await self._append_native_console_system_message(
                    f"Image generation failed: {detail}", session_id=session.id
                )
                return
            store.append_generation_message(
                session.id,
                content=generation_content_marker(prepared.prompt),
                variants=batch.successes,
                persist=True,
            )
            if len(batch.successes) < count:
                store.append_message(
                    session.id,
                    role=ConsoleMessageRole.SYSTEM,
                    content=(
                        f"{len(batch.successes)}/{count} generated "
                        f"({'; '.join(batch.errors)})"
                    ),
                )
            await self._sync_native_console_chat_ui()
        except Exception as exc:  # noqa: BLE001 - reported to the user, never a bare app crash
            # task-558: `run_generation_batch` itself raising (as opposed to
            # catching a per-variant failure and returning it in
            # `batch.errors`, the zero-success path above) used to propagate
            # straight past the draft-restore logic -- the user's typed
            # prompt was gone with no way to recover it. Mirrors the
            # zero-success restore exactly.
            if composer is not None and saved_draft:
                composer.clear_draft()
                composer.insert_text_as_paste(saved_draft)
            logger.error(
                "Image generation batch raised (exception_type={})",
                type(exc).__name__,
            )
            await self._append_native_console_system_message(
                f"Image generation failed: {exc}", session_id=session.id
            )
        finally:
            inflight.discard(session.id)
