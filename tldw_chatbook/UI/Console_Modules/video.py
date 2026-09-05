"""Controller-owned Console generated-video lifecycle policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
import inspect
import os
from pathlib import Path
import threading
from typing import Any, Literal

from loguru import logger
from rich.markup import escape as escape_markup

from ...Chat.console_command_grammar import CommandParse
from ...Chat.console_generate_video import (
    GENERATE_VIDEO_USAGE_TEXT,
    PendingVideoArtifact,
    estimate_video_cost_text,
    is_paid_backend,
    parse_generate_video_args,
    run_video_generation,
)
from ...config import get_cli_setting
from ...Video_Generation.config import get_video_generation_config
from ...Video_Generation.video_formats import canonical_video_extension
from ...Video_Generation.video_store import (
    VideoCapacityExceeded,
    VideoPublicationGate,
    VideoStore,
    VideoStoreSaveError,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.Console.console_video_card import ConsoleVideoCardSpec


class ConsoleVideoController:
    """Own generated-video state, publication, and orchestration.

    The controller keeps video policy independent from Textual DOM access;
    screen-owned behavior is supplied through constructor callbacks.
    """

    def __init__(
        self,
        *,
        app_instance: Any,
        sync_native_console_chat_ui: Callable[[], Any],
        ensure_console_chat_store: Callable[[], Any],
        wait_for_console_screen_result: Callable[[Any], Any],
        open_video_with_os: Callable[[Path], None],
        append_native_console_system_message: Callable[..., Any],
        default_console_session_settings: Callable[[], Any],
        console_composer_or_none: Callable[[], Any | None],
        clear_console_composer_draft: Callable[[], None],
    ) -> None:
        """Build the generated-video controller.

        Args:
            app_instance: Application object used for settings and the shared
                generated-video store.
            sync_native_console_chat_ui: Refresh the native Console transcript.
            ensure_console_chat_store: Return the active Console chat store.
            wait_for_console_screen_result: Await or return a pushed-screen result.
            open_video_with_os: Present a generated video with the host OS.
            append_native_console_system_message: Append a system transcript row.
            default_console_session_settings: Return default session settings.
            console_composer_or_none: Return the mounted composer when available.
            clear_console_composer_draft: Clear the current composer draft.
        """
        self.app_instance = app_instance
        self._sync_native_console_chat_ui_fn = sync_native_console_chat_ui
        self._ensure_console_chat_store_fn = ensure_console_chat_store
        self._wait_for_console_screen_result_fn = wait_for_console_screen_result
        self._open_video_with_os_fn = open_video_with_os
        self._append_native_console_system_message_fn = (
            append_native_console_system_message
        )
        self._default_console_session_settings_fn = default_console_session_settings
        self._console_composer_or_none_fn = console_composer_or_none
        self._clear_console_composer_draft_fn = clear_console_composer_draft
        self._console_videogen_inflight: set[str] = set()
        self._console_videogen_cancels: dict[str, threading.Event] = {}
        self._console_video_store: VideoStore | None = None
        self._pending_video_artifacts: dict[str, PendingVideoArtifact] = {}
        self._pending_video_artifacts_closed = False
        self._pending_video_operation_cancels: dict[str, VideoPublicationGate] = {}
        self._pending_video_active_operations: dict[
            str, tuple[PendingVideoArtifact, int]
        ] = {}
        self._pending_video_deferred_closes: dict[str, PendingVideoArtifact] = {}

    async def _sync_native_console_chat_ui(self) -> None:
        result = self._sync_native_console_chat_ui_fn()
        if inspect.isawaitable(result):
            await result

    def _ensure_console_chat_store(self) -> Any:
        return self._ensure_console_chat_store_fn()

    async def _wait_for_console_screen_result(self, screen: Any) -> Any:
        result = self._wait_for_console_screen_result_fn(screen)
        return await result if inspect.isawaitable(result) else result

    def _open_video_with_os(self, path: Path) -> None:
        self._open_video_with_os_fn(path)

    async def _append_native_console_system_message(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        result = self._append_native_console_system_message_fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def _default_console_session_settings(self) -> Any:
        return self._default_console_session_settings_fn()

    def _console_composer_or_none(self) -> Any | None:
        return self._console_composer_or_none_fn()

    def _clear_console_composer_draft(self) -> None:
        self._clear_console_composer_draft_fn()

    def _console_videogen_inflight_sessions(self) -> set[str]:
        """Per-session guard mirroring ``_console_imagegen_inflight_sessions``."""
        inflight = getattr(self, "_console_videogen_inflight", None)
        if inflight is None:
            inflight = set()
            self._console_videogen_inflight = inflight
        return inflight

    def _console_videogen_cancel_events(self) -> dict[str, "threading.Event"]:
        """Session id -> cancel event for the in-flight video generation."""
        events = getattr(self, "_console_videogen_cancels", None)
        if events is None:
            events = {}
            self._console_videogen_cancels = events
        return events

    def _ensure_console_video_store(self) -> VideoStore:
        """Return an explicit test override or the app-owned VideoStore."""
        override = getattr(self, "_console_video_store", None)
        if override is not None:
            return override
        store = getattr(self.app_instance, "generated_video_store", None)
        if store is None:
            raise RuntimeError("Console requires the app-owned generated video store")
        return store

    def _build_video_card_specs(self, messages) -> dict[str, ConsoleVideoCardSpec]:
        """Build video-generation card payloads, resolving each slug to its file.

        A message whose file is missing (restart/expiry/LRU eviction) stays
        in the map as an ``"expired"`` tombstone spec -- the card renders
        the named tombstone instead of dropping the row (ADR-044).
        """
        store = self._ensure_console_video_store()
        specs: dict[str, ConsoleVideoCardSpec] = {}
        for message in messages:
            meta = getattr(message, "video_metadata", None)
            if meta is None:
                continue
            extension = canonical_video_extension(meta.container)
            path = store.resolve(
                self._video_storage_message_id(message),
                meta.name,
                extension=extension,
            )
            specs[message.id] = ConsoleVideoCardSpec(
                message_id=message.id,
                meta=meta,
                status="ready" if path is not None else "expired",
                file_path=str(path) if path is not None else None,
            )
        return specs

    @staticmethod
    def _video_storage_message_id(message) -> str:
        """Return the durable key used by the ephemeral video store."""
        return message.persisted_message_id or message.id

    def _pending_console_video_artifacts(
        self,
    ) -> dict[str, PendingVideoArtifact]:
        """Return the lazily owned pending-video registry for this screen."""
        artifacts = getattr(self, "_pending_video_artifacts", None)
        if artifacts is None:
            artifacts = {}
            self._pending_video_artifacts = artifacts
        return artifacts

    def _owns_pending_console_video(self, artifact: PendingVideoArtifact) -> bool:
        """Whether this mounted screen still owns this exact staged result."""
        return (
            not getattr(self, "_pending_video_artifacts_closed", False)
            and self._pending_console_video_artifacts().get(artifact.message_id)
            is artifact
        )

    @staticmethod
    def _close_pending_console_video(artifact: PendingVideoArtifact) -> None:
        """Close one artifact without letting cleanup interrupt its caller."""
        if getattr(getattr(artifact, "stream", None), "closed", False):
            return
        try:
            artifact.close()
        except Exception as exc:  # noqa: BLE001 - teardown must continue
            logger.warning(
                "Console video operation={} failed error_type={}",
                "artifact_close",
                type(exc).__name__,
            )

    def _register_console_video_publication_gate(
        self, message_id: str
    ) -> VideoPublicationGate:
        """Register one pre-generation gate so teardown can cancel publication."""
        gates = getattr(self, "_pending_video_operation_cancels", None)
        if gates is None:
            gates = {}
            self._pending_video_operation_cancels = gates
        gate = VideoPublicationGate()
        gates[message_id] = gate
        if getattr(self, "_pending_video_artifacts_closed", False):
            gate.cancel()
        return gate

    def _release_console_video_publication_gate(
        self, message_id: str, gate: VideoPublicationGate | None
    ) -> None:
        """Drop a terminal gate once no stream operation still uses it."""
        if gate is None:
            return
        active = getattr(self, "_pending_video_active_operations", {})
        entry = active.get(message_id)
        if entry is not None and entry[1] > 0:
            return
        gates = getattr(self, "_pending_video_operation_cancels", {})
        if gates.get(message_id) is gate:
            gates.pop(message_id, None)

    def _begin_pending_console_video_operation(
        self, artifact: PendingVideoArtifact
    ) -> VideoPublicationGate | None:
        """Mark a stream-using operation active and return its cancellation gate."""
        if not self._owns_pending_console_video(artifact):
            return None
        active = getattr(self, "_pending_video_active_operations", None)
        if active is None:
            active = {}
            self._pending_video_active_operations = active
        entry = active.get(artifact.message_id)
        if entry is not None and entry[0] is not artifact:
            return None
        count = entry[1] if entry is not None else 0
        active[artifact.message_id] = (artifact, count + 1)
        gates = getattr(self, "_pending_video_operation_cancels", None)
        if gates is None:
            gates = {}
            self._pending_video_operation_cancels = gates
        gate = gates.get(artifact.message_id)
        if gate is None:
            gate = VideoPublicationGate()
            gates[artifact.message_id] = gate
        return gate

    def _end_pending_console_video_operation(
        self, artifact: PendingVideoArtifact
    ) -> None:
        """Release one active operation and perform any deferred close."""
        active = getattr(self, "_pending_video_active_operations", {})
        entry = active.get(artifact.message_id)
        if entry is None or entry[0] is not artifact:
            return
        if entry[1] > 1:
            active[artifact.message_id] = (artifact, entry[1] - 1)
            return
        active.pop(artifact.message_id, None)
        deferred = getattr(self, "_pending_video_deferred_closes", {})
        if deferred.get(artifact.message_id) is artifact:
            deferred.pop(artifact.message_id, None)
            ConsoleVideoController._close_pending_console_video(artifact)

    @staticmethod
    async def _await_shielded_console_video_task(
        task,
        *,
        cancelled_result_callback=None,
    ) -> Any:
        """Wait for a child operation to finish before propagating cancellation."""
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                result = task.result()
            except BaseException:
                raise cancellation
            if cancelled_result_callback is not None:
                cancelled_result_callback(result)
            raise cancellation

    async def _run_pending_console_video_operation(
        self,
        artifact: PendingVideoArtifact,
        function,
        *args,
        result_callback=None,
        **kwargs,
    ) -> tuple[bool, Any]:
        """Run and finalize one artifact operation before releasing ownership."""
        gate = self._begin_pending_console_video_operation(artifact)
        if gate is None:
            return False, None

        async def _run_and_finalize() -> Any:
            result = await asyncio.to_thread(
                function,
                *args,
                publication_gate=gate,
                **kwargs,
            )
            if result_callback is not None:
                result = await result_callback(result)
            return result

        operation_task = asyncio.create_task(_run_and_finalize())
        try:
            result = await ConsoleVideoController._await_shielded_console_video_task(
                operation_task
            )
            return True, result
        finally:
            self._end_pending_console_video_operation(artifact)

    async def _run_console_video_generation_operation(
        self,
        *,
        session_id: str,
        message_id: str,
        **generation_kwargs,
    ) -> None:
        """Run generation and make every committed tuple durable in one child."""

        async def _generate_and_finalize() -> tuple[Any, Path] | PendingVideoArtifact:
            outcome = await asyncio.to_thread(
                run_video_generation,
                message_id=message_id,
                **generation_kwargs,
            )
            if not isinstance(outcome, PendingVideoArtifact):
                ConsoleVideoController._persist_generated_video_tuple(
                    self,
                    outcome,
                    session_id=session_id,
                    message_id=message_id,
                )
            return outcome

        def _close_cancelled_pending(outcome: Any) -> None:
            if isinstance(outcome, PendingVideoArtifact):
                ConsoleVideoController._close_pending_console_video(outcome)

        operation_task = asyncio.create_task(_generate_and_finalize())
        outcome = await ConsoleVideoController._await_shielded_console_video_task(
            operation_task,
            cancelled_result_callback=_close_cancelled_pending,
        )
        if isinstance(outcome, PendingVideoArtifact):
            await ConsoleVideoController._resolve_generated_video_outcome(
                self,
                outcome,
                session_id=session_id,
                message_id=message_id,
            )
            return
        if getattr(self, "_pending_video_artifacts_closed", False):
            return
        await self._sync_native_console_chat_ui()

    def _drain_pending_console_videos(self) -> None:
        """Atomically detach and close every staged video owned by the screen."""
        self._pending_video_artifacts_closed = True
        adapter_cancels = list(getattr(self, "_console_videogen_cancels", {}).values())
        publication_cancels = list(
            getattr(self, "_pending_video_operation_cancels", {}).values()
        )
        self._pending_video_operation_cancels = {}
        artifacts = getattr(self, "_pending_video_artifacts", None)
        self._pending_video_artifacts = {}
        for cancel in adapter_cancels:
            try:
                cancel.set()
            except Exception as exc:  # noqa: BLE001 - teardown must continue
                logger.warning(
                    "Console video operation={} failed error_type={}",
                    "adapter_cancel",
                    type(exc).__name__,
                )
        for cancel in publication_cancels:
            try:
                cancel.cancel()
            except Exception as exc:  # noqa: BLE001 - teardown must continue
                logger.warning(
                    "Console video operation={} failed error_type={}",
                    "publication_cancel",
                    type(exc).__name__,
                )
        if not artifacts:
            return
        active = getattr(self, "_pending_video_active_operations", {})
        deferred = getattr(self, "_pending_video_deferred_closes", None)
        if deferred is None:
            deferred = {}
            self._pending_video_deferred_closes = deferred
        for artifact in artifacts.values():
            entry = active.get(artifact.message_id)
            if entry is not None and entry[0] is artifact and entry[1] > 0:
                deferred[artifact.message_id] = artifact
                continue
            ConsoleVideoController._close_pending_console_video(artifact)

    @staticmethod
    def _external_video_target_identity(path: Path) -> tuple[int, int, int, int, int]:
        """Capture one target's non-following identity for overwrite consent."""
        metadata = path.lstat()
        return ConsoleVideoController._external_video_stat_identity(metadata)

    @staticmethod
    def _external_video_stat_identity(
        metadata,
    ) -> tuple[int, int, int, int, int]:
        """Return the non-following fields used for race revalidation."""
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_mode,
        )

    @staticmethod
    def _external_video_cleanup_identity(metadata) -> tuple[int, int, int]:
        """Return immutable fields used to identify an app-owned sibling."""
        return (metadata.st_dev, metadata.st_ino, metadata.st_mode)

    @staticmethod
    def _external_video_parent_identity(
        parent: Path,
    ) -> tuple[int, int, int]:
        """Validate and identify a real, non-reparse destination directory."""
        import stat

        metadata = parent.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or bool(reparse_flag and attributes & reparse_flag)
        ):
            raise OSError("unsafe external video parent")
        return (metadata.st_dev, metadata.st_ino, metadata.st_mode)

    @staticmethod
    def _require_external_video_pinned_capabilities() -> None:
        """Fail closed unless every directory-relative primitive is available."""
        directory_flags = getattr(os, "O_DIRECTORY", 0)
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        required_dir_fd = (os.open, os.stat, os.link, os.unlink)
        required_nofollow = (os.stat, os.link)
        try:
            replace_parameters = inspect.signature(os.replace).parameters
        except (TypeError, ValueError) as exc:
            raise OSError("pinned external save unsupported") from exc
        if (
            not directory_flags
            or not nofollow_flag
            or any(function not in os.supports_dir_fd for function in required_dir_fd)
            or any(
                function not in os.supports_follow_symlinks
                for function in required_nofollow
            )
            or "src_dir_fd" not in replace_parameters
            or "dst_dir_fd" not in replace_parameters
        ):
            raise OSError("pinned external save unsupported")

    @staticmethod
    def _external_video_precommit_check(
        parent: Path,
        parent_fd: int | None,
        parent_identity: tuple[int, int, int],
        sibling: str | Path,
        sibling_identity: tuple[int, int, int, int, int],
    ) -> None:
        """Revalidate the pinned parent and complete sibling before commit."""
        if (
            ConsoleVideoController._external_video_parent_identity(parent)
            != parent_identity
        ):
            raise OSError("external video parent changed")
        if parent_fd is None:
            metadata = Path(sibling).lstat()
        else:
            pinned_metadata = os.fstat(parent_fd)
            if (
                pinned_metadata.st_dev,
                pinned_metadata.st_ino,
                pinned_metadata.st_mode,
            ) != parent_identity:
                raise OSError("pinned external video parent changed")
            metadata = os.stat(sibling, dir_fd=parent_fd, follow_symlinks=False)
        if (
            ConsoleVideoController._external_video_stat_identity(metadata)
            != sibling_identity
        ):
            raise OSError("external video sibling changed")

    @staticmethod
    def _copy_pending_video_external(
        artifact: PendingVideoArtifact,
        target: Path,
        confirmed_identity: tuple[int, int, int, int, int] | None,
        publication_gate: VideoPublicationGate | None = None,
    ) -> Literal["saved", "confirm"]:
        """Copy to a complete sibling, then commit without silent overwrite.

        ``confirmed_identity`` is ``None`` for a target believed absent. That
        path uses a hard-link commit, whose create-if-absent property prevents
        a concurrent creator from being overwritten. A confirmed replacement
        is revalidated immediately before ``os.replace``.
        """
        import secrets
        import shutil

        from tldw_chatbook.Utils.path_validation import validate_path_simple

        target = validate_path_simple(target, probe_existing=False)
        extension = canonical_video_extension(artifact.extension)
        if target.suffix != f".{extension}":
            raise ValueError("external video target extension does not match")
        ConsoleVideoController._require_external_video_pinned_capabilities()
        target.parent.mkdir(parents=False, exist_ok=True)
        parent_identity = ConsoleVideoController._external_video_parent_identity(
            target.parent
        )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_fd: int | None = None
        staged_fd: int | None = None
        sibling_name: str | None = None
        sibling_cleanup_identity: tuple[int, int, int] | None = None
        sibling_complete_identity: tuple[int, int, int, int, int] | None = None
        try:
            parent_fd = os.open(target.parent, directory_flags)
            pinned_metadata = os.fstat(parent_fd)
            if (
                pinned_metadata.st_dev,
                pinned_metadata.st_ino,
                pinned_metadata.st_mode,
            ) != parent_identity:
                raise OSError("external video parent changed during pin")
            os.stat(".", dir_fd=parent_fd, follow_symlinks=False)
            for _attempt in range(32):
                candidate = f".{target.name}.{secrets.token_hex(8)}.tmp"
                try:
                    staged_fd = os.open(
                        candidate,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    continue
                sibling_name = candidate
                sibling_cleanup_identity = (
                    ConsoleVideoController._external_video_cleanup_identity(
                        os.fstat(staged_fd)
                    )
                )
                break
            else:
                raise OSError("could not allocate external video sibling")
            staged_context = os.fdopen(staged_fd, "w+b")
            staged_fd = None

            with staged_context as staged:
                artifact.rewind()
                shutil.copyfileobj(artifact.stream, staged)
                staged.flush()
                os.fsync(staged.fileno())
                if os.fstat(staged.fileno()).st_size != artifact.size_bytes:
                    raise OSError("generated video payload size changed")
                sibling_complete_identity = (
                    ConsoleVideoController._external_video_stat_identity(
                        os.fstat(staged.fileno())
                    )
                )

            sibling = sibling_name
            if sibling is None:  # pragma: no cover - allocation guarantees it
                raise OSError("external video sibling unavailable")
            ConsoleVideoController._external_video_precommit_check(
                target.parent,
                parent_fd,
                parent_identity,
                sibling,
                sibling_complete_identity,
            )
            claim = (
                publication_gate.claim_publication()
                if publication_gate is not None
                else nullcontext(True)
            )
            with claim as active:
                if not active:
                    raise OSError("external video commit cancelled")
                ConsoleVideoController._external_video_precommit_check(
                    target.parent,
                    parent_fd,
                    parent_identity,
                    sibling,
                    sibling_complete_identity,
                )
                if confirmed_identity is None:
                    try:
                        os.link(
                            sibling,
                            target.name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        return "confirm"
                    try:
                        os.unlink(sibling, dir_fd=parent_fd)
                    except OSError:
                        pass
                    else:
                        sibling_name = None
                    return "saved"

                try:
                    current_identity = (
                        ConsoleVideoController._external_video_stat_identity(
                            os.stat(
                                target.name,
                                dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                        )
                    )
                except FileNotFoundError:
                    return "confirm"
                if current_identity != confirmed_identity:
                    return "confirm"
                os.replace(
                    sibling,
                    target.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                sibling_name = None
                return "saved"
        finally:
            try:
                artifact.rewind()
            except (OSError, ValueError):
                pass
            if sibling_name is not None and parent_fd is not None:
                try:
                    if sibling_cleanup_identity is None and staged_fd is not None:
                        sibling_cleanup_identity = (
                            ConsoleVideoController._external_video_cleanup_identity(
                                os.fstat(staged_fd)
                            )
                        )
                    current_sibling = (
                        ConsoleVideoController._external_video_cleanup_identity(
                            os.stat(
                                sibling_name,
                                dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                        )
                    )
                    if (
                        sibling_cleanup_identity is not None
                        and current_sibling == sibling_cleanup_identity
                    ):
                        os.unlink(sibling_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "external_stage_cleanup",
                        "OSError",
                    )
            if staged_fd is not None:
                try:
                    os.close(staged_fd)
                except OSError as exc:
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "external_stage_fd_close",
                        type(exc).__name__,
                    )
            if parent_fd is not None:
                try:
                    os.close(parent_fd)
                except OSError as exc:
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "external_parent_close",
                        type(exc).__name__,
                    )

    def _retry_pending_console_video(
        self,
        artifact: PendingVideoArtifact,
        *,
        publication_gate: VideoPublicationGate | None = None,
    ) -> Path:
        """Retry a failed ordinary managed save from the retained payload."""
        artifact.rewind()
        content = artifact.stream.read()
        artifact.rewind()
        if len(content) != artifact.size_bytes:
            raise VideoStoreSaveError("pending video payload size changed")
        outcome = self._ensure_console_video_store().save(
            artifact.message_id,
            artifact.slug,
            content,
            extension=artifact.extension,
            publication_gate=publication_gate,
        )
        if isinstance(outcome, VideoCapacityExceeded):
            raise VideoStoreSaveError("pending video no longer fits managed storage")
        return outcome

    async def _save_pending_console_video_external(
        self, artifact: PendingVideoArtifact
    ) -> Path | Literal[False] | None:
        """Choose and atomically write an external path, retaining on failure."""
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

        while self._owns_pending_console_video(artifact):
            selected = await self._wait_for_console_screen_result(
                EnhancedFileSave(
                    title="Save generated video",
                    default_filename=f"{artifact.slug}.{artifact.extension}",
                )
            )
            if not self._owns_pending_console_video(artifact) or not selected:
                return None
            try:
                target = ConsoleVideoController._normalize_pending_video_target(
                    Path(selected).expanduser(), artifact.extension
                )
            except ValueError:
                self.app_instance.notify(
                    "Choose a filename with no extension or the generated video format.",
                    severity="warning",
                )
                continue
            confirmed_identity = None
            reconfirmation_required = False

            while self._owns_pending_console_video(artifact):
                try:
                    identity = await asyncio.to_thread(
                        self._external_video_target_identity, target
                    )
                except FileNotFoundError:
                    identity = None
                except (OSError, ValueError) as exc:
                    if not self._owns_pending_console_video(artifact):
                        return None
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "external_target_inspect",
                        type(exc).__name__,
                    )
                    self.app_instance.notify(
                        "Could not inspect the selected video destination. "
                        "Choose another location or discard the result.",
                        severity="error",
                    )
                    return False

                if identity is None and reconfirmation_required:
                    confirmed = await self._wait_for_console_screen_result(
                        ConfirmationDialog(
                            title="Destination changed",
                            message=(
                                "The file previously confirmed at "
                                f"{escape_markup(str(target))} no longer exists. "
                                "Save the generated video to this path?"
                            ),
                            confirm_label="Save",
                            cancel_label="Choose another",
                        )
                    )
                    if not self._owns_pending_console_video(artifact):
                        return None
                    if not confirmed:
                        break
                    reconfirmation_required = False
                    confirmed_identity = None
                elif identity is not None and identity != confirmed_identity:
                    confirmed = await self._wait_for_console_screen_result(
                        ConfirmationDialog(
                            title="Replace existing file?",
                            message=(
                                "A file already exists at "
                                f"{escape_markup(str(target))}. Replace it?"
                            ),
                            confirm_label="Replace",
                            cancel_label="Choose another",
                        )
                    )
                    if not self._owns_pending_console_video(artifact):
                        return None
                    if not confirmed:
                        break
                    confirmed_identity = identity
                    reconfirmation_required = False
                elif identity is None:
                    confirmed_identity = None

                if not self._owns_pending_console_video(artifact):
                    return None
                try:
                    started, result = await self._run_pending_console_video_operation(
                        artifact,
                        self._copy_pending_video_external,
                        artifact,
                        target,
                        confirmed_identity,
                    )
                    if not started:
                        return None
                except (OSError, ValueError) as exc:
                    if not self._owns_pending_console_video(artifact):
                        return None
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "external_copy",
                        type(exc).__name__,
                    )
                    self.app_instance.notify(
                        "Could not save the generated video to "
                        f"{escape_markup(str(target))}. You can try again or "
                        "choose another outcome.",
                        severity="error",
                    )
                    return False
                if not self._owns_pending_console_video(artifact):
                    return None
                if result == "saved":
                    return target
                reconfirmation_required = confirmed_identity is not None
                confirmed_identity = None
            # Replacement declined: return to the picker with the stage live.
        return None

    @staticmethod
    def _normalize_pending_video_target(target: Path, extension: str) -> Path:
        """Append the canonical suffix or require its exact lowercase spelling."""
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        target = validate_path_simple(target, probe_existing=False)
        canonical = canonical_video_extension(extension)
        if not target.suffix:
            return target.with_suffix(f".{canonical}")
        if target.suffix == f".{canonical}":
            return target
        raise ValueError("external video target extension does not match")

    async def _resolve_generated_video_outcome(
        self,
        outcome: tuple[Any, Path] | PendingVideoArtifact,
        *,
        session_id: str,
        message_id: str,
    ) -> None:
        """Resolve one normal or staged generation result for either caller."""
        chat_store = self._ensure_console_chat_store()
        if not isinstance(outcome, PendingVideoArtifact):
            ConsoleVideoController._persist_generated_video_tuple(
                self, outcome, session_id=session_id, message_id=message_id
            )
            if getattr(self, "_pending_video_artifacts_closed", False):
                return
            await self._sync_native_console_chat_ui()
            return

        artifact = outcome
        if artifact.message_id != message_id or getattr(
            self, "_pending_video_artifacts_closed", False
        ):
            ConsoleVideoController._close_pending_console_video(artifact)
            return
        artifacts = self._pending_console_video_artifacts()
        existing = artifacts.get(message_id)
        if existing is not None and existing is not artifact:
            ConsoleVideoController._close_pending_console_video(artifact)
            return
        artifacts[message_id] = artifact
        try:
            while self._owns_pending_console_video(artifact):
                from ...Widgets.Console.console_video_capacity_modal import (
                    ConsoleVideoCapacityModal,
                )

                choice = await self._wait_for_console_screen_result(
                    ConsoleVideoCapacityModal(
                        reason=artifact.reason,
                        size_bytes=artifact.size_bytes,
                        max_bytes=artifact.max_bytes,
                    )
                )
                if not self._owns_pending_console_video(artifact):
                    return
                if choice != "keep" and choice != "save_external":
                    return
                if choice == "save_external":
                    external_path = await self._save_pending_console_video_external(
                        artifact
                    )
                    if not self._owns_pending_console_video(artifact):
                        return
                    if external_path is None:
                        return
                    if external_path is False:
                        continue
                    try:
                        self._open_video_with_os(external_path)
                    except Exception as exc:  # noqa: BLE001 - OS launcher boundary
                        logger.warning(
                            "Console video operation={} failed error_type={}",
                            "external_open",
                            type(exc).__name__,
                        )
                        self.app_instance.notify(
                            "Video saved, but could not open it automatically: "
                            f"{escape_markup(str(external_path))}",
                            severity="warning",
                        )
                    return

                if not self._owns_pending_console_video(artifact):
                    return
                try:
                    video_store = self._ensure_console_video_store()

                    async def _finalize_managed_result(
                        managed_path,
                    ) -> Path | None:
                        resolved_path = await asyncio.to_thread(
                            video_store.resolve,
                            artifact.message_id,
                            artifact.slug,
                            extension=artifact.extension,
                        )
                        if resolved_path is not None and Path(resolved_path) == Path(
                            managed_path
                        ):
                            chat_store.append_video_message(
                                session_id,
                                video_metadata=artifact.metadata,
                                persist=True,
                                message_id=artifact.message_id,
                            )
                            return Path(managed_path)
                        return None

                    if artifact.reason == "over_capacity":
                        (
                            started,
                            managed_path,
                        ) = await self._run_pending_console_video_operation(
                            artifact,
                            video_store.adopt_oversized,
                            artifact.message_id,
                            artifact.slug,
                            artifact.stream,
                            artifact.size_bytes,
                            extension=artifact.extension,
                            result_callback=_finalize_managed_result,
                        )
                    else:
                        (
                            started,
                            managed_path,
                        ) = await self._run_pending_console_video_operation(
                            artifact,
                            self._retry_pending_console_video,
                            artifact,
                            result_callback=_finalize_managed_result,
                        )
                    if not started:
                        return
                except Exception as exc:  # noqa: BLE001 - recoverable store boundary
                    if not self._owns_pending_console_video(artifact):
                        return
                    logger.warning(
                        "Console video operation={} failed error_type={}",
                        "managed_resolution",
                        type(exc).__name__,
                    )
                    self.app_instance.notify(
                        "The generated video could not be stored. You can try "
                        "again, save it to disk, or discard it.",
                        severity="error",
                    )
                    continue
                terminal = getattr(self, "_pending_video_artifacts_closed", False)
                if not self._owns_pending_console_video(artifact) and not terminal:
                    return
                if managed_path is None:
                    if terminal:
                        return
                    self.app_instance.notify(
                        "The generated video was not available after storage. "
                        "Choose another outcome.",
                        severity="error",
                    )
                    continue
                if terminal:
                    return
                await self._sync_native_console_chat_ui()
                return
        finally:
            current = self._pending_console_video_artifacts()
            if current.get(message_id) is artifact:
                current.pop(message_id, None)
            publication_gate = getattr(
                self, "_pending_video_operation_cancels", {}
            ).get(message_id)
            active = getattr(self, "_pending_video_active_operations", {})
            entry = active.get(message_id)
            if entry is not None and entry[0] is artifact and entry[1] > 0:
                deferred = getattr(self, "_pending_video_deferred_closes", None)
                if deferred is None:
                    deferred = {}
                    self._pending_video_deferred_closes = deferred
                deferred[message_id] = artifact
            else:
                ConsoleVideoController._close_pending_console_video(artifact)
            ConsoleVideoController._release_console_video_publication_gate(
                self, message_id, publication_gate
            )

    def _persist_generated_video_tuple(
        self,
        outcome: tuple[Any, Path],
        *,
        session_id: str,
        message_id: str,
    ) -> None:
        """Persist one already-published normal video without touching the UI."""
        metadata, _managed_path = outcome
        self._ensure_console_chat_store().append_video_message(
            session_id,
            video_metadata=metadata,
            persist=True,
            message_id=message_id,
        )

    async def _console_command_generate_video(self, parse: CommandParse) -> None:
        """Resolve and run one ``/generate-video`` generation (task-3401.5).

        Mirrors ``_console_command_generate_image``: refusals leave the
        composer draft untouched and never touch the store; the draft clears
        on dispatch and is restored on failure; the blocking generation runs
        via ``asyncio.to_thread``; the in-flight guard and cancel event are
        always cleared in a ``finally`` so a crashed/cancelled run never
        wedges the session. Cancellation is cooperative: the composer's Stop
        button (visible while a video generation is in flight) sets the
        event the MiniMax adapter polls.
        """
        from uuid import uuid4

        args = parse_generate_video_args(parse.args)
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings(),
        )
        if not args.prompt:
            await self._append_native_console_system_message(
                GENERATE_VIDEO_USAGE_TEXT, session_id=session.id
            )
            return
        # @style resolution (task-3401.12): an unknown style refuses with
        # the available catalog; a resolved style composes the prompt and
        # contributes duration/fps/ratio defaults.
        style_params: dict = {}
        prompt_text = args.prompt
        negative_text: str | None = None
        if args.style is not None:
            from tldw_chatbook.Video_Generation.video_templates import (
                apply_video_template,
                get_all_video_templates,
                get_video_template,
            )

            template = get_video_template(args.style)
            if template is None:
                available = ", ".join(sorted(get_all_video_templates()))
                await self._append_native_console_system_message(
                    f"Unknown video style '@{args.style}' — available: {available}.",
                    session_id=session.id,
                )
                return
            prompt_text, negative_text = apply_video_template(template, args.prompt)
            style_params = dict(template.default_params)
        cfg = get_video_generation_config()
        backend = args.backend or cfg.default_backend
        if not backend:
            await self._append_native_console_system_message(
                "No video generation backend configured. Set "
                "[video_generation].default_backend, or use "
                "/generate-video :backend <prompt>.",
                session_id=session.id,
            )
            return
        from tldw_chatbook.Video_Generation.adapter_registry import (
            get_registry as _get_video_registry,
        )

        if _get_video_registry().resolve_backend(backend) is None:
            await self._append_native_console_system_message(
                f"Video backend '{backend}' is not enabled. Check "
                "[video_generation].enabled_backends.",
                session_id=session.id,
            )
            return
        # Cost-confirm gate (AC: paid backends only, settings-toggleable).
        if cfg.confirm_cost_estimate and is_paid_backend(backend):
            from tldw_chatbook.Widgets.cancel_confirmation_dialog import (
                CancelConfirmationDialog,
            )

            confirmed = await self._wait_for_console_screen_result(
                CancelConfirmationDialog(
                    title="Generate video?",
                    message=estimate_video_cost_text(
                        backend, style_params.get("duration_seconds")
                    ),
                    confirm_text="Generate",
                    cancel_text="Cancel",
                )
            )
            if not confirmed:
                return
        inflight = self._console_videogen_inflight_sessions()
        if session.id in inflight:
            await self._append_native_console_system_message(
                "A video generation is already running for this session.",
                session_id=session.id,
            )
            return
        inflight.add(session.id)
        # Capture draft before clearing so we can restore it on failure.
        composer = self._console_composer_or_none()
        saved_draft = composer.draft_text() if composer is not None else ""
        self._clear_console_composer_draft()
        cancel_event = threading.Event()
        self._console_videogen_cancel_events()[session.id] = cancel_event
        message_id = str(uuid4())
        publication_gate = (
            ConsoleVideoController._register_console_video_publication_gate(
                self, message_id
            )
        )
        try:
            await self._append_native_console_system_message(
                f"⏳ Generating video on {backend}… (Stop cancels)",
                session_id=session.id,
            )
            await ConsoleVideoController._run_console_video_generation_operation(
                self,
                session_id=session.id,
                message_id=message_id,
                backend=backend,
                prompt=prompt_text,
                negative_prompt=negative_text or None,
                style_negative_prompt=args.style is not None,
                video_format="mp4",
                duration_seconds=style_params.get("duration_seconds"),
                fps=style_params.get("fps"),
                ratio=style_params.get("ratio"),
                cancel_event=cancel_event,
                publication_gate=publication_gate,
                video_store=self._ensure_console_video_store(),
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user, never a bare crash
            if composer is not None and saved_draft:
                composer.clear_draft()
                composer.insert_text_as_paste(saved_draft)
            logger.error(
                "Console video operation={} failed error_type={}",
                "generation",
                type(exc).__name__,
            )
            await self._append_native_console_system_message(
                f"Video generation failed ({type(exc).__name__}).",
                session_id=session.id,
            )
        finally:
            inflight.discard(session.id)
            self._console_videogen_cancel_events().pop(session.id, None)
            ConsoleVideoController._release_console_video_publication_gate(
                self, message_id, publication_gate
            )

    async def _play_console_video(self, message_id: str) -> None:
        """Open the ephemeral video file with the OS default player (v1).

        The in-app player screens land with tasks 3401.9/.10; until then the
        honest playback path is the system player. A missing file (tombstone)
        re-syncs so the card renders expired, then reports.
        """
        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message no longer exists.", severity="warning"
            )
            return
        meta = getattr(message, "video_metadata", None)
        if meta is None:
            return
        extension = canonical_video_extension(meta.container)
        path = self._ensure_console_video_store().resolve(
            self._video_storage_message_id(message),
            meta.name,
            extension=extension,
        )
        if path is None:
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(
                "The ephemeral video file is gone — regenerate to recreate it.",
                severity="warning",
            )
            return
        # task-3401.10: the modal player screen is the real playback surface
        # (audio + sync + seek); the OS player remains as the fallback when
        # ffmpeg/ffplay are not installed (with one guidance notice).
        from tldw_chatbook.Media_Playback.player_pipeline import (
            playback_tools_available,
        )

        tools_ok, guidance = playback_tools_available()
        if tools_ok:
            from tldw_chatbook.UI.Screens.video_player_screen import VideoPlayerScreen

            self.app_instance.push_screen(VideoPlayerScreen(str(path), title=meta.name))
            return
        self.app_instance.notify(guidance, severity="information")
        try:
            self._open_video_with_os(path)
        except Exception as exc:
            logger.warning(
                "Console video operation={} failed error_type={}",
                "managed_open",
                type(exc).__name__,
            )
            self.app_instance.notify(
                "Could not open the video with the system player.", severity="error"
            )

    async def _save_console_video_copy(self, message_id: str) -> None:
        """Copy the ephemeral video file out to the user's save location.

        The ONLY byte escape hatch for the ephemeral model -- always an
        explicit user act (ADR-044). Mirrors
        ``_save_console_message_image``'s destination/collision pattern.
        """
        import shutil

        from rich.markup import escape as escape_markup

        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message no longer exists.", severity="warning"
            )
            return
        meta = getattr(message, "video_metadata", None)
        if meta is None:
            return
        extension = canonical_video_extension(meta.container)
        path = self._ensure_console_video_store().resolve(
            self._video_storage_message_id(message),
            meta.name,
            extension=extension,
        )
        if path is None:
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(
                "The ephemeral video file is gone — regenerate to recreate it.",
                severity="warning",
            )
            return

        def _copy_to_disk() -> "Path":
            from tldw_chatbook.Utils.path_validation import validate_path_simple

            save_location = validate_path_simple(
                os.path.expanduser(
                    get_cli_setting("chat.videos", "save_location", "~/Downloads")
                )
            )
            save_location.mkdir(parents=True, exist_ok=True)
            target = save_location / f"{meta.name}.{extension}"
            counter = 1
            while target.exists():
                target = save_location / f"{meta.name}_{counter}.{extension}"
                counter += 1
            shutil.copy2(path, target)
            return target

        try:
            written = await asyncio.to_thread(_copy_to_disk)
        except Exception as exc:
            logger.warning(
                "Console video operation={} failed error_type={}",
                "managed_copy",
                type(exc).__name__,
            )
            self.app_instance.notify("Could not save the video.", severity="error")
            return
        self.app_instance.notify(f"Video saved to {escape_markup(str(written))}")

    async def _regenerate_console_video_message(self, message_id: str) -> None:
        """Regenerate a video message from its persisted facts (tombstone or not).

        Appends a NEW video message (videos have no variants) rebuilt from
        the stored ``VideoGenerationMetadata`` -- same backend/prompt/model/
        shape -- with ``seed`` forced to ``-1`` so the recreation is never a
        byte-duplicate. The per-session in-flight guard and cancel-event
        bookkeeping mirror ``_console_command_generate_video`` exactly.
        """
        from uuid import uuid4

        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message no longer exists.", severity="warning"
            )
            return
        meta = getattr(message, "video_metadata", None)
        if meta is None:
            return
        session_id = store.session_id_for_message(message_id)
        inflight = self._console_videogen_inflight_sessions()
        if session_id in inflight:
            await self._append_native_console_system_message(
                "A video generation is already running for this session.",
                session_id=session_id,
            )
            return
        inflight.add(session_id)
        cancel_event = threading.Event()
        self._console_videogen_cancel_events()[session_id] = cancel_event
        new_message_id = str(uuid4())
        publication_gate = (
            ConsoleVideoController._register_console_video_publication_gate(
                self, new_message_id
            )
        )
        try:
            await ConsoleVideoController._run_console_video_generation_operation(
                self,
                session_id=session_id,
                message_id=new_message_id,
                backend=meta.backend,
                prompt=meta.prompt,
                negative_prompt=meta.negative_prompt or None,
                duration_seconds=(
                    int(meta.duration_seconds) if meta.duration_seconds else None
                ),
                fps=int(meta.fps) if meta.fps else None,
                width=meta.width,
                height=meta.height,
                ratio=meta.ratio,
                seed=-1,
                model=meta.model,
                video_format=meta.container,
                cancel_event=cancel_event,
                publication_gate=publication_gate,
                video_store=self._ensure_console_video_store(),
            )
        except Exception as exc:  # noqa: BLE001 - reported, never a bare crash
            logger.error(
                "Console video operation={} failed error_type={}",
                "regeneration",
                type(exc).__name__,
            )
            await self._append_native_console_system_message(
                f"Video regeneration failed ({type(exc).__name__}).",
                session_id=session_id,
            )
        finally:
            inflight.discard(session_id)
            self._console_videogen_cancel_events().pop(session_id, None)
            ConsoleVideoController._release_console_video_publication_gate(
                self, new_message_id, publication_gate
            )

    async def _console_command_stream_video(self, parse: CommandParse) -> None:
        """Resolve and play one ``/stream-video <url>`` (task-3401.11).

        Resolution (egress-gated redirect walk, yt-dlp subprocess fallback,
        seekability probe) is blocking network I/O, so it runs via
        ``asyncio.to_thread``; the player screen opens on success with the
        stream's seek capability and the AC5 time box. Nothing is written
        to disk on any path.
        """
        url = (parse.args or "").strip()
        if not url:
            await self._append_native_console_system_message(
                "Usage: /stream-video <url>"
            )
            return
        from tldw_chatbook.Media_Playback.stream_resolve import (
            MAX_STREAM_SECONDS,
            StreamResolutionError,
            resolve_stream_url,
        )

        try:
            resolution = await asyncio.to_thread(resolve_stream_url, url)
        except StreamResolutionError as exc:
            await self._append_native_console_system_message(
                f"Cannot stream that URL: {exc}"
            )
            return
        except Exception as exc:  # egress refusal or unexpected resolution failure
            logger.warning(
                "stream resolution failed (error_type={})", type(exc).__name__
            )
            await self._append_native_console_system_message(
                f"Cannot stream that URL: {exc}"
            )
            return
        from tldw_chatbook.UI.Screens.video_player_screen import VideoPlayerScreen

        self.app_instance.push_screen(
            VideoPlayerScreen(
                resolution.final_url,
                title="stream",
                seekable=resolution.seekable,
                max_seconds=MAX_STREAM_SECONDS,
            )
        )
