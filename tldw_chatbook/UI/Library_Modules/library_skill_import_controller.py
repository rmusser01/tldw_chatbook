"""App-scoped single-flight owner for Library skill imports."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger

from ...Skills_Interop.skill_remote_fetch import (
    RemoteSkillError,
    classify_skill_source_url,
    install_skill_from_url,
)
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Utils.path_validation import validate_path_simple


LIBRARY_SKILLS_IMPORT_WORKER_GROUP = "library_skills_import"
_SKILL_MD_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class LibrarySkillImportSnapshot:
    """One app-owned import row and its accepted operation receipt."""

    row_open: bool = False
    path: str = ""
    status: str = ""
    review_name: str = ""
    in_flight: bool = False
    generation: int = 0


@dataclass(frozen=True)
class _LibrarySkillImportOutcome:
    status: str
    review_name: str = ""
    clear_path: bool = False
    refresh_sources: bool = False


class LibrarySkillImportCoordinator:
    """Own one non-cancellable import across replaceable Library screens.

    ``app_instance`` is the stable service owner for the coordinator's
    lifetime. No screen or widget is retained: settlement looks up the current
    routed screen through the mounted runtime app passed to :meth:`run`.
    """

    def __init__(self, app_instance: Any) -> None:
        self._app_instance = app_instance
        self._snapshot = LibrarySkillImportSnapshot()

    @property
    def snapshot(self) -> LibrarySkillImportSnapshot:
        return self._snapshot

    def update(self, **changes: Any) -> LibrarySkillImportSnapshot:
        """Replace selected presentation fields in the shared snapshot."""
        self._snapshot = replace(self._snapshot, **changes)
        return self._snapshot

    def open_draft(self) -> bool:
        """Open a fresh row, explicitly dismissing any prior receipt."""
        if self._snapshot.in_flight:
            return False
        self._snapshot = LibrarySkillImportSnapshot(
            row_open=True,
            generation=self._snapshot.generation + 1,
        )
        return True

    def dismiss(self) -> bool:
        """Explicitly close the row without claiming cancellation."""
        if self._snapshot.in_flight:
            return False
        self._snapshot = LibrarySkillImportSnapshot(
            generation=self._snapshot.generation + 1
        )
        return True

    def invalidate_row(self) -> None:
        """Fence callbacks from a row lifecycle that is no longer current."""
        self.update(generation=self._snapshot.generation + 1)

    def claim(self, raw_path: str) -> bool:
        """Synchronously admit one operation before its app worker exists."""
        if self._snapshot.in_flight:
            return False
        self.update(
            row_open=True,
            path=raw_path,
            status="Inspecting/importing…",
            review_name="",
            in_flight=True,
            generation=self._snapshot.generation + 1,
        )
        return True

    async def run(self, raw_path: str, *, runtime_app: Any) -> None:
        """Run the accepted mutation and publish one authoritative receipt."""
        operation = asyncio.create_task(
            self._run_and_settle(raw_path, runtime_app=runtime_app)
        )
        while True:
            try:
                await asyncio.shield(operation)
                return
            except asyncio.CancelledError:
                owner = asyncio.current_task()
                if owner is not None:
                    while owner.cancelling():
                        owner.uncancel()
                if not operation.done():
                    continue
                if operation.cancelled():
                    self._settle(
                        _LibrarySkillImportOutcome(
                            "Could not import that skill."
                        ),
                        runtime_app=runtime_app,
                    )
                    return
                operation.result()
                return

    async def _run_and_settle(self, raw_path: str, *, runtime_app: Any) -> None:
        """Settle the accepted mutation in its own cancellation-shielded task."""
        fatal_error: BaseException | None = None
        try:
            outcome = await self._import(raw_path)
        except asyncio.CancelledError:
            outcome = _LibrarySkillImportOutcome("Could not import that skill.")
        except Exception:
            logger.warning("Library skill import worker failed unexpectedly.")
            outcome = _LibrarySkillImportOutcome("Could not import that skill.")
        except BaseException as exc:
            fatal_error = exc
            outcome = _LibrarySkillImportOutcome("Could not import that skill.")

        self._settle(outcome, runtime_app=runtime_app)
        if fatal_error is not None:
            raise fatal_error

    def _settle(
        self, outcome: _LibrarySkillImportOutcome, *, runtime_app: Any
    ) -> None:
        """Publish the one terminal snapshot before the operation task ends."""
        self.update(
            path="" if outcome.clear_path else self._snapshot.path,
            status=outcome.status,
            review_name=outcome.review_name,
            in_flight=False,
        )
        self._publish_current_screen(
            runtime_app, refresh_sources=outcome.refresh_sources
        )

    def _publish_current_screen(
        self, runtime_app: Any, *, refresh_sources: bool
    ) -> None:
        """Notify only the currently routed Library presentation, if any."""
        try:
            current_screen = runtime_app.screen
        except Exception:
            return
        publish = getattr(
            current_screen, "_present_library_skills_import_snapshot", None
        )
        if callable(publish):
            publish(refresh_sources=refresh_sources)

    async def _import(self, raw_path: str) -> _LibrarySkillImportOutcome:
        if raw_path.startswith(("http://", "https://")):
            return await self._import_url(raw_path)
        try:
            validated_path = validate_path_simple(
                Path(raw_path).expanduser(), require_exists=True
            )
        except ValueError:
            logger.opt(exception=True).warning(
                "Rejected Library skills import path."
            )
            return _LibrarySkillImportOutcome(
                "Could not find that file or folder."
            )

        service = getattr(self._app_instance, "skills_scope_service", None)
        import_file = getattr(service, "import_skill_file", None)
        import_directory = getattr(service, "import_skill_directory", None)
        if not callable(import_directory) or not callable(import_file):
            return _LibrarySkillImportOutcome("Skill import is unavailable.")

        if validated_path.is_dir():
            skill_dir = validated_path
        elif validated_path.name.lower() == _SKILL_MD_FILENAME.lower():
            skill_dir = validated_path.parent
        else:
            return await self._import_file(validated_path, import_file)

        if self._find_skill_md(skill_dir) is None:
            return _LibrarySkillImportOutcome(
                "No SKILL.md found in that folder."
            )

        skill_name = skill_dir.name
        try:
            await self._call_service(
                import_directory,
                skill_dir,
                mode="local",
                name=skill_name,
                trust_approved=False,
            )
        except Exception as exc:
            return self._failure(skill_name, exc)
        return self._success(skill_name)

    async def _import_file(
        self, file_path: Path, import_file: Any
    ) -> _LibrarySkillImportOutcome:
        suffix = file_path.suffix.lower()
        if suffix == ".zip":
            content_type = "application/zip"
            try:
                data = await asyncio.to_thread(file_path.read_bytes)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not read Library skill import file."
                )
                return _LibrarySkillImportOutcome("Could not read that file.")
        elif suffix == ".md":
            content_type = "text/markdown"
            try:
                text = await asyncio.to_thread(
                    file_path.read_text, encoding="utf-8", errors="strict"
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not read Library skill import file."
                )
                return _LibrarySkillImportOutcome("Could not read that file.")
            data = text.encode("utf-8")
        else:
            return _LibrarySkillImportOutcome("Unsupported file type.")

        try:
            record = await self._call_service(
                import_file,
                data,
                mode="local",
                filename=file_path.name,
                content_type=content_type,
                trust_approved=False,
            )
        except Exception as exc:
            return self._failure(self._safe_name(file_path.stem), exc)
        stored_name = str(record.get("name") or "") if isinstance(record, dict) else ""
        return self._success(
            self._safe_name(stored_name)
            if stored_name
            else self._safe_name(file_path.stem)
        )

    async def _import_url(self, url: str) -> _LibrarySkillImportOutcome:
        service = getattr(self._app_instance, "skills_scope_service", None)
        if service is None:
            return _LibrarySkillImportOutcome("Skill import is unavailable.")
        try:
            name_guess = self._safe_name(
                classify_skill_source_url(url).suggested_name
            )
        except RemoteSkillError:
            name_guess = self._safe_name(url.rstrip("/").rsplit("/", 1)[-1])
        try:
            result = await self._call_service(
                install_skill_from_url,
                url,
                scope_service=service,
            )
        except RemoteSkillError as exc:
            return _LibrarySkillImportOutcome(str(exc))
        except Exception as exc:
            return self._failure(name_guess, exc)
        if not isinstance(result, dict):
            return _LibrarySkillImportOutcome("Could not import that skill.")
        return self._success(self._safe_name(result.get("name", "")))

    @staticmethod
    async def _call_service(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
        def invoke() -> Any:
            result = callable_obj(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result

            async def await_result() -> Any:
                return await result

            return asyncio.run(await_result())

        return await asyncio.to_thread(invoke)

    @staticmethod
    def _find_skill_md(directory: Path) -> Path | None:
        exact = directory / _SKILL_MD_FILENAME
        if exact.is_file():
            return exact
        try:
            children = list(directory.iterdir())
        except Exception:
            return None
        return next(
            (
                child
                for child in children
                if child.is_file()
                and child.name.lower() == _SKILL_MD_FILENAME.lower()
            ),
            None,
        )

    @staticmethod
    def _safe_name(value: Any) -> str:
        text = sanitize_string(str(value or ""), max_length=64).strip()
        text = text.replace("<", "").replace(">", "")
        for pattern in ("javascript:", "onclick=", "onerror="):
            text = text.replace(pattern, "")
        if validate_text_input(text, max_length=64, allow_html=False):
            return text
        return ""

    @staticmethod
    def _success(skill_name: str) -> _LibrarySkillImportOutcome:
        return _LibrarySkillImportOutcome(
            f'Imported "{skill_name}" · re-review it in the trust panel',
            review_name=skill_name,
            clear_path=True,
            refresh_sources=True,
        )

    @staticmethod
    def _failure(skill_name: str, exc: Exception) -> _LibrarySkillImportOutcome:
        logger.opt(exception=True).warning(
            "Library skill import failed for {!r}.", skill_name
        )
        if "local_skill_exists:" in str(exc):
            return _LibrarySkillImportOutcome(
                f'Skipped — a skill named "{skill_name}" already exists.',
                clear_path=True,
            )
        return _LibrarySkillImportOutcome(
            "Could not import that skill.", clear_path=True
        )


def ensure_library_skill_import_coordinator(
    app_instance: Any,
) -> LibrarySkillImportCoordinator:
    """Return the one coordinator shared by every Library screen visit."""
    coordinator = getattr(app_instance, "library_skill_import_coordinator", None)
    if not isinstance(coordinator, LibrarySkillImportCoordinator):
        coordinator = LibrarySkillImportCoordinator(app_instance)
        setattr(app_instance, "library_skill_import_coordinator", coordinator)
    return coordinator
