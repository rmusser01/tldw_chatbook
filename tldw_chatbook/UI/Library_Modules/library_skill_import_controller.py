"""App-scoped single-flight owner for Library skill imports."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from ...Skills_Interop.skill_remote_fetch import (
    RemoteSkillError,
    RemoteSkillPackage,
    import_inspected_skill,
    inspect_skill_from_url,
    re_root_skill_zip,
)
from ...Skills_Interop.skill_package_inspection import (
    SkillPackageKind,
    inspect_skill_directory,
    inspect_skill_zip,
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
    candidates: tuple[str, ...] = ()
    recovery_actions: tuple[str, ...] = ()
    package_kind: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class _LibrarySkillImportOutcome:
    status: str
    review_name: str = ""
    clear_path: bool = False
    refresh_sources: bool = False
    candidates: tuple[str, ...] = ()
    recovery_actions: tuple[str, ...] = ()
    package_kind: str = ""
    retryable: bool = False
    pending_package: RemoteSkillPackage | "_PendingDirectory" | None = None


@dataclass(frozen=True)
class _PendingDirectory:
    """A validated local repository path awaiting one explicit subdir."""

    path: Path
    candidates: tuple[str, ...]
    stamps: tuple[
        tuple[str, tuple[int, ...], tuple[int, ...]], ...
    ]


class LibrarySkillImportCoordinator:
    """Own one non-cancellable import across replaceable Library screens.

    ``app_instance`` is the stable service owner for the coordinator's
    lifetime. No screen or widget is retained: settlement looks up the current
    routed screen through the mounted runtime app passed to :meth:`run`.
    """

    def __init__(self, app_instance: Any) -> None:
        self._app_instance = app_instance
        self._snapshot = LibrarySkillImportSnapshot()
        self._pending_package: RemoteSkillPackage | _PendingDirectory | None = None
        self._accepted_input = ""
        self._selected_candidate = ""

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
        self._pending_package = None
        self._accepted_input = ""
        self._selected_candidate = ""
        self._snapshot = LibrarySkillImportSnapshot(
            row_open=True,
            generation=self._snapshot.generation + 1,
        )
        return True

    def dismiss(self) -> bool:
        """Explicitly close the row without claiming cancellation."""
        if self._snapshot.in_flight:
            return False
        self._pending_package = None
        self._accepted_input = ""
        self._selected_candidate = ""
        self._snapshot = LibrarySkillImportSnapshot(
            generation=self._snapshot.generation + 1
        )
        return True

    def invalidate_row(self) -> None:
        """Fence callbacks from a row lifecycle that is no longer current."""
        self.update(generation=self._snapshot.generation + 1)

    def update_draft_path(self, path: str) -> bool:
        """Replace an idle draft and clear every outcome tied to the old input."""
        if self._snapshot.in_flight:
            return False
        self._pending_package = None
        self._accepted_input = ""
        self._selected_candidate = ""
        self.update(
            path=path,
            status="",
            review_name="",
            candidates=(),
            recovery_actions=(),
            package_kind="",
            retryable=False,
        )
        return True

    def claim(self, raw_path: str) -> bool:
        """Synchronously admit one operation before its app worker exists."""
        if self._snapshot.in_flight:
            return False
        accepted_input = (
            self._accepted_input
            if self._accepted_input and raw_path == self._snapshot.path
            else raw_path
        )
        self._accepted_input = accepted_input
        self._selected_candidate = ""
        self.update(
            row_open=True,
            path=self._display_path(accepted_input),
            status="Inspecting/importing…",
            review_name="",
            in_flight=True,
            generation=self._snapshot.generation + 1,
            candidates=(),
            recovery_actions=(),
            package_kind="",
            retryable=False,
        )
        return True

    def cancel_choice(self) -> bool:
        """Release a pre-import candidate choice and preserve its safe draft."""
        if not self._snapshot.candidates or self._pending_package is None:
            return False
        self._pending_package = None
        self._selected_candidate = ""
        self.update(
            status="",
            in_flight=False,
            candidates=(),
            recovery_actions=(),
            package_kind="",
            retryable=False,
            generation=self._snapshot.generation + 1,
        )
        return True

    def claim_candidate(self, candidate: str) -> bool:
        """Synchronously claim one displayed candidate before scheduling."""
        if (
            self._pending_package is None
            or self._selected_candidate
            or candidate not in self._snapshot.candidates
            or not self._snapshot.in_flight
        ):
            return False
        self._selected_candidate = candidate
        self.update(
            status="Inspecting/importing…",
            candidates=(),
            generation=self._snapshot.generation + 1,
        )
        return True

    def claim_retry(self) -> str | None:
        """Re-admit the last failed input without exposing its raw URL."""
        raw_path = self._accepted_input
        if not self._snapshot.retryable or not raw_path or not self.claim(raw_path):
            return None
        return raw_path

    async def run(self, raw_path: str, *, runtime_app: Any) -> None:
        """Run the accepted mutation and publish one authoritative receipt."""
        del raw_path
        operation = asyncio.create_task(
            self._run_and_settle(
                self._accepted_input, runtime_app=runtime_app
            )
        )
        await self._await_terminal_operation(operation, runtime_app=runtime_app)

    async def _await_terminal_operation(
        self, operation: asyncio.Task[None], *, runtime_app: Any
    ) -> None:
        """Keep one accepted operation owned through repeated outer cancellation."""
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

    async def run_candidate(self, candidate: str, *, runtime_app: Any) -> None:
        """Import one explicit candidate from the retained inspected bytes."""
        package = self._pending_package
        if (
            package is None
            or candidate != self._selected_candidate
            or not self._snapshot.in_flight
        ):
            return
        operation = asyncio.create_task(
            self._run_candidate_and_settle(
                package, candidate, runtime_app=runtime_app
            )
        )
        await self._await_terminal_operation(operation, runtime_app=runtime_app)

    async def _run_candidate_and_settle(
        self,
        package: RemoteSkillPackage | _PendingDirectory,
        candidate: str,
        *,
        runtime_app: Any,
    ) -> None:
        """Settle one selected candidate inside the cancellation shield."""
        fatal_error: BaseException | None = None
        try:
            service = getattr(self._app_instance, "skills_scope_service", None)
            if isinstance(package, _PendingDirectory):
                skill_dir = self._resolve_pending_directory_candidate(
                    package, candidate
                )
                if skill_dir is None:
                    raise ValueError("local_skill_candidate_changed")
                result = await self._call_service(
                    service.import_skill_directory,
                    skill_dir,
                    mode="local",
                    name=skill_dir.name,
                    trust_approved=False,
                )
            else:
                result = await self._call_service(
                    import_inspected_skill,
                    package,
                    candidate=candidate,
                    scope_service=service,
                )
            name = self._safe_name(
                result.get("name", "") if isinstance(result, dict) else ""
            )
            outcome = self._success(name)
        except asyncio.CancelledError:
            outcome = _LibrarySkillImportOutcome("Could not import that skill.")
        except Exception:
            logger.warning("Library selected skill import failed.")
            outcome = _LibrarySkillImportOutcome(
                "Could not import that skill.", retryable=True
            )
        except BaseException as exc:
            fatal_error = exc
            outcome = _LibrarySkillImportOutcome("Could not import that skill.")
        self._pending_package = None
        self._settle(outcome, runtime_app=runtime_app)
        if fatal_error is not None:
            raise fatal_error

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
        self._pending_package = outcome.pending_package
        self._selected_candidate = ""
        choice_pending = bool(outcome.candidates and outcome.pending_package)
        self.update(
            path="" if outcome.clear_path else self._snapshot.path,
            status=outcome.status,
            review_name=outcome.review_name,
            in_flight=choice_pending,
            candidates=outcome.candidates,
            recovery_actions=outcome.recovery_actions,
            package_kind=outcome.package_kind,
            retryable=outcome.retryable,
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
        except ValueError as exc:
            logger.warning(
                "Rejected Library skills import path; exception_type={}.",
                type(exc).__name__,
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
            inspection = await asyncio.to_thread(
                inspect_skill_directory, validated_path
            )
            if inspection.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY:
                pending = self._pending_directory(
                    validated_path, inspection.candidates
                )
                if pending is None:
                    return _LibrarySkillImportOutcome(
                        "That package is malformed or unsupported.",
                        package_kind=(
                            SkillPackageKind.MALFORMED_OR_UNSUPPORTED.value
                        ),
                    )
                return _LibrarySkillImportOutcome(
                    "Choose one skill to import.",
                    candidates=inspection.candidates,
                    package_kind=inspection.kind.value,
                    pending_package=pending,
                )
            if inspection.kind is not SkillPackageKind.ROOT_SKILL:
                return _LibrarySkillImportOutcome(
                    inspection.message,
                    recovery_actions=inspection.recovery_actions,
                    package_kind=inspection.kind.value,
                )
            candidate = inspection.candidates[0]
            skill_dir = validated_path / candidate if candidate else validated_path
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
            except Exception as exc:
                logger.warning(
                    "Could not read Library skill import file; exception_type={}.",
                    type(exc).__name__,
                )
                return _LibrarySkillImportOutcome("Could not read that file.")
            inspection = inspect_skill_zip(data, repository_source=False)
            if inspection.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY:
                package = RemoteSkillPackage(
                    inspection=inspection,
                    archive_bytes=data,
                    archive_sha256=hashlib.sha256(data).hexdigest(),
                    suggested_name=file_path.stem,
                )
                return _LibrarySkillImportOutcome(
                    "Choose one skill to import.",
                    candidates=inspection.candidates,
                    package_kind=inspection.kind.value,
                    pending_package=package,
                )
            if inspection.kind is not SkillPackageKind.ROOT_SKILL:
                return _LibrarySkillImportOutcome(
                    inspection.message,
                    package_kind=inspection.kind.value,
                )
            candidate = inspection.candidates[0]
            try:
                data, final_name = re_root_skill_zip(
                    data,
                    subdir=candidate,
                    suggested_name=(candidate.rsplit("/", 1)[-1] or file_path.stem),
                )
            except RemoteSkillError:
                return _LibrarySkillImportOutcome(
                    "That package is malformed or unsupported.",
                    package_kind=SkillPackageKind.MALFORMED_OR_UNSUPPORTED.value,
                )
            file_path = file_path.with_name(f"{final_name}.zip")
        elif suffix == ".md":
            content_type = "text/markdown"
            try:
                text = await asyncio.to_thread(
                    file_path.read_text, encoding="utf-8", errors="strict"
                )
            except Exception as exc:
                logger.warning(
                    "Could not read Library skill import file; exception_type={}.",
                    type(exc).__name__,
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
            package = await self._call_service(
                inspect_skill_from_url,
                url,
                scope_service=service,
            )
        except Exception:
            logger.warning("Library remote skill inspection failed.")
            return _LibrarySkillImportOutcome(
                "Could not fetch that skill package. Retry when access is available.",
                package_kind=SkillPackageKind.FETCH_OR_AUTH_FAILURE.value,
                retryable=True,
            )
        inspection = package.inspection
        if inspection.kind is SkillPackageKind.MULTI_SKILL_REPOSITORY:
            return _LibrarySkillImportOutcome(
                "Choose one skill to import.",
                candidates=inspection.candidates,
                package_kind=inspection.kind.value,
                pending_package=package,
            )
        if inspection.kind is not SkillPackageKind.ROOT_SKILL:
            return _LibrarySkillImportOutcome(
                inspection.message,
                recovery_actions=inspection.recovery_actions,
                package_kind=inspection.kind.value,
                retryable=(
                    inspection.kind is SkillPackageKind.FETCH_OR_AUTH_FAILURE
                ),
            )
        try:
            result = await self._call_service(
                import_inspected_skill,
                package,
                scope_service=service,
            )
        except Exception as exc:
            return self._failure(
                self._safe_name(package.suggested_name), exc
            )
        if not isinstance(result, dict):
            return _LibrarySkillImportOutcome(
                "Could not import that skill.", retryable=True
            )
        return self._success(self._safe_name(result.get("name", "")))

    @staticmethod
    def _display_path(raw_path: str) -> str:
        if not raw_path.startswith(("http://", "https://")):
            return raw_path
        try:
            parsed = urlsplit(raw_path)
            hostname = parsed.hostname
            if not hostname:
                return "Remote package URL"
            netloc = f"[{hostname}]" if ":" in hostname else hostname
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
            return urlunsplit((parsed.scheme, netloc, "", "", ""))
        except ValueError:
            return "Remote package URL"

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
    def _stat_stamp(path: Path, *, directory: bool) -> tuple[int, ...] | None:
        """Capture one no-follow identity for a candidate path."""
        try:
            info = os.lstat(path)
        except OSError:
            return None
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(
            info.st_mode
        )
        if not expected or stat.S_ISLNK(info.st_mode):
            return None
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    @classmethod
    def _pending_directory(
        cls, path: Path, candidates: tuple[str, ...]
    ) -> _PendingDirectory | None:
        """Freeze no-follow identities for every displayed local candidate."""
        try:
            root = path.resolve(strict=True)
        except OSError:
            return None
        if cls._stat_stamp(root, directory=True) is None:
            return None
        stamps: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for candidate in candidates:
            candidate_path = root.joinpath(*PurePosixPath(candidate).parts)
            directory_stamp = cls._stat_stamp(candidate_path, directory=True)
            body_stamp = cls._stat_stamp(
                candidate_path / _SKILL_MD_FILENAME, directory=False
            )
            if directory_stamp is None or body_stamp is None:
                return None
            stamps.append((candidate, directory_stamp, body_stamp))
        return _PendingDirectory(root, candidates, tuple(stamps))

    @classmethod
    def _resolve_pending_directory_candidate(
        cls, package: _PendingDirectory, candidate: str
    ) -> Path | None:
        """Revalidate containment and the inspected body identity before copy."""
        if candidate not in package.candidates:
            return None
        relative = PurePosixPath(candidate)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            return None
        try:
            root = package.path.resolve(strict=True)
        except OSError:
            return None
        if root != package.path or cls._stat_stamp(root, directory=True) is None:
            return None
        current = root
        for part in relative.parts:
            current /= part
            if cls._stat_stamp(current, directory=True) is None:
                return None
        try:
            resolved = current.resolve(strict=True)
        except OSError:
            return None
        if resolved != current or root not in resolved.parents:
            return None
        body = resolved / _SKILL_MD_FILENAME
        current_directory_stamp = cls._stat_stamp(resolved, directory=True)
        current_body_stamp = cls._stat_stamp(body, directory=False)
        expected = next(
            (stamp for stamp in package.stamps if stamp[0] == candidate), None
        )
        if expected is None or (
            current_directory_stamp,
            current_body_stamp,
        ) != expected[1:]:
            return None
        return resolved

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
            package_kind=SkillPackageKind.ROOT_SKILL.value,
        )

    @staticmethod
    def _failure(skill_name: str, exc: Exception) -> _LibrarySkillImportOutcome:
        logger.warning(
            "Library skill import failed; exception_type={}.",
            type(exc).__name__,
        )
        if type(exc) is ValueError and exc.args == (
            f"local_skill_exists:{skill_name}",
        ):
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
