"""Direct-user raw CLI submission outside the provider prompt queue."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
import uuid

from ...Tools.raw_cli_executor import (
    MAX_RAW_TIMEOUT_SECONDS,
    RawCliRequest,
    validate_raw_cli_request,
)
from ...Widgets.Console.console_composer_bar import (
    ConsoleDraftStash,
    classify_console_raw_draft,
)

_EMPTY_REFUSAL = "Raw CLI command is empty. The exact draft was restored."
_LOCKED_REFUSAL = (
    "Raw CLI is Locked. Unlock it in Settings > Privacy & Security, then arm it "
    "for this launch. The exact draft was restored."
)
_UNARMED_REFUSAL = (
    "Raw CLI is Unlocked but not armed for this launch. Arm it in Settings > "
    "Privacy & Security. The exact draft was restored."
)
_AUTHORITY_CHANGED_REFUSAL = (
    "Raw CLI authority changed before launch. The exact draft was restored."
)
_CONTAINMENT_REFUSAL = (
    "Raw CLI did not launch because authority changed or process containment "
    "could not be established. The exact draft was restored."
)


def restore_refused_raw_cli_stash(
    session_id: str | None,
    stash: ConsoleDraftStash,
    *,
    composer: Any | None,
    active_session_id: str | None,
    visible_session_id: str | None,
) -> bool:
    """Restore only when the composer unambiguously belongs to the origin."""
    if composer is None:
        return False
    if active_session_id != session_id or visible_session_id != session_id:
        return False
    composer.restore_stashed_draft(stash)
    return True


class ConsoleRawCliController:
    """Capture and start one trusted user command without touching a provider."""

    def __init__(
        self,
        *,
        raw_cli_runtime: Callable[[], Any],
        active_session_id: Callable[[], str | None],
        persisted_leaf_anchor: Callable[[str], str | None],
        selected_local_root: Callable[[str], Path | None],
        private_scratch_root: Callable[[str], Path],
        refusal_stash_bank: dict[str, list[Any]],
        restore_stash: Callable[[str | None, ConsoleDraftStash], bool],
        append_local_error: Callable[[str | None, str], None],
        append_store_marker: Callable[..., Any],
        update_store_marker: Callable[..., Any],
        agent_runs_db: Callable[[], Any],
        run_log_access: Callable[[], Any],
        start_worker: Callable[..., Any],
        marshal_to_ui: Callable[..., Any],
    ) -> None:
        self._raw_cli_runtime = raw_cli_runtime
        self._active_session_id = active_session_id
        self._persisted_leaf_anchor = persisted_leaf_anchor
        self._selected_local_root = selected_local_root
        self._private_scratch_root = private_scratch_root
        self._banked_stashes_by_session = refusal_stash_bank
        self._restore_stash = restore_stash
        self._append_local_error = append_local_error
        # Task 7/8 consume these already-wired boundaries; Task 6 must not
        # invent an interim persistence or transcript representation.
        self._append_store_marker = append_store_marker
        self._update_store_marker = update_store_marker
        self._agent_runs_db = agent_runs_db
        self._run_log_access = run_log_access
        self._start_worker = start_worker
        self._marshal_to_ui = marshal_to_ui

    def start_user_command(self, stash: ConsoleDraftStash) -> bool:
        """Start one trusted raw stash in its own non-exclusive thread worker."""
        classified = classify_console_raw_draft(stash)
        if classified.kind != "raw":
            return False
        session_id = self._active_session_id()
        if not classified.text.strip():
            return self._refuse(session_id, stash, _EMPTY_REFUSAL)

        runtime = self._raw_cli_runtime()
        if runtime.permitted is not True:
            return self._refuse(session_id, stash, _LOCKED_REFUSAL)
        if runtime.armed is not True:
            return self._refuse(session_id, stash, _UNARMED_REFUSAL)

        if not session_id:
            return self._refuse(
                session_id,
                stash,
                "Raw CLI needs an active Console session. The exact draft was restored.",
            )
        try:
            request = RawCliRequest(
                invocation_id=uuid.uuid4().hex,
                caller="user",
                command=classified.text,
                shell="auto",
                initial_directory=self._submission_root(session_id),
                timeout_seconds=MAX_RAW_TIMEOUT_SECONDS,
                console_session_id=session_id,
                transcript_anchor_id=self._persisted_leaf_anchor(session_id),
            )
            validate_raw_cli_request(request)
        except ValueError as exc:
            return self._refuse(
                session_id, stash, f"Raw CLI refused: {exc}. Draft restored."
            )
        except Exception:  # noqa: BLE001 -- submission snapshot fails locally
            return self._refuse(
                session_id,
                stash,
                "Raw CLI could not capture this Console context. The exact draft "
                "was restored.",
            )

        try:
            self._start_worker(
                partial(self._execute, runtime, request, session_id, stash),
                thread=True,
                exclusive=False,
                name=f"console-raw-cli-{request.invocation_id}",
            )
        except Exception:  # noqa: BLE001 -- failed worker admission is local refusal
            return self._refuse(
                session_id,
                stash,
                "Raw CLI worker could not start. The exact draft was restored.",
            )
        return True

    def _submission_root(self, session_id: str) -> Path:
        selected = self._selected_local_root(session_id)
        root = (
            selected if selected is not None else self._private_scratch_root(session_id)
        )
        return Path(root).resolve()

    def _execute(
        self,
        runtime: Any,
        request: RawCliRequest,
        session_id: str,
        stash: ConsoleDraftStash,
    ) -> None:
        try:
            result = runtime.execute(request, lambda _event: None)
        except Exception:  # noqa: BLE001 -- runtime already owns diagnostic detail
            self._marshal_to_ui(
                self._append_local_error,
                session_id,
                "Raw CLI execution failed locally.",
            )
            return
        if result.terminal_state == "refused":
            self._marshal_to_ui(
                self._refuse,
                session_id,
                stash,
                _AUTHORITY_CHANGED_REFUSAL,
            )
        elif result.terminal_state == "containment_unavailable":
            self._marshal_to_ui(
                self._refuse,
                session_id,
                stash,
                _CONTAINMENT_REFUSAL,
            )

    def _refuse(
        self,
        session_id: str | None,
        stash: ConsoleDraftStash,
        message: str,
    ) -> bool:
        restored = self._restore_stash(session_id, stash)
        if not restored and session_id is not None:
            self._banked_stashes_by_session.setdefault(session_id, []).append(stash)
        self._append_local_error(session_id, message)
        return False

    def restore_banked_stashes(self, session_id: str, composer: Any) -> int:
        """Prepend exact refused stashes once their origin is reconciled."""
        stashes = self._banked_stashes_by_session.pop(session_id, [])
        for stash in reversed(stashes):
            composer.restore_stashed_draft(stash)
        return len(stashes)


__all__ = ["ConsoleRawCliController", "restore_refused_raw_cli_stash"]
