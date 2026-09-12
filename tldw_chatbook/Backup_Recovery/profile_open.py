"""Fixed mounted-app acknowledgement for an explicitly launched recovered profile."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .activation import ActivationStore, _flush_existing, _private, _write
from .isolated_restore import _launch_state
from .profile_paths import lexical_path


class OpenedReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: int = 1
    checks_version: int = 1
    attempt: str
    profile_id: str
    generation: str
    operation_id: str
    installation_id: str
    config: str
    data: str


@dataclass(frozen=True)
class _Selection:
    profile_id: str
    control_root: Path
    attempt: str
    process: int
    expected: OpenedReceipt


_selected: _Selection | None = None


def _expected(profile_id, control_root, attempt):
    if not isinstance(attempt, str) or re.fullmatch(r"[0-9a-f]{32}", attempt) is None:
        raise ValueError("invalid_profile_launch_attempt")
    entry, witness = _launch_state(profile_id, control_root)
    return OpenedReceipt(
        attempt=attempt,
        profile_id=profile_id,
        generation=witness["generation"],
        operation_id=witness["operation_id"],
        installation_id=entry.installation_id,
        config=entry.config,
        data=entry.data,
    )


def _select(profile_id, control_root, attempt):
    """Called only by the existing fresh-process selection before app imports."""
    global _selected
    expected = _expected(profile_id, control_root, attempt)
    if _selected is not None:
        raise ValueError("profile_requires_fresh_process")
    _selected = _Selection(profile_id, control_root, attempt, os.getpid(), expected)


def _name(attempt):
    return "opened-" + attempt + ".json"


def opened_receipt(profile_id, control_root, attempt):
    """Read only the matching current locally verified launch result."""
    expected = _expected(profile_id, control_root, attempt)
    store = ActivationStore(control_root / "activation")
    with _private(store._generation(expected.generation)) as parent:
        try:
            receipt = OpenedReceipt.model_validate(
                bootstrap._read(parent, _name(attempt))
            )
        except FileNotFoundError:
            return None
    if receipt != expected or _expected(profile_id, control_root, attempt) != expected:
        raise ValueError("profile_open_receipt_changed")
    return receipt


def _check_local_content(app, selected):
    from tldw_chatbook import config

    from .local_content_lifetime import operation, worker_databases

    expected = selected.expected
    databases = (app.chachanotes_db, app.prompts_db, app.media_db)
    paths = (
        config.get_chachanotes_db_path(),
        config.get_prompts_db_path(),
        config.get_media_db_path(),
    )
    if (
        selected.process != os.getpid()
        or not app._ui_ready
        or not app._initial_screen_pushed
        or app.client_id != expected.installation_id
        or config.CLI_APP_CLIENT_ID != expected.installation_id
        or lexical_path(os.environ["TLDW_CONFIG_PATH"]) != Path(expected.config)
        or any(
            db is None or lexical_path(db.db_path) != lexical_path(path)
            for db, path in zip(databases, paths)
        )
    ):
        raise ValueError("profile_local_content_unavailable")
    with operation(paths), worker_databases(databases):
        if (
            _expected(selected.profile_id, selected.control_root, selected.attempt)
            != expected
        ):
            raise ValueError("profile_open_generation_changed")
        databases[0].count_notes()
        databases[1].list_prompts(page=1, per_page=1)
        databases[2].get_distinct_media_types()
        if (
            _expected(selected.profile_id, selected.control_root, selected.attempt)
            != expected
        ):
            raise ValueError("profile_open_generation_changed")
        store = ActivationStore(selected.control_root / "activation")
        with _private(store._generation(expected.generation)) as parent:
            try:
                _write(parent, _name(selected.attempt), expected)
            except FileExistsError:
                _flush_existing(parent, _name(selected.attempt), expected)


async def acknowledge_mounted(app):
    """Run fixed local reads after the actual successful first-screen refresh."""
    selected = _selected
    if selected is None or getattr(app, "_recovery_open_checked", False):
        return
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError
    from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError as MediaDatabaseError
    from tldw_chatbook.DB.Prompts_DB import DatabaseError as PromptsDatabaseError

    try:
        await asyncio.to_thread(_check_local_content, app, selected)
    except (
        OSError,
        ValueError,
        RuntimeError,
        AttributeError,
        TypeError,
        CharactersRAGDBError,
        MediaDatabaseError,
        PromptsDatabaseError,
    ):
        app.loguru_logger.warning(
            "Recovered profile local-open acknowledgement unavailable"
        )
    finally:
        app._recovery_open_checked = True
