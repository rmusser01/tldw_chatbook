"""Backup helpers relocated from the retired Tools & Settings window (TASK-33244).

The legacy window (``UI/Tools_Settings_Window.py``) was navigation-unreachable
and has been deleted; its DB backup/restore UI is superseded by the
Backup_Recovery surface. These helpers survive relocation because live test
coverage drives them directly:

- the TTS profile-backup integration tests pin the staged-manifest
  publication contract, including cooperative Textual-worker cancellation
  and cleanup-unwind behavior;
- the character-card backup export tests pin the BLOB-to-base64 backup shape
  (task-15769), which the import chain round-trips;
- the ChaChaNotes migration test resolves database files via
  ``SETTINGS_DATABASES``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger
from textual.worker import NoActiveWorker, get_current_worker

from tldw_chatbook.Utils.private_paths import create_private_text

SETTINGS_DATABASES = (
    ("chachanotes", "ChaChaNotes", "tldw_chatbook_ChaChaNotes"),
    ("prompts", "Prompts", "tldw_cli_prompts"),
    ("media", "Media", "tldw_cli_media_v2"),
    ("evals", "Evals", "tldw_cli_evals"),
    ("rag", "RAG", "tldw_cli_rag_indexing"),
    ("subscriptions", "Subscriptions", "tldw_cli_subscriptions"),
)


class BackupManifestPublication(NamedTuple):
    """Immutable ownership token for one staged manifest publication."""

    stage_path: Path
    final_path: Path


def _character_backup_json_default(value: Any) -> str:
    """json.dumps fallback for the row types SQLite hands back (task-15769).

    `list_character_cards` rows carry `created_at`/`last_modified` as
    ``datetime`` objects, which crashed the whole backup dump with
    ``TypeError: Object of type datetime is not JSON serializable`` even for
    image-free cards. Anything else unexpected still raises, so a new
    non-serializable column can never silently ship garbage.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_character_cards_for_backup(characters: list[dict[str, Any]]) -> str:
    """Serialize character-card rows into the JSON backup dump (task-15769).

    The raw `image` BLOB (``bytes``) is replaced by a plain-base64
    ``image_base64`` string -- the same compatibility shape
    `Chat_Functions.load_characters` produces, and deliberately NOT the
    data-URI form `export_character_card_to_json` embeds, because the import
    chain (`parse_v1_card` -> `import_and_save_character_from_file*`)
    b64decodes the raw string, so a prefixed value would not round-trip.
    """
    serializable: list[dict[str, Any]] = []
    for card in characters:
        card = dict(card)
        image = card.pop("image", None)
        if isinstance(image, (bytes, bytearray, memoryview)):
            card["image_base64"] = base64.b64encode(bytes(image)).decode("ascii")
        elif image is not None:
            # Not a BLOB (unexpected but conceivable via TEXT affinity);
            # keep the value rather than silently dropping it.
            card["image"] = image
        serializable.append(card)
    return json.dumps(
        serializable,
        indent=2,
        ensure_ascii=False,
        default=_character_backup_json_default,
    )


def raise_if_textual_worker_cancelled() -> None:
    """Cooperatively stop executor work after its Textual worker is cancelled."""

    try:
        worker = get_current_worker()
    except NoActiveWorker:
        return
    if worker.is_cancelled:
        raise asyncio.CancelledError from None


def has_active_control_flow() -> bool:
    """Return whether cleanup is unwinding a non-Exception signal."""

    active_exception = sys.exception()
    return active_exception is not None and not isinstance(
        active_exception,
        Exception,
    )


def unlink_backup_artifact(
    path: Path,
    phase: str,
    *,
    preserve_control_flow: bool = False,
) -> None:
    """Remove an unpublished artifact without masking control flow."""

    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Database backup phase={} cleanup=unlink failed", phase)
    except BaseException:
        if not preserve_control_flow:
            raise
        logger.warning("Database backup phase={} cleanup=unlink failed", phase)


def build_backup_manifest_publication(
    backup_dir: Path,
) -> BackupManifestPublication:
    """Build a path-only token for a same-directory manifest stage."""

    return BackupManifestPublication(
        stage_path=backup_dir / ".backup_info.json.tmp",
        final_path=backup_dir / "backup_info.json",
    )


def write_backup_manifest(
    timestamp: str,
    backed_up: tuple[tuple[str, Path], ...],
    publication: BackupManifestPublication,
) -> BackupManifestPublication:
    """Serialize and sync a staged manifest without publishing it."""

    stage_created = False
    completed = False
    try:
        backup_info = {
            "timestamp": timestamp,
            "databases": [
                {"name": name, "path": str(path)} for name, path in backed_up
            ],
        }
        raise_if_textual_worker_cancelled()
        serialized = io.StringIO()
        json.dump(backup_info, serialized, indent=2)
        create_private_text(
            publication.stage_path,
            serialized.getvalue(),
            application_owned_directory=publication.stage_path.parent,
        )
        stage_created = True
        raise_if_textual_worker_cancelled()
        completed = True
        return publication
    except Exception:
        raise RuntimeError("backup_manifest_write_failed") from None
    finally:
        if stage_created and not completed:
            unlink_backup_artifact(
                publication.stage_path,
                "manifest",
                preserve_control_flow=has_active_control_flow(),
            )
