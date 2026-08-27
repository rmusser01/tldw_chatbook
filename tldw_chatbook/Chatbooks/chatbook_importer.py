# chatbook_importer.py
# Description: Service for importing chatbooks/knowledge packs
#
"""
Chatbook Importer
-----------------

Handles the import and validation of chatbooks into the application.
"""

import heapq
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import List, Dict, Any, Optional, Tuple, Mapping
from loguru import logger

from .chatbook_models import ChatbookManifest, ContentType, ChatbookVersion
from .conflict_resolver import ConflictResolver, ConflictResolution
from ..Chat.chat_conversation_service import ChatConversationService
from ..Chat.citation_service_factory import (
    build_local_citation_conversation_service,
)
from ..Chat.provider_continuation import (
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    read_provider_continuation_json,
)
from ..Chat.assistant_generation_state import (
    AssistantGenerationState,
    normalize_assistant_generation_state,
)
from ..Chat.thinking_blocks import (
    preflight_thinking_history_policy,
    thinking_exchange_to_json,
)
from ..model_capabilities import moonshot_model_returns_reasoning_content
from ..DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..DB.Prompts_DB import PromptsDatabase
from ..config import load_console_library_migration_seed
from ..Prompt_Management.prompt_chatbook_record import (
    PromptChatbookRecordError,
    decode_chatbook_prompt_record,
)
from ..Character_Chat.character_card_formats import detect_and_parse_character_card
from ..Utils.path_validation import validate_filename
from ..Utils.paths import get_user_data_dir
from ..Utils.private_paths import secure_private_directory


_PROMPT_ARCHIVE_ITEM_ID = re.compile(r"(?:[1-9][0-9]*|item-[0-9]{6,})\Z")
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_COMPRESSION_RATIO = 1_000
_ARCHIVE_COPY_CHUNK_BYTES = 64 * 1024
_ARCHIVE_LIMIT_ERROR = "Chatbook archive exceeds safety limits."
_MAX_V2_GRAPH_MESSAGES = 10_000
_MAX_V2_MESSAGE_ID_CHARS = 256
_MAX_V2_TOTAL_ID_CHARS = 1024 * 1024
_MAX_V2_MESSAGE_CONTENT_CHARS = 1024 * 1024
_MAX_V2_TOTAL_CONTENT_CHARS = 16 * 1024 * 1024
_MAX_V2_TOTAL_PRIVATE_BYTES = 16 * 1024 * 1024
_MAX_V2_TOTAL_THINKING_BYTES = 16 * 1024 * 1024
_MAX_V2_GRAPH_DEPTH = 2_048


# Outcome vocabulary shared by ``ImportTypeResult`` and ``ImportStatus``
# (task-19734). These name what actually happened, so a caller can never read
# "the import ran" as "items were imported".
IMPORT_OUTCOME_NONE = "none"  # this type was not part of the import at all
IMPORT_OUTCOME_EXCLUDED = "excluded"  # present in the chatbook, not attempted
IMPORT_OUTCOME_EMPTY = "empty"  # nothing to import (an empty chatbook)
IMPORT_OUTCOME_IMPORTED = "imported"  # every attempted item landed
IMPORT_OUTCOME_PARTIAL = "partial"  # some landed, some did not
IMPORT_OUTCOME_SKIPPED = "skipped"  # nothing landed; everything already present
IMPORT_OUTCOME_FAILED = "failed"  # nothing landed and something went wrong
#
# ``empty`` is a claim about the FILE and ``excluded`` a claim about the RUN,
# and they must never be swapped (Qodo review of PR #1945): making
# ``total_items`` count only what the run attempts meant a chatbook whose
# items were all opted out of, or all of types this importer cannot write,
# reported "this chatbook contained no items" -- false, and contradicted by
# the per-type rows and warnings the same run produced.

# The two reasons an item present in a chatbook is never attempted. Defined
# once here so the importer's return message and the wizard's banner name them
# with the same words (task-19734).
LEFT_OUT_BY_OPTIONS_NOUN = "left out by your import options"
UNSUPPORTED_BY_IMPORTER_NOUN = "not supported by this importer"

# The content types this importer can actually write, in dispatch order.
# Anything else in a chatbook's selections is reported as unsupported rather
# than silently inflating the totals (task-19734).
_IMPORTABLE_CONTENT_TYPES: Tuple["ContentType", ...] = (
    ContentType.CHARACTER,
    ContentType.CONVERSATION,
    ContentType.NOTE,
    ContentType.PROMPT,
    ContentType.MEDIA,
    ContentType.KEPT_BRIEFING,
)


class ImportTypeResult:
    """Per-content-type outcome counters for a single import run.

    ``attempted`` is how many items of this type the import was asked to
    write; the other three are what actually happened to them. Nothing here
    is ever derived from a manifest's advertised totals -- that is the whole
    point (task-19734): the UI used to tick "✓ Imported conversations" off a
    manifest count, which stays true even when every item was skipped.
    """

    def __init__(self, content_type: "ContentType"):
        self.content_type = content_type
        self.attempted = 0
        self.excluded = 0
        self.unsupported = 0
        self.successful = 0
        self.skipped = 0
        self.failed = 0

    @property
    def accounted(self) -> int:
        """Items whose fate is known (some paths can bail before recording)."""
        return self.successful + self.skipped + self.failed

    @property
    def left_out(self) -> int:
        """Items present in the chatbook that this run never attempted.

        Two different reasons, deliberately counted apart: ``excluded`` is the
        user's own choice and ``unsupported`` is this importer's limit. They
        must not be reported with each other's words.
        """
        return self.excluded + self.unsupported

    @property
    def outcome(self) -> str:
        """What actually happened to this content type.

        An attempted type that recorded no successes and no skips is
        ``failed``, not ``imported``: an early return (a missing database
        path, say) leaves every counter at zero, and silence must not read
        as success.
        """
        if self.attempted <= 0:
            if self.left_out > 0:
                return IMPORT_OUTCOME_EXCLUDED
            return IMPORT_OUTCOME_NONE
        if self.successful <= 0:
            if self.failed > 0 or self.skipped <= 0:
                return IMPORT_OUTCOME_FAILED
            return IMPORT_OUTCOME_SKIPPED
        if self.successful >= self.attempted:
            return IMPORT_OUTCOME_IMPORTED
        return IMPORT_OUTCOME_PARTIAL

    def to_dict(self) -> dict:
        """Convert this type's result to a dictionary."""
        return {
            "content_type": self.content_type.value,
            "attempted": self.attempted,
            "excluded": self.excluded,
            "unsupported": self.unsupported,
            "successful": self.successful,
            "skipped": self.skipped,
            "failed": self.failed,
            "outcome": self.outcome,
        }


class ImportStatus:
    """Track import progress and results."""

    def __init__(self):
        self.total_items = 0
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        # Per-content-type results, keyed by ``ContentType`` (task-19734).
        self.by_type: Dict["ContentType", ImportTypeResult] = {}

    def result_for(self, content_type: "ContentType") -> ImportTypeResult:
        """Return (creating if needed) the result record for one content type."""
        result = self.by_type.get(content_type)
        if result is None:
            result = ImportTypeResult(content_type)
            self.by_type[content_type] = result
        return result

    def result_snapshot(self, content_type: "ContentType") -> ImportTypeResult:
        """Read one type's result without adding it to this run's records."""
        return self.by_type.get(content_type) or ImportTypeResult(content_type)

    def plan(self, content_type: "ContentType", attempted: int) -> ImportTypeResult:
        """Record how many items of ``content_type`` this run will attempt."""
        result = self.result_for(content_type)
        result.attempted += max(0, int(attempted))
        return result

    def exclude(self, content_type: "ContentType", count: int) -> ImportTypeResult:
        """Record items present in the chatbook that the user opted out of."""
        result = self.result_for(content_type)
        result.excluded += max(0, int(count))
        return result

    def mark_unsupported(
        self, content_type: "ContentType", count: int
    ) -> ImportTypeResult:
        """Record items of a type this importer cannot write.

        Counted, not just warned about: these items were in the chatbook and
        did not arrive, and a run that attempted nothing else must be able to
        say so rather than calling the chatbook empty (task-19734).
        """
        result = self.result_for(content_type)
        result.unsupported += max(0, int(count))
        return result

    def record_processed(self, content_type: "ContentType") -> None:
        """Count one item of ``content_type`` as having been reached."""
        self.processed_items += 1
        self.result_for(content_type)

    def record_success(self, content_type: "ContentType") -> None:
        """Count one successfully imported item of ``content_type``."""
        self.successful_items += 1
        self.result_for(content_type).successful += 1

    def record_skipped(self, content_type: "ContentType") -> None:
        """Count one skipped (already present) item of ``content_type``."""
        self.skipped_items += 1
        self.result_for(content_type).skipped += 1

    def record_failure(self, content_type: "ContentType") -> None:
        """Count one failed item of ``content_type``."""
        self.failed_items += 1
        self.result_for(content_type).failed += 1

    @property
    def planned_items(self) -> int:
        """Total items the run was asked to attempt, summed over types."""
        return sum(result.attempted for result in self.by_type.values())

    @property
    def excluded_items(self) -> int:
        """Items the user's own options kept out of this run."""
        return sum(result.excluded for result in self.by_type.values())

    @property
    def unsupported_items(self) -> int:
        """Items of a type this importer cannot write."""
        return sum(result.unsupported for result in self.by_type.values())

    @property
    def left_out_items(self) -> int:
        """Items the chatbook contained and this run never attempted."""
        return self.excluded_items + self.unsupported_items

    def left_out_detail(self) -> str:
        """Name why items were left out, in the words both surfaces use."""
        parts = [
            (self.excluded_items, LEFT_OUT_BY_OPTIONS_NOUN),
            (self.unsupported_items, UNSUPPORTED_BY_IMPORTER_NOUN),
        ]
        return ", ".join(f"{count} {noun}" for count, noun in parts if count > 0)

    @property
    def accounted_items(self) -> int:
        """Items whose fate this run actually recorded."""
        return self.successful_items + self.skipped_items + self.failed_items

    @property
    def attempted_items(self) -> int:
        """Items this run was asked to import, however it found out."""
        return max(self.planned_items, self.total_items, self.accounted_items)

    @property
    def unaccounted_items(self) -> int:
        """Attempted items whose fate was never recorded.

        Non-zero when a content type bails out before recording anything (a
        missing database path, say). The completion panel has to say so:
        otherwise Total silently exceeds Imported + Skipped + Failed and the
        summary reads "0 failed" for items that never landed (task-19734).
        """
        return self.attempted_items - self.accounted_items

    @property
    def outcome(self) -> str:
        """What actually happened across the whole import.

        Mirrors :attr:`ImportTypeResult.outcome`, so a run and each of its
        types are described in the same vocabulary.

        ``EMPTY`` is reserved for a chatbook that held nothing at all.  A
        chatbook that held items this run never attempted -- media the user
        opted out of, or types this importer cannot write -- is ``EXCLUDED``:
        "there was nothing" and "there was something and we attempted none of
        it" are different facts, and only one of them is about the file.
        """
        attempted = self.attempted_items
        if attempted <= 0:
            if self.left_out_items > 0:
                return IMPORT_OUTCOME_EXCLUDED
            return IMPORT_OUTCOME_EMPTY
        if self.successful_items <= 0:
            if self.failed_items > 0 or self.skipped_items <= 0:
                return IMPORT_OUTCOME_FAILED
            return IMPORT_OUTCOME_SKIPPED
        if self.successful_items >= attempted:
            return IMPORT_OUTCOME_IMPORTED
        return IMPORT_OUTCOME_PARTIAL

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> dict:
        """Convert status to dictionary."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "excluded_items": self.excluded_items,
            "unsupported_items": self.unsupported_items,
            "outcome": self.outcome,
            "by_type": {
                content_type.value: result.to_dict()
                for content_type, result in self.by_type.items()
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ChatbookImporter:
    """Service for importing chatbooks into the application."""

    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize the chatbook importer.

        Args:
            db_paths: Dictionary mapping database names to their paths
        """
        self.db_paths = db_paths
        self.temp_dir = secure_private_directory(
            get_user_data_dir() / "temp" / "imports",
            create=True,
            application_owned=True,
        ).lexical_path
        self.conflict_resolver = ConflictResolver()

    def _create_extract_dir(self, prefix: str) -> Path:
        """Create one collision-resistant owner-only extraction directory."""

        return Path(tempfile.mkdtemp(prefix=prefix, dir=self.temp_dir))

    @staticmethod
    def _validated_archive_parts(member: zipfile.ZipInfo) -> tuple[str, ...]:
        """Return safe relative path components for one archive member."""

        filename = member.filename
        if not filename or "\x00" in filename or "\\" in filename:
            raise ValueError(f"Unsafe archive member path: {filename!r}")
        relative = PurePosixPath(filename)
        parts = relative.parts
        if (
            relative.is_absolute()
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0].endswith(":")
        ):
            raise ValueError(f"Unsafe archive member path: {filename!r}")

        archived_mode = member.external_attr >> 16
        archived_type = stat.S_IFMT(archived_mode)
        if archived_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"Unsupported archive member type: {filename!r}")
        return parts

    def _extract_private_archive(
        self,
        chatbook_path: Path,
        extract_dir: Path,
    ) -> None:
        """Extract regular ZIP members with owner-only permissions."""

        with zipfile.ZipFile(chatbook_path, "r") as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise ValueError(_ARCHIVE_LIMIT_ERROR)
            total_bytes = 0
            validated: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
            for member in members:
                parts = self._validated_archive_parts(member)
                total_bytes += member.file_size
                if (
                    member.file_size > _MAX_ARCHIVE_MEMBER_BYTES
                    or total_bytes > _MAX_ARCHIVE_TOTAL_BYTES
                    or member.file_size
                    > max(member.compress_size, 1) * _MAX_ARCHIVE_COMPRESSION_RATIO
                ):
                    raise ValueError(_ARCHIVE_LIMIT_ERROR)
                validated.append((member, parts))

            for member, parts in validated:
                target = extract_dir.joinpath(*parts)
                if member.is_dir():
                    secure_private_directory(
                        target,
                        create=True,
                        application_owned=True,
                    )
                    continue

                secure_private_directory(
                    target.parent,
                    create=True,
                    application_owned=True,
                )
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                file_fd = os.open(target, flags, 0o600)
                try:
                    if hasattr(os, "fchmod"):
                        os.fchmod(file_fd, 0o600)
                    with os.fdopen(file_fd, "wb") as destination:
                        file_fd = -1
                        with archive.open(member, "r") as source:
                            written = 0
                            while chunk := source.read(_ARCHIVE_COPY_CHUNK_BYTES):
                                written += len(chunk)
                                if (
                                    written > member.file_size
                                    or written > _MAX_ARCHIVE_MEMBER_BYTES
                                ):
                                    raise ValueError(_ARCHIVE_LIMIT_ERROR)
                                destination.write(chunk)
                            if written != member.file_size:
                                raise ValueError(_ARCHIVE_LIMIT_ERROR)
                        destination.flush()
                        os.fsync(destination.fileno())
                finally:
                    if file_fd >= 0:
                        os.close(file_fd)

    def preview_chatbook(
        self, chatbook_path: Path
    ) -> Tuple[Optional[ChatbookManifest], Optional[str]]:
        """
        Preview a chatbook without importing it.

        Args:
            chatbook_path: Path to the chatbook file

        Returns:
            Tuple of (manifest, error_message)
        """
        extract_dir: Optional[Path] = None
        try:
            if chatbook_path.suffix != ".zip":
                return (
                    None,
                    "Unsupported chatbook format. Only ZIP files are supported.",
                )
            extract_dir = self._create_extract_dir("preview_")
            self._extract_private_archive(chatbook_path, extract_dir)

            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                return None, "Invalid chatbook: manifest.json not found"

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            manifest = ChatbookManifest.from_dict(manifest_data)

            return manifest, None

        except Exception as e:
            logger.error(f"Error previewing chatbook: {e}")
            return None, f"Error previewing chatbook: {str(e)}"
        finally:
            if extract_dir is not None:
                shutil.rmtree(extract_dir, ignore_errors=True)

    def import_chatbook(
        self,
        chatbook_path: Path,
        content_selections: Optional[Dict[ContentType, List[str]]] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.ASK,
        prefix_imported: bool = False,
        import_media: bool = True,
        import_embeddings: bool = False,
        import_status: Optional[ImportStatus] = None,
    ) -> Tuple[bool, str]:
        """
        Import a chatbook into the application.

        Args:
            chatbook_path: Path to the chatbook file
            content_selections: Optional dict of content types to specific IDs to import
            conflict_resolution: How to handle conflicts
            prefix_imported: Whether to prefix imported content titles
            import_media: Whether to import media files
            import_embeddings: Whether to import embeddings
            import_status: Optional status object populated with item-level results

        Returns:
            Tuple of (success, message)
        """
        logger.info(
            f"ChatbookImporter.import_chatbook: Starting import of {chatbook_path}"
        )
        logger.info(
            f"ChatbookImporter.import_chatbook: Options - conflict_resolution={conflict_resolution}, prefix_imported={prefix_imported}, import_media={import_media}, import_embeddings={import_embeddings}"
        )
        status = import_status if import_status else ImportStatus()
        extract_dir: Optional[Path] = None

        try:
            logger.info(f"Importing chatbook from {chatbook_path}")

            if chatbook_path.suffix != ".zip":
                error_msg = "Unsupported chatbook format. Only ZIP files are supported."
                status.add_error(error_msg)
                return False, error_msg
            extract_dir = self._create_extract_dir("import_")
            self._extract_private_archive(chatbook_path, extract_dir)

            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            logger.info(
                f"ChatbookImporter.import_chatbook: Looking for manifest at {manifest_path}"
            )
            if not manifest_path.exists():
                logger.error(
                    "ChatbookImporter.import_chatbook: manifest.json not found"
                )
                error_msg = "Invalid chatbook: manifest.json not found"
                status.add_error(error_msg)
                return False, error_msg

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            logger.info(
                f"ChatbookImporter.import_chatbook: Loaded manifest with {len(manifest_data.get('content', {}))} content types"
            )

            manifest = ChatbookManifest.from_dict(manifest_data)
            logger.info(
                f"ChatbookImporter.import_chatbook: Manifest - version {manifest.version}, {manifest.total_conversations} conversations, {manifest.total_notes} notes, {manifest.total_characters} characters, {manifest.total_media_items} media"
            )

            # Check version compatibility
            if manifest.version not in {ChatbookVersion.V1, ChatbookVersion.V2}:
                status.add_warning(
                    f"Chatbook version {manifest.version.value} may not be fully compatible"
                )

            # Set up content selections
            if content_selections is None:
                # Import everything by default
                content_selections = {}
                for item in manifest.content_items:
                    if item.type not in content_selections:
                        content_selections[item.type] = []
                    content_selections[item.type].append(item.id)

            # Record what each content type was asked to do BEFORE any of it
            # runs, so a type that dies before recording a single item still
            # reads as "attempted and produced nothing" rather than as absent
            # (task-19734). Media only counts as attempted when it is actually
            # going to be imported, and a content type this importer cannot
            # write is never counted as attempted -- otherwise the totals
            # carry a permanent unexplained shortfall.
            for planned_type in _IMPORTABLE_CONTENT_TYPES:
                if planned_type not in content_selections:
                    continue
                if planned_type is ContentType.MEDIA and not import_media:
                    status.exclude(
                        ContentType.MEDIA, len(content_selections[ContentType.MEDIA])
                    )
                    continue
                status.plan(planned_type, len(content_selections[planned_type]))

            for unsupported_type, unsupported_ids in content_selections.items():
                if unsupported_type in _IMPORTABLE_CONTENT_TYPES or not unsupported_ids:
                    continue
                status.mark_unsupported(unsupported_type, len(unsupported_ids))
                status.add_warning(
                    f"{len(unsupported_ids)} {unsupported_type.value} item(s) in this "
                    "chatbook are not supported by the importer and were not imported"
                )

            # Total items to import: what the run will actually attempt.
            status.total_items = status.planned_items

            # Import each content type
            if ContentType.CHARACTER in content_selections:
                # Import characters first as they may be dependencies
                self._import_characters(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.CHARACTER],
                    conflict_resolution,
                    prefix_imported,
                    status,
                )

            if ContentType.CONVERSATION in content_selections:
                self._import_conversations(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.CONVERSATION],
                    conflict_resolution,
                    prefix_imported,
                    status,
                )

            if ContentType.NOTE in content_selections:
                self._import_notes(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.NOTE],
                    conflict_resolution,
                    prefix_imported,
                    status,
                )

            if ContentType.PROMPT in content_selections:
                self._import_prompts(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.PROMPT],
                    conflict_resolution,
                    prefix_imported,
                    status,
                )

            if import_media and ContentType.MEDIA in content_selections:
                self._import_media(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.MEDIA],
                    conflict_resolution,
                    status,
                )

            if ContentType.KEPT_BRIEFING in content_selections:
                self._import_kept_briefings(
                    extract_dir,
                    manifest,
                    content_selections[ContentType.KEPT_BRIEFING],
                    status,
                )

            # The run succeeded unless nothing landed and something went wrong
            # (task-19734). A skip is not a success: an all-skipped re-import
            # returns True here because it is not an *error*, but its message
            # says in words that nothing was imported, and callers that need
            # to branch on what happened read ``status.outcome`` rather than
            # inferring an import from this boolean.
            outcome = status.outcome
            success = outcome != IMPORT_OUTCOME_FAILED

            # Items this importer cannot write are named in every message, not
            # only logged into ``warnings`` -- otherwise a chatbook of 8 items
            # of which 2 are importable reports "Successfully imported 2/2"
            # and the other 6 vanish without a word (task-19734).
            unsupported_note = (
                f"{status.unsupported_items} {UNSUPPORTED_BY_IMPORTER_NOUN}"
                if status.unsupported_items > 0
                else ""
            )

            if outcome == IMPORT_OUTCOME_EMPTY:
                message = "No items to import"
            elif outcome == IMPORT_OUTCOME_EXCLUDED:
                # Not "no items": the chatbook had items and this run
                # attempted none of them.
                message = (
                    "No items were imported: none of the "
                    f"{status.left_out_items} item(s) in this chatbook were "
                    f"attempted ({status.left_out_detail()})"
                )
            elif outcome in (IMPORT_OUTCOME_IMPORTED, IMPORT_OUTCOME_PARTIAL):
                details = []
                if status.skipped_items > 0:
                    details.append(f"{status.skipped_items} skipped")
                if status.failed_items > 0:
                    details.append(f"{status.failed_items} failed")
                if unsupported_note:
                    details.append(unsupported_note)

                message = f"Successfully imported {status.successful_items}/{status.total_items} items"
                if details:
                    message += f" ({', '.join(details)})"
            elif outcome == IMPORT_OUTCOME_SKIPPED:
                message = (
                    "No items were imported: "
                    f"{status.skipped_items}/{status.total_items} items were already "
                    "present and were skipped"
                )
                if unsupported_note:
                    message += f" ({unsupported_note})"
            else:
                message = "Failed to import any items from chatbook"
                if unsupported_note:
                    message += f" ({unsupported_note})"

            if success:
                logger.info(message)
            else:
                logger.error(message)

            return success, message

        except Exception as e:
            error_msg = f"Fatal error: {str(e)}"
            logger.error(f"Error importing chatbook: {e}")
            status.add_error(error_msg)
            return False, error_msg
        finally:
            if extract_dir is not None:
                shutil.rmtree(extract_dir, ignore_errors=True)

    def _import_conversations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        conversation_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus,
    ) -> None:
        """Import conversations."""
        logger.info(
            f"ChatbookImporter._import_conversations: Starting import of {len(conversation_ids)} conversations"
        )
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error(
                "ChatbookImporter._import_conversations: ChaChaNotes database path not configured"
            )
            status.add_error("ChaChaNotes database path not configured")
            return

        db = CharactersRAGDB(
            db_path,
            "chatbook_importer",
            console_library_migration_seed=load_console_library_migration_seed(),
        )
        conversation_service, _, _ = build_local_citation_conversation_service(
            db,
            sidecar_path=get_user_data_dir()
            / "tldw_chatbook_chat_rag_context.json",
        )
        conv_dir = extract_dir / "content" / "conversations"
        logger.info(
            f"ChatbookImporter._import_conversations: Looking for conversations in {conv_dir}"
        )

        for conv_id in conversation_ids:
            status.record_processed(ContentType.CONVERSATION)
            logger.info(
                f"ChatbookImporter._import_conversations: Processing conversation {conv_id} ({status.processed_items}/{len(conversation_ids)})"
            )

            try:
                # Find conversation file
                conv_file = self._conversation_file_path(
                    extract_dir, conv_dir, manifest, conv_id
                )
                if not conv_file.exists():
                    logger.warning(
                        f"ChatbookImporter._import_conversations: Conversation file not found: {conv_file.name}"
                    )
                    status.add_warning(f"Conversation file not found: {conv_file.name}")
                    status.record_failure(ContentType.CONVERSATION)
                    continue

                # Load conversation data
                with open(conv_file, "r", encoding="utf-8") as f:
                    conv_data = json.load(f)

                graph_messages = None
                if manifest.version == ChatbookVersion.V2:
                    if (
                        not isinstance(conv_data, dict)
                        or type(conv_id) is not str
                        or not conv_id.strip()
                        or type(conv_data.get("id")) is not str
                        or not conv_data["id"].strip()
                        or conv_data["id"] != conv_id
                    ):
                        raise ValueError("Invalid V2 conversation identity.")
                    graph_messages = self._validate_v2_conversation_graph(conv_data)
                    thinking_policy, policy_warning = (
                        preflight_thinking_history_policy(
                            conv_data.get("thinking_history_policy")
                        )
                    )
                    if policy_warning is not None:
                        status.add_warning(policy_warning)
                else:
                    thinking_policy = "auto"

                # Check for existing conversation with same name
                conv_name = conv_data["name"]
                if prefix_imported:
                    conv_name = f"[Imported] {conv_name}"

                # Check for existing conversations with same name
                existing_conversations = db.get_conversation_by_name(conv_name)
                logger.info(
                    f"ChatbookImporter._import_conversations: Found {len(existing_conversations) if existing_conversations else 0} existing conversations with name '{conv_name}'"
                )

                if existing_conversations:
                    # Handle conflict - use the first (most recent) conversation
                    existing = existing_conversations[0]
                    resolution = self.conflict_resolver.resolve_conversation_conflict(
                        existing, conv_data, conflict_resolution
                    )

                    if resolution == ConflictResolution.SKIP:
                        logger.info(
                            "ChatbookImporter._import_conversations: Skipping conversation due to conflict resolution"
                        )
                        status.record_skipped(ContentType.CONVERSATION)
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        old_name = conv_name
                        conv_name = self._generate_unique_name(conv_name, db)
                        logger.info(
                            f"ChatbookImporter._import_conversations: Renamed conversation from '{old_name}' to '{conv_name}'"
                        )

                # Create conversation
                character_id = conv_data.get("character_id")
                conv_dict = {
                    "title": conv_name,
                    "created_at": conv_data.get(
                        "created_at", datetime.now().isoformat()
                    ),
                    "updated_at": conv_data.get(
                        "updated_at", datetime.now().isoformat()
                    ),
                    "character_id": character_id,
                    "assistant_authority_id": None,
                    "root_id": f"imported_{conv_data.get('id', 'unknown')}",
                    "thinking_history_policy": thinking_policy,
                }
                # Stage all filesystem work FIRST (attachment byte loads),
                # so the transaction below holds the write lock only for
                # pure DB writes — no disk I/O inside the transaction.
                staged_messages = []
                source_messages = (
                    graph_messages
                    if graph_messages is not None
                    else conv_data.get("messages", [])
                )
                for msg in source_messages:
                    image_kwargs, attachment_rows = self._load_message_attachments(
                        extract_dir, msg, status
                    )
                    staged_messages.append((msg, image_kwargs, attachment_rows))

                # One outer transaction per conversation — per conversation,
                # not per chatbook, to preserve error-isolation semantics
                # (one bad conversation fails alone; others still import).
                # TransactionContextManager is depth-tracked/reentrant, so
                # add_conversation/add_message/set_message_attachments'
                # own `with self.transaction():` calls become nested and
                # only this outer block commits, once, per conversation
                # (task-250 / performance audit finding A5). A failure
                # partway through the message loop rolls back the whole
                # conversation (an isolation improvement — the except below
                # already counted that case as failed). Success accounting
                # happens AFTER the block so a failed COMMIT can never be
                # double-counted as both success and failure. Citation
                # context (a JSON side-store, not this DB) also persists
                # after commit, so it neither extends the transaction nor
                # records context for rows that get rolled back.
                imported_message_context: list[tuple[str, str, dict]] = []
                new_conv_id = None
                with db.transaction() as connection:
                    new_conv_id = db.add_conversation(conv_dict)
                    logger.info(
                        f"ChatbookImporter._import_conversations: Created conversation with ID {new_conv_id}"
                    )

                    if new_conv_id:
                        logger.info(
                            f"ChatbookImporter._import_conversations: Importing {len(staged_messages)} messages"
                        )
                        message_id_map: dict[str, str] = {}
                        if graph_messages is not None:
                            message_id_map = {
                                str(msg["id"]): str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_URL,
                                        f"chatbook:{new_conv_id}:{msg['id']}",
                                    )
                                )
                                for msg in graph_messages
                            }
                        for ordinal, (
                            msg,
                            image_kwargs,
                            attachment_rows,
                        ) in enumerate(staged_messages, start=1):
                            msg_dict = {
                                "conversation_id": new_conv_id,
                                "sender": msg["role"],
                                "content": msg["content"],
                                "timestamp": msg.get(
                                    "timestamp", datetime.now().isoformat()
                                ),
                            }
                            if graph_messages is not None:
                                old_id = str(msg["id"])
                                parent_id = msg.get("parent_id")
                                msg_dict.update(
                                    {
                                        "id": message_id_map[old_id],
                                        "parent_message_id": message_id_map.get(
                                            str(parent_id)
                                            if parent_id is not None
                                            else ""
                                        ),
                                        "role": msg["role"],
                                    }
                                )
                                continuation = self._imported_continuation_json(
                                    msg,
                                    ordinal=ordinal,
                                    status=status,
                                )
                                if continuation is not None:
                                    msg_dict["provider_continuation_json"] = (
                                        continuation
                                    )
                                thinking_json = msg.get(
                                    "_thinking_canonical_json"
                                )
                                if thinking_json is not None:
                                    msg_dict["thinking_blocks_json"] = thinking_json
                                continuation_checkpoint = (
                                    parse_provider_continuation_json(continuation)
                                    if continuation is not None
                                    else None
                                )
                                raw_state = msg.get("assistant_generation_state")
                                try:
                                    generation_state = (
                                        normalize_assistant_generation_state(
                                            role=msg["role"],
                                            raw_state=raw_state,
                                            has_valid_active_continuation=(
                                                continuation_checkpoint is not None
                                                and continuation_checkpoint.state
                                                == "active"
                                            ),
                                        )
                                    )
                                except ValueError:
                                    raise ValueError(
                                        "Invalid V2 conversation graph."
                                    ) from None
                                if (
                                    generation_state
                                    is AssistantGenerationState.CONTINUATION_ACTIVE
                                    and (
                                        continuation_checkpoint is None
                                        or continuation_checkpoint.state != "active"
                                    )
                                ):
                                    raise ValueError(
                                        "Invalid V2 conversation graph."
                                    )
                                msg_dict["assistant_generation_state"] = (
                                    generation_state.value
                                    if generation_state is not None
                                    else None
                                )
                            elif msg.get("_private") is not None:
                                status.add_warning(
                                    "Exact tool continuation was discarded for "
                                    f"message {ordinal}."
                                )
                            msg_dict.update(image_kwargs)
                            new_message_id = db.add_message(msg_dict)
                            if new_message_id:
                                if graph_messages is not None:
                                    variant_of = msg.get("variant_of")
                                    connection.execute(
                                        "UPDATE messages SET variant_of = ?, "
                                        "variant_number = ?, is_selected_variant = ?, "
                                        "total_variants = ?, deleted = ? WHERE id = ?",
                                        (
                                            message_id_map.get(
                                                str(variant_of)
                                                if variant_of is not None
                                                else ""
                                            ),
                                            msg["variant_number"],
                                            int(msg["is_selected_variant"]),
                                            msg["total_variants"],
                                            int(msg["deleted"]),
                                            new_message_id,
                                        ),
                                    )
                                if attachment_rows:
                                    db.set_message_attachments(
                                        str(new_message_id), attachment_rows
                                    )
                                imported_message_context.append(
                                    (str(new_conv_id), str(new_message_id), msg)
                                )
                        if graph_messages is not None:
                            active_leaf = conv_data.get("active_leaf_message_id")
                            connection.execute(
                                "UPDATE conversations SET active_leaf_message_id = ? "
                                "WHERE id = ?",
                                (message_id_map.get(active_leaf), new_conv_id),
                            )

                if new_conv_id:
                    for (
                        context_conv_id,
                        context_message_id,
                        msg,
                    ) in imported_message_context:
                        self._persist_imported_message_citation_context(
                            conversation_service,
                            context_conv_id,
                            context_message_id,
                            msg,
                        )
                    status.record_success(ContentType.CONVERSATION)
                    logger.info(
                        f"ChatbookImporter._import_conversations: Successfully imported conversation: {conv_name}"
                    )
                else:
                    status.record_failure(ContentType.CONVERSATION)
                    status.add_error(f"Failed to create conversation: {conv_name}")
                    logger.error(
                        f"ChatbookImporter._import_conversations: Failed to create conversation: {conv_name}"
                    )

            except Exception as e:
                status.record_failure(ContentType.CONVERSATION)
                status.add_error(f"Error importing conversation {conv_id}: {str(e)}")
                logger.opt(exception=True).error(
                    "ChatbookImporter._import_conversations: Error importing conversation {}",
                    conv_id,
                )

    @staticmethod
    def _validate_v2_conversation_graph(
        conversation: object,
    ) -> list[dict[str, Any]]:
        """Validate a complete V2 graph before allocating any local owner IDs."""
        if not isinstance(conversation, dict):
            raise ValueError("Invalid V2 conversation graph.")
        raw_messages = conversation.get("messages")
        if (
            not isinstance(raw_messages, list)
            or len(raw_messages) > _MAX_V2_GRAPH_MESSAGES
        ):
            raise ValueError("Invalid V2 conversation graph.")
        messages: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        orders: set[int] = set()
        total_id_chars = 0
        total_content_chars = 0
        total_private_bytes = 0
        total_thinking_bytes = 0
        for raw in raw_messages:
            if not isinstance(raw, dict):
                raise ValueError("Invalid V2 conversation graph.")
            message_id = raw.get("id")
            order = raw.get("order")
            role = raw.get("role")
            content = raw.get("content")
            if (
                not isinstance(message_id, str)
                or not message_id
                or len(message_id) > _MAX_V2_MESSAGE_ID_CHARS
                or message_id in by_id
                or type(order) is not int
                or order < 0
                or order in orders
                or role not in {"user", "assistant", "system", "tool"}
                or not isinstance(content, str)
                or len(content) > _MAX_V2_MESSAGE_CONTENT_CHARS
                or type(raw.get("deleted")) is not bool
                or type(raw.get("variant_number")) is not int
                or raw["variant_number"] < 1
                or type(raw.get("is_selected_variant")) is not bool
                or type(raw.get("total_variants")) is not int
                or raw["total_variants"] < 1
            ):
                raise ValueError("Invalid V2 conversation graph.")
            total_id_chars += len(message_id)
            total_content_chars += len(content)
            for link in ("parent_id", "variant_of"):
                target = raw.get(link)
                if target is not None and (
                    not isinstance(target, str)
                    or len(target) > _MAX_V2_MESSAGE_ID_CHARS
                ):
                    raise ValueError("Invalid V2 conversation graph.")
                total_id_chars += len(target or "")
            if (
                total_id_chars > _MAX_V2_TOTAL_ID_CHARS
                or total_content_chars > _MAX_V2_TOTAL_CONTENT_CHARS
            ):
                raise ValueError("Invalid V2 conversation graph.")
            item = dict(raw)
            thinking = raw.get("_thinking")
            if thinking is not None:
                if role != "assistant":
                    raise ValueError("Invalid V2 conversation graph.")
                try:
                    canonical_thinking = thinking_exchange_to_json(thinking)
                except ValueError:
                    raise ValueError("Invalid V2 conversation graph.") from None
                total_thinking_bytes += len(canonical_thinking.encode("utf-8"))
                if total_thinking_bytes > _MAX_V2_TOTAL_THINKING_BYTES:
                    raise ValueError("Invalid V2 conversation graph.")
                item["_thinking_canonical_json"] = canonical_thinking
            private = raw.get("_private")
            checkpoint = None
            if (
                isinstance(private, dict)
                and set(private) == {"provider_continuation"}
                and role == "assistant"
            ):
                checkpoint = read_provider_continuation_json(
                    private.get("provider_continuation")
                ).checkpoint
                if checkpoint is not None:
                    canonical = dump_provider_continuation_json(checkpoint)
                    private_bytes = len(
                        (f'{{"provider_continuation":{canonical}}}').encode("utf-8")
                    )
                    if (
                        total_private_bytes + private_bytes
                        > _MAX_V2_TOTAL_PRIVATE_BYTES
                    ):
                        item["_private"] = {"provider_continuation": None}
                        checkpoint = None
                    else:
                        total_private_bytes += private_bytes
            raw_state = raw.get("assistant_generation_state")
            if raw_state is not None and role != "assistant":
                raise ValueError("Invalid V2 conversation graph.")
            try:
                generation_state = normalize_assistant_generation_state(
                    role=role,
                    raw_state=raw_state,
                    has_valid_active_continuation=(
                        checkpoint is not None and checkpoint.state == "active"
                    ),
                )
            except ValueError:
                raise ValueError("Invalid V2 conversation graph.") from None
            if (
                generation_state
                is AssistantGenerationState.CONTINUATION_ACTIVE
                and (checkpoint is None or checkpoint.state != "active")
            ):
                raise ValueError("Invalid V2 conversation graph.")
            item["assistant_generation_state"] = (
                generation_state.value if generation_state is not None else None
            )
            messages.append(item)
            by_id[message_id] = item
            orders.add(order)
        if orders != set(range(len(messages))):
            raise ValueError("Invalid V2 conversation graph.")
        for message in messages:
            for link in ("parent_id", "variant_of"):
                target = message.get(link)
                if target is not None and (
                    target not in by_id or target == message["id"]
                ):
                    raise ValueError("Invalid V2 conversation graph.")

        # Resolve each parent chain once. Completed paths memoize both cycle
        # state and depth, so a long chain remains linear rather than quadratic.
        states: dict[str, int] = {}
        depths: dict[str, int] = {}
        for message in messages:
            current: str | None = str(message["id"])
            path: list[str] = []
            while current is not None and states.get(current, 0) == 0:
                states[current] = 1
                path.append(current)
                parent = by_id[current].get("parent_id")
                current = str(parent) if parent is not None else None
            if current is not None and states.get(current) == 1:
                raise ValueError("Invalid V2 conversation graph.")
            depth = depths.get(current, 0) if current is not None else 0
            for message_id in reversed(path):
                depth += 1
                if depth > _MAX_V2_GRAPH_DEPTH:
                    raise ValueError("Invalid V2 conversation graph.")
                depths[message_id] = depth
                states[message_id] = 2

        groups: dict[str, list[dict[str, Any]]] = {}
        for message in messages:
            root_id = str(message.get("variant_of") or message["id"])
            root = by_id[root_id]
            if root.get("variant_of") is not None:
                raise ValueError("Invalid V2 conversation graph.")
            if (
                message.get("parent_id") != root.get("parent_id")
                or message["role"] != root["role"]
            ):
                raise ValueError("Invalid V2 conversation graph.")
            groups.setdefault(root_id, []).append(message)
        for variants in groups.values():
            count = len(variants)
            if (
                {variant["variant_number"] for variant in variants}
                != set(range(1, count + 1))
                or sum(bool(variant["is_selected_variant"]) for variant in variants)
                != 1
                or any(variant["total_variants"] != count for variant in variants)
            ):
                raise ValueError("Invalid V2 conversation graph.")

        active_leaf = conversation.get("active_leaf_message_id")
        if active_leaf is not None and (
            not isinstance(active_leaf, str)
            or active_leaf not in by_id
            or by_id[active_leaf]["deleted"]
        ):
            raise ValueError("Invalid V2 conversation graph.")
        selected_path = conversation.get("selected_path_message_ids")
        if (
            not isinstance(selected_path, list)
            or len(selected_path) > _MAX_V2_GRAPH_DEPTH
            or any(
                not isinstance(message_id, str)
                or len(message_id) > _MAX_V2_MESSAGE_ID_CHARS
                for message_id in selected_path
            )
        ):
            raise ValueError("Invalid V2 conversation graph.")
        expected_path: list[str] = []
        current = active_leaf
        while current is not None:
            if by_id[current]["deleted"]:
                raise ValueError("Invalid V2 conversation graph.")
            expected_path.append(current)
            current = by_id[current].get("parent_id")
        expected_path.reverse()
        if selected_path != expected_path:
            raise ValueError("Invalid V2 conversation graph.")

        # Kahn ordering respects both parent and variant ownership without
        # repeatedly scanning the remaining graph.
        indegrees = {str(message["id"]): 0 for message in messages}
        dependents: dict[str, list[str]] = {
            str(message["id"]): [] for message in messages
        }
        for message in messages:
            message_id = str(message["id"])
            dependencies = {
                str(target)
                for target in (message.get("parent_id"), message.get("variant_of"))
                if target is not None
            }
            indegrees[message_id] = len(dependencies)
            for dependency in dependencies:
                dependents[dependency].append(message_id)
        ready = [
            (int(message["order"]), str(message["id"]))
            for message in messages
            if indegrees[str(message["id"])] == 0
        ]
        heapq.heapify(ready)
        ordered: list[dict[str, Any]] = []
        while ready:
            _, message_id = heapq.heappop(ready)
            ordered.append(by_id[message_id])
            for dependent in dependents[message_id]:
                indegrees[dependent] -= 1
                if indegrees[dependent] == 0:
                    heapq.heappush(
                        ready, (int(by_id[dependent]["order"]), dependent)
                    )
        if len(ordered) != len(messages):
            raise ValueError("Invalid V2 conversation graph.")
        return ordered

    @staticmethod
    def _imported_continuation_json(
        message: Mapping[str, Any],
        *,
        ordinal: int,
        status: ImportStatus,
    ) -> str | None:
        """Return validated private continuation or add one redacted warning."""
        private = message.get("_private")
        if private is None:
            return None
        checkpoint: ProviderContinuationCheckpoint | None = None
        if (
            isinstance(private, dict)
            and set(private) == {"provider_continuation"}
            and message.get("role") == "assistant"
        ):
            result = read_provider_continuation_json(
                private.get("provider_continuation")
            )
            checkpoint = result.checkpoint
        # TASK-19170: the exact-owner rule for complete preserved-thinking
        # checkpoints follows the versioned kimi reasoning family; pre-19170
        # family checkpoints ending with a tool round are kept (shape guard).
        if checkpoint is not None and (
            checkpoint.provider != "moonshot"
            or not moonshot_model_returns_reasoning_content(checkpoint.model)
            or checkpoint.state != "complete"
            or bool(checkpoint.rounds[-1].calls)
            or checkpoint.rounds[-1].assistant_content == message.get("content")
        ):
            return dump_provider_continuation_json(checkpoint)
        status.add_warning(
            f"Exact tool continuation was discarded for message {ordinal}."
        )
        return None

    @staticmethod
    def _load_message_attachments(
        extract_dir: Path,
        msg: Dict[str, Any],
        status: ImportStatus,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load a message's image attachments from the extracted chatbook.

        Position 0 restores through the legacy ``image_data``/
        ``image_mime_type`` message columns; positions >= 1 become
        ``message_attachments`` rows — matching the app's live read contract.
        Entries whose resolved path escapes the extraction root (a hostile
        chatbook) or whose file is missing are skipped with a warning; the
        message itself still imports.

        Args:
            extract_dir: Root the chatbook archive was extracted into.
            msg: The message payload from the conversation JSON.
            status: Import status collector for warnings.

        Returns:
            Tuple of (legacy image kwargs for ``add_message``,
            attachment rows for ``set_message_attachments``).
        """
        image_kwargs: Dict[str, Any] = {}
        rows: List[Dict[str, Any]] = []
        raw_entries = msg.get("attachments")
        if not isinstance(raw_entries, list):
            return image_kwargs, rows
        root = extract_dir.resolve()
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            relative = str(entry.get("file") or "")
            try:
                position = int(entry.get("position"))
            except (TypeError, ValueError):
                continue
            if position < 0:
                status.add_warning(
                    f"Skipped attachment with invalid position {position}: {relative}"
                )
                continue
            if not relative:
                continue
            # NOTE: path_validation.validate_path cannot bound this read — it
            # rejects ANY resolved path containing a dot component, and the
            # importer's own extraction root lives under ~/.local/share/….
            # Same posture, expressed locally: no dot components within the
            # archive-relative path (covers ../ and hidden files), plus a
            # resolve()-based containment check as the symlink backstop.
            if any(part.startswith(".") for part in Path(relative).parts):
                status.add_warning(
                    f"Skipped attachment outside chatbook archive: {relative}"
                )
                continue
            resolved = (extract_dir / relative).resolve()
            if root != resolved and root not in resolved.parents:
                status.add_warning(
                    f"Skipped attachment outside chatbook archive: {relative}"
                )
                continue
            if not resolved.is_file():
                status.add_warning(f"Attachment file missing from chatbook: {relative}")
                continue
            try:
                data = resolved.read_bytes()
            except OSError as exc:
                status.add_warning(f"Failed to read attachment file {relative}: {exc}")
                continue
            mime_type = str(entry.get("mime_type") or "image/png")
            display_name = str(entry.get("display_name") or "")
            if position == 0:
                image_kwargs = {"image_data": data, "image_mime_type": mime_type}
            else:
                rows.append(
                    {
                        "position": position,
                        "data": data,
                        "mime_type": mime_type,
                        "display_name": display_name,
                    }
                )
        rows.sort(key=lambda row: row["position"])
        return image_kwargs, rows

    @staticmethod
    def _conversation_file_path(
        extract_dir: Path,
        conv_dir: Path,
        manifest: ChatbookManifest,
        conv_id: str,
    ) -> Path:
        for item in manifest.content_items:
            if (
                item.id == conv_id
                and item.type == ContentType.CONVERSATION
                and item.file_path
            ):
                return ChatbookImporter._safe_manifest_relative_path(
                    extract_dir, item.file_path
                )
        fallback_filename = f"conversation_{conv_id}.json"
        validate_filename(fallback_filename)
        return conv_dir / fallback_filename

    @staticmethod
    def _safe_manifest_relative_path(base_dir: Path, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("Chatbook manifest file paths must be relative")
        for part in path.parts:
            validate_filename(part)
        resolved_base = base_dir.resolve()
        resolved_path = (resolved_base / path).resolve()
        resolved_path.relative_to(resolved_base)
        return resolved_path

    @staticmethod
    def _persist_imported_message_citation_context(
        conversation_service: ChatConversationService,
        conversation_id: str,
        message_id: str,
        message_payload: Mapping[str, Any],
    ) -> None:
        rag_context = {}
        exported_rag_context = message_payload.get("rag_context")
        if isinstance(exported_rag_context, Mapping):
            rag_context.update(dict(exported_rag_context))
        for key in ("citation_validation", "evidence_bundle"):
            value = message_payload.get(key)
            if value is not None:
                rag_context[key] = value

        citations = message_payload.get("citations")
        citation_items = citations if isinstance(citations, list) else []
        if not rag_context and not citation_items:
            return

        conversation_service.record_imported_legacy_citation_context(
            conversation_id,
            message_id,
            rag_context=rag_context,
            citations=[item for item in citation_items if isinstance(item, Mapping)],
        )

    def _import_notes(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        note_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus,
    ) -> None:
        """Import notes."""
        logger.info(
            f"ChatbookImporter._import_notes: Starting import of {len(note_ids)} notes"
        )
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error(
                "ChatbookImporter._import_notes: ChaChaNotes database path not configured"
            )
            status.add_error("ChaChaNotes database path not configured")
            return

        db = CharactersRAGDB(
            db_path,
            "chatbook_importer",
            console_library_migration_seed=load_console_library_migration_seed(),
        )
        notes_dir = extract_dir / "content" / "notes"
        logger.info(f"ChatbookImporter._import_notes: Looking for notes in {notes_dir}")

        for note_id in note_ids:
            status.record_processed(ContentType.NOTE)
            logger.info(
                f"ChatbookImporter._import_notes: Processing note {note_id} ({status.processed_items}/{len(note_ids)})"
            )

            try:
                # Find note item in manifest
                note_item = None
                for item in manifest.content_items:
                    if item.id == note_id and item.type == ContentType.NOTE:
                        note_item = item
                        break

                if not note_item or not note_item.file_path:
                    logger.warning(
                        f"ChatbookImporter._import_notes: Note metadata not found for ID: {note_id}"
                    )
                    status.add_warning(f"Note metadata not found for ID: {note_id}")
                    status.record_failure(ContentType.NOTE)
                    continue

                # Load note file
                note_file = extract_dir / note_item.file_path
                logger.info(
                    f"ChatbookImporter._import_notes: Loading note file from {note_file}"
                )
                if not note_file.exists():
                    logger.warning(
                        f"ChatbookImporter._import_notes: Note file not found: {note_file}"
                    )
                    status.add_warning(f"Note file not found: {note_file}")
                    status.record_failure(ContentType.NOTE)
                    continue

                # Parse markdown with frontmatter
                with open(note_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract frontmatter if present
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        # Parse frontmatter
                        parts[1].strip()
                        note_content = parts[2].strip()
                    else:
                        note_content = content
                else:
                    note_content = content

                # Check for existing note with same title
                note_title = note_item.title
                if prefix_imported:
                    note_title = f"[Imported] {note_title}"

                # Check for existing note
                existing = db.get_note_by_title(note_title)

                if existing:
                    # Handle conflict
                    resolution = self.conflict_resolver.resolve_note_conflict(
                        existing,
                        {"title": note_title, "content": note_content},
                        conflict_resolution,
                    )

                    if resolution == ConflictResolution.SKIP:
                        status.record_skipped(ContentType.NOTE)
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        note_title = self._generate_unique_note_title(note_title, db)

                # Create note
                # Note: keywords/tags are not stored in the notes table
                new_note_id = db.add_note(title=note_title, content=note_content)

                if new_note_id:
                    status.record_success(ContentType.NOTE)
                    logger.info(f"Imported note: {note_title}")
                else:
                    status.record_failure(ContentType.NOTE)
                    status.add_error(f"Failed to create note: {note_title}")

            except Exception as e:
                status.record_failure(ContentType.NOTE)
                status.add_error(f"Error importing note {note_id}: {str(e)}")
                logger.opt(exception=True).error(
                    "ChatbookImporter._import_notes: Error importing note {}",
                    note_id,
                )

    def _import_characters(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        character_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus,
    ) -> None:
        """Import characters."""
        logger.info(
            f"ChatbookImporter._import_characters: Starting import of {len(character_ids)} characters"
        )
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error(
                "ChatbookImporter._import_characters: ChaChaNotes database path not configured"
            )
            status.add_error("ChaChaNotes database path not configured")
            return

        db = CharactersRAGDB(
            db_path,
            "chatbook_importer",
            console_library_migration_seed=load_console_library_migration_seed(),
        )
        chars_dir = extract_dir / "content" / "characters"
        logger.info(
            f"ChatbookImporter._import_characters: Looking for characters in {chars_dir}"
        )

        for char_id in character_ids:
            status.record_processed(ContentType.CHARACTER)
            logger.info(
                f"ChatbookImporter._import_characters: Processing character {char_id} ({status.processed_items}/{len(character_ids)})"
            )

            try:
                # Find character file
                char_file = chars_dir / f"character_{char_id}.json"
                if not char_file.exists():
                    logger.warning(
                        f"ChatbookImporter._import_characters: Character file not found: {char_file.name}"
                    )
                    status.add_warning(f"Character file not found: {char_file.name}")
                    status.record_failure(ContentType.CHARACTER)
                    continue

                # Load character data
                with open(char_file, "r", encoding="utf-8") as f:
                    raw_char_data = json.load(f)

                # Detect and parse character card format
                parsed_card, format_name = detect_and_parse_character_card(
                    raw_char_data
                )
                logger.info(
                    f"ChatbookImporter._import_characters: Detected format '{format_name}' for character {char_id}"
                )
                if not parsed_card:
                    logger.error(
                        f"ChatbookImporter._import_characters: Failed to parse character card for {char_id} (format: {format_name})"
                    )
                    status.add_error(
                        f"Failed to parse character card for {char_id} (format: {format_name})"
                    )
                    status.record_failure(ContentType.CHARACTER)
                    continue

                # Log the detected format
                logger.info(
                    f"ChatbookImporter._import_characters: Successfully parsed character {char_id} from {format_name} format"
                )

                # Extract character data from parsed V2 format
                char_data = parsed_card.get("data", parsed_card)

                # Check for existing character with same name
                char_name = char_data.get("name", "Unknown")
                if prefix_imported:
                    char_name = f"[Imported] {char_name}"

                # Check for existing character
                existing = db.get_character_card_by_name(char_name)
                logger.info(
                    f"ChatbookImporter._import_characters: Found existing character: {True if existing else False}"
                )

                if existing:
                    # Handle conflict
                    resolution = self.conflict_resolver.resolve_character_conflict(
                        existing, char_data, conflict_resolution
                    )

                    if resolution == ConflictResolution.SKIP:
                        logger.info(
                            "ChatbookImporter._import_characters: Skipping character due to conflict resolution"
                        )
                        status.record_skipped(ContentType.CHARACTER)
                        continue
                    elif resolution == ConflictResolution.RENAME:
                        old_name = char_name
                        char_name = self._generate_unique_character_name(char_name, db)
                        logger.info(
                            f"ChatbookImporter._import_characters: Renamed character from '{old_name}' to '{char_name}'"
                        )

                # Create character with V2 formatted data
                # Map V2 fields to database fields
                card_data = {
                    "name": char_name,
                    "description": char_data.get("description", ""),
                    "personality": char_data.get("personality", ""),
                    "scenario": char_data.get("scenario", ""),
                    "first_message": char_data.get("first_mes", ""),
                    "example_messages": char_data.get("mes_example", ""),
                    "creator_notes": char_data.get("creator_notes", ""),
                    "system_prompt": char_data.get("system_prompt", ""),
                    "post_history_instructions": char_data.get(
                        "post_history_instructions", ""
                    ),
                    "alternate_greetings": char_data.get("alternate_greetings", []),
                    "tags": char_data.get("tags", []),
                    "creator": char_data.get("creator", ""),
                    "character_version": char_data.get("character_version", ""),
                    "extensions": char_data.get("extensions", {}),
                    "character_book": char_data.get("character_book"),
                    "version": 1,  # DB schema version
                    "format": format_name,  # Store original format for reference
                }

                # If the raw data had a 'card' field with additional data, preserve it
                if "card" in raw_char_data and isinstance(raw_char_data["card"], dict):
                    # Merge any additional fields from original card
                    for key, value in raw_char_data["card"].items():
                        if key not in card_data and value is not None:
                            card_data[key] = value

                new_char_id = db.add_character_card(card_data)
                logger.info(
                    f"ChatbookImporter._import_characters: Created character with ID {new_char_id}"
                )

                if new_char_id:
                    status.record_success(ContentType.CHARACTER)
                    logger.info(
                        f"ChatbookImporter._import_characters: Successfully imported character: {char_name}"
                    )
                else:
                    status.record_failure(ContentType.CHARACTER)
                    status.add_error(f"Failed to create character: {char_name}")
                    logger.error(
                        f"ChatbookImporter._import_characters: Failed to create character: {char_name}"
                    )

            except Exception as e:
                status.record_failure(ContentType.CHARACTER)
                status.add_error(f"Error importing character {char_id}: {str(e)}")
                logger.opt(exception=True).error(
                    "ChatbookImporter._import_characters: Error importing character {}",
                    char_id,
                )

    def _import_prompts(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        prompt_ids: List[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportStatus,
    ) -> None:
        """Import versioned or historical portable Prompt records."""
        db_path = self.db_paths.get("Prompts")
        if not db_path:
            status.add_error("Prompts database path not configured")
            return

        valid_prompt_ids: list[str] = []
        for prompt_id in prompt_ids:
            if (
                not isinstance(prompt_id, str)
                or _PROMPT_ARCHIVE_ITEM_ID.fullmatch(prompt_id) is None
            ):
                status.record_processed(ContentType.PROMPT)
                status.record_failure(ContentType.PROMPT)
                status.add_error("Unable to import Prompt item.")
                logger.error(
                    "ChatbookImporter._import_prompts: Prompt import failed "
                    "item=invalid category=shape"
                )
            else:
                valid_prompt_ids.append(prompt_id)
        if not valid_prompt_ids:
            return

        try:
            db = PromptsDatabase(db_path, "chatbook_importer")
        except Exception:
            for prompt_id in valid_prompt_ids:
                status.record_processed(ContentType.PROMPT)
                status.record_failure(ContentType.PROMPT)
                status.add_error("Unable to import Prompt item.")
                logger.error(
                    "ChatbookImporter._import_prompts: Prompt import failed "
                    "item={} category=database",
                    prompt_id,
                )
            return
        prompts_dir = extract_dir / "content" / "prompts"

        for prompt_id in valid_prompt_ids:
            status.record_processed(ContentType.PROMPT)

            try:
                # Find prompt file
                prompt_file = prompts_dir / f"prompt_{prompt_id}.json"
                if not prompt_file.exists():
                    status.add_error("Unable to import Prompt item.")
                    status.record_failure(ContentType.PROMPT)
                    logger.error(
                        "ChatbookImporter._import_prompts: Prompt import failed "
                        "item={} category=missing",
                        prompt_id,
                    )
                    continue

                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                decoded = decode_chatbook_prompt_record(prompt_data)

                prompt_name = decoded["name"]
                if prefix_imported:
                    prompt_name = f"[Imported] {prompt_name}"

                result = db.add_prompt(
                    name=prompt_name,
                    author=decoded["author"],
                    details=decoded["details"],
                    system_prompt=decoded["system_prompt"],
                    user_prompt=decoded["user_prompt"],
                    keywords=decoded["keywords"],
                    overwrite=False,
                    prompt_format=decoded["prompt_format"],
                    prompt_schema_version=decoded["prompt_schema_version"],
                    prompt_definition=decoded["prompt_definition"],
                    artifact_type=decoded["artifact_type"],
                )
                new_prompt_id = result[0] if result else None

                if new_prompt_id:
                    status.record_success(ContentType.PROMPT)
                    logger.info(
                        "ChatbookImporter._import_prompts: Prompt imported "
                        "item={} category=success",
                        prompt_id,
                    )
                else:
                    status.record_failure(ContentType.PROMPT)
                    status.add_error("Unable to import Prompt item.")
                    logger.error(
                        "ChatbookImporter._import_prompts: Prompt import failed "
                        "item={} category=database",
                        prompt_id,
                    )

            except Exception as exc:
                status.record_failure(ContentType.PROMPT)
                status.add_error("Unable to import Prompt item.")
                category = (
                    exc.category
                    if isinstance(exc, PromptChatbookRecordError)
                    else "read"
                    if isinstance(exc, (OSError, json.JSONDecodeError))
                    else "database"
                )
                logger.error(
                    "ChatbookImporter._import_prompts: Prompt import failed "
                    "item={} category={}",
                    prompt_id,
                    category,
                )
        db.close_connection()

    def _import_media(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        media_ids: List[str],
        conflict_resolution: ConflictResolution,
        status: ImportStatus,
    ) -> None:
        """Import media items."""
        db_path = self.db_paths.get("Media")
        if not db_path:
            status.add_error("Media database path not configured")
            return

        db = MediaDatabase(db_path, "chatbook_importer")
        media_dir = extract_dir / "content" / "media"
        metadata_dir = media_dir / "metadata"

        for media_id in media_ids:
            status.record_processed(ContentType.MEDIA)

            try:
                # Find media metadata file
                metadata_file = metadata_dir / f"media_{media_id}.json"
                if not metadata_file.exists():
                    status.add_warning(
                        f"Media metadata file not found: {metadata_file.name}"
                    )
                    status.record_failure(ContentType.MEDIA)
                    continue

                # Load media metadata
                with open(metadata_file, "r", encoding="utf-8") as f:
                    media_data = json.load(f)

                # Check for existing media with same title and URL
                title = media_data.get("title", "Untitled")
                url = media_data.get("url")

                # Check if media already exists by URL
                existing = None
                if url:
                    existing = db.get_media_by_url(url)

                if existing:
                    # Handle conflict
                    if conflict_resolution == ConflictResolution.SKIP:
                        status.record_skipped(ContentType.MEDIA)
                        logger.info(f"Skipped existing media: {title}")
                        continue
                    elif conflict_resolution == ConflictResolution.RENAME:
                        title = self._generate_unique_media_title(title, db)

                # Load content if available
                content = ""
                content_file = media_dir / f"media_{media_id}.txt"
                if content_file.exists():
                    with open(content_file, "r", encoding="utf-8") as f:
                        content = f.read()

                # Prepare media data for import
                keywords_raw = media_data.get("metadata", {}).get("media_keywords", "")
                if isinstance(keywords_raw, str):
                    keywords = [
                        word.strip() for word in keywords_raw.split(",") if word.strip()
                    ]
                elif isinstance(keywords_raw, (list, tuple)):
                    keywords = [
                        str(word).strip() for word in keywords_raw if str(word).strip()
                    ]
                else:
                    # Any other (non-str, non-sequence) shape -- e.g. a stray
                    # int/dict from a malformed manifest -- yields no keywords
                    # rather than crashing the whole import.
                    keywords = []

                # Add media to database. NOTE: this call previously used
                # three parameter names that do not exist on
                # ``MediaDatabase.add_media_with_keywords`` (``media_keywords``
                # instead of ``keywords``, ``summary`` instead of
                # ``analysis_content``) -- both raised ``TypeError`` for
                # every media import -- and treated its return value as a
                # bare id when it is actually a
                # ``(media_id, message, status)`` tuple, so even a fixed
                # call would have miscounted every import as successful.
                try:
                    new_media_id, _add_message, _add_status = (
                        db.add_media_with_keywords(
                            url=url,
                            title=title,
                            media_type=media_data.get("media_type"),
                            content=content or media_data.get("content", ""),
                            keywords=keywords,
                            prompt=media_data.get("metadata", {}).get("prompt"),
                            analysis_content=media_data.get("metadata", {}).get(
                                "summary"
                            ),
                            transcription_model=media_data.get("metadata", {}).get(
                                "transcription_model"
                            ),
                            transcription_provenance=media_data.get("metadata", {}).get(
                                "transcription_provenance"
                            ),
                            author=media_data.get("author"),
                            ingestion_date=media_data.get("metadata", {}).get(
                                "ingestion_date"
                            ),
                        )
                    )

                    if new_media_id:
                        status.record_success(ContentType.MEDIA)
                        logger.info(f"Imported media: {title}")
                    else:
                        status.record_failure(ContentType.MEDIA)
                        status.add_error(f"Failed to create media: {title}")

                except Exception as e:
                    status.record_failure(ContentType.MEDIA)
                    status.add_error(
                        f"Database error importing media '{title}': {str(e)}"
                    )

            except Exception as e:
                status.record_failure(ContentType.MEDIA)
                status.add_error(f"Error importing media {media_id}: {str(e)}")
                logger.error(f"Error importing media {media_id}: {e}")

    # Mirrors ChatbookCreator._KEPT_SCRIPTS_EXPORT_PAGE_SIZE, but this read is
    # only used to de-duplicate scripts with no `source_script_id` against
    # rows already present in the *target* DB (see the match loop below), not
    # to enumerate every script for export -- a single page is intentional
    # here, unlike the paginated export path.
    _KEPT_SCRIPTS_IMPORT_LIMIT = 1000

    @staticmethod
    def _kept_dt_key(value: Any) -> Any:
        """Normalize a kept-row datetime-ish value for equality comparison.

        The importer's `payload` values are always ISO strings (JSON has no
        datetime type); a freshly-queried `existing` row's `DATETIME`
        columns come back as real `datetime` objects (the connection's
        registered converter). Rendering both sides through `.isoformat()`
        (when present) lets the two representations compare equal.
        """
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    @classmethod
    def _kept_briefing_content_matches(
        cls, existing: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> bool:
        """True if a locally-existing kept briefing is byte-identical to an
        incoming one sharing the same `source_briefing_id` (already-present,
        safe to skip silently) vs. genuinely different content (a conflict
        that must never be silently overwritten).

        `kept_at` is deliberately excluded from this comparison: it is
        provenance of *when* the briefing was kept, not part of the
        briefing's content, so the same artifact kept at different moments
        (e.g. re-exported from a second device) must still compare equal
        and skip as already-present rather than spuriously conflict.
        """
        plain_fields = (
            "watchlist_name",
            "body_markdown",
            "covers_through_item_id",
            "selection_mode",
            "model_used",
            "item_count",
            "featured_count",
            "overflow_count",
            "origin",
        )
        if any(existing.get(f) != payload.get(f) for f in plain_fields):
            return False
        dt_fields = ("covers_from_ts", "original_created_at")
        return all(
            cls._kept_dt_key(existing.get(f)) == cls._kept_dt_key(payload.get(f))
            for f in dt_fields
        )

    @classmethod
    def _kept_script_content_matches(
        cls, existing: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> bool:
        """Same byte-identity check as `_kept_briefing_content_matches`, for
        one kept script.

        `kept_at` is deliberately excluded here too, for the same reason:
        it is provenance of the keeping, not content, so a script kept at
        different moments must still skip as already-present rather than
        spam a conflict.
        """
        plain_fields = ("preset_name", "roster_snapshot_json", "turns_json", "model_used")
        if any(existing.get(f) != payload.get(f) for f in plain_fields):
            return False
        return cls._kept_dt_key(existing.get("original_created_at")) == cls._kept_dt_key(
            payload.get("original_created_at")
        )

    @staticmethod
    def _kept_briefing_file_path(
        extract_dir: Path,
        kept_dir: Path,
        manifest: ChatbookManifest,
        kept_id: str,
    ) -> Path:
        for item in manifest.content_items:
            if (
                item.id == kept_id
                and item.type == ContentType.KEPT_BRIEFING
                and item.file_path
            ):
                return ChatbookImporter._safe_manifest_relative_path(
                    extract_dir, item.file_path
                )
        fallback_filename = f"kept_briefing_{kept_id}.json"
        validate_filename(fallback_filename)
        return kept_dir / fallback_filename

    def _import_kept_briefings(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        kept_briefing_ids: List[str],
        status: ImportStatus,
    ) -> None:
        """Import kept briefings and their kept scripts (task-1870).

        Policy: `source_briefing_id` is a device-local Subscriptions_DB id,
        so a cross-device import can collide with a *different* local kept
        briefing that happens to share the same source id. Rather than
        force this through the display-name-keyed ask/skip/rename/replace
        machinery in `ConflictResolver` (built for conversations/notes/
        characters, not a UNIQUE-source-id-keyed idempotent artifact), this
        mirrors the "raced keep" handling the keep service itself already
        uses (`Subscriptions/briefing_keep.py` -- see the kept-briefings
        design doc's delivery notes): try the insert; if `create_kept_
        briefing` raises `ConflictError` because a row for this source id
        already exists, fall back to the existing row -- silently if its
        content is byte-identical (an ordinary idempotent re-import), with
        an honest warning if it differs (a genuine conflict; the existing
        row is never overwritten). Kept scripts ride under the (possibly
        pre-existing) parent under the same policy, except NULL-source
        scripts (cast directly from a kept briefing, no subscriptions-side
        source) are deduped by content match within the parent instead,
        since NULL carries no identity of its own.
        """
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.error(
                "ChatbookImporter._import_kept_briefings: ChaChaNotes database path not configured"
            )
            status.add_error("ChaChaNotes database path not configured")
            return

        db = CharactersRAGDB(
            db_path,
            "chatbook_importer",
            console_library_migration_seed=load_console_library_migration_seed(),
        )
        kept_dir = extract_dir / "content" / "kept_briefings"

        for kept_id in kept_briefing_ids:
            status.record_processed(ContentType.KEPT_BRIEFING)
            try:
                kept_file = self._kept_briefing_file_path(
                    extract_dir, kept_dir, manifest, kept_id
                )
                if not kept_file.exists():
                    status.add_warning(
                        f"Kept briefing file not found: {kept_file.name}"
                    )
                    status.record_failure(ContentType.KEPT_BRIEFING)
                    continue

                with open(kept_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                source_briefing_id = payload["source_briefing_id"]

                newly_inserted = False
                conflict = False
                target_kept_id: Optional[int] = None
                try:
                    target_kept_id = db.create_kept_briefing(
                        source_briefing_id=source_briefing_id,
                        watchlist_name=payload.get("watchlist_name"),
                        body_markdown=payload["body_markdown"],
                        covers_through_item_id=payload.get(
                            "covers_through_item_id"
                        ),
                        covers_from_ts=payload.get("covers_from_ts"),
                        selection_mode=payload.get("selection_mode"),
                        model_used=payload.get("model_used"),
                        item_count=payload.get("item_count", 0),
                        featured_count=payload.get("featured_count", 0),
                        overflow_count=payload.get("overflow_count", 0),
                        origin=payload.get("origin", "manual"),
                        original_created_at=payload.get("original_created_at"),
                        kept_at=payload.get("kept_at"),
                    )
                    newly_inserted = True
                except ConflictError:
                    existing = db.get_kept_briefing_by_source(source_briefing_id)
                    if existing is None:
                        # Lost a race with another writer between the
                        # failed insert and this read -- a hard failure
                        # rather than a guess.
                        status.record_failure(ContentType.KEPT_BRIEFING)
                        status.add_error(
                            "Kept briefing conflict for "
                            f"source_briefing_id={source_briefing_id} could not "
                            "be resolved (row vanished mid-import)."
                        )
                        continue
                    target_kept_id = existing["id"]
                    if not self._kept_briefing_content_matches(existing, payload):
                        conflict = True

                # Count the briefing's own outcome now, before touching its
                # kept scripts: the row is already durably present (either
                # freshly inserted, or an existing row we deliberately left
                # alone), so a script-level failure below must not turn an
                # already-successful briefing insert into a false "failed"
                # count (task-1870 fix-wave F5 -- see the per-item try
                # around `_import_kept_scripts`).
                if newly_inserted:
                    status.record_success(ContentType.KEPT_BRIEFING)
                else:
                    status.record_skipped(ContentType.KEPT_BRIEFING)
                    if conflict:
                        status.add_warning(
                            "Kept briefing conflict: source_briefing_id="
                            f"{source_briefing_id} already exists locally with "
                            "different content; the existing kept briefing "
                            "and its kept script(s) were not modified."
                        )

                if conflict:
                    # Refuse the whole incoming item as a unit -- parent AND
                    # children. `target_kept_id` here is the *unrelated*
                    # local briefing that merely happens to share the same
                    # source id, so importing the incoming scripts under it
                    # would graft someone else's cast history onto the
                    # user's own briefing while the warning above claims
                    # nothing was touched (task-1870 fix-wave F1). The
                    # byte-identical (non-conflict) branch above is
                    # unaffected -- scripts still import additively there,
                    # which is the ordinary re-keep/idempotent-import path.
                    logger.info(
                        "ChatbookImporter._import_kept_briefings: kept briefing "
                        f"source_briefing_id={source_briefing_id} conflicts with "
                        "an existing local row; its kept scripts were not imported."
                    )
                    continue

                try:
                    scripts_inserted, scripts_present, scripts_conflicted = (
                        self._import_kept_scripts(
                            db, target_kept_id, payload.get("scripts") or []
                        )
                    )
                except Exception as script_exc:
                    # The briefing itself is already counted above and is
                    # durably in the DB -- an honest report says so, and
                    # names the script failure as a warning instead of
                    # reporting the whole item as failed (task-1870
                    # fix-wave F5).
                    status.add_warning(
                        f"Kept briefing (source_briefing_id={source_briefing_id}): "
                        f"kept scripts could not be imported: {script_exc}"
                    )
                    logger.opt(exception=True).error(
                        "ChatbookImporter._import_kept_briefings: Error importing "
                        "kept scripts for source_briefing_id={}",
                        source_briefing_id,
                    )
                    continue

                if scripts_conflicted:
                    status.add_warning(
                        f"Kept briefing (source_briefing_id={source_briefing_id}): "
                        f"{scripts_conflicted} kept script(s) already present "
                        "locally with different content and were not modified."
                    )
                logger.info(
                    "ChatbookImporter._import_kept_briefings: kept briefing "
                    f"source_briefing_id={source_briefing_id} "
                    f"({'inserted' if newly_inserted else 'already present'}); "
                    f"scripts inserted={scripts_inserted} present={scripts_present} "
                    f"conflicted={scripts_conflicted}"
                )

            except Exception as e:
                status.record_failure(ContentType.KEPT_BRIEFING)
                status.add_error(
                    f"Error importing kept briefing {kept_id}: {str(e)}"
                )
                logger.opt(exception=True).error(
                    "ChatbookImporter._import_kept_briefings: Error importing kept briefing {}",
                    kept_id,
                )

    def _import_kept_scripts(
        self,
        db: CharactersRAGDB,
        kept_briefing_id: int,
        script_payloads: List[Dict[str, Any]],
    ) -> Tuple[int, int, int]:
        """Import one kept briefing's kept scripts under its (possibly
        pre-existing) parent.

        Returns (inserted, already_present, conflicted) counts. These are
        deliberately kept out of `ImportStatus`'s top-level counters --
        scripts are not independently selectable content items, so they
        would inflate the "X/Y items" accounting beyond the selected kept
        briefing count; the caller surfaces conflicts via a warning and logs
        the full breakdown instead.
        """
        inserted = 0
        already_present = 0
        conflicted = 0
        # Lazily fetched, and only for NULL-source scripts: the DB state for
        # this kept briefing *before* this call touches it. A matched
        # candidate is popped out of this pool (not merely flagged) so each
        # pre-existing row can satisfy at most one incoming script -- two
        # incoming scripts with genuinely identical content (legal: NULLs
        # are mutually distinct) still both insert if only one matching row
        # pre-existed, while re-importing the same chatbook twice matches
        # one-for-one and adds nothing. Rows inserted earlier in *this same*
        # loop are deliberately never added to the pool, so a source export
        # that legitimately contains two distinct byte-identical scripts
        # still round-trips as two rows, not one.
        existing_scripts: Optional[List[Dict[str, Any]]] = None

        for script_payload in script_payloads:
            source_script_id = script_payload.get("source_script_id")
            preset_name = script_payload.get("preset_name", "")
            roster_snapshot_json = script_payload.get("roster_snapshot_json", "{}")
            turns_json = script_payload.get("turns_json", "[]")
            model_used = script_payload.get("model_used")
            original_created_at = script_payload.get("original_created_at")
            kept_at = script_payload.get("kept_at")

            if source_script_id is not None:
                try:
                    db.create_kept_script(
                        kept_briefing_id,
                        source_script_id=source_script_id,
                        preset_name=preset_name,
                        roster_snapshot_json=roster_snapshot_json,
                        turns_json=turns_json,
                        model_used=model_used,
                        original_created_at=original_created_at,
                        kept_at=kept_at,
                    )
                    inserted += 1
                except ConflictError:
                    existing = db.get_kept_script_by_source(source_script_id)
                    if existing is not None and self._kept_script_content_matches(
                        existing, script_payload
                    ):
                        already_present += 1
                    else:
                        conflicted += 1
                continue

            if existing_scripts is None:
                existing_scripts = db.list_kept_scripts(
                    kept_briefing_id, limit=self._KEPT_SCRIPTS_IMPORT_LIMIT
                )
            match_index = next(
                (
                    idx
                    for idx, row in enumerate(existing_scripts)
                    if row.get("source_script_id") is None
                    and self._kept_script_content_matches(row, script_payload)
                ),
                None,
            )
            if match_index is not None:
                existing_scripts.pop(match_index)
                already_present += 1
                continue

            db.create_kept_script(
                kept_briefing_id,
                source_script_id=None,
                preset_name=preset_name,
                roster_snapshot_json=roster_snapshot_json,
                turns_json=turns_json,
                model_used=model_used,
                original_created_at=original_created_at,
                kept_at=kept_at,
            )
            inserted += 1

        return inserted, already_present, conflicted

    def _generate_unique_media_title(self, base_title: str, db: MediaDatabase) -> str:
        """Generate a unique media title."""
        counter = 1
        new_title = f"{base_title} ({counter})"
        # MediaDatabase doesn't have a get_by_title method, so we'll just append a counter
        # This is fine since media is primarily identified by URL
        return new_title

    def _generate_unique_name(self, base_name: str, db: CharactersRAGDB) -> str:
        """Generate a unique conversation name."""
        counter = 1
        while True:
            new_name = f"{base_name} ({counter})"
            # Check if any conversations exist with this name
            if not db.get_conversation_by_name(new_name):  # Empty list is falsy
                return new_name
            counter += 1

    def _generate_unique_note_title(self, base_title: str, db: CharactersRAGDB) -> str:
        """Generate a unique note title."""
        counter = 1
        while True:
            new_title = f"{base_title} ({counter})"
            if not db.get_note_by_title(new_title):
                return new_title
            counter += 1

    def _generate_unique_character_name(
        self, base_name: str, db: CharactersRAGDB
    ) -> str:
        """Generate a unique character name."""
        counter = 1
        while True:
            new_name = f"{base_name} ({counter})"
            if not db.get_character_card_by_name(new_name):
                return new_name
            counter += 1
