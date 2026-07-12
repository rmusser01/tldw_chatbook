# chatbook_creator.py
# Description: Service for creating chatbooks/knowledge packs
#
"""
Chatbook Creator
----------------

Handles the creation and packaging of chatbooks from database content.
"""

import json
import os
import shutil
import tempfile
import threading
import zipfile
import hashlib
import html
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Set, Tuple
from loguru import logger

from .chatbook_models import (
    Chatbook, ChatbookManifest, ChatbookContent, 
    ContentItem, ContentType, ChatbookVersion, Relationship
)
from .error_handler import ChatbookErrorHandler, ChatbookErrorType, safe_chatbook_operation
from ..Chat.chat_conversation_service import ChatConversationService
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..DB.Prompts_DB import PromptsDatabase
from ..DB.RAG_Indexing_DB import RAGIndexingDB
from ..DB.Evals_DB import EvalsDB
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Utils.input_validation import sanitize_string
from ..Utils.path_validation import validate_filename
from ..Utils.text import sanitize_filename
from ..Utils.paths import get_user_data_dir


CITATION_MESSAGE_EXPORT_KEYS = ("citation_validation", "evidence_bundle", "citations")
MAX_CITATION_REPORT_SNIPPET_CHARS = 1000
MAX_CITATION_REPORT_MESSAGES = 100
MAX_CITATION_REPORT_REFERENCES_PER_MESSAGE = 50
MAX_CITATION_REPORT_TOTAL_REFERENCES = 200


def _coerce_media_timestamp(value: Any) -> Optional[datetime]:
    """Normalize a Media DB ``DATETIME`` column to a ``datetime``.

    ``MediaDatabase`` opens its connection with ``PARSE_DECLTYPES`` and a
    registered ``DATETIME`` converter, so ``ingestion_date``/
    ``last_modified`` come back as real ``datetime`` instances (not ISO
    strings) -- ``datetime.fromisoformat`` would raise on them directly.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ExportProgress:
    """A single progress tick emitted during chatbook creation."""
    phase: str
    current: int
    total: int


class ChatbookExportCancelled(Exception):
    """Raised internally when cancel_check() returns True at a checkpoint."""


class ChatbookCreator:
    """Service for creating chatbooks from database content."""

    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize the chatbook creator.
        
        Args:
            db_paths: Dictionary mapping database names to their paths
        """
        logger.info(f"ChatbookCreator.__init__: Initializing with db_paths={db_paths}")
        self.db_paths = db_paths
        # Use cross-platform user data directory
        user_data_dir = get_user_data_dir()
        self.temp_dir = user_data_dir / "temp" / "chatbooks"
        logger.info(f"ChatbookCreator.__init__: Creating temp directory at {self.temp_dir}")
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_root = Path(tempfile.gettempdir()) / "tldw_chatbook" / "chatbooks"
            logger.warning(
                f"ChatbookCreator.__init__: Failed to create configured temp directory "
                f"{self.temp_dir}: {exc}. Falling back to {fallback_root}"
            )
            self.temp_dir = fallback_root
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.missing_dependencies: Set[int] = set()
        self.auto_included_characters: Set[int] = set()
        self._selected_characters: Set[str] = set()  # Track explicitly selected characters
        # Progress/cancel hooks are stored per-thread so a single ChatbookCreator
        # instance reused across exports (e.g. ChatbookCreationWindow keeps one)
        # can never have one export's callbacks overwrite another's if two ever
        # run concurrently on different threads.
        self._thread_local = threading.local()
        logger.info("ChatbookCreator.__init__: Initialization complete")

    def _emit_progress(self, phase: str, current: int, total: int) -> None:
        cb = getattr(self._thread_local, "progress_callback", None)
        if cb is None:
            return
        try:
            cb(ExportProgress(phase=phase, current=current, total=total))
        except Exception:
            logger.opt(exception=True).debug("ChatbookCreator: progress_callback raised; ignored")

    def _check_cancel(self) -> None:
        cancel_check = getattr(self._thread_local, "cancel_check", None)
        if cancel_check is not None and cancel_check():
            raise ChatbookExportCancelled()

    def _cleanup_run(self, work_dir: Optional[Path], partial_path: Optional[Path]) -> None:
        """Remove this run's temp artifacts on any exit path.

        ``work_dir`` is a temp directory we created (safe to rmtree).
        ``partial_path`` is a sibling of the user-chosen destination and is only
        ever a *file* we wrote, so it is unlinked as a file and NEVER rmtree'd —
        a directory unexpectedly sitting at ``<dest>.partial`` must not be
        recursively deleted.
        """
        if work_dir is not None:
            try:
                if work_dir.is_dir():
                    shutil.rmtree(work_dir, ignore_errors=True)
            except OSError:
                logger.opt(exception=True).debug(f"ChatbookCreator: could not remove work_dir {work_dir}")
        if partial_path is not None:
            try:
                if partial_path.is_file():
                    partial_path.unlink()
            except OSError:
                logger.opt(exception=True).debug(f"ChatbookCreator: could not remove partial {partial_path}")

    def create_chatbook(
        self,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        output_path: Path,
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "thumbnail",
        include_embeddings: bool = False,
        tags: List[str] = None,
        categories: List[str] = None,
        auto_include_dependencies: bool = True,
        progress_callback: Optional[Callable[["ExportProgress"], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a chatbook from selected content.
        
        Args:
            name: Name of the chatbook
            description: Description of the chatbook
            content_selections: Dictionary mapping content types to lists of IDs
            output_path: Path where the chatbook should be saved
            author: Author name (optional)
            include_media: Whether to include media files
            media_quality: Quality of media to include (thumbnail/compressed/original)
            include_embeddings: Whether to include embeddings
            tags: List of tags for the chatbook
            categories: List of categories for the chatbook
            auto_include_dependencies: Whether to automatically include missing character dependencies
            
        Returns:
            Tuple of (success, message, dependency_info)
            dependency_info contains:
                - missing_dependencies: List of character IDs that were referenced but not included
                - auto_included: List of character IDs that were automatically included
        """
        logger.info(f"ChatbookCreator.create_chatbook: Starting creation of '{name}'")
        logger.info(f"ChatbookCreator.create_chatbook: Options - include_media={include_media}, media_quality={media_quality}, include_embeddings={include_embeddings}, auto_include_dependencies={auto_include_dependencies}")
        logger.info(f"ChatbookCreator.create_chatbook: Content selections - {[(t.value, len(ids)) for t, ids in content_selections.items()]}")

        self._thread_local.progress_callback = progress_callback
        self._thread_local.cancel_check = cancel_check
        work_dir: Optional[Path] = None
        partial_path: Optional[Path] = None
        try:
            # Reset dependency tracking
            self.missing_dependencies.clear()
            self.auto_included_characters.clear()
            self._selected_characters = set(content_selections.get(ContentType.CHARACTER, []))
            logger.info(f"ChatbookCreator.create_chatbook: Tracking {len(self._selected_characters)} explicitly selected characters")
            
            # Create temporary working directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = Path(tempfile.mkdtemp(prefix=f"chatbook_{timestamp}_", dir=self.temp_dir))
            logger.info(f"ChatbookCreator.create_chatbook: Working directory created at {work_dir}")
            
            # Initialize manifest
            logger.info("ChatbookCreator.create_chatbook: Initializing manifest")
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=name,
                description=description,
                author=author,
                include_media=include_media,
                include_embeddings=include_embeddings,
                media_quality=media_quality,
                tags=tags or [],
                categories=categories or []
            )
            logger.info(f"ChatbookCreator.create_chatbook: Manifest created - version {manifest.version}, media={include_media}, embeddings={include_embeddings}")
            
            # Initialize content container
            content = ChatbookContent()
            
            # Process each content type
            logger.info(f"ChatbookCreator.create_chatbook: Processing content types")
            
            # Collect conversations
            if ContentType.CONVERSATION in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.CONVERSATION])} conversations")
                self._collect_conversations(
                    content_selections[ContentType.CONVERSATION],
                    work_dir,
                    manifest,
                    content,
                    auto_include_dependencies
                )
            
            # Collect notes
            if ContentType.NOTE in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.NOTE])} notes")
                self._collect_notes(
                    content_selections[ContentType.NOTE],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect characters
            if ContentType.CHARACTER in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.CHARACTER])} characters")
                self._collect_characters(
                    content_selections[ContentType.CHARACTER],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect media (if enabled)
            if include_media and ContentType.MEDIA in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.MEDIA])} media items (quality={media_quality})")
                self._collect_media(
                    content_selections[ContentType.MEDIA],
                    work_dir,
                    manifest,
                    content,
                    media_quality
                )
            
            # Collect prompts
            if ContentType.PROMPT in content_selections:
                logger.info(f"ChatbookCreator.create_chatbook: Collecting {len(content_selections[ContentType.PROMPT])} prompts")
                self._collect_prompts(
                    content_selections[ContentType.PROMPT],
                    work_dir,
                    manifest,
                    content
                )
            
            # Auto-discover relationships
            logger.info("ChatbookCreator.create_chatbook: Discovering relationships")
            self._discover_relationships(manifest, content)
            
            # Update statistics
            manifest.total_conversations = len(content.conversations)
            manifest.total_notes = len(content.notes)
            manifest.total_characters = len(content.characters)
            manifest.total_media_items = len(content.media_items)
            manifest.total_prompts = len(content.prompts)
            logger.info(f"ChatbookCreator.create_chatbook: Final stats - conversations={manifest.total_conversations}, notes={manifest.total_notes}, characters={manifest.total_characters}, media={manifest.total_media_items}, prompts={manifest.total_prompts}")
            
            # Write manifest
            manifest_path = work_dir / "manifest.json"
            logger.info(f"ChatbookCreator.create_chatbook: Writing manifest to {manifest_path}")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Create README
            logger.info("ChatbookCreator.create_chatbook: Creating README")
            self._create_readme(work_dir, manifest)
            
            # Package into archive (atomic finalize: write .partial, then os.replace)
            if output_path.suffix != '.zip':
                output_path = output_path.with_suffix('.zip')
            partial_path = output_path.with_name(output_path.name + ".partial")
            logger.info(f"ChatbookCreator.create_chatbook: Creating ZIP archive at {output_path}")
            self._create_zip_archive(work_dir, output_path, partial_path)
            
            # Best-effort size calc: the archive is already finalized on disk
            # (os.replace done inside _create_zip_archive), so a stat() failure
            # here must NOT flip a successful export into a reported failure.
            try:
                manifest.total_size_bytes = output_path.stat().st_size
                logger.info(f"ChatbookCreator.create_chatbook: Archive size: {manifest.total_size_bytes} bytes")
            except OSError:
                logger.opt(exception=True).debug("ChatbookCreator.create_chatbook: could not stat finalized archive")
            
            # (temp work_dir + any leftover .partial are cleaned up in the
            # `finally` below, on every exit path — success, cancel, error.)

            # Prepare dependency info
            dependency_info = {
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters)
            }
            
            # Build success message
            message = f"Chatbook created successfully at {output_path}"
            if self.auto_included_characters:
                message += f". Auto-included {len(self.auto_included_characters)} character dependencies"
            if self.missing_dependencies:
                message += f". Warning: {len(self.missing_dependencies)} character dependencies are missing"
            
            logger.info(f"ChatbookCreator.create_chatbook: Success - {message}")
            return True, message, dependency_info
            
        except ChatbookExportCancelled:
            logger.info("ChatbookCreator.create_chatbook: cancelled by request")
            return False, "Export cancelled", {
                "cancelled": True,
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters),
            }
        except Exception as e:
            logger.opt(exception=True).error("ChatbookCreator.create_chatbook: Error creating chatbook")
            dependency_info = {
                "missing_dependencies": list(self.missing_dependencies),
                "auto_included": list(self.auto_included_characters)
            }
            return False, f"Error creating chatbook: {str(e)}", dependency_info
        finally:
            # Single cleanup point for every exit path (success/cancel/error):
            # remove the temp work_dir and any leftover .partial archive, and
            # clear this thread's hooks so a reused instance never leaks them.
            self._cleanup_run(work_dir, partial_path)
            self._thread_local.progress_callback = None
            self._thread_local.cancel_check = None

    def _collect_conversations(
        self,
        conversation_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        auto_include_dependencies: bool
    ) -> None:
        """Collect conversations and their messages."""
        logger.info(f"ChatbookCreator._collect_conversations: Collecting {len(conversation_ids)} conversations")
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            logger.warning("ChatbookCreator._collect_conversations: ChaChaNotes database path not configured")
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        conversation_service = ChatConversationService(
            db,
            rag_context_store_path=get_user_data_dir() / "tldw_chatbook_chat_rag_context.json",
        )
        conv_dir = work_dir / "content" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChatbookCreator._collect_conversations: Created conversations directory at {conv_dir}")
        
        total = len(conversation_ids)
        for idx, conv_id in enumerate(conversation_ids):
            self._check_cancel()
            self._emit_progress("conversations", idx + 1, total)
            logger.debug(f"ChatbookCreator._collect_conversations: Processing conversation {conv_id}")
            try:
                # Get conversation details
                conv = db.get_conversation_by_id(conv_id)
                if not conv:
                    logger.warning(f"ChatbookCreator._collect_conversations: Conversation {conv_id} not found")
                    continue
                
                # Get DB messages and merge any sidecar RAG/citation context.
                db_messages = db.get_messages_for_conversation(conv_id)
                context_messages = conversation_service.get_messages_with_context(conv_id)
                messages = self._merge_message_context(db_messages, context_messages)
                logger.debug(f"ChatbookCreator._collect_conversations: Found {len(messages) if messages else 0} messages for conversation {conv_id}")
                
                # Create conversation data
                # Convert datetime to string if needed
                created_at = conv['created_at']
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                    
                last_modified = conv.get('last_modified', conv['created_at'])
                if hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()
                    
                exported_messages = []
                citation_messages = []
                for msg in messages:
                    timestamp = msg.get('timestamp') or msg.get('created_at') or datetime.now().isoformat()
                    if hasattr(timestamp, 'isoformat'):
                        timestamp = timestamp.isoformat()
                    message_id = msg.get('id')
                    message_data = {
                        "id": message_id,
                        "role": msg.get('role') or msg.get('sender'),
                        "content": msg['message'] if 'message' in msg else msg.get('content', ''),
                        "timestamp": timestamp,
                    }
                    citation_payload = self._message_citation_export_payload(msg)
                    if citation_payload:
                        message_data.update(citation_payload)
                        citation_messages.append(message_data)
                    exported_messages.append(message_data)

                title = conv.get('title', conv.get('conversation_name', 'Untitled'))
                citation_metadata: dict[str, Any] = {}
                if citation_messages:
                    citation_report_path, citation_report_metadata = self._write_conversation_citation_report(
                        conv_dir,
                        str(conv_id),
                        title,
                        citation_messages,
                    )
                    citation_metadata = self._conversation_citation_export_metadata(
                        citation_messages,
                        citation_report_path,
                    )
                    citation_metadata.update(citation_report_metadata)

                conv_data = {
                    "id": conv['id'],
                    "name": title,
                    "created_at": created_at,
                    "updated_at": last_modified,
                    "character_id": conv.get('character_id'),
                    "messages": exported_messages
                }
                
                # Write conversation file
                conv_file = self._conversation_export_path(conv_dir, str(conv_id), ".json")
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.conversations.append(conv_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=conv_id,
                    type=ContentType.CONVERSATION,
                    title=title,
                    created_at=datetime.fromisoformat(conv_data['created_at']),
                    updated_at=datetime.fromisoformat(conv_data['updated_at']),
                    metadata=citation_metadata,
                    file_path=f"content/conversations/{conv_file.name}"
                ))
                
                # Track character dependency if present
                if conv.get('character_id'):
                    self._add_character_dependency(conv['character_id'], manifest, content, work_dir, auto_include_dependencies)
                    
            except Exception as e:
                logger.error(f"Error collecting conversation {conv_id}: {e}")

    @staticmethod
    def _merge_message_context(
        db_messages: list[Mapping[str, Any]],
        context_messages: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        context_by_id = {str(message.get("id")): message for message in context_messages}
        merged_messages: list[dict[str, Any]] = []
        for message in db_messages:
            merged = dict(message)
            context = context_by_id.get(str(message.get("id")))
            if context:
                if context.get("rag_context") is not None:
                    merged["rag_context"] = context.get("rag_context")
                if context.get("citations") is not None:
                    merged["citations"] = context.get("citations")
            merged_messages.append(merged)
        return merged_messages

    @staticmethod
    def _message_citation_export_payload(message: Mapping[str, Any]) -> dict[str, Any]:
        """Return JSON-safe citation/evidence payloads attached to an exported message."""
        metadata = message.get("metadata") if isinstance(message.get("metadata"), Mapping) else {}
        rag_context = message.get("rag_context") if isinstance(message.get("rag_context"), Mapping) else {}
        payload: dict[str, Any] = {}
        for key in CITATION_MESSAGE_EXPORT_KEYS:
            value = message.get(key)
            if value is None:
                value = metadata.get(key)
            if value is None:
                value = rag_context.get(key)
            if not ChatbookCreator._has_exportable_citation_value(value):
                continue
            payload[key] = ChatbookCreator._json_safe_payload(value)
        return payload

    @staticmethod
    def _conversation_export_path(conv_dir: Path, conv_id: str, suffix: str) -> Path:
        filename = f"conversation_{ChatbookCreator._safe_conversation_file_id(conv_id)}{suffix}"
        validate_filename(filename)
        return conv_dir / filename

    @staticmethod
    def _safe_conversation_file_id(conv_id: str) -> str:
        candidate = sanitize_filename(str(conv_id)).strip().strip(".")
        if candidate:
            filename = f"conversation_{candidate}.json"
            try:
                validate_filename(filename)
                return candidate
            except ValueError:
                pass
        digest = hashlib.sha256(str(conv_id).encode("utf-8")).hexdigest()[:12]
        return f"id-{digest}"

    @staticmethod
    def _has_exportable_citation_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes, Mapping, list, tuple, set)):
            return bool(value)
        return True

    @staticmethod
    def _json_safe_payload(value: Any) -> Any:
        to_payload = getattr(value, "to_payload", None)
        if callable(to_payload):
            return ChatbookCreator._json_safe_payload(to_payload())

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return ChatbookCreator._json_safe_payload(model_dump(mode="json", exclude_none=True))

        if isinstance(value, Mapping):
            return {
                str(key): ChatbookCreator._json_safe_payload(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [ChatbookCreator._json_safe_payload(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _conversation_citation_export_metadata(
        citation_messages: list[dict[str, Any]],
        citation_report_path: str,
    ) -> dict[str, Any]:
        evidence_source_count = 0
        evidence_snippet_count = 0
        for message in citation_messages:
            evidence_bundle = message.get("evidence_bundle")
            if not isinstance(evidence_bundle, Mapping):
                continue
            references = evidence_bundle.get("references")
            if not isinstance(references, list):
                continue
            evidence_source_count += len(references)
            evidence_snippet_count += sum(
                1
                for reference in references
                if isinstance(reference, Mapping)
                and ChatbookCreator._citation_report_snippet(reference.get("snippet"))
            )

        return {
            "citation_report_path": citation_report_path,
            "citation_message_count": len(citation_messages),
            "evidence_source_count": evidence_source_count,
            "evidence_snippet_count": evidence_snippet_count,
        }

    def _write_conversation_citation_report(
        self,
        conv_dir: Path,
        conv_id: str,
        title: str,
        citation_messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Write a human-readable citation/source report for exported conversations."""
        report_file = self._conversation_export_path(conv_dir, conv_id, "_citations.md")
        report_metadata: dict[str, Any] = {}
        total_references_written = 0
        truncated = len(citation_messages) > MAX_CITATION_REPORT_MESSAGES
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Citations and Evidence: {self._markdown_report_text(title)}\n\n")
            for message in citation_messages[:MAX_CITATION_REPORT_MESSAGES]:
                references_remaining = MAX_CITATION_REPORT_TOTAL_REFERENCES - total_references_written
                if references_remaining <= 0:
                    truncated = True
                    break
                written, section_truncated = self._write_message_citation_report_section(
                    f,
                    message,
                    references_remaining=references_remaining,
                )
                total_references_written += written
                truncated = truncated or section_truncated
            if truncated:
                f.write(
                    "Report truncated: full citation and evidence payloads remain "
                    "available in the conversation JSON.\n"
                )
        if truncated:
            report_metadata["citation_report_truncated"] = True
        return f"content/conversations/{report_file.name}", report_metadata

    @staticmethod
    def _write_message_citation_report_section(
        handle: Any,
        message: Mapping[str, Any],
        *,
        references_remaining: int,
    ) -> tuple[int, bool]:
        message_id = ChatbookCreator._markdown_report_text(message.get("id", "unknown"))
        handle.write(f"## Message {message_id}\n\n")
        handle.write(f"Role: {ChatbookCreator._markdown_report_text(message.get('role', 'unknown'))}\n\n")

        validation = message.get("citation_validation")
        if isinstance(validation, Mapping):
            status = validation.get("status")
            if status is not None:
                handle.write(f"Citation status: {ChatbookCreator._markdown_report_text(status)}\n")
            cited_ids = validation.get("cited_evidence_ids")
            if isinstance(cited_ids, list) and cited_ids:
                handle.write(
                    "Cited evidence: "
                    f"{', '.join(ChatbookCreator._markdown_report_text(item) for item in cited_ids)}\n"
                )
            unknown_ids = validation.get("unknown_citation_ids")
            if isinstance(unknown_ids, list) and unknown_ids:
                handle.write(
                    "Unknown evidence: "
                    f"{', '.join(ChatbookCreator._markdown_report_text(item) for item in unknown_ids)}\n"
                )
            handle.write("\n")

        evidence_bundle = message.get("evidence_bundle")
        if not isinstance(evidence_bundle, Mapping):
            return 0, False

        bundle_id = evidence_bundle.get("bundle_id")
        query = evidence_bundle.get("query")
        source = evidence_bundle.get("source")
        if bundle_id is not None:
            handle.write(f"Evidence bundle: {ChatbookCreator._markdown_report_text(bundle_id)}\n")
        if source is not None:
            handle.write(f"Source: {ChatbookCreator._markdown_report_text(source)}\n")
        if query is not None:
            handle.write(f"Query: {ChatbookCreator._markdown_report_text(query)}\n")
        handle.write("\n")

        references = evidence_bundle.get("references")
        if not isinstance(references, list) or not references:
            return 0, False

        handle.write("### Sources\n\n")
        reference_limit = min(
            len(references),
            references_remaining,
            MAX_CITATION_REPORT_REFERENCES_PER_MESSAGE,
        )
        references_written = 0
        for reference in references[:reference_limit]:
            if not isinstance(reference, Mapping):
                continue
            evidence_id = ChatbookCreator._markdown_report_text(reference.get("evidence_id", "unknown"))
            title = ChatbookCreator._markdown_report_text(reference.get("title", "Untitled source"))
            source_type = ChatbookCreator._markdown_report_text(reference.get("source_type", "source"))
            authority = ChatbookCreator._markdown_report_text(reference.get("authority_label", "unknown authority"))
            status = ChatbookCreator._markdown_report_text(reference.get("status", "unknown"))
            handle.write(f"- {evidence_id} | {source_type} | {title} | {authority} | {status}\n")
            source_id = reference.get("source_id")
            if source_id is not None:
                handle.write(f"  - Source id: {ChatbookCreator._markdown_report_text(source_id)}\n")
            snippet = ChatbookCreator._citation_report_snippet(reference.get("snippet"))
            if snippet:
                handle.write(f"  - Snippet: {snippet}\n")
            references_written += 1
        handle.write("\n")
        section_truncated = len(references) > reference_limit
        return references_written, section_truncated

    @staticmethod
    def _citation_report_snippet(value: Any) -> str:
        raw = "" if value is None else str(value)
        text = " ".join(raw.split())
        if len(text) > MAX_CITATION_REPORT_SNIPPET_CHARS:
            text = text[: MAX_CITATION_REPORT_SNIPPET_CHARS - 3].rstrip() + "..."
        return ChatbookCreator._markdown_report_text(text, max_length=MAX_CITATION_REPORT_SNIPPET_CHARS)

    @staticmethod
    def _markdown_report_text(value: Any, *, max_length: int = MAX_CITATION_REPORT_SNIPPET_CHARS) -> str:
        raw = "" if value is None else str(value)
        text = " ".join(sanitize_string(raw, max_length=max_length).split())
        return html.escape(text, quote=False).replace("|", "\\|")
    
    def _collect_notes(
        self,
        note_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect notes and export as markdown."""
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        notes_dir = work_dir / "content" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(note_ids)
        for idx, note_id in enumerate(note_ids):
            self._check_cancel()
            self._emit_progress("notes", idx + 1, total)
            try:
                # Get note details
                note = db.get_note_by_id(note_id)
                if not note:
                    continue
                
                # Create note metadata
                # Convert datetime to string if needed
                created_at = note['created_at']
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                    
                last_modified = note.get('last_modified', note['created_at'])
                if hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()
                    
                note_data = {
                    "id": note['id'],
                    "title": note['title'],
                    "content": note['content'],
                    "created_at": created_at,
                    "updated_at": last_modified,
                    "tags": note.get('keywords', '').split(',') if note.get('keywords') else []
                }
                
                # Write markdown file
                note_file = notes_dir / f"{sanitize_filename(note['title'])}.md"
                with open(note_file, 'w', encoding='utf-8') as f:
                    # Write frontmatter
                    f.write("---\n")
                    f.write(f"id: {note['id']}\n")
                    f.write(f"title: {note['title']}\n")
                    f.write(f"created_at: {note_data['created_at']}\n")
                    f.write(f"updated_at: {note_data['updated_at']}\n")
                    if note_data['tags']:
                        f.write(f"tags: {', '.join(note_data['tags'])}\n")
                    f.write("---\n\n")
                    
                    # Write content
                    f.write(note['content'])
                
                # Add to content
                content.notes.append(note_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=note_id,
                    type=ContentType.NOTE,
                    title=note['title'],
                    created_at=datetime.fromisoformat(note_data['created_at']),
                    updated_at=datetime.fromisoformat(note_data['updated_at']),
                    tags=note_data['tags'],
                    file_path=f"content/notes/{note_file.name}"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting note {note_id}: {e}")
    
    def _collect_characters(
        self,
        character_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect character cards."""
        db_path = self.db_paths.get("ChaChaNotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        chars_dir = work_dir / "content" / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(character_ids)
        for idx, char_id in enumerate(character_ids):
            self._check_cancel()
            self._emit_progress("characters", idx + 1, total)
            try:
                # Get character card (which includes all details)
                char = db.get_character_card_by_id(int(char_id))
                if not char:
                    continue
                
                # Create character data
                char_data = {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "personality": char.get('personality', ''),
                    "created_at": datetime.now().isoformat(),  # Characters don't have timestamps in DB
                    "updated_at": datetime.now().isoformat(),
                    "avatar_path": char.get('avatar_path'),
                    "card": char  # The full character card data
                }
                
                # Write character file (remove 'card' field which may have non-serializable objects)
                char_file = chars_dir / f"character_{char_id}.json"
                char_data_for_json = {k: v for k, v in char_data.items() if k != 'card'}
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data_for_json, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters.append(char_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=char_id,
                    type=ContentType.CHARACTER,
                    title=char['name'],
                    description=char.get('description'),
                    created_at=datetime.fromisoformat(char_data['created_at']),
                    updated_at=datetime.fromisoformat(char_data['updated_at']),
                    file_path=f"content/characters/character_{char_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting character {char_id}: {e}")
    
    def _collect_media(
        self,
        media_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        quality: str
    ) -> None:
        """Collect media items and their files."""
        db_path = self.db_paths.get("Media")
        if not db_path:
            logger.warning("Media database path not configured")
            return
            
        db = MediaDatabase(db_path, "chatbook_creator")
        media_dir = work_dir / "content" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        metadata_dir = media_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        total = len(media_ids)
        for idx, media_id in enumerate(media_ids):
            self._check_cancel()
            self._emit_progress("media", idx + 1, total)
            try:
                # Get media details from database
                media_item = db.get_media_by_id(int(media_id))
                if not media_item:
                    logger.warning(f"Media item not found: {media_id}")
                    continue
                
                # Create media data structure.
                # NOTE: the Media table's real columns are ``type``,
                # ``ingestion_date``, and ``last_modified`` -- there is no
                # ``media_type``/``created_at``/``updated_at`` column, so
                # reading those keys off ``media_item`` always resolved to
                # ``None`` and silently dropped the media type and both
                # timestamps from every export.
                media_data = {
                    "id": media_item['id'],
                    "title": media_item.get('title', 'Untitled'),
                    "media_type": media_item.get('type'),
                    "url": media_item.get('url'),
                    "author": media_item.get('author'),
                    "content": media_item.get('content', ''),
                    "created_at": media_item.get('ingestion_date'),
                    "updated_at": media_item.get('last_modified'),
                    "metadata": {
                        "ingestion_date": media_item.get('ingestion_date'),
                        "media_keywords": media_item.get('media_keywords'),
                        "prompt": media_item.get('prompt'),
                        "summary": media_item.get('summary'),
                        "transcription_model": media_item.get('transcription_model')
                    }
                }
                
                # Handle media file if it exists
                # Note: The current MediaDatabase doesn't store actual file paths,
                # so we'll store the textual content and metadata
                media_filename = f"media_{media_id}"
                
                # Save media metadata. ``media_item`` comes straight from
                # ``MediaDatabase.get_media_by_id`` -- ``DATETIME``-typed
                # columns (e.g. ``ingestion_date``) round-trip through
                # sqlite3's registered converter as real ``datetime``
                # objects (see ``DB/sqlite_datetime_fix.py``), which plain
                # ``json.dump`` cannot serialize. ``default=str`` renders
                # any such value as its ISO string instead of raising --
                # this previously failed silently (caught by this method's
                # own broad ``except Exception`` below), dropping every
                # media item from the export with no surfaced error.
                metadata_file = metadata_dir / f"{media_filename}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(media_data, f, indent=2, ensure_ascii=False, default=str)
                
                # If media has content (transcription, text, etc.), save it
                if media_item.get('content'):
                    content_file = media_dir / f"{media_filename}.txt"
                    with open(content_file, 'w', encoding='utf-8') as f:
                        f.write(media_item['content'])
                
                # Add to content
                content.media_items.append(media_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=media_id,
                    type=ContentType.MEDIA,
                    title=media_item.get('title', 'Untitled'),
                    description=media_item.get('summary'),
                    created_at=_coerce_media_timestamp(media_item.get('ingestion_date')) or datetime.now(),
                    updated_at=_coerce_media_timestamp(media_item.get('last_modified')) or datetime.now(),
                    metadata={
                        "media_type": media_item.get('type'),
                        "quality": quality,
                        "has_content": bool(media_item.get('content'))
                    },
                    file_path=f"content/media/metadata/{media_filename}.json"
                ))
                
                logger.info(f"Collected media item: {media_item.get('title', 'Untitled')}")
                
            except Exception as e:
                logger.error(f"Error collecting media {media_id}: {e}")
    
    def _collect_prompts(
        self,
        prompt_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect prompts."""
        db_path = self.db_paths.get("Prompts")
        if not db_path:
            return
            
        db = PromptsDatabase(db_path, "chatbook_creator")
        prompts_dir = work_dir / "content" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(prompt_ids)
        for idx, prompt_id in enumerate(prompt_ids):
            self._check_cancel()
            self._emit_progress("prompts", idx + 1, total)
            try:
                # Get prompt details
                prompt = db.get_prompt_by_id(int(prompt_id))
                if not prompt:
                    continue
                
                # Create prompt data
                prompt_data = {
                    "id": prompt['id'],
                    "name": prompt['name'],
                    "description": prompt.get('details', ''),
                    "content": prompt.get('system_prompt', '') or prompt.get('user_prompt', ''),
                    "created_at": prompt.get('created_at', datetime.now().isoformat()),
                    "updated_at": prompt.get('updated_at', datetime.now().isoformat())
                }
                
                # Write prompt file
                prompt_file = prompts_dir / f"prompt_{prompt_id}.json"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.prompts.append(prompt_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=prompt_id,
                    type=ContentType.PROMPT,
                    title=prompt['name'],
                    description=prompt.get('description'),
                    created_at=datetime.fromisoformat(prompt['created_at']),
                    updated_at=datetime.fromisoformat(prompt['updated_at']),
                    file_path=f"content/prompts/prompt_{prompt_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting prompt {prompt_id}: {e}")
    
    def _add_character_dependency(self, character_id: int, manifest: ChatbookManifest, 
                                  content: ChatbookContent, work_dir: Path, 
                                  auto_include: bool) -> None:
        """Add a character as a dependency if not already included."""
        # Check if character already in manifest
        for item in manifest.content_items:
            if item.type == ContentType.CHARACTER and item.id == str(character_id):
                return
        
        # Check if character is explicitly selected
        if str(character_id) in self._selected_characters:
            return  # Will be collected later in _collect_characters
        
        # Track missing dependency
        self.missing_dependencies.add(character_id)
        
        if auto_include:
            # Auto-include the character
            db_path = self.db_paths.get("ChaChaNotes")
            if not db_path:
                logger.error(f"Cannot auto-include character {character_id}: Database path not configured")
                return
                
            db = CharactersRAGDB(db_path, "chatbook_creator")
            
            try:
                # Get character card (which includes all details)
                char = db.get_character_card_by_id(character_id)
                if not char:
                    logger.error(f"Character {character_id} not found in database")
                    return
                
                # Create character data
                char_data = {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "personality": char.get('personality', ''),
                    "created_at": datetime.now().isoformat(),  # Characters don't have timestamps in DB
                    "updated_at": datetime.now().isoformat(),
                    "avatar_path": char.get('avatar_path'),
                    "card": char  # The full character card data
                }
                
                # Create characters directory if it doesn't exist
                chars_dir = work_dir / "content" / "characters"
                chars_dir.mkdir(parents=True, exist_ok=True)
                
                # Write character file (remove 'card' field which may have non-serializable objects)
                char_file = chars_dir / f"character_{character_id}.json"
                char_data_for_json = {k: v for k, v in char_data.items() if k != 'card'}
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data_for_json, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters.append(char_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=str(character_id),
                    type=ContentType.CHARACTER,
                    title=char['name'],
                    description=char.get('description'),
                    created_at=datetime.fromisoformat(char_data['created_at']),
                    updated_at=datetime.fromisoformat(char_data['updated_at']),
                    file_path=f"content/characters/character_{character_id}.json",
                    metadata={"auto_included": True}
                ))
                
                # Track auto-included character
                self.auto_included_characters.add(character_id)
                self.missing_dependencies.remove(character_id)
                
                logger.info(f"Auto-included character dependency: {char['name']} (ID: {character_id})")
                
            except Exception as e:
                logger.error(f"Error auto-including character {character_id}: {e}")
        else:
            logger.warning(f"Character {character_id} is referenced but not included in chatbook")
    
    def _discover_relationships(self, manifest: ChatbookManifest, content: ChatbookContent) -> None:
        """Discover relationships between content items."""
        self._check_cancel()
        self._emit_progress("relationships", 1, 1)
        # Find conversation-character relationships
        for conv in content.conversations:
            if conv.get('character_id'):
                # Find if character is in manifest
                char_id = str(conv['character_id'])
                for item in manifest.content_items:
                    if item.type == ContentType.CHARACTER and item.id == char_id:
                        manifest.relationships.append(Relationship(
                            source_id=str(conv['id']),
                            target_id=char_id,
                            relationship_type="uses_character"
                        ))
                        break
        
        # TODO: Add more relationship discovery logic
        # - Notes mentioning conversations
        # - Media used in conversations
        # - etc.
    
    def _create_readme(self, work_dir: Path, manifest: ChatbookManifest) -> None:
        """Create a README file for the chatbook."""
        readme_path = work_dir / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {manifest.name}\n\n")
            f.write(f"{manifest.description}\n\n")
            
            if manifest.author:
                f.write(f"**Author:** {manifest.author}\n\n")
            
            f.write(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Contents\n\n")
            
            if manifest.total_conversations > 0:
                f.write(f"- **Conversations:** {manifest.total_conversations}\n")
            if manifest.total_notes > 0:
                f.write(f"- **Notes:** {manifest.total_notes}\n")
            if manifest.total_characters > 0:
                f.write(f"- **Characters:** {manifest.total_characters}\n")
            if manifest.total_media_items > 0:
                f.write(f"- **Media Items:** {manifest.total_media_items}\n")
            if manifest.total_prompts > 0:
                f.write(f"- **Prompts:** {manifest.total_prompts}\n")
            
            if manifest.tags:
                f.write(f"\n## Tags\n\n")
                f.write(", ".join(manifest.tags))
                f.write("\n")
            
            f.write("\n## Structure\n\n")
            f.write("```\n")
            f.write("chatbook/\n")
            f.write("├── manifest.json     # Chatbook metadata\n")
            f.write("├── README.md        # This file\n")
            f.write("└── content/         # Content files\n")
            if manifest.total_conversations > 0:
                f.write("    ├── conversations/   # Chat conversations\n")
            if manifest.total_notes > 0:
                f.write("    ├── notes/          # Markdown notes\n")
            if manifest.total_characters > 0:
                f.write("    ├── characters/     # Character cards\n")
            if manifest.total_media_items > 0:
                f.write("    ├── media/          # Media files and content\n")
                f.write("    │   └── metadata/   # Media metadata JSON files\n")
            if manifest.total_prompts > 0:
                f.write("    ├── prompts/        # Prompts\n")
            f.write("```\n")
            
            f.write("\n## License\n\n")
            if manifest.license:
                f.write(manifest.license)
            else:
                f.write("See individual content files for licensing information.")
    
    def _create_zip_archive(self, work_dir: Path, output_path: Path, partial_path: Path) -> None:
        """Zip work_dir into a sibling .partial, then atomically replace output_path."""
        files = [p for p in work_dir.rglob('*') if p.is_file()]
        total = len(files)
        with zipfile.ZipFile(partial_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, file_path in enumerate(files):
                self._check_cancel()
                arcname = file_path.relative_to(work_dir)
                zf.write(file_path, arcname)
                self._emit_progress("packaging", idx + 1, total)
        os.replace(partial_path, output_path)
