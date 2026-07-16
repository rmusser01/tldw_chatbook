import base64
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger as _logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

logger = _logger.bind(module="ChatPersistenceService")


class ChatPersistenceService:
    def __init__(self, db: CharactersRAGDB, workspace_registry: Any | None = None):
        self.db = db
        self.workspace_registry = workspace_registry

    @staticmethod
    def derive_conversation_title(
        *,
        character_name: Optional[str] = None,
        assistant_kind: Optional[str] = None,
        assistant_id: Optional[str] = None,
        explicit_title: Optional[str] = None,
    ) -> str:
        if explicit_title:
            return explicit_title
        if character_name:
            return f"Chat with {character_name}"

        normalized_kind = (assistant_kind or "").strip().lower() or None
        normalized_id = (assistant_id or "").strip() or None

        if normalized_kind == "persona":
            return f"Chat with {normalized_id}" if normalized_id else "Chat with Persona"
        if normalized_kind == "character":
            return f"Chat with {normalized_id}" if normalized_id else "Chat with Character"
        return "New Chat"

    def create_conversation(
        self,
        *,
        character_id: Optional[int] = None,
        character_name: Optional[str] = None,
        assistant_kind: Optional[str] = None,
        assistant_id: Optional[str] = None,
        persona_memory_mode: Optional[str] = None,
        runtime_backend: Optional[str] = None,
        discovery_owner: Optional[str] = None,
        discovery_entity_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        conversation_title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        safe_workspace_id = self._require_workspace_scope(
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        title = self.derive_conversation_title(
            character_name=character_name,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
            explicit_title=conversation_title,
        )
        conversation_id = self.db.add_conversation({
            "character_id": character_id,
            "assistant_kind": assistant_kind,
            "assistant_id": assistant_id,
            "persona_memory_mode": persona_memory_mode,
            "runtime_backend": runtime_backend,
            "discovery_owner": discovery_owner,
            "discovery_entity_id": discovery_entity_id,
            "scope_type": scope_type,
            "workspace_id": safe_workspace_id if safe_workspace_id is not None else workspace_id,
            "title": title,
            "system_prompt": system_prompt,
            "client_id": self.db.client_id,
        })
        if safe_workspace_id is not None:
            try:
                self._link_workspace_conversation(
                    workspace_id=safe_workspace_id,
                    conversation_id=conversation_id,
                    title=title,
                )
            except Exception:
                self._discard_created_conversation(conversation_id)
                raise
        return conversation_id

    def fork_conversation_into_workspace(
        self,
        *,
        conversation_id: str,
        target_workspace_id: str,
    ) -> Any:
        """Record a workspace conversation link without mutating global history.

        Args:
            conversation_id: Existing conversation id to expose in the target
                workspace context.
            target_workspace_id: Workspace id that should receive the
                conversation membership link.

        Returns:
            The workspace membership returned by the registry service.

        Raises:
            ValueError: If the conversation or target workspace cannot be
                resolved.
            Exception: Propagates registry storage failures from the workspace
                membership link operation.
        """

        conversation = self.db.get_conversation_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        safe_workspace_id = self._require_workspace_scope(
            scope_type="workspace",
            workspace_id=target_workspace_id,
        )
        if safe_workspace_id is None:
            raise ValueError("Failed to resolve a valid workspace ID")
        title = str(conversation.get("title") or "Workspace conversation")
        return self._link_workspace_conversation(
            workspace_id=safe_workspace_id,
            conversation_id=conversation_id,
            title=title,
        )

    def _require_workspace_scope(
        self,
        *,
        scope_type: Optional[str],
        workspace_id: Optional[str],
    ) -> Optional[str]:
        normalized_scope = (scope_type or "").strip().lower()
        if normalized_scope != "workspace":
            return None
        safe_workspace_id = (workspace_id or "").strip()
        if not safe_workspace_id:
            raise ValueError("Workspace conversation requires a workspace_id")
        if self.workspace_registry is None:
            raise ValueError("Workspace registry is required for workspace conversations")
        workspace = self.workspace_registry.get_workspace(safe_workspace_id)
        if workspace is None:
            raise ValueError(f"Unknown workspace: {safe_workspace_id}")
        return safe_workspace_id

    def _link_workspace_conversation(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        title: str,
    ) -> Any:
        return self.workspace_registry.link_membership(
            workspace_id,
            item_type="conversation",
            item_id=conversation_id,
            role="workspace-thread",
            title=title,
        )

    def _discard_created_conversation(self, conversation_id: str) -> None:
        conversation = self.db.get_conversation_by_id(
            conversation_id,
            include_deleted=True,
        )
        if conversation is None or conversation.get("deleted"):
            return
        try:
            expected_version = int(conversation["version"])
            self.db.soft_delete_conversation(
                conversation_id,
                expected_version=expected_version,
            )
        except Exception:
            logger.bind(conversation_id=conversation_id).opt(exception=True).error(
                "Failed to soft-delete workspace conversation after membership link failure",
            )
            raise

    def update_conversation_system_prompt(
        self,
        *,
        conversation_id: str,
        system_prompt: Optional[str],
    ) -> bool:
        """Update the persisted system prompt for an existing conversation.

        Args:
            conversation_id: UUID of the conversation to update.
            system_prompt: New system prompt text, or ``None``/blank to clear it.

        Returns:
            True if the update was applied.

        Raises:
            ValueError: If the conversation cannot be found.
        """
        current_conversation = self.db.get_conversation_by_id(conversation_id)
        if not current_conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        return bool(
            self.db.update_conversation(
                conversation_id,
                {"system_prompt": system_prompt},
                expected_version=current_conversation["version"],
            )
        )

    def update_message_content(
        self,
        *,
        message_id: str,
        content: str,
        image_data: Optional[bytes],
        image_mime_type: Optional[str],
        parent_message_id: Optional[str] = None,
        feedback: Optional[str] = None,
        update_parent: bool = False,
        update_feedback: bool = False,
        attachments: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> bool:
        """Update a message's content, optionally its parent/feedback, and its images.

        Two mutually exclusive contracts govern how images are updated:

        Legacy (``attachments=None``): ``image_data``/``image_mime_type`` are
        the sole source of the message's single legacy image. Passing
        ``image_data=None`` leaves any already-persisted image untouched
        (it does NOT clear it) -- callers may pass ``None`` simply because
        in-memory bytes were never rehydrated. Passing non-``None``
        ``image_data`` replaces the persisted image.

        Split addressing (``attachments`` is a sequence): an authoritative,
        full rewrite of every position. Position 0 (if present) replaces the
        legacy ``image_data``/``image_mime_type`` columns -- even when its
        ``data``/``mime_type`` are ``None``, since supplying ``attachments``
        at all means the caller intends to overwrite. Positions >= 1 replace
        the ``message_attachments`` table rows via
        ``CharactersRAGDB.set_message_attachments`` (an empty list clears any
        stale rows). The row update and the table rewrite happen inside one
        transaction so a table-write failure rolls back the row update too;
        conversely, if the row update itself does not succeed (returns a
        falsy result without raising), the table write is skipped entirely
        so attachments never drift out of sync with unrevised content.
        ``image_data``/``image_mime_type`` are ignored when ``attachments``
        is supplied.

        Args:
            message_id: UUID of the message to update.
            content: New message text content.
            image_data: Legacy single-image bytes; ignored when
                ``attachments`` is supplied. See the legacy contract above
                for how ``None`` is handled.
            image_mime_type: Legacy single-image MIME type; ignored when
                ``attachments`` is supplied.
            parent_message_id: New parent message id, applied only when
                ``update_parent`` is True.
            feedback: New feedback value, applied only when
                ``update_feedback`` is True.
            update_parent: Whether to update ``parent_message_id``.
            update_feedback: Whether to update ``feedback``.
            attachments: Optional full 0..N-1 position list of attachment
                rows (each a mapping with ``position``, ``data``,
                ``mime_type``, and optional ``display_name``). When
                supplied, this is the sole, authoritative source for both
                the legacy image columns (position 0) and the
                ``message_attachments`` table (positions >= 1). ``None``
                leaves all attachments/images untouched by this call except
                via the legacy ``image_data``/``image_mime_type`` kwargs.

        Returns:
            True if the row update was applied; False if the underlying
            update reported failure without raising (attachments are left
            untouched in that case).

        Raises:
            ValueError: If the message cannot be found.
            ConflictError: If the message was concurrently modified or
                soft-deleted (optimistic-lock version mismatch).
        """
        current_message = self.db.get_message_by_id(message_id)
        if not current_message:
            raise ValueError(f"Message {message_id} not found")

        update_data: Dict[str, Any] = {"content": content}
        # Only include the image columns when new image bytes are supplied.
        # ``ChaChaNotes_DB.update_message`` treats an *included* ``image_data``
        # key of ``None`` as an explicit request to NULL both image columns,
        # but omitting the key entirely leaves any persisted image untouched.
        # Callers here (e.g. the Console store) may pass ``image_data=None``
        # simply because in-memory bytes were never rehydrated -- that must
        # not wipe an image that already exists in the database.
        #
        # ``attachments`` (split addressing): ``None`` means "leave
        # attachments untouched" -- the byte-identical #621/#628-era behavior
        # above. When a caller passes a full 0..N-1 position list, position 0
        # replaces the legacy image columns (even when ``None``, since an
        # explicit attachments list is an authoritative rewrite) and
        # positions >= 1 replace the ``message_attachments`` table rows.
        if attachments is not None:
            position_zero = next(
                (row for row in attachments if int(row["position"]) == 0), None
            )
            update_data["image_data"] = position_zero["data"] if position_zero else None
            update_data["image_mime_type"] = (
                position_zero["mime_type"] if position_zero else None
            )
        elif image_data is not None:
            update_data["image_data"] = image_data
            update_data["image_mime_type"] = image_mime_type
        if update_parent:
            update_data["parent_message_id"] = parent_message_id
        if update_feedback:
            update_data["feedback"] = feedback

        if attachments is not None:
            extra_rows = [
                {
                    "position": int(row["position"]),
                    "data": row["data"],
                    "mime_type": row["mime_type"],
                    "display_name": row.get("display_name", ""),
                }
                for row in attachments
                if int(row["position"]) >= 1
            ]
            # One atomic unit: inside this outer transaction the nested
            # update_message/set_message_attachments transactions are no-ops,
            # so a failed table write rolls back the row update (content and
            # legacy image columns) too. Conversely, if the row update itself
            # fails without raising (e.g. an optimistic-lock miss reported as
            # a plain ``False`` return instead of a ``ConflictError``), the
            # attachments table write must be skipped -- otherwise
            # attachments would be rewritten while content/version were not,
            # leaving the two out of sync.
            with self.db.transaction():
                result = bool(
                    self.db.update_message(
                        message_id,
                        update_data,
                        expected_version=current_message["version"],
                    )
                )
                if result:
                    self.db.set_message_attachments(message_id, extra_rows)
            return result

        return bool(
            self.db.update_message(
                message_id,
                update_data,
                expected_version=current_message["version"],
            )
        )

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        message_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        feedback: Optional[str] = None,
        attachments: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        """Create a new message, optionally with a legacy image or a full attachment list.

        Two mutually exclusive contracts govern how images are stored,
        mirroring ``update_message_content``:

        Legacy (``attachments=None``): ``image_data``/``image_mime_type``
        are the sole source of the message's single legacy image, stored
        directly on the ``messages`` row. The ``message_attachments`` table
        is never touched.

        Split addressing (``attachments`` is a sequence): an authoritative
        full 0..N-1 position list. Position 0 (if present) overrides the
        scalar ``image_data``/``image_mime_type`` kwargs -- even overriding
        them with ``None`` when no position-0 entry is present, since
        supplying ``attachments`` at all means it is authoritative. Positions
        >= 1 are written to the ``message_attachments`` table via
        ``CharactersRAGDB.set_message_attachments``, always -- even an empty
        list, so any stale rows a prior attempt at this same ``message_id``
        left behind are cleared. The row insert and the table write happen
        inside one transaction, so a table-write failure rolls back the row
        insert too.

        Args:
            conversation_id: UUID of the parent conversation.
            sender: Message sender/role (e.g. ``"user"``, ``"assistant"``).
            content: Message text content.
            image_data: Legacy single-image bytes; ignored when
                ``attachments`` is supplied.
            image_mime_type: Legacy single-image MIME type; ignored when
                ``attachments`` is supplied.
            message_id: Optional explicit message id; the DB generates one
                when omitted.
            parent_message_id: Optional parent message id for threading.
            feedback: Optional feedback value applied via a follow-up update
                once the message exists (feedback is not part of the initial
                insert payload).
            attachments: Optional full 0..N-1 position list of attachment
                rows (each a mapping with ``position``, ``data``,
                ``mime_type``, and optional ``display_name``). When
                supplied, this is the sole, authoritative source for both
                the legacy image columns (position 0) and the
                ``message_attachments`` table (positions >= 1).

        Returns:
            The newly created message's id.

        Raises:
            CharactersRAGDBError: For database integrity errors during the
                insert or the attachment-table write.
        """
        # Split addressing: when ``attachments`` is supplied it covers ALL
        # positions (0..N-1) and is authoritative -- position 0 overrides the
        # scalar ``image_data``/``image_mime_type`` kwargs (even overriding
        # them with ``None`` when no position-0 entry is present), and
        # positions >= 1 land in the ``message_attachments`` table via
        # ``set_message_attachments``. ``attachments=None`` leaves the
        # scalar kwargs as the sole source of the legacy image columns and
        # never touches the attachments table -- byte-identical to the
        # pre-split behavior.
        effective_image_data = image_data
        effective_image_mime_type = image_mime_type
        extra_rows: List[Dict[str, Any]] = []
        if attachments is not None:
            position_zero = next(
                (row for row in attachments if int(row["position"]) == 0), None
            )
            effective_image_data = position_zero["data"] if position_zero else None
            effective_image_mime_type = (
                position_zero["mime_type"] if position_zero else None
            )
            extra_rows = [
                {
                    "position": int(row["position"]),
                    "data": row["data"],
                    "mime_type": row["mime_type"],
                    "display_name": row.get("display_name", ""),
                }
                for row in attachments
                if int(row["position"]) >= 1
            ]

        message_payload = {
            "id": message_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "sender": sender,
            "content": content,
            "image_data": effective_image_data,
            "image_mime_type": effective_image_mime_type,
            "client_id": self.db.client_id,
        }
        if attachments is not None:
            # One atomic unit: inside this outer transaction the nested
            # add_message/set_message_attachments transactions are no-ops, so
            # a failed table write rolls the message row back too. The table
            # write always runs -- an empty list still clears any stale rows
            # a prior attempt at this same message_id may have left behind.
            with self.db.transaction():
                created_message_id = self.db.add_message(message_payload)
                self.db.set_message_attachments(created_message_id, extra_rows)
        else:
            created_message_id = self.db.add_message(message_payload)
        if feedback is not None:
            created_message = self.db.get_message_by_id(created_message_id)
            self.db.update_message(
                created_message_id,
                {"feedback": feedback},
                expected_version=created_message["version"],
            )
        return created_message_id

    def get_attachments_for_messages(
        self, message_ids: Sequence[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch-fetch extra (position >= 1) attachments for messages.

        Passthrough to ``CharactersRAGDB.get_attachments_for_messages``.
        Legacy position-0 images are not included here -- they live on the
        ``messages`` row itself (``image_data``/``image_mime_type``).

        Args:
            message_ids: Message ids to fetch attachment rows for.

        Returns:
            A mapping of message id to its list of attachment row dicts
            (each with ``position``, ``data``, ``mime_type``,
            ``display_name``), ordered by position. Message ids with no
            extra (position >= 1) attachments are omitted from the result.
        """
        return self.db.get_attachments_for_messages(message_ids)

    def save_history(
        self,
        *,
        conversation_id: str,
        chatbot_history: List[Dict[str, Any]],
    ) -> int:
        existing_messages = self.db.get_messages_for_conversation(
            conversation_id,
            limit=10000,
            order_by_timestamp="ASC",
        )
        existing_by_id = {message["id"]: message for message in existing_messages}
        existing_by_position = [
            message for message in existing_messages
            if message.get("variant_of") is None
        ]
        consumed_existing_ids = set()

        saved_count = 0
        fallback_index = 0

        for message_obj in chatbot_history:
            sender = message_obj.get("role")
            if not sender or sender == "system":
                continue

            content, image_data, image_mime_type = self._extract_message_payload(message_obj)
            if not content and not image_data:
                continue

            message_id = message_obj.get("id")
            parent_message_id = message_obj.get("parent_message_id")
            feedback = message_obj.get("feedback")

            if message_id and message_id in existing_by_id:
                self.update_message_content(
                    message_id=message_id,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                    update_parent="parent_message_id" in message_obj,
                    update_feedback="feedback" in message_obj,
                )
                consumed_existing_ids.add(message_id)
            elif message_id:
                self.create_message(
                    conversation_id=conversation_id,
                    sender=sender,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    message_id=message_id,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                )
            else:
                while (
                    fallback_index < len(existing_by_position)
                    and existing_by_position[fallback_index]["id"] in consumed_existing_ids
                ):
                    fallback_index += 1

            if not message_id and fallback_index < len(existing_by_position):
                existing_message = existing_by_position[fallback_index]
                self.update_message_content(
                    message_id=existing_message["id"],
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                    update_parent="parent_message_id" in message_obj,
                    update_feedback="feedback" in message_obj,
                )
                consumed_existing_ids.add(existing_message["id"])
                fallback_index += 1
            elif not message_id:
                self.create_message(
                    conversation_id=conversation_id,
                    sender=sender,
                    content=content,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    parent_message_id=parent_message_id,
                    feedback=feedback,
                )

            saved_count += 1

        retained_existing_ids = set(consumed_existing_ids)
        variants_added = True
        while variants_added:
            variants_added = False
            for existing_message in existing_messages:
                if existing_message["id"] in retained_existing_ids:
                    continue
                if existing_message.get("variant_of") in retained_existing_ids:
                    retained_existing_ids.add(existing_message["id"])
                    variants_added = True

        for existing_message in existing_messages:
            if existing_message["id"] not in retained_existing_ids:
                self.db.soft_delete_message(
                    existing_message["id"],
                    existing_message["version"],
                )

        return saved_count

    @staticmethod
    def _extract_message_payload(message_obj: Dict[str, Any]) -> Tuple[str, Optional[bytes], Optional[str]]:
        text_content_parts: List[str] = []
        image_data_bytes: Optional[bytes] = None
        image_mime_type_str: Optional[str] = None
        content_data = message_obj.get("content")

        if isinstance(content_data, str):
            text_content_parts.append(content_data)
        elif isinstance(content_data, list):
            for part in content_data:
                part_type = part.get("type")
                if part_type == "text":
                    text_content_parts.append(part.get("text", ""))
                elif part_type == "image_url":
                    image_url_dict = part.get("image_url", {})
                    url_str = image_url_dict.get("url", "")
                    if url_str.startswith("data:") and ";base64," in url_str:
                        try:
                            header, b64_data = url_str.split(";base64,", 1)
                            image_mime_type_str = header.split("data:", 1)[1] if "data:" in header else None
                            if image_mime_type_str:
                                image_data_bytes = base64.b64decode(b64_data)
                            else:
                                text_content_parts.append("<Error: Malformed image data URI in history>")
                        except Exception:
                            image_data_bytes = None
                            image_mime_type_str = None
                            text_content_parts.append("<Error: Failed to decode image data from history>")
                    elif url_str:
                        text_content_parts.append(f"<Image URL: {url_str}>")
        elif content_data is not None:
            text_content_parts.append(f"<Unsupported content type: {type(content_data)}>")

        return "\n".join(text_content_parts).strip(), image_data_bytes, image_mime_type_str
