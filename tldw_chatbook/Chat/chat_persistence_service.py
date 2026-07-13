import base64
from typing import Any, Dict, List, Optional, Tuple

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
    ) -> bool:
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
        if image_data is not None:
            update_data["image_data"] = image_data
            update_data["image_mime_type"] = image_mime_type
        if update_parent:
            update_data["parent_message_id"] = parent_message_id
        if update_feedback:
            update_data["feedback"] = feedback

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
        image_data: Optional[bytes],
        image_mime_type: Optional[str],
        message_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        created_message_id = self.db.add_message({
            "id": message_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "client_id": self.db.client_id,
        })
        if feedback is not None:
            created_message = self.db.get_message_by_id(created_message_id)
            self.db.update_message(
                created_message_id,
                {"feedback": feedback},
                expected_version=created_message["version"],
            )
        return created_message_id

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
