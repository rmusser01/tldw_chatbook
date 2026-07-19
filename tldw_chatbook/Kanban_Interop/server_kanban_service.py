"""Policy-gated active-server Kanban service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    KanbanBoardCreate,
    KanbanBoardExportRequest,
    KanbanBoardImportRequest,
    KanbanBoardUpdate,
    KanbanBulkArchiveCardsRequest,
    KanbanBulkCardLinksRequest,
    KanbanBulkDeleteCardsRequest,
    KanbanBulkLabelCardsRequest,
    KanbanBulkMoveCardsRequest,
    KanbanCardCopyRequest,
    KanbanCardCopyWithChecklistsRequest,
    KanbanCardCreate,
    KanbanCardLinkCreate,
    KanbanCardMoveRequest,
    KanbanCardSearchRequest,
    KanbanCardUpdate,
    KanbanChecklistCreate,
    KanbanChecklistItemCreate,
    KanbanChecklistItemReorderRequest,
    KanbanChecklistItemUpdate,
    KanbanChecklistReorderRequest,
    KanbanChecklistUpdate,
    KanbanCommentCreate,
    KanbanCommentUpdate,
    KanbanLabelCreate,
    KanbanLabelUpdate,
    KanbanListCreate,
    KanbanListUpdate,
    KanbanReorderRequest,
    KanbanSearchRequest,
    KanbanToggleAllChecklistItemsRequest,
)
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


@dataclass(frozen=True, slots=True)
class KanbanOperationSpec:
    action_id: str
    client_method: str
    kind: str | None = None
    request_model: type[Any] | None = None
    request_arg_index: int | None = None
    identifier_arg_indexes: tuple[int, ...] = ()


KANBAN_OPERATION_SPECS: dict[str, KanbanOperationSpec] = {
    "create_board": KanbanOperationSpec("kanban.boards.create.server", "create_kanban_board", "board", KanbanBoardCreate, 0),
    "list_boards": KanbanOperationSpec("kanban.boards.list.server", "list_kanban_boards", "board"),
    "get_board": KanbanOperationSpec("kanban.boards.detail.server", "get_kanban_board", "board", identifier_arg_indexes=(0,)),
    "update_board": KanbanOperationSpec("kanban.boards.update.server", "update_kanban_board", "board", KanbanBoardUpdate, 1, (0,)),
    "archive_board": KanbanOperationSpec("kanban.boards.archive.server", "archive_kanban_board", "board", identifier_arg_indexes=(0,)),
    "unarchive_board": KanbanOperationSpec("kanban.boards.restore.server", "unarchive_kanban_board", "board", identifier_arg_indexes=(0,)),
    "delete_board": KanbanOperationSpec("kanban.boards.delete.server", "delete_kanban_board", "board", identifier_arg_indexes=(0,)),
    "restore_board": KanbanOperationSpec("kanban.boards.restore.server", "restore_kanban_board", "board", identifier_arg_indexes=(0,)),
    "create_list": KanbanOperationSpec("kanban.lists.create.server", "create_kanban_list", "list", KanbanListCreate, 1),
    "list_lists": KanbanOperationSpec("kanban.lists.list.server", "list_kanban_lists", "list", identifier_arg_indexes=(0,)),
    "reorder_lists": KanbanOperationSpec("kanban.lists.reorder.server", "reorder_kanban_lists", "reorder", KanbanReorderRequest, 1, (0,)),
    "get_list": KanbanOperationSpec("kanban.lists.detail.server", "get_kanban_list", "list", identifier_arg_indexes=(0,)),
    "update_list": KanbanOperationSpec("kanban.lists.update.server", "update_kanban_list", "list", KanbanListUpdate, 1, (0,)),
    "archive_list": KanbanOperationSpec("kanban.lists.archive.server", "archive_kanban_list", "list", identifier_arg_indexes=(0,)),
    "unarchive_list": KanbanOperationSpec("kanban.lists.restore.server", "unarchive_kanban_list", "list", identifier_arg_indexes=(0,)),
    "delete_list": KanbanOperationSpec("kanban.lists.delete.server", "delete_kanban_list", "list", identifier_arg_indexes=(0,)),
    "restore_list": KanbanOperationSpec("kanban.lists.restore.server", "restore_kanban_list", "list", identifier_arg_indexes=(0,)),
    "create_card": KanbanOperationSpec("kanban.cards.create.server", "create_kanban_card", "card", KanbanCardCreate, 1),
    "list_cards": KanbanOperationSpec("kanban.cards.list.server", "list_kanban_cards", "card", identifier_arg_indexes=(0,)),
    "reorder_cards": KanbanOperationSpec("kanban.cards.reorder.server", "reorder_kanban_cards", "reorder", KanbanReorderRequest, 1, (0,)),
    "get_card": KanbanOperationSpec("kanban.cards.detail.server", "get_kanban_card", "card", identifier_arg_indexes=(0,)),
    "update_card": KanbanOperationSpec("kanban.cards.update.server", "update_kanban_card", "card", KanbanCardUpdate, 1, (0,)),
    "move_card": KanbanOperationSpec("kanban.cards.launch.server", "move_kanban_card", "card", KanbanCardMoveRequest, 1, (0,)),
    "copy_card": KanbanOperationSpec("kanban.cards.launch.server", "copy_kanban_card", "card", KanbanCardCopyRequest, 1, (0,)),
    "archive_card": KanbanOperationSpec("kanban.cards.archive.server", "archive_kanban_card", "card", identifier_arg_indexes=(0,)),
    "unarchive_card": KanbanOperationSpec("kanban.cards.restore.server", "unarchive_kanban_card", "card", identifier_arg_indexes=(0,)),
    "delete_card": KanbanOperationSpec("kanban.cards.delete.server", "delete_kanban_card", "card", identifier_arg_indexes=(0,)),
    "restore_card": KanbanOperationSpec("kanban.cards.restore.server", "restore_kanban_card", "card", identifier_arg_indexes=(0,)),
    "list_board_activities": KanbanOperationSpec("kanban.activities.list.server", "list_kanban_board_activities", "activity", identifier_arg_indexes=(0,)),
    "list_card_activities": KanbanOperationSpec("kanban.activities.list.server", "list_kanban_card_activities", "activity", identifier_arg_indexes=(0,)),
    "get_board_export": KanbanOperationSpec("kanban.boards.export.server", "get_kanban_board_export", "board_export", identifier_arg_indexes=(0,)),
    "export_board": KanbanOperationSpec("kanban.boards.export.server", "export_kanban_board", "board_export", KanbanBoardExportRequest, 1, (0,)),
    "import_board": KanbanOperationSpec("kanban.boards.import.server", "import_kanban_board", "board_import", KanbanBoardImportRequest, 0),
    "search_cards_basic": KanbanOperationSpec("kanban.search.list.server", "search_kanban_cards_basic", "card_search", KanbanCardSearchRequest, 0),
    "search_cards_basic_get": KanbanOperationSpec("kanban.search.list.server", "search_kanban_cards_basic_get", "card_search"),
    "bulk_move_cards": KanbanOperationSpec("kanban.cards.update.server", "bulk_move_kanban_cards", "card_bulk", KanbanBulkMoveCardsRequest, 0),
    "bulk_archive_cards": KanbanOperationSpec("kanban.cards.archive.server", "bulk_archive_kanban_cards", "card_bulk", KanbanBulkArchiveCardsRequest, 0),
    "bulk_unarchive_cards": KanbanOperationSpec("kanban.cards.restore.server", "bulk_unarchive_kanban_cards", "card_bulk", KanbanBulkArchiveCardsRequest, 0),
    "bulk_delete_cards": KanbanOperationSpec("kanban.cards.delete.server", "bulk_delete_kanban_cards", "card_bulk", KanbanBulkDeleteCardsRequest, 0),
    "bulk_label_cards": KanbanOperationSpec("kanban.cards.update.server", "bulk_label_kanban_cards", "card_bulk", KanbanBulkLabelCardsRequest, 0),
    "filter_board_cards": KanbanOperationSpec("kanban.cards.list.server", "filter_kanban_board_cards", "card", identifier_arg_indexes=(0,)),
    "copy_card_with_checklists": KanbanOperationSpec("kanban.cards.launch.server", "copy_kanban_card_with_checklists", "card", KanbanCardCopyWithChecklistsRequest, 1, (0,)),
    "create_label": KanbanOperationSpec("kanban.labels.create.server", "create_kanban_label", "label", KanbanLabelCreate, 1),
    "list_labels": KanbanOperationSpec("kanban.labels.list.server", "list_kanban_labels", "label", identifier_arg_indexes=(0,)),
    "get_label": KanbanOperationSpec("kanban.labels.detail.server", "get_kanban_label", "label", identifier_arg_indexes=(0,)),
    "update_label": KanbanOperationSpec("kanban.labels.update.server", "update_kanban_label", "label", KanbanLabelUpdate, 1, (0,)),
    "delete_label": KanbanOperationSpec("kanban.labels.delete.server", "delete_kanban_label", "label", identifier_arg_indexes=(0,)),
    "assign_label_to_card": KanbanOperationSpec("kanban.card_labels.create.server", "assign_kanban_label_to_card", "card_label", identifier_arg_indexes=(0, 1)),
    "remove_label_from_card": KanbanOperationSpec("kanban.card_labels.delete.server", "remove_kanban_label_from_card", "card_label", identifier_arg_indexes=(0, 1)),
    "list_card_labels": KanbanOperationSpec("kanban.card_labels.list.server", "list_kanban_card_labels", "label", identifier_arg_indexes=(0,)),
    "create_checklist": KanbanOperationSpec("kanban.checklists.create.server", "create_kanban_checklist", "checklist", KanbanChecklistCreate, 1),
    "list_checklists": KanbanOperationSpec("kanban.checklists.list.server", "list_kanban_checklists", "checklist", identifier_arg_indexes=(0,)),
    "get_checklist": KanbanOperationSpec("kanban.checklists.detail.server", "get_kanban_checklist", "checklist", identifier_arg_indexes=(0,)),
    "update_checklist": KanbanOperationSpec("kanban.checklists.update.server", "update_kanban_checklist", "checklist", KanbanChecklistUpdate, 1, (0,)),
    "delete_checklist": KanbanOperationSpec("kanban.checklists.delete.server", "delete_kanban_checklist", "checklist", identifier_arg_indexes=(0,)),
    "reorder_checklists": KanbanOperationSpec("kanban.checklists.reorder.server", "reorder_kanban_checklists", "reorder", KanbanChecklistReorderRequest, 1, (0,)),
    "create_checklist_item": KanbanOperationSpec("kanban.checklist_items.create.server", "create_kanban_checklist_item", "checklist_item", KanbanChecklistItemCreate, 1),
    "list_checklist_items": KanbanOperationSpec("kanban.checklist_items.list.server", "list_kanban_checklist_items", "checklist_item", identifier_arg_indexes=(0,)),
    "get_checklist_item": KanbanOperationSpec("kanban.checklist_items.detail.server", "get_kanban_checklist_item", "checklist_item", identifier_arg_indexes=(0,)),
    "update_checklist_item": KanbanOperationSpec("kanban.checklist_items.update.server", "update_kanban_checklist_item", "checklist_item", KanbanChecklistItemUpdate, 1, (0,)),
    "delete_checklist_item": KanbanOperationSpec("kanban.checklist_items.delete.server", "delete_kanban_checklist_item", "checklist_item", identifier_arg_indexes=(0,)),
    "reorder_checklist_items": KanbanOperationSpec("kanban.checklist_items.reorder.server", "reorder_kanban_checklist_items", "checklist_item", KanbanChecklistItemReorderRequest, 1, (0,)),
    "check_checklist_item": KanbanOperationSpec("kanban.checklist_items.update.server", "check_kanban_checklist_item", "checklist_item", identifier_arg_indexes=(0,)),
    "uncheck_checklist_item": KanbanOperationSpec("kanban.checklist_items.update.server", "uncheck_kanban_checklist_item", "checklist_item", identifier_arg_indexes=(0,)),
    "toggle_all_checklist_items": KanbanOperationSpec("kanban.checklist_items.update.server", "toggle_all_kanban_checklist_items", "checklist", KanbanToggleAllChecklistItemsRequest, 1, (0,)),
    "create_comment": KanbanOperationSpec("kanban.comments.create.server", "create_kanban_comment", "comment", KanbanCommentCreate, 1),
    "list_comments": KanbanOperationSpec("kanban.comments.list.server", "list_kanban_comments", "comment", identifier_arg_indexes=(0,)),
    "get_comment": KanbanOperationSpec("kanban.comments.detail.server", "get_kanban_comment", "comment", identifier_arg_indexes=(0,)),
    "update_comment": KanbanOperationSpec("kanban.comments.update.server", "update_kanban_comment", "comment", KanbanCommentUpdate, 1, (0,)),
    "delete_comment": KanbanOperationSpec("kanban.comments.delete.server", "delete_kanban_comment", "comment", identifier_arg_indexes=(0,)),
    "search_cards_get": KanbanOperationSpec("kanban.search.list.server", "search_kanban_cards_get", "search"),
    "search_cards": KanbanOperationSpec("kanban.search.list.server", "search_kanban_cards", "search", KanbanSearchRequest, 0),
    "get_search_status": KanbanOperationSpec("kanban.search.detail.server", "get_kanban_search_status", "search_status", identifier_arg_indexes=()),
    "add_card_link": KanbanOperationSpec("kanban.card_links.create.server", "add_kanban_card_link", "card_link", KanbanCardLinkCreate, 1),
    "list_card_links": KanbanOperationSpec("kanban.card_links.list.server", "list_kanban_card_links", "card_link", identifier_arg_indexes=(0,)),
    "get_card_link_counts": KanbanOperationSpec("kanban.card_links.detail.server", "get_kanban_card_link_counts", "card_link_counts", identifier_arg_indexes=(0,)),
    "remove_card_link": KanbanOperationSpec("kanban.card_links.delete.server", "remove_kanban_card_link", "card_link", identifier_arg_indexes=(0, 1, 2)),
    "remove_card_link_by_id_for_card": KanbanOperationSpec("kanban.card_links.delete.server", "remove_kanban_card_link_by_id_for_card", "card_link", identifier_arg_indexes=(0, 1)),
    "remove_card_link_by_id": KanbanOperationSpec("kanban.card_links.delete.server", "remove_kanban_card_link_by_id", "card_link", identifier_arg_indexes=(0,)),
    "bulk_add_card_links": KanbanOperationSpec("kanban.card_links.create.server", "bulk_add_kanban_card_links", "card_link_bulk", KanbanBulkCardLinksRequest, 1, (0,)),
    "bulk_remove_card_links": KanbanOperationSpec("kanban.card_links.delete.server", "bulk_remove_kanban_card_links", "card_link_bulk", KanbanBulkCardLinksRequest, 1, (0,)),
    "list_cards_by_linked_content": KanbanOperationSpec("kanban.card_links.list.server", "list_kanban_cards_by_linked_content", "linked_cards", identifier_arg_indexes=(0, 1)),
}


_COLLECTION_KEYS = {
    "boards": "board",
    "lists": "list",
    "cards": "card",
    "labels": "label",
    "checklists": "checklist",
    "items": "checklist_item",
    "comments": "comment",
    "activities": "activity",
    "links": "card_link",
    "results": "search_result",
}


class ServerKanbanService:
    """Execute stable non-workflow Kanban operations against the active server."""

    operations = KANBAN_OPERATION_SPECS

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerKanbanService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerKanbanService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def __getattr__(self, name: str) -> Any:
        if name not in self.operations:
            raise AttributeError(name)

        async def _bound_operation(*args: Any, **kwargs: Any) -> Any:
            return await self.invoke(name, *args, **kwargs)

        return _bound_operation

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server Kanban operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server Kanban action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python", by_alias=True)
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: cls._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _model(request_data: Any, model_type: type[Any]) -> Any:
        if isinstance(request_data, model_type):
            return request_data
        return model_type(**dict(request_data or {}))

    @classmethod
    def _coerce_request_args(
        cls,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        spec: KanbanOperationSpec,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if spec.request_model is None or spec.request_arg_index is None:
            return args, kwargs
        if spec.request_arg_index < len(args):
            next_args = list(args)
            next_args[spec.request_arg_index] = cls._model(next_args[spec.request_arg_index], spec.request_model)
            return tuple(next_args), kwargs
        if "request_data" in kwargs:
            next_kwargs = dict(kwargs)
            next_kwargs["request_data"] = cls._model(next_kwargs["request_data"], spec.request_model)
            return args, next_kwargs
        return args, kwargs

    @staticmethod
    def _identifier_from_args(args: tuple[Any, ...], spec: KanbanOperationSpec) -> str | None:
        if not spec.identifier_arg_indexes:
            return None
        values: list[str] = []
        for index in spec.identifier_arg_indexes:
            if index >= len(args):
                return None
            values.append(str(args[index]))
        return ":".join(values)

    @staticmethod
    def _with_record_id(kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "server")
        if identifier is None:
            identifier = record.get("id") or record.get("card_id") or record.get("board_id") or record.get("list_id")
        if identifier is not None:
            record.setdefault("record_id", f"server:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_record(
        cls,
        payload: Any,
        *,
        kind: str | None = None,
        identifier: Any | None = None,
    ) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [cls._normalize_record(item, kind=kind, identifier=identifier) for item in payload]
        if not isinstance(payload, dict):
            if kind is not None:
                return cls._with_record_id(cls._record_kind_name(kind), {"success": bool(payload)}, identifier)
            return payload

        record = {key: cls._normalize_collection_value(key, value) for key, value in payload.items()}
        record.setdefault("backend", "server")
        record_kind = cls._record_kind_name(kind)
        if record_kind is not None:
            if kind == "board_import":
                identifier = identifier or record.get("import_stats", {}).get("board_id") or record.get("board", {}).get("id")
            elif kind == "search_status":
                identifier = "active"
            elif kind == "board_export":
                identifier = identifier or record.get("board", {}).get("id")
            elif kind == "card_link_counts":
                identifier = identifier or record.get("card_id")
            elif kind == "reorder":
                identifier = identifier or "operation"
            return cls._with_record_id(record_kind, record, identifier)
        return record

    @classmethod
    def _normalize_collection_value(cls, key: str, value: Any) -> Any:
        collection_kind = _COLLECTION_KEYS.get(key)
        if collection_kind is not None and isinstance(value, list):
            return [cls._normalize_record(item, kind=collection_kind) for item in value]
        if key == "board" and isinstance(value, dict):
            return cls._normalize_record(value, kind="board")
        if isinstance(value, dict):
            return {nested_key: cls._normalize_collection_value(nested_key, nested_value) for nested_key, nested_value in value.items()}
        if isinstance(value, list):
            return [cls._normalize_collection_value("", item) for item in value]
        return value

    @staticmethod
    def _record_kind_name(kind: str | None) -> str | None:
        if kind == "board":
            return "kanban_board"
        if kind == "list":
            return "kanban_list"
        if kind == "card":
            return "kanban_card"
        if kind == "activity":
            return "kanban_activity"
        if kind == "board_export":
            return "kanban_board_export"
        if kind == "board_import":
            return "kanban_board_import"
        if kind == "label":
            return "kanban_label"
        if kind == "card_label":
            return "kanban_card_label"
        if kind == "checklist":
            return "kanban_checklist"
        if kind == "checklist_item":
            return "kanban_checklist_item"
        if kind == "comment":
            return "kanban_comment"
        if kind == "search":
            return "kanban_search"
        if kind == "search_result":
            return "kanban_search_result"
        if kind == "card_search":
            return "kanban_card_search"
        if kind == "search_status":
            return "kanban_search_status"
        if kind == "card_link":
            return "kanban_card_link"
        if kind == "card_link_counts":
            return "kanban_card_link_counts"
        if kind == "card_link_bulk":
            return "kanban_card_link_bulk"
        if kind == "linked_cards":
            return "kanban_linked_cards"
        if kind == "card_bulk":
            return "kanban_card_bulk"
        if kind == "reorder":
            return "kanban_reorder"
        return None

    @classmethod
    def _normalize_response(
        cls,
        payload: Any,
        *,
        kind: str | None = None,
        identifier: Any | None = None,
    ) -> Any:
        return cls._normalize_record(payload, kind=kind, identifier=identifier)

    async def invoke(self, operation_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            spec = self.operations[operation_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Kanban operation: {operation_name}") from exc
        self._enforce(spec.action_id)
        coerced_args, coerced_kwargs = self._coerce_request_args(args, dict(kwargs), spec)
        result = await getattr(self._require_client(), spec.client_method)(*coerced_args, **coerced_kwargs)
        return self._normalize_response(
            result,
            kind=spec.kind,
            identifier=self._identifier_from_args(coerced_args, spec),
        )
