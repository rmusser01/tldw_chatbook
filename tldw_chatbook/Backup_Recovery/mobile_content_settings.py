"""Private C2a native settings projections, never an insertion authorization."""

import json
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import ConfigDict, TypeAdapter

# Native postponed annotations must also resolve on the declared Pydantic floor.
from tldw_chatbook.Chat.message_metadata import (  # noqa: F401
    AgentChatStartMetadata,
    CanvasCardMetadata,
    CanvasCardOriginMetadata,
    CharacterEmoteEventMetadata,
    CharacterEmoteMetadata,
    MessageMetadata,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

from .mobile_content_core import _revision
from .mobile_content_graph import (
    _ContentBlocker,
    _ContentGraph,
    _ContentRow,
    _GraphBuilder,
    _UnprovedReference,
)
from .mobile_workspace_plan import _object

_TABLES = (
    "conversations",
    "messages",
    "console_conversation_capture_policy",
    "console_conversation_library_policy",
)
_MESSAGE_FIELDS = (
    "usage_json",
    "metadata_json",
    "provider_continuation_json",
    "assistant_generation_state",
    "thinking_blocks_json",
)
_METADATA_JSON = TypeAdapter(tuple[MessageMetadata], config=ConfigDict(extra="forbid"))
_USAGE_JSON = TypeAdapter(tuple[ProviderUsage], config=ConfigDict(extra="forbid"))
_VIDEO_JSON = TypeAdapter(
    tuple[VideoGenerationMetadata], config=ConfigDict(extra="forbid")
)


@dataclass(frozen=True)
class _SettingsSuccessor:
    """A named native consumer still owes this exact row a proof."""

    table: str
    original: tuple[str | int, ...]
    handler: Literal["C2b", "C2c", "C3", "T", "S", "A-runs", "I"]
    domain: str
    original_id: str | int
    mapped_id: str | int


@dataclass(frozen=True)
class _LocalAuthoritySlot:
    """Non-insertable C2a value: T must prove both captured native contexts."""

    conversation_id: str
    mapped_conversation_id: str
    original_character_id: int
    mapped_character_id: int
    original_authority_id: str = field(repr=False)
    context_name: Literal["default"] = "default"
    handler: Literal["T"] = "T"
    column: Literal["assistant_authority_id"] = "assistant_authority_id"


@dataclass(frozen=True)
class _DictionaryDerivation:
    conversation_id: str
    mapped_conversation_id: str
    original_attachments: tuple[int, ...]
    mapped_attachments: tuple[int, ...]
    original_unresolved: bool
    mapped_unresolved: bool
    original_active: tuple[int, ...]
    mapped_active: tuple[int, ...]


@dataclass(frozen=True)
class _SettingsPlan:
    graph: _ContentGraph = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    successors: tuple[_SettingsSuccessor, ...]
    dictionary_derivations: tuple[_DictionaryDerivation, ...]
    authority_slots: tuple[_LocalAuthoritySlot, ...]
    pending_handlers: tuple[str, ...] = ("C2b", "C2c", "C3", "T", "S", "A-runs")

    @property
    def identities(self):
        return self.graph.identities

    def require_complete(self) -> None:
        raise ValueError("mobile_content_handlers_pending")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _typed(adapter, encoded):
    _object(encoded)  # Reject duplicate keys/non-finite numbers before native parsing.
    return adapter.validate_json("[" + encoded + "]", strict=True)[0]


def _scope(builder, value):
    from tldw_chatbook.Chat.rag_scope import parse_scope

    if (
        type(value) is not dict
        or set(value) - {"version", "updated_at", "items", "empty_is_scoped"}
        or type(value.get("version")) is not int
        or value["version"] not in (1, 2)
        or type(value.get("empty_is_scoped", False)) is not bool
        or parse_scope(value) is None
    ):
        raise ValueError("invalid_rag_scope")
    items = []
    for item in value["items"]:
        if (
            type(item) is not dict
            or set(item) != {"source_type", "source_id"}
            or type(item["source_id"]) is not str
        ):
            raise ValueError("invalid_rag_scope")
        if item["source_type"] == "media":
            raise _UnprovedReference("media", item["source_id"])
        if item["source_type"] != "note":
            raise ValueError("invalid_rag_scope")
        items.append(item | {"source_id": builder.ref("note", item["source_id"])})
    return value | {"items": items}


def _dictionary_shape(encoded):
    """Installed v34→35 predicate for the strictly admitted metadata subset."""
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
        LocalChatDictionaryService,
    )

    if encoded is None:
        return (), False, ()
    metadata = _object(encoded)
    values = metadata.get("active_dictionaries", [])
    if type(values) is not list:
        raise ValueError("dictionary_list_unrepresentable")
    if any(
        type(value) is not int
        and not (type(value) is str and re.fullmatch(r"[1-9][0-9]*", value))
        for value in values
    ):
        raise ValueError("dictionary_identity_unrepresentable")
    marker = "active_dictionaries"
    # Native LIKE '%active_dictionaries%' folds ASCII only; '_' is one
    # Unicode character, rather than a literal underscore.
    visible = (
        re.search(r"active.dictionaries", encoded, re.ASCII | re.IGNORECASE | re.DOTALL)
        is not None
    )
    attachments = (
        tuple(sorted({value for value in values if type(value) is int}))
        if visible
        else ()
    )
    unresolved = visible and (
        encoded.count(marker) != 1 or any(type(value) is str for value in values)
    )
    active = tuple(
        LocalChatDictionaryService._active_dictionaries({"metadata": encoded})
    )
    return attachments, unresolved, active


def _metadata(builder, row, successor):
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        ConsoleGenerationSettingsReadStatus,
        parse_console_generation_settings,
    )
    from tldw_chatbook.Chat.console_roleplay_metadata import (
        ConsoleRoleplayContext,
        parse_console_roleplay_context,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        parse_persisted_console_session_settings,
    )
    from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

    raw = row["metadata"]
    if raw is None:
        return None
    value = _object(raw)
    known = {
        "console_session_settings",
        "console_generation_settings",
        "console_roleplay_context",
        "console_speech",
        "pinned_response_prefill",
        "rag_scope",
        "active_dictionaries",
        "console_agent_handoff",
        "goal_launch",
        "console_appearance",
    }
    if set(value) - known:
        raise ValueError("unknown_conversation_metadata")
    if "console_appearance" in value:
        from tldw_chatbook.Chat.console_appearance import ConsoleConversationAppearance

        appearance = value["console_appearance"]
        if type(appearance) is not dict or set(appearance) != {"icon", "color"}:
            raise ValueError("invalid_appearance")
        ConsoleConversationAppearance(**appearance)
    if (
        "console_session_settings" in value
        and parse_persisted_console_session_settings(raw) is None
    ):
        raise ValueError("invalid_session_settings")
    if (
        "console_generation_settings" in value
        and parse_console_generation_settings(value).status
        is not ConsoleGenerationSettingsReadStatus.VALID
    ):
        raise ValueError("invalid_generation_settings")
    if "console_roleplay_context" in value:
        context = value["console_roleplay_context"]
        if (
            type(context) is not dict
            or type(context.get("version")) is not int
            or context["version"] not in (1, 2)
        ):
            raise ValueError("invalid_roleplay_context")
        fields = {
            "user_name_override",
            "character_system_template",
            "persona_system_template",
        }
        if context["version"] == 2:
            fields.add("character_name_snapshot")
        if set(context) - fields - {"version"}:
            raise ValueError("unknown_roleplay_context")
        expected = ConsoleRoleplayContext(
            **{key: item for key, item in context.items() if key != "version"}
        )
        if parse_console_roleplay_context(value) != expected:
            raise ValueError("lossy_roleplay_context")
    if "console_speech" in value:
        speech = value["console_speech"]
        if type(speech) is not dict or set(speech) != {
            "auto_speak",
            "paused",
            "consent_destination",
            "consent_version",
        }:
            raise ValueError("invalid_speech_context")
        ConsoleSpeechPreferences(**speech)
    if "pinned_response_prefill" in value and type(
        value["pinned_response_prefill"]
    ) not in (str, type(None)):
        raise ValueError("invalid_prefill")
    if "rag_scope" in value:
        value["rag_scope"] = _scope(builder, value["rag_scope"])
    if "active_dictionaries" in value:
        _dictionary_shape(raw)
        value["active_dictionaries"] = [
            builder.ref("dictionary", item)
            if type(item) is int
            else str(builder.ref("dictionary", int(item)))
            for item in value["active_dictionaries"]
        ]
    if "console_agent_handoff" in value:
        from tldw_chatbook.Chat.message_metadata import AgentHandoffLaunchMetadata

        handoff = value["console_agent_handoff"]
        if type(handoff) is not dict:
            raise ValueError("invalid_handoff")
        common = {"draft", "created_via", "source_run_id"}
        expected = (
            common | {"version", "state", "draft_revision", "launch"}
            if "version" in handoff
            else common
        )
        if set(handoff) != expected or any(
            type(handoff[key]) is not str for key in common
        ):
            raise ValueError("invalid_handoff")
        if "version" in handoff and (
            type(handoff["version"]) is not int
            or handoff["version"] != 2
            or handoff["state"] not in {"pending", "consumed"}
            or type(handoff["draft_revision"]) is not int
            or handoff["draft_revision"] < 1
            or AgentHandoffLaunchMetadata.read(handoff["launch"]) is None
        ):
            raise ValueError("invalid_handoff")
        source = handoff["source_run_id"]
        mapped = builder.ref("run", source)
        value["console_agent_handoff"] = handoff | {"source_run_id": mapped}
        successor("C2b", "handoff_run", source, mapped)
        successor("A-runs", "run", source, mapped)
    if "goal_launch" in value:
        launch = value["goal_launch"]
        if (
            type(launch) is not dict
            or set(launch) != {"goal_id", "launch_id", "payload_hash"}
            or any(type(v) is not str or not v for v in launch.values())
        ):
            raise ValueError("invalid_goal_launch")
        mapped = builder.ref("goal", launch["goal_id"])
        value["goal_launch"] = launch | {"goal_id": mapped}
        successor("A-runs", "goal_launch", launch["goal_id"], mapped)
    # No reserialization is needed when the declared ownership references stayed put.
    return raw if value == _object(raw) else _json(value)


def _project_context(builder, raw, successor):
    from tldw_chatbook.Chat.console_project_instructions import (
        decode_project_context_json,
        encode_project_context_json,
    )

    if raw is None:
        return None
    value = _object(raw)
    native = decode_project_context_json(raw)
    if _object(encode_project_context_json(native)) != value:
        raise ValueError("invalid_project_context")
    if native.working_folder_binding_id is None:
        return raw
    identity = native.working_folder_binding_id
    mapped = builder.ref("binding", identity)
    successor("I", "binding", identity, mapped)
    return encode_project_context_json(
        replace(native, working_folder_binding_id=mapped)
    )


def _message_metadata(builder, raw, successor, *, message_id=None):
    if raw is None:
        return None
    value = _object(raw)
    if "video_generation" in value:
        from tldw_chatbook.Chat.conversation_local_marks_service import (
            ConversationLocalMarksService,
        )
        from tldw_chatbook.Video_Generation.video_metadata import video_relative_path

        if set(value) - {"video_generation", "terminal_receipt_id"}:
            raise ValueError("mixed_message_metadata")
        if "terminal_receipt_id" in value:
            ConversationLocalMarksService.validate_terminal_receipt_id(
                value["terminal_receipt_id"]
            )
        video = _typed(_VIDEO_JSON, _json(value["video_generation"]))
        source = video.source_image_message_id
        if source is not None:
            video = replace(
                video, source_image_message_id=builder.ref("message", source)
            )
        if not video.is_unavailable_tombstone:
            # No captured physical-member proof/publisher is provided by C2a.
            # A-files owns agents.history, not this native VideoStore root.
            member = video_relative_path(message_id, video.name, video.container)
            raise _UnprovedReference("generated_video_member", str(member))
        encoded = video.to_json()
        if "terminal_receipt_id" in value:
            return _json(
                _object(encoded) | {"terminal_receipt_id": value["terminal_receipt_id"]}
            )
        return encoded
    if "console_fork" in value:
        from tldw_chatbook.Chat.console_chat_fork import (
            parse_console_fork_message_metadata,
        )

        parsed = parse_console_fork_message_metadata(raw)
        if parsed is None:
            raise ValueError("invalid_fork_metadata")
        if parsed[2] is not None:
            successor("T", "trace_turn", parsed[2], parsed[2])
        return raw
    native = _typed(_METADATA_JSON, raw)
    if native.agent_chat_start is not None:
        source = native.agent_chat_start
        mapped = replace(
            source,
            attempt_id=builder.ref("chat_start_attempt", source.attempt_id),
            source_run_id=builder.ref("run", source.source_run_id),
            source_conversation_id=builder.ref(
                "conversation", source.source_conversation_id
            ),
        )
        native = replace(native, agent_chat_start=mapped)
        successor("C2b", "chat_start_attempt", source.attempt_id, mapped.attempt_id)
        successor("A-runs", "run", source.source_run_id, mapped.source_run_id)
    cards = []
    for card in native.canvas_cards:
        cards.append(
            replace(
                card,
                canvas_id=builder.ref("canvas", card.canvas_id),
                revision_id=builder.ref(
                    "canvas_revision", card.revision_id, optional=True
                ),
                origin=replace(
                    card.origin,
                    message_id=builder.ref("message", card.origin.message_id),
                ),
            )
        )
        successor("C2b", "canvas_turn", card.origin.run_id, card.origin.run_id)
    native = replace(native, canvas_cards=tuple(cards))
    if native.character_emote is not None:
        emote = native.character_emote
        mapped = {}
        for name, domain in (
            ("actor_id", "character"),
            ("pack_id", "visual_pack"),
            ("pack_version_id", "visual_version"),
            ("expression_id", "character_expression"),
            ("asset_id", "visual_asset"),
        ):
            identity = getattr(emote, name)
            mapped[name] = builder.ref(domain, identity, optional=True)
            if identity is not None:
                successor("C2c", domain, identity, mapped[name])
        native = replace(native, character_emote=replace(emote, **mapped))
    return native.to_json()


def _validate_conversation_scalars(row):
    """Validate installed DB scalars without its lossy input normalization."""
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        _CONVERSATION_IDENTITY_TEXT_MAX_BYTES,
        CharactersRAGDB,
    )

    _revision(row)
    for name in ("created_at", "last_modified", "client_id"):
        if type(row[name]) is not str:
            raise ValueError("invalid_conversation_text")
    for name in (
        "title",
        "topic_label",
        "topic_last_tagged_at",
        "cluster_id",
        "source",
        "external_ref",
        "system_prompt",
        "context_summary",
        "discovery_entity_id",
    ):
        if row[name] is not None and type(row[name]) is not str:
            raise ValueError("invalid_conversation_text")
    if row["rating"] is not None and (
        type(row["rating"]) is not int or not 1 <= row["rating"] <= 5
    ):
        raise ValueError("invalid_rating")
    if type(row["archived"]) is not int or row["archived"] not in (0, 1):
        raise ValueError("invalid_archive_state")
    for name in ("assistant_id", "assistant_authority_id"):
        value = row[name]
        if value is not None and (
            type(value) is not str
            or not value
            or value != value.strip()
            or len(value.encode("utf-8")) > _CONVERSATION_IDENTITY_TEXT_MAX_BYTES
        ):
            raise ValueError("invalid_assistant_identity")
    kind = row["assistant_kind"]
    if kind not in {None, *CharactersRAGDB._ALLOWED_CONVERSATION_ASSISTANT_KINDS}:
        raise ValueError("unknown_assistant_kind")
    if kind == "character" and row["runtime_backend"] == "local":
        character = row["character_id"]
        if (
            type(character) is not int
            or CharactersRAGDB._normalize_positive_character_id(character) != character
            or row["assistant_id"] != str(character)
        ):
            raise ValueError("invalid_local_character")
    elif row["character_id"] is not None:
        raise ValueError("nonlocal_character_reference")
    if kind in {"character", "persona"} and row["assistant_id"] is None:
        raise ValueError("missing_assistant_identity")
    if kind is None and row["assistant_id"] is not None:
        raise ValueError("untyped_assistant_identity")
    if (
        row["scope_type"] == "workspace"
        and row["workspace_id"] is None
        or row["scope_type"] == "global"
        and row["workspace_id"] is not None
    ):
        raise ValueError("invalid_workspace_scope")


def _plan_content_settings(reads):
    from tldw_chatbook.Chat.assistant_generation_state import AssistantGenerationState
    from tldw_chatbook.Chat.console_capture_policy_repository import (
        ConsoleCapturePolicyRepository,
    )
    from tldw_chatbook.Chat.console_library_policy_repository import (
        ConsoleLibraryPolicyRepository,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Chat.thinking_blocks import parse_thinking_blocks_json
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        CharactersRAGDB,
        _validated_thinking_history_policy,
    )

    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_content_native_snapshot_required")
    reads.check()
    graph = reads.content_graph_plan()
    builder = _GraphBuilder(reads, graph.identities, _TABLES)
    builder.blockers.extend(graph.blockers)
    builder.rows.extend(row for row in graph.rows if row.table != "messages")
    partial_messages = {
        row.original: row for row in graph.rows if row.table == "messages"
    }
    successors, derivations, authority_slots = [], [], []

    def successor_for(table, original, pending):
        def require(handler, domain, source, mapped):
            pending.append(
                _SettingsSuccessor(table, original, handler, domain, source, mapped)
            )

        return require

    def conversation(row):
        _validate_conversation_scalars(row)
        pending = []
        successor = successor_for("conversations", (row["id"],), pending)
        if (
            row["runtime_backend"] not in {"local", "server"}
            or row["scope_type"] not in CharactersRAGDB._ALLOWED_SCOPE_TYPES
            or row["state"] not in CharactersRAGDB._ALLOWED_CONVERSATION_STATES
            or row["topic_label_source"] not in {None, "manual", "auto"}
        ):
            raise ValueError("invalid_conversation_settings")
        _validated_thinking_history_policy(row["thinking_history_policy"])
        if row["thinking_history_policy"] not in {None, "auto", "include", "exclude"}:
            raise ValueError("unknown_thinking_policy")
        if row["archived"] not in (0, 1) or row["discovery_owner"] not in {
            "general_chat",
            "ccp_character",
            "ccp_persona",
        }:
            raise ValueError("invalid_conversation_settings")
        values = dict(row)
        for column, domain in (
            ("id", "conversation"),
            ("root_id", "conversation"),
            ("parent_conversation_id", "conversation"),
            ("forked_from_message_id", "message"),
            ("active_leaf_message_id", "message"),
            ("active_leaf_before_message_id", "message"),
            ("summary_boundary_message_id", "message"),
            ("topic_last_tagged_message_id", "message"),
            ("character_id", "character"),
            ("workspace_id", "workspace"),
        ):
            values[column] = builder.ref(domain, row[column], optional=column != "id")
        kind = row["assistant_kind"]
        if kind not in {None, *CharactersRAGDB._ALLOWED_CONVERSATION_ASSISTANT_KINDS}:
            raise ValueError("unknown_assistant_kind")
        if kind == "character" and row["runtime_backend"] == "local":
            if type(row["character_id"]) is not int or row["assistant_id"] != str(
                row["character_id"]
            ):
                raise ValueError("invalid_local_character")
            values["assistant_id"] = str(values["character_id"])
            if row["assistant_authority_id"] is not None:
                authority_slots.append(
                    _LocalAuthoritySlot(
                        row["id"],
                        values["id"],
                        row["character_id"],
                        values["character_id"],
                        values.pop("assistant_authority_id"),
                    )
                )
                successor(
                    "T",
                    "character_authority",
                    row["character_id"],
                    values["character_id"],
                )
            successor("C2c", "character", row["character_id"], values["character_id"])
        elif kind == "persona" and row["runtime_backend"] == "local":
            try:
                reads.authority_reference("persona", row["assistant_id"])
            except ValueError:
                raise _UnprovedReference("persona", row["assistant_id"]) from None
            successor("C2c", "persona", row["assistant_id"], row["assistant_id"])
        discovery = row["discovery_entity_id"]
        if (
            row["runtime_backend"] == "local"
            and row["discovery_owner"] == "ccp_character"
            and discovery is not None
        ):
            if not re.fullmatch(r"[1-9][0-9]*", discovery):
                raise ValueError("invalid_discovery_character")
            mapped_discovery = builder.ref("character", int(discovery))
            values["discovery_entity_id"] = str(mapped_discovery)
            successor("C2c", "discovery_character", int(discovery), mapped_discovery)
        elif (
            row["runtime_backend"] == "local"
            and row["discovery_owner"] == "ccp_persona"
            and discovery is not None
        ):
            try:
                reads.authority_reference("persona", discovery)
            except ValueError:
                raise _UnprovedReference("persona", discovery) from None
            successor("C2c", "discovery_persona", discovery, discovery)
        if row["runtime_backend"] == "server" and row["assistant_id"] is not None:
            successor(
                "C2c", "remote_assistant", row["assistant_id"], row["assistant_id"]
            )
        if (
            kind != "character"
            and row["assistant_authority_id"] is not None
            or kind != "persona"
            and row["persona_memory_mode"] is not None
            or row["persona_memory_mode"]
            not in {None, *CharactersRAGDB._ALLOWED_PERSONA_MEMORY_MODES}
        ):
            raise ValueError("invalid_assistant_policy")
        values["metadata"] = _metadata(builder, row, successor)
        values["console_project_context_json"] = _project_context(
            builder, row["console_project_context_json"], successor
        )
        before = _dictionary_shape(row["metadata"])
        after = _dictionary_shape(values["metadata"])
        original_edges = tuple(
            sorted(
                dict(r.values)["dictionary_id"]
                for r in graph.source_rows
                if r.table == "conversation_dictionary_attachments"
                and dict(r.values)["conversation_id"] == row["id"]
            )
        )
        unresolved = any(
            r.table == "conversation_dictionary_unresolved"
            and r.original == (row["id"],)
            for r in graph.source_rows
        )
        if before[:2] != (original_edges, unresolved):
            raise ValueError("native_dictionary_index_mismatch")
        derivations.append(
            _DictionaryDerivation(
                row["id"],
                values["id"],
                before[0],
                after[0],
                before[1],
                after[1],
                before[2],
                after[2],
            )
        )
        if (
            row["context_summary"] is not None
            or row["summary_boundary_message_id"] is not None
        ):
            successor("C3", "conversation_memory", row["id"], values["id"])
        missing = (
            ("assistant_authority_id",)
            if "assistant_authority_id" not in values
            else ()
        )
        builder.emit(
            "conversations",
            row,
            values,
            pending=missing,
            proofs=tuple(
                sorted({item.handler + ":" + item.domain for item in pending})
            ),
        )
        successors.extend(pending)

    builder.handle("conversations", conversation)

    def message(row):
        partial = partial_messages.get((row["id"],))
        if partial is None:
            return  # C1's exact blocker is already present.
        pending = []
        successor = successor_for("messages", (row["id"],), pending)
        fields = {name: row[name] for name in _MESSAGE_FIELDS}
        if fields["usage_json"] is not None:
            usage = _typed(_USAGE_JSON, fields["usage_json"])
            if any(
                getattr(usage, name) < 0
                for name in (
                    "uncached_input",
                    "cache_read",
                    "cache_write",
                    "output",
                    "audio_input",
                    "audio_output",
                    "transcription_seconds",
                )
            ):
                raise ValueError("negative_usage")
        fields["metadata_json"] = _message_metadata(
            builder, fields["metadata_json"], successor, message_id=row["id"]
        )
        if fields["provider_continuation_json"] is not None:
            parse_provider_continuation_json(fields["provider_continuation_json"])
            successor(
                "C2b",
                "provider_continuation",
                row["id"],
                graph.identities.mapped("message", row["id"]),
            )
        if fields["thinking_blocks_json"] is not None:
            parse_thinking_blocks_json(fields["thinking_blocks_json"])
        if fields["assistant_generation_state"] is not None:
            if row["role"] != "assistant":
                raise ValueError("generation_state_nonassistant")
            state = AssistantGenerationState(fields["assistant_generation_state"])
            if state in {
                AssistantGenerationState.ACCEPTED,
                AssistantGenerationState.DISPATCH_STARTED,
                AssistantGenerationState.CONTINUATION_ACTIVE,
            }:
                successor(
                    "C2b",
                    "dispatch_owner",
                    row["id"],
                    graph.identities.mapped("message", row["id"]),
                )
        builder.dependencies.extend(partial.dependencies)
        builder.emit(
            "messages",
            row,
            dict(partial.values) | fields,
            proofs=tuple(
                sorted({item.handler + ":" + item.domain for item in pending})
            ),
        )
        successors.extend(pending)

    builder.handle("messages", message)
    for table in _TABLES[2:]:

        def policy(row, table=table):
            if table == "console_conversation_capture_policy":
                if ConsoleCapturePolicyRepository._from_row(row) is None:
                    raise ValueError("invalid_capture_policy")
            elif (
                ConsoleLibraryPolicyRepository._result_from_row(
                    row | {"conversation_deleted": 0}
                ).durable_policy
                is None
            ):
                raise ValueError("invalid_library_policy")
            builder.emit(
                table,
                row,
                row
                | {
                    "conversation_id": builder.ref(
                        "conversation", row["conversation_id"]
                    )
                },
            )

        builder.handle(table, policy)
    builder.close_dependencies()
    accepted = {(r.table, r.original) for r in builder.rows}
    reads.check()
    return _SettingsPlan(
        graph,
        tuple(row for row in builder.rows if row.table in _TABLES),
        builder.retained(),
        tuple(builder.blockers),
        tuple(item for item in successors if (item.table, item.original) in accepted),
        tuple(
            item
            for item in derivations
            if ("conversations", (item.conversation_id,)) in accepted
        ),
        tuple(
            item
            for item in authority_slots
            if ("conversations", (item.conversation_id,)) in accepted
        ),
    )
