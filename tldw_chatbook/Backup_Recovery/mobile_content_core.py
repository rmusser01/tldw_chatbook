"""Fixed native core graph fields for captured C1 content plans."""

import json

from .mobile_content_graph import (
    _ConversationStructure,
    _NativeConflict,
    _UnprovedReference,
)

_CORE_TABLES = (
    "conversations",
    "notes",
    "note_links",
    "character_cards",
    "character_expression_images",
    "keywords",
    "keyword_collections",
    "chat_dictionaries",
    "world_books",
    "world_book_entries",
    "note_folders",
    "note_folder_memberships",
    "note_keywords",
    "collection_keywords",
    "conversation_keywords",
    "conversation_dictionaries",
    "conversation_dictionary_attachments",
    "conversation_dictionary_unresolved",
    "conversation_local_marks",
    "conversation_world_books",
    "messages",
    "message_attachments",
)
_MESSAGE_PENDING = (
    "usage_json",
    "metadata_json",
    "provider_continuation_json",
    "assistant_generation_state",
    "thinking_blocks_json",
)


def _revision(row):
    if "version" in row and (type(row["version"]) is not int or row["version"] < 1):
        raise ValueError("invalid_version")
    if "deleted" in row and (
        type(row["deleted"]) is not int or row["deleted"] not in (0, 1)
    ):
        raise ValueError("invalid_deletion")


def _unique(builder, table, row, column, *, nocase=False, active=False):
    # SQLite's installed NOCASE is ASCII-only, unlike Python casefold().
    def key(value):
        if type(value) is not str:
            raise ValueError("invalid_unique_text")
        return (
            value.translate(
                str.maketrans(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
                )
            )
            if nocase
            else value
        )

    if nocase and type(row[column]) is str and "\x00" in row[column]:
        raise _UnprovedReference("native_nocase_text", row["id"])
    if active and row["deleted"]:
        return
    if any(
        (not active or not other["deleted"]) and key(other[column]) == key(row[column])
        for other in builder.canonical[table]
    ):
        raise _NativeConflict()


def _sync_identity(builder, table, row):
    if row.get("sync_id") is not None:
        if type(row["sync_id"]) is not str:
            raise ValueError("invalid_sync_identity")
        if any(
            other["sync_id"] == row["sync_id"] for other in builder.canonical[table]
        ):
            raise _NativeConflict()


def _ancestor_chain(builder, table, row, parent_column):
    seen = {row["id"]}
    current = row
    while current[parent_column] is not None:
        parent = current[parent_column]
        if parent in seen:
            raise ValueError("cyclic_native_graph")
        seen.add(parent)
        current = builder.by_id[table].get(parent)
        if current is None:
            raise _UnprovedReference(table, parent)


def _entities(builder):
    def character(row):
        _revision(row)
        _unique(builder, "character_cards", row, "name")
        # Native card fields and embedded dictionary/lore snapshots are bodies;
        # their JSON is not a second identity registry.
        builder.emit(
            "character_cards", row, row | {"id": builder.ref("character", row["id"])}
        )

    builder.handle("character_cards", character)

    def expression(row):
        _revision(row)
        if (
            type(row["image"]) is not bytes
            or row["mime"] is not None
            and type(row["mime"]) is not str
        ):
            raise ValueError("invalid_expression_image")
        builder.emit(
            "character_expression_images",
            row,
            row
            | {
                "id": builder.ref("character_expression", row["id"]),
                "character_id": builder.ref("character", row["character_id"]),
            },
        )

    builder.handle("character_expression_images", expression)

    def keyword(row):
        _revision(row)
        _unique(builder, "keywords", row, "keyword", nocase=True)
        _sync_identity(builder, "keywords", row)
        builder.emit("keywords", row, row | {"id": builder.ref("keyword", row["id"])})

    builder.handle("keywords", keyword)

    def collection(row):
        _revision(row)
        _unique(builder, "keyword_collections", row, "name", nocase=True)
        _sync_identity(builder, "keyword_collections", row)
        _ancestor_chain(builder, "keyword_collections", row, "parent_id")
        builder.emit(
            "keyword_collections",
            row,
            row
            | {
                "id": builder.ref("keyword_collection", row["id"]),
                "parent_id": builder.ref(
                    "keyword_collection", row["parent_id"], optional=True
                ),
            },
        )

    builder.handle("keyword_collections", collection)

    def dictionary(row):
        from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary

        _revision(row)
        _unique(builder, "chat_dictionaries", row, "name")
        if row["file_path"] is not None:
            raise _UnprovedReference("dictionary_file", row["id"])
        if row["entries_json"]:
            entries = json.loads(row["entries_json"])
            if type(entries) is not list:
                raise ValueError("invalid_dictionary_entries")
            for entry in entries:
                ChatDictionary.from_dict(entry)
        builder.emit(
            "chat_dictionaries", row, row | {"id": builder.ref("dictionary", row["id"])}
        )

    builder.handle("chat_dictionaries", dictionary)

    def book(row):
        _revision(row)
        _unique(builder, "world_books", row, "name")
        builder.emit(
            "world_books", row, row | {"id": builder.ref("world_book", row["id"])}
        )

    builder.handle("world_books", book)

    def entry(row):
        for column in ("keys", "secondary_keys"):
            if row[column] is not None:
                value = json.loads(row[column])
                if type(value) is not list or any(
                    type(key) is not str for key in value
                ):
                    raise ValueError("invalid_lore_keys")
        builder.emit(
            "world_book_entries",
            row,
            row
            | {
                "id": builder.ref("world_book_entry", row["id"]),
                "world_book_id": builder.ref("world_book", row["world_book_id"]),
            },
        )

    builder.handle("world_book_entries", entry)


def _folders(builder):
    from tldw_chatbook.Notes.note_folder_models import (
        join_normalized_folder_path,
        normalize_folder_name,
    )
    from tldw_chatbook.Notes.note_folder_repository import _join_display_folder_path

    def folder(row):
        _revision(row)
        _ancestor_chain(builder, "note_folders", row, "parent_id")
        name = normalize_folder_name(row["name"])
        parent = builder.by_id["note_folders"].get(row["parent_id"])
        path = _join_display_folder_path(parent["path"] if parent else "", name.display)
        normalized = join_normalized_folder_path(
            parent["normalized_path"] if parent else "", name.key
        )
        if (
            row["name"],
            row["normalized_name"],
            row["path"],
            row["normalized_path"],
        ) != (name.display, name.key, path, normalized):
            raise ValueError("invalid_folder_path")
        _unique(builder, "note_folders", row, "normalized_path", active=True)
        _sync_identity(builder, "note_folders", row)
        builder.emit(
            "note_folders",
            row,
            row
            | {
                "id": builder.ref("note_folder", row["id"]),
                "parent_id": builder.ref(
                    "note_folder", row["parent_id"], optional=True
                ),
            },
        )

    builder.handle("note_folders", folder)

    def membership(row):
        _revision(row)
        if row["ownership"] == "managed":
            raise _UnprovedReference("managed_note_owner", row["owner_id"])
        if (
            row["ownership"] != "manual"
            or row["owner_id"] != ""
            or row["owner_active"] != 1
        ):
            raise ValueError("invalid_manual_membership")
        builder.emit(
            "note_folder_memberships",
            row,
            row
            | {
                "id": builder.ref("note_folder_membership", row["id"]),
                "folder_id": builder.ref("note_folder", row["folder_id"]),
                "note_id": builder.ref("note", row["note_id"]),
            },
        )

    builder.handle("note_folder_memberships", membership)


def _associations(builder):
    # These six native junctions contain only the two declared references and
    # their preserved local order/timestamp fields. No JSON or guessed key scan.
    junctions = (
        ("note_keywords", "note_id", "note", "keyword_id", "keyword"),
        (
            "collection_keywords",
            "collection_id",
            "keyword_collection",
            "keyword_id",
            "keyword",
        ),
        (
            "conversation_keywords",
            "conversation_id",
            "conversation",
            "keyword_id",
            "keyword",
        ),
        (
            "conversation_dictionaries",
            "conversation_id",
            "conversation",
            "dictionary_id",
            "dictionary",
        ),
        (
            "conversation_dictionary_attachments",
            "conversation_id",
            "conversation",
            "dictionary_id",
            "dictionary",
        ),
        (
            "conversation_world_books",
            "conversation_id",
            "conversation",
            "world_book_id",
            "world_book",
        ),
    )
    for table, left, left_domain, right, right_domain in junctions:

        def junction(
            row,
            table=table,
            left=left,
            left_domain=left_domain,
            right=right,
            right_domain=right_domain,
        ):
            builder.emit(
                table,
                row,
                row
                | {
                    left: builder.ref(left_domain, row[left]),
                    right: builder.ref(right_domain, row[right]),
                },
                proofs=("C2_native_dictionary_metadata_derivation",)
                if table == "conversation_dictionary_attachments"
                else (),
            )

        builder.handle(table, junction)
    for table in ("conversation_dictionary_unresolved", "conversation_local_marks"):

        def local(row, table=table):
            builder.emit(
                table,
                row,
                row
                | {
                    "conversation_id": builder.ref(
                        "conversation", row["conversation_id"]
                    )
                },
                proofs=("C2_native_dictionary_metadata_derivation",)
                if table == "conversation_dictionary_unresolved"
                else (),
            )

        builder.handle(table, local)


def _messages(builder):
    def message(row):
        _revision(row)
        _ancestor_chain(builder, "messages", row, "parent_message_id")
        for column in ("parent_message_id", "variant_of"):
            identity = row[column]
            if identity is None:
                continue
            other = builder.by_id["messages"].get(identity)
            if other is None:
                raise _UnprovedReference("message", identity)
            if (
                other["conversation_id"] != row["conversation_id"]
                or identity == row["id"]
            ):
                raise ValueError("invalid_message_branch")
            if column == "variant_of" and (
                other["variant_of"] is not None
                or other["parent_message_id"] != row["parent_message_id"]
            ):
                raise ValueError("invalid_message_variant")
        if row["image_data"] is not None and type(row["image_data"]) is not bytes:
            raise ValueError("invalid_message_image")
        root = row["variant_of"] or row["id"]
        variants = [
            other
            for other in builder.source["messages"]
            if other["id"] == root or other["variant_of"] == root
        ]
        live = [other for other in variants if not other["deleted"]]
        if (
            len(live) > 1
            and sum(other["is_selected_variant"] == 1 for other in live) != 1
        ):
            raise ValueError("invalid_variant_selection")
        values = {
            key: value for key, value in row.items() if key not in _MESSAGE_PENDING
        }
        values.update(
            id=builder.ref("message", row["id"]),
            conversation_id=builder.ref("conversation", row["conversation_id"]),
            parent_message_id=builder.ref(
                "message", row["parent_message_id"], optional=True
            ),
            variant_of=builder.ref("message", row["variant_of"], optional=True),
        )
        builder.emit(
            "messages",
            row,
            values,
            pending=_MESSAGE_PENDING,
            proofs=("C2_message_fields",),
        )

    builder.handle("messages", message)

    def attachment(row):
        if (
            type(row["position"]) is not int
            or row["position"] < 1
            or type(row["data"]) is not bytes
            or not row["mime_type"]
        ):
            raise ValueError("invalid_message_attachment")
        builder.emit(
            "message_attachments",
            row,
            row | {"message_id": builder.ref("message", row["message_id"])},
        )

    builder.handle("message_attachments", attachment)


def _conversations(builder):
    def structure(row):
        references = {}
        for column, domain in (
            ("root_id", "conversation"),
            ("parent_conversation_id", "conversation"),
            ("forked_from_message_id", "message"),
            ("active_leaf_message_id", "message"),
            ("active_leaf_before_message_id", "message"),
            ("summary_boundary_message_id", "message"),
            ("character_id", "character"),
            ("workspace_id", "workspace"),
        ):
            references[column] = builder.ref(domain, row[column], optional=True)
        builder.inputs.append(
            _ConversationStructure(
                row["id"],
                builder.ref("conversation", row["id"]),
                tuple(references.items()),
            )
        )

    builder.handle("conversations", structure)


def _core(builder):
    _entities(builder)
    _folders(builder)
    _associations(builder)
    _messages(builder)
    _conversations(builder)
