"""Fixed logical actor graphs; physical assets and live authority stay unproved."""

from dataclasses import dataclass, field, replace

from .mobile_captured_authority import _AuthorityReference
from .mobile_content_graph import (
    _ContentBlocker,
    _ContentRow,
    _GraphBuilder,
    _NativeConflict,
    _UnprovedReference,
)
from .mobile_content_settings import _MESSAGE_FIELDS, _json, _SettingsSuccessor
from .mobile_content_workflows import _integer, _shape, _text, _WorkflowPlan
from .mobile_workspace_plan import _object

_TABLES = (
    "actor_portable_identities",
    "buddy_profiles",
    "buddy_visual_bindings",
    "persona_visual_assets",
    "persona_visual_bindings",
    "persona_visual_pack_versions",
    "persona_visual_packs",
    "visual_identity_assets",
    "visual_identity_bindings",
    "visual_identity_pack_versions",
    "visual_identity_packs",
)


@dataclass(frozen=True)
class _ActorPlan:
    workflows: _WorkflowPlan = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    logical_rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    successors: tuple[_SettingsSuccessor, ...]
    discharged: tuple[_SettingsSuccessor, ...]
    field_ownership: tuple[tuple[str, tuple[str, ...]], ...]
    authority_references: tuple[_AuthorityReference, ...] = field(repr=False)
    pending_handlers: tuple[str, ...] = ("C3", "T", "S", "A-runs")

    @property
    def identities(self):
        return self.workflows.identities

    @property
    def authority_slots(self):
        return self.workflows.settings.authority_slots

    @property
    def dictionary_derivations(self):
        return self.workflows.settings.dictionary_derivations

    def require_complete(self) -> None:
        raise ValueError("mobile_content_handlers_pending")


def _actor(builder, kind, identity):
    from tldw_chatbook.Actor_Packs.repository import _actor_id

    if kind == "character":
        if (
            type(identity) is not str
            or not identity.isascii()
            or not identity.isdigit()
        ):
            raise ValueError("invalid_character_identity")
        original = int(identity)
        _actor_id(kind, original)
        return str(builder.ref(kind, original))
    _actor_id(kind, identity)
    builder.dependencies.append(("persona", identity))
    return identity


def _record(builder, table, identity, domain):
    value = builder.by_id[table].get(identity)
    if value is None:
        raise _UnprovedReference(domain, identity)
    return value


def _relation(builder, row, *, persona, binding=False):
    prefix = "persona_visual" if persona else "visual_identity"
    domain = "persona_visual" if persona else "visual"
    pack = _record(builder, prefix + "_packs", row["pack_id"], domain + "_pack")
    version_key = "active_version_id" if binding else "pack_version_id"
    version = _record(
        builder, prefix + "_pack_versions", row[version_key], domain + "_version"
    )
    if version["pack_id"] != pack["id"]:
        raise ValueError("visual_version_pack_mismatch")
    if (
        persona
        and binding
        and row["status"] == "active"
        and pack["active_version_id"] != version["id"]
    ):
        raise ValueError("persona_active_version_mismatch")
    return pack, version


def _visual_context(builder, raw, *, digest=None):
    from tldw_chatbook.Character_Chat.artwork_attribution import artwork_context

    value = _object(raw)
    _shape(
        value,
        (),
        (
            "source_id",
            "source",
            "provenance",
            "license_value",
            "pack_content_sha256",
            "profile_pack_id",
            "forked_from_pack_id",
            "forked_from_version_id",
            "retained_asset_id",
            "publication_source",
            "tldw/artwork",
            "tldw/buddy_conversion",
            "date",
            "source_portrait",
            "strategy",
            "tool",
            "visual_direction",
        ),
    )
    artwork_context(value, expected_sha256=digest)
    mapped = dict(value)
    for key, domain in (
        ("forked_from_pack_id", "visual_pack"),
        ("forked_from_version_id", "visual_version"),
        ("retained_asset_id", "visual_asset"),
    ):
        if key in value:
            _integer(value[key], 1)
            mapped[key] = builder.ref(domain, value[key])
    for key in (
        "source_id",
        "profile_pack_id",
        "publication_source",
        "date",
        "strategy",
        "tool",
        "visual_direction",
    ):
        if key in value:
            _text(value[key])
    for key in ("source", "provenance", "license_value"):
        if key in value:
            _text(value[key], limit=256)
    if "pack_content_sha256" in value:
        from tldw_chatbook.Character_Chat.visual_identity import _lower_sha256

        _lower_sha256(value["pack_content_sha256"])
    if "source_portrait" in value:
        from tldw_chatbook.Character_Chat.visual_identity import _lower_sha256

        _shape(value["source_portrait"], ("filename", "sha256"))
        _text(value["source_portrait"]["filename"])
        _lower_sha256(value["source_portrait"]["sha256"])
    if ("forked_from_pack_id" in value) != ("forked_from_version_id" in value):
        raise ValueError("incomplete_visual_lineage")
    if "forked_from_pack_id" in value:
        version = _record(
            builder,
            "visual_identity_pack_versions",
            value["forked_from_version_id"],
            "visual_version",
        )
        if version["pack_id"] != value["forked_from_pack_id"]:
            raise ValueError("visual_lineage_mismatch")
    return raw if mapped == value else _json(mapped)


def _visual_asset(row):
    from tldw_chatbook.Character_Chat.visual_identity import _validate_manifest_asset

    _integer(row["is_animated"], 0, 1)
    _integer(row["deleted"], 0, 1)
    return _validate_manifest_asset(
        {
            "expression_key": row["expression_key"],
            "original_label": row["original_expression_key"],
            "display_label": row["display_label"],
            "storage_relpath": row["storage_relpath"],
            "content_type": row["content_type"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "is_animated": bool(row["is_animated"]),
            "frame_count": 1 if row["frame_count"] is None else row["frame_count"],
            "duration_ms": row["duration_ms"],
        }
    )


def _logical_actors(builder):
    """Transform fixed native rows and typed maps, without issuing any proof."""
    from tldw_chatbook.Actor_Packs.repository import _decode_identity
    from tldw_chatbook.DB.VisualIdentity_DB import LOCAL_OWNER_ID
    from tldw_chatbook.Persona_Visual import repository as native

    def emit(table, row, values, proofs=()):
        if any(domain == "persona" for domain, _ in builder.dependencies):
            proofs += ("authority:persona",)
        builder.emit(table, row, values, proofs=proofs)

    def portable(row):
        _decode_identity(row)
        for key in ("created_at", "updated_at"):
            _text(row[key])
        values = row | {
            "local_actor_id": _actor(builder, row["actor_kind"], row["local_actor_id"])
        }
        _decode_identity(values)
        if any(
            other["portable_uuid"] == row["portable_uuid"]
            or (other["actor_kind"], other["local_actor_id"])
            == (values["actor_kind"], values["local_actor_id"])
            for other in builder.canonical["actor_portable_identities"]
        ):
            raise _NativeConflict()
        emit("actor_portable_identities", row, values)

    builder.handle("actor_portable_identities", portable)

    def buddy(row):
        _text(row["id"], limit=128)
        _text(row["name"], limit=256)
        _integer(row["version"], 1)
        native._db_enum(row["status"], {"active", "archived", "deleted"})
        for key in ("created_at", "updated_at"):
            _text(row[key])
        _text(row["source_key"], optional=True)
        if row["source_key"] is not None and any(
            other["source_key"] == row["source_key"]
            for other in builder.canonical["buddy_profiles"]
        ):
            raise _NativeConflict()
        emit("buddy_profiles", row, row | {"id": builder.ref("buddy", row["id"])})

    builder.handle("buddy_profiles", buddy)

    def persona_asset(row):
        native._decode_asset(row)
        _relation(builder, row, persona=True)
        values = row | {
            "id": builder.ref("persona_visual_asset", row["id"]),
            "pack_id": builder.ref("persona_visual_pack", row["pack_id"]),
            "pack_version_id": builder.ref(
                "persona_visual_version", row["pack_version_id"]
            ),
        }
        native._decode_asset(values)
        emit("persona_visual_assets", row, values, ("physical:persona_visual_asset",))

    builder.handle("persona_visual_assets", persona_asset)

    def persona_version(row):
        assets = [
            a
            for a in builder.source["persona_visual_assets"]
            if a["pack_version_id"] == row["id"]
        ]
        if any(a["pack_id"] != row["pack_id"] for a in assets):
            raise ValueError("persona_asset_pack_mismatch")
        native._decode_version(row, tuple(native._decode_asset(a) for a in assets))
        values = row | {
            "id": builder.ref("persona_visual_version", row["id"]),
            "pack_id": builder.ref("persona_visual_pack", row["pack_id"]),
        }
        for asset in assets:
            builder.ref("persona_visual_asset", asset["id"])
        native._decode_version(
            values,
            tuple(
                replace(
                    native._decode_asset(a),
                    id=builder.ref("persona_visual_asset", a["id"]),
                    pack_id=values["pack_id"],
                    pack_version_id=values["id"],
                )
                for a in assets
            ),
        )
        emit(
            "persona_visual_pack_versions",
            row,
            values,
            ("physical:persona_visual_manifest",),
        )

    builder.handle("persona_visual_pack_versions", persona_version)

    def persona_pack(row):
        # NULL is native unbound history; no active graph decoder may invent a version.
        if row["active_version_id"] is None:
            native._db_positive_int(row["id"])
            native._db_text(row["title"], 256)
            native._db_text(row["description"], 4096, allow_empty=True)
            native._db_enum(row["status"], {"active", "archived", "deleted"})
            native._db_positive_int(row["version"])
            native._validate_stored_source_context(row["source_context_json"])
            kind = native._db_text(row["source_kind"], 64)
            if native._SOURCE_KIND_PATTERN.fullmatch(kind) is None:
                raise ValueError("invalid_source_kind")
            for key in ("created_at", "updated_at"):
                native._db_timestamp(row[key])
        else:
            native._decode_pack(row)
            version = _record(
                builder,
                "persona_visual_pack_versions",
                row["active_version_id"],
                "persona_visual_version",
            )
            if version["pack_id"] != row["id"]:
                raise ValueError("persona_active_version_mismatch")
        values = row | {
            "id": builder.ref("persona_visual_pack", row["id"]),
            "active_version_id": builder.ref(
                "persona_visual_version", row["active_version_id"], optional=True
            ),
        }
        emit("persona_visual_packs", row, values)

    builder.handle("persona_visual_packs", persona_pack)

    def persona_binding(row, *, buddy_owner):
        table = "buddy_visual_bindings" if buddy_owner else "persona_visual_bindings"
        shape = (
            row | {"persona_id": None, "persona_revision": 0} if buddy_owner else row
        )
        native._decode_binding(shape)
        _relation(builder, row, persona=True, binding=True)
        values = row | {
            "id": builder.ref(
                "buddy_visual_binding" if buddy_owner else "persona_visual_binding",
                row["id"],
            ),
            "pack_id": builder.ref("persona_visual_pack", row["pack_id"]),
            "active_version_id": builder.ref(
                "persona_visual_version", row["active_version_id"]
            ),
        }
        if buddy_owner:
            owner = _record(builder, "buddy_profiles", row["buddy_id"], "buddy")
            values["buddy_id"] = builder.ref("buddy", row["buddy_id"])
            if row["status"] == "active" and (
                owner["version"] != row["buddy_revision"] or owner["status"] != "active"
            ):
                raise _UnprovedReference(
                    "buddy_revision", (row["buddy_id"], row["buddy_revision"])
                )
            proofs = ()
            owner_key = "buddy_id"
        else:
            values["persona_id"] = _actor(builder, "persona", row["persona_id"])
            proofs = ("authority:persona_revision",)
            owner_key = "persona_id"
        if row["status"] == "active" and any(
            other["status"] == "active" and other[owner_key] == values[owner_key]
            for other in builder.canonical[table]
        ):
            raise _NativeConflict()
        native._decode_binding(
            values | {"persona_id": None, "persona_revision": 0}
            if buddy_owner
            else values
        )
        emit(table, row, values, proofs)

    builder.handle(
        "persona_visual_bindings", lambda row: persona_binding(row, buddy_owner=False)
    )
    builder.handle(
        "buddy_visual_bindings", lambda row: persona_binding(row, buddy_owner=True)
    )

    def visual(row, table):
        from tldw_chatbook.Character_Chat import visual_identity as contract

        if (
            type(row["owner_user_id"]) is not int
            or row["owner_user_id"] != LOCAL_OWNER_ID
        ):
            raise _UnprovedReference("visual_local_owner", row["owner_user_id"])
        for key in ("created_at", "updated_at"):
            if key in row:
                _text(row[key])
        domain = {
            "visual_identity_packs": "visual_pack",
            "visual_identity_pack_versions": "visual_version",
            "visual_identity_assets": "visual_asset",
            "visual_identity_bindings": "visual_binding",
        }[table]
        values = row | {"id": builder.ref(domain, row["id"])}
        proofs = ()
        if table == "visual_identity_packs":
            _integer(row["version"], 1)
            native._db_enum(row["status"], {"active", "archived", "deleted"})
            for key in ("title", "description", "source_kind"):
                _text(row[key])
            contract._valid_expression_key(row["default_expression_key"])
            values["source_context_json"] = _visual_context(
                builder, row["source_context_json"]
            )
            values["active_version_id"] = builder.ref(
                "visual_version", row["active_version_id"], optional=True
            )
            if row["active_version_id"] is not None:
                version = _record(
                    builder,
                    "visual_identity_pack_versions",
                    row["active_version_id"],
                    "visual_version",
                )
                if version["pack_id"] != row["id"]:
                    raise ValueError("visual_active_version_mismatch")
        elif table == "visual_identity_pack_versions":
            _integer(row["version_number"], 1)
            values["pack_id"] = builder.ref("visual_pack", row["pack_id"])
            manifest = _object(row["manifest_json"])
            _shape(
                manifest,
                (
                    "schema_id",
                    "pack_id",
                    "license",
                    "default_expression_key",
                    "pack_content_sha256",
                    "assets",
                ),
                ("title", "source_server_commit"),
            )
            projected_assets = []
            for asset in manifest["assets"]:
                _shape(
                    asset,
                    (
                        "expression_key",
                        "original_label",
                        "display_label",
                        "content_type",
                        "bytes",
                        "width",
                        "height",
                        "sha256",
                        "is_animated",
                        "frame_count",
                    ),
                    (
                        "duration_ms",
                        "storage_relpath",
                        "relative_filename",
                        "generation",
                    ),
                )
                if "generation" in asset:
                    asset = asset | {
                        "generation": _object(
                            _visual_context(builder, _json(asset["generation"]))
                        )
                    }
                projected_assets.append(asset)
            # Reserved bundles require physical directory measurement; retain that exact dependency.
            directory_bytes = None
            if manifest.get("pack_id") == contract.SAMIRA_PACK_ID:
                raise _UnprovedReference("visual_bundle_directory", row["id"])
            parsed = contract.validate_visual_identity_manifest(
                manifest, directory_bytes=directory_bytes
            )
            if projected_assets != manifest["assets"]:
                projected = manifest | {"assets": projected_assets}
                if contract.validate_visual_identity_manifest(projected) != parsed:
                    raise ValueError("visual_manifest_identity_changed")
                values["manifest_json"] = _json(projected)
            if parsed.default_expression_key != row["default_expression_key"]:
                raise ValueError("visual_default_mismatch")
            assets = [
                a
                for a in builder.source["visual_identity_assets"]
                if a["pack_version_id"] == row["id"] and not a["deleted"]
            ]
            if len(assets) != len(parsed.assets):
                raise ValueError("visual_manifest_asset_mismatch")
            declared = {a.expression_key: a for a in parsed.assets}
            for asset in assets:
                if asset["pack_id"] not in (None, row["pack_id"]) or declared.get(
                    asset["expression_key"]
                ) != _visual_asset(asset):
                    raise ValueError("visual_manifest_asset_mismatch")
                builder.ref("visual_asset", asset["id"])
        elif table == "visual_identity_assets":
            _visual_asset(row)
            version = _record(
                builder,
                "visual_identity_pack_versions",
                row["pack_version_id"],
                "visual_version",
            )
            if row["pack_id"] is not None and row["pack_id"] != version["pack_id"]:
                raise ValueError("visual_asset_pack_mismatch")
            values["pack_id"] = builder.ref(
                "visual_pack", row["pack_id"], optional=True
            )
            values["pack_version_id"] = builder.ref(
                "visual_version", row["pack_version_id"]
            )
            _text(row["source_filename"])
            if row["preview_relpath"] is not None:
                contract._safe_relative_parts(row["preview_relpath"])
            values["source_context_json"] = _visual_context(
                builder, row["source_context_json"], digest=row["sha256"]
            )
            proofs = ("physical:visual_asset",)
        else:
            _integer(row["version"], 1)
            native._db_enum(row["status"], {"active", "deleted"})
            _relation(builder, row, persona=False, binding=True)
            values["actor_id"] = _actor(builder, row["actor_kind"], row["actor_id"])
            values["pack_id"] = builder.ref("visual_pack", row["pack_id"])
            values["active_version_id"] = builder.ref(
                "visual_version", row["active_version_id"]
            )
            if row["status"] == "active" and any(
                other["status"] == "active"
                and (other["owner_user_id"], other["actor_kind"], other["actor_id"])
                == (LOCAL_OWNER_ID, values["actor_kind"], values["actor_id"])
                for other in builder.canonical[table]
            ):
                raise _NativeConflict()
        emit(table, row, values, proofs)

    for table in (
        "visual_identity_assets",
        "visual_identity_pack_versions",
        "visual_identity_packs",
        "visual_identity_bindings",
    ):
        builder.handle(table, lambda row, table=table: visual(row, table))

    _close_actor_uniqueness(builder)


def _close_actor_uniqueness(builder):
    """Native decimal aliases may collapse despite injective typed I maps."""
    groups = {}
    for row in builder.rows:
        if row.table not in _TABLES:
            continue
        values = dict(row.values)
        keys = [
            (
                row.table,
                "primary",
                tuple(values[c] for c in builder.specs[row.table].identity_columns),
            )
        ]
        if row.table.endswith("_bindings") and values["status"] == "active":
            columns = {
                "buddy_visual_bindings": ("buddy_id",),
                "persona_visual_bindings": ("persona_id",),
                "visual_identity_bindings": ("owner_user_id", "actor_kind", "actor_id"),
            }[row.table]
            keys.append((row.table, "active_owner", tuple(values[c] for c in columns)))
        for key in keys:
            groups.setdefault(key, []).append(row)
    blocked = {
        (row.table, row.original)
        for group in groups.values()
        if len(group) > 1
        for row in group
    }
    for table, original in sorted(blocked):
        builder.blockers.append(
            _ContentBlocker(table, original, "native_unique_conflict")
        )
    builder.rows = [
        row for row in builder.rows if (row.table, row.original) not in blocked
    ]


def _field_ownership(specs):
    from .mobile_content_settings import _TABLES as settings
    from .mobile_content_workflows import _TABLES as workflows

    ownership = tuple(
        (table, _MESSAGE_FIELDS if table == "messages" else specs[table].columns)
        for table in settings + workflows + _TABLES
    )
    if (
        len({table for table, _ in ownership}) != len(ownership)
        or sum(len(columns) for _, columns in ownership) != 246
    ):
        raise ValueError("mobile_c2_ownership_mismatch")
    return ownership


def _close_dictionary_derivations(builder, settings):
    """Discharge only the actual retained native metadata/index derivations."""
    from .mobile_content_settings import _dictionary_shape

    proof = "C2_native_dictionary_metadata_derivation"
    originals = {
        r.original[0]: dict(r.values)
        for r in settings.source_rows
        if r.table == "conversations"
    }
    conversations = {
        r.original[0]: dict(r.values)
        for r in builder.rows
        if r.table == "conversations"
    }
    derivations = {d.conversation_id: d for d in settings.dictionary_derivations}
    attachments = "conversation_dictionary_attachments"
    unresolved = "conversation_dictionary_unresolved"

    def indices(rows, conversation):
        edges, missing = [], False
        for row in rows:
            if row.table not in {attachments, unresolved}:
                continue
            value = dict(row.values)
            if value["conversation_id"] == conversation:
                if row.table == attachments:
                    edges.append(value["dictionary_id"])
                else:
                    missing = True
        return tuple(sorted(edges)), missing

    verified = set()
    for identity, mapped in conversations.items():
        derivation = derivations.get(identity)
        try:
            if derivation is None or mapped["id"] != derivation.mapped_conversation_id:
                raise ValueError("dictionary_derivation_missing")
            before = _dictionary_shape(originals[identity]["metadata"])
            after = _dictionary_shape(mapped["metadata"])
            if (
                before
                != (
                    derivation.original_attachments,
                    derivation.original_unresolved,
                    derivation.original_active,
                )
                or after
                != (
                    derivation.mapped_attachments,
                    derivation.mapped_unresolved,
                    derivation.mapped_active,
                )
                or indices(settings.graph.source_rows, identity) != before[:2]
                or indices(builder.rows, mapped["id"]) != after[:2]
            ):
                raise ValueError("dictionary_derivation_mismatch")
        except (ValueError, KeyError, TypeError):
            builder.blockers.append(
                _ContentBlocker(
                    "conversations",
                    (identity,),
                    "native_dictionary_derivation_mismatch",
                )
            )
        else:
            verified.add(identity)
    builder.rows = [
        replace(
            row, required_proofs=tuple(p for p in row.required_proofs if p != proof)
        )
        if row.table in {attachments, unresolved} and row.original[0] in verified
        else row
        for row in builder.rows
    ]


def _plan_content_actors(reads):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_content_native_snapshot_required")
    reads.check()
    workflows = reads.content_workflow_plan()
    builder = _GraphBuilder(reads, workflows.identities, _TABLES)
    ownership = _field_ownership(builder.specs)
    builder.rows.extend(workflows.rows)
    builder.blockers.extend(workflows.blockers)
    _logical_actors(builder)
    references = {}
    logical = tuple(r for r in builder.rows if r.table in _TABLES)
    for row in logical:
        original = dict(
            next(
                r.values
                for r in builder.retained()
                if r.table == row.table and r.original == row.original
            )
        )
        for proof in row.required_proofs:
            domain = proof.split(":", 1)[1]
            identity = (
                (original["persona_id"], original["persona_revision"])
                if domain == "persona_revision"
                else next(
                    identity for name, identity in row.dependencies if name == "persona"
                )
                if domain == "persona"
                else row.original[0]
            )
            if domain == "persona":
                try:
                    references[identity] = reads.authority_reference(
                        "persona", identity
                    )
                    continue
                except ValueError:
                    pass
            builder.blockers.append(
                _ContentBlocker(
                    row.table, row.original, "dependency_unproven", domain, identity
                )
            )
    accepted = {(r.table, r.original) for r in builder.rows}
    successors, discharged = [], []
    domains = {
        "character": "character_cards",
        "discovery_character": "character_cards",
        "character_expression": "character_expression_images",
        "visual_pack": "visual_identity_packs",
        "visual_version": "visual_identity_pack_versions",
        "visual_asset": "visual_identity_assets",
    }
    for item in workflows.successors:
        if item.handler != "C2c":
            successors.append(item)
            continue
        if item.domain in {"persona", "discovery_persona"}:
            try:
                references[item.original_id] = reads.authority_reference(
                    "persona", item.original_id
                )
            except ValueError:
                builder.blockers.append(
                    _ContentBlocker(
                        item.table,
                        item.original,
                        "dependency_unproven",
                        "persona",
                        item.original_id,
                    )
                )
                continue
        elif item.domain in domains:
            if (domains[item.domain], (item.original_id,)) not in accepted:
                builder.blockers.append(
                    _ContentBlocker(
                        item.table,
                        item.original,
                        "dependency_blocked",
                        item.domain,
                        item.original_id,
                    )
                )
                continue
        elif item.domain == "remote_assistant":
            builder.blockers.append(
                _ContentBlocker(
                    item.table,
                    item.original,
                    "dependency_unproven",
                    item.domain,
                    item.original_id,
                )
            )
            continue
        else:
            raise ValueError("unknown_actor_successor")
        discharged.append(item)
    # Native historical emotes bind one exact graph, never the current actor binding.
    original_messages = {
        r.original[0]: dict(r.values)
        for r in workflows.settings.source_rows
        if r.table == "messages"
    }
    for identity, row in original_messages.items():
        if not any(
            s.handler == "C2c" and s.table == "messages" and s.original == (identity,)
            for s in workflows.successors
        ):
            continue
        from tldw_chatbook.Chat.message_metadata import MessageMetadata

        emote = MessageMetadata.from_json(row["metadata_json"]).character_emote
        if emote is None:
            continue
        asset = builder.by_id["visual_identity_assets"].get(emote.asset_id)
        version = builder.by_id["visual_identity_pack_versions"].get(
            emote.pack_version_id
        )
        if (
            asset is None
            or version is None
            or asset["pack_id"] != emote.pack_id
            or version["pack_id"] != emote.pack_id
            or asset["pack_version_id"] != emote.pack_version_id
            or asset["expression_key"] != emote.expression_key
            or asset["deleted"]
        ):
            builder.blockers.append(
                _ContentBlocker("messages", (identity,), "native_emote_graph_mismatch")
            )
    _close_dictionary_derivations(builder, workflows.settings)
    blocked = {(b.table, b.original) for b in builder.blockers}
    builder.rows = [r for r in builder.rows if (r.table, r.original) not in blocked]
    builder.close_dependencies()
    accepted = {(r.table, r.original) for r in builder.rows}
    closed = {(s.table, s.original, "C2c:" + s.domain) for s in discharged}
    rows = tuple(
        replace(
            r,
            required_proofs=tuple(
                p
                for p in r.required_proofs
                if (r.table, r.original, p) not in closed
                and not (
                    p == "authority:persona"
                    and all(
                        identity in references
                        for domain, identity in r.dependencies
                        if domain == "persona"
                    )
                )
            ),
        )
        for r in builder.rows
    )
    for row in rows:
        if row.table in dict(ownership) and set(dict(row.values)) | set(
            row.pending_columns
        ) != set(builder.specs[row.table].columns):
            raise ValueError("mobile_c2_field_ownership_mismatch")
    reads.check()
    return _ActorPlan(
        workflows,
        rows,
        logical,
        builder.retained(),
        tuple(builder.blockers),
        tuple(s for s in successors if (s.table, s.original) in accepted),
        tuple(s for s in discharged if (s.table, s.original) in accepted),
        ownership,
        tuple(references.values()),
    )
