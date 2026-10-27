"""Native captured actor projections are logical history, never authority."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def _seed_actors(connection, purpose):
    _seed_core(connection, purpose)
    connection.execute(
        "INSERT INTO buddy_profiles(id,name,source_key,version) VALUES (?,?,NULL,3)",
        ("buddy-one", "Historical Buddy"),
    )
    if purpose == "legacy":
        connection.execute(
            "INSERT INTO actor_portable_identities(actor_kind,local_actor_id,portable_uuid,version) "
            "VALUES ('character','1','f476a5e1-01cc-4389-9daf-21c619516846',7)"
        )


def test_captured_actors_map_native_identity_and_complete_c2_ownership(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Actor_Packs.repository import _decode_identity

    with content_reads(tmp_path, monkeypatch, _seed_actors) as reads:
        build = getattr(reads, "content_actor_plan", None)
        assert callable(build), "captured native reader needs logical C2c planning"
        plan = build()
        assert not plan.blockers
        actor = next(
            dict(r.values) for r in plan.rows if r.table == "actor_portable_identities"
        )
        native = _decode_identity(actor)
        assert native.local_actor_id == str(plan.identities.mapped("character", 1))
        assert native.portable_uuid == "f476a5e1-01cc-4389-9daf-21c619516846"
        buddy = next(dict(r.values) for r in plan.rows if r.table == "buddy_profiles")
        assert buddy["id"] == plan.identities.mapped("buddy", "buddy-one")
        assert buddy["version"] == 3
        assert sum(len(columns) for _, columns in plan.field_ownership) == 246
        assert len(plan.source_rows) == 2
        assert not any(s.handler in {"C2b", "C2c"} for s in plan.successors)
        assert plan.pending_handlers == ("C3", "T", "S", "A-runs")
        for candidate in (
            plan,
            replace(plan, rows=(), blockers=(), successors=(), pending_handlers=()),
        ):
            with pytest.raises(ValueError, match="mobile_content_handlers_pending"):
                candidate.require_complete()


def _seed_visuals(connection, purpose):
    import hashlib
    import json

    from Tests.Persona_Visual.test_persona_visual_repository import _valid_manifest
    from tldw_chatbook.Character_Chat.visual_identity import compute_pack_content_sha256

    _seed_actors(connection, purpose)
    if purpose != "legacy":
        return
    manifest = json.dumps(_valid_manifest())
    connection.execute(
        "INSERT INTO persona_visual_packs(id,title,active_version_id) VALUES (111,'Sprite history',NULL)"
    )
    connection.execute(
        "INSERT INTO persona_visual_pack_versions(id,pack_id,version_number,renderer_type,manifest_version,manifest_json,manifest_sha256,storage_relpath) VALUES (111,111,1,'sprite_frames',1,?,?,?)",
        (
            manifest,
            hashlib.sha256(manifest.encode()).hexdigest(),
            "persona_visual/history/manifest.json",
        ),
    )
    connection.execute(
        "UPDATE persona_visual_packs SET active_version_id=111 WHERE id=111"
    )
    connection.execute(
        "INSERT INTO persona_visual_assets(id,pack_id,pack_version_id,asset_key,role,storage_relpath,mime_type,bytes,sha256,width,height,frame_count) VALUES (111,111,111,'idle','frame','persona_visual/history/idle.png','image/png',12,?,4,5,1)",
        ("b" * 64,),
    )
    connection.execute(
        "INSERT INTO buddy_visual_bindings(id,buddy_id,buddy_revision,pack_id,active_version_id) VALUES (111,'buddy-one',3,111,111)"
    )
    connection.execute(
        "INSERT INTO persona_visual_bindings(id,persona_id,persona_revision,pack_id,active_version_id) VALUES (111,'persona-1',7,111,111)"
    )
    connection.execute(
        "INSERT INTO actor_portable_identities(actor_kind,local_actor_id,portable_uuid) VALUES ('persona','persona-1','497d518c-2f9e-40a7-a408-72a8442f20a7')"
    )
    asset = {
        "expression_key": "happy",
        "original_label": "happy",
        "display_label": "Happy",
        "storage_relpath": "history/happy.png",
        "content_type": "image/png",
        "bytes": 12,
        "sha256": "c" * 64,
        "width": 4,
        "height": 5,
        "is_animated": False,
        "frame_count": 1,
        "duration_ms": None,
    }
    visual = {
        "schema_id": "tldw.visual_identity_pack/v1",
        "pack_id": "private.history",
        "title": "History",
        "license": "CC0",
        "default_expression_key": "happy",
        "assets": [asset],
    }
    visual["pack_content_sha256"] = compute_pack_content_sha256(visual)
    connection.execute(
        "INSERT INTO visual_identity_packs(id,owner_user_id,title,default_expression_key) VALUES (221,0,'Expression history','happy')"
    )
    connection.execute(
        "INSERT INTO visual_identity_pack_versions(id,pack_id,owner_user_id,version_number,default_expression_key,manifest_json) VALUES (221,221,0,1,'happy',?)",
        (json.dumps(visual),),
    )
    connection.execute(
        "UPDATE visual_identity_packs SET active_version_id=221 WHERE id=221"
    )
    connection.execute(
        "INSERT INTO visual_identity_assets(id,owner_user_id,pack_id,pack_version_id,expression_key,original_expression_key,display_label,source_filename,storage_relpath,content_type,bytes,sha256,width,height,frame_count) VALUES (221,0,221,221,'happy','happy','Happy','happy.png','history/happy.png','image/png',12,?,4,5,1)",
        ("c" * 64,),
    )
    connection.execute(
        "INSERT INTO visual_identity_bindings(id,owner_user_id,actor_kind,actor_id,pack_id,active_version_id) VALUES (221,0,'character','11',221,221)"
    )


def _persona_seed(tmp_path):
    def native_seed(db, purpose):
        import json

        from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
        from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

        config = read_profile_config(tmp_path / "profile" / "config.toml")
        path = user_data_dir(config) / "tldw_chatbook_personas.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "id": "persona-1",
                            "name": "Private persona",
                            "version": 7,
                            "is_active": True,
                            "policy_rules": [],
                        }
                    ]
                }
            )
        )
        path.chmod(0o600)

    return native_seed


def _logical_inputs():
    """Native rows with explicit data-only maps; no capture or I authority."""
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.mobile_content_actors import _TABLES
    from tldw_chatbook.Backup_Recovery.mobile_content_graph import (
        _GraphBuilder,
        _UnprovedReference,
    )
    from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _DOMAINS
    from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES as specs
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    class LogicalInputs(_GraphBuilder):
        def __init__(self, source):
            self.specs = {s.table: s for s in specs if s.family == "chat"}
            self.source = source
            self.canonical = {t: [] for t in _TABLES}
            self.by_id = {
                t: {r["id"]: r for r in rows}
                for t, rows in source.items()
                if "id" in self.specs[t].identity_columns
            }
            self.rows, self.blockers, self.dependencies = [], [], []
            self.maps = {("character", 1): 1001, ("character", 11): 1011}
            for d in _DOMAINS:
                if d.family == "chat" and not d.repeated and d.table in source:
                    for row in source[d.table]:
                        identity = row[self.specs[d.table].identity_columns[0]]
                        self.maps[d.name, identity] = (
                            identity + 1000
                            if type(identity) is int
                            else "mapped-" + identity
                        )

        def ref(self, domain, identity, *, optional=False):
            if identity is None and optional:
                return None
            if (domain, identity) not in self.maps:
                raise _UnprovedReference(domain, identity)
            self.dependencies.append((domain, identity))
            return self.maps[domain, identity]

    with closing(CharactersRAGDB(":memory:", "logical-input")) as db:
        db.add_conversation({"id": "conversation", "title": "Logical input"})
        with db.transaction() as cursor:
            _seed_visuals(cursor.connection, "legacy")
        source = {
            t: [
                dict(r)
                for r in db.get_connection().execute('SELECT * FROM "' + t + '"')
            ]
            for t in _TABLES
        }
    return LogicalInputs(source)


def test_all_logical_actor_tables_reach_native_consumers_with_pending_proofs():
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.mobile_content_actors import (
        _TABLES,
        _logical_actors,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

    data = _logical_inputs()
    _logical_actors(data)
    assert not data.blockers
    assert {r.table for r in data.rows} == set(_TABLES)
    assert sum(len(data.specs[t].columns) for t in _TABLES) == 116
    assert {p for r in data.rows for p in r.required_proofs} == {
        "physical:persona_visual_asset",
        "physical:persona_visual_manifest",
        "authority:persona_revision",
        "authority:persona",
        "physical:visual_asset",
    }
    with closing(CharactersRAGDB(":memory:", "actor-consumer")) as db:
        with db.transaction() as cursor:
            cursor.execute("PRAGMA defer_foreign_keys=ON")
            for row in data.rows:
                values = dict(row.values)
                columns = ",".join('"' + key + '"' for key in values)
                cursor.execute(
                    'INSERT INTO "'
                    + row.table
                    + '" ('
                    + columns
                    + ") VALUES ("
                    + ",".join("?" for _ in values)
                    + ")",
                    tuple(values.values()),
                )
        persona = PersonaVisualRepository(db)
        graph = persona.get_active_persona_pack("persona-1")
        assert graph.identity.pack_id == data.maps["persona_visual_pack", 111]
        buddy = persona.get_active_persona_pack(
            None, buddy_id=data.maps["buddy", "buddy-one"]
        )
        assert buddy.identity.buddy_revision == 3
        visual = VisualIdentityRepository(db).get_active_actor_pack(
            "character", data.maps["character", 11]
        )
        assert visual["assets"][0]["id"] == data.maps["visual_asset", 221]
        assert (
            visual["version"]["manifest_json"]
            == data.source["visual_identity_pack_versions"][0]["manifest_json"]
        )


def test_asset_populated_native_admission_refuses_without_capture(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import (
        prepare,
        prepared,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )

    native_seed = mobile_consolidation_fixture._seed

    def seed(stores, **kwargs):
        def populate(db, legacy):
            with db.transaction() as cursor:
                _seed_visuals(cursor.connection, "legacy" if legacy else "canonical")

        return native_seed(stores, **kwargs, chat_seed=populate)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        assert "mobile_missing_file_dependency" in preview.issues
        assert not inventory.complete
        before = _fingerprints(tmp_path / "profile")
        with journal._locked(exclusive=False) as fd:
            events = tuple(journal._records(fd))
        with session._discovery_reads(private_sqlite=True):
            actual = _preview_selected_mobile(config, config)
        assert "mobile_missing_file_dependency" in actual.issues
        with pytest.raises(ValueError, match="mobile_source_changed"):
            prepare(session, journal, config, actual, inventory)
        assert _fingerprints(tmp_path / "profile") == before
        with journal._locked(exclusive=False) as fd:
            assert tuple(journal._records(fd)) == events
    assert not (tmp_path / "archives").exists()


def test_complete_c2_discharges_retained_dictionary_derivations(tmp_path, monkeypatch):
    with content_reads(tmp_path, monkeypatch, _seed_actors) as reads:
        plan = reads.content_actor_plan()
        pending = [
            (r.table, p)
            for r in plan.rows
            for p in r.required_proofs
            if p == "C2_native_dictionary_metadata_derivation"
        ]
        assert not pending


def test_captured_persona_identity_retains_admitted_reference_provenance(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_actors(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                "INSERT INTO actor_portable_identities(actor_kind,local_actor_id,portable_uuid) VALUES ('persona','persona-1','497d518c-2f9e-40a7-a408-72a8442f20a7')"
            )

    with content_reads(
        tmp_path, monkeypatch, seed, native_seed=_persona_seed(tmp_path)
    ) as reads:
        plan = reads.content_actor_plan()
        assert not plan.blockers
        actor = next(
            dict(r.values)
            for r in plan.rows
            if r.table == "actor_portable_identities"
            and r.original == ("persona", "persona-1")
        )
        assert actor["local_actor_id"] == "persona-1"
        assert len(plan.authority_references) == 1
        proof = plan.authority_references[0]
        assert proof.kind == "persona" and proof.reference_id == "persona-1"
        assert proof.member_digest and proof.provenance_digest


def test_complete_c2_retains_native_authority_slot_none_and_dictionary_proof(
    tmp_path, monkeypatch
):
    source = {}

    def native_seed(db, purpose):
        source[purpose] = db.get_local_authority_id()
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET character_id=1,assistant_kind='character',assistant_id='1',assistant_authority_id=? WHERE id='conversation'",
                (source[purpose],),
            )
        db.add_conversation(
            {
                "id": "null-authority",
                "title": "Native supplied None",
                "character_id": 1,
                "assistant_kind": "character",
                "assistant_id": "1",
                "assistant_authority_id": None,
            }
        )

    with content_reads(
        tmp_path, monkeypatch, _seed_actors, native_seed=native_seed
    ) as reads:
        plan = reads.content_actor_plan()
        assert not plan.blockers
        rows = {r.original[0]: r for r in plan.rows if r.table == "conversations"}
        assert rows["conversation"].pending_columns == ("assistant_authority_id",)
        assert "assistant_authority_id" not in dict(rows["conversation"].values)
        assert dict(rows["null-authority"].values)["assistant_authority_id"] is None
        assert rows["null-authority"].pending_columns == ()
        assert len(plan.authority_slots) == 1
        assert plan.authority_slots[0].original_authority_id == source["legacy"]
        assert plan.authority_slots[0].mapped_character_id == plan.identities.mapped(
            "character", 1
        )
        assert plan.dictionary_derivations
        assert any(
            s.handler == "T" and s.domain == "character_authority"
            for s in plan.successors
        )
        assert all(not p.startswith("C2") for r in plan.rows for p in r.required_proofs)


def test_captured_actor_uniqueness_and_c1_blockers_propagate_without_identity_invention(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_actors(connection, purpose)
        connection.execute("UPDATE buddy_profiles SET source_key='same-native-source'")
        if purpose == "canonical":
            connection.execute(
                "INSERT INTO actor_portable_identities(actor_kind,local_actor_id,portable_uuid) VALUES ('character','1','f476a5e1-01cc-4389-9daf-21c619516846')"
            )
        else:
            connection.execute(
                "UPDATE character_cards SET name='canonical name' WHERE id=11"
            )
            connection.execute(
                "UPDATE conversations SET character_id=11,assistant_kind='character',assistant_id='11',assistant_authority_id=NULL WHERE id='conversation'"
            )
            connection.execute(
                "INSERT INTO actor_portable_identities(actor_kind,local_actor_id,portable_uuid) VALUES ('character','11','c082e4a5-f7a9-4ae7-9204-89c117b61fee')"
            )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_actor_plan()
        assert any(
            b.table == "buddy_profiles" and b.reason == "native_unique_conflict"
            for b in plan.blockers
        )
        assert any(
            b.table == "actor_portable_identities"
            and b.original == ("character", "1")
            and b.reason == "native_unique_conflict"
            for b in plan.blockers
        )
        assert any(
            b.table == "actor_portable_identities"
            and b.original == ("character", "11")
            and b.reason == "dependency_blocked"
            for b in plan.blockers
        )
        assert not any(
            r.table == "conversations" and r.original == ("conversation",)
            for r in plan.rows
        )
        assert not any(r.table == "messages" for r in plan.rows)
        assert any(
            r.table == "actor_portable_identities" and r.original == ("character", "11")
            for r in plan.source_rows
        )
        assert plan.identities.mapped("character", 11)


def test_native_actor_import_provenance_is_preserved_as_inert_display_history():
    import json

    from tldw_chatbook.Backup_Recovery.mobile_content_actors import _logical_actors

    data = _logical_inputs()
    for table, context in (
        ("visual_identity_assets", {"provenance": "actor-pack-import"}),
        (
            "visual_identity_packs",
            {
                "source": "local",
                "license_value": "unspecified",
                "profile_pack_id": "profile-history",
            },
        ),
    ):
        data.source[table][0]["source_context_json"] = json.dumps(context)
    _logical_actors(data)
    assert not data.blockers
    for table in ("visual_identity_assets", "visual_identity_packs"):
        row = next(r for r in data.rows if r.table == table)
        assert (
            dict(row.values)["source_context_json"]
            == data.source[table][0]["source_context_json"]
        )


def test_native_decimal_actor_aliases_refuse_mapped_identity_collision():
    from tldw_chatbook.Backup_Recovery.mobile_content_actors import _logical_actors

    data = _logical_inputs()
    original = next(
        r
        for r in data.source["actor_portable_identities"]
        if r["actor_kind"] == "character"
    )
    data.source["actor_portable_identities"].append(
        original
        | {
            "local_actor_id": "01",
            "portable_uuid": "c082e4a5-f7a9-4ae7-9204-89c117b61fee",
        }
    )
    _logical_actors(data)
    conflicts = {
        b.original
        for b in data.blockers
        if b.table == "actor_portable_identities"
        and b.reason == "native_unique_conflict"
    }
    assert conflicts == {("character", "1"), ("character", "01")}
    assert not any(
        r.table == "actor_portable_identities"
        and dict(r.values)["actor_kind"] == "character"
        for r in data.rows
    )


def test_captured_remote_assistant_is_an_exact_external_dependency(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_actors(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                "UPDATE conversations SET runtime_backend='server',assistant_kind='character',assistant_id='remote-actor',character_id=NULL,assistant_authority_id=NULL WHERE id='conversation'"
            )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_actor_plan()
        assert any(
            b.table == "conversations"
            and b.original == ("conversation",)
            and b.reason == "dependency_unproven"
            and b.dependency_domain == "remote_assistant"
            and b.dependency_id == "remote-actor"
            for b in plan.blockers
        )
        assert not any(r.table == "messages" for r in plan.rows)
        original = next(
            dict(r.values)
            for r in plan.workflows.settings.source_rows
            if r.table == "conversations" and r.original == ("conversation",)
        )
        assert original["assistant_id"] == "remote-actor"


@pytest.mark.parametrize(
    "table,column,value,reason,domain",
    [
        (
            "visual_identity_assets",
            "owner_user_id",
            True,
            "dependency_unproven",
            "visual_local_owner",
        ),
        (
            "visual_identity_assets",
            "source_context_json",
            '{"unknown_reference":111}',
            "native_content_invalid",
            None,
        ),
        (
            "visual_identity_pack_versions",
            "manifest_json",
            '{"assets":[]}',
            "native_content_invalid",
            None,
        ),
        (
            "persona_visual_assets",
            "pack_id",
            999,
            "dependency_unproven",
            "persona_visual_pack",
        ),
        (
            "buddy_visual_bindings",
            "buddy_revision",
            2,
            "dependency_unproven",
            "buddy_revision",
        ),
    ],
)
def test_logical_actor_native_invalid_or_unproved_associations_refuse(
    table, column, value, reason, domain
):
    from tldw_chatbook.Backup_Recovery.mobile_content_actors import _logical_actors

    data = _logical_inputs()
    data.source[table][0][column] = value
    _logical_actors(data)
    assert any(
        b.table == table and b.reason == reason and b.dependency_domain == domain
        for b in data.blockers
    )
    assert not any(r.table == table for r in data.rows)


def test_native_manifest_generation_maps_declared_reference_without_rehashing_history():
    import json

    from tldw_chatbook.Backup_Recovery.mobile_content_actors import _logical_actors
    from tldw_chatbook.Character_Chat.visual_identity import (
        validate_visual_identity_manifest,
    )

    data = _logical_inputs()
    original = data.source["visual_identity_pack_versions"][0]
    manifest = json.loads(original["manifest_json"])
    manifest["assets"][0]["generation"] = {
        "retained_asset_id": 221,
        "tool": "Literal 221 remains display history",
    }
    original["manifest_json"] = json.dumps(manifest)
    source_bytes = original["manifest_json"]
    before = validate_visual_identity_manifest(manifest)
    _logical_actors(data)
    assert not data.blockers
    row = next(r for r in data.rows if r.table == "visual_identity_pack_versions")
    mapped = json.loads(dict(row.values)["manifest_json"])
    assert mapped["assets"][0]["generation"] == {
        "retained_asset_id": data.maps["visual_asset", 221],
        "tool": "Literal 221 remains display history",
    }
    after = validate_visual_identity_manifest(mapped)
    assert before.pack_content_sha256 == after.pack_content_sha256
    assert ("visual_asset", 221) in row.dependencies
    assert original["manifest_json"] == source_bytes


def test_captured_complete_c2_preserves_native_emote_fallback_and_legacy_expression(
    tmp_path, monkeypatch
):
    from contextlib import closing

    from Tests.Backup_Recovery.test_mobile_content_graph import _install_rows
    from Tests.Character_Chat.test_visual_identity_resolution import _png_bytes
    from tldw_chatbook.Character_Chat.visual_identity import resolve_visual_identity
    from tldw_chatbook.Chat.message_metadata import (
        CharacterEmoteMetadata,
        MessageMetadata,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    encoded = {}
    image = _png_bytes((20, 30, 40))

    def seed(connection, purpose):
        _seed_actors(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                "INSERT INTO visual_identity_packs(id,owner_user_id,title) VALUES (221,0,'Unbound historical pack')"
            )
            connection.execute(
                "UPDATE character_expression_images SET state_id='thinking' WHERE id=11"
            )

    def native_seed(db, purpose):
        if purpose != "legacy":
            return
        db.set_character_expression_image(11, "thinking", image, "image/png")
        native = resolve_visual_identity(
            db,
            actor_kind="character",
            actor_id=11,
            requested_state="thinking",
            user_data_dir=tmp_path / "source-native-assets",
        )
        assert native.resolution_source == "legacy_expression"
        assert native.expression_id == 11
        assert (native.pack_id, native.pack_version_id, native.asset_id) == (
            None,
            None,
            None,
        )
        forms = {
            "graph-answer": CharacterEmoteMetadata(
                0, actor_kind="character", actor_id=1, fallback_reason="no_active_pack"
            ),
            "graph-variant": CharacterEmoteMetadata(
                0,
                actor_kind="character",
                actor_id=11,
                expression_id=native.expression_id,
                expression_key=native.resolved_expression_key,
            ),
            "emote-cross-owner": CharacterEmoteMetadata(
                0,
                actor_kind="character",
                actor_id=1,
                expression_id=11,
                expression_key=native.resolved_expression_key,
            ),
            "emote-partial": CharacterEmoteMetadata(
                0,
                actor_kind="character",
                actor_id=11,
                pack_id=221,
            ),
        }
        for identity, emote in forms.items():
            if identity.startswith("emote-"):
                db.add_message(
                    {
                        "id": identity,
                        "conversation_id": "conversation",
                        "sender": "Assistant",
                        "role": "assistant",
                        "content": "Exact emote history",
                        "parent_message_id": "graph-user",
                    }
                )
            encoded[identity] = MessageMetadata(character_emote=emote).to_json()
            assert MessageMetadata.from_json(encoded[identity]).character_emote == emote
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE messages SET metadata_json=? WHERE id=?",
                    (encoded[identity], identity),
                )

    with content_reads(tmp_path, monkeypatch, seed, native_seed=native_seed) as reads:
        plan = reads.content_actor_plan()
        messages = {r.original[0]: r for r in plan.rows if r.table == "messages"}
        mismatched = {
            b.original[0]
            for b in plan.blockers
            if b.table == "messages" and b.reason == "native_emote_graph_mismatch"
        }
        checks = {
            "actor_only_fallback_preserved": "graph-answer" in messages,
            "native_legacy_expression_preserved": "graph-variant" in messages,
            "cross_owner_refuses": "emote-cross-owner" in mismatched
            and "emote-cross-owner" not in messages,
            "partial_pack_refuses": "emote-partial" in mismatched
            and "emote-partial" not in messages,
        }
        assert all(checks.values()), checks
        originals = {
            r.original[0]: dict(r.values)["metadata_json"]
            for r in plan.workflows.settings.source_rows
            if r.table == "messages"
        }
        for identity in ("graph-answer", "graph-variant"):
            assert originals[identity] == encoded[identity]
            expected = MessageMetadata.from_json(encoded[identity]).character_emote
            actual = MessageMetadata.from_json(
                dict(messages[identity].values)["metadata_json"]
            ).character_emote
            assert actual == replace(
                expected,
                actor_id=plan.identities.mapped("character", expected.actor_id),
                expression_id=plan.identities.mapped(
                    "character_expression", expected.expression_id
                )
                if expected.expression_id is not None
                else None,
            )
            assert not any(
                p.startswith("C2c:") for p in messages[identity].required_proofs
            )
        with closing(CharactersRAGDB(":memory:", "mapped-emote-consumer")) as sink:
            _install_rows(
                sink,
                [
                    r
                    for r in plan.rows
                    if r.table in {"character_cards", "character_expression_images"}
                ],
            )
            mapped_actor = plan.identities.mapped("character", 11)
            assert (
                sink.get_character_expression_image(mapped_actor, "thinking") == image
            )
            assert (
                sink.get_character_expression_image(
                    plan.identities.mapped("character", 1), "thinking"
                )
                is None
            )
            resolved = resolve_visual_identity(
                sink,
                actor_kind="character",
                actor_id=mapped_actor,
                requested_state="thinking",
                user_data_dir=tmp_path / "mapped-native-assets",
            )
            assert resolved.resolution_source == "legacy_expression"
            assert resolved.expression_id == plan.identities.mapped(
                "character_expression", 11
            )
            assert (resolved.pack_id, resolved.pack_version_id, resolved.asset_id) == (
                None,
                None,
                None,
            )
