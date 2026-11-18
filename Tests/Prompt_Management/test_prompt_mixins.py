import pytest


def test_mixin_v3_keeps_literal_body_and_has_no_system_lane():
    from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
        decode_prompt_artifact,
    )

    body = "  <svg>{pose}</svg>\n<style>.x { color: red; }</style>\n"
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": body,
        "placement": "after",
    }
    decoded = decode_prompt_artifact(
        {
            "artifact_type": "mixin",
            "prompt_format": "structured",
            "prompt_schema_version": 3,
            "prompt_definition": definition,
            "system_prompt": "",
            "user_prompt": body,
        }
    )
    assert decoded.state == "supported_v3"
    assert decoded.definition.body == body
    assert body not in repr(decoded.definition)
    assert (decoded.compiled_system, decoded.compiled_user) == ("", body)


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_schema_version": True},
        {"prompt_schema_version": 4},
        {
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": True,
                "body": "x",
                "placement": "before",
            }
        },
        {
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": "x",
                "placement": "before",
                "extra": 1,
            }
        },
        {
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": "  ",
                "placement": "before",
            }
        },
        {
            "prompt_definition": {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": "x",
                "placement": "middle",
            }
        },
        {
            "prompt_definition": {
                "kind": "block_prompt",
                "schema_version": 3,
                "body": "x",
                "placement": "before",
            }
        },
        {"prompt_format": "legacy"},
        {"artifact_type": "prompt"},
    ],
)
def test_mixin_rejects_invalid_shape(change):
    from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
        decode_prompt_artifact,
    )

    record = {
        "artifact_type": "mixin",
        "prompt_format": "structured",
        "prompt_schema_version": 3,
        "prompt_definition": {
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "x",
            "placement": "before",
        },
    }
    record.update(change)
    assert decode_prompt_artifact(record).state != "supported_v3"


def test_local_capabilities_validate_and_canonicalize_mixin():
    from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
        local_prompt_capabilities,
        validate_console_artifact_payload,
    )

    body = " literal {value} \n"
    payload = {
        "artifact_type": "mixin",
        "prompt_format": "structured",
        "prompt_schema_version": 3,
        "prompt_definition": {
            "kind": "text_mixin",
            "schema_version": 3,
            "body": body,
            "placement": "before",
        },
        "system_prompt": "stale",
        "user_prompt": "stale",
    }
    result = validate_console_artifact_payload(payload, local_prompt_capabilities())
    assert result["prompt_definition"] == payload["prompt_definition"]
    assert (result["system_prompt"], result["user_prompt"]) == ("", body)


@pytest.mark.asyncio
async def test_local_service_saves_filters_and_reads_snapshot(tmp_path):
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    db = PromptsDatabase(tmp_path / "prompts.db", client_id="mixin-test")
    service = PromptScopeService(LocalPromptService(db), None)
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": " literal ",
        "placement": "after",
    }
    saved = await service.save_prompt(
        mode="local",
        name="mixin marker",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=definition,
        system_prompt="stale",
        user_prompt="stale",
    )
    await service.save_prompt(mode="local", name="normal marker", user_prompt="normal")
    assert saved["artifact_type"] == "mixin"
    assert saved["system_prompt"] == "" and saved["user_prompt"] == " literal "
    assert await service.count_prompts(mode="local", artifact_type="mixin") == 1
    page = await service.browse_prompts(mode="local", artifact_type="mixin")
    assert page["total_items"] == 1 and page["items"][0]["uuid"] == saved["uuid"]
    assert (
        len(
            await service.search_prompts(
                mode="local", query="marker", artifact_type="mixin"
            )
        )
        == 1
    )
    records = await service.read_mixin_records(
        mode="local", record_uuids=(saved["uuid"],)
    )
    assert records[0]["version"] == saved["version"]
    assert records[0]["prompt_definition"] == definition
    with pytest.raises(ValueError):
        await service.save_prompt(
            mode="local",
            prompt_identifier=saved["uuid"],
            artifact_type="prompt",
            user_prompt="escape",
        )
    metadata_update = await service.save_prompt(
        mode="local",
        prompt_identifier=saved["uuid"],
        artifact_type="mixin",
        details="new detail",
        expected_version=saved["version"],
    )
    assert metadata_update["user_prompt"] == definition["body"]
    saved = metadata_update
    versions = await service.list_prompt_versions(
        mode="local", prompt_identifier=saved["uuid"]
    )
    first = versions["items"][0]
    assert first["restore_eligible"]
    updated_definition = {**definition, "body": "updated"}
    updated = await service.save_prompt(
        mode="local",
        prompt_identifier=saved["uuid"],
        name="mixin marker",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=updated_definition,
        expected_version=saved["version"],
    )
    restored = await service.restore_prompt_version(
        mode="local",
        prompt_identifier=saved["uuid"],
        version=first["version"],
        change_id=first["change_id"],
        expected_version=updated["version"],
    )
    assert restored
    current = await service.get_prompt(mode="local", prompt_identifier=saved["uuid"])
    assert current["user_prompt"] == definition["body"]
    with pytest.raises(ValueError):
        await service.record_prompt_usage(mode="local", prompt_identifier=saved["uuid"])
    with pytest.raises(ValueError):
        await service.read_mixin_records(mode="local", record_uuids=("missing",))
    with pytest.raises(ValueError):
        await service.read_mixin_records(mode="server", record_uuids=(saved["uuid"],))


def test_mixin_is_not_a_normal_console_prompt():
    from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController

    assert ConsolePromptsController._is_recipe_prompt_record({"artifact_type": "mixin"})


@pytest.mark.asyncio
async def test_direct_local_service_compiles_mixin_on_create_and_update(tmp_path):
    from types import SimpleNamespace

    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.local_prompt_service import LocalPromptService

    db = PromptsDatabase(tmp_path / "direct.db", "direct")
    interop = SimpleNamespace(
        add_prompt=db.add_prompt,
        fetch_prompt_details=db.fetch_prompt_details,
        get_db_instance=lambda: db,
    )
    service = LocalPromptService(interop)
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "{literal}",
        "placement": "before",
    }
    with pytest.raises(ValueError):
        await service.create_prompt(
            name="invalid",
            artifact_type="prompt",
            prompt_format="structured",
            prompt_schema_version=3,
            prompt_definition=definition,
        )
    result = await service.create_prompt(
        name="direct",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=definition,
        user_prompt="wrong",
    )
    assert result["user_prompt"] == "{literal}"
    updated = await service.update_prompt(
        result["uuid"], details="changed", user_prompt="wrong"
    )
    assert updated["user_prompt"] == "{literal}"
    with pytest.raises(ValueError):
        await service.update_prompt(result["uuid"], artifact_type="prompt")


def test_snapshot_batches_and_refuses_wrong_type_and_deleted():
    import json

    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    db = PromptsDatabase(":memory:", "snapshot")
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "secret",
        "placement": "before",
    }
    uuids = tuple(f"00000000-0000-4000-8000-{index:012x}" for index in range(901))
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO Prompts (name, uuid, last_modified, client_id, artifact_type, prompt_format, prompt_schema_version, prompt_definition) VALUES (?, ?, 'now', 'snapshot', 'mixin', 'structured', 3, ?)",
            ((uid, uid, json.dumps(definition)) for uid in uuids),
        )
    import sqlite3

    db.get_connection().setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 20)
    records = db.fetch_mixin_snapshot(uuids)
    assert tuple(row["uuid"] for row in records) == uuids
    assert all(row["prompt_definition"] == definition for row in records)
    db.add_prompt("ordinary", None, None, user_prompt="body")
    with pytest.raises(ValueError):
        db.fetch_mixin_snapshot(
            ("ordinary",)
        )  # Name lookup must never substitute for UUID.
    with pytest.raises(ValueError):
        db.fetch_mixin_snapshot((uuids[0], uuids[0]))
    _, deleted_uuid, _ = db.add_prompt(
        "deleted",
        None,
        None,
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=definition,
        user_prompt="secret",
    )
    assert db.soft_delete_prompt(deleted_uuid)
    with pytest.raises(ValueError):
        db.fetch_mixin_snapshot((deleted_uuid,))
    db.restore_deleted_prompt(deleted_uuid, expected_version=2)
    assert db.fetch_mixin_snapshot((deleted_uuid,))[0]["version"] == 3


def test_mixin_body_respects_local_compiled_limit():
    from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
        LOCAL_COMPILED_LANE_LIMIT,
        local_prompt_capabilities,
        validate_console_artifact_payload,
    )

    payload = {
        "artifact_type": "mixin",
        "prompt_format": "structured",
        "prompt_schema_version": 3,
        "prompt_definition": {
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "x" * (LOCAL_COMPILED_LANE_LIMIT + 1),
            "placement": "before",
        },
    }
    with pytest.raises(ValueError, match="exceeds"):
        validate_console_artifact_payload(payload, local_prompt_capabilities())


def test_scope_local_adapter_validates_mixin_without_outer_scope():
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService

    db = PromptsDatabase(":memory:", "adapter")
    service = LocalPromptService(db)
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "canonical",
        "placement": "after",
    }
    saved = service.create_prompt(
        {
            "name": "adapter",
            "artifact_type": "mixin",
            "prompt_format": "structured",
            "prompt_schema_version": 3,
            "prompt_definition": definition,
            "user_prompt": "wrong",
        }
    )
    assert saved["user_prompt"] == "canonical"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_name", ["scope", "direct"])
async def test_mixin_update_with_none_version_refuses_interleaved_write(
    tmp_path, monkeypatch, adapter_name
):
    from types import SimpleNamespace

    from tldw_chatbook.DB.Prompts_DB import (
        ExpectedVersionConflictError,
        PromptsDatabase,
    )
    from tldw_chatbook.Prompt_Management.local_prompt_service import (
        LocalPromptService as DirectService,
    )
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService as ScopeService,
    )

    path = tmp_path / "race.db"
    db = PromptsDatabase(path, "first")
    other = PromptsDatabase(path, "other")
    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "original",
        "placement": "after",
    }
    prompt_id, prompt_uuid, _ = db.add_prompt(
        "race",
        None,
        None,
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=definition,
        user_prompt="original",
    )
    newer = {**definition, "body": "concurrent"}
    original_update = db.update_prompt_by_id

    def interleave(identifier, payload, *, expected_version=None):
        other.update_prompt_by_id(
            prompt_id,
            {"prompt_definition": newer, "user_prompt": "concurrent"},
            expected_version=1,
        )
        return original_update(identifier, payload, expected_version=expected_version)

    monkeypatch.setattr(db, "update_prompt_by_id", interleave)
    with pytest.raises(ExpectedVersionConflictError):
        if adapter_name == "scope":
            ScopeService(db).update_prompt(
                prompt_uuid, {"details": "metadata", "expected_version": None}
            )
        else:
            interop = SimpleNamespace(
                fetch_prompt_details=db.fetch_prompt_details, get_db_instance=lambda: db
            )
            await DirectService(interop).update_prompt(
                prompt_uuid, details="metadata", expected_version=None
            )
    result = other.fetch_prompt_details(prompt_uuid)
    assert result["user_prompt"] == "concurrent"
    from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
        deserialize_definition,
    )

    assert deserialize_definition(result["prompt_definition"])["body"] == "concurrent"
    assert result["version"] == 2
    db.close()
    other.close()


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_type": "prompt"},
        {"artifact_type": "recipe"},
        {"prompt_format": "legacy"},
        {"prompt_schema_version": 4},
        {"prompt_schema_version": True},
        {"definition_change": {"schema_version": True}},
        {"definition_change": {"kind": "block_prompt"}},
        {"definition_change": {"extra": 1}},
        {"definition_change": {"body": "  "}},
        {"definition_change": {"placement": "middle"}},
        {"prompt_definition": "not JSON"},
        {"prompt_definition": "[]"},
    ],
)
def test_snapshot_model_decoder_refuses_mislabeled_or_malformed_rows(change):
    import json

    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    definition = {
        "kind": "text_mixin",
        "schema_version": 3,
        "body": "literal {untouched}",
        "placement": "before",
    }
    definition.update(change.get("definition_change", {}))
    record = {
        "artifact_type": "mixin",
        "prompt_format": "structured",
        "prompt_schema_version": 3,
        "prompt_definition": json.dumps(definition),
    }
    record.update(
        {key: value for key, value in change.items() if key != "definition_change"}
    )
    database = PromptsDatabase(":memory:", "literal-model-boundary")
    uid = "00000000-0000-4000-8000-000000000001"
    try:
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO Prompts (name, uuid, last_modified, client_id, artifact_type, "
                "prompt_format, prompt_schema_version, prompt_definition) "
                "VALUES ('Literal', ?, 'now', 'fixture', ?, ?, ?, ?)",
                (
                    uid,
                    record["artifact_type"],
                    record["prompt_format"],
                    record["prompt_schema_version"],
                    record["prompt_definition"],
                ),
            )
        with pytest.raises(ValueError, match="Requested mix-in is unsupported"):
            database.fetch_mixin_snapshot((uid,))
        stored = (
            database.get_connection()
            .execute("SELECT prompt_definition FROM Prompts WHERE uuid=?", (uid,))
            .fetchone()
        )
        assert stored[0] == record["prompt_definition"]
    finally:
        database.close()
