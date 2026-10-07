"""Negotiated note snapshots preserve source and private pending intent."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.notes_outbox_producer import NotesSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope

KEY = b"x" * 32
SOURCE = '\ufeff{ "x": 1, "x": 2 }\r\n'
LANGUAGE = "Future/Exact Mode"
SCOPE = {
    "server_profile_id": "server-a",
    "authenticated_principal_id": "user-a",
    "workspace_scope": None,
}


class SQLiteNotesStore:
    """Fixture bridge writes receiver results to the actual canonical database."""

    def __init__(self, db):
        self.db = db
        self.hashes = {}

    def upsert_note(self, object_id, payload, *, object_revision):
        from tldw_chatbook.Sync_Interop.notes_local_store import (
            NotesMaterializationUnsupported,
        )

        if any(
            payload.get(field) is not None
            for field in ("conversation_id", "message_id")
        ):
            raise NotesMaterializationUnsupported(
                "Canonical Notes has no backlink storage"
            )
        old = self.db.get_note_by_id(object_id)
        fields = {
            key: payload[key]
            for key in ("title", "content", "content_language")
            if key in payload
        }
        if old:
            self.db.update_note(object_id, fields, old["version"])
        else:
            self.db.add_note(
                payload["title"],
                payload.get("content", payload.get("body", "")),
                note_id=object_id,
                **(
                    {"content_language": payload["content_language"]}
                    if "content_language" in payload
                    else {}
                ),
            )

    def get_note_content_hash(self, note_id):
        return self.hashes.get(note_id)

    def upsert_note_content(self, note_id, payload, payload_hash):
        self.upsert_note(
            note_id, {**payload, "content": payload["body"]}, object_revision=1
        )
        self.hashes[note_id] = payload_hash


@pytest.fixture
def db(tmp_path):
    store = CharactersRAGDB(tmp_path / "notes.db", "language-test")
    yield store
    store.close_connection()


def builder(**kwargs):
    return SyncEnvelopeBuilder(
        dataset_id="dataset-1", device_id="device-1", dataset_key=KEY, **kwargs
    )


@pytest.mark.parametrize("domain", ["notes.note", "notes"])
@pytest.mark.parametrize("language", [LANGUAGE, None])
def test_negotiated_serialized_receiver_real_sqlite(db, domain, language):
    b = builder(notes_adapter_versions={domain: [2]})
    env = (
        b.build_notes_note_upsert(
            note_id="n1", title="Exact", content=SOURCE, content_language=language
        )
        if domain == "notes.note"
        else b.build_note_upsert(
            note_id="n1", title="Exact", body=SOURCE, content_language=language
        )
    )
    assert (env.adapter_version, env.schema_version) == (2, 2)
    if domain == "notes":
        wire = env.model_dump_json()
        assert LANGUAGE not in wire and SOURCE not in wire
        assert (
            decrypt_sync_payload(json.loads(env.payload_ciphertext), key=KEY)[
                "content_language"
            ]
            == language
        )
    else:
        assert env.payload == {
            "title": "Exact",
            "content": SOURCE,
            "conversation_id": None,
            "message_id": None,
            "content_language": language,
        }
        assert env.payload_hash == canonical_payload_hash(env.payload)
    serialized = SyncV2Envelope.model_validate_json(env.model_dump_json()).model_copy(
        update={"object_revision": 1, "server_cursor": 1}
    )
    receiver = SyncEnvelopeApplier(
        local_store=SQLiteNotesStore(db),
        dataset_key=KEY,
        notes_mirror=NotesMirror(":memory:"),
        dataset_id="dataset-1",
    )
    assert receiver.apply(serialized)["status"] == "applied"
    row = db.get_note_by_id("n1")
    assert (row["content"], row["content_language"]) == (SOURCE, language)


@pytest.mark.parametrize("domain", ["notes.note", "notes"])
def test_old_omission_preserves_existing_language(db, domain):
    db.add_note("Old", "old", note_id="n1", content_language=LANGUAGE)
    b = builder()
    env = (
        b.build_notes_note_upsert(note_id="n1", title="New", content=SOURCE)
        if domain == "notes.note"
        else b.build_note_upsert(note_id="n1", title="New", body=SOURCE)
    )
    receiver = SyncEnvelopeApplier(
        local_store=SQLiteNotesStore(db),
        dataset_key=KEY,
        notes_mirror=NotesMirror(":memory:"),
        dataset_id="dataset-1",
    )
    assert (
        receiver.apply(env.model_copy(update={"object_revision": 1}))["status"]
        == "applied"
    )
    assert db.get_note_by_id("n1")["content_language"] == LANGUAGE


@pytest.mark.parametrize("domain", ["notes.note", "notes"])
def test_builder_requires_explicit_complete_peer(domain):
    with pytest.raises(ValueError, match="notes_language_upgrade_required"):
        if domain == "notes.note":
            builder().build_notes_note_upsert(
                note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
            )
        else:
            builder().build_note_upsert(
                note_id="n", title="T", body=SOURCE, content_language=LANGUAGE
            )


def test_private_language_rejected_in_all_clear_metadata():
    base = builder().build_note_upsert(note_id="n", title="T", body="B").model_dump()
    for field in (
        "payload",
        "payload_clear",
        "routing_metadata",
        "encryption_metadata",
    ):
        with pytest.raises(ValueError, match="content_language"):
            SyncV2Envelope.model_validate(
                {**base, field: {"status": {"content_language": "json"}}}
            )


def test_pending_private_ciphertext_and_reason_survive_restart_and_body_ack(
    tmp_path, db
):
    path = tmp_path / "sync.db"
    repo = SyncStateRepository(path)
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )
    result = producer.enqueue_note_upsert(
        **SCOPE, note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    assert result["reason"] == "notes_language_upgrade_required"
    repo = SyncStateRepository(path)
    entries = repo.list_pending_sync_v2_outbox_envelopes(
        **SCOPE, dataset_id="dataset-1"
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["last_error"]["error_code"] == "notes_language_upgrade_required"
    assert (
        decrypt_sync_payload(
            json.loads(entry["envelope"]["payload_ciphertext"]), key=KEY
        )["content_language"]
        == LANGUAGE
    )
    repo.mark_sync_v2_outbox_push_results(
        **SCOPE,
        dataset_id="dataset-1",
        accepted=[
            {
                "client_envelope_id": entry["client_envelope_id"],
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )
    remaining = repo.list_pending_sync_v2_outbox_envelopes(
        **SCOPE, dataset_id="dataset-1"
    )
    assert remaining[0]["last_error"]["error_code"] == "notes_language_ack_required"
    assert (
        repo.list_pending_sync_v2_outbox_envelopes(
            **{**SCOPE, "authenticated_principal_id": "user-b"}, dataset_id="dataset-1"
        )
        == []
    )


@pytest.mark.asyncio
async def test_publication_rechecks_private_peer_and_retains_scope(tmp_path, db):
    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )

    class UnsupportedPeer:
        async def capture_notes_authority(self, server_profile_id, **scope):
            assert server_profile_id == SCOPE["server_profile_id"]
            assert (
                scope["authenticated_principal_id"]
                == SCOPE["authenticated_principal_id"]
            )
            assert scope["workspace_scope"] == SCOPE["workspace_scope"]
            return self, lambda: None

        def __init__(self):
            self.probes = 0
            self.pushes = 0

        async def get_v2_capabilities(self, *, dataset_id, **kwargs):
            self.probes += 1
            return {
                "domains": ["notes.note"],
                "writable_adapter_versions": {"notes.note": [2]},
            }

        async def push_v2_envelopes(self, **kwargs):
            self.pushes += 1
            return {}

    repo = SyncStateRepository(tmp_path / "state.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
        capabilities={
            "domains": ["notes"],
            "writable_adapter_versions": {"notes": [2]},
        },
    )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )
    producer.enqueue_note_upsert(
        **SCOPE, note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    peer = UnsupportedPeer()
    sync = LocalFirstSyncService(
        server_service=peer,
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        dataset_keys={"dataset-1": KEY},
    )
    with pytest.raises(ValueError, match="notes_language_upgrade_required"):
        await sync.sync_once(**SCOPE, domains=["notes"])
    assert (peer.probes, peer.pushes) == (1, 0)
    restarted = SyncStateRepository(tmp_path / "state.db")
    assert (
        restarted.list_pending_sync_v2_outbox_envelopes(
            **SCOPE, dataset_id="dataset-1"
        )[0]["last_error"]["error_code"]
        == "notes_language_upgrade_required"
    )


@pytest.mark.asyncio
async def test_local_save_captures_exact_intent_even_when_enqueue_fails(tmp_path, db):
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

    class LocalNotes:
        def note_transaction(self, user_id):
            return db.transaction(immediate=True)

        def get_note_by_id(self, user_id, note_id, **kwargs):
            return db.get_note_by_id(note_id, **kwargs)

        def update_note(self, user_id, note_id, fields, version):
            return db.update_note(note_id, fields, version)

        def add_note(self, user_id, title, content, **kwargs):
            return db.add_note(title, content, **kwargs)

    repo = SyncStateRepository(tmp_path / "state.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )

    def fail(**kwargs):
        raise RuntimeError("simulated interruption after canonical capture")

    producer.enqueue_note_upsert = fail
    scope = NotesScopeService(
        local_notes_service=LocalNotes(),
        server_service=None,
        sync_v2_notes_producer=producer,
    )
    note_id = await scope.save_note(
        scope="local_note",
        user_id="u",
        title="T",
        content=SOURCE,
        content_language=LANGUAGE,
        sync_v2_profile=SCOPE,
    )
    rows = db.list_latest_dispatchable_note_sync_entries(
        server_profile_id="server-a", dataset_id="dataset-1"
    )
    assert len(rows) == 1
    captured = json.loads(rows[0]["payload"])
    assert (captured["content"], captured["content_language"], captured["version"]) == (
        SOURCE,
        LANGUAGE,
        1,
    )
    await scope.save_note(
        scope="local_note",
        user_id="u",
        note_id=note_id,
        version=1,
        title="Next",
        content="next",
        sync_v2_profile=SCOPE,
    )
    rows = db.list_latest_dispatchable_note_sync_entries(
        server_profile_id="server-a", dataset_id="dataset-1"
    )
    assert json.loads(rows[0]["payload"])["content"] == SOURCE
    assert json.loads(rows[1]["payload"])["content_language"] == LANGUAGE
    await scope.save_note(
        scope="local_note",
        user_id="u",
        note_id=note_id,
        version=2,
        title="Reset",
        content="reset",
        content_language=None,
        sync_v2_profile=SCOPE,
    )
    rows = db.list_latest_dispatchable_note_sync_entries(
        server_profile_id="server-a", dataset_id="dataset-1"
    )
    assert json.loads(rows[2]["payload"])["content_language"] is None


@pytest.mark.parametrize(
    "patch",
    [
        {"object_id": "other"},
        {"domain": "chat"},
        {"apply_status": "failed"},
        {"object_revision": 99},
        {"server_cursor": None},
        {"object_revision": None},
    ],
)
def test_insufficient_complete_ack_retains_intent(tmp_path, patch):
    repo = SyncStateRepository(tmp_path / "state.db")
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    repo.enqueue_sync_v2_outbox_envelope(**SCOPE, dataset_id="dataset-1", envelope=env)
    ack = {
        "client_envelope_id": env.client_envelope_id,
        "domain": "notes.note",
        "object_id": "n",
        "apply_status": "applied",
        "object_revision": 1,
        "server_cursor": 8,
        **patch,
    }
    repo.mark_sync_v2_outbox_push_results(
        **SCOPE, dataset_id="dataset-1", accepted=[ack], rejected=[], conflicts=[]
    )
    assert (
        len(repo.list_pending_sync_v2_outbox_envelopes(**SCOPE, dataset_id="dataset-1"))
        == 1
    )


def test_actual_complete_server_ack_marks_exact_scoped_intent(tmp_path):
    from tldw_chatbook.tldw_api.sync_schemas import SyncV2PushAcceptedEnvelope

    repo = SyncStateRepository(tmp_path / "state.db")
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    repo.enqueue_sync_v2_outbox_envelope(**SCOPE, dataset_id="dataset-1", envelope=env)
    ack = SyncV2PushAcceptedEnvelope(
        client_envelope_id=env.client_envelope_id,
        domain="notes.note",
        object_id="n",
        apply_status="applied",
        object_revision=1,
        server_cursor=8,
    )
    assert repo.mark_sync_v2_outbox_push_results(
        **SCOPE,
        dataset_id="dataset-1",
        accepted=[ack.model_dump()],
        rejected=[],
        conflicts=[],
    ) == {"dispatched": 1, "retained": 0}
    assert (
        repo.list_pending_sync_v2_outbox_envelopes(**SCOPE, dataset_id="dataset-1")
        == []
    )


@pytest.mark.asyncio
async def test_transport_pins_probe_and_rejects_changed_authority(tmp_path):
    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    class Client:
        def __init__(self):
            self.calls = []

        async def get_sync_v2_profile(self, **kwargs):
            return {"user_id": "user-a"}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            return {
                "datasets": [
                    {
                        "dataset_id": "dataset-1",
                        "scope_type": "personal",
                        "workspace_id": None,
                    }
                ]
            }

        async def get_sync_v2_capabilities(self, **kwargs):
            self.calls.append("probe")
            return {"writable_adapter_versions": {"notes.note": [2]}}

        async def push_sync_v2_envelopes(self, request):
            self.calls.append("push")
            provider.current = Client()
            return {
                "dataset_id": "dataset-1",
                "accepted": [
                    {"client_envelope_id": request.envelopes[0].client_envelope_id}
                ],
            }

    class Provider:
        def __init__(self):
            self.current = Client()

        def build_client(self):
            return self.current

        def get_active_context(self):
            return context

    context = type("Context", (), {"active_server_id": "server-a"})()
    provider = Provider()
    transport = ServerSyncService(None, client_provider=provider)
    authority = await transport.capture_notes_authority(
        "server-a",
        authenticated_principal_id="user-a",
        workspace_scope=None,
        dataset_id="dataset-1",
        device_id="device-1",
    )
    await transport.get_v2_capabilities(
        dataset_id="dataset-1", notes_authority=authority
    )
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    with pytest.raises(ValueError, match="notes_authority_changed"):
        await transport.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[env],
            notes_authority=authority,
        )
    assert authority[0].calls == ["probe", "push"]


@pytest.mark.asyncio
async def test_clear_service_requires_exact_mirror_scope_and_applies_real_db(
    tmp_path, db
):
    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )

    env = (
        builder(notes_adapter_versions={"notes.note": [2]})
        .build_notes_note_upsert(
            note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
        )
        .model_copy(
            update={"device_id": "remote", "object_revision": 1, "server_cursor": 7}
        )
    )

    class Peer:
        async def capture_notes_authority(self, server_profile_id, **scope):
            assert server_profile_id == SCOPE["server_profile_id"]
            assert (
                scope["authenticated_principal_id"]
                == SCOPE["authenticated_principal_id"]
            )
            assert scope["workspace_scope"] == SCOPE["workspace_scope"]
            return self, lambda: None

        async def pull_v2_envelopes(self, **kwargs):
            return {
                "dataset_id": "dataset-1",
                "envelopes": [env.model_dump()],
                "next_cursor": "7",
                "has_more": False,
            }

    repo = SyncStateRepository(tmp_path / "state.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    mirror = NotesMirror(":memory:")
    service = LocalFirstSyncService(
        server_service=Peer(),
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        dataset_keys={"dataset-1": KEY},
        notes_mirror=mirror,
        notes_mirror_scope=("server-a", "user-a", None),
    )
    await service.sync_once(**SCOPE, domains=["notes.note"])
    assert db.get_note_by_id("n")["content_language"] == LANGUAGE
    assert mirror.get("dataset-1", "n").object_revision == 1
    repo.set_sync_v2_profile_state(
        **{**SCOPE, "authenticated_principal_id": "user-b"},
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    with pytest.raises(ValueError, match="notes_mirror_scope_unavailable"):
        await service.sync_once(
            **{**SCOPE, "authenticated_principal_id": "user-b"}, domains=["notes.note"]
        )


@pytest.mark.asyncio
async def test_clear_flow_rechecks_before_publication_and_mirror_cleanup(db):
    from tldw_chatbook.Sync_Interop.notes_m1_flow import NotesM1SyncFlow
    from tldw_chatbook.tldw_api import SyncV2PushResponse

    mirror = NotesMirror(":memory:")
    b = builder(notes_adapter_versions={"notes.note": [2]}, notes_mirror=mirror)
    env = b.build_notes_note_upsert(
        note_id="n", title="T", content=SOURCE, content_language=LANGUAGE
    )
    current = True

    class Client:
        async def get_sync_v2_capabilities(self, **kwargs):
            return {"writable_adapter_versions": {"notes.note": [2]}}

        async def push_sync_v2_envelopes(self, request):
            nonlocal current
            current = False
            return SyncV2PushResponse(
                dataset_id="dataset-1",
                accepted=[
                    {
                        "client_envelope_id": env.client_envelope_id,
                        "domain": "notes.note",
                        "object_id": "n",
                        "object_revision": 1,
                        "server_cursor": 7,
                        "apply_status": "applied",
                    }
                ],
            )

    def assert_current():
        if not current:
            raise ValueError("notes_authority_changed")

    flow = NotesM1SyncFlow(
        client=Client(),
        builder=b,
        mirror=mirror,
        local_store=SQLiteNotesStore(db),
        dataset_id="dataset-1",
        device_id="device-1",
        assert_authority_current=assert_current,
    )
    with pytest.raises(ValueError, match="notes_authority_changed"):
        await flow.push([env])
    assert mirror.get("dataset-1", "n") is None


@pytest.mark.asyncio
async def test_old_private_full_snapshot_stays_pending_even_with_v2_peer(tmp_path, db):
    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )

    repo = SyncStateRepository(tmp_path / "state.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    old = builder().build_note_upsert(note_id="n", title="Old", body=SOURCE)
    repo.enqueue_sync_v2_outbox_envelope(**SCOPE, dataset_id="dataset-1", envelope=old)

    class Peer:
        async def capture_notes_authority(self, server_profile_id, **scope):
            assert server_profile_id == SCOPE["server_profile_id"]
            assert (
                scope["authenticated_principal_id"]
                == SCOPE["authenticated_principal_id"]
            )
            assert scope["workspace_scope"] == SCOPE["workspace_scope"]
            return self, lambda: None

        async def get_v2_capabilities(self, **kwargs):
            return {"writable_adapter_versions": {"notes": [1, 2]}}

        async def push_v2_envelopes(self, **kwargs):
            pytest.fail("old immutable private snapshot must not be published")

    service = LocalFirstSyncService(
        server_service=Peer(),
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        dataset_keys={"dataset-1": KEY},
    )
    with pytest.raises(ValueError, match="notes_language_upgrade_required"):
        await service.sync_once(**SCOPE, domains=["notes"])
    entry = SyncStateRepository(
        tmp_path / "state.db"
    ).list_pending_sync_v2_outbox_envelopes(**SCOPE, dataset_id="dataset-1")[0]
    assert entry["envelope"] == old.model_dump(mode="json")
    assert entry["last_error"]["error_code"] == "notes_language_upgrade_required"


@pytest.mark.parametrize("domain", ["notes.note", "notes"])
def test_complete_receiver_refuses_missing_or_invalid_language_before_write(db, domain):
    from tldw_chatbook.Sync_Interop.crypto import encrypt_sync_payload

    b = builder(notes_adapter_versions={domain: [2]})
    env = (
        b.build_notes_note_upsert(
            note_id="n", title="T", content=SOURCE, content_language="json"
        )
        if domain == "notes.note"
        else b.build_note_upsert(
            note_id="n", title="T", body=SOURCE, content_language="json"
        )
    )
    for language in (123, "", "x" * 101):
        if domain == "notes.note":
            payload = {**env.payload, "content_language": language}
            invalid = env.model_copy(
                update={
                    "payload": payload,
                    "payload_hash": canonical_payload_hash(payload),
                }
            )
        else:
            payload = {"title": "T", "body": SOURCE, "content_language": language}
            invalid = env.model_copy(
                update={
                    "payload_ciphertext": encrypt_sync_payload(
                        payload, key=KEY
                    ).model_dump_json(),
                    "payload_hash": canonical_payload_hash(payload),
                }
            )
        receiver = SyncEnvelopeApplier(
            local_store=SQLiteNotesStore(db),
            dataset_key=KEY,
            notes_mirror=NotesMirror(":memory:"),
            dataset_id="dataset-1",
        )
        assert receiver.apply(invalid)["status"] == "conflict"
        assert db.get_note_by_id("n") is None


@pytest.mark.asyncio
async def test_authenticated_python_capability_absence_failure_and_refresh():
    from unittest.mock import AsyncMock

    from tldw_chatbook.tldw_api.client import TLDWAPIClient
    from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError

    client = TLDWAPIClient(base_url="https://a.example", token="token")
    client._request = AsyncMock(
        side_effect=[
            {},
            {"content_language": True},
            APIResponseError(404, "missing"),
            APIConnectionError("offline"),
        ]
    )
    for expected in [
        (False, "unsupported_server"),
        (True, None),
        (False, "unsupported_server"),
        (False, "transport_unavailable"),
    ]:
        capability = await client.get_notes_language_capability()
        assert (capability.available, capability.reason) == expected
    assert client._request.await_count == 4
    await client.close()


@pytest.mark.parametrize(
    "origin",
    [
        {"authenticated_principal_id": "p", "workspace_scope": "w"},
        {"authenticated_principal_id": None, "workspace_scope": None},
    ],
)
def test_immutable_intent_drain_requires_exact_origin_scope_after_restart(
    tmp_path, db, origin
):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
        NotesOrganizationSyncService,
    )

    scope = {"server_profile_id": "server-a", **origin}
    repo = SyncStateRepository(tmp_path / "origin.db")
    repo.set_sync_v2_profile_state(
        **scope,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    note_id = db.add_note("T", SOURCE, content_language=LANGUAGE)
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )
    with db.transaction() as cursor:
        producer.capture_note_intent(
            cursor=cursor, note=db.get_note_by_id(note_id), **scope, base_version=None
        )
    repo = SyncStateRepository(tmp_path / "origin.db")
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )
    service = NotesOrganizationSyncService(
        notes_repository=SimpleNamespace(db=db),
        state_repository=repo,
        notes_producer=producer,
    )
    service.finalize_pending_note_organization_receipts = lambda **kwargs: None
    foreign = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": "other",
        "workspace_scope": "other",
    }
    repo.set_sync_v2_profile_state(
        **foreign,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    assert service.drain_pending_note_intents(**foreign)["copied"] == 0
    assert not repo.list_sync_v2_outbox_entries(**foreign, dataset_id="dataset-1")
    assert service.drain_pending_note_intents(**scope)["copied"] == 1
    payload = json.loads(
        db.list_latest_dispatchable_note_sync_entries(
            server_profile_id="server-a", dataset_id="dataset-1"
        )[0]["payload"]
    )
    assert payload["_source_scope"] == origin


def test_missing_origin_proof_is_not_explicit_null(tmp_path, db):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
        NotesOrganizationSyncService,
    )

    scope = {
        "server_profile_id": "server-a",
        "authenticated_principal_id": None,
        "workspace_scope": None,
    }
    repo = SyncStateRepository(tmp_path / "unknown.db")
    repo.set_sync_v2_profile_state(
        **scope,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    note_id = db.add_note("T", SOURCE, content_language=LANGUAGE)
    with db.transaction() as cursor:
        NotesOrganizationSyncService._insert_note_sync_intent(
            cursor,
            db.get_note_by_id(note_id),
            intent_id="unknown",
            server_profile_id="server-a",
            dataset_id="dataset-1",
            base_version=None,
            request_fingerprint="f" * 64,
        )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=db, dataset_keys={"dataset-1": KEY}
    )
    service = NotesOrganizationSyncService(
        notes_repository=SimpleNamespace(db=db),
        state_repository=repo,
        notes_producer=producer,
    )
    service.finalize_pending_note_organization_receipts = lambda **kwargs: None
    assert service.drain_pending_note_intents(**scope)["copied"] == 0


def test_advertised_v2_does_not_upgrade_unknown_language():
    with pytest.raises(ValueError, match="notes_language_upgrade_required"):
        builder(notes_adapter_versions={"notes": [2]}).build_note_upsert(
            note_id="unknown", title="T", body=SOURCE
        )
    envelope = builder(notes_adapter_versions={"notes": [1, 2]}).build_note_upsert(
        note_id="unknown", title="T", body=SOURCE
    )
    assert envelope.adapter_version == envelope.schema_version == 1
    complete = builder(notes_adapter_versions={"notes": [2]}).build_note_upsert(
        note_id="known", title="T", body=SOURCE, content_language=None
    )
    assert complete.adapter_version == complete.schema_version == 2


@pytest.mark.asyncio
async def test_restore_failed_enqueue_keeps_exact_origin_snapshot_after_restart(
    tmp_path,
):
    from types import SimpleNamespace

    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
        NotesOrganizationSyncService,
    )

    path = tmp_path / "restored-notes.db"
    store = CharactersRAGDB(path, "restore-test")
    note_id = store.add_note("Exact", SOURCE, content_language=LANGUAGE)
    assert store.soft_delete_note(note_id, 1)
    repo = SyncStateRepository(tmp_path / "restore-state.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=store, dataset_keys={"dataset-1": KEY}
    )
    failures = []

    def fail(**kwargs):
        failures.append(kwargs)
        raise RuntimeError("interrupted after restore capture")

    producer.enqueue_note_upsert = fail

    class LocalNotes:
        def note_transaction(self, user_id):
            return store.transaction(immediate=True)

        def restore_note(self, user_id, note_id, version):
            return store.restore_note(note_id, version)

        def get_note_by_id(self, user_id, note_id):
            return store.get_note_by_id(note_id)

        def get_keywords_for_note(self, user_id, note_id):
            return store.get_keywords_for_note(note_id)

    service = NotesScopeService(
        local_notes_service=LocalNotes(),
        server_service=None,
        sync_v2_notes_producer=producer,
    )
    restored = await service.restore_note(
        scope="local_note",
        note_id=note_id,
        version=2,
        user_id="u",
        sync_v2_profile=SCOPE,
    )
    assert restored["version"] == 3
    assert len(failures) == 1
    store.close_connection()
    store = CharactersRAGDB(path, "restore-test")
    repo = SyncStateRepository(tmp_path / "restore-state.db")
    rows = store.list_latest_dispatchable_note_sync_entries(
        server_profile_id="server-a", dataset_id="dataset-1"
    )
    snapshots = [
        json.loads(row["payload"])
        for row in rows
        if "_source_scope" in json.loads(row["payload"])
    ]
    assert len(snapshots) == 1
    payload = snapshots[0]
    assert (payload["content"], payload["content_language"], payload["version"]) == (
        SOURCE,
        LANGUAGE,
        3,
    )
    assert payload["_source_scope"] == {
        "authenticated_principal_id": "user-a",
        "workspace_scope": None,
    }
    canonical = store.get_note_by_id(note_id)
    assert (
        canonical["content"],
        canonical["content_language"],
        canonical["version"],
        canonical["deleted"],
    ) == (SOURCE, LANGUAGE, 3, 0)
    producer = NotesSyncV2OutboxProducer(
        state_repository=repo, notes_db=store, dataset_keys={"dataset-1": KEY}
    )
    drain = NotesOrganizationSyncService(
        notes_repository=SimpleNamespace(db=store),
        state_repository=repo,
        notes_producer=producer,
    )
    drain.finalize_pending_note_organization_receipts = lambda **kwargs: None
    assert drain.drain_pending_note_intents(**SCOPE)["copied"] == 1
    envelope = repo.list_pending_sync_v2_outbox_envelopes(
        **SCOPE, dataset_id="dataset-1"
    )[0]
    assert envelope["last_error"]["error_code"] == "notes_language_upgrade_required"
    store.close_connection()


@pytest.mark.parametrize(
    "backlinks",
    [{"conversation_id": {}}, {"message_id": 4}, {"conversation_id": False}],
)
def test_complete_backlink_types_refused_before_store_or_mirror(db, backlinks):
    mirror = NotesMirror(":memory:")
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="bad-links", title="T", content=SOURCE, content_language=LANGUAGE
    )
    payload = {**env.payload, **backlinks}
    env = env.model_copy(
        update={
            "payload": payload,
            "payload_hash": canonical_payload_hash(payload),
            "object_revision": 1,
            "server_cursor": 1,
        }
    )
    result = SyncEnvelopeApplier(
        local_store=SQLiteNotesStore(db), notes_mirror=mirror, dataset_id="dataset-1"
    ).apply(env)
    assert result["status"] == "conflict"
    assert db.get_note_by_id("bad-links") is None
    assert mirror.get("dataset-1", "bad-links") is None


def test_complete_backlinks_reach_and_survive_existing_materializer():
    from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore

    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="links",
        title="T",
        content=SOURCE,
        content_language=LANGUAGE,
        conversation_id="c",
        message_id="m",
    )
    result = SyncEnvelopeApplier(
        local_store=store, notes_mirror=mirror, dataset_id="dataset-1"
    ).apply(env)
    assert result["status"] == "applied"
    assert store.get("links")["conversation_id"] == "c"
    assert store.get("links")["message_id"] == "m"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_user,actual_user,actual_workspace,manifest_count,has_provider",
    [
        ("user-a", "user-b", None, 1, True),
        ("user-a", "user-a", "foreign-workspace", 1, True),
        (None, "user-a", None, 1, True),
        ("user-a", None, None, 1, True),
        ("user-a", "user-a", None, 0, True),
        ("user-a", "user-a", None, 2, True),
        ("user-a", "user-a", None, 1, False),
    ],
)
async def test_already_foreign_authenticated_scope_refuses_before_pull_or_apply(
    tmp_path,
    db,
    requested_user,
    actual_user,
    actual_workspace,
    manifest_count,
    has_provider,
):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )
    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    scope = {**SCOPE, "authenticated_principal_id": requested_user}
    env = (
        builder(notes_adapter_versions={"notes.note": [2]})
        .build_notes_note_upsert(
            note_id="foreign", title="T", content=SOURCE, content_language=LANGUAGE
        )
        .model_copy(
            update={
                "object_revision": 1,
                "server_cursor": 9,
                "device_id": "other-device",
            }
        )
    )

    class Client:
        def __init__(self):
            self.calls = []

        async def get_sync_v2_profile(self, **kwargs):
            self.calls.append("identity")
            return {"user_id": actual_user}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            self.calls.append("manifest")
            return {
                "datasets": [
                    {
                        "dataset_id": "dataset-1",
                        "scope_type": "workspace" if actual_workspace else "personal",
                        "workspace_id": actual_workspace,
                    }
                ]
                * manifest_count
            }

        async def get_sync_v2_capabilities(self, **kwargs):
            self.calls.append("probe")
            return {"writable_adapter_versions": {"notes.note": [2]}}

        async def push_sync_v2_envelopes(self, request):
            self.calls.append("push")
            return {
                "dataset_id": "dataset-1",
                "accepted": [
                    {
                        "client_envelope_id": item.client_envelope_id,
                        "domain": item.domain,
                        "object_id": item.object_id,
                        "object_revision": 1,
                        "server_cursor": 4,
                        "apply_status": "applied",
                    }
                    for item in request.envelopes
                ],
            }

        async def pull_sync_v2_envelopes(self, **kwargs):
            self.calls.append("pull")
            return {
                "dataset_id": "dataset-1",
                "envelopes": [env.model_dump()],
                "next_cursor": 9,
                "has_more": False,
            }

    client = Client()

    class Provider:
        def build_client(self):
            return client

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    repo = SyncStateRepository(tmp_path / "stable-authority.db")
    repo.set_sync_v2_profile_state(
        **scope,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_cursors={"dataset-1": 3},
    )
    pending = builder(
        notes_adapter_versions={"notes.note": [2]}
    ).build_notes_note_upsert(
        note_id="pending", title="T", content=SOURCE, content_language=LANGUAGE
    )
    repo.enqueue_sync_v2_outbox_envelope(
        **scope, dataset_id="dataset-1", envelope=pending
    )
    mirror = NotesMirror(":memory:")
    service = LocalFirstSyncService(
        server_service=(
            ServerSyncService(None, client_provider=Provider())
            if has_provider
            else ServerSyncService(client)
        ),
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        dataset_keys={"dataset-1": KEY},
        notes_mirror=mirror,
        notes_mirror_scope=("server-a", requested_user, None),
    )
    with pytest.raises(ValueError, match="notes_authority_(changed|unavailable)"):
        await service.sync_once(**scope, domains=["notes.note"])
    assert "push" not in client.calls
    assert "pull" not in client.calls
    assert "probe" not in client.calls
    assert db.get_note_by_id("foreign") is None
    assert mirror.get("dataset-1", "foreign") is None
    assert (
        len(repo.list_pending_sync_v2_outbox_envelopes(**scope, dataset_id="dataset-1"))
        == 1
    )
    assert repo.get_sync_v2_profile_state(**scope)["dataset_cursors"] == {
        "dataset-1": 3
    }


@pytest.mark.parametrize("field", ["conversation_id", "message_id"])
def test_canonical_materializer_refuses_unsupported_backlink_without_writes(db, field):
    mirror = NotesMirror(":memory:")
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="unsupported",
        title="T",
        content=SOURCE,
        content_language=LANGUAGE,
        **{field: "non-null"},
    )
    result = SyncEnvelopeApplier(
        local_store=SQLiteNotesStore(db), notes_mirror=mirror, dataset_id="dataset-1"
    ).apply(env)
    assert result["status"] == "conflict"
    assert result["conflict"]["conflict_type"] == "notes_materialization_unsupported"
    assert db.get_note_by_id("unsupported") is None
    assert mirror.get("dataset-1", "unsupported") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", [None, "workspace-a"])
async def test_typed_authenticated_profile_manifest_proves_exact_scope(workspace):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
    from tldw_chatbook.tldw_api.sync_schemas import (
        SyncV2ProfileResponse,
        SyncV2RestoreManifestResponse,
    )

    class Client:
        async def get_sync_v2_profile(self, **kwargs):
            assert kwargs == {"device_id": "device-1"}
            return SyncV2ProfileResponse(user_id="7")

        async def get_sync_v2_restore_manifest(self, **kwargs):
            assert kwargs == {"dataset_ids": ["dataset-1"]}
            return SyncV2RestoreManifestResponse(
                datasets=[
                    {
                        "dataset_id": "dataset-1",
                        "scope_type": "workspace" if workspace else "personal",
                        "workspace_id": workspace,
                        "encryption_policy": "server_trusted_v1",
                    }
                ]
            )

    client = Client()

    class Provider:
        def build_client(self):
            return client

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    service = ServerSyncService(None, client_provider=Provider())
    pinned, assert_current = await service.capture_notes_authority(
        "server-a",
        authenticated_principal_id="7",
        workspace_scope=workspace,
        dataset_id="dataset-1",
        device_id="device-1",
    )
    assert pinned is client
    assert_current()


@pytest.mark.asyncio
async def test_fixed_transport_without_target_proof_is_actionably_unavailable():
    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    class Client:
        async def get_sync_v2_profile(self, **kwargs):
            raise AssertionError(
                "unproved target must refuse before authenticated reads"
            )

    with pytest.raises(ValueError, match="notes_authority_unavailable"):
        await ServerSyncService(Client()).capture_notes_authority(
            "server-a",
            authenticated_principal_id="7",
            workspace_scope=None,
            dataset_id="dataset-1",
            device_id="device-1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "datasets",
    [
        [],
        [{"dataset_id": "dataset-1", "scope_type": "personal", "workspace_id": None}]
        * 2,
        [{"dataset_id": "dataset-1", "scope_type": "personal"}],
    ],
)
async def test_missing_or_ambiguous_authenticated_manifest_refuses(datasets):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    class Client:
        async def get_sync_v2_profile(self, **kwargs):
            return {"user_id": "7"}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            return {"datasets": datasets}

    client = Client()

    class Provider:
        def build_client(self):
            return client

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    with pytest.raises(ValueError, match="notes_authority_unavailable"):
        await ServerSyncService(
            None, client_provider=Provider()
        ).capture_notes_authority(
            "server-a",
            authenticated_principal_id="7",
            workspace_scope=None,
            dataset_id="dataset-1",
            device_id="device-1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_result", ["absent", "none", "bad_callback"])
async def test_injected_notes_peer_without_capture_refuses_and_retains_pending(
    tmp_path, db, capture_result
):
    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )

    class UnprovedPeer:
        def __init__(self):
            self.calls = []

        async def get_v2_capabilities(self, **kwargs):
            self.calls.append("probe")
            return {"writable_adapter_versions": {"notes.note": [2]}}

        async def push_v2_envelopes(self, **kwargs):
            self.calls.append("push")
            return {
                "dataset_id": "dataset-1",
                "accepted": [
                    {
                        "client_envelope_id": env["client_envelope_id"],
                        "domain": env["domain"],
                        "object_id": env["object_id"],
                        "object_revision": 1,
                        "server_cursor": 4,
                        "apply_status": "applied",
                    }
                    for env in kwargs["envelopes"]
                ],
            }

        async def pull_v2_envelopes(self, **kwargs):
            self.calls.append("pull")
            return {
                "dataset_id": "dataset-1",
                "envelopes": [],
                "next_cursor": 4,
                "has_more": False,
            }

    repo = SyncStateRepository(tmp_path / "missing-guard.db")
    repo.set_sync_v2_profile_state(
        **SCOPE,
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
        dataset_cursors={"dataset-1": 3},
    )
    env = builder(notes_adapter_versions={"notes.note": [2]}).build_notes_note_upsert(
        note_id="pending", title="T", content=SOURCE, content_language=LANGUAGE
    )
    repo.enqueue_sync_v2_outbox_envelope(**SCOPE, dataset_id="dataset-1", envelope=env)
    mirror = NotesMirror(":memory:")
    peer = UnprovedPeer()
    if capture_result != "absent":

        async def invalid_capture(*args, **kwargs):
            return None if capture_result == "none" else (peer, None)

        peer.capture_notes_authority = invalid_capture
    service = LocalFirstSyncService(
        server_service=peer,
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        dataset_keys={"dataset-1": KEY},
        notes_mirror=mirror,
        notes_mirror_scope=("server-a", "user-a", None),
    )
    with pytest.raises(ValueError, match="notes_authority_unavailable"):
        await service.sync_once(**SCOPE, domains=["notes.note"])
    assert peer.calls == []
    assert (
        SyncStateRepository(tmp_path / "missing-guard.db").get_sync_v2_profile_state(
            **SCOPE
        )["last_error"]
        == "notes_authority_unavailable"
    )
    pending = repo.list_pending_sync_v2_outbox_envelopes(
        **SCOPE, dataset_id="dataset-1"
    )
    assert len(pending) == 1
    assert pending[0]["client_envelope_id"] == env.client_envelope_id
    assert repo.get_sync_v2_profile_state(**SCOPE)["dataset_cursors"] == {
        "dataset-1": 3
    }
    assert mirror.get("dataset-1", "pending") is None
    assert db.get_note_by_id("pending") is None

    await service.sync_once(**SCOPE, domains=["chat"])
    assert peer.calls == ["pull"]
    assert (
        SyncStateRepository(tmp_path / "missing-guard.db").get_sync_v2_profile_state(
            **SCOPE
        )["last_error"]
        == "notes_authority_unavailable"
    )
    assert (
        len(repo.list_pending_sync_v2_outbox_envelopes(**SCOPE, dataset_id="dataset-1"))
        == 1
    )


@pytest.mark.asyncio
async def test_mixed_fixed_client_provider_cannot_borrow_target_proof():
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    class Client:
        async def get_sync_v2_profile(self, **kwargs):
            return {"user_id": "same-user"}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            return {
                "datasets": [
                    {
                        "dataset_id": "same-dataset",
                        "scope_type": "personal",
                        "workspace_id": None,
                    }
                ]
            }

    fixed = Client()
    provider_client = Client()

    class Provider:
        def build_client(self):
            return provider_client

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    with pytest.raises(ValueError, match="notes_authority_unavailable"):
        await ServerSyncService(
            fixed, client_provider=Provider()
        ).capture_notes_authority(
            "server-a",
            authenticated_principal_id="same-user",
            workspace_scope=None,
            dataset_id="same-dataset",
            device_id="device-1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["fixed", "provider", "provider_client"])
async def test_captured_notes_authority_rechecks_provider_ownership(change):
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService

    class Client:
        async def get_sync_v2_profile(self, **kwargs):
            return {"user_id": "same-user"}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            return {
                "datasets": [
                    {
                        "dataset_id": "same-dataset",
                        "scope_type": "personal",
                        "workspace_id": None,
                    }
                ]
            }

        async def get_sync_v2_capabilities(self, **kwargs):
            raise AssertionError("stale provider must refuse before probe")

    class Provider:
        def __init__(self):
            self.current = Client()

        def build_client(self):
            return self.current

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    provider = Provider()
    service = ServerSyncService(None, client_provider=provider)
    authority = await service.capture_notes_authority(
        "server-a",
        authenticated_principal_id="same-user",
        workspace_scope=None,
        dataset_id="same-dataset",
        device_id="device-1",
    )
    assert authority[0] is provider.current
    authority[1]()
    if change == "fixed":
        service.client = provider.current
    elif change == "provider":
        service.client_provider = Provider()
    else:
        provider.current = Client()
    with pytest.raises(ValueError, match="notes_authority_changed"):
        authority[1]()
    with pytest.raises(ValueError, match="notes_authority_changed"):
        await service.get_v2_capabilities(
            dataset_id="same-dataset", notes_authority=authority
        )
    with pytest.raises(ValueError, match="notes_authority_changed"):
        await service._push_v2_envelopes(
            dataset_id="same-dataset",
            device_id="device-1",
            envelopes=[],
            notes_authority=authority,
        )
    with pytest.raises(ValueError, match="notes_authority_changed"):
        await service._pull_v2_envelopes(
            dataset_id="same-dataset", device_id="device-1", notes_authority=authority
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["identity", "manifest"])
@pytest.mark.parametrize(
    "failure", ["connection", "authentication", "response", "cancel", "programming"]
)
async def test_notes_proof_read_failure_preserves_real_store_pending(
    tmp_path, db, stage, failure
):
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.Sync_Interop.local_first_sync_service import (
        LocalFirstSyncService,
    )
    from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
    from tldw_chatbook.tldw_api.exceptions import (
        APIConnectionError,
        APIResponseError,
        AuthenticationError,
    )

    errors = {
        "connection": APIConnectionError("private-network-detail"),
        "authentication": AuthenticationError("private-auth-detail"),
        "response": APIResponseError(503, "private-response-detail"),
        "cancel": asyncio.CancelledError(),
        "programming": RuntimeError("programming defect"),
    }

    class Client:
        def __init__(self):
            self.calls = []

        async def get_sync_v2_profile(self, **kwargs):
            self.calls.append("identity")
            if stage == "identity":
                raise errors[failure]
            return {"user_id": "user-a"}

        async def get_sync_v2_restore_manifest(self, **kwargs):
            self.calls.append("manifest")
            raise errors[failure]

        async def get_sync_v2_capabilities(self, **kwargs):
            raise AssertionError("no capability call after failed proof")

        async def push_sync_v2_envelopes(self, request):
            raise AssertionError("no publication after failed proof")

        async def pull_sync_v2_envelopes(self, **kwargs):
            raise AssertionError("no pull after failed proof")

    client = Client()

    class Provider:
        def build_client(self):
            return client

        def get_active_context(self):
            return SimpleNamespace(active_server_id="server-a")

    scope = {**SCOPE, "authenticated_principal_id": "user-a"}
    path = tmp_path / "proof-errors.db"
    repo = SyncStateRepository(path)
    repo.set_sync_v2_profile_state(
        **scope,
        profile_mode="local_first",
        dataset_id="dataset-1",
        device_id="device-1",
        dataset_cursors={"dataset-1": 3},
    )
    envelope = builder(
        notes_adapter_versions={"notes.note": [2]}
    ).build_notes_note_upsert(
        note_id="pending", title="T", content=SOURCE, content_language=LANGUAGE
    )
    repo.enqueue_sync_v2_outbox_envelope(
        **scope, dataset_id="dataset-1", envelope=envelope
    )
    before = repo.list_pending_sync_v2_outbox_envelopes(**scope, dataset_id="dataset-1")
    mirror = NotesMirror(":memory:")
    service = LocalFirstSyncService(
        server_service=ServerSyncService(None, client_provider=Provider()),
        state_repository=repo,
        local_store=SQLiteNotesStore(db),
        notes_mirror=mirror,
        notes_mirror_scope=("server-a", "user-a", None),
    )
    expected = (
        type(errors[failure]) if failure in {"cancel", "programming"} else ValueError
    )
    with pytest.raises(expected):
        await service.sync_once(**scope, domains=["notes.note"])
    reopened = SyncStateRepository(path)
    state = reopened.get_sync_v2_profile_state(**scope)
    assert state["dataset_cursors"] == {"dataset-1": 3}
    assert state["last_error"] == (
        None if failure in {"cancel", "programming"} else "notes_authority_unavailable"
    )
    assert (
        reopened.list_pending_sync_v2_outbox_envelopes(**scope, dataset_id="dataset-1")
        == before
    )
    assert client.calls == (
        ["identity"] if stage == "identity" else ["identity", "manifest"]
    )
    assert mirror.get("dataset-1", "pending") is None
    assert db.get_note_by_id("pending") is None
