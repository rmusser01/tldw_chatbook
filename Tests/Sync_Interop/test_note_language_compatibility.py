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
    authority = transport.capture_notes_authority("server-a")
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
