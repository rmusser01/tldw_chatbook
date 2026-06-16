"""P2: client parses the live M1 push/pull response shapes."""

from tldw_chatbook.tldw_api import SyncV2PushResponse, SyncV2PullResponse

LIVE_PUSH_RESPONSE = {
    "dataset_id": "ds_1",
    "server_cursor": 129,
    "accepted": [
        {
            "client_envelope_id": "c1",
            "envelope_id": "srv_env_129",
            "server_cursor": 129,
            "object_id": "note_1",
            "object_revision": 1,
            "apply_status": "applied",
        }
    ],
    "idempotent": [],
    "rejected": [],
    "conflicts": [],
    "apply_errors": [],
    "next_cursor": "129",
}

LIVE_PULL_RESPONSE = {
    "dataset_id": "ds_1",
    "from_cursor": 0,
    "next_cursor": "130",
    "has_more": False,
    "envelopes": [
        {
            "envelope_id": "srv_env_130",
            "client_envelope_id": "c2",
            "dataset_id": "ds_1",
            "server_cursor": 130,
            "domain": "notes.note",
            "operation": "upsert",
            "object_id": "note_1",
            "schema_version": 1,
            "payload": {"title": "T", "content": "B"},
            "payload_hash": "sha256:x",
            "object_revision": 1,
            "deleted": False,
            "encryption_metadata": {"policy": "server_trusted_v1"},
        }
    ],
}


def test_push_response_parses_m1_buckets():
    resp = SyncV2PushResponse.model_validate(LIVE_PUSH_RESPONSE)
    assert resp.accepted[0].object_id == "note_1"
    assert resp.accepted[0].object_revision == 1
    assert resp.accepted[0].apply_status == "applied"
    assert resp.server_cursor == 129
    assert resp.idempotent == []
    assert resp.apply_errors == []


def test_pull_response_parses_m1_envelopes():
    resp = SyncV2PullResponse.model_validate(LIVE_PULL_RESPONSE)
    assert resp.from_cursor == 0
    env = resp.envelopes[0]
    assert env.object_id == "note_1"
    assert env.payload == {"title": "T", "content": "B"}
    assert env.server_cursor == 130
