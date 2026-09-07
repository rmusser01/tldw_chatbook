import hashlib
import json

from tldw_chatbook.Scheduling.slot_keys import (
    build_manual_run_idempotency_payload,
    build_scheduled_run_idempotency_key,
    canonical_hash,
)


def test_scheduled_key_matches_server_recipe_byte_for_byte():
    payload = {
        "definition_id": "d1",
        "definition_version": 3,
        "schedule_slot": "2026-09-01T09:00:00+00:00",
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    key = build_scheduled_run_idempotency_key(
        definition_id="d1",
        definition_version=3,
        schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert key == f"scheduled-task-rq:{expected_digest}"
    # Golden digest pinned for this exact payload (computed once, hardcoded
    # here) so a change to the recipe that happens to agree with itself
    # can't silently pass the reimplementation check above.
    assert key == (
        "scheduled-task-rq:"
        "3ffcacb8c8dd370f97175787b144e08c521b9888ac91512e001de88f684075bb"
    )


def test_canonical_hash_is_key_order_independent():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_manual_payload_matches_server_shape():
    assert build_manual_run_idempotency_payload(definition_id="d9") == {
        "action": "create_manual_run",
        "definition_id": "d9",
        "trigger_reason": "manual",
    }
