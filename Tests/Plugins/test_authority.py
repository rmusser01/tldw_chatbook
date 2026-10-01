"""Authenticated plugin authority is distinct from prepared metadata."""

import pytest


def test_prepared_intent_is_not_commit_proof():
    from tldw_chatbook.Plugins.authority import authority_message

    payload = {"generation": 1, "operation_id": "op", "digest": "abc"}
    assert authority_message("prepared", payload) != authority_message(
        "committed", payload
    )
    assert authority_message("prepared", payload) == authority_message(
        "prepared", payload
    )


@pytest.mark.parametrize(
    "payload", [{"n": float("nan")}, {"n": float("inf")}, {1: "x"}, {"x": (1,)}]
)
def test_message_rejects_non_json_authority(payload):
    from tldw_chatbook.Plugins.authority import authority_message

    with pytest.raises(ValueError):
        authority_message("snapshot", payload)


def test_unknown_purpose_is_rejected():
    from tldw_chatbook.Plugins.authority import authority_message

    with pytest.raises(ValueError):
        authority_message("arbitrary", {})


def test_plugin_keys_are_separate_from_standalone_and_each_other():
    from dataclasses import astuple

    from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
        derive_plugin_authority_keys,
        derive_skill_trust_keys,
    )

    salt = b"s" * 32
    plugin = derive_plugin_authority_keys("passphrase", salt=salt)
    standalone = derive_skill_trust_keys("passphrase", salt=salt)
    assert len(set(astuple(plugin))) == 3
    assert not set(astuple(plugin)) & set(astuple(standalone))
    assert plugin == derive_plugin_authority_keys("passphrase", salt=salt)
    assert all(str(key) not in repr(plugin) for key in astuple(plugin))


def empty_snapshot():
    return {
        "schema_version": 1,
        "installations": [],
        "revisions": [],
        "components": [],
        "selections": [],
        "activation": [],
        "mappings": [],
        "authority_generations": [],
        "revision_trust": [],
        "tombstones": [],
        "data_roots": [],
        "operation_result": None,
    }


def test_complete_schema_rejects_omission_unknown_and_bool_generation():
    from tldw_chatbook.Plugins.authority import PluginMarker, canonical_snapshot

    snapshot = empty_snapshot()
    assert canonical_snapshot(snapshot) == snapshot
    for name in snapshot:
        bad = dict(snapshot)
        del bad[name]
        with pytest.raises(ValueError):
            canonical_snapshot(bad)
    with pytest.raises(ValueError):
        canonical_snapshot({**snapshot, "permission_default": "allow"})
    with pytest.raises(ValueError):
        PluginMarker(
            generation=True, operation_id="op", recovery_snapshot_digest="a" * 64
        )


def test_canonical_authority_keeps_absent_inherit_and_disabled_distinct():
    from tldw_chatbook.Plugins.authority import canonical_snapshot, snapshot_digest

    snapshot = empty_snapshot()
    snapshot["installations"] = [
        {"installation_id": "b", "revision_digest": None, "activation_default": False},
        {"installation_id": "a", "revision_digest": None, "activation_default": False},
    ]
    base = snapshot_digest(snapshot)
    snapshot["activation"] = [
        {"installation_id": "a", "workspace_id": "work", "intent": "inherit"}
    ]
    inherited = snapshot_digest(snapshot)
    snapshot["activation"][0]["intent"] = "disabled"
    assert len({base, inherited, snapshot_digest(snapshot)}) == 3
    assert canonical_snapshot(snapshot)["installations"][0]["installation_id"] == "a"
    snapshot["installations"].append(snapshot["installations"][0])
    with pytest.raises(ValueError):
        canonical_snapshot(snapshot)


def test_root_reference_requires_absolute_safe_path():
    from tldw_chatbook.Plugins.authority import canonical_snapshot

    snapshot = empty_snapshot()
    snapshot["data_roots"] = [
        {
            "root_id": "root",
            "installation_id": "removed",
            "workspace_id": None,
            "path": "../../escape",
            "generation": 1,
            "deletion_fenced": True,
        }
    ]
    with pytest.raises(ValueError):
        canonical_snapshot(snapshot)
