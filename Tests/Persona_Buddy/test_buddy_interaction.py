"""Exact Buddy scope and private preference admission."""

from types import SimpleNamespace

import pytest


def interaction_module():
    import importlib.util

    assert (
        importlib.util.find_spec("tldw_chatbook.Persona_Buddy.interaction") is not None
    ), "Buddy scope contract is missing"
    from tldw_chatbook.Persona_Buddy import interaction

    return interaction


def session(
    id="live-a",
    persisted="conversation-a",
    revision=0,
    backend="local",
    ephemeral=False,
):
    return SimpleNamespace(
        id=id,
        persisted_conversation_id=persisted,
        conversation_binding_revision=revision,
        runtime_backend=backend,
        ephemeral=ephemeral,
        workspace_id="workspace-a",
    )


def test_binding_resolves_its_conversation_without_using_active_selection():
    m = interaction_module()
    a, b = session(), session("live-b", "conversation-b")
    target = m.BuddyBinding.for_session(a)
    assert target.resolve_session([b, a]) is a
    assert target.resolve_session([b]) is None


def test_repurposed_live_slot_never_receives_a_bound_reply():
    m = interaction_module()
    target = m.BuddyBinding.for_session(session())
    assert (
        target.resolve_session([session(persisted="conversation-b", revision=1)])
        is None
    )
    assert target.resolve_session([session(revision=1)]) is None


def test_durable_binding_can_reconnect_after_restart_but_never_across_sources():
    m = interaction_module()
    target = m.BuddyBinding.for_session(session())
    restored = session("new-live-id")
    assert target.resolve_session([restored]) is restored
    assert target.resolve_session([session("server-live", backend="server")]) is None


def test_temporary_binding_does_not_enter_profile_config():
    m = interaction_module()
    target = m.BuddyBinding.for_session(session(persisted=None, ephemeral=True))
    prefs = m.BuddyInteractionPreferences(binding=target, animated=False)
    encoded = m.serialize_preferences(prefs)
    assert "live-a" not in str(encoded)
    assert m.parse_preferences(encoded).binding is None
    assert m.parse_preferences(encoded).animated is False


def test_workspace_binding_matches_only_its_own_local_sessions():
    m = interaction_module()
    target = m.BuddyBinding(kind="workspace", target_id="workspace-a")
    assert target.includes(session())
    assert not target.includes(session(backend="server"))
    different = session()
    different.workspace_id = "workspace-b"
    assert not target.includes(different)


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "other", "target_id": "ok"},
        {"kind": "conversation", "target_id": ""},
        {"kind": "workspace", "target_id": "../../private"},
        {"kind": "conversation", "target_id": "a", "binding_revision": True},
    ],
)
def test_invalid_scope_is_rejected(raw):
    m = interaction_module()
    with pytest.raises(ValueError):
        m.BuddyBinding(**raw)
