"""Apply's preference transaction preserves the last usable selection on failure."""

import importlib.util
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Persona_Buddy.interaction import (
    BuddyBinding,
    BuddyInteractionPreferences,
)
from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
    BuddyTargetChoice,
)


def coordinator_module():
    assert (
        importlib.util.find_spec("tldw_chatbook.UI.Navigation.buddy_management")
        is not None
    ), "Shared Buddy management coordinator is missing"
    from tldw_chatbook.UI.Navigation import buddy_management

    return buddy_management


@pytest.mark.asyncio
async def test_failed_batch_persistence_restores_selection_and_scope(monkeypatch):
    m = coordinator_module()
    from tldw_chatbook import config
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import (
        BuddySelection,
        PersonaBuddyPreferences,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementChoice,
    )

    old = PersonaBuddyPreferences(enabled=True, selection=BuddySelection("old"))
    controller = PersonaBuddyController(preferences=old)
    app = SimpleNamespace(
        app_config={"buddy_interaction": {"animated": True}}, console_runtime=None
    )
    manager = m.BuddyManagementCoordinator(
        app,
        controller=controller,
        library=SimpleNamespace(get_buddy=lambda id: SimpleNamespace(id=id)),
    )
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda sections: False)
    with pytest.raises(ValueError, match="save"):
        await manager.apply_choice(
            BuddyManagementChoice(enabled=True, buddy_id="new", animated=False)
        )
    assert controller.current_preferences() == old
    assert app.app_config["buddy_interaction"] == {"animated": True}


@pytest.mark.asyncio
async def test_apply_batches_artwork_and_scope_without_creating_persona(monkeypatch):
    m = coordinator_module()
    from tldw_chatbook import config
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementChoice,
    )

    controller = PersonaBuddyController()
    app = SimpleNamespace(app_config={}, console_runtime=None)
    manager = m.BuddyManagementCoordinator(
        app,
        controller=controller,
        library=SimpleNamespace(get_buddy=lambda id: SimpleNamespace(id=id)),
    )
    writes = []
    monkeypatch.setattr(
        config,
        "save_settings_to_cli_config",
        lambda sections: writes.append(sections) or True,
    )
    await manager.apply_choice(
        BuddyManagementChoice(enabled=True, buddy_id="migu", animated=False)
    )
    assert len(writes) == 1
    assert writes[0]["persona_buddy"]["buddy_id"] == "migu"
    assert writes[0]["buddy_interaction"]["animated"] is False
    assert controller.current_preferences().selection.buddy_id == "migu"
    assert app.app_config["buddy_interaction"]["animated"] is False


@pytest.mark.parametrize("restored", [False, True])
def test_form_canonicalizes_same_conversation_after_first_save_or_restore(restored):
    m = coordinator_module()
    saved = BuddyBinding(
        kind="conversation",
        target_id="old" if restored else "live",
        conversation_id="saved" if restored else None,
    )
    live = ConsoleChatSession(id="live", persisted_conversation_id="saved")
    target = BuddyTargetChoice(
        "conversation:live", "Saved conversation", BuddyBinding.for_session(live)
    )
    app = SimpleNamespace(
        app_config={},
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: (live,))
        ),
    )
    manager = m.BuddyManagementCoordinator(app)
    manager.preferences = BuddyInteractionPreferences(binding=saved)
    assert manager._binding_for_form((target,)) == target.binding
    assert manager.preferences.binding == saved  # Opening remains read-only.


def test_form_never_canonicalizes_a_repurposed_live_slot():
    m = coordinator_module()
    saved = BuddyBinding(
        kind="conversation", target_id="live", conversation_id="former"
    )
    live = ConsoleChatSession(
        id="live",
        persisted_conversation_id="different",
        conversation_binding_revision=1,
    )
    target = BuddyTargetChoice(
        "conversation:live", "Different", BuddyBinding.for_session(live)
    )
    app = SimpleNamespace(
        app_config={},
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: (live,))
        ),
    )
    manager = m.BuddyManagementCoordinator(app)
    manager.preferences = BuddyInteractionPreferences(binding=saved)
    assert manager._binding_for_form((target,)) == saved


@pytest.mark.asyncio
async def test_first_save_promotes_pin_for_restart_without_creating_a_conversation(
    monkeypatch,
):
    from tldw_chatbook import config
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.interaction import parse_preferences

    original = BuddyBinding(kind="conversation", target_id="live")
    live = ConsoleChatSession(id="live", persisted_conversation_id="saved")
    app = SimpleNamespace(
        app_config={},
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: (live,))
        ),
    )
    manager = coordinator_module().BuddyManagementCoordinator(
        app, controller=PersonaBuddyController()
    )
    manager.preferences = BuddyInteractionPreferences(binding=original, animated=False)
    writes = []
    monkeypatch.setattr(
        config,
        "save_settings_to_cli_config",
        lambda value: writes.append(value) or True,
    )
    assert await manager._promote_saved_binding(original)
    restored = parse_preferences(writes[0]["buddy_interaction"])
    next_runtime = ConsoleChatSession(id="new-live", persisted_conversation_id="saved")
    assert restored.binding.resolve_session((next_runtime,)) is next_runtime
    assert not restored.animated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["repurposed", "temporary", "rebound", "write_failed"]
)
async def test_first_save_promotion_preserves_stale_or_unsavable_preferences(
    monkeypatch, reason
):
    from tldw_chatbook import config
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController

    original = BuddyBinding(
        kind="conversation", target_id="live", ephemeral=reason == "temporary"
    )
    live = ConsoleChatSession(
        id="live",
        persisted_conversation_id=None if reason == "temporary" else "saved",
        ephemeral=reason == "temporary",
        conversation_binding_revision=int(reason == "repurposed"),
    )
    app = SimpleNamespace(
        app_config={},
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: (live,))
        ),
    )
    manager = coordinator_module().BuddyManagementCoordinator(
        app, controller=PersonaBuddyController()
    )
    manager.preferences = BuddyInteractionPreferences(
        binding=BuddyBinding(kind="workspace", target_id="another")
        if reason == "rebound"
        else original
    )
    previous = manager.preferences
    writes = []
    monkeypatch.setattr(
        config,
        "save_settings_to_cli_config",
        lambda value: writes.append(value) or False,
    )
    if reason == "write_failed":
        with pytest.raises(ValueError, match="save"):
            await manager._promote_saved_binding(original)
    else:
        assert not await manager._promote_saved_binding(original)
        assert writes == []
    assert manager.preferences == previous
    assert app.app_config == {}
