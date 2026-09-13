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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,enabled,opened",
    [("show", True, True), ("close", True, False), ("disable", False, True)],
)
async def test_independent_visibility_does_not_read_or_change_persona(
    action, enabled, opened
):
    from unittest.mock import AsyncMock

    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import (
        BuddySelection,
        PersonaBuddyPreferences,
    )

    initial = PersonaBuddyPreferences(
        enabled=True, open=action != "show", selection=BuddySelection("independent")
    )
    writes = []
    controller = PersonaBuddyController(
        preferences=initial, preference_writer=lambda p: writes.append(p) or True
    )
    app = SimpleNamespace(
        app_config={}, console_runtime=None, reconcile_persona_buddy_view=AsyncMock()
    )
    manager = coordinator_module().BuddyManagementCoordinator(
        app, controller=controller
    )
    await manager._set_visibility(action)
    result = controller.current_preferences()
    assert result.selection == initial.selection
    assert result.enabled is enabled
    assert result.open is opened
    assert writes[-1] == result
    app.reconcile_persona_buddy_view.assert_awaited_once()


@pytest.mark.asyncio
async def test_import_retry_after_save_failure_reuses_publication_and_keeps_revision_guard(
    monkeypatch,
    tmp_path,
):
    from tldw_chatbook import config
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import BuddySelection
    from tldw_chatbook.Persona_Visual.snapshot import BuddySnapshot
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementChoice,
    )

    published = []
    record = SimpleNamespace(id="imported")
    path = tmp_path / "pack.zip"
    path.write_bytes(b"reviewed by test library")
    review = BuddySnapshot(
        title="Reviewed Buddy",
        manifest_json="{}",
        assets=(),
        artwork={},
        source_sha256="0" * 64,
        _guard=lambda: True,
    )
    library = SimpleNamespace(
        review_archive=lambda _: review,
        publish_review=lambda _: published.append(path) or record,
        get_buddy=lambda _: record,
    )
    controller = PersonaBuddyController()
    manager = coordinator_module().BuddyManagementCoordinator(
        SimpleNamespace(app_config={}, console_runtime=None),
        controller=controller,
        library=library,
    )
    choice = BuddyManagementChoice(enabled=True, import_path=str(path))
    imports = {}
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: False)
    with pytest.raises(ValueError) as failed:
        await manager.apply_choice(choice, expected_revision=0, imports=imports)
    revision = failed.value.buddy_retry_revision
    assert imports == {str(path): "imported"}
    assert controller.current_preferences().selection is None
    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: True)
    await manager.apply_choice(choice, expected_revision=revision, imports=imports)
    assert published == [path]
    assert controller.current_preferences().selection == BuddySelection("imported")
    controller.apply_preferences_patch(selection=BuddySelection("newer"))
    with pytest.raises(ValueError, match="changed elsewhere"):
        await manager.apply_choice(choice, expected_revision=revision, imports=imports)
    assert controller.current_preferences().selection == BuddySelection("newer")


@pytest.mark.asyncio
async def test_invalid_import_error_is_actionable_and_does_not_expose_internal_code(
    tmp_path,
):
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementChoice,
    )

    def invalid(_):
        raise ValueError("persona_visual_import_invalid")

    path = tmp_path / "invalid.zip"
    path.write_bytes(b"invalid")
    manager = coordinator_module().BuddyManagementCoordinator(
        SimpleNamespace(app_config={}, console_runtime=None),
        controller=PersonaBuddyController(),
        library=SimpleNamespace(review_archive=invalid),
    )
    with pytest.raises(ValueError) as failed:
        await manager.apply_choice(BuddyManagementChoice(import_path=str(path)))
    assert "Check the path" in str(failed.value)
    assert "persona_visual_import_invalid" not in str(failed.value)


def test_current_persona_labels_use_names_none_and_existing_workspace_resolution():
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    manager = coordinator_module().BuddyManagementCoordinator(
        SimpleNamespace(
            app_config={},
            local_character_persona_service=SimpleNamespace(
                get_persona_profile=lambda key: (
                    {"id": key, "name": "Archivist"} if key == "known" else None
                )
            ),
        )
    )
    assert manager._persona_label("known") == "Archivist"
    assert manager._persona_label(None) == "None"
    assert manager._persona_label("secret-id") == "Unavailable"
    assert (
        manager._workspace_persona_label(
            SimpleNamespace(
                assistant_defaults=WorkspaceAssistantDefaults(assistant_id="known")
            )
        )
        == "Archivist"
    )
    assert (
        manager._workspace_persona_label(
            SimpleNamespace(
                assistant_defaults=WorkspaceAssistantDefaults(assistant_id="missing")
            )
        )
        == "Unavailable (new conversations use None)"
    )
    assert (
        manager._workspace_persona_label(SimpleNamespace(assistant_defaults=None))
        == "None"
    )


@pytest.mark.asyncio
async def test_partial_persona_failure_reports_saved_buddy_and_safe_retry_revision(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_persona_assignment
    from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
    from tldw_chatbook.Persona_Buddy.preferences import BuddySelection
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementChoice,
    )

    monkeypatch.setattr(config, "save_settings_to_cli_config", lambda _: True)
    assignment = SimpleNamespace(
        apply=AsyncMock(side_effect=ValueError("Persona changed"))
    )
    monkeypatch.setattr(
        console_persona_assignment,
        "prepare_buddy_persona_assignment",
        lambda *_: assignment,
    )
    controller = PersonaBuddyController()
    manager = coordinator_module().BuddyManagementCoordinator(
        SimpleNamespace(app_config={}, console_runtime=None),
        controller=controller,
        library=SimpleNamespace(get_buddy=lambda _: SimpleNamespace(id="migu")),
    )
    with pytest.raises(ValueError, match="Buddy settings were saved") as failed:
        await manager.apply_choice(
            BuddyManagementChoice(
                enabled=True, buddy_id="migu", persona_choice="selected-persona"
            ),
            expected_revision=0,
        )
    assert controller.current_preferences().selection == BuddySelection("migu")
    assert (
        failed.value.buddy_retry_revision
        == controller.snapshot().preferences_generation
    )
    await manager.apply_choice(
        BuddyManagementChoice(enabled=True, buddy_id="migu"),
        expected_revision=failed.value.buddy_retry_revision,
    )
    assert assignment.apply.await_count == 1
