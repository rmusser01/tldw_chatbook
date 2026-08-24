"""Console Library retrieval-mode setting (task-1337, plan Task 7).

Covers the visible ``Use direct Library tools`` toggle under Settings >
Library/RAG (with the full approved privacy/scope copy rendered below it, not
in a tooltip) and the ChatScreen provider factory that reads
``[console].direct_library_tools`` fresh for every Console run, injecting a
``LibraryToolProvider`` (direct mode) or ``LibraryRagToolProvider`` (RAG
fallback) into the cached ``ConsoleChatController`` without rebuilding it.
"""

import json
from types import SimpleNamespace

import pytest
from textual.widgets import Checkbox

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Library.library_tool_contract import ERROR_FEATURE_UNAVAILABLE


def _patch_cli_config(monkeypatch, config):
    """Point the config module's load seam at a fake mapping."""
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda *args, **kwargs: config,
    )


def _build_screen():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "test-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "test-model"}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    return app, ChatScreen(app)


def _turn_context(*, direct: bool):
    return SimpleNamespace(
        library_authority=SimpleNamespace(direct_library_tools=direct)
    )


# --- ChatScreen provider factory --------------------------------------------


def test_factory_direct_mode_builds_library_tool_provider(monkeypatch):
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.Library.local_library_tool_service import (
        LocalLibraryToolService,
    )

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    _app, screen = _build_screen()

    provider = screen._console_library_provider_factory(_turn_context(direct=True))

    assert isinstance(provider, LibraryToolProvider)
    assert isinstance(provider._service, LocalLibraryToolService)


def test_factory_off_mode_builds_bounded_rag_provider(monkeypatch):
    from tldw_chatbook.Agents.library_rag_tool_provider import (
        LibraryRagToolProvider,
    )

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": False}})
    app, screen = _build_screen()

    provider = screen._console_library_provider_factory(_turn_context(direct=False))

    assert isinstance(provider, LibraryRagToolProvider)
    assert provider._rag_service is getattr(app, "library_rag_search_service", None)


def test_factory_fails_closed_when_turn_context_is_missing(monkeypatch):
    _patch_cli_config(monkeypatch, {})
    _app, screen = _build_screen()

    assert screen._console_library_provider_factory() is None


def test_factory_reads_captured_context_without_rebuilding_controller(monkeypatch):
    """Each captured authority selects its provider without rebuilding."""
    from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    config = {"console": {"direct_library_tools": True}}
    _patch_cli_config(monkeypatch, config)
    _app, screen = _build_screen()

    controller = screen._ensure_console_chat_controller()
    assert controller._library_provider_factory is not None

    first = controller._library_provider_factory(_turn_context(direct=True))
    config["console"]["direct_library_tools"] = False
    second = controller._library_provider_factory(_turn_context(direct=False))

    assert isinstance(first, LibraryToolProvider)
    assert isinstance(second, LibraryRagToolProvider)
    assert screen._console_chat_controller is controller


def test_factory_assembles_service_only_from_local_app_attributes(monkeypatch):
    """The direct service is wired from the app's local service attributes --
    and only those (identity, not reconstruction)."""
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    app, screen = _build_screen()
    app.local_media_reading_service = SimpleNamespace(marker="media")
    app.notes_service = SimpleNamespace(marker="notes")
    app.local_prompt_service = SimpleNamespace(marker="prompts")
    app.local_skills_service = SimpleNamespace(marker="skills")
    app.local_chat_conversation_service = SimpleNamespace(marker="conversations")
    app.local_library_collections_service = SimpleNamespace(marker="collections")

    provider = screen._console_library_provider_factory(_turn_context(direct=True))

    assert isinstance(provider, LibraryToolProvider)
    service = provider._service
    assert service._media is app.local_media_reading_service
    assert service._notes is app.notes_service
    assert service._prompts is app.local_prompt_service
    assert service._skills is app.local_skills_service
    assert service._conversations is app.local_chat_conversation_service
    assert service._collections is app.local_library_collections_service


def test_factory_wires_the_policy_enforcer_into_the_chunk_tool_service(monkeypatch):
    """Task 5 (chunking-agent-tools, spec §6): the Console-direct chunk tool
    service receives the APP's policy enforcer -- the writing chunk tools
    (`library_save_chunk_spec`, `library_rechunk_media`) are service-level
    gated on the Console path, closing the ungated Console-direct gap."""
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    app, screen = _build_screen()
    app.local_media_reading_service = SimpleNamespace(marker="media")

    provider = screen._console_library_provider_factory()

    assert isinstance(provider, LibraryToolProvider)
    chunk_service = provider._service._media_chunk
    assert chunk_service is not None
    # Identity with the app's own enforcer -- a REAL enforcer, not None and
    # not a reconstruction (the Console gate is closed).
    assert isinstance(app.service_policy_enforcer, ServicePolicyEnforcer)
    assert chunk_service._policy_enforcer is app.service_policy_enforcer


def test_factory_chunk_read_tools_degrade_when_one_media_handle_is_missing(
    monkeypatch, tmp_path
):
    """Qodo review (PR #1976): the factory constructs the chunk service when
    EITHER media handle resolves, so the one-present/one-absent shape must
    degrade the read tools to the NAMED feature_unavailable payload -- not
    scrub an AttributeError on the missing handle to storage_error."""
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    app, screen = _build_screen()
    # Media DB present, reading service absent: the service IS constructed
    # (the factory's either-handle guard), with a None reading handle.
    app.media_db = MediaDatabase(
        tmp_path / "console-degrade.db", client_id="console-degrade-tests"
    )
    app.local_media_reading_service = None

    provider = screen._console_library_provider_factory()

    assert isinstance(provider, LibraryToolProvider)
    assert provider._service._media_chunk is not None
    degrade_cases = (
        ("library:library_get_media_structure", {"id": "media:irrelevant"}),
        (
            "library:library_get_media_chunk",
            {"id": "media:irrelevant", "chunk_index": 0},
        ),
    )
    for tool_id, args in degrade_cases:
        result = provider.invoke(tool_id, args)
        assert result.ok is False
        payload = json.loads(result.error)
        assert payload["error"]["code"] == ERROR_FEATURE_UNAVAILABLE


def test_factory_missing_backend_yields_per_tool_feature_unavailable(monkeypatch):
    """A backend that is None must degrade that backend's tools to
    ``feature_unavailable`` -- never fail the whole provider."""
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    app, screen = _build_screen()
    app.local_media_reading_service = None
    app.notes_service = None
    app.local_prompt_service = None
    app.local_skills_service = None
    app.local_chat_conversation_service = None
    app.local_library_collections_service = None

    provider = screen._console_library_provider_factory(_turn_context(direct=True))

    assert isinstance(provider, LibraryToolProvider)
    # Catalog still exposes the full descriptor set (no total failure).
    assert len(provider.list_catalog()) == 24
    for tool_id in ("library:library_list_notes", "library:library_list_media"):
        result = provider.invoke(tool_id, {})
        assert result.ok is False
        payload = json.loads(result.error)
        assert payload["error"]["code"] == ERROR_FEATURE_UNAVAILABLE


def test_factory_present_backend_serves_its_tool(monkeypatch):
    """One present backend answers its own tools even while others are None."""
    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    app, screen = _build_screen()
    calls = []

    class _FakeNotes:
        def list_library_notes(self, user_id, *, limit, offset):
            calls.append((user_id, limit, offset))
            return {"items": [], "total": 0}

    app.local_media_reading_service = None
    app.notes_service = _FakeNotes()
    app.local_prompt_service = None
    app.local_skills_service = None
    app.local_chat_conversation_service = None
    app.local_library_collections_service = None

    provider = screen._console_library_provider_factory(_turn_context(direct=True))
    result = provider.invoke("library:library_list_notes", {"limit": 5})

    assert result.ok is True
    assert json.loads(result.content)["total"] == 0
    assert calls == [("local_library", 5, 0)]
    # ...while the missing media backend still reports feature_unavailable.
    media_result = provider.invoke("library:library_list_media", {})
    assert media_result.ok is False
    assert (
        json.loads(media_result.error)["error"]["code"] == ERROR_FEATURE_UNAVAILABLE
    )


# --- Settings > Library/RAG compose -----------------------------------------


@pytest.mark.asyncio
async def test_settings_library_rag_renders_console_retrieval_toggle(
    monkeypatch, tmp_path
):
    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _visible_text,
    )
    from Tests.UI.test_settings_configuration_hub import (
        _open_settings_category,
        _wire_rag_profile_adapter,
    )

    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    _patch_cli_config(monkeypatch, {})

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-library-rag")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        checkbox = screen.query_one(
            "#settings-library-rag-direct-library-tools", Checkbox
        )
        assert checkbox.value is True
        # `_visible_text` collects Static/Button only -- the toggle's label is
        # asserted on the Checkbox itself.
        assert str(checkbox.label) == "Use direct Library tools"

        # The approved spec-section-8 copy is visible below the toggle, not
        # hidden in a tooltip.
        assert "Console agent retrieval" in text
        assert (
            "Console agents may automatically list, count, read, and lexically "
            "search" in text
        )
        assert (
            "Direct list, count, view, and lexical search tools are unavailable"
            in text
        )
        assert "Library RAG as the default retrieval method" in text
        assert "Notes, Media, and Conversations" in text
        assert "requires an available, populated index" in text
        assert "leaves your device" in text
        assert "Use a local model" in text
        assert "Console agents only" in text
        assert "MCP Library access is controlled separately" in text
        # And the privacy copy really is a rendered Static, not a tooltip.
        assert checkbox.tooltip is None or "Privacy" not in str(checkbox.tooltip)

        # Toggling stages a draft and enables Save through the existing path.
        checkbox.value = False
        screen.handle_library_rag_direct_library_tools_changed(
            Checkbox.Changed(checkbox, False)
        )
        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert "Unsaved" in _visible_text(screen)


# --- Save path ---------------------------------------------------------------


def test_persist_library_rag_save_writes_console_section_after_profile(
    monkeypatch, tmp_path
):
    """Profile save lands first; the [console] section follows in the same
    logical save, deep-merged so unrelated Console keys survive."""
    from Tests.UI.test_settings_configuration_hub import _wire_rag_profile_adapter
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        SettingsLibraryRagDefaults,
        build_library_rag_save_sections,
    )
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    _wire_rag_profile_adapter(monkeypatch, tmp_path)

    saved_sections: list[dict] = []

    def fake_save_sections(self, sections):
        saved_sections.append(dict(sections))
        return True

    monkeypatch.setattr(
        SettingsConfigAdapter, "save_sections", fake_save_sections
    )

    app = _build_test_app()
    app.app_config["console"] = {"max_parallel_runs": 3}
    screen = SettingsScreen(app)
    values = SettingsLibraryRagDefaults(direct_library_tools=False)
    sections = build_library_rag_save_sections(screen._app_config_mapping(), values)

    saved, reason, applied = screen._persist_library_rag_save(values, sections)

    assert saved is True
    assert reason == ""
    assert applied == sections
    assert len(saved_sections) == 1
    console = saved_sections[0]["console"]
    assert console["direct_library_tools"] is False
    assert console["max_parallel_runs"] == 3


def test_persist_library_rag_save_skips_console_write_when_profile_refuses(
    monkeypatch, tmp_path
):
    """Read-only (builtin) profile: the whole save is refused -- the console
    section is NOT written on its own (all-or-nothing)."""
    from Tests.UI.test_settings_configuration_hub import _wire_rag_profile_adapter
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        SettingsLibraryRagDefaults,
        build_library_rag_save_sections,
    )
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    # Active profile stays the read-only builtin default.
    _wire_rag_profile_adapter(monkeypatch, tmp_path, active_id="hybrid_basic")

    calls: list[dict] = []
    monkeypatch.setattr(
        SettingsConfigAdapter,
        "save_sections",
        lambda self, sections: calls.append(dict(sections)) or True,
    )

    screen = SettingsScreen(_build_test_app())
    values = SettingsLibraryRagDefaults(direct_library_tools=False)
    sections = build_library_rag_save_sections({}, values)

    saved, reason, applied = screen._persist_library_rag_save(values, sections)

    assert saved is False
    assert reason == "builtin"
    assert applied is None
    assert calls == []


def test_persist_library_rag_save_reports_console_write_failure(
    monkeypatch, tmp_path
):
    """A failed config write surfaces as an unsuccessful save (draft stays)."""
    from Tests.UI.test_settings_configuration_hub import _wire_rag_profile_adapter
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        SettingsLibraryRagDefaults,
        build_library_rag_save_sections,
    )
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    _wire_rag_profile_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(
        SettingsConfigAdapter, "save_sections", lambda self, sections: False
    )

    screen = SettingsScreen(_build_test_app())
    values = SettingsLibraryRagDefaults(direct_library_tools=True)
    sections = build_library_rag_save_sections({}, values)

    saved, reason, applied = screen._persist_library_rag_save(values, sections)

    assert saved is False
    assert reason
    assert applied is None
