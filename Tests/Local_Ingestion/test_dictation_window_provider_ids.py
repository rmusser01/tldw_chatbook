"""Improved Dictation Window provider dropdowns must offer real dispatch ids.

task-1282: `Dictation_Window_Improved.py` once offered
`("Lightning Whisper", "lightning-whisper")` -- a value nothing in
`transcription_service.py`'s dispatch chain, or `Utils/local_stt_providers.py`'s
`LOCAL_PROVIDER_MODULES`, recognizes. The real id is `"lightning-whisper-mlx"`.
`console_voice_input.py`'s privacy allowlist had the identical typo previously
(see `Utils/local_stt_providers.py`'s module docstring) and was fixed there.
This pins the retained production window so the same typo cannot silently return.

Deliberately excluded from the "must be a real dispatch id" check below:

* `"auto"` -- a documented sentinel
  (`LazyLiveDictationService(transcription_provider="auto")`'s "let the
  transcription service choose" default), not itself a provider
  `transcription_service.py` ever branches on.
* `"openai-whisper"` / `"google-speech"` -- two options
  `Dictation_Window_Improved.py`'s non-privacy list still offers that appear
  *nowhere* in `transcription_service.py` at all (the real remote path is
  `"remote-whisper"`). Those were never wired to anything -- unlike
  `"lightning-whisper"`, which was a typo of a real id -- so that is a
  separate, pre-existing gap outside task-1282's scope. Excluded here rather
  than silently patched alongside an unrelated fix.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# The ids `Local_Ingestion/transcription_service.py` actually branches on in
# its provider dispatch chains (`transcribe`, `transcribe_buffer`,
# `get_available_providers`). Mirrors `Utils/local_stt_providers.py`'s
# `LOCAL_PROVIDER_MODULES` for the local providers, plus the one remote path.
DISPATCHABLE_PROVIDER_IDS = frozenset(
    {
        "parakeet-onnx",
        "parakeet-mlx",
        "lightning-whisper-mlx",
        "faster-whisper",
        "qwen2audio",
        "parakeet",
        "canary",
        "remote-whisper",
    }
)

# Values a dropdown may legitimately offer without being a literal dispatch
# id -- see the module docstring.
NON_DISPATCH_EXCEPTIONS = frozenset({"auto", "openai-whisper", "google-speech"})


def _assert_all_ids_are_real(provider_ids: list[str]) -> None:
    assert provider_ids, "no provider ids found -- test needs updating"
    for provider_id in provider_ids:
        assert (
            provider_id in NON_DISPATCH_EXCEPTIONS
            or provider_id in DISPATCHABLE_PROVIDER_IDS
        ), f"{provider_id!r} is not a real transcription_service dispatch id"


def test_dictation_window_improved_provider_option_ids_are_real() -> None:
    """Drives the real `_get_provider_options()` method (not a source parse):
    its two branches (privacy-mode local-only vs. the full list) are plain
    Python control flow, not two separate `Select(...)` literals, so calling
    it directly exercises the exact same code the widget composes with.
    """
    from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow

    window = ImprovedDictationWindow.__new__(ImprovedDictationWindow)

    window.settings = {"privacy": {"local_only": True}}
    privacy_ids = [option_id for _label, option_id in window._get_provider_options()]

    window.settings = {"privacy": {"local_only": False}}
    all_ids = [option_id for _label, option_id in window._get_provider_options()]

    _assert_all_ids_are_real(privacy_ids)
    _assert_all_ids_are_real(all_ids)


def test_no_dropdown_offers_the_bare_misspelled_lightning_whisper_id() -> None:
    """Belt-and-suspenders: the specific typo task-1282 fixed, by name.

    `_assert_all_ids_are_real` above would already fail if this reappeared
    (`"lightning-whisper"` is in neither allowlist), but this pins the exact
    regression instead of relying on that more general check alone.
    """
    from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow

    window = ImprovedDictationWindow.__new__(ImprovedDictationWindow)
    window.settings = {"privacy": {"local_only": True}}
    privacy_ids = [option_id for _label, option_id in window._get_provider_options()]
    window.settings = {"privacy": {"local_only": False}}
    all_ids = [option_id for _label, option_id in window._get_provider_options()]

    assert "lightning-whisper" not in privacy_ids
    assert "lightning-whisper" not in all_ids
    assert "lightning-whisper-mlx" not in privacy_ids
    assert "lightning-whisper-mlx" not in all_ids


# -- 0.2.4: retain a saved legacy provider until an explicit replacement --


def _fake_get_cli_setting_legacy_provider(section, key=None, default=None):
    """Stand-in for `config.get_cli_setting` with an empty config file,
    except `dictation.provider`, which answers as if a user's saved config
    still holds the pre-task-1282 misspelling. Every other setting mirrors
    an unset config: the production window calls `get_cli_setting("section.key",
    default)` (the dotted two-positional-arg form), so echoing `key` back
    (it carries the caller's default in that form) reproduces "key absent,
    default applies" for everything that isn't the provider.
    """
    if section == "dictation.provider":
        return "lightning-whisper"
    return key


def test_load_settings_preserves_raw_deferred_legacy_provider_id(monkeypatch) -> None:
    """Opening a saved legacy selection must retain its exact value, without writes."""
    import tldw_chatbook.UI.Dictation_Window_Improved as dwi

    monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)
    save_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        dwi,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    settings = dwi.ImprovedDictationWindow.__new__(
        dwi.ImprovedDictationWindow
    )._load_settings()

    assert settings["provider"] == "lightning-whisper"
    assert save_calls == []


def test_initialize_service_refuses_deferred_legacy_before_construction(
    monkeypatch,
) -> None:
    """A saved deferred alias refuses initialization before the service constructor."""
    import tldw_chatbook.UI.Dictation_Window_Improved as dwi

    monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)

    constructed_kwargs = {}

    class _FakeDictationService:
        def __init__(self, **kwargs):
            constructed_kwargs.update(kwargs)

        def update_privacy_settings(self, *_args, **_kwargs):
            pass

        def set_buffer_duration(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(dwi, "LazyLiveDictationService", _FakeDictationService)

    window = dwi.ImprovedDictationWindow()
    window._show_status = lambda *_args, **_kwargs: None

    assert window.settings["provider"] == "lightning-whisper"
    assert window._initialize_service() is False
    assert constructed_kwargs == {}
    assert window.dictation_service is None
    assert "Your saved settings have not been changed." in window.initialization_error


@pytest.mark.parametrize("stored", ["lightning-whisper-mlx", "lightning-whisper"])
def test_provider_select_accepts_only_its_retained_unavailable_value(stored) -> None:
    """Only the exact saved deferred value is retained as an unavailable option."""
    from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow
    from textual.widgets import Select
    from textual.widgets._select import InvalidSelectValueError

    window = ImprovedDictationWindow.__new__(ImprovedDictationWindow)
    window.settings = {"provider": stored, "privacy": {"local_only": True}}
    options = window._get_provider_options()

    select = Select(options=options, value=Select.NULL)
    select._init_selected_option(stored)
    assert select.value == stored
    assert "unavailable" in next(label for label, value in options if value == stored).lower()

    unnormalized_select = Select(options=options, value=Select.NULL)
    other = "lightning-whisper" if stored == "lightning-whisper-mlx" else "lightning-whisper-mlx"
    with pytest.raises(InvalidSelectValueError):
        unnormalized_select._init_selected_option(other)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["lightning-whisper-mlx", "lightning-whisper"])
async def test_passive_provider_events_keep_raw_saved_selection_without_writes(monkeypatch, stored):
    from textual.widgets import Button, Select

    import tldw_chatbook.UI.Dictation_Window_Improved as dwi
    from Tests.UI.test_dictation_settings_debounce import _DictationHarness

    monkeypatch.setattr(dwi, "get_cli_setting", lambda section, key=None: stored if section == "dictation.provider" else key)
    writes = []
    monkeypatch.setattr(dwi, "save_settings_to_cli_config", writes.append)
    monkeypatch.setattr(dwi, "LazyLiveDictationService", lambda **kwargs: pytest.fail("service created"))
    async with _DictationHarness().run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        view = pilot.app.query_one(dwi.ImprovedDictationWindow)
        assert view.settings["provider"] == stored
        assert view._settings_snapshot()["dictation"]["provider"] == stored
        button = view.query_one("#dictation-toggle-btn", Button)
        assert button.disabled
        assert "Your saved settings have not been changed." in str(button.tooltip)
        select = view.query_one("#provider-select", Select)
        select.value = "faster-whisper"
        select.post_message(Select.Changed(select, stored))
        select.value = Select.NULL
        select.set_options(view._get_provider_options())
        select.value = stored
        await pilot.pause()
        assert view.settings["provider"] == stored
        assert view.dictation_service is None
        assert writes == []
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", ["lightning-whisper", "auto"])
async def test_deliberate_supported_replacement_persists_once_without_service_creation(
    monkeypatch, saved
):
    import copy

    from textual.widgets import Collapsible, Select

    import tldw_chatbook.UI.Dictation_Window_Improved as dwi
    from Tests.UI.test_dictation_settings_debounce import _DictationHarness

    def setting(section, default=None):
        if section == "dictation.provider":
            return saved
        if section == "transcription.default_provider":
            return "lightning-whisper"
        return default

    monkeypatch.setattr(dwi, "get_cli_setting", setting)
    writes = []
    monkeypatch.setattr(dwi, "save_settings_to_cli_config", lambda snapshot: writes.append(copy.deepcopy(snapshot)))
    monkeypatch.setattr(dwi, "LazyLiveDictationService", lambda **kwargs: pytest.fail("replacement initialized service"))
    async with _DictationHarness().run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        view = pilot.app.query_one(dwi.ImprovedDictationWindow)
        for section in view.query(Collapsible):
            section.collapsed = False
        select = view.query_one("#provider-select", Select)
        select.focus()
        await pilot.press("enter", "home", "down", "down", "down", "enter")
        await pilot.pause()
        assert view.settings["provider"] == "faster-whisper"
        assert view.dictation_service is None
        assert view.is_initialized is False
        await view.remove()
    assert len(writes) == 1
    assert writes[0]["dictation"]["provider"] == "faster-whisper"


@pytest.mark.asyncio
async def test_privacy_edit_refreshes_options_without_rewriting_saved_alias(monkeypatch):
    from textual.widgets import Select, Switch

    import tldw_chatbook.UI.Dictation_Window_Improved as dwi
    from Tests.UI.test_dictation_settings_debounce import _DictationHarness

    monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)
    writes = []
    monkeypatch.setattr(dwi, "save_settings_to_cli_config", writes.append)
    async with _DictationHarness().run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        view = pilot.app.query_one(dwi.ImprovedDictationWindow)
        view.query_one("#local-only-switch", Switch).value = False
        await pilot.pause()
        assert view.query_one("#provider-select", Select).value == "lightning-whisper"
        await view.remove()
    assert len(writes) == 1
    assert writes[0]["dictation.privacy"]["local_only"] is False
    assert writes[0]["dictation"]["provider"] == "lightning-whisper"


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [False, True])
async def test_replacement_keeps_existing_service_and_terminal_retirement(monkeypatch, retired):
    from types import SimpleNamespace

    from textual.widgets import Collapsible, Select

    import tldw_chatbook.UI.Dictation_Window_Improved as dwi
    from Tests.UI.test_dictation_settings_debounce import _DictationHarness

    monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)
    writes = []
    monkeypatch.setattr(dwi, "save_settings_to_cli_config", writes.append)
    owner = SimpleNamespace(
        state=dwi.DictationState.IDLE,
        processing_thread=None,
        abandon=lambda: pytest.fail("replacement abandoned an existing owner"),
    )
    async with _DictationHarness().run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        view = pilot.app.query_one(dwi.ImprovedDictationWindow)
        view.dictation_service = owner
        if retired:
            view.cleanup()
        for section in view.query(Collapsible):
            section.collapsed = False
        select = view.query_one("#provider-select", Select)
        select.focus()
        await pilot.press("enter", "home", "down", "down", "down", "enter")
        await pilot.pause()
        assert view.settings["provider"] == "lightning-whisper"
        assert view.dictation_service is owner
        assert view._retired is retired
        assert writes == []
    assert view._retired is True
    assert view.dictation_service is owner
    assert writes == []
