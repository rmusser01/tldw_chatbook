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
    assert "lightning-whisper-mlx" in privacy_ids
    assert "lightning-whisper-mlx" in all_ids


# -- task-1282 follow-up: normalize an already-persisted legacy provider id --


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


def test_load_settings_normalizes_legacy_provider_id(monkeypatch) -> None:
    """A `dictation.provider` config value saved under the pre-task-1282
    dropdown (`"lightning-whisper"`) must resolve to the real dispatch id
    (`"lightning-whisper-mlx"`) when the production window loads its settings.
    Correcting the dropdown's own options (tested above) does nothing for a
    config file a user already saved with the old, broken value -- this is
    what actually unbreaks that user without them touching Settings again.
    Read-side only: `_load_settings()` must not write the normalized value
    back to config as a side effect of merely loading it.
    """
    import tldw_chatbook.UI.Dictation_Window_Improved as dwi

    monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)
    save_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        dwi,
        "save_setting_to_cli_config",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    settings = dwi.ImprovedDictationWindow.__new__(
        dwi.ImprovedDictationWindow
    )._load_settings()

    assert settings["provider"] == "lightning-whisper-mlx"
    assert save_calls == []


def test_initialize_service_constructs_with_normalized_legacy_provider_id(
    monkeypatch,
) -> None:
    """Drives the exact path the bug report described:
    `ImprovedDictationWindow._initialize_service()` forwards
    `self.settings["provider"]` straight into `LazyLiveDictationService`
    unmodified, so if `_load_settings()` ever stopped normalizing, the
    service itself would be constructed with the dead legacy id. Patches
    only the service constructor (and the two calls `_initialize_service()`
    makes on the constructed instance) plus `_show_status`, which otherwise
    needs a mounted widget; no audio device is touched anywhere here.

    Constructs the window via its real `__init__` rather than `__new__`:
    `is_initialized`/`initialization_error` are Textual `reactive`
    attributes, and `_initialize_service()` assigns to them on both its
    success and failure paths, which raises `ReactiveError` on an instance
    that skipped `Widget.__init__` (no mounting/DOM is otherwise needed --
    `__init__` alone is enough to satisfy the reactive machinery). This also
    means `get_cli_setting` must be patched *before* construction, since
    `__init__` itself calls `_load_settings()`.
    """
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

    assert window.settings["provider"] == "lightning-whisper-mlx"
    assert window._initialize_service() is True
    assert constructed_kwargs["transcription_provider"] == "lightning-whisper-mlx"


def test_provider_select_accepts_the_normalized_legacy_value() -> None:
    """The dropdown must tolerate a normalized legacy value at init time.

    `Select.value` raises `InvalidSelectValueError` when set (at mount, via
    `_init_selected_option`) to something absent from its own options --
    exactly what would happen if the raw legacy `"lightning-whisper"` config
    value reached `Select(value=...)` unnormalized, since the production
    window's option list has not offered that id since task-1282. Calls
    `_init_selected_option` directly (the real method `_on_mount` uses) so
    this exercises actual `Select` validation without needing a running app.
    """
    from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow
    from tldw_chatbook.Utils.local_stt_providers import normalize_provider_id
    from textual.widgets import Select
    from textual.widgets._select import InvalidSelectValueError

    window = ImprovedDictationWindow.__new__(ImprovedDictationWindow)
    window.settings = {"privacy": {"local_only": True}}
    options = window._get_provider_options()

    normalized = normalize_provider_id("lightning-whisper")
    select = Select(options=options, value=Select.NULL)
    select._init_selected_option(normalized)  # must not raise
    assert select.value == normalized

    unnormalized_select = Select(options=options, value=Select.NULL)
    with pytest.raises(InvalidSelectValueError):
        unnormalized_select._init_selected_option("lightning-whisper")
