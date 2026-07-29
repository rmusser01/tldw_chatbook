"""Legacy Dictation Window provider dropdowns must offer real dispatch ids.

task-1282: `Dictation_Window_Improved.py` (both of `_get_provider_options()`'s
branches) and `Dictation_Window.py`'s `"provider-select"` all offered
`("Lightning Whisper", "lightning-whisper")` -- a value nothing in
`transcription_service.py`'s dispatch chain, or `Utils/local_stt_providers.py`'s
`LOCAL_PROVIDER_MODULES`, recognizes. The real id is `"lightning-whisper-mlx"`.
`console_voice_input.py`'s privacy allowlist had the identical typo previously
(see `Utils/local_stt_providers.py`'s module docstring) and was fixed there;
this pins the two legacy windows so the same typo cannot silently return.

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

import ast
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def _callee_name(func: ast.expr) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _select_option_ids(source: str, select_id: str) -> list[str]:
    """Ids from a `Select(options=[(label, id), ...], id=select_id)` literal.

    Parses the real module source rather than importing+instantiating the
    widget, so this test does not depend on Textual's widget lifecycle at
    all -- it only needs the literal the production code actually ships.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _callee_name(node.func) == "Select"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        id_node = keywords.get("id")
        if not (isinstance(id_node, ast.Constant) and id_node.value == select_id):
            continue
        options_node = keywords.get("options")
        assert isinstance(options_node, ast.List), (
            f"Select(id={select_id!r}) options is not a literal list; "
            "this test needs updating to match"
        )
        return [
            elt.elts[1].value
            for elt in options_node.elts
            if isinstance(elt, ast.Tuple)
        ]
    raise AssertionError(f"no Select(id={select_id!r}) call found in source")


def _assert_all_ids_are_real(provider_ids: list[str]) -> None:
    assert provider_ids, "no provider ids found -- test needs updating"
    for provider_id in provider_ids:
        assert (
            provider_id in NON_DISPATCH_EXCEPTIONS
            or provider_id in DISPATCHABLE_PROVIDER_IDS
        ), f"{provider_id!r} is not a real transcription_service dispatch id"


def test_dictation_window_provider_select_ids_are_real() -> None:
    source = (REPO_ROOT / "tldw_chatbook" / "UI" / "Dictation_Window.py").read_text()

    _assert_all_ids_are_real(_select_option_ids(source, "provider-select"))


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

    dictation_window_source = (
        REPO_ROOT / "tldw_chatbook" / "UI" / "Dictation_Window.py"
    ).read_text()
    plain_window_ids = _select_option_ids(dictation_window_source, "provider-select")

    assert "lightning-whisper" not in privacy_ids
    assert "lightning-whisper" not in all_ids
    assert "lightning-whisper" not in plain_window_ids
    assert "lightning-whisper-mlx" in privacy_ids
    assert "lightning-whisper-mlx" in all_ids
    assert "lightning-whisper-mlx" in plain_window_ids
