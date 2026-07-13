"""Regression guard: every app-level Filters spec must use callable testers.

A glob *string* (e.g. ``"*.json"``) passed where ``Filter`` expects a
``Callable[[Path], bool]`` makes ``Filter.__call__`` do ``"*.json"(path)`` ->
``TypeError: 'str' object is not callable``, which tears down the whole app
session the moment the file dialog opens. Several screens shipped this bug;
this test builds each Filters collection and asserts the testers are callable
and return a bool without raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Third_Party.textual_fspicker import Filters


def _assert_filters_callable(filters: Filters) -> None:
    # Filters stores Filter objects; calling one runs its tester(path).
    for index, (_name, _id) in enumerate(filters.selections):
        f = filters[index]
        result = f(Path("sample.json"))
        assert isinstance(result, bool), f"filter {index} returned {result!r}"
        # A non-callable tester would raise TypeError above, not return a bool.


def test_eval_default_filters_are_callable():
    from tldw_chatbook.Widgets.file_picker_dialog import EvalFilePickerDialog

    dialog = EvalFilePickerDialog()
    _assert_filters_callable(dialog._get_default_filters())


def test_create_filter_helper_is_callable_and_matches():
    from tldw_chatbook.Widgets.file_picker_dialog import create_filter

    f = create_filter("*.yaml;*.yml;*.json")
    assert callable(f)
    assert f(Path("x.json")) is True
    assert f(Path("x.YAML")) is False  # case-sensitive fnmatch on name
    assert f(Path("x.txt")) is False


def test_eval_dialogs_dataset_filters_are_callable():
    import tldw_chatbook.Widgets.Evals.eval_dialogs as ed

    # The Filters live inline in a method/function; rebuild the exact spec.
    filters = Filters(
        ("Dataset Files", lambda p: p.suffix.lower() in (".json", ".jsonl", ".csv", ".parquet")),
        ("JSON Files", lambda p: p.suffix.lower() in (".json", ".jsonl")),
        ("CSV Files", lambda p: p.suffix.lower() == ".csv"),
        ("Parquet Files", lambda p: p.suffix.lower() == ".parquet"),
        ("All Files", lambda p: True),
    )
    _assert_filters_callable(filters)
    # And guard the source no longer contains a list/glob tester.
    src = Path(ed.__file__).read_text(encoding="utf-8")
    assert '("Dataset Files", ["' not in src


@pytest.mark.parametrize(
    "module_path",
    [
        "tldw_chatbook/Widgets/file_picker_dialog.py",
        "tldw_chatbook/Widgets/Evals/eval_dialogs.py",
        "tldw_chatbook/Widgets/transcription_history_viewer.py",
        "tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py",
        "tldw_chatbook/UI/CCP_Modules/ccp_dictionary_handler.py",
        "tldw_chatbook/UI/Screens/personas_screen.py",
    ],
)
def test_no_glob_string_filter_tester_in_source(module_path):
    """Static guard: no ``("Name", "*.ext")`` / ``("Name", ["*.ext"])`` in app Filters."""
    import re

    text = Path(module_path).read_text(encoding="utf-8")
    # A filter tuple whose second element is a string/list literal beginning with
    # a glob is the crash signature. create_filter("...") and lambdas are fine.
    bad = re.findall(r'\(\s*"[^"]+"\s*,\s*(?:"\*|\[\s*"\*)', text)
    assert not bad, f"glob-string Filters tester(s) in {module_path}: {bad}"
