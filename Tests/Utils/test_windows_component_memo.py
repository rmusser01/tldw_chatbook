"""Accepted Windows path components are remembered; the answers do not change.

``_component`` decides whether one path component is an ordinary filename by
scanning its characters. One approved-edit turn split 7,656 paths and checked
108,960 components, 2.4 us each: 0.23 s of the turn (TASK-40608, counted
2026-11-26). The answer depends on the characters alone, so an accepted name
is remembered and costs one set lookup afterwards.

What has to stay true: every name gets the answer the full check gives, a
refused name is never remembered, anything that is not exactly a ``str`` still
takes the full check, and the remembered set stays bounded.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import _component

REFUSED = [
    "..", ".", "a/b", "a\\b", "x:y", "x\x00y", "", "file.", "file ", "CON", "nul.txt",
    "com1", "LPT¹.log", "aux", "conin$", "tab\there", 'quo"te', "a<b", "a|b", "a?b", "a*b",
]


def full_check(name):
    """The check as it stood before anything was remembered (the specification)."""
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or any(char in name for char in '/\\:\x00<>|?*"')
        or any(ord(char) < 32 for char in name)
        or name[-1] in " ."
    ):
        raise ValueError("invalid_windows_component")
    base = name.split(".", 1)[0].upper()
    if base in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or (
        len(base) == 4 and base[:3] in {"COM", "LPT"} and base[3] in "123456789¹²³"
    ):
        raise ValueError("reserved_windows_component")
    return name


def answer(check, name):
    try:
        return ("accepted", check(name))
    except ValueError as error:
        return ("refused", str(error))


class _Remembered(set):
    """A set that records what was asked of it and what was added."""

    def __init__(self):
        super().__init__()
        self.asked, self.added = [], []

    def __contains__(self, name):
        self.asked.append(name)
        return super().__contains__(name)

    def add(self, name):
        self.added.append(name)
        super().add(name)


@pytest.fixture
def remembered(monkeypatch):
    names = _Remembered()
    monkeypatch.setattr(windows_files, "_ACCEPTED_COMPONENTS", names)
    return names


def test_an_accepted_name_is_scanned_once(remembered):
    for _ in range(1000):
        assert _component("journal-123.json") == "journal-123.json"
    assert remembered.added == ["journal-123.json"]
    assert remembered == {"journal-123.json"}
    assert len(remembered.asked) == 1000


@pytest.mark.parametrize("name", REFUSED)
def test_a_refused_name_is_refused_every_time_and_never_remembered(remembered, name):
    for _ in range(3):
        assert answer(_component, name) == answer(full_check, name)
        assert answer(_component, name)[0] == "refused"
    assert remembered.added == [] and not remembered


@pytest.mark.parametrize("name", [None, 5, b"journal", ["journal"], {"journal"}, ("j",)])
def test_a_name_that_is_not_text_is_refused_as_before(remembered, name):
    # An unhashable value must not reach the set: the answer is a ValueError.
    assert answer(_component, name) == ("refused", "invalid_windows_component")
    assert remembered.asked == [] and remembered.added == []


def test_a_string_subclass_takes_the_full_check_and_is_returned_unchanged(remembered):
    class Name(str):
        pass

    name = Name("journal-123.json")
    assert _component(name) is name
    with pytest.raises(ValueError, match="reserved_windows_component"):
        _component(Name("CON"))
    assert remembered.asked == [] and remembered.added == []


def test_the_remembered_names_stay_bounded(remembered, monkeypatch):
    monkeypatch.setattr(windows_files, "_ACCEPTED_COMPONENTS_LIMIT", 8)
    for index in range(100):
        name = f"file-{index}.txt"
        assert _component(name) == name
        assert len(remembered) <= 8
    # Emptied when full, so the latest names are still remembered.
    assert "file-99.txt" in remembered and "file-0.txt" not in set(remembered)
    assert _component("file-0.txt") == "file-0.txt"


_ALPHABET = st.sampled_from(list('abcCONnulPRTLcom1239 ./\\:<>|?*"$¹²\x00\x1f-_é'))
_NAMES = st.one_of(
    st.sampled_from(REFUSED + ["journal-123.json", "config.toml", "COM0", "COMA", "LPT10", "con.fig"]),
    st.text(alphabet=_ALPHABET, max_size=9),
    st.text(max_size=6),
)


@settings(max_examples=600, deadline=None)
@given(names=st.lists(_NAMES, min_size=1, max_size=12))
def test_every_name_gets_the_full_checks_answer_however_often_it_is_asked(names):
    saved = windows_files._ACCEPTED_COMPONENTS
    windows_files._ACCEPTED_COMPONENTS = set()
    try:
        for name in names + names:
            assert answer(_component, name) == answer(full_check, name), name
        assert all(
            answer(full_check, name)[0] == "accepted"
            for name in windows_files._ACCEPTED_COMPONENTS
        )
    finally:
        windows_files._ACCEPTED_COMPONENTS = saved
