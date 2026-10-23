"""TASK-34000.11 final fix round (I-2): the Preview link policy as a pure table.

The behaviours that decide what a clicked Preview href may do used to be
pinned only by ``test_library_note_links_picker_and_preview_extended.py``,
which boots the app and never runs in a PR lane. This table runs over the
pure helpers in ``tldw_chatbook/Notes/note_link_policy.py`` in well under a
second, with no Hypothesis and no app import, and sits in FAST_LANE right
after ``Tests/Notes/test_note_links_typed_titles.py``. It lives beside that file, not under ``Tests/UI/``, whose conftest imports the whole app for every test.

Rules pinned: every non-http(s) scheme, a scheme-relative ``//host`` and an
http(s) url carrying whitespace, a C0/C1 control or a bidirectional control
are refused (fix round 1 M-9; final round M-3); the scheme is matched
case-insensitively, so ``HTTPS://x`` opens (M-2); a scheme-less href with no
host points nowhere Chatbook can open and is told apart from an unsupported
scheme (M-2); the notice text drops C0/C1 and bidi controls (M-3).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Notes.note_link_policy import (
    BIDI_CONTROLS,
    is_clean_http_link,
    link_kind,
    link_notice_text,
)

REFUSED_SCHEMES = (
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    "file:///etc/passwd",
    "data:text/html,hi",
    "vbscript:msgbox(1)",
    "ftp://a/[b]x[/]",
    "mailto:a@b.example",
    "//evil.example/",
    "//evil.example",
)

CONTROL_LADEN_HTTP = (
    "https://a.example/\x1b]0;x\x07",
    "https://a.example/with space",
    "https://a.example/tab\there",
    "https://a.example/\x85nel",
    "https://a.example/‮right-to-left",
    "https://a.example/⁦isolate",
    "https://a.example/‎mark",
    "HTTP://a.example/‭bidi",
)

NO_TARGET = ("#heading", "docs/a.md", "../up.md", "", "?query=1")


@pytest.mark.parametrize("href", REFUSED_SCHEMES)
def test_every_non_http_scheme_and_a_scheme_relative_host_is_refused(href: str) -> None:
    assert not is_clean_http_link(href)
    assert link_kind(href) == "refused"


@pytest.mark.parametrize("href", CONTROL_LADEN_HTTP)
def test_an_http_href_carrying_whitespace_or_a_control_is_refused(href: str) -> None:
    """M-9 / M-3: an escape sequence or a bidi override never reaches the
    browser, whatever the scheme's case."""
    assert not is_clean_http_link(href)
    assert link_kind(href) == "refused"


@pytest.mark.parametrize(
    "href",
    ["https://x.example/[b]bold[/]", "HTTPS://x", "HTTP://up.example/", "Https://Mixed.Example/p?q=1#f"],
)
def test_the_scheme_is_matched_case_insensitively(href: str) -> None:
    """M-2: ``HTTPS://x`` is an https link, not an unsupported scheme."""
    assert is_clean_http_link(href)
    assert link_kind(href) == "http"


@pytest.mark.parametrize("href", NO_TARGET)
def test_a_scheme_less_href_with_no_host_points_nowhere_not_an_unsupported_scheme(
    href: str,
) -> None:
    """M-2: a fragment or a relative path gets its own notice, never the
    "unsupported scheme" one."""
    assert not is_clean_http_link(href)
    assert link_kind(href) == "no_target"


def test_a_note_link_is_told_apart_before_any_other_rule() -> None:
    assert link_kind("note://3b3468f5-d19c-478c-bdd0-bba0d9770616") == "note"
    assert not is_clean_http_link("note://x")


def test_an_href_urlsplit_rejects_is_refused_not_raised() -> None:
    assert link_kind("https://[bad") == "refused"
    assert not is_clean_http_link("https://[bad")


def test_the_notice_text_drops_c0_c1_and_bidi_controls() -> None:
    """M-9 / M-3: the toast never carries an escape sequence or a bidi
    override that would make it read backwards."""
    assert link_notice_text("https://a.example/\x1b]0;x\x07") == "https://a.example/]0;x"
    assert link_notice_text("https://a.example/\x85nel") == "https://a.example/nel"
    assert link_notice_text("abc‮‪‫‬‭def") == "abcdef"
    assert link_notice_text("abc⁦⁧⁨⁩def") == "abcdef"
    assert link_notice_text("abc‎‏def") == "abcdef"
    assert link_notice_text("https://x.example/[b]bold[/]") == "https://x.example/[b]bold[/]"


def test_the_bidi_control_set_is_exactly_the_eleven_ruled_on() -> None:
    expected = {chr(c) for c in range(0x202A, 0x202F)} | {chr(c) for c in range(0x2066, 0x206A)}
    expected |= {"‎", "‏"}
    assert BIDI_CONTROLS == frozenset(expected)
    assert len(BIDI_CONTROLS) == 11
