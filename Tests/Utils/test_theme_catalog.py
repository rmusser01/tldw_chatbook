import re
from types import SimpleNamespace

import pytest
from textual.app import InvalidThemeError
from textual.theme import BUILTIN_THEMES, Theme

from tldw_chatbook.css.Themes import theme_catalog as tc
from tldw_chatbook.css.Themes.theme_catalog import (
    BASE_KEYS,
    STRIP_KEYS,
    build_catalog,
    display_name,
    is_catalog_theme,
)
from tldw_chatbook.css.Themes.themes import ALL_THEMES

SHIPPED = {t.name: t for t in ALL_THEMES}
_HEX6 = re.compile(r"^#[0-9A-F]{6}$")


def _available(**extra):
    themes = dict(BUILTIN_THEMES)
    themes.update(SHIPPED)
    themes.update(extra)
    return themes


def test_display_name_titles_slug():
    assert display_name("modern_dark_dracula") == "Modern Dark Dracula"
    assert display_name("textual-light") == "Textual Light"


def test_groups_and_order():
    mine = Theme(name="warm_paper", primary="#BD6B32", dark=False)
    entries = build_catalog(_available(warm_paper=mine), {"warm_paper"}, "warm_paper", "textual-dark")
    origins = [e.origin for e in entries]
    assert origins == sorted(origins, key=["yours", "shipped", "textual"].index)
    assert entries[0].id == "warm_paper" and entries[0].origin == "yours"
    assert {e.id for e in entries if e.origin == "textual"} == set(BUILTIN_THEMES)


def test_custom_prefixed_registrations_are_hidden():
    scratch = Theme(name="custom_apricot", primary="#123456")
    entries = build_catalog(_available(custom_apricot=scratch), set(), "custom_apricot", "textual-dark")
    assert all(not e.id.startswith("custom_") for e in entries)


def test_markers():
    entries = build_catalog(_available(), set(), "nord", "apricot")
    by_id = {e.id: e for e in entries}
    assert by_id["nord"].is_active and not by_id["nord"].is_launch_default
    assert by_id["apricot"].is_launch_default and not by_id["apricot"].is_active
    assert sum(e.is_active for e in entries) == 1


def test_custom_prefixed_active_marks_its_base_theme():
    mine = Theme(name="warm_paper", primary="#BD6B32", dark=False)
    scratch = Theme(name="custom_warm_paper", primary="#123456", dark=False)
    entries = build_catalog(
        _available(warm_paper=mine, custom_warm_paper=scratch), {"warm_paper"}, "custom_warm_paper", "x"
    )
    assert [e.id for e in entries if e.is_active] == ["warm_paper"]


def test_launch_default_not_registered_marks_nothing():
    entries = build_catalog(_available(), set(), "textual-dark", "deleted_theme")
    assert not any(e.is_launch_default for e in entries)


def test_user_file_overriding_shipped_and_textual():
    mine_apricot = Theme(name="apricot", primary="#000000")
    mine_nord = Theme(name="nord", primary="#111111")
    entries = build_catalog(
        _available(apricot=mine_apricot, nord=mine_nord), {"apricot", "nord"}, "nord", "nord"
    )
    by_id = {e.id: e for e in entries}
    assert by_id["apricot"].origin == "yours" and by_id["apricot"].overrides == "shipped"
    assert by_id["nord"].origin == "yours" and by_id["nord"].overrides == "textual"
    assert [e.id for e in entries].count("apricot") == 1


def test_strip_is_seven_resolved_hex_colours():
    entry = next(e for e in build_catalog(_available(), set(), "textual-dark", "x") if e.id == "textual-dark")
    assert len(entry.strip) == len(STRIP_KEYS) == 7
    assert all(c.startswith("#") and len(c) == 7 and c == c.upper() for c in entry.strip)


def test_every_colour_of_every_entry_is_uppercase_rrggbb():
    """No key of any built entry may be an 8-digit alpha hex or an ANSI name.

    deep_dive_cyberspace's `error` colour has alpha < 1
    (`Color.hex` -> `"#FF33AACC"`), and ansi-dark/ansi-light resolve every key
    to an ANSI colour name (`Color.hex` -> `"ansi_default"` etc, since those
    colours are resolved against the terminal's own palette at render time
    and have no RGB of their own). Both used to leak through `.upper()`
    unparsed, which `ThemePreview.paint`'s `set_styles` silently drops --
    highlighting one of those themes left the card showing the PREVIOUS
    theme's colours under the new title.
    """
    entries = build_catalog(_available(), set(), "textual-dark", "textual-dark")
    assert {e.id for e in entries} >= {"deep_dive_cyberspace", "ansi-dark", "ansi-light"}
    bad = [
        (e.id, key, value)
        for e in entries
        for key, value in e.colours
        if not _HEX6.match(value)
    ]
    assert not bad, f"non-#RRGGBB colours leaked through: {bad}"
    assert all(len(e.colours) == len(BASE_KEYS) for e in entries)


def test_duplicate_display_names_get_id_suffix():
    twin = Theme(name="solarized_light", primary="#268BD2", dark=False)
    entries = build_catalog(_available(solarized_light=twin), set(), "textual-dark", "x")
    names = [e.display_name for e in entries]
    assert len(names) == len(set(names))
    assert any(n.endswith("· solarized-light") for n in names)


def test_is_catalog_theme():
    assert is_catalog_theme("textual-dark") and is_catalog_theme("nord") and is_catalog_theme("apricot")
    assert not is_catalog_theme("warm_paper")


class _FakeApp:
    def __init__(self, theme="textual-dark", known=("textual-dark", "nord", "apricot")):
        self._theme = theme
        self._known = set(known)
        self.app_config = {"general": {"default_theme": "textual-dark"}}

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if value not in self._known:
            raise InvalidThemeError(value)
        self._theme = value


@pytest.fixture
def writes(monkeypatch):
    calls = []

    def fake_apply(mutation):
        calls.append(mutation)
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", fake_apply)
    monkeypatch.setattr(tc, "current_launch_default", lambda: "textual-dark")
    return calls


def test_use_theme_persist_writes_config_and_app_config(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    assert app.theme == "nord"
    assert writes == [{"general": {"default_theme": "nord"}}]
    assert app.app_config["general"]["default_theme"] == "nord"
    assert change == tc.ThemeChange("textual-dark", "textual-dark", True)


def test_try_does_not_write(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=False)
    assert app.theme == "nord" and writes == [] and change.persisted is False


def test_use_theme_unknown_name_raises_and_writes_nothing(writes):
    app = _FakeApp()
    with pytest.raises(InvalidThemeError):
        tc.use_theme(app, "no_such_theme", persist=True)
    assert writes == [] and app.theme == "textual-dark"


def test_use_theme_persist_failure_reports_not_persisted(monkeypatch):
    monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=False, caches_reloaded=False))
    monkeypatch.setattr(tc, "current_launch_default", lambda: "textual-dark")
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    assert app.theme == "nord" and change.persisted is False
    assert app.app_config["general"]["default_theme"] == "textual-dark"


def test_revert_restores_active_and_launch_default(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    tc.revert_theme(app, change)
    assert app.theme == "textual-dark"
    assert writes[-1] == {"general": {"default_theme": "textual-dark"}}


def test_revert_of_try_writes_nothing(writes):
    app = _FakeApp()
    tc.revert_theme(app, tc.use_theme(app, "nord", persist=False))
    assert app.theme == "textual-dark" and writes == []


def test_merge_keeps_first_previous_values(writes):
    app = _FakeApp()
    first = tc.use_theme(app, "apricot", persist=False)   # Try apricot
    second = tc.use_theme(app, "nord", persist=True)      # then Use nord
    merged = first.merge(second)
    assert merged == tc.ThemeChange("textual-dark", "textual-dark", True)
    tc.revert_theme(app, merged)
    assert app.theme == "textual-dark"
    assert writes[-1] == {"general": {"default_theme": "textual-dark"}}


def test_user_theme_names_reads_toml_stems(tmp_path):
    (tmp_path / "warm_paper.toml").write_text("[theme]\nname='warm_paper'\n")
    (tmp_path / "notes.txt").write_text("x")
    assert tc.user_theme_names(tmp_path) == {"warm_paper"}
    assert tc.user_theme_names(tmp_path / "missing") == set()
