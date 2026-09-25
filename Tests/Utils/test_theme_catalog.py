from textual.theme import BUILTIN_THEMES, Theme

from tldw_chatbook.css.Themes.theme_catalog import (
    STRIP_KEYS,
    build_catalog,
    display_name,
    is_catalog_theme,
)
from tldw_chatbook.css.Themes.themes import ALL_THEMES

SHIPPED = {t.name: t for t in ALL_THEMES}


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


def test_duplicate_display_names_get_id_suffix():
    twin = Theme(name="solarized_light", primary="#268BD2", dark=False)
    entries = build_catalog(_available(solarized_light=twin), set(), "textual-dark", "x")
    names = [e.display_name for e in entries]
    assert len(names) == len(set(names))
    assert any(n.endswith("· solarized-light") for n in names)


def test_is_catalog_theme():
    assert is_catalog_theme("textual-dark") and is_catalog_theme("nord") and is_catalog_theme("apricot")
    assert not is_catalog_theme("warm_paper")
