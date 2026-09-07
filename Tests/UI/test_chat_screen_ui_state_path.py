"""TASK-865: ChatScreen's sidebar-state persistence must honor
``TLDW_CONFIG_PATH`` -- not a hardcoded ``Path.home() / ".config" /
"tldw_cli" / "ui_state.toml"`` literal that always lands in the real
``~/.config/tldw_cli`` regardless of which profile is active.

``_save_sidebar_state``/``_load_sidebar_state`` are plain methods on
``ChatScreen`` that only touch ``self.ui_state``/``self.sidebar_state`` --
this drives the real, unmodified methods directly (via ``object.__new__``,
bypassing Textual's mount lifecycle, which is irrelevant to path
derivation) rather than re-implementing their logic in the test.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _bare_screen_with_ui_state():
    screen = ChatScreen.__new__(ChatScreen)
    screen.ui_state = SimpleNamespace(
        collapsible_states={"notes": True},
        sidebar_search_query="hello",
        last_active_section="notes",
    )
    return screen


def test_save_sidebar_state_writes_under_the_active_profiles_config_dir(
    monkeypatch, tmp_path
):
    """AC #4: with TLDW_CONFIG_PATH pointed at a profile, saving the sidebar
    state must write under THAT profile's directory, not the real
    ``~/.config/tldw_cli``."""
    profile_config = tmp_path / "profile-one" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_config))

    screen = _bare_screen_with_ui_state()
    screen._save_sidebar_state()

    expected_path = tmp_path / "profile-one" / "ui_state.toml"
    assert expected_path.exists()
    assert "hello" in expected_path.read_text(encoding="utf-8")


def test_save_then_reread_round_trips_under_a_retargeted_profile(monkeypatch, tmp_path):
    """Confirms the saved content is actually readable back from the SAME
    path the (real, unmodified) save method wrote to. Reads the file
    directly rather than calling ``_load_sidebar_state()`` -- that method
    also assigns a Textual ``reactive`` attribute, which requires a fully
    mounted widget node unrelated to what this test verifies (path
    derivation and content round-tripping)."""
    import toml

    from tldw_chatbook.config import _get_effective_config_path

    profile_config = tmp_path / "profile-two" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_config))

    saving_screen = _bare_screen_with_ui_state()
    saving_screen._save_sidebar_state()

    reread_path = _get_effective_config_path().parent / "ui_state.toml"
    data = toml.load(reread_path)

    assert data["sidebar"]["collapsible_states"] == {"notes": True}
    assert data["sidebar"]["search_query"] == "hello"
    assert data["sidebar"]["last_active_section"] == "notes"


def test_two_profiles_do_not_share_sidebar_state(monkeypatch, tmp_path):
    """Two different active profiles must persist to two different files,
    not collide into the one real ``~/.config/tldw_cli/ui_state.toml``."""
    profile_a_config = tmp_path / "profile-a" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_a_config))
    screen_a = _bare_screen_with_ui_state()
    screen_a._save_sidebar_state()

    profile_b_config = tmp_path / "profile-b" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_b_config))
    screen_b = ChatScreen.__new__(ChatScreen)
    screen_b.ui_state = SimpleNamespace(
        collapsible_states={"chat": False},
        sidebar_search_query="other",
        last_active_section="chat",
    )
    screen_b._save_sidebar_state()

    path_a = tmp_path / "profile-a" / "ui_state.toml"
    path_b = tmp_path / "profile-b" / "ui_state.toml"
    assert path_a.exists() and path_b.exists()
    assert path_a != path_b
    assert "hello" in path_a.read_text(encoding="utf-8")
    assert "other" in path_b.read_text(encoding="utf-8")
