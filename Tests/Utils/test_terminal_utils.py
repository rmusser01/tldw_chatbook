"""Terminal capability detection tests (task-1532).

The avatar/inline-image mode resolver trusts ``detect_terminal_capabilities``.
Inside tmux the host terminal's env (``TERM_PROGRAM``, ``ITERM_SESSION_ID``)
leaks into the pane while tmux does NOT pass graphics escape sequences
through, so recommending "regular" (graphics) there paints nothing. These
tests pin the tmux override and guard the non-tmux graphics path.
"""

from tldw_chatbook.Utils.terminal_utils import detect_terminal_capabilities

_ENV_VARS = (
    "TERM",
    "TERM_PROGRAM",
    "ITERM_SESSION_ID",
    "TMUX",
    "VTE_VERSION",
    "WT_SESSION",
)


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_tmux_forces_pixels_despite_iterm_env_leak(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,1234,0")
    monkeypatch.setenv("TERM", "screen-256color")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setenv("ITERM_SESSION_ID", "w0t0p0:0000")

    caps = detect_terminal_capabilities()

    assert caps["recommended_mode"] == "pixels"
    assert caps["tgp"] is False
    assert caps["sixel"] is False
    assert caps["terminal_type"] == "tmux"


def test_tmux_style_term_without_tmux_var_forces_pixels(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TERM", "tmux-256color")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")

    caps = detect_terminal_capabilities()

    assert caps["recommended_mode"] == "pixels"
    assert caps["terminal_type"] == "tmux"


def test_iterm_outside_tmux_keeps_graphics(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")

    caps = detect_terminal_capabilities()

    assert caps["recommended_mode"] == "regular"
    assert caps["terminal_type"] == "iterm2"
