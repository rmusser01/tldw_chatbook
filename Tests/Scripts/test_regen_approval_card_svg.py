"""Destination-validation tests for scripts/regen_approval_card_svg.py (task-32290, Qodo #2).

``main()`` must reject a CLI-supplied destination that escapes the
repository's ``Docs/`` directory -- via traversal or an absolute path
outside the tree -- before anything is written, and must still accept (and
write to) a path that stays under ``Docs/``.
"""

from __future__ import annotations

from importlib import import_module, util

import pytest

pytestmark = pytest.mark.bootstrap_profile


def _entry():
    spec = util.find_spec("scripts.regen_approval_card_svg")
    assert spec is not None, "regen_approval_card_svg script is unavailable"
    return import_module("scripts.regen_approval_card_svg")


def test_traversal_argument_rejected(monkeypatch):
    mod = _entry()
    target = mod._DOCS_ROOT.parent.parent / "etc" / "x.svg"
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", "../../etc/x.svg"])
    assert mod.main() == 1
    assert not target.exists()


def test_absolute_path_outside_docs_rejected(tmp_path, monkeypatch):
    mod = _entry()
    outside = tmp_path / "x.svg"
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(outside)])
    assert mod.main() == 1
    assert not outside.exists()


def test_path_under_docs_accepted(monkeypatch):
    mod = _entry()
    dest = (
        mod._DOCS_ROOT
        / "User_Guide"
        / "images"
        / "console"
        / "_test_regen_approval_card.svg"
    )
    saved: list[object] = []
    monkeypatch.setattr(mod, "write_guide_svg", lambda path, _svg: saved.append(path))
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(dest)])
    assert mod.main() == 0
    assert saved == [dest]
    assert not dest.exists()


def test_artwork_shows_captured_action_target_scope_and_immediate_choices(monkeypatch):
    import html
    import re

    mod = _entry()
    rendered = []
    monkeypatch.setattr(mod, "write_guide_svg", lambda _path, svg: rendered.append(svg))
    target = mod._DOCS_ROOT / "User_Guide/images/console/_test_approval_capture.svg"
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(target)])
    assert mod.main() == 0
    text = html.unescape("".join(re.findall(r">([^<>]*)</text>", rendered[0]))).replace(
        "\xa0", " "
    )
    for fact in (
        "Read file",
        "notes/release.md",
        "Writer",
        "Chat scratch",
        "Allow once",
        "Deny",
        "More options",
        "Details",
    ):
        assert fact in text
    assert "Auto-denies" not in text
    assert "Approve all" not in text
    assert not target.exists()


def test_generator_supplies_theme_variables_without_test_runner_pins(monkeypatch):
    from dataclasses import replace

    from textual.theme import BUILTIN_THEMES

    mod = _entry()
    original_init = mod._CardApp.__init__

    def use_bare_theme(app, *args, **kwargs):
        original_init(app, *args, **kwargs)
        bare = replace(
            BUILTIN_THEMES["textual-dark"], name="artwork-bare-theme", variables={}
        )
        app.register_theme(bare)
        app.theme = bare.name

    monkeypatch.setattr(mod._CardApp, "__init__", use_bare_theme)
    saved = []
    monkeypatch.setattr(mod, "write_guide_svg", lambda path, _svg: saved.append(path))
    target = mod._DOCS_ROOT / "User_Guide/images/console/_test_approval_theme.svg"
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(target)])
    assert mod.main() == 0
    assert saved == [target]
    assert not target.exists()


def test_written_card_loads_no_remote_fonts(tmp_path, monkeypatch):
    """TASK-38969: the regenerated SVG carries no cdnjs font URL."""
    mod = _entry()
    monkeypatch.setattr(mod, "_DOCS_ROOT", tmp_path)
    target = tmp_path / "approval-card.svg"
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(target)])
    assert mod.main() == 0
    svg = target.read_text(encoding="utf-8")
    assert 'local("FiraCode-Regular")' in svg
    assert "cdnjs.cloudflare.com" not in svg
