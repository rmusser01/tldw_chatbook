"""Destination-validation tests for scripts/regen_approval_card_svg.py (task-32290, Qodo #2).

``main()`` must reject a CLI-supplied destination that escapes the
repository's ``Docs/`` directory -- via traversal or an absolute path
outside the tree -- before anything is written, and must still accept (and
write to) a path that stays under ``Docs/``.
"""

from __future__ import annotations

from importlib import import_module, util


def _entry():
    spec = util.find_spec("scripts.regen_approval_card_svg")
    assert spec is not None, "regen_approval_card_svg script is unavailable"
    return import_module("scripts.regen_approval_card_svg")


def test_traversal_argument_rejected(monkeypatch):
    mod = _entry()
    target = mod._DOCS_ROOT.parent.parent / "etc" / "x.svg"
    monkeypatch.setattr(
        "sys.argv", ["regen_approval_card_svg.py", "../../etc/x.svg"]
    )
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
    saved: list[str | None] = []
    monkeypatch.setattr(
        "textual.app.App.save_screenshot",
        lambda self, filename=None, path=None, time_format=None: saved.append(
            filename
        )
        or filename,
    )
    monkeypatch.setattr("sys.argv", ["regen_approval_card_svg.py", str(dest)])
    assert mod.main() == 0
    assert saved == [str(dest)]
    assert not dest.exists()
