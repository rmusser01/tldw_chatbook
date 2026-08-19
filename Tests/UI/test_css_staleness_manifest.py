"""Content-hash staleness for the generated CSS sheets (TASK-18910).

Measured motivation: the boot-time rebuild costs ~0.7 s (interpreter spawn
+ full package AST scan) and fired on every boot where any input's *mtime*
moved past the build -- including the no-content-change moves a branch
switch, ``git checkout`` of a file, or a stash pop produce. With a content
manifest, an mtime move is only a hint to hash-check; identical content is
not stale. The manifest is also authoritative in the other direction: a
content change is stale even when the sheets themselves were regenerated
elsewhere (the pure-mtime rule's masking gap).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tldw_chatbook.css import build_css
from tldw_chatbook.app import _generated_css_is_stale


@pytest.fixture()
def css_tree(tmp_path: Path) -> Path:
    """A minimal source tree: package root with css/ inputs and outputs."""
    css_dir = tmp_path / "css"
    css_dir.mkdir()
    (css_dir / "core").mkdir()
    (css_dir / "core" / "_base.tcss").write_text("/* base */\n")
    (css_dir / "tldw_cli_modular.tcss").write_text("/* bundle */\n")
    (css_dir / build_css.WIDGET_DEFAULTS_SELF_FILENAME).write_text("/* w1 */\n")
    (css_dir / build_css.WIDGET_DEFAULTS_SCOPED_FILENAME).write_text("/* w2 */\n")
    (css_dir / build_css.SCREEN_CSS_SELF_FILENAME).write_text("/* s1 */\n")
    (css_dir / build_css.SCREEN_CSS_SCOPED_FILENAME).write_text("/* s2 */\n")
    return tmp_path


def _write_manifest(css_tree: Path, entries: dict[str, list]) -> None:
    (css_tree / "css" / build_css.BUILD_MANIFEST_FILENAME).write_text(
        json.dumps(entries, indent=2, sort_keys=True)
    )


def _manifest_with_current(css_tree: Path) -> dict[str, list]:
    css_dir = css_tree / "css"
    source = css_dir / "core" / "_base.tcss"
    return {
        "css/core/_base.tcss": [build_css._file_sha256(source), source.stat().st_mtime]
    }


def _bump_mtime(path: Path, past_manifest: bool = True) -> None:
    now = time.time()
    offset = 10 if past_manifest else -10
    os.utime(path, (now + offset, now + offset))


def test_mtime_move_with_unchanged_content_is_not_stale(css_tree: Path) -> None:
    """The TASK-18910 case: checkout/branch-switch bumps mtimes, content identical."""
    _write_manifest(css_tree, _manifest_with_current(css_tree))
    _bump_mtime(css_tree / "css" / "core" / "_base.tcss")

    stale, reason = _generated_css_is_stale(css_tree)
    assert not stale, f"unchanged content must not be stale (reason={reason!r})"


def test_real_content_change_is_stale(css_tree: Path) -> None:
    """An actual edit must still trigger the rebuild."""
    _write_manifest(css_tree, _manifest_with_current(css_tree))
    module = css_tree / "css" / "core" / "_base.tcss"
    module.write_text("/* base EDITED */\n")

    stale, reason = _generated_css_is_stale(css_tree)
    assert stale
    assert "_base.tcss" in reason


def test_content_change_masked_by_newer_sheets_is_stale(css_tree: Path) -> None:
    """The pure-mtime rule's gap: pull brings NEW sheets, edit slipped in unbuilt.

    Sheets get fresh mtimes (regenerated upstream), so the edited source is
    not 'newer than the build' -- mtime alone never fired. The manifest's
    recorded mtime still predates the edit, so the hash arbitrates.
    """
    _write_manifest(css_tree, _manifest_with_current(css_tree))
    module = css_tree / "css" / "core" / "_base.tcss"
    module.write_text("/* base EDITED */\n")
    # The pull also rewrote every generated sheet (all mtimes now).
    for name in (
        "tldw_cli_modular.tcss",
        build_css.WIDGET_DEFAULTS_SELF_FILENAME,
        build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
        build_css.SCREEN_CSS_SELF_FILENAME,
        build_css.SCREEN_CSS_SCOPED_FILENAME,
    ):
        _bump_mtime(css_tree / "css" / name)

    stale, reason = _generated_css_is_stale(css_tree)
    assert stale, (
        "a real edit must be caught even when sheets were regenerated elsewhere"
    )
    assert "_base.tcss" in reason


def test_deleted_manifest_input_is_stale(css_tree: Path) -> None:
    """Deleting a recorded input must rebuild (the old rule never saw this)."""
    _write_manifest(css_tree, _manifest_with_current(css_tree))
    (css_tree / "css" / "core" / "_base.tcss").unlink()

    stale, reason = _generated_css_is_stale(css_tree)
    assert stale
    assert "deleted" in reason


def test_gained_bundled_css_declaration_is_stale(css_tree: Path) -> None:
    """A NEW carrier module (not in the manifest) must trigger a rebuild."""
    _write_manifest(css_tree, _manifest_with_current(css_tree))
    carrier = css_tree / "Widgets" / "demo_widget.py"
    carrier.parent.mkdir()
    carrier.write_text("class Demo:\n    BUNDLED_CSS = 'Demo { color: red; }'\n")

    stale, reason = _generated_css_is_stale(css_tree)
    assert stale
    assert "gained a BUNDLED_CSS declaration" in reason


def test_missing_manifest_falls_back_to_mtime_rule(css_tree: Path) -> None:
    """No manifest (first boot after this change): rebuild once, then heal."""
    _bump_mtime(css_tree / "css" / "core" / "_base.tcss")

    stale, _ = _generated_css_is_stale(css_tree)
    assert stale


def test_malformed_manifest_falls_back_to_mtime_rule(css_tree: Path) -> None:
    """A corrupt manifest must never SKIP a needed rebuild (fail-safe)."""
    (css_tree / "css" / build_css.BUILD_MANIFEST_FILENAME).write_text("{not json")
    _bump_mtime(css_tree / "css" / "core" / "_base.tcss")

    stale, _ = _generated_css_is_stale(css_tree)
    assert stale


def test_write_build_manifest_records_hash_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder records [sha256, mtime] per input beside the sheets."""
    css_dir = tmp_path / "css"
    css_dir.mkdir()
    (css_dir / "core").mkdir()
    source = css_dir / "core" / "_base.tcss"
    source.write_text("/* base */\n")

    monkeypatch.setattr(build_css, "CSS_MODULES", ["core/_base.tcss"])
    monkeypatch.setattr(
        build_css.widget_css,
        "iter_blocks",
        lambda package_root, attr: [],
    )

    build_css.write_build_manifest(css_dir)

    manifest = json.loads((css_dir / build_css.BUILD_MANIFEST_FILENAME).read_text())
    entry = manifest["css/core/_base.tcss"]
    assert entry[0] == build_css._file_sha256(source)
    assert entry[1] == source.stat().st_mtime
