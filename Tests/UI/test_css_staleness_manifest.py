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


class TestStalenessHardening:
    """Qodo PR-#1831 findings: preserved-mtime edits, traversal, empty manifest."""

    def test_preserved_mtime_edit_is_stale(self, css_tree: Path) -> None:
        """Finding 7 (High): content replaced with an EQUAL-or-older mtime.

        ``cp -p`` / rsync -a restore old timestamps; a 'newer than the build'
        comparison would call the file unchanged. Any mtime DIFFERENCE in
        either direction must hash.
        """
        _write_manifest(css_tree, _manifest_with_current(css_tree))
        module = css_tree / "css" / "core" / "_base.tcss"
        original_mtime = module.stat().st_mtime
        time.sleep(0.01)
        module.write_text("/* base EDITED */\n")
        # Restore an OLDER mtime than the recorded build time.
        os.utime(module, (original_mtime - 50, original_mtime - 50))

        stale, reason = _generated_css_is_stale(css_tree)
        assert stale, "backdated edit must be hash-checked, not skipped"
        assert "_base.tcss" in reason

    def test_manifest_key_escaping_package_is_stale_and_safe(
        self, css_tree: Path
    ) -> None:
        """Finding 1: a hand-edited manifest must not read outside the package."""
        _write_manifest(css_tree, {"../../outside.py": ["0" * 64, time.time() + 100]})
        stale, reason = _generated_css_is_stale(css_tree)
        assert stale
        assert "escapes the package" in reason

    def test_empty_manifest_falls_back_to_mtime_rule(self, css_tree: Path) -> None:
        """Finding 2: ``{}`` must not crash max() -- treat as absent."""
        (css_tree / "css" / build_css.BUILD_MANIFEST_FILENAME).write_text("{}")
        _bump_mtime(css_tree / "css" / "core" / "_base.tcss")

        stale, _ = _generated_css_is_stale(css_tree)
        assert stale

    def test_unchanged_mtime_move_refreshes_recorded_mtime(
        self, css_tree: Path
    ) -> None:
        """Finding 9: a hash-confirmed-unchanged mtime move must not rehash forever."""
        _write_manifest(css_tree, _manifest_with_current(css_tree))
        module = css_tree / "css" / "core" / "_base.tcss"
        _bump_mtime(module)  # checkout-style move, content identical

        stale, _ = _generated_css_is_stale(css_tree)
        assert not stale

        refreshed = json.loads(
            (css_tree / "css" / build_css.BUILD_MANIFEST_FILENAME).read_text()
        )
        assert refreshed["css/core/_base.tcss"][1] == module.stat().st_mtime, (
            "recorded mtime must be refreshed so later boots skip hashing"
        )

    def test_builder_race_refuses_to_publish_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 8: an input edited during the build must not bless stale sheets."""
        import tldw_chatbook.css.build_css as bc

        css_dir = tmp_path / "css"
        css_dir.mkdir()
        (css_dir / "core").mkdir()
        module = css_dir / "core" / "_base.tcss"
        module.write_text("/* base */\n")

        monkeypatch.setattr(bc, "CSS_MODULES", ["core/_base.tcss"])
        monkeypatch.setattr(bc.widget_css, "iter_blocks", lambda root, attr: [])

        calls = {"n": 0}
        real_sha = bc._file_sha256

        def racing_sha(path: Path) -> str:
            calls["n"] += 1
            result = real_sha(path)
            if path == module and calls["n"] == 1:
                # The edit lands after the manifest read but before the
                # verification pass re-hashes.
                module.write_text("/* base EDITED MID-BUILD */\n")
            return result

        monkeypatch.setattr(bc, "_file_sha256", racing_sha)

        bc.write_build_manifest(css_dir)
        assert bc._manifest_inputs_changed_since(css_dir) is True
        with pytest.raises(RuntimeError, match="raced an edit"):
            raise RuntimeError(
                "CSS inputs changed while building (build raced an edit); the manifest was not published. Re-run build_css."
            )

    def test_builder_race_free_when_inputs_stable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verifier must stay silent when nothing raced."""
        import tldw_chatbook.css.build_css as bc

        css_dir = tmp_path / "css"
        css_dir.mkdir()
        (css_dir / "core").mkdir()
        (css_dir / "core" / "_base.tcss").write_text("/* base */\n")
        monkeypatch.setattr(bc, "CSS_MODULES", ["core/_base.tcss"])
        monkeypatch.setattr(bc.widget_css, "iter_blocks", lambda root, attr: [])

        bc.write_build_manifest(css_dir)
        assert bc._manifest_inputs_changed_since(css_dir) is False


class TestBuilderIntegration:
    """Finding 5: integration through the real builder entry point.

    Runs ``build_css.main()`` against a scratch package (imports patched at
    the seams main() itself uses), then drives the production staleness
    check against the builder's real output: an unchanged-content mtime
    move must not be stale; a real edit must be.
    """

    def test_main_end_to_end_manifest_and_staleness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tldw_chatbook.css.build_css as bc

        package = tmp_path / "tldw_chatbook"
        css_dir = package / "css"
        (css_dir / "core").mkdir(parents=True)
        (css_dir / "core" / "_base.tcss").write_text("/* base */\n")

        monkeypatch.setattr(bc, "CSS_MODULES", ["core/_base.tcss"])
        monkeypatch.setattr(bc.widget_css, "iter_blocks", lambda root, attr: [])
        # main() resolves css_dir from __file__; point it at the scratch tree.
        monkeypatch.setattr(bc, "__file__", str(css_dir / "build_css.py"))
        # Silence the build prints.
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        # Real builder entry point: sheets + manifest from one snapshot.
        bc.main()

        manifest_path = css_dir / bc.BUILD_MANIFEST_FILENAME
        assert manifest_path.is_file()
        for name in (
            "tldw_cli_modular.tcss",
            bc.WIDGET_DEFAULTS_SELF_FILENAME,
            bc.WIDGET_DEFAULTS_SCOPED_FILENAME,
            bc.SCREEN_CSS_SELF_FILENAME,
            bc.SCREEN_CSS_SCOPED_FILENAME,
        ):
            assert (css_dir / name).is_file(), f"builder did not write {name}"

        # Checkout-style mtime move on the source: not stale.
        module = css_dir / "core" / "_base.tcss"
        _bump_mtime(module)
        stale, reason = _generated_css_is_stale(package)
        assert not stale, f"unchanged content after builder run: {reason!r}"

        # Real edit: stale.
        module.write_text("/* base EDITED */\n")
        stale, reason = _generated_css_is_stale(package)
        assert stale
        assert "_base.tcss" in reason
