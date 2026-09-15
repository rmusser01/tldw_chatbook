"""Contracts for the shared last-used picker directory (task-32174).

PR #2554 review: the remembered directory is persisted user state read back
from ``config.toml``, so it is a trust boundary -- it must be validated
through ``Utils/path_validation.py`` before any filesystem probe or picker
use, and the write for one key must be latest-wins rather than
last-worker-to-finish.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Library import library_browse_location as module
from tldw_chatbook.Library.library_browse_location import (
    claim_browse_directory,
    picker_recent_context,
    remember_browse_directory,
    validated_browse_directory,
)


def _capture_writes(monkeypatch, saved: list, *, succeeds: bool = True):
    """Patch the ONE config-write seam this module uses.

    task-32643 folded the recent-roots list into the same write as the start
    directory, so the seam is now ``save_settings_to_cli_config`` (one dict of
    sections) rather than ``save_setting_to_cli_config`` (one key). These pins
    follow the rename: a stale patch would leave the real writer running and
    assert against a list nothing ever appended to.
    """

    def _write(settings):
        saved.append(settings)
        return succeeds

    monkeypatch.setattr(module, "save_settings_to_cli_config", _write)


def test_a_real_directory_comes_back_normalized(tmp_path):
    """The happy path: an existing directory survives, fully resolved."""
    assert validated_browse_directory(str(tmp_path)) == tmp_path.resolve()


@pytest.mark.parametrize(
    "remembered",
    [
        None,
        "",
        "../../etc",  # traversal
        "relative/dir",  # not absolute
        "/definitely/not/here/32174",  # gone since it was remembered
        "/tmp/nul\x00byte",  # malformed
        12345,  # not even a path
    ],
)
def test_an_unusable_remembered_value_is_refused(remembered):
    """Every malformed spelling falls back to the caller's own default."""
    assert validated_browse_directory(remembered) is None


def test_a_remembered_file_is_not_offered_as_a_directory(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# hi", encoding="utf-8")
    assert validated_browse_directory(str(note)) is None


def test_a_superseded_selection_cannot_overwrite_a_newer_one(tmp_path, monkeypatch):
    """Two picks in flight: the newer directory is the one that survives."""
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    saved: list[dict] = []
    _capture_writes(monkeypatch, saved)

    first = claim_browse_directory("library.test_32174", "last_directory")
    second = claim_browse_directory("library.test_32174", "last_directory")
    # The newer worker wins the race, then the older one finishes late.
    remember_browse_directory("library.test_32174", "last_directory", newer, second)
    remember_browse_directory("library.test_32174", "last_directory", older, first)

    assert saved == [{"library.test_32174": {"last_directory": str(newer)}}]


def test_a_picked_file_remembers_its_parent_directory(tmp_path, monkeypatch):
    picked = tmp_path / "note.md"
    picked.write_text("# hi", encoding="utf-8")
    saved: list[dict] = []
    _capture_writes(monkeypatch, saved)

    generation = claim_browse_directory("library.test_32174_parent", "last_directory")
    remember_browse_directory(
        "library.test_32174_parent", "last_directory", picked, generation
    )

    assert saved == [
        {"library.test_32174_parent": {"last_directory": str(tmp_path)}}
    ]


def test_a_refused_config_write_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    """A failed save used to leave no trace at all -- the picker just
    silently reopened somewhere else next time."""
    _capture_writes(monkeypatch, [], succeeds=False)
    logged: list[str] = []
    sink_id = module.logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        generation = claim_browse_directory("library.test_32174_fail", "last_directory")
        remember_browse_directory(
            "library.test_32174_fail", "last_directory", tmp_path, generation
        )
    finally:
        module.logger.remove(sink_id)

    assert any("library.test_32174_fail" in line for line in logged)


def test_a_raising_config_write_is_reported_not_swallowed(tmp_path, monkeypatch):
    """The Folder files worker used to ``except Exception: pass``."""

    def _boom(*args, **kwargs):
        raise RuntimeError("config is on fire")

    monkeypatch.setattr(module, "save_settings_to_cli_config", _boom)
    logged: list[str] = []
    sink_id = module.logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        generation = claim_browse_directory("library.test_32174_boom", "last_directory")
        # Must not propagate: this runs inside a Textual worker.
        remember_browse_directory(
            "library.test_32174_boom", "last_directory", tmp_path, generation
        )
    finally:
        module.logger.remove(sink_id)

    assert any("library.test_32174_boom" in line for line in logged)


def test_home_is_never_used_as_a_silent_stand_in_for_a_real_directory():
    """``validated_browse_directory`` reports refusal; the fallback choice
    stays with each caller (all three use ``Path.home()``)."""
    assert validated_browse_directory(str(Path.home())) == Path.home().resolve()


# --- task-32251 AC#5: the configured notes folder beats bare $HOME ----------


def test_a_remembered_directory_still_wins(tmp_path, monkeypatch):
    from tldw_chatbook.Library.library_browse_location import browse_start_directory

    remembered = tmp_path / "remembered"
    configured = tmp_path / "configured"
    remembered.mkdir()
    configured.mkdir()
    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda section, key=None, default=None: str(configured),
    )
    assert browse_start_directory(str(remembered)) == remembered.resolve()


def test_the_configured_notes_folder_is_offered_before_home(tmp_path, monkeypatch):
    """AC#5: with nothing remembered, open where the user keeps notes."""
    from tldw_chatbook.Library.library_browse_location import browse_start_directory

    configured = tmp_path / "configured"
    configured.mkdir()
    seen: list[tuple] = []

    def fake_get(section, key=None, default=None):
        seen.append((section, key))
        return str(configured)

    monkeypatch.setattr(module, "get_cli_setting", fake_get)
    assert browse_start_directory(None) == configured.resolve()
    assert seen == [("notes", "sync_directory")]


def test_home_is_still_the_last_resort(monkeypatch):
    from tldw_chatbook.Library.library_browse_location import browse_start_directory

    monkeypatch.setattr(
        module, "get_cli_setting", lambda section, key=None, default=None: None
    )
    assert browse_start_directory(None) == Path.home()


def test_an_unusable_configured_notes_folder_is_skipped(monkeypatch):
    from tldw_chatbook.Library.library_browse_location import browse_start_directory

    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda section, key=None, default=None: "~/definitely/not/here/32251",
    )
    assert browse_start_directory(None) == Path.home()


def test_a_notes_door_also_records_the_root_it_returned(tmp_path, monkeypatch):
    """task-32643 AC#3: the recents list rides the write that already happens.

    One config write per selection, carrying both facts -- the start directory
    this picker reopens at, and the roots it offers on ctrl+r. Asserted on the
    single settings dict so a second rewrite per pick would fail here.
    """
    chosen = tmp_path / "vault"
    chosen.mkdir()
    saved: list[dict] = []
    _capture_writes(monkeypatch, saved)

    context = picker_recent_context("library.test_32643")
    generation = claim_browse_directory("library.test_32643", "last_directory")
    remember_browse_directory(
        "library.test_32643",
        "last_directory",
        chosen,
        generation,
        recent_context=context,
    )

    assert len(saved) == 1, "one selection must not rewrite config.toml twice"
    settings = saved[0]
    assert settings["library.test_32643"] == {"last_directory": str(chosen)}
    recent = settings["filepicker"][f"recent_{context}"]
    assert [entry["path"] for entry in recent] == [str(chosen.resolve())]
    assert recent[0]["type"] == "directory"


def test_picking_a_FILE_records_where_to_reopen_but_offers_no_new_root(
    tmp_path, monkeypatch
):
    """Review round 1, finding 2: the two facts are not the same fact.

    Import once can return a file, and the start directory rightly takes that
    file's PARENT -- that is where to reopen. The recents list must not: its
    rows are offered as answers (`base_dialog._on_recent_selected` dismisses
    with the row), so a folder the user only browsed THROUGH on the way to a
    file would come back as a one-keystroke whole-folder adoption of a folder
    they never chose.
    """
    folder = tmp_path / "browsed-through"
    folder.mkdir()
    picked_file = folder / "one-note.md"
    picked_file.write_text("x", encoding="utf-8")
    saved: list[dict] = []
    _capture_writes(monkeypatch, saved)

    context = picker_recent_context("library.notes_import")
    generation = claim_browse_directory("library.notes_import", "last_directory")
    remember_browse_directory(
        "library.notes_import",
        "last_directory",
        picked_file,
        generation,
        recent_context=context,
    )

    assert len(saved) == 1
    settings = saved[0]
    # Still reopens where the user was...
    assert settings["library.notes_import"] == {"last_directory": str(folder)}
    # ...and still offers nothing new to adopt wholesale.
    assert "filepicker" not in settings, (
        "a picked FILE must not add its parent to the roots ctrl+r offers as "
        f"answers; it wrote {settings.get('filepicker')!r}"
    )


def test_a_picker_that_offers_no_recents_records_none(tmp_path, monkeypatch):
    """Negative control: a blank ``recent_context`` writes no recents key.

    The Library ingest browser calls this function too and never shows a
    recents list; accumulating a key nothing reads would be config litter.
    """
    chosen = tmp_path / "somewhere"
    chosen.mkdir()
    saved: list[dict] = []
    _capture_writes(monkeypatch, saved)

    generation = claim_browse_directory("library.test_32643_none", "last_directory")
    remember_browse_directory(
        "library.test_32643_none", "last_directory", chosen, generation
    )

    assert saved == [{"library.test_32643_none": {"last_directory": str(chosen)}}]


def test_the_recents_context_is_derived_from_the_config_section():
    """One string decides both ends, so a caller cannot read one context and
    persist under another (task-32643 AC#3)."""
    assert picker_recent_context("library.notes_sync") == "library_notes_sync"
    assert picker_recent_context("file_notes") == "file_notes"
