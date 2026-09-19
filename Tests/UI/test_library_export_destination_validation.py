"""TASK-31204: the normalized Library export destination must flow through
the shared path-validation seam.

``_apply_library_export_destination`` validated the RAW ``FileSave`` pick but
then rewrote its suffix to ``.zip`` and stored THAT path without re-checking
-- the path the export actually writes never saw ``validate_path_simple``
(Qodo finding on PR #2344, relocated verbatim from LibraryScreen).

These tests wire the minimum object graph the method touches (state accessor,
screen stub) rather than mounting the full Library shell.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tldw_chatbook.UI.Library_Modules import library_export_controller as lec


def _controller(form: dict) -> lec.LibraryExportController:
    """A bare controller whose only live wiring is what this method reads."""
    screen = SimpleNamespace(refresh=lambda **kwargs: None)
    controller = lec.LibraryExportController.__new__(lec.LibraryExportController)
    controller._screen = screen
    controller._export_state_accessor = lambda: SimpleNamespace(form=form)
    return controller


def test_normalized_destination_flows_through_the_validation_seam():
    """The ``.zip``-normalized path (the one actually written) is validated,
    not only the raw dialog pick."""
    form: dict = {}
    controller = _controller(form)

    with patch.object(
        lec, "validate_path_simple", side_effect=lambda p, **kw: p
    ) as validate:
        controller._apply_library_export_destination(Path("/tmp/notes.txt"))

    validated = [call.args[0] for call in validate.call_args_list]
    assert Path("/tmp/notes.zip") in validated, (
        "the normalized destination must itself pass through "
        f"validate_path_simple; only saw {validated}"
    )
    assert form["destination"] == "/tmp/notes.zip"


def test_rejecting_the_normalized_path_clears_destination_with_inline_reason():
    """If validation rejects the normalized path, the destination is NOT
    retained and the form carries the reason without losing its other fields."""
    form = {
        "destination": "/tmp/previous.zip",
        "destination_exists": True,
        "name": "Keep this title",
    }
    controller = _controller(form)

    def _validate(path, **kwargs):
        if path.name == "notes.zip":
            raise ValueError("hostile normalized component")
        return path

    with patch.object(lec, "validate_path_simple", side_effect=_validate):
        controller._apply_library_export_destination(Path("/tmp/notes.txt"))

    assert form == {
        "destination": "",
        "destination_exists": False,
        "name": "Keep this title",
        "destination_error": "Can't save there: hostile normalized component",
    }


def test_hostile_raw_pick_is_rejected_with_a_reason():
    """The pre-existing first gate: a hostile raw pick never reaches the form.

    PR #2634 review: drives the REAL shared validator (no stub), so the
    traversal-shaped input exercises path_validation.py's actual checks.
    """
    form: dict = {}
    controller = _controller(form)

    controller._apply_library_export_destination(Path("/tmp/../../..evil.txt"))

    assert form["destination"] == ""
    assert form["destination_exists"] is False
    assert form["destination_error"].startswith("Can't save there:")
    assert "dangerous pattern" in form["destination_error"]


def test_uppercase_zip_pick_normalizes_to_the_writers_path(tmp_path):
    """PR #2634 review: a ``foo.ZIP`` pick must store the lowercase
    ``foo.zip`` the writer actually replaces, so the overwrite line names
    the real file."""
    form: dict = {}
    controller = _controller(form)

    with patch.object(lec, "validate_path_simple", side_effect=lambda p, **kw: p):
        controller._apply_library_export_destination(tmp_path / "notes.ZIP")

    assert form["destination"] == str(tmp_path / "notes.zip")
