"""Undo observes the current app attachment without changing refusal ordering."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Workspaces.change_revert import RevertRefusedError
from tldw_chatbook.Workspaces.change_revert_ownership import _UndoConfirmation


def _confirmation(app, *, attached=True):
    confirmation = object.__new__(_UndoConfirmation)
    confirmation.app = app
    confirmation.runtime = SimpleNamespace(
        _disposed=False, _app=app if attached else None
    )
    confirmation.issuer = SimpleNamespace(closed=False)
    confirmation.provider = None
    return confirmation


def test_missing_undo_app_runtime_is_refused():
    with pytest.raises(RevertRefusedError, match="original Undo source changed"):
        _confirmation(SimpleNamespace()).check_memory()


def test_undo_app_descriptor_is_read_once_after_earlier_guard():
    class App:
        reads = 0

        @property
        def console_runtime(self):
            self.reads += 1
            raise AttributeError("removed")

    app = App()
    with pytest.raises(RevertRefusedError, match="original Undo source changed"):
        _confirmation(app, attached=False).check_memory()
    assert app.reads == 0
    with pytest.raises(RevertRefusedError, match="original Undo source changed"):
        _confirmation(app).check_memory()
    assert app.reads == 1


def test_undo_app_descriptor_other_exception_is_not_masked():
    class App:
        @property
        def console_runtime(self):
            raise ValueError("original descriptor error")

    with pytest.raises(ValueError, match="original descriptor error"):
        _confirmation(App()).check_memory()
