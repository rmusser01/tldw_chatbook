"""TASK-545 P2: note tools must attribute notes to the real configured user.

Both tools hardcoded `user_id="default_user"` with a "Would be actual user
in production" comment, while the app resolves `notes_user_id` from
`load_settings()["USERS_NAME"]`.

`user_id` is an ATTRIBUTION value, not a visibility partition -- the `notes`
table has no user column, and `NotesInteropService.add_note` documents that
"the user_id will be used as the client_id". So a user who set
[general] users_name still SAW their agent-created notes; those notes were
attributed to a fabricated "default_user" client, which is what sync and
conflict resolution key off.
"""

import pathlib

import pytest

import tldw_chatbook.Tools.note_management_tools as nmt


class _FakeNotesService:
    """Captures the user_id every call was made with."""

    def __init__(self, **kwargs):
        _FakeNotesService.last = self
        self.calls = []

    def add_note(self, user_id, title, content):
        self.calls.append(("add_note", user_id))
        return "note-1"

    def get_note_by_id(self, user_id, note_id):
        self.calls.append(("get_note_by_id", user_id))
        return {"id": note_id, "version": 1}

    def update_note(self, user_id, note_id, update_data, expected_version):
        self.calls.append(("update_note", user_id))
        return True


@pytest.fixture
def fake_service(monkeypatch):
    monkeypatch.setattr(nmt, "NotesInteropService", _FakeNotesService)
    return _FakeNotesService


@pytest.mark.asyncio
async def test_create_note_uses_the_configured_user(monkeypatch, fake_service):
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    result = await nmt.CreateNoteTool().execute(title="t", content="c")
    assert "error" not in result
    assert fake_service.last.calls == [("add_note", "alice")]


@pytest.mark.asyncio
async def test_update_note_uses_the_configured_user_on_every_call(
    monkeypatch, fake_service
):
    """Both the existence check and the write must use the same id -- a
    mismatch would read one user's note and write another's."""
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    result = await nmt.UpdateNoteTool().execute(note_id="n1", title="t2")
    assert "error" not in result
    assert fake_service.last.calls == [
        ("get_note_by_id", "alice"),
        ("update_note", "alice"),
    ]


def test_resolver_reads_users_name_from_load_settings(monkeypatch):
    monkeypatch.setattr(nmt, "load_settings", lambda: {"USERS_NAME": "bob"})
    assert nmt._resolve_user_id() == "bob"


def test_resolver_honors_the_env_var_override(monkeypatch):
    """The real value is os.getenv("USERS_NAME", <toml>) resolved INSIDE
    load_settings -- reading TOML directly would diverge from
    app.notes_user_id and stamp a third distinct client_id."""
    import tldw_chatbook.config as config_module

    monkeypatch.setenv("USERS_NAME", "env_user")
    try:
        settings = config_module.load_settings(force_reload=True)
        assert settings["USERS_NAME"] == "env_user"
        monkeypatch.setattr(nmt, "load_settings", lambda: settings)
        assert nmt._resolve_user_id() == "env_user"
    finally:
        # force_reload=True writes config.py's process-global settings cache.
        # conftest's autouse per-test TLDW_CONFIG_PATH currently masks the
        # leak by changing the cache key, but reset explicitly rather than
        # relying on that -- the convention three other test files follow
        # (e.g. Tests/test_user_data_dir_isolation.py).
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


def test_resolver_falls_back_when_settings_are_unavailable(monkeypatch):
    """A tool must never crash because config could not be read."""

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(nmt, "load_settings", boom)
    assert nmt._resolve_user_id() == "default_user"


# -- Importing the module must not touch the filesystem ----------------------


@pytest.fixture
def reloadable_module():
    """Restore `nmt`'s own bindings after a test reloads it.

    `importlib.reload()` re-executes `from ..config import
    get_chachanotes_db_path`, so a reload performed while that function is
    patched rebinds it INSIDE `nmt` to the patched object. `monkeypatch`
    then restores `config`, not `nmt` -- leaving `nmt` bound to the fake for
    the rest of the pytest session. That poisoned every later test in the
    process that created a note (caught when it broke
    `test_builtin_gate_live_tools.py`).

    Reloading once more here, after every patch is undone, re-binds `nmt`
    to the real functions.
    """
    yield nmt
    import importlib

    importlib.reload(nmt)


def test_importing_the_module_does_not_resolve_the_db_path(reloadable_module):
    """Qodo/whole-branch finding: this was computed at module scope.

    `get_chachanotes_db_path()` reaches `get_user_data_dir()`, which
    `mkdir`s. Evaluating that at import scope meant merely importing this
    module created a directory, and an unwritable $HOME made the import
    RAISE -- which `tool_catalog`'s registration loop turns into
    "create_note is silently missing" rather than a normal tool error.
    """
    import importlib
    from unittest.mock import patch

    import tldw_chatbook.config as config_module

    def landmine():
        raise AssertionError("get_chachanotes_db_path() called at import time")

    # A context manager, not monkeypatch: the patch must be gone BEFORE the
    # fixture's restoring reload runs, or that reload re-binds the fake.
    with patch.object(config_module, "get_chachanotes_db_path", landmine):
        importlib.reload(nmt)  # the import itself must not call it

        # ...but it must still be reachable lazily, per call.
        with patch.object(
            nmt, "get_chachanotes_db_path", lambda: pathlib.Path("/tmp/x/db.sqlite")
        ):
            assert str(nmt._notes_db_base_dir()) == "/tmp/x"


def test_an_unwritable_data_dir_does_not_break_the_import(reloadable_module):
    """The failure mode that made this a bug rather than a style nit."""
    import importlib
    from unittest.mock import patch

    import tldw_chatbook.config as config_module

    def boom():
        raise PermissionError("read-only file system")

    with patch.object(config_module, "get_chachanotes_db_path", boom):
        importlib.reload(nmt)  # must not raise
        assert nmt.CreateNoteTool().name == "create_note"


def test_a_reload_does_not_leak_a_patched_binding(reloadable_module):
    """Regression guard for the leak itself, not just its symptom."""
    import importlib
    from unittest.mock import patch

    import tldw_chatbook.config as config_module

    with patch.object(
        config_module, "get_chachanotes_db_path", lambda: pathlib.Path("/fake/db")
    ):
        importlib.reload(nmt)
    importlib.reload(nmt)  # what the fixture does, asserted inline

    assert nmt.get_chachanotes_db_path is config_module.get_chachanotes_db_path
    assert "/fake" not in str(nmt._notes_db_base_dir())
