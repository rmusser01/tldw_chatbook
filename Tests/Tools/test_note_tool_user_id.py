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


@pytest.fixture(autouse=True)
def _clear_notes_service_cache():
    """Drop the module-global service cache around every test (task-692).

    ``_notes_service()`` caches the built service so the tool path stops
    re-opening the DB per call. That cache is module state: without this,
    a real service built by one test could be served to a later test that
    monkeypatches ``NotesInteropService``, and the fake would never be
    used. Today each test happens to get its own config path, so the key
    differs -- this does not rely on that accident.
    """
    nmt._reset_notes_service_cache()
    yield
    nmt._reset_notes_service_cache()


@pytest.fixture
def fake_service(monkeypatch, tmp_path):
    from types import SimpleNamespace

    stub = SimpleNamespace(db_path_str=str(tmp_path / "stub.db"))
    monkeypatch.setattr(nmt, "_default_db_provider", lambda: stub)
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


@pytest.mark.asyncio
async def test_note_tools_reuse_one_service_across_calls(monkeypatch, fake_service):
    """task-692: the service (and therefore its per-user CharactersRAGDB
    cache) is built once, not once per tool call.

    Before this, every ``execute()`` constructed its own service, whose
    ``_db_instances`` cache is an INSTANCE attribute -- so ``_get_db``
    missed every time and opened a fresh ``CharactersRAGDB`` (a real DB
    open plus schema-version check) on each call, then threw it away.
    """
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    built = []
    real_init = _FakeNotesService.__init__

    def counting_init(self, **kwargs):
        built.append(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(_FakeNotesService, "__init__", counting_init)

    await nmt.CreateNoteTool().execute(title="a", content="c")
    await nmt.CreateNoteTool().execute(title="b", content="c")
    await nmt.SearchNotesTool().execute(query="q")

    assert len(built) == 1, f"expected one service build, got {len(built)}"


@pytest.mark.asyncio
async def test_service_cache_rebuilds_when_the_supplied_db_changes(
    monkeypatch, fake_service, tmp_path
):
    """The cache is keyed on the DB the host supplies, so handing a tool a
    different database must NOT serve a service bound to the previous one."""
    from types import SimpleNamespace

    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    built = []
    real_init = _FakeNotesService.__init__

    def counting_init(self, **kwargs):
        built.append(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(_FakeNotesService, "__init__", counting_init)

    first = SimpleNamespace(db_path_str=str(tmp_path / "one" / "c.db"))
    second = SimpleNamespace(db_path_str=str(tmp_path / "two" / "c.db"))

    await nmt.CreateNoteTool(db_provider=lambda: first).execute(title="a", content="c")
    assert len(built) == 1
    await nmt.CreateNoteTool(db_provider=lambda: first).execute(title="b", content="c")
    assert len(built) == 1
    await nmt.CreateNoteTool(db_provider=lambda: second).execute(title="b", content="c")
    assert len(built) == 2, "a different DB must rebuild the cached service"


# -- TASK-36945.4: the real service, fed by the host-supplied database -------


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_note_tools_use_the_real_service_on_the_supplied_database(tmp_path):
    """Tool.execute -> real NotesInteropService -> real DB, nothing faked."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "tool-path.db", "task-36945-4")
    nmt._reset_notes_service_cache()
    try:
        created = await nmt.CreateNoteTool(db_provider=lambda: db).execute(
            title="provider note", content="stored via the tool path"
        )
        assert "error" not in created, created
        found = await nmt.SearchNotesTool(db_provider=lambda: db).execute(
            query="provider"
        )
        assert "error" not in found, found
        assert created["note_id"] in str(found)
    finally:
        nmt._reset_notes_service_cache()


_FRESH_PROCESS_SCRIPT = """
import asyncio, json, os
from pathlib import Path
import tldw_chatbook.config as config
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

assert config.chachanotes_db is None, "something already created the DB"
real = config.get_cli_setting
config.get_cli_setting = lambda s, k, d=None: True if s == "tools" else real(s, k, d)
tool = BuiltinToolProvider()._tools["create_note"]
result = asyncio.run(tool.execute(title="fresh", content="process"))
db = config.chachanotes_db
print(json.dumps({"result": result, "db_path": str(Path(db.db_path_str).resolve()) if db else None}))
"""


def test_note_tools_succeed_in_a_process_where_nothing_created_the_db():
    """AC#3: base failed with 'No unified database template for
    NotesInteropService' because config.chachanotes_db was still None."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "error" not in out["result"], out
    assert out["result"]["note_id"]
    # Never the real profile: the DB must sit under the scratch HOME/XDG.
    scratch = Path(env["HOME"]).resolve()
    assert Path(out["db_path"]).is_relative_to(scratch), out["db_path"]
