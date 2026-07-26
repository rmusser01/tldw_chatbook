"""TASK-545 P2: note tools must write as the real configured user.

Both tools hardcoded `user_id="default_user"` with a "Would be actual user
in production" comment, while the app resolves `notes_user_id` from
`load_settings()["USERS_NAME"]`. A user who set [general] users_name got
agent-created notes in a bucket their Notes UI never reads.
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
    app.notes_user_id and create a third bucket."""
    import tldw_chatbook.config as config_module

    monkeypatch.setenv("USERS_NAME", "env_user")
    settings = config_module.load_settings(force_reload=True)
    assert settings["USERS_NAME"] == "env_user"
    monkeypatch.setattr(nmt, "load_settings", lambda: settings)
    assert nmt._resolve_user_id() == "env_user"


def test_resolver_falls_back_when_settings_are_unavailable(monkeypatch):
    """A tool must never crash because config could not be read."""

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(nmt, "load_settings", boom)
    assert nmt._resolve_user_id() == "default_user"
