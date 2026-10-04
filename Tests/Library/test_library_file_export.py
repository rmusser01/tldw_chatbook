"""TASK-34000.3 (N-11): the single-file export seam.

Pins the shared seam every one-file Library export goes through
(``UI/Library_Modules/library_file_export.py``): the replace prompt, the
atomic write, the remembered export folder, and -- AC#3 -- that the Report
artifact export and the Collections legacy-recovery export ask before
writing over an existing file. Those two are driven through the real
controller methods with a fake app that records what is pushed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

PRECIOUS = b"PRECIOUS USER FILE\n"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _precious(tmp_path: Path, name: str = "precious.md") -> Path:
    folder = tmp_path / "exp"
    folder.mkdir(exist_ok=True)
    destination = folder / name
    destination.write_bytes(PRECIOUS)
    return destination


class _FakeApp:
    """Records ``push_screen`` and ``notify`` instead of running Textual."""

    def __init__(self) -> None:
        self.pushed: list[tuple[object, object]] = []
        self.notices: list[tuple[str, str]] = []

    def push_screen(self, screen, callback=None, **_kwargs):
        self.pushed.append((screen, callback))

    def notify(self, message, severity="information", **_kwargs):
        self.notices.append((str(message), severity))


async def _answer(app: _FakeApp, answer):
    """Answer the one pushed prompt the way Textual would deliver it."""
    assert len(app.pushed) == 1, f"expected one prompt, saw {len(app.pushed)}"
    _, callback = app.pushed.pop()
    outcome = callback(answer)
    if inspect.isawaitable(outcome):
        await outcome


def _prompt_texts(dialog: ConfirmationDialog) -> tuple[str, str, str, str]:
    return (
        str(dialog.title),
        str(dialog.message),
        str(dialog.cancel_label),
        str(dialog.confirm_label),
    )


# --- the seam's primitives ----------------------------------------------------


def test_a_failed_write_leaves_the_existing_file_intact_and_no_temp_file(
    tmp_path, monkeypatch
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    destination = _precious(tmp_path)
    before = _md5(destination)

    def _refuse(*_args, **_kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", _refuse)
    with pytest.raises(OSError):
        write_export_text(destination, "new content that must never land")

    assert _md5(destination) == before
    assert destination.read_bytes() == PRECIOUS
    assert not list(destination.parent.glob(".*")), "temp file left behind"


def test_write_export_text_replaces_the_file_and_keeps_its_mode(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    destination = _precious(tmp_path)
    destination.chmod(0o600)

    write_export_text(destination, "# Exported\n")

    assert destination.read_text(encoding="utf-8") == "# Exported\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(destination.parent.glob(".*")), "temp file left behind"


def test_write_export_text_does_not_invent_a_missing_folder(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    destination = tmp_path / "typo" / "note.md"
    with pytest.raises(FileNotFoundError):
        write_export_text(destination, "x")
    assert not (tmp_path / "typo").exists()


@pytest.mark.asyncio
async def test_a_new_destination_is_written_without_asking(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        request_export_write,
    )

    app = _FakeApp()
    calls: list[str] = []
    outcome = request_export_write(
        app,
        tmp_path / "new.md",
        on_replace=lambda **kw: calls.append(("replace", kw["overwrite"])),
        on_cancel=lambda: calls.append("cancel"),
    )
    assert outcome is None
    assert calls == [("replace", False)], "a new file is published no-clobber"
    assert app.pushed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [False, None])
async def test_an_existing_destination_asks_with_cancel_first_and_a_negative_keeps_it(
    tmp_path, answer
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        request_export_write,
    )

    destination = _precious(tmp_path)
    app = _FakeApp()
    calls: list[str] = []
    request_export_write(
        app,
        destination,
        on_replace=lambda **kw: calls.append(("replace", kw["overwrite"])),
        on_cancel=lambda: calls.append("cancel"),
    )
    assert calls == []
    dialog, _ = app.pushed[0]
    assert isinstance(dialog, ConfirmationDialog)
    title, message, cancel_label, confirm_label = _prompt_texts(dialog)
    assert cancel_label == "Cancel" and confirm_label == "Replace"
    assert 'Replace "precious.md"' in message
    assert str(destination.parent) in message, message
    assert "Replace" in title

    await _answer(app, answer)
    assert calls == ["cancel"]
    assert destination.read_bytes() == PRECIOUS


@pytest.mark.asyncio
async def test_replace_runs_the_write_once(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        request_export_write,
    )

    destination = _precious(tmp_path)
    app = _FakeApp()
    calls: list[str] = []
    request_export_write(
        app,
        destination,
        on_replace=lambda **kw: calls.append(("replace", kw["overwrite"])),
        on_cancel=lambda: calls.append("cancel"),
    )
    await _answer(app, True)
    assert calls == [("replace", True)]


def test_a_dangling_symlink_destination_still_asks(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        export_destination_exists,
    )

    link = tmp_path / "gone.md"
    link.symlink_to(tmp_path / "missing-target.md")
    assert export_destination_exists(link)
    assert not export_destination_exists(tmp_path / "absent.md")


@pytest.mark.asyncio
async def test_a_symlink_to_an_existing_file_is_written_through_and_named(
    tmp_path, monkeypatch
):
    """Review Important #1: Replace writes the link's target, the link stays,
    and the prompt names the resolved folder plus the link that led there."""
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        replace_prompt_message,
        request_export_write,
        resolve_export_destination,
        write_export_text,
    )

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    vault = tmp_path / "vault"
    vault.mkdir()
    target = vault / "x.md"
    target.write_bytes(PRECIOUS)
    notes = tmp_path / "notes"
    notes.mkdir()
    link = notes / "x.md"
    link.symlink_to(target)

    resolved = resolve_export_destination(link)
    assert resolved == target.resolve()
    message = replace_prompt_message(resolved, link)
    assert 'Replace "x.md" in ~/vault?' in message, message
    assert '"x.md" in ~/notes links to it' in message, message

    app = _FakeApp()
    writes: list[bool] = []

    def _write(*, overwrite: bool = True) -> None:
        writes.append(overwrite)
        write_export_text(resolved, "# through the link\n", overwrite=overwrite)

    request_export_write(
        app, resolved, on_replace=_write, on_cancel=lambda: None, picked=link
    )
    assert writes == [], "an existing target must ask first"
    assert target.read_bytes() == PRECIOUS
    dialog, _ = app.pushed[0]
    assert "links to it" in str(dialog.message)

    await _answer(app, True)
    assert writes == [True]
    assert target.read_text(encoding="utf-8") == "# through the link\n"
    assert link.is_symlink(), "the link must survive; only its target changes"
    assert link.resolve() == target.resolve()
    assert link.read_text(encoding="utf-8") == "# through the link\n"


@pytest.mark.asyncio
async def test_a_file_that_appears_after_the_check_is_not_overwritten(tmp_path):
    """Review Minor #1: the "nothing there" branch publishes no-clobber; a
    FileExistsError falls into the prompt instead of a silent overwrite."""
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        request_export_write,
    )

    destination = tmp_path / "late.md"
    app = _FakeApp()
    calls: list[bool] = []

    def _write(*, overwrite: bool = True) -> None:
        calls.append(overwrite)
        if not overwrite:
            destination.write_bytes(PRECIOUS)  # someone else got there first
            raise FileExistsError(destination)

    outcome = request_export_write(
        app, destination, on_replace=_write, on_cancel=lambda: None
    )
    assert outcome is None
    assert calls == [False]
    assert destination.read_bytes() == PRECIOUS
    assert len(app.pushed) == 1, "the race must fall into the replace prompt"

    await _answer(app, True)
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_an_async_writer_hitting_the_race_also_falls_into_the_prompt(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        request_export_write,
    )

    app = _FakeApp()
    calls: list[bool] = []

    async def _write(*, overwrite: bool = True) -> None:
        calls.append(overwrite)
        if not overwrite:
            raise FileExistsError("appeared")

    outcome = request_export_write(
        app, tmp_path / "late.md", on_replace=_write, on_cancel=lambda: None
    )
    assert inspect.isawaitable(outcome)
    await outcome
    assert calls == [False]
    assert len(app.pushed) == 1


@pytest.mark.asyncio
async def test_prompt_export_wrapper_asks_when_a_file_appears_mid_write(
    tmp_path, monkeypatch
):
    """The prompt writer must let the no-clobber FileExistsError reach the
    seam (so it asks) rather than report it as an export error."""
    from tldw_chatbook.UI.Library_Modules import library_file_export as seam

    destination = tmp_path / "exp" / "late.md"
    destination.parent.mkdir()
    real_write = seam.write_export_text
    attempts: list[bool] = []

    def _racing_write(path, content, *, overwrite=True):
        attempts.append(overwrite)
        if not overwrite:
            path.write_bytes(PRECIOUS)
            raise FileExistsError(path)
        return real_write(path, content, overwrite=overwrite)

    monkeypatch.setattr(seam, "write_export_text", _racing_write)
    app = _FakeApp()
    notices: list[tuple[str, str]] = []
    screen = SimpleNamespace(app=app)
    detail = {
        "name": "Late prompt",
        "author": "",
        "details": "",
        "system_prompt": "",
        "user_prompt": "body",
        "keywords": [],
    }

    seam.export_library_prompt_file(
        screen,
        destination,
        detail,
        7,
        lambda message, severity="information": notices.append((message, severity)),
    )

    assert attempts == [False]
    assert destination.read_bytes() == PRECIOUS
    assert not [m for m, _ in notices if "Error exporting" in m], notices
    assert len(app.pushed) == 1, "the race must fall into the replace prompt"
    await _answer(app, True)
    assert attempts == [False, True]
    assert "Late prompt" in destination.read_text(encoding="utf-8")
    assert any("exported successfully" in m for m, _ in notices), notices


def test_no_clobber_write_refuses_an_existing_file_and_leaves_it(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    destination = _precious(tmp_path)
    with pytest.raises(FileExistsError):
        write_export_text(destination, "must not land", overwrite=False)
    assert destination.read_bytes() == PRECIOUS
    assert not list(destination.parent.glob(".*")), "temp file left behind"
    write_export_text(tmp_path / "exp" / "fresh.md", "fresh\n", overwrite=False)
    assert (tmp_path / "exp" / "fresh.md").read_text(encoding="utf-8") == "fresh\n"


def test_a_dangling_symlink_is_written_at_the_chosen_path(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        resolve_export_destination,
    )

    link = tmp_path / "gone.md"
    link.symlink_to(tmp_path / "missing-target.md")
    assert resolve_export_destination(link) == link
    plain = tmp_path / "plain.md"
    assert resolve_export_destination(plain) == plain


def test_replace_prompt_abbreviates_the_home_folder(tmp_path, monkeypatch):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        describe_export_folder,
    )

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert describe_export_folder(tmp_path / "exp") == "~/exp"
    assert describe_export_folder(tmp_path) == "~"
    elsewhere = Path("/var/tmp/elsewhere")
    assert describe_export_folder(elsewhere) == str(elsewhere)


def test_receipts_name_the_folder_and_the_file(tmp_path, monkeypatch):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        describe_export_destination,
    )

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert describe_export_destination(tmp_path / "exp" / "a.md") == "~/exp/a.md"
    assert describe_export_destination(tmp_path / "a.md") == "~/a.md"
    assert describe_export_destination(Path("/var/tmp/x/a.md")) == "/var/tmp/x/a.md"


def test_the_next_picker_opens_in_the_last_export_folder_and_falls_back(
    tmp_path, monkeypatch
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        library_export_picker_location,
        remember_export_directory,
    )

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir()
    app = _FakeApp()
    assert library_export_picker_location(app) == str(tmp_path / "home")

    exported = _precious(tmp_path)
    remember_export_directory(app, exported)
    assert library_export_picker_location(app) == str(exported.parent)

    other = _FakeApp()
    assert library_export_picker_location(other) == str(tmp_path / "home"), (
        "the remembered folder is per app session"
    )

    exported.unlink()
    exported.parent.rmdir()
    assert library_export_picker_location(app) == str(tmp_path / "home"), (
        "a folder that no longer exists must not be offered"
    )


# --- AC#3: the Report artifact export --------------------------------------


def _artifacts_controller(app: _FakeApp):
    from tldw_chatbook.UI.Library_Modules.library_artifacts_controller import (
        LibraryArtifactsController,
    )

    controller = LibraryArtifactsController.__new__(LibraryArtifactsController)
    controller.disposed = False
    controller.screen = SimpleNamespace(
        app=app,
        app_instance=SimpleNamespace(
            subscriptions_db=None, chachanotes_db=None, local_chatbook_service=None
        ),
        notify=app.notify,
    )
    return controller


_REPORT = {
    "id": 7,
    "body_markdown": "# Weekly digest\n\nThree items.",
    "watchlist_name": "Research",
    "status": "complete",
    "created_at": "2026-10-01T09:00:00+00:00",
}


@pytest.mark.asyncio
async def test_report_export_asks_before_replacing_and_cancel_keeps_the_file(
    tmp_path,
):
    destination = _precious(tmp_path, "Research report.md")
    before = _md5(destination)
    app = _FakeApp()
    controller = _artifacts_controller(app)

    await controller._write_export(destination, _REPORT, controller.profile())

    assert _md5(destination) == before, (
        f"the report export replaced {destination} before asking"
    )
    assert len(app.pushed) == 1
    dialog, _ = app.pushed[0]
    assert isinstance(dialog, ConfirmationDialog)
    assert "Research report.md" in str(dialog.message)

    await _answer(app, False)
    assert destination.read_bytes() == PRECIOUS
    assert not [m for m, _ in app.notices if "exported to" in m], app.notices


@pytest.mark.asyncio
async def test_report_export_replace_writes_and_names_the_full_path(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        library_export_picker_location,
    )

    destination = _precious(tmp_path, "Research report.md")
    app = _FakeApp()
    controller = _artifacts_controller(app)

    await controller._write_export(destination, _REPORT, controller.profile())
    await _answer(app, True)

    written = destination.read_text(encoding="utf-8")
    assert "Three items." in written
    assert PRECIOUS.decode() not in written
    assert not list(destination.parent.glob(".*")), "temp file left behind"
    assert any(str(destination) in m for m, _ in app.notices), app.notices
    assert library_export_picker_location(app) == str(destination.parent)


# --- AC#3: the Collections legacy-recovery export ---------------------------


class _FakeRecovery:
    """Stands in for the recovery service: a plain write, like the real one's effect."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, object]] = []

    def export_json(self, destination, *, overwrite_identity):
        self.calls.append((Path(destination), overwrite_identity))
        Path(destination).write_text(
            json.dumps({"collections": [], "memberships": []}), encoding="utf-8"
        )
        return Path(destination)


def _collections_controller(app: _FakeApp, recovery: _FakeRecovery):
    from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )

    controller = LibraryCollectionsController.__new__(LibraryCollectionsController)
    state = SimpleNamespace(action_status="")
    controller._collections_state_accessor = lambda: state
    controller._screen = SimpleNamespace(
        app=app,
        app_instance=SimpleNamespace(
            collections_legacy_recovery_service=recovery, notify=app.notify
        ),
    )
    refreshes: list[int] = []
    controller._refresh_library_collections_capture_reader = lambda: refreshes.append(1)
    return controller, state, refreshes


@pytest.mark.asyncio
async def test_legacy_recovery_export_asks_before_replacing_and_cancel_keeps_the_file(
    tmp_path,
):
    destination = _precious(tmp_path, "legacy-collections-recovery.json")
    before = _md5(destination)
    app = _FakeApp()
    recovery = _FakeRecovery()
    controller, state, _ = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(destination)

    assert _md5(destination) == before, (
        f"the legacy export replaced {destination} before asking"
    )
    assert recovery.calls == []
    assert len(app.pushed) == 1
    dialog, _ = app.pushed[0]
    assert isinstance(dialog, ConfirmationDialog)
    assert "legacy-collections-recovery.json" in str(dialog.message)

    await _answer(app, False)
    assert destination.read_bytes() == PRECIOUS
    assert recovery.calls == []
    assert "complete" not in state.action_status.lower(), state.action_status


@pytest.mark.asyncio
async def test_legacy_recovery_export_replace_writes_through_the_service(tmp_path):
    from tldw_chatbook.UI.Library_Modules.library_file_export import (
        library_export_picker_location,
    )

    destination = _precious(tmp_path, "legacy-collections-recovery.json")
    identity = destination.lstat()
    app = _FakeApp()
    recovery = _FakeRecovery()
    controller, state, refreshes = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(destination)
    await _answer(app, True)

    assert recovery.calls == [(destination, (identity.st_dev, identity.st_ino))], (
        "the service must still receive the pre-confirmation identity race guard"
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "collections": [],
        "memberships": [],
    }
    assert str(destination) in state.action_status, state.action_status
    assert refreshes
    assert library_export_picker_location(app) == str(destination.parent)


@pytest.mark.asyncio
async def test_legacy_recovery_export_confirms_the_json_normalized_destination(
    tmp_path,
):
    """The picker may hand back ``recovery`` while ``recovery.json`` exists."""
    existing = _precious(tmp_path, "recovery.json")
    app = _FakeApp()
    recovery = _FakeRecovery()
    controller, _, _ = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(
        existing.with_suffix("")
    )

    assert recovery.calls == []
    assert len(app.pushed) == 1
    assert "recovery.json" in str(app.pushed[0][0].message)
    assert existing.read_bytes() == PRECIOUS
    assert not os.path.exists(existing.with_suffix(""))


@pytest.mark.asyncio
async def test_legacy_recovery_export_never_replaces_a_file_that_appeared_unasked(
    tmp_path, monkeypatch
):
    """Final review M3: nothing was at the destination when the seam checked,
    so no one was asked. A file that appears before the publish must be left
    alone. The publish used to read the target's identity at publish time
    whatever ``overwrite`` said, which handed the real recovery service
    permission to replace that file. Driven through the real service."""
    from Tests.Library.test_collections_legacy_recovery import _seed_legacy
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
    )
    from tldw_chatbook.UI.Library_Modules import library_file_export

    database = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(database, count=1)
    folder = tmp_path / "exp"
    folder.mkdir()
    destination = folder / "legacy-collections-recovery.json"
    real_exists = library_file_export.export_destination_exists

    def appears_after_the_check(path):
        seen = real_exists(path)
        Path(path).write_bytes(PRECIOUS)  # another program writes it just now
        return seen

    monkeypatch.setattr(
        library_file_export, "export_destination_exists", appears_after_the_check
    )
    app = _FakeApp()
    controller, state, _ = _collections_controller(
        app, LegacyCollectionsRecovery(database)
    )
    try:
        await controller._export_library_collection_legacy_recovery(destination)
    finally:
        database.close()

    assert destination.read_bytes() == PRECIOUS, (
        "the export replaced a file that appeared after the check, without asking"
    )
    assert "complete" not in state.action_status.lower(), state.action_status
    assert state.action_status.startswith(
        "Legacy export failed: legacy export target"
    ), state.action_status
    assert app.pushed == []
