"""TASK-34000.3 (N-11): the single-file export seam.

Pins the shared seam every one-file Library export goes through
(``UI/Library_Modules/library_file_export.py``): the replace prompt, the
atomic write, the remembered export folder, and -- AC#3 -- that the Report
artifact export and the Collections legacy-recovery export ask before
writing over an existing file. Those two are driven through the real
controller methods with a fake app that records what is pushed.
"""

from __future__ import annotations

import errno
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


# --- PR #3021 review: a destination folder that cannot hard-link -----------
# FAT32/exFAT sticks and many SMB/NFS mounts refuse ``link()`` with
# PermissionError or OSError(ENOTSUP) -- not FileExistsError. The no-clobber
# publish of a NEW file had no other way to land, so the note, prompt and
# report exports failed there, where the plain write before this seam
# worked. These fail on 502bd89efc.


def _eperm() -> OSError:
    return PermissionError(errno.EPERM, "Operation not permitted")


def _enotsup() -> OSError:
    return OSError(errno.ENOTSUP, "Operation not supported")


LINK_REFUSALS = [
    pytest.param(_eperm, id="PermissionError"),
    pytest.param(_enotsup, id="ENOTSUP"),
]


def _refuse_hard_links(monkeypatch, refusal, *, first=None) -> list[Path]:
    """Make ``os.link`` fail the way a volume without hard links does.

    ``first`` runs before the refusal, with the destination: another program
    getting there between the seam's look and the publish.
    """
    attempts: list[Path] = []

    def refuse(_source, destination, *_args, **_kwargs):
        attempts.append(Path(destination))
        if first is not None:
            first(Path(destination))
        raise refusal()

    monkeypatch.setattr(os, "link", refuse)
    return attempts


def _export_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "exp"
    folder.mkdir(exist_ok=True)
    return folder


def _names(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


_PROMPT_DETAIL = {
    "name": "Weekly review coach",
    "author": "",
    "details": "",
    "system_prompt": "",
    "user_prompt": "Review my week.",
    "keywords": [],
}


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    attempts = _refuse_hard_links(monkeypatch, refusal)
    destination = _export_folder(tmp_path) / "fresh.md"

    write_export_text(destination, "fresh\n", overwrite=False)

    assert attempts == [destination], "the link publish is still tried first"
    assert destination.read_bytes() == b"fresh\n"
    assert _names(destination.parent) == ["fresh.md"], (
        "temp file or placeholder left behind"
    )


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_no_clobber_write_still_refuses_an_existing_file_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    _refuse_hard_links(monkeypatch, refusal)
    destination = _precious(tmp_path)

    with pytest.raises(FileExistsError):
        write_export_text(destination, "must not land", overwrite=False)

    assert destination.read_bytes() == PRECIOUS
    assert _names(destination.parent) == ["precious.md"], "temp file left behind"


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_a_failed_no_clobber_write_leaves_no_placeholder_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    from tldw_chatbook.UI.Library_Modules.library_file_export import write_export_text

    _refuse_hard_links(monkeypatch, refusal)

    def _disk_went_away(*_args, **_kwargs):
        raise OSError(errno.EIO, "disk went away")

    monkeypatch.setattr(os, "replace", _disk_went_away)
    destination = _export_folder(tmp_path) / "fresh.md"

    with pytest.raises(OSError) as caught:
        write_export_text(destination, "never lands", overwrite=False)

    assert caught.value.errno == errno.EIO
    assert not os.path.lexists(destination), "an empty placeholder was left behind"
    assert _names(destination.parent) == []


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_prompt_export_to_a_new_file_succeeds_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    from tldw_chatbook.UI.Library_Modules import library_file_export as seam

    _refuse_hard_links(monkeypatch, refusal)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    destination = _export_folder(tmp_path) / "Weekly review coach.md"
    app = _FakeApp()
    notices: list[tuple[str, str]] = []

    seam.export_library_prompt_file(
        SimpleNamespace(app=app),
        destination,
        _PROMPT_DETAIL,
        7,
        lambda message, severity="information": notices.append((message, severity)),
    )

    assert notices == [
        (
            "Prompt exported successfully to ~/exp/Weekly review coach.md",
            "information",
        )
    ], notices
    assert "Review my week." in destination.read_text(encoding="utf-8")
    assert app.pushed == [], "nothing was there: no replace prompt"
    assert _names(destination.parent) == ["Weekly review coach.md"]


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_note_export_to_a_new_file_succeeds_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    from tldw_chatbook.UI.Library_Modules import library_file_export as seam

    _refuse_hard_links(monkeypatch, refusal)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    destination = _export_folder(tmp_path) / "Reading list.md"
    app = _FakeApp()
    finished: list[tuple[object, dict]] = []
    screen = SimpleNamespace(
        app=app,
        app_instance=SimpleNamespace(notify=app.notify),
        _finish_library_notes_operation=lambda operation, **kw: finished.append(
            (operation, kw)
        ),
    )

    seam.export_library_note_file(
        screen,
        destination,
        "markdown",
        "Reading list",
        "Three books.",
        "books",
        "note-1",
        "operation-token",
    )

    assert app.notices == [
        ("Note exported successfully to ~/exp/Reading list.md", "information")
    ], app.notices
    assert finished == [("operation-token", {"success": True})], finished
    assert "Three books." in destination.read_text(encoding="utf-8")
    assert app.pushed == []
    assert _names(destination.parent) == ["Reading list.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", LINK_REFUSALS)
async def test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    _refuse_hard_links(monkeypatch, refusal)
    destination = _export_folder(tmp_path) / "Research report.md"
    app = _FakeApp()
    controller = _artifacts_controller(app)

    await controller._write_export(destination, _REPORT, controller.profile())

    assert "Three items." in destination.read_text(encoding="utf-8")
    assert [severity for _, severity in app.notices] == ["information"], app.notices
    assert str(destination) in app.notices[0][0], app.notices
    assert app.pushed == []
    assert _names(destination.parent) == ["Research report.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", LINK_REFUSALS)
async def test_a_file_appearing_mid_export_still_asks_where_hard_links_are_unsupported(
    tmp_path, monkeypatch, refusal
):
    """The fallback keeps the no-clobber promise: a file another program
    wrote after the seam looked is never replaced unasked, and the
    ``FileExistsError`` still reaches the Replace prompt."""
    from tldw_chatbook.UI.Library_Modules import library_file_export as seam

    attempts = _refuse_hard_links(
        monkeypatch, refusal, first=lambda path: path.write_bytes(PRECIOUS)
    )
    destination = _export_folder(tmp_path) / "late.md"
    app = _FakeApp()
    notices: list[tuple[str, str]] = []

    seam.export_library_prompt_file(
        SimpleNamespace(app=app),
        destination,
        _PROMPT_DETAIL,
        7,
        lambda message, severity="information": notices.append((message, severity)),
    )

    assert attempts == [destination]
    assert destination.read_bytes() == PRECIOUS, "replaced a file without asking"
    assert notices == [], notices
    assert len(app.pushed) == 1, "the collision must fall into the replace prompt"
    assert _names(destination.parent) == ["late.md"], "temp file left behind"

    await _answer(app, True)
    assert "Review my week." in destination.read_text(encoding="utf-8")
    assert any("exported successfully" in m for m, _ in notices), notices
    assert _names(destination.parent) == ["late.md"]


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


def _seeded_recovery_database(tmp_path: Path):
    from Tests.Library.test_collections_legacy_recovery import _seed_legacy
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB

    database = LibraryCollectionsDB(tmp_path / "collections.db")
    _seed_legacy(database, count=1)
    return database


@pytest.mark.asyncio
async def test_legacy_recovery_export_never_replaces_a_file_that_appeared_unasked(
    tmp_path, monkeypatch
):
    """Final review M3: nothing was at the destination when the seam checked,
    so no one was asked. A file that appears before the publish must be left
    alone. The publish used to read the target's identity at publish time
    whatever ``overwrite`` said, which handed the real recovery service
    permission to replace that file. Driven through the real service.

    PR #3021 review: the service's refusal (``legacy_export_target_exists``)
    was then reported as "Legacy export failed" and the user had to start
    over. It is the same late collision the note, prompt and report exports
    turn into the Replace prompt, so it opens that prompt too -- and Replace
    then publishes through the service with the identity the user was shown.
    """
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
    )
    from tldw_chatbook.UI.Library_Modules import library_file_export

    database = _seeded_recovery_database(tmp_path)
    destination = _export_folder(tmp_path) / "legacy-collections-recovery.json"
    real_exists = library_file_export.export_destination_exists
    appeared: list[Path] = []

    def appears_after_the_check(path):
        seen = real_exists(path)
        if not appeared:  # another program writes it just now, once
            appeared.append(Path(path))
            Path(path).write_bytes(PRECIOUS)
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

        assert destination.read_bytes() == PRECIOUS, (
            "the export replaced a file that appeared after the check, without asking"
        )
        assert "complete" not in state.action_status.lower(), state.action_status
        assert "failed" not in state.action_status.lower(), (
            f"a late collision was reported as a failure: {state.action_status!r}"
        )
        assert app.notices == [], app.notices
        assert len(app.pushed) == 1, "the late collision must open the Replace prompt"
        dialog, _ = app.pushed[0]
        assert isinstance(dialog, ConfirmationDialog)
        assert "legacy-collections-recovery.json" in str(dialog.message)

        await _answer(app, True)
    finally:
        database.close()

    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert [row["collection_id"] for row in exported["collections"]] == ["legacy-000"]
    assert state.action_status.startswith("Legacy recovery export complete: ")
    assert str(destination) in state.action_status, state.action_status
    assert _names(destination.parent) == ["legacy-collections-recovery.json"]


@pytest.mark.asyncio
async def test_legacy_recovery_export_asks_when_a_file_appears_during_the_export(
    tmp_path,
):
    """The later window: the service has accepted the destination and written
    its temporary file, and the name is taken before it publishes. The real
    service refuses with ``legacy_export_target_changed``; with nothing
    confirmed and a file there now, that is the same question. Cancel keeps
    the file that appeared."""
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
    )

    database = _seeded_recovery_database(tmp_path)
    destination = _export_folder(tmp_path) / "legacy-collections-recovery.json"
    recovery = LegacyCollectionsRecovery(database)
    recovery._before_publish = lambda: destination.write_bytes(PRECIOUS)
    app = _FakeApp()
    controller, state, _ = _collections_controller(app, recovery)
    try:
        await controller._export_library_collection_legacy_recovery(destination)
    finally:
        database.close()

    assert destination.read_bytes() == PRECIOUS
    assert app.notices == [], app.notices
    assert len(app.pushed) == 1, "the late collision must open the Replace prompt"
    assert _names(destination.parent) == ["legacy-collections-recovery.json"], (
        "the service's temporary file was left behind"
    )

    await _answer(app, False)
    assert destination.read_bytes() == PRECIOUS
    assert state.action_status == (
        "Legacy export cancelled. legacy-collections-recovery.json was left unchanged."
    ), state.action_status


class _RefusingRecovery:
    """A recovery service that refuses every publish with one reason."""

    def __init__(self, error: Exception, *, before=None) -> None:
        self.error = error
        self.before = before
        self.calls: list[tuple[Path, object]] = []

    def export_json(self, destination, *, overwrite_identity):
        self.calls.append((Path(destination), overwrite_identity))
        if self.before is not None:
            self.before(Path(destination))
        raise self.error


def _recovery_error(reason: str) -> Exception:
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecoveryError,
    )

    return LegacyCollectionsRecoveryError(reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        pytest.param(
            lambda: _recovery_error("legacy_export_failed"),
            "Legacy export failed: legacy export failed.",
            id="write-failed",
        ),
        pytest.param(
            lambda: _recovery_error("invalid_legacy_export_destination"),
            "Legacy export failed: invalid legacy export destination.",
            id="invalid-destination",
        ),
        pytest.param(
            lambda: _recovery_error("legacy_export_parent_changed"),
            "Legacy export failed: legacy export parent changed.",
            id="parent-changed",
        ),
        pytest.param(
            # The name was taken and released again: nothing is there to ask about.
            lambda: _recovery_error("legacy_export_target_changed"),
            "Legacy export failed: legacy export target changed.",
            id="target-changed-but-nothing-there",
        ),
        pytest.param(
            lambda: RuntimeError("boom"),
            "Legacy export failed: legacy export failed.",
            id="unexpected-error",
        ),
    ],
)
async def test_legacy_recovery_export_reports_every_other_failure_as_failed(
    tmp_path, error, status
):
    destination = _export_folder(tmp_path) / "legacy-collections-recovery.json"
    app = _FakeApp()
    recovery = _RefusingRecovery(error())
    controller, state, refreshes = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(destination)

    assert recovery.calls == [(destination, None)]
    assert state.action_status == status, state.action_status
    assert app.pushed == [], "only a destination that is taken asks"
    assert [severity for _, severity in app.notices] == ["warning"], app.notices
    assert refreshes
    assert not os.path.lexists(destination)


@pytest.mark.asyncio
async def test_legacy_recovery_export_does_not_ask_about_a_file_it_could_not_replace(
    tmp_path,
):
    """A destination the service will not write at all (here: not a regular
    single-link file) stays a failure -- asking "Replace?" and then failing
    would be a question with no good answer."""
    destination = _export_folder(tmp_path) / "legacy-collections-recovery.json"
    app = _FakeApp()
    recovery = _RefusingRecovery(
        _recovery_error("invalid_legacy_export_destination"),
        before=lambda path: path.mkdir(),
    )
    controller, state, _ = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(destination)

    assert state.action_status == (
        "Legacy export failed: invalid legacy export destination."
    ), state.action_status
    assert app.pushed == []
    assert destination.is_dir()


@pytest.mark.asyncio
async def test_legacy_recovery_export_refused_after_replace_is_a_failure_not_a_second_prompt(
    tmp_path,
):
    """The user answered Replace for the file they were shown. If that file
    changes before the publish, the service refuses -- and the answer is not
    silently carried over to a different file, nor re-asked from inside the
    prompt's own callback (where a ``FileExistsError`` has no handler)."""
    destination = _precious(tmp_path, "legacy-collections-recovery.json")
    app = _FakeApp()
    recovery = _RefusingRecovery(_recovery_error("legacy_export_target_changed"))
    controller, state, _ = _collections_controller(app, recovery)

    await controller._export_library_collection_legacy_recovery(destination)
    assert recovery.calls == []
    await _answer(app, True)

    identity = destination.lstat()
    assert recovery.calls == [(destination, (identity.st_dev, identity.st_ino))]
    assert state.action_status == (
        "Legacy export failed: legacy export target changed."
    ), state.action_status
    assert app.pushed == [], "no second prompt"
    assert destination.read_bytes() == PRECIOUS
