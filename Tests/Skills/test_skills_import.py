"""Real-service tests for the Library skills Import row (Task 5 of the
Skills sub-project).

Mirrors ``Tests/Skills/test_skills_library_flow.py``'s real-service
posture: a real ``LocalSkillsService``/``SkillsScopeService`` wired onto a
real ``LibraryScreen`` via ``App.run_test()`` -- no hand-rolled fakes for
the service layer.

Per the sub-project's compat directive, this suite imports REAL SKILL.md
files copied from the ``obra/superpowers`` skillset (see
``Tests/fixtures/superpowers_skills/README.md`` for provenance) through
the actual Library Import row -- not synthetic content -- so any
incompatibility between what that real-world skillset actually writes and
what ``local_skills_service``/``skills_schemas`` accepts surfaces here
honestly rather than being assumed away. A handful of additional,
CLEARLY synthetic edge cases (name too long, oversized content, a nested
reference subfolder) are constructed inline with ``tmp_path`` rather than
committed as fixture files, to keep the fixtures directory small.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
import zipfile
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger
from textual.screen import Screen
from textual.widgets import Button, Input

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.tldw_api.skills_schemas import SKILL_NAME_PATTERN
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from tldw_chatbook.UI.Library_Modules import library_skill_import_controller
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Library_Modules.skill_import_choice_modal import (
    SkillImportChoiceModal,
)
from tldw_chatbook.Skills_Interop.skill_package_inspection import (
    FRAMEWORK_MESSAGE,
    SkillPackageInspection,
    SkillPackageKind,
)
from tldw_chatbook.Skills_Interop.skill_remote_fetch import RemoteSkillPackage

from Tests.Skills.test_skills_library_flow import (
    _real_trust_service,
    _wire_empty_non_skill_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skills_canvas import _CanvasHost, _two_row_state


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "superpowers_skills"


def _write_single_skill_zip(path: Path, *, name: str) -> None:
    """Write one minimal importable skill archive for route-parity tests."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "SKILL.md",
            f"---\nname: {name}\ndescription: Route parity fixture.\n---\n\nBody.\n",
        )


def _real_skills_scope_service_with_trust(tmp_path):
    """Build a real ``LocalSkillsService``/``SkillsScopeService`` pair backed
    by a real (unlocked, NOT bootstrapped) trust service.

    Unlike ``test_skills_library_flow._real_skills_scope_service`` (which
    defaults to ``allow_untrusted_without_trust_service=True`` -- a compat
    mode that reports EVERY skill as already trusted, making trust
    irrelevant), this suite is specifically about asserting imported
    skills land TRUST-PENDING, so it always wires a real trust service.
    Never bootstrapped: a freshly imported skill in a trust store that has
    never been bootstrapped at all reports ``trust_uninitialized``/
    ``trust_blocked=True`` -- itself a genuine "needs review before use"
    state, not a synthetic stand-in for one.
    """
    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    return local_service, service


# The real superpowers skills copied into the fixtures dir (see its
# README.md for provenance) -- five distinct skills, one of which
# (requesting-code-review) has a real supporting reference file.
REAL_FIXTURE_SKILLS = (
    "executing-plans",
    "requesting-code-review",
    "using-superpowers",
    "verification-before-completion",
    "writing-plans",
)


async def _open_skills_import_row(screen, pilot) -> None:
    """Open the Skills rail row, then the inline Import row below its toolbar."""
    skills_row = await _wait_for_selector(
        screen, pilot, "#library-row-browse-skills"
    )
    assert isinstance(skills_row, Button)
    skills_row.press()
    # The default empty Library can already be showing Skills. Let that
    # same-row rail press settle before resolving the canvas action, or the
    # query can return the about-to-be-recomposed button and its event is lost.
    await pilot.pause()
    import_row = await _wait_for_selector(screen, pilot, "#library-skills-import")
    assert isinstance(import_row, Button)
    assert import_row.disabled is False
    import_row.press()
    await _wait_for_selector(screen, pilot, "#library-skills-import-path")


async def _run_skills_import_via_ui(
    screen, pilot, path: Path, *, deadline_seconds: float = 30.0
) -> str:
    """Type ``path`` into the Import row and press Import, returning the
    outcome line once it changes from whatever it showed before this call.

    Waiting for a CHANGE (not just "non-empty") matters here: the Import
    row's outcome ``Static`` is never cleared between successive imports
    (only "Cancel" clears it), so blindly waiting for non-empty text would
    read a STALE outcome left over from a previous import in the same
    Import-row session.

    The wait is a WALL-CLOCK deadline, not a fixed iteration count: each
    import spins up a fresh OS thread + ``asyncio.run()`` loop + real file
    I/O (~3.5s even unloaded), so the previous 150x0.02s = 3.0s iteration
    ceiling sat below the zero-load baseline and flaked whenever the suite
    ran under contention.

    task-291 closed the residual stale-text false-pass: the success copy now
    carries the imported skill's NAME, so callers assert the name-specific
    line and a stale success from the previous import can never satisfy it.
    """
    status_widgets = list(screen.query("#library-skills-import-status"))
    assert status_widgets, "Skills import status did not mount"
    previous = str(status_widgets[0].renderable)
    screen.query_one("#library-skills-import-path", Input).value = str(path)
    await pilot.pause()
    screen.query_one("#library-skills-import-run", Button).press()
    await pilot.pause()
    status_text = previous
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        status_widgets = list(screen.query("#library-skills-import-status"))
        if status_widgets:
            status_text = str(status_widgets[0].renderable)
            if status_text != previous:
                return status_text
        await pilot.pause(0.02)
    return status_text


async def _wait_for_active_library_screen(app, pilot) -> LibraryScreen:
    """Return the freshly mounted production Library route."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if isinstance(app.screen, LibraryScreen):
            await _wait_for_library_shell(app.screen, pilot)
            return app.screen
        await pilot.pause(0.02)
    raise AssertionError("Library screen did not mount")


async def _wait_for_skill_import_terminal(
    screen: LibraryScreen,
    pilot,
    *,
    expected_status: str,
) -> None:
    """Wait for the shared import receipt to reach one literal outcome."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if (
            screen._library_skills_import_in_flight is False
            and screen._library_skills_import_status == expected_status
        ):
            return
        await pilot.pause(0.02)
    raise AssertionError(
        "Skill import did not settle: "
        f"in_flight={screen._library_skills_import_in_flight!r}, "
        f"status={screen._library_skills_import_status!r}"
    )


def test_library_screen_has_no_parallel_skill_import_pipeline():
    """Every supported route mutates through the app-owned coordinator."""
    retired_screen_owners = (
        "_run_library_skills_import_single_flight",
        "_run_library_skills_import",
        "_import_library_skill_from_loose_file",
        "_install_library_skill_from_url",
        "_apply_library_skills_import_success",
        "_apply_library_skills_import_outcome_from_exception",
    )

    assert all(
        not hasattr(LibraryScreen, name) for name in retired_screen_owners
    )


def _multi_package() -> RemoteSkillPackage:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("repo/skills/a/SKILL.md", "a")
        archive.writestr("repo/skills/b/SKILL.md", "b")
    payload = buffer.getvalue()
    import hashlib

    return RemoteSkillPackage(
        inspection=SkillPackageInspection(
            SkillPackageKind.MULTI_SKILL_REPOSITORY,
            ("skills/a", "skills/b"),
            "Choose one installable skill.",
        ),
        archive_bytes=payload,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        suggested_name="repo",
    )


def _root_package(name: str = "remote-skill") -> RemoteSkillPackage:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "repo/SKILL.md",
            f"---\nname: {name}\ndescription: Remote fixture.\n---\n\nBody.\n",
        )
    payload = buffer.getvalue()
    import hashlib

    return RemoteSkillPackage(
        inspection=SkillPackageInspection(SkillPackageKind.ROOT_SKILL, ("",)),
        archive_bytes=payload,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        suggested_name=name,
    )


@pytest.mark.asyncio
async def test_coordinator_holds_single_flight_through_candidate_choice(monkeypatch):
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    package = _multi_package()

    async def inspect(*args, **kwargs):
        return package

    monkeypatch.setattr(library_skill_import_controller, "inspect_skill_from_url", inspect)
    coordinator.open_draft()
    url = "https://github.com/o/repo"
    assert coordinator.claim(url) is True
    await coordinator.run(url, runtime_app=SimpleNamespace(screen=SimpleNamespace()))

    assert coordinator.snapshot.in_flight is True
    assert coordinator.snapshot.candidates == ("skills/a", "skills/b")
    assert coordinator.snapshot.status == "Choose one skill to import."
    assert coordinator.claim("/competing.zip") is False
    assert coordinator.cancel_choice() is True
    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.path == "https://github.com"
    assert coordinator.snapshot.status == ""


@pytest.mark.asyncio
async def test_coordinator_imports_explicit_retained_candidate_once(monkeypatch):
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    package = _multi_package()
    imports: list[tuple[RemoteSkillPackage, str]] = []

    async def inspect(*args, **kwargs):
        return package

    async def import_one(pending, *, candidate, scope_service):
        imports.append((pending, candidate))
        return {"name": "b"}

    monkeypatch.setattr(library_skill_import_controller, "inspect_skill_from_url", inspect)
    monkeypatch.setattr(library_skill_import_controller, "import_inspected_skill", import_one)
    coordinator.open_draft()
    assert coordinator.claim("https://github.com/o/repo") is True
    await coordinator.run(
        "https://github.com/o/repo", runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    assert coordinator.claim_candidate("skills/b") is True
    assert coordinator.claim_candidate("skills/a") is False
    await coordinator.run_candidate(
        "skills/b", runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )

    assert imports == [(package, "skills/b")]
    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.candidates == ()
    assert coordinator.snapshot.review_name == "b"


@pytest.mark.asyncio
async def test_selected_candidate_owner_survives_repeated_cancellation(monkeypatch):
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    package = _multi_package()
    started = threading.Event()
    release = threading.Event()
    imports = 0

    async def inspect(*args, **kwargs):
        return package

    async def import_one(*args, **kwargs):
        nonlocal imports
        started.set()
        await asyncio.to_thread(release.wait)
        imports += 1
        return {"name": "a"}

    monkeypatch.setattr(library_skill_import_controller, "inspect_skill_from_url", inspect)
    monkeypatch.setattr(library_skill_import_controller, "import_inspected_skill", import_one)
    coordinator.open_draft()
    assert coordinator.claim("https://github.com/o/repo") is True
    await coordinator.run(
        "https://github.com/o/repo", runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    assert coordinator.claim_candidate("skills/a") is True
    owner = asyncio.create_task(
        coordinator.run_candidate(
            "skills/a", runtime_app=SimpleNamespace(screen=SimpleNamespace())
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    owner.cancel()
    await asyncio.sleep(0)
    owner.cancel()
    await asyncio.sleep(0)
    assert owner.done() is False
    assert coordinator.snapshot.in_flight is True

    release.set()
    await owner
    assert imports == 1
    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.review_name == "a"


@pytest.mark.asyncio
async def test_coordinator_surfaces_framework_and_redacted_fetch_failure(monkeypatch):
    outcomes = (
        RemoteSkillPackage(
            SkillPackageInspection(
                SkillPackageKind.FRAMEWORK_REPOSITORY,
                message=FRAMEWORK_MESSAGE,
                recovery_actions=("Use project instructions.",),
            )
        ),
        RemoteSkillPackage(
            SkillPackageInspection(
                SkillPackageKind.FETCH_OR_AUTH_FAILURE,
                message="Could not fetch that skill package. Retry when access is available.",
            )
        ),
    )
    for package in outcomes:
        coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
            SimpleNamespace(skills_scope_service=SimpleNamespace())
        )

        async def inspect(*args, **kwargs):
            return package

        monkeypatch.setattr(
            library_skill_import_controller, "inspect_skill_from_url", inspect
        )
        coordinator.open_draft()
        assert coordinator.claim("https://example.com/pkg.zip?token=SECRET") is True
        await coordinator.run(
            "https://example.com/pkg.zip?token=SECRET",
            runtime_app=SimpleNamespace(screen=SimpleNamespace()),
        )
        assert "SECRET" not in repr(coordinator.snapshot)
        assert coordinator.snapshot.package_kind == package.inspection.kind.value
        assert coordinator.snapshot.retryable is (
            package.inspection.kind is SkillPackageKind.FETCH_OR_AUTH_FAILURE
        )


def test_coordinator_keeps_retry_url_private_and_display_url_credential_free():
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    coordinator.open_draft()

    assert coordinator.claim(
        "https://alice:URL-SECRET@example.com/pkg.zip?token=QUERY-SECRET"
    )

    assert coordinator.snapshot.path == "https://example.com"
    rendered = repr(coordinator.snapshot)
    assert "URL-SECRET" not in rendered
    assert "QUERY-SECRET" not in rendered

    coordinator.update(in_flight=False, retryable=True)
    assert coordinator.claim_retry() == (
        "https://alice:URL-SECRET@example.com/pkg.zip?token=QUERY-SECRET"
    )
    assert coordinator.claim_retry() is None
    assert coordinator.snapshot.path == "https://example.com"


@pytest.mark.asyncio
async def test_candidate_cancel_reuses_unchanged_private_signed_url(monkeypatch):
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    inspected: list[str] = []

    async def inspect_url(url, **kwargs):
        inspected.append(url)
        return _multi_package()

    monkeypatch.setattr(
        library_skill_import_controller, "inspect_skill_from_url", inspect_url
    )
    raw = "https://example.com/PATH-SECRET.zip?token=QUERY-SECRET"
    coordinator.open_draft()
    assert coordinator.claim(raw)
    await coordinator.run(
        raw, runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    safe_receipt = repr(coordinator.snapshot)
    assert "PATH-SECRET" not in safe_receipt
    assert "QUERY-SECRET" not in safe_receipt
    assert coordinator.cancel_choice()

    unchanged_display = coordinator.snapshot.path
    assert unchanged_display == "https://example.com"
    assert coordinator.claim(unchanged_display)
    await coordinator.run(
        unchanged_display, runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )

    assert inspected == [raw, raw]


@pytest.mark.asyncio
async def test_edit_after_candidate_cancel_replaces_private_signed_url(monkeypatch):
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    inspected: list[str] = []

    async def inspect_url(url, **kwargs):
        inspected.append(url)
        return _multi_package()

    monkeypatch.setattr(
        library_skill_import_controller, "inspect_skill_from_url", inspect_url
    )
    original = "https://example.com/old.zip?token=OLD-SECRET"
    edited = "https://example.net/new.zip?token=NEW-SECRET"
    coordinator.open_draft()
    assert coordinator.claim(original)
    await coordinator.run(
        original, runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    assert coordinator.cancel_choice()

    assert coordinator.update_draft_path(edited)
    assert coordinator.claim(edited)
    await coordinator.run(
        edited, runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )

    assert inspected == [original, edited]


@pytest.mark.asyncio
async def test_local_multi_skill_folder_requires_choice_and_imports_only_selected(tmp_path):
    source = tmp_path / "repository"
    for relative in ("skills/a", "skills/b"):
        skill_dir = source / relative
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("body", encoding="utf-8")
    calls: list[tuple[Path, dict]] = []

    async def import_directory(path, **kwargs):
        calls.append((path, kwargs))
        return {"name": kwargs["name"]}

    service = SimpleNamespace(
        import_skill_directory=import_directory,
        import_skill_file=lambda *args, **kwargs: None,
    )
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=service)
    )
    coordinator.open_draft()
    assert coordinator.claim(str(source)) is True
    await coordinator.run(
        str(source), runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    assert coordinator.snapshot.candidates == ("skills/a", "skills/b")

    assert coordinator.claim_candidate("skills/a") is True
    await coordinator.run_candidate(
        "skills/a", runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )

    assert calls == [
        (
            source / "skills/a",
            {"mode": "local", "name": "a", "trust_approved": False},
        )
    ]
    assert coordinator.snapshot.review_name == "a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "swap_kind", ("candidate_symlink", "body_symlink", "body_replacement")
)
async def test_local_candidate_rejects_post_inspection_swap(tmp_path, swap_kind):
    source = tmp_path / "repository"
    candidate = source / "skills" / "a"
    other = source / "skills" / "b"
    for skill_dir in (candidate, other):
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_body = outside / "SKILL.md"
    outside_body.write_text("outside", encoding="utf-8")
    calls: list[Path] = []

    async def import_directory(path, **kwargs):
        calls.append(Path(path))
        return {"name": "a"}

    service = SimpleNamespace(
        import_skill_directory=import_directory,
        import_skill_file=lambda *args, **kwargs: None,
    )
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=service)
    )
    coordinator.open_draft()
    assert coordinator.claim(str(source))
    await coordinator.run(
        str(source), runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )
    assert coordinator.snapshot.candidates == ("skills/a", "skills/b")

    body = candidate / "SKILL.md"
    body.unlink()
    try:
        if swap_kind == "candidate_symlink":
            candidate.rmdir()
            os.symlink(outside, candidate, target_is_directory=True)
        elif swap_kind == "body_symlink":
            os.symlink(outside_body, body)
        else:
            body.write_text("replacement", encoding="utf-8")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    assert coordinator.claim_candidate("skills/a")
    await coordinator.run_candidate(
        "skills/a", runtime_app=SimpleNamespace(screen=SimpleNamespace())
    )

    assert calls == []
    assert coordinator.snapshot.review_name == ""
    assert coordinator.snapshot.in_flight is False


@pytest.mark.asyncio
async def test_mounted_library_multi_skill_choice_imports_one_to_trust_review(tmp_path):
    source = tmp_path / "repository"
    for name in ("alpha", "zeta"):
        skill_dir = source / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Candidate fixture.\n---\n\nBody.\n",
            encoding="utf-8",
        )
    local_service, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)
        screen.query_one("#library-skills-import-path", Input).value = str(source)
        await pilot.pause()
        screen.query_one("#library-skills-import-run", Button).press()
        for _ in range(200):
            if isinstance(host.screen, SkillImportChoiceModal):
                break
            await pilot.pause(0.01)
        assert isinstance(host.screen, SkillImportChoiceModal)
        assert app.library_skill_import_coordinator.snapshot.in_flight is True
        choices = host.screen.query_one("#skill-import-choice-list")
        choices.highlighted = 1
        await pilot.pause()
        host.screen.query_one("#skill-import-choice-import", Button).press()
        for _ in range(300):
            if (
                host.screen is screen
                and not screen._library_skills_import_in_flight
                and screen._library_skills_import_review_name
            ):
                break
            await pilot.pause(0.01)

        assert host.screen is screen
        assert screen._library_skills_import_review_name == "zeta"
        assert (
            screen._library_skills_import_status
            == 'Imported "zeta" · re-review it in the trust panel'
        )
        await host.workers.wait_for_complete()

    context = await service.get_context(mode="local")
    assert [skill["name"] for skill in context["blocked_skills"]] == ["zeta"]
    with pytest.raises(ValueError, match="local_skill_not_found:alpha"):
        await local_service.get_skill("alpha")


@pytest.mark.asyncio
async def test_candidate_modal_replays_when_inspection_finishes_while_away(
    tmp_path, monkeypatch
):
    _local, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)
    started = threading.Event()
    release = threading.Event()

    async def inspect_url(*args, **kwargs):
        started.set()
        await asyncio.to_thread(release.wait)
        return _multi_package()

    monkeypatch.setattr(
        library_skill_import_controller, "inspect_skill_from_url", inspect_url
    )
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_skills_import_row(screen, pilot)
            screen.query_one("#library-skills-import-path", Input).value = (
                "https://github.com/o/repo"
            )
            await pilot.pause()
            screen.query_one("#library-skills-import-run", Button).press()
            assert await asyncio.to_thread(started.wait, 5)

            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-canvas")
            release.set()
            for _ in range(250):
                if app.library_skill_import_coordinator.snapshot.candidates:
                    break
                await pilot.pause(0.02)
            assert app.library_skill_import_coordinator.snapshot.candidates
            assert host.screen is screen

            screen.query_one("#library-row-browse-skills", Button).press()
            await _wait_for_selector(screen, pilot, "#library-skills-import-path")
            for _ in range(10):
                if isinstance(host.screen, SkillImportChoiceModal):
                    break
                await pilot.pause()
            assert isinstance(host.screen, SkillImportChoiceModal)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_replacement_library_hydrates_pending_candidate_modal(tmp_path):
    _local, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        original = _active_library_screen(host)
        await _wait_for_library_shell(original, pilot)
        coordinator = library_skill_import_controller.ensure_library_skill_import_coordinator(
            app
        )
        package = _multi_package()
        coordinator._pending_package = package
        coordinator.update(
            row_open=True,
            path="https://example.com",
            status="Choose one skill to import.",
            in_flight=True,
            candidates=package.inspection.candidates,
            package_kind=SkillPackageKind.MULTI_SKILL_REPOSITORY.value,
            generation=7,
        )

        await host.switch_screen(Screen())
        replacement = LibraryScreen(app)
        replacement._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
        await host.switch_screen(replacement)
        await _wait_for_library_shell(replacement, pilot)
        for _ in range(10):
            if isinstance(host.screen, SkillImportChoiceModal):
                break
            await pilot.pause()

        assert isinstance(host.screen, SkillImportChoiceModal)
        assert coordinator.snapshot.in_flight is True


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 36), (72, 22)))
async def test_framework_recovery_and_failed_retry_render_at_supported_sizes(size):
    recovery = (
        "Choose a repository subdirectory that contains SKILL.md.",
        "Use its project instructions when that is the intended integration.",
        "Use the framework's external CLI outside Chatbook.",
        "Create a separately reviewed wrapper skill.",
    )
    framework = _CanvasHost(
        _two_row_state(),
        import_open=True,
        import_status=FRAMEWORK_MESSAGE,
        import_package_kind=SkillPackageKind.FRAMEWORK_REPOSITORY.value,
        import_recovery_actions=recovery,
    )
    async with framework.run_test(size=size):
        status = framework.query_one("#library-skills-import-status")
        assert str(status.renderable) == FRAMEWORK_MESSAGE
        guidance = framework.query_one("#library-skills-import-recovery")
        guidance_text = str(guidance.renderable)
        assert all(action in guidance_text for action in recovery)
        assert not list(framework.query("#library-skills-import-retry"))
        assert str(framework.query_one("#library-skills-import", Button).label) == (
            "Import skill…"
        )

    failed = _CanvasHost(
        _two_row_state(),
        import_open=True,
        import_status="Could not fetch that skill package.",
        import_package_kind=SkillPackageKind.FETCH_OR_AUTH_FAILURE.value,
        import_retryable=True,
    )
    async with failed.run_test(size=size):
        assert failed.query_one("#library-skills-import-retry", Button)


@pytest.mark.asyncio
async def test_import_exception_text_and_traceback_are_not_logged(monkeypatch):
    source = _multi_package()
    package = RemoteSkillPackage(
        inspection=SkillPackageInspection(SkillPackageKind.ROOT_SKILL, ("",)),
        archive_bytes=source.archive_bytes,
        archive_sha256=source.archive_sha256,
        suggested_name="safe-name",
    )

    async def inspect_url(*args, **kwargs):
        return package

    async def fail_import(*args, **kwargs):
        raise RuntimeError("TOKEN-SECRET raw failure body")

    monkeypatch.setattr(
        library_skill_import_controller, "inspect_skill_from_url", inspect_url
    )
    monkeypatch.setattr(
        library_skill_import_controller, "import_inspected_skill", fail_import
    )
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message} | {exception}")
    try:
        coordinator.open_draft()
        raw = "https://example.com/safe-name.zip"
        assert coordinator.claim(raw)
        await coordinator.run(
            raw, runtime_app=SimpleNamespace(screen=SimpleNamespace())
        )
    finally:
        logger.remove(sink)

    rendered = "".join(messages)
    assert "TOKEN-SECRET" not in rendered
    assert "Traceback" not in rendered


@pytest.mark.asyncio
async def test_duplicate_collision_copy_never_requires_stringifying_other_errors(
    monkeypatch,
):
    source = _multi_package()
    package = RemoteSkillPackage(
        inspection=SkillPackageInspection(SkillPackageKind.ROOT_SKILL, ("",)),
        archive_bytes=source.archive_bytes,
        archive_sha256=source.archive_sha256,
        suggested_name="safe-name",
    )

    async def inspect_url(*args, **kwargs):
        return package

    async def duplicate(*args, **kwargs):
        raise ValueError("local_skill_exists:safe-name")

    monkeypatch.setattr(
        library_skill_import_controller, "inspect_skill_from_url", inspect_url
    )
    monkeypatch.setattr(
        library_skill_import_controller, "import_inspected_skill", duplicate
    )
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    coordinator.open_draft()
    assert coordinator.claim("https://example.com/safe-name.zip")
    await coordinator.run(
        "https://example.com/safe-name.zip",
        runtime_app=SimpleNamespace(screen=SimpleNamespace()),
    )

    assert coordinator.snapshot.status == (
        'Skipped — a skill named "safe-name" already exists.'
    )


def test_stale_candidate_modal_callback_cannot_target_a_new_package(monkeypatch):
    """A callback owns the generation it displayed, never a later package."""
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    first = _multi_package()
    coordinator._pending_package = first
    coordinator.update(
        row_open=True,
        path="https://github.com/o/first",
        status="Choose one skill to import.",
        in_flight=True,
        candidates=first.inspection.candidates,
        generation=4,
    )
    pushed: dict[str, object] = {}
    workers: list[object] = []
    fake = SimpleNamespace(
        is_mounted=True,
        _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
        _library_skill_import_coordinator=coordinator,
        _library_skill_choice_presented_generation=-1,
    )
    fake.app = SimpleNamespace(
        screen=fake,
        push_screen=lambda modal, callback: pushed.update(
            modal=modal, callback=callback
        ),
        run_worker=lambda coroutine, **kwargs: workers.append(coroutine),
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )

    LibraryScreen._present_library_skills_import_choice_if_needed(fake)
    old_callback = pushed["callback"]
    assert coordinator.cancel_choice() is True
    second = _multi_package()
    coordinator._pending_package = second
    coordinator.update(
        status="Choose one skill to import.",
        in_flight=True,
        candidates=second.inspection.candidates,
        generation=5,
    )

    old_callback("skills/a")

    assert workers == []
    assert coordinator.snapshot.generation == 5


@pytest.mark.parametrize("stale_choice", (None, "skills/a"))
def test_replaced_library_screen_fences_stale_choice_callback(
    monkeypatch, stale_choice
):
    """Only the current screen may resolve one app-owned candidate choice."""
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace(skills_scope_service=SimpleNamespace())
    )
    package = _multi_package()
    coordinator._pending_package = package
    coordinator.update(
        row_open=True,
        path="https://example.com",
        status="Choose one skill to import.",
        in_flight=True,
        candidates=package.inspection.candidates,
        package_kind=SkillPackageKind.MULTI_SKILL_REPOSITORY.value,
        generation=11,
    )
    pushed: list[tuple[object, object]] = []
    workers: list[object] = []

    def screen():
        return SimpleNamespace(
            is_mounted=True,
            _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
            _library_skill_import_coordinator=coordinator,
            _library_skill_choice_presented_generation=-1,
        )

    def run_worker(coroutine, **kwargs):
        workers.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )
    old = screen()
    old.app = SimpleNamespace(
        screen=old,
        push_screen=lambda modal, callback: pushed.append((modal, callback)),
        run_worker=run_worker,
    )
    LibraryScreen._present_library_skills_import_choice_if_needed(old)
    old_callback = pushed[-1][1]

    replacement = screen()
    replacement.app = old.app
    replacement.app.screen = replacement
    LibraryScreen._present_library_skills_import_choice_if_needed(replacement)
    LibraryScreen._present_library_skills_import_choice_if_needed(replacement)
    assert len(pushed) == 2

    old_callback(stale_choice)

    assert coordinator.snapshot.candidates == package.inspection.candidates
    assert coordinator.snapshot.in_flight is True
    assert workers == []


@pytest.mark.asyncio
async def test_coordinator_settles_accepted_import_before_consuming_cancellation(
    monkeypatch,
):
    """Cancellation cannot release admission while the mutation still runs."""
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace()
    )
    started = asyncio.Event()
    release = asyncio.Event()
    presentations: list[bool] = []

    async def blocked_import(_raw_path):
        started.set()
        await release.wait()
        return library_skill_import_controller._LibrarySkillImportOutcome(
            'Imported "cancel-safe" · re-review it in the trust panel',
            review_name="cancel-safe",
            clear_path=True,
        )

    screen = SimpleNamespace(
        _present_library_skills_import_snapshot=lambda *, refresh_sources: (
            presentations.append(refresh_sources)
        )
    )
    runtime_app = SimpleNamespace(screen=screen)
    monkeypatch.setattr(coordinator, "_import", blocked_import)
    coordinator.open_draft()
    assert coordinator.claim("/accepted") is True
    worker = asyncio.create_task(
        coordinator.run("/accepted", runtime_app=runtime_app)
    )
    await started.wait()

    worker.cancel()
    await asyncio.sleep(0)
    assert coordinator.snapshot.in_flight is True
    assert worker.done() is False

    release.set()
    await worker
    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.review_name == "cancel-safe"
    assert coordinator.snapshot.status.startswith('Imported "cancel-safe"')
    assert presentations == [False]


def _claimed_import_coordinator():
    """Return one coordinator holding an accepted import."""
    coordinator = library_skill_import_controller.LibrarySkillImportCoordinator(
        SimpleNamespace()
    )
    assert coordinator.open_draft() is True
    assert coordinator.claim("/accepted") is True
    return coordinator


@pytest.mark.asyncio
async def test_coordinator_inner_cancellation_settles_fail_closed_once(
    monkeypatch,
):
    """A cancelled underlying operation releases and publishes one failure."""
    coordinator = _claimed_import_coordinator()
    presentations: list[bool] = []

    async def cancelled_import(_raw_path):
        raise asyncio.CancelledError

    monkeypatch.setattr(coordinator, "_import", cancelled_import)
    runtime_app = SimpleNamespace(
        screen=SimpleNamespace(
            _present_library_skills_import_snapshot=(
                lambda *, refresh_sources: presentations.append(refresh_sources)
            )
        )
    )

    await coordinator.run("/accepted", runtime_app=runtime_app)

    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.status == "Could not import that skill."
    assert presentations == [False]


@pytest.mark.asyncio
async def test_coordinator_fatal_base_exception_settles_once_before_reraise(
    monkeypatch,
    capsys,
):
    """A fatal operation outcome cannot strand or leak its exception text."""
    coordinator = _claimed_import_coordinator()
    presentations: list[bool] = []

    class FatalProbe(BaseException):
        pass

    async def fatal_import(_raw_path):
        raise FatalProbe("sensitive diagnostic probe")

    monkeypatch.setattr(coordinator, "_import", fatal_import)
    runtime_app = SimpleNamespace(
        screen=SimpleNamespace(
            _present_library_skills_import_snapshot=(
                lambda *, refresh_sources: presentations.append(refresh_sources)
            )
        )
    )

    with pytest.raises(FatalProbe):
        await coordinator.run("/accepted", runtime_app=runtime_app)

    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.status == "Could not import that skill."
    assert presentations == [False]
    captured = capsys.readouterr()
    assert "sensitive diagnostic probe" not in captured.out
    assert "sensitive diagnostic probe" not in captured.err


@pytest.mark.asyncio
async def test_coordinator_repeated_outer_cancellation_waits_for_one_outcome(
    monkeypatch,
):
    """Repeated owner cancellation cannot detach the accepted operation."""
    coordinator = _claimed_import_coordinator()
    started = asyncio.Event()
    release = asyncio.Event()
    landed = asyncio.Event()
    presentations: list[bool] = []
    lands = 0

    async def blocked_import(_raw_path):
        nonlocal lands
        started.set()
        await release.wait()
        lands += 1
        landed.set()
        return library_skill_import_controller._LibrarySkillImportOutcome(
            'Imported "one" · re-review it in the trust panel',
            review_name="one",
            clear_path=True,
        )

    monkeypatch.setattr(coordinator, "_import", blocked_import)
    runtime_app = SimpleNamespace(
        screen=SimpleNamespace(
            _present_library_skills_import_snapshot=(
                lambda *, refresh_sources: presentations.append(refresh_sources)
            )
        )
    )
    owner = asyncio.create_task(
        coordinator.run("/accepted", runtime_app=runtime_app)
    )
    await started.wait()

    try:
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        assert owner.done() is False
        assert coordinator.snapshot.in_flight is True
    finally:
        release.set()
        await landed.wait()
    assert lands == 1
    assert coordinator.snapshot.in_flight is False
    assert coordinator.snapshot.review_name == "one"
    assert presentations == [False]
    await owner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "second_trigger", "expected_name"),
    (
        ("loose", "enter", "loose-skill"),
        ("folder", "button", "folder-skill"),
        ("zip", "enter", "zip-skill"),
        ("url", "button", "remote-skill"),
    ),
)
async def test_skill_import_is_single_flight_across_every_route_and_navigation(
    tmp_path, monkeypatch, route, second_trigger, expected_name
):
    """A second submit cannot replace a real blocked threaded import.

    The service barrier proves that the first filesystem/network-shaped call is
    still running while both the presentation and handler authorization gates
    are exercised. Leaving Skills remains available; returning exposes the
    accepted operation's actual terminal outcome rather than a cancelled or
    replacement worker's status.
    """
    store_dir = tmp_path / "store"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _local_service, service = _real_skills_scope_service_with_trust(store_dir)
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    if route == "loose":
        import_value = source_dir / "loose-skill.md"
        import_value.write_text(
            "---\ndescription: Loose route fixture.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        owner, attribute = service, "import_skill_file"
    elif route == "folder":
        import_value = source_dir / "folder-skill"
        import_value.mkdir()
        (import_value / "SKILL.md").write_text(
            "---\nname: folder-skill\ndescription: Folder route fixture.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        owner, attribute = service, "import_skill_directory"
    elif route == "zip":
        import_value = source_dir / "zip-skill.zip"
        _write_single_skill_zip(import_value, name="zip-skill")
        owner, attribute = service, "import_skill_file"
    else:
        import_value = "https://github.com/example/remote-skill"
        owner, attribute = library_skill_import_controller, "inspect_skill_from_url"

    original = getattr(owner, attribute)
    first_started = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls: list[int] = []
    calls_lock = threading.Lock()

    async def blocked_call(*args, **kwargs):
        with calls_lock:
            calls.append(len(calls) + 1)
            call_number = len(calls)
        (first_started if call_number == 1 else second_started).set()
        await asyncio.to_thread(release.wait)
        try:
            if route == "url":
                return _root_package()
            return await original(*args, **kwargs)
        finally:
            if call_number == 1:
                finished.set()

    monkeypatch.setattr(owner, attribute, blocked_call)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-skills", Button).press()
            await _wait_for_selector(screen, pilot, "#library-skills-import")
            screen.handle_library_skills_import(SimpleNamespace(stop=lambda: None))
            await _wait_for_selector(screen, pilot, "#library-skills-import-path")
            path_input = screen.query_one("#library-skills-import-path", Input)
            path_input.value = str(import_value)
            await pilot.pause()
            screen.query_one("#library-skills-import-run", Button).press()
            assert await asyncio.to_thread(first_started.wait, 5)
            await pilot.pause()

            assert screen._library_skills_import_in_flight is True
            assert screen._library_skills_import_status == "Inspecting/importing…"
            for selector in (
                "#library-skills-import-path",
                "#library-skills-import-browse",
                "#library-skills-import-browse-folder",
                "#library-skills-import-run",
                "#library-skills-import-cancel",
            ):
                assert screen.query_one(selector).disabled is True

            event = SimpleNamespace(stop=lambda: None)
            if second_trigger == "enter":
                screen.handle_library_skills_import_path_submitted(event)
            else:
                screen.handle_library_skills_import_run(event)
            await pilot.pause()
            assert second_started.is_set() is False
            assert calls == [1]
            assert (
                screen._library_skills_import_status
                == "An import is already in progress."
            )

            media_row = screen.query_one("#library-row-browse-media", Button)
            assert media_row.disabled is False
            media_row.press()
            await _wait_for_selector(screen, pilot, "#library-media-canvas")
            assert screen._library_skills_import_in_flight is True

            release.set()
            assert await asyncio.to_thread(finished.wait, 10)
            for _ in range(100):
                if not screen._library_skills_import_in_flight:
                    break
                await pilot.pause()
            assert screen._library_skills_import_in_flight is False

            screen.query_one("#library-row-browse-skills", Button).press()
            await _wait_for_selector(screen, pilot, "#library-skills-import-path")
            status = str(
                screen.query_one("#library-skills-import-status").renderable
            )
            assert (
                status
                == f'Imported "{expected_name}" · re-review it in the trust panel'
            )
            assert calls == [1]
    finally:
        release.set()


@pytest.mark.asyncio
async def test_routed_library_replacement_observes_and_refuses_app_owned_import(
    tmp_path, monkeypatch
):
    """A fresh routed screen cannot admit work hidden by its predecessor."""
    store_dir = tmp_path / "store"
    source = tmp_path / "routed-skill.md"
    source.write_text(
        "---\ndescription: Routed owner fixture.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    _local_service, service = _real_skills_scope_service_with_trust(store_dir)
    original = service.import_skill_file
    started = threading.Event()
    release = threading.Event()
    landed = threading.Event()
    calls: list[int] = []

    def blocking_import(*args, **kwargs):
        calls.append(len(calls) + 1)
        started.set()
        release.wait(15)
        try:
            return asyncio.run(original(*args, **kwargs))
        finally:
            landed.set()

    monkeypatch.setattr(service, "import_skill_file", blocking_import)
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            original_screen = _active_library_screen(host)
            await _wait_for_library_shell(original_screen, pilot)
            await _open_skills_import_row(original_screen, pilot)
            original_screen.query_one(
                "#library-skills-import-path", Input
            ).value = str(source)
            await pilot.pause()
            original_screen.query_one("#library-skills-import-run", Button).press()
            assert await asyncio.to_thread(started.wait, 5)

            await host.switch_screen(Screen())
            replacement = LibraryScreen(app)
            await host.switch_screen(replacement)
            await _wait_for_library_shell(replacement, pilot)

            assert host.screen is replacement
            assert original_screen not in host.screen_stack
            assert replacement._library_skills_import_in_flight is True
            assert replacement._library_skills_import_open is True
            skills_row = await _wait_for_selector(
                replacement, pilot, "#library-row-browse-skills"
            )
            assert isinstance(skills_row, Button)
            skills_row.press()
            await _wait_for_selector(
                replacement, pilot, "#library-skills-import-path"
            )
            for selector in (
                "#library-skills-import-path",
                "#library-skills-import-browse",
                "#library-skills-import-browse-folder",
                "#library-skills-import-run",
                "#library-skills-import-cancel",
            ):
                assert replacement.query_one(selector).disabled is True

            replacement._start_library_skills_import()
            await pilot.pause()
            assert calls == [1]
            assert (
                replacement._library_skills_import_status
                == "An import is already in progress."
            )

            release.set()
            assert await asyncio.to_thread(landed.wait, 10)
            await _wait_for_skill_import_terminal(
                replacement,
                pilot,
                expected_status=(
                    'Imported "routed-skill" · re-review it in the trust panel'
                ),
            )
            assert replacement._library_skills_import_review_name == "routed-skill"
            assert calls == [1]
    finally:
        release.set()


@pytest.mark.asyncio
async def test_completed_import_receipt_survives_rail_departure_and_return(tmp_path):
    """A completed receipt remains visible after ordinary Library navigation."""
    source = tmp_path / "completed-skill.md"
    source.write_text(
        "---\ndescription: Completed receipt fixture.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    _local_service, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)
    expected = 'Imported "completed-skill" · re-review it in the trust panel'

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)
        screen.query_one("#library-skills-import-path", Input).value = str(source)
        await pilot.pause()
        screen.query_one("#library-skills-import-run", Button).press()
        await _wait_for_skill_import_terminal(
            screen, pilot, expected_status=expected
        )

        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        assert screen._library_skills_import_open is True
        assert screen._library_skills_import_status == expected

        screen.query_one("#library-row-browse-skills", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skills-import-path")
        assert (
            str(screen.query_one("#library-skills-import-status").renderable)
            == expected
        )
        assert screen._library_skills_import_review_name == "completed-skill"


@pytest.mark.asyncio
async def test_import_completed_while_away_survives_another_rail_move(
    tmp_path, monkeypatch
):
    """Off-canvas settlement remains after later navigation before return."""
    source = tmp_path / "away-skill.md"
    source.write_text(
        "---\ndescription: Away receipt fixture.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    _local_service, service = _real_skills_scope_service_with_trust(tmp_path / "store")
    original = service.import_skill_file
    started = threading.Event()
    release = threading.Event()

    async def blocked_import(*args, **kwargs):
        started.set()
        await asyncio.to_thread(release.wait)
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "import_skill_file", blocked_import)
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)
    expected = 'Imported "away-skill" · re-review it in the trust panel'

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_skills_import_row(screen, pilot)
            screen.query_one("#library-skills-import-path", Input).value = str(source)
            await pilot.pause()
            screen.query_one("#library-skills-import-run", Button).press()
            assert await asyncio.to_thread(started.wait, 5)

            screen.query_one("#library-row-browse-media", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-canvas")
            release.set()
            await _wait_for_skill_import_terminal(
                screen, pilot, expected_status=expected
            )

            screen.query_one("#library-row-browse-notes", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-canvas")
            assert screen._library_skills_import_open is True
            assert screen._library_skills_import_status == expected

            screen.query_one("#library-row-browse-skills", Button).press()
            await _wait_for_selector(screen, pilot, "#library-skills-import-path")
            assert (
                str(screen.query_one("#library-skills-import-status").renderable)
                == expected
            )
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "browse_handler",
    (
        LibraryScreen.handle_library_skills_import_browse,
        LibraryScreen.handle_library_skills_import_browse_folder,
    ),
)
async def test_abandoned_picker_cannot_write_into_reopened_import_row(
    tmp_path, monkeypatch, browse_handler
):
    """Cancel/reopen advances the row lifecycle beyond an old picker."""
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = _real_skills_scope_service_with_trust(
        tmp_path / "store"
    )[1]
    host = LibraryHarness(app)
    pushed: dict[str, object] = {}

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)
        monkeypatch.setattr(
            host,
            "push_screen",
            lambda dialog, callback=None: pushed.update(
                dialog=dialog, callback=callback
            ),
        )
        browse_handler(screen, SimpleNamespace(stop=lambda: None))
        callback = pushed["callback"]

        screen.query_one("#library-skills-import-cancel", Button).press()
        await _wait_for_selector(screen, pilot, "#library-skills-import")
        screen.query_one("#library-skills-import", Button).press()
        path_input = await _wait_for_selector(
            screen, pilot, "#library-skills-import-path"
        )
        assert isinstance(path_input, Input)
        path_input.value = "/new-draft"
        await pilot.pause()

        await callback(Path("/stale/SKILL.md"))

        assert screen._library_skills_import_path == "/new-draft"
        assert screen.query_one(
            "#library-skills-import-path", Input
        ).value == "/new-draft"


@pytest.mark.asyncio
async def test_import_real_superpowers_skills_lands_trust_pending(tmp_path):
    """Import every real superpowers fixture skill through the actual
    Library Import row (rail row -> Import… -> path -> Import), then
    assert each one lands TRUST-PENDING (blocked) per the spec, with the
    exact outcome-line copy the brief pins, and that its persisted
    name/description validate against ``skills_schemas``'s own
    constraints (the real names are lowercase-hyphenated and well under
    the 64-char limit; the real descriptions are well under 1000 chars).
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app(configured_default="library")
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service

    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        for _ in range(200):
            if isinstance(app.screen, LibraryScreen):
                break
            await pilot.pause(0.01)
        assert isinstance(app.screen, LibraryScreen)
        screen = app.screen
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        for name in REAL_FIXTURE_SKILLS:
            status = await _run_skills_import_via_ui(screen, pilot, FIXTURES_DIR / name)
            # Name-specific outcome copy (task-291): a stale success line
            # from the PREVIOUS skill can never satisfy this assertion.
            assert status == f'Imported "{name}" · re-review it in the trust panel', (
                f"unexpected outcome importing {name!r}: {status!r}"
            )

    context = await service.get_context(mode="local")
    blocked_names = {item["name"] for item in context["blocked_skills"]}
    assert blocked_names == set(REAL_FIXTURE_SKILLS)
    # TRUST-PENDING means blocked, not merely absent from the trusted list:
    # every imported skill must be blocked, none available yet.
    assert context["available_skills"] == []

    for name in REAL_FIXTURE_SKILLS:
        record = await local_service.get_skill(name)
        assert record["trust_blocked"] is True
        assert SKILL_NAME_PATTERN.match(record["name"]), record["name"]
        assert record["description"], f"{name} has no description"
        assert len(record["description"]) <= 1000


@pytest.mark.asyncio
async def test_import_skill_via_skill_md_file_path_derives_name_from_parent_directory(
    tmp_path,
):
    """Pointing the Import row at the ``SKILL.md`` FILE itself (not its
    parent directory) must resolve to the SAME correct skill name as
    pointing it at the directory -- the incompatibility this guards
    against: every real skill package uses the literal filename
    ``SKILL.md`` for every skill, so naively deriving the name from that
    file's own basename (as a generic ``import_skill_file(filename=...)``
    call would) produces the same wrong name ("skill") for every import
    regardless of which skill it actually is. The coordinator must use the
    PARENT DIRECTORY's name instead for this exact shape.

    Merge-gate regression (PR #784, IMPORTANT finding): this shape used to
    fall through to a flat, text-only read (dropping nested subfolders,
    binaries, and the executable bit) instead of routing through the SAME
    faithful ``import_skill_directory`` copy the directory-path shape
    already used -- since a SKILL.md file's PARENT dir IS the skill
    directory, both shapes must behave identically. Proven here by copying
    the real fixture (never mutating the committed copy -- see the
    fixtures README's "unmodified copies" provenance note) into ``tmp_path``
    and adding a nested ``references/`` subfolder plus an executable
    sibling script before importing via the SKILL.md FILE path.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = tmp_path / "verification-before-completion"
    shutil.copytree(
        FIXTURES_DIR / "verification-before-completion", skill_dir
    )
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    (references_dir / "note.md").write_text(
        "A nested reference file.", encoding="utf-8"
    )
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_path = scripts_dir / "run.sh"
    script_path.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o100)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        skill_md_path = skill_dir / "SKILL.md"
        status = await _run_skills_import_via_ui(screen, pilot, skill_md_path)
        assert (
            status
            == 'Imported "verification-before-completion" · re-review it in the trust panel'
        )

    record = await local_service.get_skill("verification-before-completion")
    assert record["name"] == "verification-before-completion"
    assert record["trust_blocked"] is True
    with pytest.raises(Exception):
        await local_service.get_skill("skill")

    # The nested subfolder must be imported and readable by its nested key
    # -- the flat-read fallback used to skip it entirely (nested
    # subdirectories are NOT recursed into by a flat sibling scan).
    supporting_files = record.get("supporting_files") or {}
    assert supporting_files.get("references/note.md") == "A nested reference file."

    # The sibling script's executable bit must survive the copy too --
    # further proof this shape now goes through the faithful bundle copy
    # (which preserves owner-exec), not the text-only flat read (which
    # never even considered file modes).
    bundle_files = {item["path"]: item for item in record.get("bundle_files") or []}
    assert "scripts/run.sh" in bundle_files
    assert bundle_files["scripts/run.sh"]["executable"] is True


@pytest.mark.asyncio
async def test_loose_file_import_success_line_names_the_service_derived_skill(tmp_path):
    """task-291 review: the loose-file branch imports via
    ``import_skill_file`` with NO explicit name, so the service derives it
    (lowercase kebab). The success line must show that STORED name, not the
    raw file stem -- 'My Notes.md' imports as 'my-notes' and must say so."""
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    loose = tmp_path / "My Notes.md"
    loose.write_text(
        "---\ndescription: A loose skill file.\n---\n\nLoose body.\n",
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, loose)
        assert status == 'Imported "my-notes" · re-review it in the trust panel'

    record = await local_service.get_skill("my-notes")
    assert record["name"] == "my-notes"


@pytest.mark.asyncio
async def test_import_skill_with_supporting_reference_file_threads_it_through(tmp_path):
    """``requesting-code-review`` has one real flat sibling file
    (``code-reviewer.md``) alongside its ``SKILL.md`` -- importing it by
    directory path must carry that sibling through as a supporting file
    with its exact real content, not silently drop it.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = FIXTURES_DIR / "requesting-code-review"
    real_supporting_content = (skill_dir / "code-reviewer.md").read_text(
        encoding="utf-8"
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert (
            status
            == 'Imported "requesting-code-review" · re-review it in the trust panel'
        )

    record = await local_service.get_skill("requesting-code-review")
    assert record["supporting_files"] == {"code-reviewer.md": real_supporting_content}


@pytest.mark.asyncio
async def test_import_skill_with_extra_frontmatter_fields_applies_recognized_and_drops_unknown(
    tmp_path,
):
    """``executing-plans-with-metadata`` is a synthetic fixture (real
    executing-plans description/body, augmented frontmatter -- see the
    fixtures README) that exercises frontmatter fields the real
    superpowers skillset never actually uses on its own
    (argument_hint/allowed-tools/license/compatibility/model/context/
    metadata). Recognized fields must be applied; the two UNRECOGNIZED
    fields it also carries (``priority``, ``tags``) must be silently
    dropped -- ``local_skills_service``'s own frontmatter parser filters
    to a fixed known-fields allowlist with no rejection/warning path, so
    this is the actual (not hypothetical) incompatibility surface between
    this schema and an arbitrary external skill spec.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(
            screen,
            pilot,
            FIXTURES_DIR / "executing-plans-with-metadata",
        )
        assert (
            status
            == 'Imported "executing-plans-with-metadata" · re-review it in the trust panel'
        )

    record = await local_service.get_skill("executing-plans-with-metadata")
    assert record["argument_hint"] == "plan file path"
    assert record["allowed_tools"] == ["Read", "Write", "Bash"]
    assert record["model"] == "claude-sonnet-5"
    assert record["context"] == "fork"
    assert record["license"] == "MIT"
    assert record["compatibility"] == "Claude Code, Codex CLI"
    assert record["metadata"] == {
        "origin": "superpowers",
        "upstream_skill": "executing-plans",
    }
    assert record["validation_status"] == "valid"
    # The unrecognized fields never reach the persisted record at all --
    # not present, not surfaced as a validation error either. (``version``
    # is deliberately NOT checked here: it collides with the skill's own
    # legitimate revision-counter field, always present regardless of
    # frontmatter content.)
    assert "priority" not in record
    assert "tags" not in record
    assert not any(
        "priority" in error or "tags" in error for error in record["validation_errors"]
    )


@pytest.mark.asyncio
async def test_reimporting_the_same_skill_name_is_skipped_not_duplicated(tmp_path):
    """Importing the same real skill twice must skip the second attempt
    (never silently overwrite, matching the prompts import's own
    duplicate-name posture) and report which name collided.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = FIXTURES_DIR / "executing-plans"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        first_status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert (
            first_status
            == 'Imported "executing-plans" · re-review it in the trust panel'
        )

        second_status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert (
            second_status == 'Skipped — a skill named "executing-plans" already exists.'
        )

    record = await local_service.get_skill("executing-plans")
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_import_row_reports_missing_skill_md_and_unknown_path_gracefully(
    tmp_path,
):
    """A folder with no ``SKILL.md`` and a path that does not exist at all
    both surface a specific, honest outcome line -- never a crash, never a
    silent no-op that leaves the user guessing.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    empty_folder = tmp_path / "not-a-skill"
    empty_folder.mkdir()
    missing_path = tmp_path / "does-not-exist"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, empty_folder)
        assert status == "That package is malformed or unsupported."

        status = await _run_skills_import_via_ui(screen, pilot, missing_path)
        assert status == "Could not find that file or folder."

        loose_file = tmp_path / "notes.txt"
        loose_file.write_text("not a skill", encoding="utf-8")
        status = await _run_skills_import_via_ui(screen, pilot, loose_file)
        assert status == "Unsupported file type."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_rejects_name_too_long_without_partial_state(tmp_path):
    """A skill whose DIRECTORY name exceeds the 64-character
    ``skills_schemas`` limit must fail cleanly (a real incompatibility a
    misnamed import folder could trigger) -- reported as a specific
    failure, and nothing partially written (``import_skill`` validates
    before any disk write, so a failed import must leave the skill store
    completely untouched).
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    long_name = "a" + "-oversized-skill-name" * 4  # well over 64 characters
    assert len(long_name) > 64
    skill_dir = tmp_path / "oversized" / long_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {long_name}\ndescription: Too long to import.\n---\n\nBody.\n",
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "Could not import that skill."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_rejects_oversized_content_without_partial_state(tmp_path):
    """A ``SKILL.md`` whose content exceeds ``skills_schemas``'s 500,000
    character limit must fail cleanly rather than crash or silently
    truncate -- constructed inline (not a committed fixture) to keep the
    fixtures directory small.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = tmp_path / "oversized-content"
    skill_dir.mkdir()
    oversized_body = "x" * 600_000
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: oversized-content\ndescription: Body is too large.\n---\n\n{oversized_body}\n",
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "Could not import that skill."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_imports_nested_reference_subfolder(
    tmp_path,
):
    """A skill directory with a NESTED reference subfolder (the real
    ``using-superpowers`` skill's own ``references/`` layout -- not
    copied into the fixtures dir to keep it small, reproduced here
    structurally) must import successfully AND carry the nested file
    through as a supporting file, keyed by its nested relative path.
    ``local_skills_service``'s bundle-file walk recurses into
    subdirectories (junk pruned, symlinks skipped, caps enforced) so the
    real skill's ``references/`` layout round-trips faithfully instead
    of being silently dropped.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = tmp_path / "nested-refs-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: nested-refs-skill\ndescription: Has a nested references folder.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    (references_dir / "note.md").write_text(
        "A nested reference file.", encoding="utf-8"
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert (
            status == 'Imported "nested-refs-skill" · re-review it in the trust panel'
        )

    record = await local_service.get_skill("nested-refs-skill")
    assert record["trust_blocked"] is True
    supporting_files = record.get("supporting_files") or {}
    assert "references/note.md" in supporting_files
    assert supporting_files["references/note.md"] == "A nested reference file."


# ---------------------------------------------------------------------------
# Task 5: the Import row's URL branch. ``install_skill_from_url`` (the
# network-touching seam) is monkeypatched IN THE SCREEN MODULE'S OWN
# NAMESPACE -- it is imported there directly (``from
# ...Skills_Interop.skill_remote_fetch import install_skill_from_url``),
# not looked up dynamically off a service object -- so these tests prove
# routing/outcome-translation only, never real network I/O. The seam's own
# network/classification/re-root behavior is covered by
# ``Tests/Skills/test_skill_remote_fetch.py``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_row_url_routes_to_remote_install_and_primes_review(
    tmp_path, monkeypatch
):
    """A pasted ``http(s)://`` URL in the Import row's path field routes to
    ``install_skill_from_url`` instead of the local file/folder path, and a
    successful install reports the SAME outcome shape as every other
    import path: the outcome line names the service-reported skill, and
    the Review button is primed with it (mirrors
    ``test_skills_import_success_offers_review_button`` in
    ``Tests/UI/test_library_skills_canvas.py``).
    """
    _local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    captured: dict = {}

    async def _fake_inspect_skill_from_url(url, *, scope_service, **kwargs):
        captured["url"] = url
        captured["scope_service"] = scope_service
        captured["kwargs"] = kwargs
        return _root_package()

    monkeypatch.setattr(
        library_skill_import_controller,
        "inspect_skill_from_url",
        _fake_inspect_skill_from_url,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(
            screen,
            pilot,
            "https://github.com/o/remote-skill",
            deadline_seconds=2,
        )
        assert captured, (
            app.library_skill_import_coordinator.snapshot,
            host.screen is screen,
            screen.is_mounted,
        )
        expected = 'Imported "remote-skill" · re-review it in the trust panel'
        assert status == expected, (
            app.library_skill_import_coordinator.snapshot,
            getattr(
                screen.query_one("#library-skills-canvas"),
                "import_status",
                None,
            ),
            [
                str(widget.renderable)
                for widget in screen.query("#library-skills-import-status")
            ],
        )
        review = screen.query_one("#library-skills-import-review", Button)
        assert "remote-skill" in str(review.label)

    assert captured["url"] == "https://github.com/o/remote-skill"
    assert captured["scope_service"] is service


@pytest.mark.asyncio
async def test_import_row_url_remote_skill_error_becomes_status_line(
    tmp_path, monkeypatch
):
    """Malformed remote input is surfaced as one classified safe outcome."""
    _local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async def _fake_inspect_skill_from_url(url, *, scope_service, **kwargs):
        return RemoteSkillPackage(
            SkillPackageInspection(
                SkillPackageKind.MALFORMED_OR_UNSUPPORTED,
                message="That URL is malformed or unsupported.",
            )
        )

    monkeypatch.setattr(
        library_skill_import_controller,
        "inspect_skill_from_url",
        _fake_inspect_skill_from_url,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(
            screen, pilot, "https://github.com/o/r/releases/download/v1/pkg"
        )
        assert status == "That URL is malformed or unsupported."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_url_generic_failure_uses_classified_name_guess(
    tmp_path, monkeypatch
):
    """A non-``RemoteSkillError`` failure (e.g. the underlying
    ``import_skill_file`` call rejecting a duplicate name) routes through
    the SAME coordinator exception translator every other import path uses,
    with a name guess derived from classifying the URL -- the same "derive a plausible
    skill name from what the user typed" convention the loose-file branch
    uses (``file_path.stem``), here via the seam's own classification so
    the guess matches what the seam would actually have named the skill.
    """
    _local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async def _fake_inspect_skill_from_url(url, *, scope_service, **kwargs):
        return _root_package("brainstorm")

    async def _fake_import_inspected_skill(*args, **kwargs):
        raise ValueError("local_skill_exists:brainstorm")

    monkeypatch.setattr(
        library_skill_import_controller,
        "inspect_skill_from_url",
        _fake_inspect_skill_from_url,
    )
    monkeypatch.setattr(
        library_skill_import_controller,
        "import_inspected_skill",
        _fake_import_inspected_skill,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(
            screen, pilot, "https://github.com/o/brainstorm/tree/main"
        )
        assert status == 'Skipped — a skill named "brainstorm" already exists.'
