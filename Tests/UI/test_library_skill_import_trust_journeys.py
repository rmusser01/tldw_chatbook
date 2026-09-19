"""Production-style import and exact trust-review keyboard journeys."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input, Static, TextArea

from Tests.Skills.test_skills_library_flow import (
    _real_uninitialized_trust_service,
    _wire_empty_non_skill_services,
)
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_assistant_turn import (
    _contrast,
    _painted_foreground_and_background,
)
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skill_editor_journeys import _activate
from Tests.UI.test_library_skills_canvas import _build_test_app
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.UI.Library_Modules.skill_import_choice_modal import (
    SkillImportChoiceModal,
)
from tldw_chatbook.UI.Screens.skills_screen import (
    SkillTrustBootstrapModal,
    SkillTrustPassphraseModal,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("bootstrap", [True, False])
async def test_trust_dialog_text_and_validation_are_readable(
    monkeypatch, size, theme, bootstrap
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    host = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    host.theme = theme
    modal = (
        SkillTrustBootstrapModal()
        if bootstrap
        else SkillTrustPassphraseModal(confirm_bootstrap=False)
    )
    prefix = "skill-trust-bootstrap" if bootstrap else "skill-trust-passphrase"
    results = []
    async with host.run_test(size=size) as pilot:
        host.push_screen(modal, results.append)
        await pilot.pause()
        assert modal.focused is modal.query_one(f"#{prefix}-input")
        await pilot.press("enter")
        await pilot.pause()
        error = modal.query_one(f"#{prefix}-error", Static)
        assert str(error.renderable) == "Passphrase cannot be blank."
        for widget in (modal.query_one(f"#{prefix}-message"), error):
            foreground, background = _painted_foreground_and_background(host, widget)
            assert _contrast(foreground, background) >= 4.5
        assert not results
        modal.query_one(f"#{prefix}-input", Input).value = "synthetic-qa-passphrase"
        if bootstrap:
            confirm = modal.query_one(f"#{prefix}-confirm-input", Input)
            confirm.value = "mismatch"
            await pilot.press("enter")
            await pilot.pause()
            assert str(error.renderable) == "Passphrases do not match."
            assert not results
            confirm.value = "synthetic-qa-passphrase"
        await _activate(modal, host, pilot, f"#{prefix}-submit", "Submit")
        assert results == ["synthetic-qa-passphrase"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_import_choice_and_exact_trust_approval_journey(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    store = tmp_path / "store"
    trust = _real_uninitialized_trust_service(store)
    await asyncio.to_thread(trust.bootstrap_trust, "synthetic-qa-passphrase")
    local = LocalSkillsService(store_dir=store, trust_service=trust)
    source = tmp_path / "package"
    for name in ("alpha", "zeta"):
        folder = source / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Imported café\n---\nBody {name} [literal].\n"
        )
    app = _build_test_app()
    app.library_new_profile_admission = True
    _wire_empty_non_skill_services(app)
    app.local_skills_service = local
    app.local_skill_trust_service = trust
    app.skills_scope_service = SkillsScopeService(
        local_service=local, server_service=None
    )
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    app.notify = host.notify
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-skills")
        await _wait_for_selector(screen, pilot, "#library-skills-items-grip")
        shell = screen.query_one("#library-skills-reader-shell")
        await _wait_for_condition(
            pilot,
            lambda: shell.effective_layout.reader_width > 0,
            message="Skills layout did not settle",
        )
        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _activate(screen, host, pilot, "#library-skills-sort", "sort")
        await _activate(screen, host, pilot, "#library-skills-sort-status", "Status")
        await _activate(screen, host, pilot, "#library-skills-import", "Import skill")
        path = await _wait_for_selector(screen, pilot, "#library-skills-import-path")
        path.value = str(tmp_path / "missing")
        await pilot.pause()
        await _activate(screen, host, pilot, "#library-skills-import-run", "Import")
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._library_skills_import_in_flight
                and bool(screen._library_skills_import_status)
            ),
            message="Missing path did not report an outcome",
        )
        await _focus(
            screen,
            host,
            pilot,
            "#library-skills-import-path",
            str(tmp_path / "missing")[-7:],
        )
        assert not screen._library_skills_import_review_name
        path = screen.query_one("#library-skills-import-path", Input)
        path.value = str(source)
        await pilot.pause()
        work = screen.query_one("#library-skill-work-pane")
        after_recompose = work._after_recompose
        move_after_cancel = False
        newer_focus = None

        def tab_before_cancel_callback():
            nonlocal move_after_cancel, newer_focus
            after_recompose()
            if (
                move_after_cancel
                and host.screen is screen
                and not work.import_in_flight
            ):
                move_after_cancel = False
                screen.focus_next()
                newer_focus = screen.focused

        monkeypatch.setattr(work, "_after_recompose", tab_before_cancel_callback)
        for cancel in (True, False):
            await _activate(screen, host, pilot, "#library-skills-import-run", "Import")
            await _wait_for_condition(
                pilot,
                lambda: isinstance(host.screen, SkillImportChoiceModal),
                message="Candidate choice did not open",
            )
            choice = host.screen
            await pilot.pause()
            if cancel:
                move_after_cancel = size == (170, 48) and theme == "textual-light"
                await _activate(
                    choice, host, pilot, "#skill-import-choice-cancel", "Cancel"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        host.screen is screen
                        and not screen._library_skills_import_in_flight
                    ),
                    message="Candidate Cancel did not settle",
                )
                if size == (170, 48) and theme == "textual-light":
                    await _wait_for_condition(
                        pilot,
                        lambda: newer_focus is not None,
                        message="Newer Tab did not land",
                    )
                    await pilot.pause()
                    assert screen.focused is newer_focus
                else:
                    await _wait_for_condition(
                        pilot,
                        lambda: (
                            screen.focused
                            is screen.query_one("#library-skills-import-path")
                        ),
                        message="Candidate Cancel lost path focus",
                    )
                assert not list(local.skills_dir.glob("*/SKILL.md"))
            else:
                await pilot.press("down")
                assert choice.query_one("#skill-import-choice-list").highlighted == 1
                await _activate(
                    choice, host, pilot, "#skill-import-choice-import", "Import skill"
                )
        await _wait_for_condition(
            pilot,
            lambda: (
                host.screen is screen
                and screen._library_skills_import_review_name == "zeta"
                and not screen._library_skills_import_in_flight
            ),
            message=lambda: (
                f"Selected import: {host.screen!r}; {screen._library_skill_import_coordinator.snapshot!r}"
            ),
        )
        await _focus(
            screen, host, pilot, "#library-skills-import-review", 'Review "zeta"'
        )
        assert (await local.get_skill("zeta"))["trust_blocked"]
        assert not (local.skills_dir / "alpha").exists()
        await _activate(
            screen, host, pilot, "#library-skills-import-review", 'Review "zeta"'
        )
        await _wait_for_selector(screen, pilot, "#library-skill-trust-review")
        await _focus(
            screen, host, pilot, "#library-skill-trust-review", "Review changes"
        )
        assert screen._skills_state.reader_mode == "trust"
        await _activate(
            screen, host, pilot, "#library-skill-trust-review", "Review changes"
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._skills_state.active_review is not None,
            message="Trust review was not captured",
        )
        assert "Body zeta [literal]." in str(
            screen.query_one("#library-skill-trust-review-content", Static).renderable
        )
        await _focus(
            screen, host, pilot, "#library-skill-trust-review", "Review changes"
        )
        # Approval must reject a file changed after capture, then permit a fresh review.
        skill_path = local.skills_dir / "zeta" / "SKILL.md"
        skill_path.write_text(skill_path.read_text() + "Newer on disk.\n")
        for stale in (True, False):
            await _activate(
                screen, host, pilot, "#library-skill-trust-approve", "Approve"
            )
            await _wait_for_condition(
                pilot,
                lambda: isinstance(host.screen, SkillTrustPassphraseModal),
                message="Approval passphrase did not open",
            )
            modal = host.screen
            modal.query_one(
                "#skill-trust-passphrase-input", Input
            ).value = "synthetic-qa-passphrase"
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    host.screen is screen and screen._skills_state.active_review is None
                ),
                message="Approval did not settle",
            )
            if stale:
                assert trust.status_for_skill("zeta").trust_blocked
                await _activate(
                    screen, host, pilot, "#library-skill-trust-review", "Review changes"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: screen._skills_state.active_review is not None,
                    message="Fresh review did not settle",
                )
                assert "Newer on disk." in str(
                    screen.query_one(
                        "#library-skill-trust-review-content", Static
                    ).renderable
                )
            else:
                await _wait_for_condition(
                    pilot,
                    lambda: not screen._skills_state.editor_state.trust_blocked,
                    message="Approved trust state did not refresh",
                )
                assert screen.query_one("#library-skill-trust-approve").disabled
                await _focus(screen, host, pilot, "#library-skill-mode-trust", "Trust")
                assert not trust.status_for_skill("zeta").trust_blocked

        # Closed compact Items intentionally disables its rows until reopened.
        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_skills_browse_controller.freshness == "fresh"
                and screen.query_one("#library-skill-row-zeta").focusable
            ),
            message="Trust approval left the visible Skills row stale or disabled",
        )
        # Import remains available beside a selected Skill, but cannot hide a draft.
        await _activate(screen, host, pilot, "#library-skills-import", "Import skill")
        await _wait_for_selector(screen, pilot, "#library-skills-import-path")
        await _activate(screen, host, pilot, "#library-skills-import-cancel", "Cancel")
        await _focus(screen, host, pilot, "#library-skills-import", "Import skill")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_skills_browse_controller.freshness == "fresh"
                and not screen.query_one("#library-skills-canvas")._recompose_required
                and not screen.query_one(
                    "#library-skills-canvas"
                ).has_pending_recompose_callback
            ),
            message="Refreshed Skills list did not settle",
        )
        await _activate(screen, host, pilot, "#library-skill-row-zeta", "zeta")
        await _activate(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        body = await _wait_for_selector(screen, pilot, "#library-skill-body")
        assert isinstance(body, TextArea)
        body.text = "Keep this unsaved draft."
        await pilot.pause()
        assert screen._skills_state.dirty
        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _activate(screen, host, pilot, "#library-skills-import", "Import skill")
        assert not screen._library_skills_import_open
        assert screen._skills_state.dirty
        assert screen.query_one("#library-skill-body") is body
        assert body.text == "Keep this unsaved draft."


@pytest.mark.asyncio
async def test_import_receipt_ignores_delayed_echo_and_stale_presentation(
    tmp_path, monkeypatch
):
    """Force a busy-field echo and a newer edit across deferred presentation."""
    import tldw_chatbook.UI.Screens.library_screen as library_module

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    local = LocalSkillsService(store_dir=tmp_path / "store")
    app.skills_scope_service = SkillsScopeService(
        local_service=local, server_service=None
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-skills")
        await _wait_for_selector(screen, pilot, "#library-skills-import")
        coordinator = screen._library_skill_import_coordinator
        coordinator.open_draft()
        coordinator.claim("/old-package")
        screen._present_library_skills_import_snapshot(refresh_sources=False)
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(screen.query("#library-skills-import-path"))
                and screen.query_one("#library-skills-import-path").disabled
            ),
            message="Busy path did not mount",
        )
        busy_input = screen.query_one("#library-skills-import-path", Input)
        terminal = coordinator.update(
            in_flight=False, path="", status='Imported "zeta"', review_name="zeta"
        )
        # The busy widget is still current while its replacement is queued.
        screen.handle_library_skills_import_path_changed(
            Input.Changed(busy_input, "/old-package")
        )
        assert coordinator.snapshot is terminal
        screen._present_library_skills_import_snapshot(refresh_sources=False)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen.query_one("#library-skills-import-path") is not busy_input
                and not screen.query_one("#library-skills-import-path").disabled
            ),
            message="Editable result path did not mount",
        )
        field = screen.query_one("#library-skills-import-path", Input)
        deferred = []

        def defer_presentation(*args, then=None, **kwargs):
            if then is not None:
                deferred.append(then)
            return True

        monkeypatch.setattr(library_module, "_sync_library_canvas", defer_presentation)
        screen._present_library_skills_import_snapshot(refresh_sources=False)
        field.value = "/new-draft"
        screen.handle_library_skills_import_path_changed(
            Input.Changed(field, field.value)
        )
        edited = coordinator.snapshot
        for callback in deferred:
            callback()
        await pilot.pause()
        assert coordinator.snapshot == edited
        assert edited.path == "/new-draft"
        assert edited.status == ""
        assert edited.review_name == ""
        assert edited.generation == terminal.generation + 1
