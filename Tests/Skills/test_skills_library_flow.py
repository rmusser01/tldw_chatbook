"""Real-service end-to-end tests for the Library Skills detail/trust editor
(Task 4 of the Skills sub-project).

Mirrors ``Tests/UI/test_library_prompts_canvas.py``'s real-service section:
a real ``LocalSkillsService``/``SkillsScopeService`` (and, for the trust
scenarios, a real ``SkillTrustService``) wired onto a real ``LibraryScreen``
via ``App.run_test()`` -- no hand-rolled fakes for the service layer, since
the conflict/trust-quarantine/policy-enforcement scenarios below depend on
the REAL service's actual return-value/exception shapes (see
``test_library_prompt_row_opens_editor_under_real_runtime_policy_enforcer``'s
docstring for why a fake-with-no-policy-enforcer would hide the exact class
of regression this suite guards against).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _real_skills_scope_service(
    tmp_path, *, trust_service=None, allow_untrusted: bool = True, policy_enforcer=None,
):
    """Build a real ``LocalSkillsService``/``SkillsScopeService`` pair.

    Args:
        tmp_path: The skill store's root directory.
        trust_service: A real ``SkillTrustService`` to wire in, or ``None``
            for the simple compat mode (``allow_untrusted``) most scenarios
            below use.
        allow_untrusted: Passed straight through to ``LocalSkillsService``
            -- irrelevant once a real ``trust_service`` is supplied.
        policy_enforcer: Wired into BOTH the local service and the scope
            service, mirroring how ``app.py`` wires the same enforcer
            instance into every scope-service layer.
    """
    local_service = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=trust_service,
        allow_untrusted_without_trust_service=allow_untrusted,
        policy_enforcer=policy_enforcer,
    )
    service = SkillsScopeService(
        local_service=local_service, server_service=None, policy_enforcer=policy_enforcer,
    )
    return local_service, service


def _real_trust_service(tmp_path) -> SkillTrustService:
    """A real, already-unlocked (but not yet bootstrapped) trust service."""
    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("trust-passphrase", salt=b"7" * 32)
    return trust_service


def _wire_empty_non_skill_services(app) -> None:
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])


async def _open_skill_editor(screen, pilot, skill_name: str) -> None:
    """Open the rail's Skills row, then a specific skill's row."""
    screen.query_one("#library-row-browse-skills").press()
    await pilot.pause()
    await pilot.pause()
    screen.query_one(f"#library-skill-row-{skill_name}", Button).press()
    await pilot.pause()
    for _ in range(150):
        if screen._library_skill_detail is not None:
            break
        await pilot.pause(0.02)
    await pilot.pause()


async def _wait_for_skill_status(screen, pilot, *, attempts: int = 150) -> str:
    status_text = ""
    for _ in range(attempts):
        status_text = str(screen.query_one("#library-skill-save-status").renderable)
        if status_text:
            return status_text
        await pilot.pause(0.02)
    return status_text


_SIMPLE_SKILL_CONTENT = (
    "---\n"
    "description: {description}\n"
    "argument_hint: note id\n"
    "context: fork\n"
    "---\n"
    "# {title}\n"
    "{title} body text.\n"
)


def _skill_content(*, title: str, description: str) -> str:
    return _SIMPLE_SKILL_CONTENT.format(title=title, description=description)


@pytest.mark.asyncio
async def test_open_skill_row_populates_editor_fields_and_save_bumps_version(tmp_path):
    local_service, service = _real_skills_scope_service(tmp_path)
    await local_service.create_skill(
        name="summarize-notes",
        content=_skill_content(title="Summarize", description="Summarize notes"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "summarize-notes")

        assert screen._library_skills_view == "editor"
        assert screen.query_one("#library-skill-name", Input).value == "summarize-notes"
        assert screen.query_one("#library-skill-description", Input).value == "Summarize notes"
        assert screen.query_one("#library-skill-argument-hint", Input).value == "note id"
        assert "fork" in str(screen.query_one("#library-skill-context", Button).label)
        body_area = screen.query_one("#library-skill-body", TextArea)
        assert "Summarize body text." in body_area.text

        body_area.text = "# Summarize\nChanged body text.\n"
        await pilot.pause()
        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._library_skill_dirty is False
        assert screen._library_skill_editor_state.version == 2

        persisted = await local_service.get_skill("summarize-notes")
        assert persisted["version"] == 2
        assert "Changed body" in persisted["content"]


@pytest.mark.asyncio
async def test_skill_editor_shows_shadow_warning_and_does_not_block_save(tmp_path):
    local_service, service = _real_skills_scope_service(tmp_path)
    await local_service.create_skill(
        name="draft-helper",
        content=_skill_content(title="Draft", description="Helps rewrite drafts"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "draft-helper")

        name_input = screen.query_one("#library-skill-name", Input)
        name_input.value = "calculator"
        await pilot.pause()

        # ``allow_untrusted_without_trust_service=True`` reports EVERY skill
        # as trust_status="trusted"/not blocked (LocalSkillsService's own
        # compat default), so ``save_marks_needs_review`` also fires here
        # -- checking the exact shadow copy is a SUBSTRING (not the whole
        # Static's text) keeps this test about the shadow warning only.
        warnings_text = str(screen.query_one("#library-skill-warnings", Static).renderable)
        assert (
            'Name shadows a built-in command/tool ("calculator") — it will not be '
            "invocable as /calculator or as an agent tool."
        ) in warnings_text

        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."


@pytest.mark.asyncio
async def test_saving_a_trusted_skill_warns_and_requeues_needs_review(tmp_path):
    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="reviewer", content=_skill_content(title="Reviewer", description="Reviews a diff"),
    )
    trust_service.bootstrap_trust()

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "reviewer")

        assert screen._library_skill_editor_state.trust_status == "trusted"
        warnings_text = str(screen.query_one("#library-skill-warnings", Static).renderable)
        assert warnings_text == (
            'Saving marks this skill "needs review" — re-approve it in the trust '
            "panel after saving."
        )

        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."

        context = await service.get_context(mode="local")
        blocked_names = [item["name"] for item in context["blocked_skills"]]
        assert "reviewer" in blocked_names
        available_names = [item["name"] for item in context["available_skills"]]
        assert "reviewer" not in available_names


@pytest.mark.asyncio
async def test_trust_panel_review_then_approve_moves_skill_to_available(tmp_path):
    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="approver", content=_skill_content(title="Approver", description="v1"),
    )
    trust_service.bootstrap_trust()
    # An edit landing after the trusted baseline re-quarantines the skill --
    # simulates "someone already saved a change" so the editor opens on an
    # ALREADY needs-review skill (the brief's exact scenario shape).
    await local_service.update_skill(
        "approver", content=_skill_content(title="Approver", description="v2"),
    )

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "approver")

        assert screen._library_skill_editor_state.trust_blocked is True
        review_files_before = str(
            screen.query_one("#library-skill-trust-review-files", Static).renderable
        )
        assert review_files_before == ""

        screen.query_one("#library-skill-trust-review", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skill_active_review is not None:
                break
            await pilot.pause(0.02)

        review_files_after = str(
            screen.query_one("#library-skill-trust-review-files", Static).renderable
        )
        assert "SKILL.md" in review_files_after

        pilot.app.push_screen_wait = AsyncMock(return_value="trust-passphrase")
        screen.query_one("#library-skill-trust-approve", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skill_active_review is None:
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert screen._library_skill_editor_state.trust_status == "trusted"
        context = await service.get_context(mode="local")
        available_names = [item["name"] for item in context["available_skills"]]
        assert "approver" in available_names
        blocked_names = [item["name"] for item in context["blocked_skills"]]
        assert "approver" not in blocked_names


@pytest.mark.asyncio
async def test_delete_skill_returns_to_list_and_decrements_count(tmp_path):
    local_service, service = _real_skills_scope_service(tmp_path)
    await local_service.create_skill(
        name="throwaway", content=_skill_content(title="Temp", description="temp"),
    )
    await local_service.create_skill(
        name="keeper", content=_skill_content(title="Keep", description="keep"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "throwaway")

        screen.query_one("#library-skill-delete", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skills_view == "list":
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert screen._library_skills_view == "list"
        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-skills").label)
            if "(1)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(1)" in rail_label
        assert len(screen.query("#library-skill-row-throwaway")) == 0
        assert screen.query_one("#library-skill-row-keeper", Button)


@pytest.mark.asyncio
async def test_flush_pending_work_vetoes_dirty_skill_editor(tmp_path):
    local_service, service = _real_skills_scope_service(tmp_path)
    await local_service.create_skill(
        name="dirty-check", content=_skill_content(title="D", description="d"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "dirty-check")

        screen.query_one("#library-skill-description", Input).value = "Changed mid switch"
        await pilot.pause()
        assert screen._library_skill_dirty is True

        allowed = await screen.flush_pending_work()

        assert allowed is False
        assert screen._library_skill_dirty is True


@pytest.mark.asyncio
async def test_skill_editor_opens_under_real_runtime_policy_enforcer(tmp_path):
    """Regression test for the Phase-1 gate defect class (see
    ``test_library_prompt_row_opens_editor_under_real_runtime_policy_enforcer``
    in ``Tests/UI/test_library_prompts_canvas.py``): wires the REAL
    production runtime-policy seam (``ServicePolicyEnforcer`` bound to the
    real ``CAPABILITY_REGISTRY``) rather than leaving ``policy_enforcer``
    unset, so a missing ``skills.detail.local``/``skills.create.local``/
    ``skills.update.local``/``skills.delete.local`` registry row would be
    caught here instead of silently swallowed by
    ``_refresh_library_skill_detail``'s bare ``except Exception``.
    """
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    local_service, service = _real_skills_scope_service(tmp_path, policy_enforcer=policy_enforcer)
    await local_service.create_skill(
        name="policy-check", content=_skill_content(title="P", description="p"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "policy-check")

        assert screen._library_skills_view == "editor"
        assert screen._library_skill_detail is not None
        assert screen.query_one("#library-skill-name", Input).value == "policy-check"

        # Save/Delete also route through the same enforcer -- exercise both
        # so a missing skills.update.local/skills.delete.local row would
        # fail loudly here too, not just on open.
        screen.query_one("#library-skill-description", Input).value = "policy ok"
        await pilot.pause()
        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."

        screen.query_one("#library-skill-delete", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skills_view == "list":
                break
            await pilot.pause(0.02)
        assert screen._library_skills_view == "list"
