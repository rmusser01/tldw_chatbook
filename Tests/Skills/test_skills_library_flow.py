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

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_CREATE_SKILL
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
    _wait_for_selector,
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


def _real_uninitialized_trust_service(tmp_path) -> SkillTrustService:
    """A real trust service with NO manifest and NO in-memory keys yet --
    the true first-run state (``trust_uninitialized``), unlike
    ``_real_trust_service`` above, which is already unlocked. This is the
    exact fresh-install shape the Phase-1 gate flagged as having no live-UI
    bootstrap path (FIX 2)."""
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )


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
async def test_renaming_an_existing_skill_is_refused_and_never_corrupts_it(tmp_path):
    """Fix wave for the review Critical: renaming an existing skill used to
    silently corrupt it. ``_save_library_skill`` baked the LIVE Name-input
    value into the frontmatter but always wrote under the ORIGINAL
    directory name (``update_skill`` has no rename primitive), so the
    service's own validation (``name`` must match the parent directory
    name) marked the persisted record ``validation_status: "invalid"``
    while the editor still reported "Saved." -- an unusable skill
    (``skills_screen`` gates on ``validation_status == "valid"``) with no
    visible error.

    The fix makes the Name Input disabled for an existing skill (there is
    no rename primitive to build a real rename feature on) AND, belt and
    braces, forces the persisted frontmatter name to stay pinned to the
    skill's own directory name no matter what the Input reports. This test
    proves the belt half directly: ``disabled`` only blocks focus/keyboard
    editing, not a programmatic attribute write, so setting ``.value``
    still lets this simulate a divergent Name -- and the save must still
    persist under the original name with a VALID record. The forced value
    ("calculator") is also a shadowed builtin name, so this keeps the
    shadow-warning line's coverage too (this test replaces the former
    rename-based shadow-warning scenario, which is no longer a real user
    flow now that renaming is refused).
    """
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
        assert name_input.disabled is True

        # Force a divergent value directly -- ``disabled`` blocks
        # keyboard/focus editing, not a programmatic attribute write, so
        # this simulates "the Input's value somehow diverges" without
        # relying on a real rename UI (there is none).
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

        # The core corruption check: "Saved." must never be reported
        # alongside an invalid persisted record, and the skill must still
        # live under its ORIGINAL name -- never renamed to the diverged
        # (disabled-input) value.
        persisted = await local_service.get_skill("draft-helper")
        assert persisted["name"] == "draft-helper"
        assert persisted["validation_status"] == "valid"
        assert screen._library_skill_editor_state.name == "draft-helper"

        with pytest.raises(Exception):
            await local_service.get_skill("calculator")


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
async def test_skill_editor_canvas_scrolls_trust_panel_into_view(tmp_path):
    """Gate fix wave FIX 1: at the recipe's default terminal size the
    editor's lower content (Trust panel, Save/Delete) sits below the fold.
    Before the fix, ``LibrarySkillsListCanvas`` was a plain ``Vertical``
    (clips overflow, no scrollbar, no mouse-wheel/keyboard scroll); the fix
    makes it a real ``VerticalScroll`` (the same house pattern already used
    by ``LibraryExportCanvas``/``LibraryIngestCanvas``). Mirrors
    ``test_personas_dictionaries.py``'s AC5b Entries-tab scroll geometry
    test: geometry + a real keyboard scroll + a real focus-jump, not a
    hand-simulated mouse event.
    """
    from textual.containers import VerticalScroll

    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="scroll-check", content=_skill_content(title="Scroll", description="v1"),
    )
    trust_service.bootstrap_trust()
    # Re-quarantine after bootstrap so the Trust panel's "Review changes"
    # button is enabled (focusable) -- a disabled Button cannot take focus,
    # so the focus-jump-into-view assertion below needs a real target.
    await local_service.update_skill(
        "scroll-check", content=_skill_content(title="Scroll", description="v2"),
    )

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "scroll-check")

        canvas = screen.query_one("#library-skills-canvas", VerticalScroll)
        # Structural proof this is a real scrolling container (the fix),
        # not a clipping plain Vertical -- and that mouse-wheel scroll is
        # actually enabled (the same flag Textual's own mouse-wheel handler
        # checks before scrolling at all).
        assert isinstance(canvas, VerticalScroll)
        assert canvas.allow_vertical_scroll is True
        # Prove there's genuinely content below the fold to scroll to --
        # otherwise the rest of this test would pass vacuously.
        assert canvas.max_scroll_y > 0
        assert canvas.scroll_offset.y == 0

        # Keyboard scrolling: focus the canvas itself and page down, the
        # same key binding a real terminal user gets for free from
        # ``VerticalScroll``.
        canvas.focus()
        await pilot.pause()
        await pilot.press("pagedown")
        await pilot.pause()
        assert canvas.scroll_offset.y > 0

        # Reset, then prove a focus JUMP (e.g. tabbing into the Trust
        # panel) auto-scrolls it into view -- not just a manual scroll.
        canvas.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert canvas.scroll_offset.y == 0

        review_button = screen.query_one("#library-skill-trust-review", Button)
        assert review_button.disabled is False
        review_button.focus()
        await pilot.pause()
        await pilot.pause()

        assert canvas.scroll_offset.y > 0
        canvas_region = canvas.region
        button_region = review_button.region
        assert canvas_region.y <= button_region.y < canvas_region.y + canvas_region.height


@pytest.mark.asyncio
async def test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_approve_flow(tmp_path):
    """Gate fix wave FIX 2: a brand-new (never-bootstrapped) trust store had
    no live-UI path to create the passphrase at all -- the Library editor's
    Unlock only ever unlocked an EXISTING manifest. This proves the fix end
    to end: the Trust panel shows the "Set up skill trust" state instead of
    a dead Unlock/Review/Approve row, driving the real ``bootstrap_trust``
    primitive through a confirm-passphrase modal genuinely initializes the
    on-disk trust store, the panel refreshes into the normal flow, and a
    SECOND skill can then be reviewed/approved end to end through that now-
    initialized store."""
    trust_service = _real_uninitialized_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="onboarding-check", content=_skill_content(title="Onboard", description="v1"),
    )

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "onboarding-check")

        assert screen._library_skill_editor_state.trust_status == "trust_uninitialized"
        assert not trust_service.trust_store.has_manifest()
        assert str(
            screen.query_one("#library-skill-trust-state", Static).renderable
        ) == "Trust: not initialized"
        assert screen.query_one("#library-skill-trust-setup", Button)
        # The normal Unlock/Review/Approve row must NOT render while
        # uninitialized -- there is nothing yet to unlock or review.
        assert len(screen.query("#library-skill-trust-unlock")) == 0
        assert len(screen.query("#library-skill-trust-review")) == 0
        assert len(screen.query("#library-skill-trust-approve")) == 0

        pilot.app.push_screen_wait = AsyncMock(return_value="fresh-passphrase")
        screen.query_one("#library-skill-trust-setup", Button).press()
        await pilot.pause()
        for _ in range(150):
            state = screen._library_skill_editor_state
            if state is not None and state.trust_status != "trust_uninitialized":
                break
            await pilot.pause(0.02)
        await pilot.pause()

        # Bootstrapping trusts every currently on-disk skill as the initial
        # baseline -- the just-opened skill becomes trusted, the real
        # on-disk store is genuinely initialized, and the panel has
        # switched to the normal (non-setup) layout.
        assert screen._library_skill_editor_state.trust_status == "trusted"
        assert trust_service.trust_store.has_manifest()
        assert len(screen.query("#library-skill-trust-setup")) == 0
        assert screen.query_one("#library-skill-trust-unlock", Button)
        assert screen.query_one("#library-skill-trust-review", Button)
        assert screen.query_one("#library-skill-trust-approve", Button)

        # Prove the store is genuinely usable now, not just flagged
        # "trusted" cosmetically: edit the skill (re-quarantines it under
        # the freshly-created manifest), then run a full review/approve
        # cycle through the now-normal panel.
        await local_service.update_skill(
            "onboarding-check",
            content=_skill_content(title="Onboard", description="v2"),
        )
        screen.query_one("#library-skill-back", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skills_view == "list":
                break
            await pilot.pause(0.02)
        for _ in range(150):
            if screen.query("#library-skill-row-onboarding-check"):
                break
            await pilot.pause(0.02)
        screen.query_one("#library-skill-row-onboarding-check", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skill_detail is not None:
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert screen._library_skill_editor_state.trust_blocked is True
        screen.query_one("#library-skill-trust-review", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_skill_active_review is not None:
                break
            await pilot.pause(0.02)

        pilot.app.push_screen_wait = AsyncMock(return_value="fresh-passphrase")
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
        assert "onboarding-check" in available_names


@pytest.mark.asyncio
async def test_already_bootstrapped_store_never_shows_setup_state(tmp_path):
    """Once trust has been bootstrapped (by any means), the Trust panel must
    render its NORMAL Unlock/Review/Approve row, never the first-run "Set up
    skill trust" state -- even for a freshly needs-review (blocked) skill."""
    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="already-bootstrapped", content=_skill_content(title="A", description="v1"),
    )
    trust_service.bootstrap_trust()
    await local_service.update_skill(
        "already-bootstrapped", content=_skill_content(title="A", description="v2"),
    )

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skill_editor(screen, pilot, "already-bootstrapped")

        assert screen._library_skill_editor_state.trust_status != "trust_uninitialized"
        assert len(screen.query("#library-skill-trust-setup")) == 0
        assert screen.query_one("#library-skill-trust-unlock", Button)
        assert screen.query_one("#library-skill-trust-review", Button)
        assert screen.query_one("#library-skill-trust-approve", Button)


@pytest.mark.asyncio
async def test_uninitialized_trust_store_list_still_shows_needs_review_glyph(tmp_path):
    """A never-bootstrapped trust store must degrade gracefully everywhere
    else in the Library UI, not just in the editor's Trust panel -- the
    list view still shows the skill as needs-review (``⚠``), same as any
    other trust-blocked skill, and the rail count still includes it."""
    trust_service = _real_uninitialized_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    await local_service.create_skill(
        name="pre-bootstrap", content=_skill_content(title="P", description="p"),
    )

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause()
        await pilot.pause()

        row = screen.query_one("#library-skill-row-pre-bootstrap", Button)
        assert row.has_class("library-skill-row-blocked")
        assert "⚠" in str(row.label)
        rail_label = str(screen.query_one("#library-row-browse-skills").label)
        assert "(1)" in rail_label


@pytest.mark.asyncio
async def test_skill_trust_bootstrap_modal_rejects_mismatched_confirmation():
    """The bootstrap modal creates a BRAND-NEW passphrase (unlike every
    other trust action, which unlocks an existing one) -- it must ask twice
    and refuse to dismiss on a mismatch instead of silently proceeding with
    a possibly-mistyped passphrase nobody could recover."""
    from textual import work
    from textual.app import App
    from tldw_chatbook.UI.Screens.skills_screen import SkillTrustBootstrapModal

    class _ModalHost(App):
        def __init__(self) -> None:
            super().__init__()
            self.result: str | None = "unset"

        def on_mount(self) -> None:
            self._await_modal()

        @work
        async def _await_modal(self) -> None:
            self.result = await self.push_screen_wait(SkillTrustBootstrapModal())

    app = _ModalHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#skill-trust-bootstrap-input", Input).value = "correct-horse"
        modal.query_one("#skill-trust-bootstrap-confirm-input", Input).value = "mismatched"
        await pilot.pause()
        modal.query_one("#skill-trust-bootstrap-submit", Button).press()
        await pilot.pause()

        # Rejected: the modal must still be on screen with an inline error
        # -- never dismissed with an unconfirmed passphrase.
        assert app.screen is modal
        error_text = str(modal.query_one("#skill-trust-bootstrap-error", Static).renderable)
        assert "match" in error_text.lower()
        assert app.result == "unset"

        modal.query_one("#skill-trust-bootstrap-confirm-input", Input).value = "correct-horse"
        await pilot.pause()
        modal.query_one("#skill-trust-bootstrap-submit", Button).press()
        await pilot.pause()

        assert app.result == "correct-horse"


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


# ---------------------------------------------------------------------------
# Create rail ("New skill") -- skills-200 spec's named-but-previously-
# unscheduled entry point. Mirrors ``Tests/UI/test_library_prompts_canvas.py``'s
# D1 create-row tests exactly (blank editor / save-creates / invalid-name),
# plus a trust-specific case: a brand-new skill's ``create_skill`` call never
# passes ``trust_approved=True``, so it must arrive needs-review.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_shell_create_skill_row_opens_blank_editor(tmp_path):
    """The Create rail's "New skill" row opens the in-canvas editor on a
    blank, not-yet-saved record -- empty fields, Name Input editable (no
    rename hint), ``_selected_skill_name`` empty. Mirrors
    ``test_library_shell_create_prompt_row_opens_blank_editor``."""
    local_service, service = _real_skills_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_SKILL}").press()
        await _wait_for_selector(screen, pilot, "#library-skill-name")

        assert screen._library_skills_view == "editor"
        assert screen._selected_skill_name == ""
        name_input = screen.query_one("#library-skill-name", Input)
        assert name_input.value == ""
        assert name_input.disabled is False
        assert len(screen.query("#library-skill-name-hint")) == 0
        assert screen.query_one("#library-skill-description", Input).value == ""
        assert screen.query_one("#library-skill-argument-hint", Input).value == ""
        assert screen.query_one("#library-skill-allowed-tools", Input).value == ""
        assert screen.query_one("#library-skill-body", TextArea).text == ""


@pytest.mark.asyncio
async def test_library_shell_create_skill_save_creates_and_increments_count(tmp_path):
    """Save with a fresh valid name CREATES via the scope service's create
    path (not update) -- the Skills rail count increments, the record is
    real (fetchable via the service), and the editor adopts the new name.
    Mirrors ``test_library_shell_create_prompt_save_creates_and_increments_count``."""
    local_service, service = _real_skills_scope_service(tmp_path)
    await local_service.create_skill(
        name="existing-one", content=_skill_content(title="Existing", description="d"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_SKILL}").press()
        await _wait_for_selector(screen, pilot, "#library-skill-name")

        screen.query_one("#library-skill-name", Input).value = "brand-new-skill"
        await pilot.pause()
        screen.query_one("#library-skill-description", Input).value = "A brand new skill"
        await pilot.pause()
        screen.query_one("#library-skill-body", TextArea).text = "# Brand new\nDo the thing.\n"
        await pilot.pause()
        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._selected_skill_name == "brand-new-skill"

        persisted = await local_service.get_skill("brand-new-skill")
        assert persisted["name"] == "brand-new-skill"
        assert persisted["description"] == "A brand new skill"
        assert "Do the thing." in persisted["content"]

        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-skills").label)
            if "(2)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(2)" in rail_label


@pytest.mark.asyncio
async def test_library_shell_create_skill_save_invalid_name_shows_classify_outcome(tmp_path):
    """D1's three save outcomes apply to skills too -- an invalid-shaped
    name shows the same ``classify_skill_save_error`` outcome the update
    path already shows, and creates nothing."""
    local_service, service = _real_skills_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_SKILL}").press()
        await _wait_for_selector(screen, pilot, "#library-skill-name")

        screen.query_one("#library-skill-name", Input).value = "Not A Valid Name"
        await pilot.pause()
        screen.query_one("#library-skill-body", TextArea).text = "# Body\nHi.\n"
        await pilot.pause()
        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Skill name must use lowercase letters, numbers, and hyphens."
        assert screen._selected_skill_name == ""

        context = await service.get_context(mode="local")
        assert context["available_skills"] == []
        assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_library_shell_create_skill_save_arrives_needs_review_with_panel_primed(tmp_path):
    """A brand-new skill created via the "New skill" row arrives
    trust-pending -- ``create_skill``'s default never passes
    ``trust_approved=True`` -- so the trust panel must reflect that
    immediately after Save, with no second editor open needed."""
    trust_service = _real_trust_service(tmp_path)
    # Bootstrap BEFORE any skill exists on disk: an empty trusted baseline,
    # so the skill this test creates afterward is unambiguously "added
    # since the baseline" (``quarantined_added``), not merely unreviewed.
    trust_service.bootstrap_trust()
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_SKILL}").press()
        await _wait_for_selector(screen, pilot, "#library-skill-name")

        screen.query_one("#library-skill-name", Input).value = "fresh-skill"
        await pilot.pause()
        screen.query_one("#library-skill-description", Input).value = "Fresh"
        await pilot.pause()
        screen.query_one("#library-skill-body", TextArea).text = "# Fresh\nDo it.\n"
        await pilot.pause()
        screen.query_one("#library-skill-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_skill_status(screen, pilot)
        assert status_text == "Saved."

        assert screen._library_skill_editor_state.trust_status == "quarantined_added"
        assert screen._library_skill_editor_state.trust_blocked is True
        trust_state_text = str(screen.query_one("#library-skill-trust-state", Static).renderable)
        assert trust_state_text == "Trust: new untrusted file (SKILL.md)"
        review_button = screen.query_one("#library-skill-trust-review", Button)
        assert review_button.disabled is False

        context = await service.get_context(mode="local")
        blocked_names = [item["name"] for item in context["blocked_skills"]]
        assert "fresh-skill" in blocked_names
        available_names = [item["name"] for item in context["available_skills"]]
        assert "fresh-skill" not in available_names
