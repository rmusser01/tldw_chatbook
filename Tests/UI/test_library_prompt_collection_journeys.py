"""Keyboard journeys through local Prompt collections with production CSS."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.test_library_prompt_action_journeys import _activate_more_action
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
)


async def _collection_host(tmp_path, theme):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="Collection journey",
        author="",
        details="",
        user_prompt="Saved reusable message.",
    )
    collection = await service.create_prompt_collection(
        mode="local", name="[bold] café"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return db, service, prompt_id, collection["collection_id"], host


async def _manager(host, screen, pilot):
    await _wait_for_condition(
        pilot,
        lambda: (
            host.screen is not screen
            and host.screen._catalog.status in {"ready", "empty"}
            and host.screen.focused
            is host.screen.query_one("#prompt-collection-manager-search")
        ),
        message=lambda: (
            f"Manager screen={host.screen!r}; catalog={getattr(host.screen, '_catalog', None)!r}; focus={host.screen.focused!r}"
        ),
    )
    return host.screen


async def _focus(screen, host, pilot, selector, label):
    await _wait_for_condition(
        pilot,
        lambda: (
            screen.focused is screen.query_one(selector)
            and label in _painted_text(host, screen.focused.region)
        ),
        message=lambda: (
            f"Expected readable {selector}; focus={screen.focused!r}; region={screen.query_one(selector).region}; paint={_painted_text(host, screen.query_one(selector).region)!r}"
        ),
    )


async def _tab_to(screen, host, pilot, selector, label):
    for _ in range(12):
        if screen.focused is screen.query_one(selector):
            break
        await pilot.press("tab")
    await _focus(screen, host, pilot, selector, label)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_more_collections_done_apply_and_cancel_preserve_draft(
    tmp_path, monkeypatch, size, theme
):
    db, service, prompt_id, collection_id, host = await _collection_host(
        tmp_path, theme
    )
    original = db.fetch_prompt_details(prompt_id)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        user = screen.query_one("#library-prompt-user", TextArea)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.can_manage
            ),
            message="Membership controls did not become ready",
        )
        fields = tuple(screen.query(TextArea))
        await _activate_more_action(screen, pilot, "more-collections")
        modal = await _manager(host, screen, pilot)
        await _tab_to(
            modal,
            host,
            pilot,
            f"#prompt-collection-manager-member-{collection_id}",
            "[bold] café",
        )
        await pilot.press("space")
        await _tab_to(modal, host, pilot, "#prompt-collection-manager-done", "Done")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                host.screen is screen
                and screen._library_prompt_collections_controller.membership_state.can_apply
            ),
            message="Done did not stage memberships",
        )
        assert screen._prompts_state.editor_mode == "info"
        assert not screen.query_one("#library-prompt-more-actions-region").display
        await _focus(
            screen,
            host,
            pilot,
            "#library-prompt-memberships-manage",
            "Manage collections",
        )
        screen.query_one(
            "#library-prompt-name", Input
        ).value = "Unsaved collection journey"
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Draft change not captured",
        )
        await pilot.press("tab")
        await _focus(screen, host, pilot, "#library-prompt-memberships-apply", "Apply")
        assert (
            await service.list_prompt_collection_memberships(
                mode="local", prompt_id=prompt_id
            )
        )["collection_ids"] == ()
        replace_memberships = service.replace_prompt_collection_memberships
        calls = 0

        async def fail_once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected membership storage failure")
            return await replace_memberships(**kwargs)

        monkeypatch.setattr(service, "replace_prompt_collection_memberships", fail_once)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.status
                == "apply_error"
            ),
            message="Membership failure did not offer retry",
        )
        await _focus(screen, host, pilot, "#library-prompt-memberships-apply", "Apply")
        assert (
            await service.list_prompt_collection_memberships(
                mode="local", prompt_id=prompt_id
            )
        )["collection_ids"] == ()
        await _wait_for_condition(
            pilot,
            lambda: not screen.focused.has_class("-active"),
            message="Apply press still active",
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_collections_controller.membership_state.status
                == "success"
            ),
            message="Membership Apply did not complete",
        )
        assert (
            await service.list_prompt_collection_memberships(
                mode="local", prompt_id=prompt_id
            )
        )["collection_ids"] == (collection_id,)
        screen.query_one("#library-prompt-memberships-manage", Button).focus()
        await pilot.press("enter")
        modal = await _manager(host, screen, pilot)
        await _tab_to(
            modal,
            host,
            pilot,
            f"#prompt-collection-manager-member-{collection_id}",
            "[bold] café",
        )
        await pilot.press("space", "escape")
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, message="Cancel did not close"
        )
        await _focus(
            screen,
            host,
            pilot,
            "#library-prompt-memberships-manage",
            "Manage collections",
        )
        assert (
            screen._library_prompt_collections_controller.membership_state.staged_ids
            == (collection_id,)
        )
        assert db.fetch_prompt_details(prompt_id) == original
        assert user.text == "Saved reusable message."
        assert (
            screen.query_one("#library-prompt-name", Input).value
            == "Unsaved collection journey"
        )
        assert screen._prompts_state.dirty
        assert all(field.is_attached for field in fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme", [((170, 48), "textual-dark"), ((80, 24), "textual-light")]
)
async def test_catalog_selection_and_name_collision_preserve_entered_name(
    tmp_path, size, theme
):
    _, _, _, collection_id, host = await _collection_host(tmp_path, theme)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_browse_controller.result.status == "ready"
                and bool(screen.query("#library-prompts-collection"))
            ),
            message="Prompt list",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-prompts-filter"),
            message="List entry focus not settled",
        )
        screen.query_one("#library-prompts-collection", Button).focus()
        await _focus(screen, host, pilot, "#library-prompts-collection", "collection:")
        await pilot.press("enter")
        modal = await _manager(host, screen, pilot)
        modal.query_one(
            "#prompt-collection-manager-new-name", Input
        ).value = "[bold] CAFÉ"
        modal.query_one(
            "#prompt-collection-manager-search", Input
        ).value = "pending filter"
        await _tab_to(
            modal,
            host,
            pilot,
            f"#prompt-collection-manager-row-{collection_id}",
            "[bold] café",
        )
        await pilot.press("enter")
        await _focus(
            modal,
            host,
            pilot,
            f"#prompt-collection-manager-row-{collection_id}",
            "[bold] café",
        )
        assert (
            modal.query_one("#prompt-collection-manager-new-name", Input).value
            == "[bold] CAFÉ"
        )
        assert (
            modal.query_one("#prompt-collection-manager-search", Input).value
            == "pending filter"
        )
        assert modal._catalog.query == ""
        await _tab_to(
            modal, host, pilot, "#prompt-collection-manager-create", "New collection"
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: modal._outcome == "Name already exists — choose another.",
            message="Name collision not reported",
        )
        await _focus(
            modal, host, pilot, "#prompt-collection-manager-new-name", "[bold] CAFÉ"
        )
        assert (
            modal.query_one("#prompt-collection-manager-new-name", Input).value
            == "[bold] CAFÉ"
        )
        assert not modal.query_one("#prompt-collection-manager-retry").display
        modal.query_one("#prompt-collection-manager-search", Input).focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: modal._catalog.status == "empty",
            message="Search did not settle",
        )
        assert (
            modal.query_one("#prompt-collection-manager-new-name", Input).value
            == "[bold] CAFÉ"
        )
        await pilot.press("escape")
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, message="Cancel did not close catalog"
        )
        assert screen._library_prompt_browse_controller.scope.collection_id is None
