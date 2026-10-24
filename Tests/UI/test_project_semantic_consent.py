"""Mounted, per-folder semantic consent remains separate from keyword choices."""

import asyncio
import sys

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_workspace_project_index_settings import (
    CardApp,
    card_module,
    wait_for_card,
)
from Tests.Workspaces.test_project_index_service import Rig


@private_profile_test
async def test_mounted_semantic_controls_and_status_do_not_load_model(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    before = {name for name in sys.modules if name.startswith("sentence_transformers")}
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(80, 24)) as pilot:
            await wait_for_card(card, pilot)
            assert card.query_one(".project-semantic-model", Input).value
            assert "Semantic index: Off" in str(
                card.query_one(".project-semantic-status", Static).render()
            )
            assert {button.name for button in card.query(Button)} >= {
                "semantic-stage",
                "semantic-approve",
                "semantic-refresh",
                "semantic-disable",
            }
            assert {
                name for name in sys.modules if name.startswith("sentence_transformers")
            } == before
    finally:
        await rig.close()


@private_profile_test
async def test_stage_without_disclosure_preserves_keyword_and_cancel_makes_no_provider_call(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    before_keyword = rig.service.status(rig.key)
    from tldw_chatbook.Workspaces.project_index import semantic

    def forbidden_provider(*_args, **_kwargs):
        raise AssertionError("cancelled_disclosure_reached_provider")

    monkeypatch.setattr(semantic, "_issue_project_projection_scope", forbidden_provider)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            card.query_one(".project-semantic-model", Input).value = "openai-3-small"
            next(b for b in card.query(Button) if b.name == "semantic-stage").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            state = rig.service.semantic_state(rig.key)
            assert state.enabled and state.state == "Paused"
            assert state.reason == "destination_approval_required"
            assert "https://api.openai.com/v1/embeddings" in str(
                card.query_one(".project-semantic-destination", Static).render()
            )
            assert rig.service.status(rig.key) == before_keyword
            approve = next(
                b for b in card.query(Button) if b.name == "semantic-approve"
            )
            approve.press()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert rig.service.semantic_state(rig.key) == state
            assert rig.service.status(rig.key) == before_keyword
    finally:
        await rig.close()


@private_profile_test
async def test_catalog_destination_change_before_disclosure_refuses_old_staged_approval(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Workspaces.project_index import semantic

    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    keyword = rig.service.status(rig.key)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            card.query_one(".project-semantic-model", Input).value = "openai-3-small"
            next(b for b in card.query(Button) if b.name == "semantic-stage").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            staged = rig.service.semantic_state(rig.key)
            original = semantic._resolved_spec

            def changed(model_id):
                spec, destination = original(model_id)
                if model_id == "openai-3-small":
                    return spec, "https://changed.example/v1/embeddings"
                return spec, destination

            monkeypatch.setattr(semantic, "_resolved_spec", changed)
            next(b for b in card.query(Button) if b.name == "semantic-approve").press()
            await pilot.pause()
            assert not list(card.app.screen.query("#project-semantic-confirm"))
            assert rig.service.semantic_state(rig.key) == staged
            assert rig.service.status(rig.key) == keyword
    finally:
        await rig.close()


@private_profile_test
async def test_stale_semantic_action_after_clear_cannot_enable_new_epoch(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            action = card.SemanticActionRequested(
                card.binding_id,
                card.workspace_id,
                card.status,
                card.semantic_status,
                "stage",
                "all-MiniLM-L6-v2",
            )
            rig.service.clear(rig.key, expected_scope=card.status.scope)
            with pytest.raises(ValueError, match="stale_scope"):
                card._execute_semantic(card.app, action)
            assert not rig.service.semantic_state(rig.key).enabled
    finally:
        await rig.close()


@private_profile_test
async def test_stale_disable_after_clear_and_reenable_cannot_revoke_new_epoch(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            selected, _ = card._inspect_semantic_destination("all-MiniLM-L6-v2")
            semantic = rig.service.enable_semantic(
                rig.key,
                selected,
                expected_epoch=card.status.scope.epoch,
                expected_semantic_revision=card.semantic_status.semantic_revision,
            )
            card.reload_status()
            await wait_for_card(card, pilot)
            stale = card.SemanticActionRequested(
                card.binding_id,
                card.workspace_id,
                card.status,
                semantic,
                "disable",
                "all-MiniLM-L6-v2",
            )
            rig.service.clear(rig.key, expected_scope=card.status.scope)
            new_keyword = rig.enable()
            newer = rig.service.enable_semantic(
                rig.key,
                selected,
                expected_epoch=new_keyword.scope.epoch,
                expected_semantic_revision=rig.service.semantic_state(
                    rig.key
                ).semantic_revision,
            )
            with pytest.raises(ValueError, match="stale_scope|stale_semantic_revision"):
                card._execute_semantic(card.app, stale)
            assert rig.service.semantic_state(rig.key) == newer
    finally:
        await rig.close()


@private_profile_test
async def test_approval_then_destination_change_pauses_and_preserves_keyword(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    before_keyword = rig.service.status(rig.key)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            model = card.query_one(".project-semantic-model", Input)
            model.value = "openai-3-small"
            next(b for b in card.query(Button) if b.name == "semantic-stage").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            next(b for b in card.query(Button) if b.name == "semantic-approve").press()
            await pilot.pause()
            assert any(
                "https://api.openai.com/v1/embeddings" in str(row.render())
                for row in card.app.screen.query(Static)
            )
            card.app.screen.query_one("#project-semantic-confirm", Button).press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            approved = rig.service.semantic_state(rig.key)
            assert approved.enabled and approved.settings
            assert (
                approved.settings.approved_destination_fingerprint
                == approved.settings.destination_fingerprint
            )
            model.value = "all-MiniLM-L6-v2"
            next(b for b in card.query(Button) if b.name == "semantic-stage").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            paused = rig.service.semantic_state(rig.key)
            assert (
                paused.state == "Paused"
                and paused.reason == "destination_approval_required"
            )
            assert (
                paused.settings
                and paused.settings.approved_destination_fingerprint is None
            )
            assert rig.service.status(rig.key) == before_keyword
    finally:
        await rig.close()


@private_profile_test
async def test_semantic_action_error_preserves_keyword_and_dirty_drafts(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    before_keyword = rig.service.status(rig.key)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            card.query_one(".project-index-prefixes", Input).value = "dirty-folder"
            card.query_one(".project-semantic-model", Input).value = "openai-3-small"
            monkeypatch.setattr(
                rig.service,
                "enable_semantic",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    ValueError("provider_failed")
                ),
            )
            next(b for b in card.query(Button) if b.name == "semantic-stage").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            assert card.status == before_keyword
            assert (
                card.query_one(".project-index-prefixes", Input).value == "dirty-folder"
            )
            assert (
                card.query_one(".project-semantic-model", Input).value
                == "openai-3-small"
            )
            assert rig.service.status(rig.key) == before_keyword
    finally:
        await rig.close()


@private_profile_test
async def test_pending_cleanup_retries_the_captured_owner_after_app_admission_closes(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven,
    )

    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    closed = False
    pending = True
    retried = []

    def get_service(app):
        if closed:
            raise RuntimeError("project_index_closed")
        return rig.service

    def failed_disable(*args, **kwargs):
        raise ProjectIndexRetirementUnproven("project_embedding_close_pending")

    def retry(key):
        nonlocal pending
        retried.append(key)
        pending = False
        return True

    monkeypatch.setattr(mod, "get_project_index_service", get_service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    monkeypatch.setattr(rig.service, "disable_semantic", failed_disable)
    monkeypatch.setattr(rig.service, "semantic_close_pending", lambda key: pending)
    monkeypatch.setattr(rig.service, "retry_semantic_close", retry)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            candidate = card.SemanticActionRequested(
                card.binding_id,
                card.workspace_id,
                card.status,
                card.semantic_status,
                "disable",
                "all-MiniLM-L6-v2",
            )
            card.post_message(candidate)
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert "cleanup pending" in str(
                card.query_one(".project-semantic-status", Static).render()
            )
            closed = True
            rig.service._closed = True
            next(
                b for b in card.query(Button) if b.name == "semantic-retry-cleanup"
            ).press()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert retried == [rig.key]
            assert "cleanup pending" not in str(
                card.query_one(".project-semantic-status", Static).render()
            )
    finally:
        rig.service._closed = False
        await rig.close()


@private_profile_test
async def test_cleanup_retry_never_displays_revoked_old_semantic_state(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven,
    )

    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    actual_disable = rig.service.disable_semantic
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            selected, _ = card._inspect_semantic_destination("all-MiniLM-L6-v2")
            rig.service.enable_semantic(
                rig.key,
                selected,
                expected_epoch=card.status.scope.epoch,
                expected_semantic_revision=card.semantic_status.semantic_revision,
            )
            card.reload_status()
            await wait_for_card(card, pilot)
            assert card.semantic_status.enabled

            def revoked_then_pending(*args, **kwargs):
                result = actual_disable(*args, **kwargs)
                assert result.state == "Off" and not result.enabled
                raise ProjectIndexRetirementUnproven("project_embedding_close_pending")

            monkeypatch.setattr(rig.service, "disable_semantic", revoked_then_pending)
            monkeypatch.setattr(rig.service, "semantic_close_pending", lambda key: True)
            next(b for b in card.query(Button) if b.name == "semantic-disable").press()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert not rig.service.semantic_state(rig.key).enabled
            assert "cleanup pending" in str(
                card.query_one(".project-semantic-status", Static).render()
            )
            monkeypatch.setattr(
                rig.service, "semantic_close_pending", lambda key: False
            )
            monkeypatch.setattr(rig.service, "retry_semantic_close", lambda key: True)
            next(
                b for b in card.query(Button) if b.name == "semantic-retry-cleanup"
            ).press()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            semantic_text = str(
                card.query_one(".project-semantic-status", Static).render()
            )
            assert "Semantic index: Off" in semantic_text
            assert "cleanup pending" not in semantic_text
            assert not next(
                b for b in card.query(Button) if b.name == "semantic-disable"
            ).display
    finally:
        await rig.close()


@private_profile_test
async def test_semantic_read_error_preserves_last_known_state_and_retry_guidance(
    tmp_path, request, monkeypatch
):
    mod = card_module()
    rig = Rig(tmp_path)
    rig.enable()
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    try:
        async with CardApp(card).run_test(size=(100, 35)) as pilot:
            await wait_for_card(card, pilot)
            known = card.semantic_status
            keyword = card.status
            card.query_one(".project-index-prefixes", Input).value = "dirty-folder"
            actual_state = rig.service.semantic_state

            def failed_read(key):
                raise RuntimeError("semantic_read_failed")

            monkeypatch.setattr(rig.service, "semantic_state", failed_read)
            card.reload_status()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert card.semantic_status == known
            assert card.status == keyword
            assert (
                card.query_one(".project-index-prefixes", Input).value == "dirty-folder"
            )
            text = str(card.query_one(".project-semantic-status", Static).render())
            assert "unavailable" in text.lower() and "Check status" in text
            assert not next(
                b for b in card.query(Button) if b.name == "semantic-stage"
            ).display
            monkeypatch.setattr(rig.service, "semantic_state", actual_state)
            card.reload_status()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert "Semantic index: Off" in str(
                card.query_one(".project-semantic-status", Static).render()
            )
            from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

            def busy_read(key):
                raise LifetimeBusy("index is busy")

            monkeypatch.setattr(rig.service, "semantic_state", busy_read)
            card.reload_status()
            await pilot.pause()
            while card._pending:
                await pilot.pause(0.02)
            assert "temporarily unavailable" in str(
                card.query_one(".project-semantic-status", Static).render()
            )
    finally:
        await rig.close()


@pytest.mark.parametrize(
    "theme,size", [("textual-dark", (80, 24)), ("textual-light", (170, 48))]
)
@private_profile_test
async def test_actual_settings_semantic_controls_are_focusable_in_mounted_pane(
    tmp_path, request, theme, size
):
    from pathlib import Path

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_rag_result_focus import _assert_painted
    from Tests.UI.test_settings_configuration_hub import _wait_for_selector
    from Tests.UI.test_settings_workspaces_category import (
        _settings_without_splash,
        _wait_for_settings_screen,
    )
    from Tests.app_module_patches import patch_app_global
    from textual.worker import WorkerState

    app = _build_test_app(configured_default="settings")
    app.theme = theme
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="semantic-ui", name="Semantic UI")
    root = tmp_path / "project"
    root.mkdir()
    registry.add_folder_binding("semantic-ui", root)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=size) as pilot:
            boot_worker = None
            for _ in range(900):
                boot_worker = app._boot_worker_handles.get("chachanotes_fts_backfill")
                if boot_worker is not None:
                    break
                await pilot.pause(0.05)
            assert boot_worker is not None, "chachanotes_backfill_worker_not_admitted"
            screen = await _wait_for_settings_screen(app, pilot)
            app.theme = theme
            await _wait_for_selector(screen, pilot, "#settings-category-workspaces")
            screen.query_one("#settings-category-workspaces", Button).press()
            for _ in range(300):
                rows = [
                    b
                    for b in screen.query(".settings-workspace-row")
                    if "Semantic UI" in str(b.label)
                ]
                if rows:
                    break
                await pilot.pause(0.02)
            assert rows
            rows[0].press()
            for _ in range(300):
                await pilot.pause(0.02)
                cards = list(screen.query(card_module().ProjectIndexBindingCard))
                if cards and cards[0].semantic_status is not None:
                    break
            assert cards and cards[0].semantic_status is not None, (
                cards[0].status if cards else None,
                cards[0]._pending if cards else None,
                str(cards[0].query_one(".project-index-feedback", Static).render())
                if cards
                else "",
            )
            card = cards[0]
            for name in ("semantic-stage",):
                button = next(b for b in card.query(Button) if b.name == name)
                button.focus()
                button.scroll_visible(animate=False)
                await pilot.pause()
                _assert_painted(screen, button)
                assert app.focused is button
                assert button.region.width <= screen.region.width
            qa = (
                Path(__file__).resolve().parents[2]
                / ".superpowers/sdd/2026-10-20-workspace-project-semantic-index"
            )
            app.save_screenshot(
                filename=f"task4-settings-semantic-{theme}-{size[0]}.svg", path=str(qa)
            )
            await asyncio.wait_for(boot_worker.wait(), timeout=120)
            assert boot_worker.state is WorkerState.SUCCESS
            assert boot_worker._task is not None and boot_worker._task.done()
            button = next(b for b in card.query(Button) if b.name == "semantic-stage")
            button.focus()
            button.scroll_visible(animate=False)
            await pilot.pause()
            _assert_painted(screen, button)
            assert app.focused is button
