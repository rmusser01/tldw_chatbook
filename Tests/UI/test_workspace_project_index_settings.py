"""Mounted folder-index controls and exact-scope management fencing."""

import importlib
from dataclasses import replace

import pytest
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.Workspaces.test_project_index_service import Rig, POLICY


def card_module():
    name = "tldw_chatbook.Widgets.Settings_Widgets.project_index_binding_card"
    assert importlib.util.find_spec(
        name
    ), "Settings needs a folder index management card"
    return importlib.import_module(name)


async def wait_for_card(card, pilot):
    import time

    deadline = time.monotonic() + 60
    while card.status is None or card._pending:
        assert time.monotonic() < deadline, (
            card.is_mounted,
            card._request,
            card._pending,
        )
        await pilot.pause(0.02)


class CardApp(App):
    CSS_PATH = "../../tldw_chatbook/css/tldw_cli_modular.tcss"

    def __init__(self, card):
        super().__init__()
        self.card = card

    def compose(self):
        with VerticalScroll():
            yield self.card


@pytest.mark.parametrize(
    "state,actions",
    [
        ("Off", {"enable", "clear"}),
        ("Building", {"pause", "disable", "clear"}),
        ("Partial", {"refresh", "disable", "clear", "apply"}),
        ("Ready", {"refresh", "disable", "clear", "apply"}),
        ("Paused", {"refresh", "resume", "disable", "clear", "apply"}),
        ("Interrupted", {"refresh", "resume", "disable", "clear", "apply"}),
        ("Error", {"refresh", "disable", "clear", "apply"}),
    ],
)
@private_profile_test
async def test_mounted_state_controls_and_counts(
    tmp_path, request, monkeypatch, state, actions
):
    mod = card_module()
    rig = Rig(tmp_path)
    status = replace(
        rig.service.status(rig.key),
        state=state,
        discovered=27,
        indexed=11,
        job_id="job" if state != "Off" else None,
        last_scan_at="2026-10-21T12:00:00Z",
    )
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id,
        workspace_id=rig.key.workspace_id,
        root=str(rig.root),
        status=status,
        policy=POLICY,
    )
    monkeypatch.setattr(card, "reload_status", lambda: None)
    host = CardApp(card)
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            text = " ".join(str(x.render()) for x in card.query(Static))
            assert state in text and "27 discovered" in text and "11 indexed" in text
            assert "2026-10-21" in text
            actual = {
                b.name for b in card.query(Button) if b.display and not b.disabled
            }
            assert actual == actions | {"reload"}
            assert not card.query("#semantic")
    finally:
        await rig.close()


@private_profile_test
async def test_saved_policy_and_stale_actions(tmp_path, request, monkeypatch):
    mod = card_module()
    rig = Rig(tmp_path)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    host = CardApp(card)
    try:
        async with host.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            await host.workers.wait_for_complete()
            await wait_for_card(card, pilot)
            off = card.status
            assert off is not None, (
                card.is_mounted,
                card._visit,
                card._request,
                card._pending,
                str(card.query_one(".project-index-feedback", Static).render()),
                list(host.workers),
            )
            assert off.state == "Off"
            saved = mod.ProjectIndexPolicy(("docs",), (".md",))
            enable = card.ActionRequested(
                card.binding_id, card.workspace_id, off, "enable", saved
            )
            cleared = rig.service.clear(rig.key, expected_scope=off.scope)
            with pytest.raises(ValueError, match="stale_scope"):
                card._execute(host, enable)
            assert rig.service.status(rig.key).scope == cleared.scope
            enabled = rig.service.enable(rig.key, saved, expected_scope=cleared.scope)
            old_clear = card.ActionRequested(
                card.binding_id, card.workspace_id, cleared, "clear", saved
            )
            with pytest.raises(ValueError, match="stale_scope"):
                card._execute(host, old_clear)
            card.reload_status()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert card.status.scope == enabled.scope
            assert card.query_one(".project-index-prefixes", Input).value == "docs"
            assert card.query_one(".project-index-suffixes", Input).value == ".md"
    finally:
        await rig.close()


@private_profile_test
async def test_queued_enable_cannot_reverse_completed_clear(
    tmp_path, request, monkeypatch
):
    import asyncio
    import threading

    mod = card_module()
    rig = Rig(tmp_path)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    host = CardApp(card)
    held, release = threading.Event(), threading.Event()
    execute = card._execute
    errors = []

    def delayed(app, action):
        if action.action == "enable":
            held.set()
            assert release.wait(60)
        try:
            return execute(app, action)
        except Exception as exc:
            errors.append(str(exc))
            raise

    monkeypatch.setattr(card, "_execute", delayed)
    try:
        async with host.run_test() as pilot:
            await pilot.pause()
            await host.workers.wait_for_complete()
            await wait_for_card(card, pilot)
            original = card.status
            card.post_message(
                card.ActionRequested(
                    card.binding_id, card.workspace_id, original, "enable", POLICY
                )
            )
            assert await asyncio.to_thread(held.wait, 5)
            card.post_message(
                card.ActionRequested(
                    card.binding_id, card.workspace_id, original, "clear", POLICY
                )
            )
            deadline = asyncio.get_running_loop().time() + 30
            while card.status.scope == original.scope:
                assert asyncio.get_running_loop().time() < deadline, (
                    card._pending,
                    card._request,
                    errors,
                )
                await pilot.pause(0.02)
            cleared = card.status
            assert cleared.state == "Off" and cleared.scope != original.scope
            release.set()
            await host.workers.wait_for_complete()
            assert errors == ["stale_scope"]
            assert card.status == cleared
            assert rig.service.status(rig.key).job_id is None
            assert not rig.service._runs
    finally:
        release.set()
        await rig.close()


@pytest.mark.parametrize(
    "theme,size", [("textual-dark", (80, 24)), ("textual-light", (170, 48))]
)
@private_profile_test
async def test_actual_settings_card_enable_build_clear_preserves_library(
    tmp_path, request, monkeypatch, theme, size
):
    import asyncio
    from pathlib import Path
    from Tests.UI.app_factory import _build_test_app
    from Tests.app_module_patches import patch_app_global
    from Tests.UI.test_settings_workspaces_category import (
        _settings_without_splash,
        _wait_for_settings_screen,
    )
    from Tests.UI.test_settings_configuration_hub import _wait_for_selector
    from Tests.UI.test_library_rag_result_focus import _assert_painted
    from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service
    from Tests.Agents.test_project_index_tool import provider, invoke
    from types import SimpleNamespace

    app = _build_test_app(configured_default="settings")
    app.theme = theme
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="k5-project", name="Keyword project")
    root = tmp_path / "project"
    (root / "docs/deep").mkdir(parents=True)
    (root / "docs/deep/marker.txt").write_text("deepmountedmarker", encoding="utf-8")
    binding = registry.add_folder_binding("k5-project", root)
    before_library = (
        repr(app.app_config.get("AppRAGSearchConfig")),
        repr(app.app_config.get("search")),
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=size) as pilot:
            screen = await _wait_for_settings_screen(app, pilot)
            app.theme = theme
            await pilot.pause()
            assert app.theme == theme
            assert app.current_theme.dark == (theme == "textual-dark")
            await _wait_for_selector(screen, pilot, "#settings-category-workspaces")
            screen.query_one("#settings-category-workspaces", Button).press()
            for _ in range(100):
                await pilot.pause(0.02)
                if not screen._category_pane_swap_pending and screen.query(
                    "#settings-workspace-row-k5-project"
                ):
                    break
            await _wait_for_selector(
                screen, pilot, "#settings-workspace-row-k5-project"
            )
            screen.query_one("#settings-workspace-row-k5-project", Button).press()
            mod = card_module()
            for _ in range(500):
                await pilot.pause(0.02)
                cards = list(screen.query(mod.ProjectIndexBindingCard))
                if cards and cards[0].status:
                    break
            card = cards[0]
            assert card.status and card.status.state == "Off", (
                card.is_mounted,
                card._visit,
                card._request,
                card._pending,
                str(card.query_one(".project-index-feedback", Static).render()),
                [(w.name, w.state, str(w.error)) for w in app.workers],
                len(cards),
                __import__(
                    "Tests.private_profile", fromlist=["_stacks_text"]
                )._stacks_text("card incomplete"),
            )
            assert card.binding_id == binding.binding_id

            async def action(name, state):
                button = next(b for b in card.query(Button) if b.name == name)
                button.focus()
                await pilot.pause()
                button.scroll_visible(animate=False)
                await pilot.pause()
                _assert_painted(screen, button)
                assert app.focused is button
                await pilot.press("enter")
                import time

                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    await pilot.pause(0.02)
                    if not card._pending and card.status and card.status.state == state:
                        return
                raise AssertionError(
                    str(card.query_one(".project-index-feedback", Static).render())
                )

            await action("enable", "Paused")
            await action("refresh", "Ready")
            service = await asyncio.to_thread(get_project_index_service, app)
            rig = SimpleNamespace(service=service, root=root, key=card.status.scope.key)
            tool = await asyncio.to_thread(provider, rig)
            result = await asyncio.to_thread(
                invoke, tool, {"query": "deepmountedmarker"}
            )
            assert result.ok and "deepmountedmarker" in result.content
            qa = (
                Path(__file__).resolve().parents[2]
                / ".superpowers/sdd/2026-10-20-workspace-project-keyword-index"
            )
            assert app.theme == theme and app.current_theme.dark == (
                theme == "textual-dark"
            )
            app.save_screenshot(
                filename=f"k5-settings-{theme}-{size[0]}.svg", path=str(qa)
            )
            stable_request = card._request
            await pilot.pause(2.2)
            assert (
                card.is_mounted
                and card._pending == 0
                and card._request == stable_request
            )
            assert list(screen.query(mod.ProjectIndexBindingCard)) == [card]
            await action("clear", "Off")
            app.save_screenshot(
                filename=f"k5-settings-clear-{theme}-{size[0]}.svg", path=str(qa)
            )
            assert card.status.indexed == 0
            assert before_library == (
                repr(app.app_config.get("AppRAGSearchConfig")),
                repr(app.app_config.get("search")),
            )


@pytest.mark.parametrize(
    "available,supported,copy",
    [
        (False, True, "Restore folder access"),
        (True, False, "SSH folders are not supported"),
    ],
)
@private_profile_test
async def test_unavailable_card_has_no_authority_io(
    request, monkeypatch, available, supported, copy
):
    mod = card_module()

    def forbidden(*args):
        raise AssertionError("unavailable display tried to derive authority")

    monkeypatch.setattr(mod, "get_project_index_service", forbidden)
    card = mod.ProjectIndexBindingCard(
        "binding",
        workspace_id="workspace",
        root="unavailable",
        available=available,
        supported=supported,
    )
    host = CardApp(card)
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert copy in str(card.query_one(".project-index-status", Static).render())
        assert not any(button.display for button in card.query(Button))
        assert all(field.disabled for field in card.query(Input))
        assert card._request == 0


@private_profile_test
async def test_busy_action_keeps_exact_scope_and_readable_retry(
    request, monkeypatch, tmp_path
):
    mod = card_module()
    rig = Rig(tmp_path)
    status = rig.service.status(rig.key)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id,
        workspace_id=rig.key.workspace_id,
        root=str(rig.root),
        status=status,
    )
    monkeypatch.setattr(card, "reload_status", lambda: None)

    def busy(*args):
        raise mod.LifetimeBusy("index_reads_busy")

    monkeypatch.setattr(card, "_execute", busy)
    host = CardApp(card)
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            card.post_message(
                card.ActionRequested(
                    card.binding_id, card.workspace_id, status, "enable", POLICY
                )
            )
            for _ in range(50):
                await pilot.pause(0.02)
                if "Retry" in str(
                    card.query_one(".project-index-feedback", Static).render()
                ):
                    break
            assert "Retry" in str(
                card.query_one(".project-index-feedback", Static).render()
            )
            assert card.status == status
    finally:
        await rig.close()


@pytest.mark.parametrize(
    "policy",
    [
        POLICY.__class__((), ()),
        POLICY.__class__(("src",), ()),
        POLICY.__class__((), (".py",)),
    ],
)
@private_profile_test
async def test_empty_processing_choices_round_trip_on_mounted_card(
    tmp_path, request, monkeypatch, policy
):
    mod = card_module()
    rig = Rig(tmp_path)
    monkeypatch.setattr(mod, "get_project_index_service", lambda app: rig.service)
    monkeypatch.setattr(mod, "project_index_key", lambda *args: rig.key)
    current = rig.service.status(rig.key)
    rig.service.enable(rig.key, policy, expected_scope=current.scope)
    card = mod.ProjectIndexBindingCard(
        rig.key.binding_id, workspace_id=rig.key.workspace_id, root=str(rig.root)
    )
    host = CardApp(card)
    try:
        async with host.run_test() as pilot:
            await pilot.pause()
            await wait_for_card(card, pilot)
            assert card.policy == policy
            assert card.query_one(".project-index-prefixes", Input).value == "; ".join(
                policy.include_prefixes
            )
            assert card.query_one(".project-index-suffixes", Input).value == "; ".join(
                policy.text_suffixes
            )
            button = next(b for b in card.query(Button) if b.name == "apply")
            button.press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            assert rig.service.policy(card.status.scope) == policy
    finally:
        await rig.close()


@private_profile_test
async def test_default_policy_indexes_extensionless_and_manifests(tmp_path, request):
    mod = card_module()
    assert mod.DEFAULT_POLICY == POLICY.__class__((), ())
    rig = Rig(tmp_path)
    try:
        for name in ("README", "project.toml", "package.json", "module.py"):
            (rig.root / name).write_text("defaultcoverage marker", encoding="utf-8")
        initial = rig.service.status(rig.key)
        rig.service.enable(rig.key, mod.DEFAULT_POLICY, expected_scope=initial.scope)
        rig.refresh()
        assert rig.settle().indexed == 4
    finally:
        await rig.close()


@private_profile_test
async def test_mounted_legacy_management_clear_needs_no_upgrade_space(
    tmp_path, request, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.project_index import (
        runtime,
        service as service_module,
    )
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    assert hasattr(service_module.ProjectIndexService, "management_snapshot")
    root = tmp_path / "project"
    root.mkdir()
    (root / "marker.txt").write_text("legacy marker")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="legacy-ws", name="Legacy")
    binding = registry.add_folder_binding("legacy-ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "legacy-ws", binding.binding_id, root)
    initial = service.status(key)
    enabled = service.enable(key, POLICY, expected_scope=initial.scope)
    job = service.refresh(
        key, expected_scope=enabled.scope, expected_job_revision=enabled.job_revision
    )
    await asyncio.to_thread(service._runs[job].supervisor.join)
    ready = service.status(key)
    with service._factory.retain(key) as pin, pin.bind():
        store = service._factory(key)
        try:
            with store.transaction() as conn:
                conn.execute("DROP TABLE chunk_fts")
                conn.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(text)")
                conn.execute(
                    "INSERT INTO chunk_fts(rowid,text) SELECT id,text FROM chunks"
                )
                conn.execute("DROP INDEX directories_eligible")
                conn.execute("ALTER TABLE directories DROP COLUMN attempt_revision")
                conn.execute("DROP TABLE schema_version")
                conn.execute(
                    "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=2))"
                )
                conn.execute("INSERT INTO schema_version VALUES(2)")
        finally:
            store.close_native()
    path = index_store_path(service._factory.data_root, key)
    cap = 1024**2 + 2 * path.stat().st_size - 8192
    monkeypatch.setattr(runtime, "MAX_CACHE_BYTES", cap)
    monkeypatch.setattr(service_module, "MAX_CACHE_BYTES", cap)
    with pytest.raises(ValueError, match="index_cache_limit"):
        service.status(key)
    mod = card_module()
    card = mod.ProjectIndexBindingCard(
        binding.binding_id, workspace_id="legacy-ws", root=str(root)
    )
    host = CardApp(card)
    host.workspace_registry_service = registry
    host._project_index_service = service
    try:
        async with host.run_test() as pilot:
            await pilot.pause()
            await wait_for_card(card, pilot)
            assert card.requires_upgrade and card.status.scope == ready.scope
            text = str(card.query_one(".project-index-status", Static).render())
            assert "upgrade required" in text and "Ready" not in text
            assert card.policy == POLICY
            original = card.status.scope
            next(b for b in card.query(Button) if b.name == "clear").press()
            await pilot.pause()
            await wait_for_card(card, pilot)
            assert card.status.state == "Off" and card.status.scope != original
            assert card.status.indexed == 0 and card.requires_upgrade
            with service._factory.retain(key) as pin, pin.bind():
                store = service._factory.open_cleanup(key)
                try:
                    with store.connection() as conn:
                        assert (
                            conn.execute(
                                "SELECT version FROM schema_version"
                            ).fetchone()[0]
                            == 2
                        )
                        assert not conn.execute(
                            "SELECT 1 FROM sqlite_schema WHERE name='directories_eligible'"
                        ).fetchone()
                finally:
                    store.close_native()
    finally:
        await runtime.close_project_index_service(app)
        registry.db.close()
