"""Artifact compatibility routes and exact-claim ownership in Library."""

from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope
from tldw_chatbook.UI.Library_Modules.library_artifacts_navigation import (
    LibraryArtifactsNavigation,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route
from tldw_chatbook.UI.Navigation.shell_destinations import resolve_shell_route

CHANNEL = HandoffChannel.ARTIFACT_CHATBOOK_TARGET


class Controller:
    def __init__(self):
        self.scope = ArtifactScope(view="chatbooks", query="old-filter", sort="title")
        self.generation = 0
        self.profile_identity = (1, 2, 3)
        self.opened = []

    def profile(self):
        return self.profile_identity

    def enter_view(self, view):
        self.scope = ArtifactScope(view=view, query="old-filter")

    def open_target(self, key):
        self.generation += 1
        self.opened.append(key)


@pytest.fixture
def navigation():
    controller = Controller()
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(pending_handoffs=PendingHandoffStore()),
        _artifacts_controller=controller,
        _library_selected_row_id="artifacts-chatbooks",
        is_mounted=True,
        applied=[],
    )
    screen.app = SimpleNamespace(screen=screen)
    screen.apply_navigation_context = lambda context: screen.applied.append(context)
    adapter = LibraryArtifactsNavigation(screen)
    return adapter, controller, screen, screen.app_instance.pending_handoffs


def admit(adapter, store, target="local:chatbook:41"):
    store.stage(CHANNEL, target)
    context = adapter.prepare_context({"mode": "artifacts-all"})
    assert context == {"mode": "artifacts-chatbooks"}
    adapter.resume()


@private_profile_test
def test_artifacts_resolves_library_but_chatbooks_remains_manager(request):
    assert resolve_shell_route("artifacts").destination_id == "library"
    assert resolve_screen_route("artifacts").class_name == "LibraryScreen"
    assert resolve_shell_route("chatbooks").destination_id == "library"
    assert resolve_screen_route("chatbooks").class_name == "ChatbooksScreen"


@private_profile_test
def test_exact_claim_overrides_filter_and_settles_only_after_applied_target(
    navigation, request
):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    key = ArtifactKey("chatbook", 41)
    assert controller.scope == ArtifactScope(view="chatbooks")
    assert controller.opened == [key]
    assert store.exact_revision_status(CHANNEL, 1) == "in_flight"
    adapter.resume()
    assert controller.opened == [key]
    assert adapter.target_is_current(key, controller.generation)
    adapter.target_finished(key, controller.generation)
    assert store.exact_revision_status(CHANNEL, 1) == "settled"
    assert not store.has_pending(CHANNEL)


@private_profile_test
def test_new_pending_target_rejects_old_result_and_release_cannot_overwrite_it(
    navigation, request
):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    old_key, old_generation = controller.opened[-1], controller.generation
    store.stage(CHANNEL, "local:chatbook:99")
    assert not adapter.target_is_current(old_key, old_generation)
    adapter.invalidate()
    adapter.prepare_context({"mode": "artifacts-all"})
    adapter.resume()
    assert controller.opened[-1] == ArtifactKey("chatbook", 99)
    assert not adapter.target_is_current(old_key, old_generation)
    adapter.target_finished(old_key, old_generation)
    assert store.exact_revision_status(CHANNEL, 2) == "in_flight"
    adapter.target_finished(controller.opened[-1], controller.generation, missing=True)
    assert store.exact_revision_status(CHANNEL, 2) == "settled"


@private_profile_test
def test_source_failure_releases_target_for_explicit_retry(navigation, request):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    key, generation = controller.opened[-1], controller.generation
    adapter.target_failed(key, generation)
    assert store.has_pending(CHANNEL)
    assert not adapter.target_is_current(key, generation)
    assert adapter.retry()
    assert controller.opened == [key, key]
    assert controller.generation > generation
    assert adapter.target_is_current(key, controller.generation)


@pytest.mark.parametrize(
    "context",
    [
        {"mode": "notes"},
        {"note_id": "exact-note"},
        {"mode": "artifacts-reports"},
        {"conversation_id": "conversation"},
    ],
)
@private_profile_test
def test_unrelated_explicit_library_context_keeps_precedence(
    navigation, context, request
):
    adapter, controller, screen, store = navigation
    store.stage(CHANNEL, "local:chatbook:41")
    assert adapter.prepare_context(context) == context
    adapter.resume()
    assert controller.opened == []
    assert store.has_pending(CHANNEL)
    assert screen.applied == []


@private_profile_test
def test_route_admission_uses_existing_dirty_guards_before_claiming(
    navigation, request
):
    adapter, controller, screen, store = navigation
    store.stage(CHANNEL, "local:chatbook:41")
    adapter.prepare_context({"mode": "artifacts-all"})
    screen._library_selected_row_id = "browse-notes"
    adapter.resume()
    assert screen.applied == [{"mode": "artifacts-chatbooks"}]
    assert store.has_pending(CHANNEL)
    assert controller.opened == []


@private_profile_test
def test_profile_change_and_departure_reject_late_target_without_consuming_it(
    navigation, request
):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    key, generation = controller.opened[-1], controller.generation
    controller.profile_identity = (4, 5, 6)
    assert not adapter.target_is_current(key, generation)
    adapter.invalidate()
    adapter.target_finished(key, generation)
    assert store.has_pending(CHANNEL)
    # Ordinary Keep navigation is not an artifact handoff claim.
    controller.generation += 1
    assert adapter.target_is_current(
        ArtifactKey("kept_report", 7), controller.generation
    )


@private_profile_test
def test_explicit_chatbook_context_stages_validated_existing_channel(
    navigation, request
):
    adapter, controller, _, store = navigation
    context = adapter.prepare_context({"artifact_chatbook_id": "local:chatbook:42"})
    assert context == {"mode": "artifacts-chatbooks"}
    adapter.resume()
    assert controller.opened == [ArtifactKey("chatbook", 42)]
    assert store.exact_revision_status(CHANNEL, 1) == "in_flight"


@private_profile_test
def test_new_navigation_retries_pending_source_target(navigation, request):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    adapter.target_failed(controller.opened[-1], controller.generation)
    adapter.prepare_context({"artifact_chatbook_id": "local:chatbook:99"})
    adapter.resume()
    assert controller.opened[-1] == ArtifactKey("chatbook", 99)
    assert store.exact_revision_status(CHANNEL, 2) == "in_flight"


@pytest.mark.parametrize(
    "target", ["local:chatbook:chatbook-77", "local:chatbook:01", "local:chatbook:-1"]
)
@private_profile_test
def test_invalid_legacy_target_is_terminal_missing_without_another_selection(
    navigation, target, request
):
    adapter, controller, _, store = navigation
    controller.unavailable = []
    controller.target_unavailable = lambda: controller.unavailable.append(True)
    admit(adapter, store, target)
    assert controller.opened == []
    assert controller.unavailable == [True]
    assert controller.generation == 1
    assert store.exact_revision_status(CHANNEL, 1) == "settled"


@private_profile_test
async def test_artifacts_default_and_shortcut_use_same_library_context(
    request, monkeypatch
):
    from unittest.mock import AsyncMock, Mock

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.app import TldwCli

    app = _build_test_app(configured_default="artifacts")
    assert app._resolve_initial_shell_route() == "artifacts"
    name, tab, screen_class = app._resolve_screen_navigation_target("artifacts")
    assert (name, tab, screen_class.__name__) == ("library", "library", "LibraryScreen")
    initial = Mock()
    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda route: (name, tab, lambda _: initial),
    )
    monkeypatch.setattr(app, "_ensure_screen_owned_css", Mock())
    monkeypatch.setattr(app, "_retain_reusable_navigation_screen", Mock())
    monkeypatch.setattr(app, "push_screen", AsyncMock())
    await app._push_initial_screen()
    initial.apply_navigation_context.assert_called_once_with({"mode": "artifacts-all"})
    posted = []
    TldwCli.action_shell_destination(
        SimpleNamespace(post_message=posted.append), "artifacts"
    )
    assert posted[0].screen_name == "artifacts"
    bindings = {binding.key: binding.action for binding in TldwCli.BINDINGS}
    assert bindings["ctrl+6"] == "shell_destination('artifacts')"
    expected = {
        "ctrl+7": "schedules",
        "ctrl+8": "workflows",
        "ctrl+9": "mcp",
        "ctrl+0": "acp",
        "f2": "lab",
        "f3": "logs",
        "f4": "settings",
        "f5": "research",
        "f7": "meetings",
    }
    assert all(
        bindings[key] == f"shell_destination({destination!r})"
        for key, destination in expected.items()
    )


@private_profile_test
async def test_palette_artifacts_has_exact_library_command(request):
    from textual.screen import Screen

    from tldw_chatbook.app import TabNavigationProvider

    provider = TabNavigationProvider(Screen())
    hits = [hit async for hit in provider.search("artifacts")]
    exact = [
        hit for hit in hits if str(hit.text) == "Tab Navigation: Library — Artifacts"
    ]
    assert len(exact) == 1
    assert exact[0].command.args == ("artifacts",)


@private_profile_test
def test_explicit_new_target_releases_old_claim_immediately(navigation, request):
    adapter, controller, _, store = navigation
    admit(adapter, store)
    old_key, old_generation = controller.opened[-1], controller.generation
    adapter.prepare_context({"artifact_chatbook_id": "local:chatbook:99"})
    assert not adapter.target_is_current(old_key, old_generation)
    assert store.exact_revision_status(CHANNEL, 1) == "superseded"
    adapter.resume()
    assert controller.opened[-1] == ArtifactKey("chatbook", 99)
    assert store.exact_revision_status(CHANNEL, 2) == "in_flight"
