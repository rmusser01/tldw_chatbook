"""App-wiring guard for lazy RAG admin service construction (task-254).

The RAG admin trio (``server_rag_admin_service`` / ``local_rag_admin_service``
/ ``rag_admin_scope_service``) has no reachable UI consumer since the legacy
SearchWindow stack was deleted (PR #669), so ``TldwCli.__init__`` must not pay
for constructing it at every launch. The services are instead built lazily on
first property access and cached, so any future surface (e.g. a rebuilt admin
screen or task-251 indexing controls) still gets the identically-wired trio.

These tests construct a real ``TldwCli`` (with the same heavy-init patches the
screen-navigation wiring tests use) while the three service classes are
replaced with mocks in the ``tldw_chatbook.app`` namespace, then assert:

1. zero constructions after ``__init__`` (startup pays nothing), and
2. exactly-once construction, caching, and correct trio wiring on access.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


@contextlib.contextmanager
def _cheap_app_init_patches():
    """Patch heavy ``TldwCli.__init__`` collaborators, mirroring the proven
    ``_build_test_app`` recipe in ``Tests/UI/test_screen_navigation.py``."""
    user_data_dir = Path(tempfile.mkdtemp(prefix="tldw-chatbook-rag254-test-"))

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app.current_runtime_source = "local"
        app.current_runtime_backend = "local"
        return context

    def fake_cli_setting(_section, _key=None, default=None):
        return default

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "tldw_chatbook.app.load_settings",
                return_value={"tldw_api": {"base_url": "http://localhost:8000"}},
            )
        )
        stack.enter_context(patch("tldw_chatbook.app.get_cli_setting", side_effect=fake_cli_setting))
        stack.enter_context(patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None))
        stack.enter_context(
            patch("tldw_chatbook.app.ServerNotesWorkspaceService.from_config", return_value=MagicMock())
        )
        stack.enter_context(
            patch("tldw_chatbook.app.ServerCharacterPersonaService.from_config", return_value=MagicMock())
        )
        stack.enter_context(
            patch.object(TldwCli, "_init_notes_service", lambda self, _user: setattr(self, "notes_service", None))
        )
        stack.enter_context(
            patch.object(
                TldwCli, "_init_prompts_service", lambda self: setattr(self, "prompts_service_initialized", False)
            )
        )
        stack.enter_context(
            patch.object(TldwCli, "_init_providers_models", lambda self: setattr(self, "providers_models", {}))
        )
        stack.enter_context(
            patch.object(
                TldwCli,
                "_init_media_db",
                lambda self: (
                    setattr(self, "media_db", None),
                    setattr(self, "_media_types_for_ui", ["All Media"]),
                ),
            )
        )
        stack.enter_context(
            patch("tldw_chatbook.app.load_runtime_policy_for_app", side_effect=fake_runtime_policy)
        )
        stack.enter_context(patch("tldw_chatbook.app.get_notifications_db_path", return_value=":memory:"))
        stack.enter_context(patch("tldw_chatbook.app.get_subscriptions_db_path", return_value=":memory:"))
        stack.enter_context(patch("tldw_chatbook.app.get_research_db_path", return_value=":memory:"))
        stack.enter_context(patch("tldw_chatbook.app.get_writing_db_path", return_value=":memory:"))
        stack.enter_context(patch("tldw_chatbook.app.get_user_data_dir", return_value=user_data_dir))
        stack.enter_context(
            patch(
                "tldw_chatbook.app.get_workspaces_db_path",
                return_value=user_data_dir / "workspaces.sqlite",
            )
        )
        yield


@contextlib.contextmanager
def _patched_rag_admin_classes():
    """Replace the RAG admin service classes referenced by the lazy builder."""
    with patch("tldw_chatbook.app.ServerRAGAdminService") as server_cls, patch(
        "tldw_chatbook.app.LocalRAGAdminService"
    ) as local_cls, patch("tldw_chatbook.app.RAGAdminScopeService") as scope_cls:
        yield server_cls, local_cls, scope_cls


def test_app_init_does_not_construct_rag_admin_services():
    """Startup must not build the RAG admin trio (task-254 AC #1)."""
    with _cheap_app_init_patches(), _patched_rag_admin_classes() as (server_cls, local_cls, scope_cls):
        app = TldwCli()

        assert server_cls.from_config.call_count == 0
        assert server_cls.call_count == 0
        assert local_cls.call_count == 0
        assert scope_cls.call_count == 0
        # The lazy slots exist but are unbuilt.
        assert app._server_rag_admin_service is None
        assert app._local_rag_admin_service is None
        assert app._rag_admin_scope_service is None


def test_first_access_builds_and_caches_rag_admin_trio_exactly_once():
    """First property access builds the identically-wired trio; later accesses reuse it."""
    with _cheap_app_init_patches(), _patched_rag_admin_classes() as (server_cls, local_cls, scope_cls):
        app = TldwCli()

        scope = app.rag_admin_scope_service

        # Config-driven server service, local service over the media DB, scope
        # service routing between them — same semantics as the old eager wiring.
        server_cls.from_config.assert_called_once_with(
            app.app_config,
            policy_enforcer=app.service_policy_enforcer,
        )
        local_cls.assert_called_once_with(
            app.media_db,
            media_service=app.local_media_reading_service,
        )
        scope_cls.assert_called_once_with(
            local_service=local_cls.return_value,
            server_service=server_cls.from_config.return_value,
            policy_enforcer=app.service_policy_enforcer,
        )
        assert scope is scope_cls.return_value

        # Every property is cached: repeated/companion access constructs nothing new.
        assert app.rag_admin_scope_service is scope
        assert app.server_rag_admin_service is server_cls.from_config.return_value
        assert app.local_rag_admin_service is local_cls.return_value
        assert server_cls.from_config.call_count == 1
        assert local_cls.call_count == 1
        assert scope_cls.call_count == 1


def test_server_service_config_failure_falls_back_to_clientless_service():
    """A ValueError from ``from_config`` must fall back to ``client=None`` wiring."""
    with _cheap_app_init_patches(), _patched_rag_admin_classes() as (server_cls, local_cls, scope_cls):
        server_cls.from_config.side_effect = ValueError("no server configured")
        app = TldwCli()

        server = app.server_rag_admin_service

        server_cls.assert_called_once_with(
            client=None,
            policy_enforcer=app.service_policy_enforcer,
        )
        assert server is server_cls.return_value
        scope_cls.assert_called_once_with(
            local_service=local_cls.return_value,
            server_service=server_cls.return_value,
            policy_enforcer=app.service_policy_enforcer,
        )
