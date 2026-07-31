# app_factory.py
# Description: The shared TldwCli test-app factory (task-1458).
#
# `_build_test_app` was born inside test_screen_navigation.py and ended up
# imported by 90+ test modules and called ~1,300 times per full run — a test
# module is the wrong home for suite-wide infrastructure, and its per-call
# `tempfile.mkdtemp` was never cleaned up (the 2026-07-30 audit found 324k
# leaked sandboxes totalling ~285GB on one dev machine). The factory now
# records every directory it creates and the root conftest drains them after
# each test (`drain_created_dirs`).
#
# Deliberately NOT here: any form of app-instance caching or reuse across
# tests. That was tried, and it produced wedged compositors and dead message
# pumps (see the regression coverage in test_screen_navigation.py). Every
# caller gets a fresh TldwCli.

from __future__ import annotations

import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy import RuntimeSourceState

# Every user-data dir handed to a TldwCli built here; drained (rmtree'd) by the
# root conftest's autouse cleanup after each test.
_created_dirs: list[Path] = []


def drain_created_dirs() -> int:
    """Remove every user-data dir created since the last drain.

    Called by the root conftest's autouse cleanup fixture after each test, so
    a test that builds several apps leaks nothing. Removal happens while the
    app objects may still hold open sqlite handles — POSIX unlink semantics
    make that safe, and the per-test gc in app-mounting dirs (task-1468)
    closes the handles promptly afterwards.

    Returns:
        The number of directories removed.
    """
    drained = 0
    while _created_dirs:
        path = _created_dirs.pop()
        shutil.rmtree(path, ignore_errors=True)
        drained += 1
    return drained


def _build_test_app(
    configured_default: str | None = None,
    *,
    first_run_setup_completed: bool = True,
) -> TldwCli:
    """Build a TldwCli instance with every real I/O seam faked out.

    Args:
        configured_default: Value returned for the ``general.default_tab``
            CLI setting, letting a test choose the app's initial route.
        first_run_setup_completed: Defaults to True: task-11 added a
            first-run setup wizard that FirstRunSetupWizard.first_run_setup_state.
            should_offer_wizard() auto-offers (pushed on top of whatever the
            initial screen is) whenever it sees no configured provider and no
            wizard state -- which the synthetic config below otherwise looks
            exactly like. Every pre-existing caller of this builder predates
            the wizard and asserts against its target screen/route appearing
            directly, not a modal pushed on top of it, so setup is marked
            already-completed by default here -- mirroring how a real,
            already-configured user opts out of the offer. Pass ``False`` for
            a test that specifically wants to exercise the auto-offer itself
            (see test_product_maturity_phase1_first_run.py's
            test_fresh_config_auto_offers_wizard_over_initial_screen).

    Returns:
        A freshly constructed ``TldwCli`` whose config, DB paths, and service
        initialisers were all patched for the duration of ``__init__``.
    """
    user_data_dir = Path(
        tempfile.mkdtemp(prefix="tldw-chatbook-test-")
        # `.resolve(strict=True)` is load-bearing, not tidiness: on macOS
        # mkdtemp returns /var/folders/..., /var is a symlink, and the
        # private-path guard refuses to traverse a symlinked component.
        # Without it every test on this harness dies with
        # `PrivatePathError: link_or_non_regular` before its first assertion.
    ).resolve(strict=True)
    _created_dirs.append(user_data_dir)

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app._publish_runtime_policy_projection(context.state)
        return context

    def fake_cli_setting(_section, _key=None, default=None):
        if (
            _section == "general"
            and _key == "default_tab"
            and configured_default is not None
        ):
            return configured_default
        return default

    fake_app_config: dict = {"tldw_api": {"base_url": "http://localhost:8000"}}
    if first_run_setup_completed:
        fake_app_config["first_run"] = {"setup_completed": True}

    with ExitStack() as stack:
        for ctx in (
            patch("tldw_chatbook.app.load_settings", return_value=fake_app_config),
            patch("tldw_chatbook.app.get_cli_setting", side_effect=fake_cli_setting),
            patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None),
            patch(
                "tldw_chatbook.app.ServerNotesWorkspaceService.from_config",
                return_value=MagicMock(),
            ),
            patch(
                "tldw_chatbook.app.ServerCharacterPersonaService.from_config",
                return_value=MagicMock(),
            ),
            patch.object(
                TldwCli,
                "_init_notes_service",
                lambda self, _user: setattr(self, "notes_service", None),
            ),
            patch.object(
                TldwCli,
                "_init_prompts_service",
                lambda self: setattr(self, "prompts_service_initialized", False),
            ),
            patch.object(
                TldwCli,
                "_init_providers_models",
                lambda self: setattr(self, "providers_models", {}),
            ),
            patch.object(
                TldwCli,
                "_init_media_db",
                lambda self: (
                    setattr(self, "media_db", None),
                    setattr(self, "_media_types_for_ui", ["All Media"]),
                ),
            ),
            patch(
                "tldw_chatbook.app.load_runtime_policy_for_app",
                side_effect=fake_runtime_policy,
            ),
            patch(
                "tldw_chatbook.app.get_notifications_db_path",
                return_value=":memory:",
            ),
            patch(
                "tldw_chatbook.app.get_subscriptions_db_path",
                return_value=user_data_dir / "subscriptions.sqlite",
            ),
            patch(
                "tldw_chatbook.app.get_research_db_path",
                return_value=user_data_dir / "research.sqlite",
            ),
            patch(
                "tldw_chatbook.app.get_writing_db_path",
                return_value=user_data_dir / "writing.sqlite",
            ),
            patch(
                "tldw_chatbook.app.get_user_data_dir",
                return_value=user_data_dir,
            ),
            patch(
                "tldw_chatbook.app.get_workspaces_db_path",
                return_value=user_data_dir / "workspaces.sqlite",
            ),
            patch(
                "tldw_chatbook.app.get_scheduled_tasks_db_path",
                return_value=user_data_dir / "scheduled_tasks.sqlite",
            ),
        ):
            stack.enter_context(ctx)
        return TldwCli()
