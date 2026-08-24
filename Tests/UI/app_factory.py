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
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import load_settings
from tldw_chatbook.runtime_policy import RuntimeSourceState

# Every user-data dir handed to a TldwCli built here; drained (rmtree'd) by the
# root conftest's autouse cleanup after each test.
_created_dirs: list[Path] = []

# Every still-running `get_subscriptions_db_path` patch started by
# `_build_test_app` (task-1631); stopped by the root conftest's autouse
# cleanup after each test. See `_build_test_app`'s own comment for why this
# one patch cannot simply live inside the function's `ExitStack` like the
# others.
_active_service_patches: list = []


def _deep_merge_into(target: dict, overrides: Mapping[str, Any]) -> dict:
    """Merge ``overrides`` into ``target`` in place, section by section."""
    for key, value in overrides.items():
        existing = target.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            _deep_merge_into(existing, value)
        else:
            target[key] = deepcopy(value)
    return target


def build_test_app_config(
    *,
    first_run_setup_completed: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Return the ``app_config`` a test-built ``TldwCli`` boots with.

    task-15270. This used to be a three-key synthetic dict, and that quietly
    hollowed out every mounted Console test. `ChatScreen`'s readiness/turn
    context seam (`_provider_readiness_app_config`) re-sources its config
    from `load_settings()` only when the snapshot it was handed *looks*
    disk-loaded -- it checks for the sections `load_settings()` always emits
    (`_CONSOLE_LIVE_CONFIG_MARKER_SECTIONS`: ``general``, ``logging``) and
    otherwise honours the snapshot verbatim, deliberately, so an injected
    test config is never overwritten by the developer's real one. The
    synthetic dict carried neither marker AND no `[chat_defaults]` /
    `[console]` section, so every
    `ConsoleSessionController._build_console_turn_execution_context` read
    defaults no matter what the test had persisted -- the mechanism behind
    the vacuous pass in `test_send_proceeds_when_auto_retrieve_fails`
    (task-15210), where a deliberately exploding retrieval backend was never
    once called.

    So do what the shipping app does: `app.py` assigns
    ``self.app_config = load_settings()``. That is hermetic here because the
    root conftest re-points ``TLDW_CONFIG_PATH``/``HOME``/``XDG_*`` at a
    per-test sandbox before anything imports the app, so `load_settings()`
    reads the *test's* config file -- the same file
    `save_setting_to_cli_config` writes and `get_cli_setting` reads. One
    config, one truth, for the app snapshot and for every later refresh.

    The dict is deliberately NOT deep-copied off `load_settings()`'s cache:
    production shares that object too (`load_settings` returns the cached
    dict by reference), so a test that mutates `app.app_config` sees the
    same follow-on behaviour it would see in the real app -- and a seam that
    refreshes via `load_settings()` gets back the very same object, synthetic
    overrides included. The one seam in that armour: a `save_setting_*` call
    made after the app is built invalidates the cache, so the next refresh
    re-reads the file and sees the sandbox's own `[tldw_api] base_url`
    (`http://127.0.0.1:8000`, the same endpoint spelled differently) rather
    than the override below.

    Args:
        first_run_setup_completed: When True (the default, see
            `_build_test_app`) mark the first-run wizard already completed so
            it does not auto-offer over the screen under test. When False the
            flag is removed, leaving a config that looks genuinely fresh.
        overrides: Extra config sections merged over the loaded config, for a
            caller that wants a value without persisting it. Prefer
            `save_setting_to_cli_config` for anything the app may re-read:
            these overrides live only in the snapshot, exactly like the old
            synthetic dict, so a seam that refreshes from disk will not see
            them.

    Returns:
        The app config dict, sourced from the per-test sandbox config file.
    """
    config = load_settings()
    _deep_merge_into(config, {"tldw_api": {"base_url": "http://localhost:8000"}})
    first_run = config.get("first_run")
    if not isinstance(first_run, dict):
        first_run = {}
        config["first_run"] = first_run
    if first_run_setup_completed:
        first_run["setup_completed"] = True
    else:
        first_run.pop("setup_completed", None)
    if overrides:
        _deep_merge_into(config, overrides)
    return config


def attach_chachanotes_db(app, *, client_id: str = "test-client"):
    """Give a factory-built app the durable ChaChaNotes DB a real send needs.

    TASK-21590. `_build_test_app` patches `get_chachanotes_db_lazy` to `None`,
    so a factory app boots with `chachanotes_db = None`. `ConsoleRuntime.
    ensure_chat_store` then builds the Console store with `persistence=None` --
    and since TASK-19900.3's review-fix commit `56db75386` a durable Console
    turn (any non-ephemeral manual or queued send) *fails closed* unless the
    persistence adapter exposes a callable ``commit_durable_turn``, which a
    `None` adapter cannot. That refusal returns a bare `ConsoleSubmitResult`
    instead of going through `_block`, so it writes no system row and raises no
    toast: 26 mounted send tests kept pressing Send and asserting against a
    transcript production had silently refused to write.

    ``:memory:`` is load-bearing, not a shortcut. `ConsoleRuntime.
    ensure_agent_bridge` deliberately refuses to build an agent bridge for a
    `:memory:` DB ("an in-memory harness still builds neither"), so this
    restores exactly the precondition the send path lost -- a durable-capable
    persistence adapter -- without also switching the caller onto the agent
    loop, which a file-backed DB does and which these tests were never written
    against. A test that is *about* the agent runtime must attach a
    file-backed DB itself.

    Attach BEFORE mounting: the store is built lazily on first use and caches
    its persistence adapter.

    Args:
        app: A `_build_test_app` product, not yet mounted.
        client_id: Client id recorded on the DB's rows.

    Returns:
        The `CharactersRAGDB` now assigned to ``app.chachanotes_db``.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(":memory:", client_id)
    app.chachanotes_db = db
    return db


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


def drain_active_service_patches() -> int:
    """Stop every still-running service patch started by `_build_test_app`.

    Called by the root conftest's autouse cleanup fixture after each test
    (mirroring `drain_created_dirs`), so a patch that must outlive
    `_build_test_app`'s own call (task-1631: `get_subscriptions_db_path`)
    never leaks into the next test.

    Returns:
        The number of patches stopped.
    """
    drained = 0
    while _active_service_patches:
        patcher = _active_service_patches.pop()
        patcher.stop()
        drained += 1
    return drained


def _build_test_app(
    configured_default: str | None = None,
    *,
    first_run_setup_completed: bool = True,
    preserve_profile_admission: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> TldwCli:
    """Build a TldwCli instance with every real I/O seam faked out.

    Args:
        configured_default: Value returned for the ``general.default_tab``
            CLI setting, letting a test choose the app's initial route.
        config_overrides: Config sections merged over the sandbox config for
            this app's snapshot only -- see `build_test_app_config`, and
            prefer `save_setting_to_cli_config` for anything a refreshing
            seam must also see.
        preserve_profile_admission: Defaults to False, which clears
            ``library_new_profile_admission``. `app.py` sets that flag from
            `first_profile_created_this_session()`, and the per-test config
            sandbox creates a profile for every test -- so every factory-built
            app claimed to be a brand-new profile. The Library rail answers
            that claim by composing a compact starter rail (two rows plus
            "Explore all tools") and returning before the search input, the
            Browse/Create sections and the Details disclosure, which made
            rows like ``#library-row-browse-media`` unreachable for the many
            tests written before progressive disclosure existed.
            Three Library test modules had already hand-rolled exactly this
            clearing in local `_build_test_app` wrappers; this hoists it to
            the one factory they all go through. Cleared rather than pinning
            a lifecycle value, so the screen still derives its own state --
            an existing profile with no persisted lifecycle settles to
            Expanded, which is the product's own contract (see
            `test_library_real_existing_config_without_lifecycle_defaults_expanded`).
            Pass ``True`` for a test that is *about* new-profile admission.
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
        initialisers were all patched for the duration of ``__init__`` --
        except ``get_subscriptions_db_path``, which stays patched for the
        rest of the test (see the comment where it is started, below).
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

    # task-1631: started (not entered via the `with ExitStack()` below) and
    # left running -- `LocalWatchlistsService.db_factory` (wired inside
    # `TldwCli._wire_watchlists_and_notifications_services`) is a lambda that
    # re-resolves `get_subscriptions_db_path()` fresh on every call, not once
    # at construction, so any call made after this function returns -- i.e.
    # every call the running screen makes -- must still see this same patch,
    # not the real, unpatched fallback. The eager, init-time consumers
    # (`subscriptions_db` / `WatchlistProjection` / `watchlist_bundle_service`,
    # all built while this patch is live either way) and the lazy
    # `db_factory` therefore now agree on one on-disk file for the app's
    # whole life. `drain_active_service_patches` (called by the root
    # conftest's autouse teardown) stops it once the test ends.
    subscriptions_patcher = patch(
        "tldw_chatbook.app.get_subscriptions_db_path",
        return_value=user_data_dir / "subscriptions.sqlite",
    )
    subscriptions_patcher.start()
    _active_service_patches.append(subscriptions_patcher)

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app._publish_runtime_policy_projection(context.state)
        return context

    # NOTE (task-15270): this stays synthetic while `app_config` above is
    # now the real sandbox config, so the two can disagree -- but only for
    # `tldw_chatbook.app`'s own reads, and only during `__init__` (this patch
    # is scoped to the ExitStack below). Every other module's
    # `get_cli_setting` already reads the sandbox config file, i.e. the same
    # source as `app_config`. Making this one honest too is a separate,
    # larger change: `app.py` reads `splash_screen.enabled`,
    # `general.default_theme`, the scheduling toggles and ~25 more through
    # it, all of which ship enabled-by-default in the config template, so
    # every app-building test would start booting splash screens and
    # background schedulers.
    def fake_cli_setting(_section, _key=None, default=None):
        if (
            _section == "general"
            and _key == "default_tab"
            and configured_default is not None
        ):
            return configured_default
        return default

    fake_app_config = build_test_app_config(
        first_run_setup_completed=first_run_setup_completed,
        overrides=config_overrides,
    )

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
                "tldw_chatbook.Video_Generation.video_store.get_user_data_dir",
                return_value=user_data_dir,
            ),
            patch(
                "tldw_chatbook.Video_Generation.video_store.get_video_store_policy",
                return_value=SimpleNamespace(
                    retention="session",
                    retention_ttl_hours=24,
                    max_store_mb=2048,
                ),
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
        app = TldwCli()
        # PR-3 Task 4: the Library RAG answer worker runs a real provider
        # call automatically once a rag-mode retrieval settles -- no button
        # of its own. `LibraryScreen._library_rag_answer_chat_kwargs` treats
        # a present-but-not-callable `library_rag_answer_chat` as "generation
        # disabled", so setting it to None here keeps every pilot that never
        # opted in off the network (the default, when the attribute is
        # absent entirely, is the real `chat_api_call` -- which is what the
        # shipping app must use). A test that wants generation assigns its
        # own fake callable.
        app.library_rag_answer_chat = None
        # See `preserve_profile_admission` above: the config sandbox creates a
        # profile per test, so this flag is True for every factory-built app
        # and the Library rail answers it with the compact starter rail.
        if not preserve_profile_admission:
            app.library_new_profile_admission = False
        return app
