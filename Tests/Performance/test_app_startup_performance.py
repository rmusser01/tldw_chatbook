"""Performance guardrails for startup and import-time behavior."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Tests.app_module_patches import set_app_global

REPO_ROOT = Path(__file__).resolve().parents[2]


async def _wait_until(
    condition,
    *,
    pause,
    timeout_seconds: float = 3.0,
    interval_seconds: float = 0.05,
) -> None:
    """Wait for a test-app condition without sleeping the host process."""

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _run_isolated_python(
    tmp_path: Path,
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with isolated Chatbook config/data directories."""

    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": os.pathsep.join(
            (str(REPO_ROOT), str(REPO_ROOT / "packages/tldw_profile_core/src"))
        ),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_optional_deps_import_does_not_eagerly_check_embeddings(tmp_path: Path) -> None:
    """Importing optional_deps should not import heavyweight embeddings packages."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.Utils.optional_deps  # noqa: F401

        guards = ("torch", "transformers", "chromadb", "sentence_transformers")
        print(json.dumps({"loaded": [name for name in guards if name in sys.modules]}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []
    assert "Checking embeddings dependencies early" not in result.stderr


def test_optional_deps_eager_env_still_initializes_dependency_checks(
    tmp_path: Path,
) -> None:
    """Explicit eager dependency mode should still run dependency initialization."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json

        import tldw_chatbook.Utils.optional_deps as optional_deps

        print(json.dumps({"initialized": optional_deps._initialized}))
        """,
        extra_env={"TLDW_EAGER_DEPENDENCY_CHECK": "true"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["initialized"] is True
    assert (
        "Eager dependency checking enabled via TLDW_EAGER_DEPENDENCY_CHECK"
        in result.stderr
    )


def test_app_import_does_not_load_legacy_feature_windows(tmp_path: Path) -> None:
    """Plain app import should not load heavy destination/legacy windows."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.app  # noqa: F401

        guards = (
            "tldw_chatbook.UI.STTS_Window",
            "tldw_chatbook.UI.Splash_Screens.classic.glitch_reveal",
            "tldw_chatbook.UI.Splash_Screens.tech.code_scroll",
        )
        print(json.dumps({"loaded": [name for name in guards if name in sys.modules]}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []


def test_mcp_connection_health_import_keeps_native_cleanup_deferred(
    tmp_path: Path,
) -> None:
    """Ordinary MCP session checks keep native plugin root custody deferred."""
    result = _run_isolated_python(
        tmp_path,
        """
        import sys
        from types import SimpleNamespace
        from Tests import network_guard, real_profile_guard

        network_guard.install()
        real_profile_guard.install()
        assert "tldw_chatbook.MCP.connection_ownership" not in sys.modules
        from tldw_chatbook.MCP.connection_ownership import ConnectionOwnership

        assert not ConnectionOwnership._session_unavailable(SimpleNamespace())
        assert ConnectionOwnership._session_unavailable(SimpleNamespace(_closed=True))
        assert "tldw_chatbook.Plugins.data_cleanup" not in sys.modules
        """,
    )

    assert result.returncode == 0, result.stderr


def test_tool_pack_facade_export_is_lazy(tmp_path: Path) -> None:
    """Package symbols must not pull service owners in until first access."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.Tool_Packs as tool_packs

        before = "tldw_chatbook.Tool_Packs.service" in sys.modules
        facade = tool_packs.ToolPackService
        after = "tldw_chatbook.Tool_Packs.service" in sys.modules
        print(json.dumps({"before": before, "after": after, "name": facade.__name__}))
        """,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "before": False,
        "after": True,
        "name": "ToolPackService",
    }


def test_tool_pack_implementation_imports_only_in_deferred_worker(
    tmp_path: Path,
) -> None:
    """App import stays cheap; the post-ready worker owns composition and I/O."""

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import tldw_chatbook.app as app_module
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore
        from tldw_chatbook.Workspaces.registry_service import DeferredWorkspaceToolProfileGuard

        guarded = (
            "tldw_chatbook.Tool_Packs.service",
            "tldw_chatbook.Tool_Packs.export",
            "tldw_chatbook.Tool_Packs.publication",
            "tldw_chatbook.Tool_Packs.importer",
            "tldw_chatbook.Tool_Packs.activation",
            "tldw_chatbook.Tool_Packs.receipt_store",
            "tldw_chatbook.Tool_Packs.removal",
        )
        before = [name for name in guarded if name in sys.modules]
        root = Path(__import__("os").environ["XDG_DATA_HOME"])
        # TASK-33011: the tool-pack composition moved to app_service_wiring.
        import tldw_chatbook.app_service_wiring as wiring_module
        wiring_module.get_user_data_dir = lambda: root

        class Registry:
            def __init__(self, guard): self.guard = guard
            def list_workspaces(self, *, include_archived=False): return ()
            def get_workspace(self, workspace_id): return None
            def attach_tool_profile_guard(self, guard): self.guard = guard
            @property
            def tool_profile_guard(self): return self.guard

        bootstrap = DeferredWorkspaceToolProfileGuard()
        registry = Registry(bootstrap)
        fake = SimpleNamespace(
            unified_mcp_service=SimpleNamespace(
                permission_store=MCPPermissionStore(root / "permissions.json")
            ),
            local_mcp_control_service=object(),
            workspace_registry_service=registry,
            _tool_pack_guard_bootstrap=bootstrap,
            tool_pack_service=None,
            tool_pack_service_unavailable_reason="starting",
            tool_pack_receipt_reconciliation_unavailable_reason="not_run",
        )
        fake.call_from_thread = lambda callback, *args: callback(*args)
        fake._mark_tool_pack_service_unavailable = lambda category: (
            app_module.TldwCli._mark_tool_pack_service_unavailable(fake, category)
        )
        fake._attach_tool_pack_service = lambda service, owner, guard_bootstrap: (
            app_module.TldwCli._attach_tool_pack_service(
                fake, service, owner, guard_bootstrap
            )
        )
        fake._record_tool_pack_receipt_reconciliation = lambda service, category: (
            app_module.TldwCli._record_tool_pack_receipt_reconciliation(
                fake, service, category
            )
        )

        app_module.TldwCli._compose_tool_pack_service_off_thread(fake)
        after = [name for name in guarded if name in sys.modules]
        print(json.dumps({
            "before": before,
            "after": after,
            "attached": fake.tool_pack_service is not None and registry.guard is not None,
            "root": (
                str(fake.tool_pack_service.receipt_root)
                if fake.tool_pack_service is not None else None
            ),
            "unavailable_reason": fake.tool_pack_service_unavailable_reason,
            "reconciliation_reason": fake.tool_pack_receipt_reconciliation_unavailable_reason,
            "guard_active": bootstrap.active_guard is not None,
            "guard_unchanged": registry.guard is bootstrap,
        }))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"] == []
    assert payload["after"] == [
        "tldw_chatbook.Tool_Packs.service",
        "tldw_chatbook.Tool_Packs.export",
        "tldw_chatbook.Tool_Packs.publication",
        "tldw_chatbook.Tool_Packs.importer",
        "tldw_chatbook.Tool_Packs.activation",
        "tldw_chatbook.Tool_Packs.receipt_store",
        "tldw_chatbook.Tool_Packs.removal",
    ]
    if os.name == "nt":
        # Native Windows receipt durability remains unqualified (ADR-107).
        assert payload["attached"] is False
        assert payload["root"] is None
        assert payload["unavailable_reason"] == "composition_unavailable"
        assert payload["reconciliation_reason"] == "not_run"
        assert payload["guard_active"] is False
        assert payload["guard_unchanged"] is True
        return
    assert payload["attached"] is True
    assert payload["root"] == str(tmp_path / "data" / "tool_pack_receipts")


@pytest.mark.parametrize(("enabled", "expected_tasks"), [(False, 0), (True, 1)])
def test_citation_artifact_reconciliation_is_deferred_and_policy_gated(
    enabled: bool,
    expected_tasks: int,
) -> None:
    """Ownership recovery starts after readiness and only under the write switch."""

    from tldw_chatbook.app import TldwCli

    scheduled: list[tuple[object, str]] = []

    def capture(coroutine, *, name: str):
        scheduled.append((coroutine, name))
        coroutine.close()

    async def reconcile() -> None:
        return None

    # A `Mock(spec=TldwCli)` rather than a hand-listed `SimpleNamespace`
    # (TASK-21566). The namespace had to name every attribute the method
    # touches, so each time deferred startup grew a step these tests died on
    # the new one -- most recently `run_worker`, added by task-21106 to move
    # Actor Pack recovery off the construction path. `spec=` tracks the class:
    # a genuinely new dependency is satisfied, while a typo or an attribute
    # that does not exist on TldwCli still raises. Only the seams under test
    # are wired for real.
    fake_app = Mock(spec=TldwCli)
    fake_app.citation_artifact_ownership_coordinator = SimpleNamespace(
        writes_enabled=enabled
    )
    fake_app._reconcile_citation_artifact_ownership = reconcile
    fake_app._create_deferred_startup_task = capture

    TldwCli._schedule_deferred_startup_work(fake_app)

    # Assert on this task by NAME, not by position or by the total. Deferred
    # startup schedules unrelated work too (a subscription reconcile lands
    # first, unconditionally), so a count or an index pins the shape of the
    # whole method instead of the policy gate this test is about -- and breaks
    # every time an unrelated step is added.
    citation_tasks = [
        name
        for _, name in scheduled
        if name == "deferred_citation_artifact_reconciliation"
    ]
    assert len(citation_tasks) == expected_tasks


@pytest.mark.parametrize(("enabled", "expected_tasks"), [(False, 0), (True, 1)])
def test_legacy_citation_migration_is_deferred_and_policy_gated(
    enabled: bool,
    expected_tasks: int,
) -> None:
    """One bounded migration idle unit starts only after the write switch."""

    from tldw_chatbook.app import TldwCli

    scheduled: list[tuple[object, str]] = []

    def capture(coroutine, *, name: str):
        scheduled.append((coroutine, name))
        coroutine.close()

    async def migrate() -> None:
        return None

    # See the sibling test above for why this is a spec'd Mock and why the
    # assertion is by name rather than by count (TASK-21566).
    fake_app = Mock(spec=TldwCli)
    fake_app.citation_artifact_ownership_coordinator = None
    fake_app.citation_legacy_migration_service = SimpleNamespace(
        writes_enabled=enabled,
        ready=enabled,
    )
    fake_app._migrate_legacy_citations_idle_unit = migrate
    fake_app._create_deferred_startup_task = capture

    TldwCli._schedule_deferred_startup_work(fake_app)

    migration_tasks = [
        name for _, name in scheduled if name == "deferred_legacy_citation_migration"
    ]
    assert len(migration_tasks) == expected_tasks


@pytest.mark.asyncio
async def test_deferred_migration_drains_bounded_batches_and_multiple_conversations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Deferred work yields between 100-message units until every conversation ends."""

    import tldw_chatbook.app as app_module
    from Tests.Chat.test_citation_legacy_migration import (
        CODEC,
        _record,
        _repository,
        _write_sidecar,
    )
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
        LegacyMigrationState,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(
        tmp_path / "deferred-migration.sqlite",
        client_id="deferred-migration-test",
    )
    try:
        first_conversation = db.add_conversation(
            {
                "id": "a-conversation",
                "title": "First migration",
                "character_id": None,
            }
        )
        first_ids = [f"first-message-{ordinal:04d}" for ordinal in range(205)]
        for message_id in first_ids:
            db.add_message(
                {
                    "id": message_id,
                    "conversation_id": first_conversation,
                    "sender": "assistant",
                    "content": "Legacy answer [1].",
                }
            )
        second_conversation = db.add_conversation(
            {
                "id": "b-conversation",
                "title": "Second migration",
                "character_id": None,
            }
        )
        second_ids = ["second-message-0000"]
        db.add_message(
            {
                "id": second_ids[0],
                "conversation_id": second_conversation,
                "sender": "assistant",
                "content": "Legacy answer [1].",
            }
        )
        sidecar = tmp_path / "chat_rag_context.json"
        _write_sidecar(sidecar, first_conversation, first_ids)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["conversations"][second_conversation] = {
            message_id: {
                **_record(message_id),
                "conversation_id": second_conversation,
            }
            for message_id in second_ids
        }
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        migration = CitationLegacyMigrationService(
            db=db,
            repository=_repository(db),
            sidecar_path=sidecar,
            fingerprint_codec=CODEC,
        )
        processed: list[int] = []
        original_migrate = migration.migrate_idle_unit

        def record_migration_unit():
            result = original_migrate()
            processed.append(result.processed_messages)
            return result

        migration.migrate_idle_unit = record_migration_unit
        real_sleep = asyncio.sleep
        yielded: list[float] = []

        async def record_yield(delay: float) -> None:
            yielded.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(app_module.asyncio, "sleep", record_yield)
        fake_app = SimpleNamespace(
            citation_legacy_migration_service=migration,
            loguru_logger=Mock(),
        )

        await app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)

        assert processed == [100, 100, 5, 1]
        assert yielded == [0, 0, 0]
        assert (
            migration.get_journal(first_conversation).state
            is LegacyMigrationState.COMPLETE
        )
        assert (
            migration.get_journal(second_conversation).state
            is LegacyMigrationState.COMPLETE
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_deferred_migration_is_single_flight(monkeypatch) -> None:
    """Concurrent scheduler calls share one migration driver."""

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat.citation_legacy_migration import (
        LegacyMigrationBatchResult,
        LegacyMigrationState,
    )

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocked_to_thread(function):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return function()

    migration = SimpleNamespace(
        ready=True,
        writes_enabled=True,  # TASK-33620.8: the driver gates on the switch
        migrate_idle_unit=lambda: LegacyMigrationBatchResult(
            state=LegacyMigrationState.COMPLETE
        ),
    )
    fake_app = SimpleNamespace(
        citation_legacy_migration_service=migration,
        loguru_logger=Mock(),
    )
    monkeypatch.setattr(app_module.asyncio, "to_thread", blocked_to_thread)

    first = asyncio.create_task(
        app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)
    )
    await started.wait()
    second = asyncio.create_task(
        app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)
    )
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert calls == 1


@pytest.mark.asyncio
async def test_deferred_migration_retries_exceptions_with_bounded_backoff(
    monkeypatch,
) -> None:
    """A transient worker error gets one bounded retry path instead of a task storm."""

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat.citation_legacy_migration import (
        LegacyMigrationBatchResult,
        LegacyMigrationState,
    )

    calls = 0
    delays: list[float] = []

    def migrate_idle_unit():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return LegacyMigrationBatchResult(state=LegacyMigrationState.COMPLETE)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    fake_app = SimpleNamespace(
        citation_legacy_migration_service=SimpleNamespace(
            ready=True,
            writes_enabled=True,  # TASK-33620.8
            migrate_idle_unit=migrate_idle_unit,
        ),
        loguru_logger=Mock(),
    )
    monkeypatch.setattr(app_module.asyncio, "sleep", record_sleep)

    await app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)

    assert calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_deferred_migration_backs_off_running_guard_failures(
    monkeypatch,
) -> None:
    """A retryable guard result yields bounded backoff instead of a hot loop."""

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat.citation_legacy_migration import (
        LegacyMigrationBatchResult,
        LegacyMigrationState,
    )

    calls = 0
    delays: list[float] = []

    def migrate_idle_unit():
        nonlocal calls
        calls += 1
        return LegacyMigrationBatchResult(
            state=(
                LegacyMigrationState.RUNNING
                if calls < 3
                else LegacyMigrationState.COMPLETE
            ),
            reason_code=("legacy_cutover_guard_failed" if calls < 3 else None),
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    fake_app = SimpleNamespace(
        citation_legacy_migration_service=SimpleNamespace(
            ready=True,
            writes_enabled=True,  # TASK-33620.8
            migrate_idle_unit=migrate_idle_unit,
        ),
        loguru_logger=Mock(),
    )
    monkeypatch.setattr(app_module.asyncio, "sleep", record_sleep)

    await app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)

    assert calls == 3
    assert delays == [1, 2]


@pytest.mark.asyncio
async def test_deferred_migration_isolates_terminal_failures_and_drains_later_work(
    monkeypatch,
) -> None:
    """Malformed conversations do not consume the retry budget for later work."""

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat.citation_legacy_migration import (
        LegacyMigrationBatchResult,
        LegacyMigrationState,
    )

    results = iter(
        (
            LegacyMigrationBatchResult(
                state=LegacyMigrationState.RUNNING,
                reason_code="legacy_cutover_guard_failed",
            ),
            LegacyMigrationBatchResult(
                state=LegacyMigrationState.RUNNING,
                reason_code="legacy_batch_invalid",
            ),
            LegacyMigrationBatchResult(
                state=LegacyMigrationState.RUNNING,
                reason_code="legacy_field_too_large",
            ),
            LegacyMigrationBatchResult(
                state=LegacyMigrationState.RUNNING,
                reason_code="legacy_source_unavailable",
            ),
            LegacyMigrationBatchResult(
                state=LegacyMigrationState.COMPLETE,
                processed_messages=1,
            ),
        )
    )
    calls = 0
    delays: list[float] = []

    def migrate_idle_unit():
        nonlocal calls
        calls += 1
        return next(results)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    logger = Mock()
    fake_app = SimpleNamespace(
        citation_legacy_migration_service=SimpleNamespace(
            ready=True,
            writes_enabled=True,  # TASK-33620.8
            migrate_idle_unit=migrate_idle_unit,
        ),
        loguru_logger=logger,
    )
    monkeypatch.setattr(app_module.asyncio, "sleep", record_sleep)

    await app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)

    assert calls == 5
    assert delays == [1, 0, 0, 0]
    assert [call.args[0] for call in logger.warning.call_args_list] == [
        f"Legacy citation migration retained retry state: reason_code={reason_code!r}"
        for reason_code in (
            "legacy_cutover_guard_failed",
            "legacy_batch_invalid",
            "legacy_field_too_large",
            "legacy_source_unavailable",
        )
    ]


@pytest.mark.asyncio
async def test_deferred_migration_rechecks_disabled_policy_between_units() -> None:
    """Turning off canonical writes stops the driver before another batch."""

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat.citation_legacy_migration import (
        LegacyMigrationBatchResult,
        LegacyMigrationState,
    )

    class Migration:
        enabled = True
        calls = 0

        @property
        def writes_enabled(self) -> bool:
            # TASK-33620.8: the driver re-checks the canonical-write switch;
            # key readiness is acquired inside each unit (first use).
            return self.enabled

        def migrate_idle_unit(self):
            self.calls += 1
            self.enabled = False
            return LegacyMigrationBatchResult(state=LegacyMigrationState.RUNNING)

    migration = Migration()
    fake_app = SimpleNamespace(
        citation_legacy_migration_service=migration,
        loguru_logger=Mock(),
    )

    await app_module.TldwCli._migrate_legacy_citations_idle_unit(fake_app)

    assert migration.calls == 1


# bootstrap_profile (TASK-33260): these build real config/DB owners, whose
# reads go through the ADR-126 config admission. Under the per-test env
# redirect the admission fails closed with
# RecoveryRequired("raw_source_selection_changed") before the test body runs,
# which masked every assertion below; keeping the collection-time profile lets
# them report real results.
@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_ui_ready_before_nonessential_startup_services_finish(
    monkeypatch,
) -> None:
    """Optional audio/DB/cleanup startup work should not gate initial UI readiness."""

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
    from tldw_chatbook.Utils.db_status_manager import DBStatusManager

    tts_started = asyncio.Event()
    stts_started = asyncio.Event()
    db_size_started = asyncio.Event()
    media_cleanup_started = asyncio.Event()
    release_optional_work = asyncio.Event()

    async def blocked_tts_init(self) -> None:
        tts_started.set()
        await release_optional_work.wait()

    async def blocked_stts_init(self) -> None:
        stts_started.set()
        await release_optional_work.wait()

    async def blocked_db_size_update(self) -> None:
        db_size_started.set()
        await release_optional_work.wait()

    async def blocked_media_cleanup(self) -> None:
        media_cleanup_started.set()
        await release_optional_work.wait()

    def test_cli_setting(section: str, key: str | None = None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "media_cleanup" and key == "enabled":
            return True
        if section == "media_cleanup" and key == "cleanup_on_startup":
            return True
        return default

    monkeypatch.setattr(TTSEventHandler, "initialize_tts", blocked_tts_init)
    monkeypatch.setattr(STTSEventHandler, "initialize_stts", blocked_stts_init)
    monkeypatch.setattr(DBStatusManager, "update_db_sizes", blocked_db_size_update)
    monkeypatch.setattr(TldwCli, "perform_media_cleanup", blocked_media_cleanup)
    set_app_global(monkeypatch, "get_cli_setting", test_cli_setting)

    app = _build_test_app()

    def startup_failure_context():
        # Read retained state only: do not resolve app properties/lazy owners.
        fields = app.__dict__

        def task_state(task):
            if not isinstance(task, asyncio.Task):
                return {"present": task is not None}
            done, cancelled = task.done(), task.cancelled()
            error = task.exception() if done and not cancelled else None
            return {
                "present": True,
                "name": task.get_name()[:96],
                "done": done,
                "cancelled": cancelled,
                "error_class": None if error is None else type(error).__name__,
            }

        children = fields.get("_speech_initialization_children", {})
        context = {
            "ui_ready": fields.get("_ui_ready") is True,
            "tts_initializer_entered": tts_started.is_set(),
            "stts_initializer_entered": stts_started.is_set(),
            "db_size_entered": db_size_started.is_set(),
            "media_cleanup_entered": media_cleanup_started.is_set(),
            "speech_paused": fields.get("_speech_initialization_paused") is True,
            "speech_closed": fields.get("_speech_initialization_closed") is True,
            "speech_deferred": [
                kind
                for kind in ("tts", "stts")
                if kind in fields.get("_speech_initialization_deferred", ())
            ],
            "tasks": {
                "tts_wrapper": task_state(fields.get("_tts_initialization_task")),
                "stts_wrapper": task_state(fields.get("_stts_initialization_task")),
                "tts_child": task_state(children.get("tts")),
                "stts_child": task_state(children.get("stts")),
            },
            "admission_observation": "retained flags/tasks only; allowed result not retained",
        }
        try:
            import inspect
            import math
            import time
            from concurrent.futures import Future as ConcurrentFuture
            from concurrent.futures import ThreadPoolExecutor
            from concurrent.futures import thread as executor_thread
            from functools import partial
            from queue import SimpleQueue
            from types import (
                CellType,
                CodeType,
                CoroutineType,
                FrameType,
                FunctionType,
                MethodType,
                ModuleType,
            )
            from weakref import WeakSet

            def code_identity(function):
                if type(function) is not FunctionType:
                    return "unknown"
                code = function.__code__
                return [
                    code.co_filename[-80:],
                    code.co_qualname[:80],
                    code.co_firstlineno,
                ]

            def callback_identity(callback):
                for _ in range(2):
                    if type(callback) is not partial:
                        break
                    function = callback.func
                    if (
                        type(function) is MethodType
                        and function.__self__ is app
                        and function.__func__ is call_next
                        and callback.args
                    ):
                        callback = callback.args[0]
                    else:
                        callback = function
                if type(callback) is MethodType:
                    return callback.__func__, callback.__self__ is app
                return callback, False

            def native_task_state(task):
                if type(task) is not asyncio.Task:
                    return "unknown"
                return {
                    "state": task._state,
                    "error": (
                        None
                        if task._exception is None
                        else type(task._exception).__name__[:80]
                    ),
                }

            def timer_wait_state(task):
                if type(task) is not asyncio.Task:
                    return {"missing": "unknown; no native task"}

                def task_path(current):
                    path = []
                    coroutine = current._coro
                    timing = {"missing": "unknown; no retained Timer._run numbers"}
                    native_run = inspect.getattr_static(timer_class, "_run", None)
                    timer_code = (
                        native_run.__code__
                        if type(native_run) is FunctionType
                        else None
                    )
                    clock_module = sys.modules.get("textual._time")
                    clock = (
                        vars(clock_module).get("get_time")
                        if type(clock_module) is ModuleType
                        else None
                    )
                    for _ in range(5):
                        if type(coroutine) is not CoroutineType:
                            break
                        frame, code = coroutine.cr_frame, coroutine.cr_code
                        path.append(
                            [
                                code.co_qualname[:72],
                                frame.f_lineno if type(frame) is FrameType else None,
                            ]
                        )
                        if code is timer_code and type(frame) is FrameType:
                            # Only exact native Timer._run numeric fields; no worker locals.
                            numbers = {
                                key: frame.f_locals.get(key)
                                for key in ("start", "next_timer", "count", "_interval")
                            }

                            def valid(value):
                                return (
                                    type(value) in (int, float)
                                    and abs(value) < 1e12
                                    and math.isfinite(value)
                                )

                            if clock is time.perf_counter or clock is time.monotonic:
                                now = clock()
                                timing = {
                                    "age_s": round(now - numbers["start"], 3)
                                    if valid(numbers["start"])
                                    else "unknown",
                                    "late_s": round(now - numbers["next_timer"], 3)
                                    if valid(numbers["next_timer"])
                                    else "unknown",
                                    "count": numbers["count"]
                                    if type(numbers["count"]) is int
                                    and valid(numbers["count"])
                                    else "unknown",
                                    "interval_s": numbers["_interval"]
                                    if valid(numbers["_interval"])
                                    else "unknown",
                                }
                        coroutine = coroutine.cr_await
                    return {
                        "task": native_task_state(current),
                        "code": path,
                        "more_code": type(coroutine) is CoroutineType,
                        "timing": timing,
                    }

                result = task_path(task)
                waiter = task._fut_waiter
                if type(waiter) is asyncio.Task:
                    result["child"] = task_path(waiter)
                    waiter = waiter._fut_waiter
                result["waiter"] = {
                    "type": "Task"
                    if type(waiter) is asyncio.Task
                    else "Future"
                    if type(waiter) is asyncio.Future
                    else "unknown",
                    "state": waiter._state
                    if type(waiter) is asyncio.Task or type(waiter) is asyncio.Future
                    else "unknown",
                }
                source_state = "unknown"
                if type(waiter) is asyncio.Future:
                    chain_module = sys.modules.get("asyncio.futures")
                    chain = (
                        vars(chain_module).get("_chain_future")
                        if type(chain_module) is ModuleType
                        else None
                    )
                    if (
                        type(chain) is FunctionType
                        and chain.__module__ == "asyncio.futures"
                        and chain.__qualname__ == "_chain_future"
                    ):
                        check_code = next(
                            (
                                code
                                for code in chain.__code__.co_consts
                                if type(code) is CodeType
                                and code.co_name == "_call_check_cancel"
                            ),
                            None,
                        )
                        callbacks = waiter._callbacks
                        if (
                            check_code is not None
                            and type(callbacks) is list
                            and "source" in check_code.co_freevars
                        ):
                            source_index = check_code.co_freevars.index("source")
                            for entry in callbacks[:8]:
                                if type(entry) is not tuple or len(entry) != 2:
                                    continue
                                callback = entry[0]
                                if (
                                    type(callback) is not FunctionType
                                    or callback.__code__ is not check_code
                                ):
                                    continue
                                closure = callback.__closure__
                                if (
                                    type(closure) is not tuple
                                    or source_index >= len(closure)
                                    or type(closure[source_index]) is not CellType
                                ):
                                    continue
                                try:
                                    source = closure[source_index].cell_contents
                                except ValueError:
                                    continue
                                if type(source) is ConcurrentFuture:
                                    value = inspect.getattr_static(
                                        source, "_state", None
                                    )
                                    if type(value) is str and value in {
                                        "PENDING",
                                        "RUNNING",
                                        "FINISHED",
                                        "CANCELLED",
                                    }:
                                        source_state = value
                                    break
                result["waiter"]["executor_source_state"] = source_state
                return result

            def executor_state():
                executor = inspect.getattr_static(
                    asyncio.get_running_loop(), "_default_executor", None
                )
                guard_module = sys.modules.get(
                    "tldw_chatbook.Utils.text_selection_crash_guard"
                )
                fresh = (
                    vars(guard_module).get("FreshContextExecutor")
                    if type(guard_module) is ModuleType
                    else None
                )
                trusted = (
                    (ThreadPoolExecutor, fresh)
                    if type(fresh) is type and fresh.__bases__ == (ThreadPoolExecutor,)
                    else (ThreadPoolExecutor,)
                )
                if not any(type(executor) is kind for kind in trusted):
                    return {"missing": "unknown; executor type not trusted"}
                threads = inspect.getattr_static(executor, "_threads", None)
                work_queue = inspect.getattr_static(executor, "_work_queue", None)
                maximum = inspect.getattr_static(executor, "_max_workers", None)
                info = {
                    "kind": type(executor).__name__,
                    "max_workers": maximum if type(maximum) is int else "unknown",
                    "thread_count": len(threads) if type(threads) is set else "unknown",
                    "queue_size_approx": SimpleQueue.qsize(work_queue)
                    if type(work_queue) is SimpleQueue
                    else "unknown",
                    "workers": [],
                    "workers_omitted": 0,
                }
                if type(threads) is not set or len(threads) > 1024:
                    info["missing"] = (
                        "unknown; thread set unavailable or outside observation bound"
                    )
                    return info
                frames = sys._current_frames()
                worker = executor_thread._worker
                if type(worker) is not FunctionType:
                    info["missing"] = "unknown; executor worker code not native"
                    return info
                worker_code = worker.__code__
                info["workers_omitted"] = max(0, len(threads) - 6)
                for thread in tuple(threads)[:6]:
                    if type(thread) is not threading.Thread:
                        continue
                    frame = frames.get(thread.ident)
                    names, matched = [], False
                    for _ in range(24):
                        if type(frame) is not FrameType:
                            break
                        if len(names) < 2:
                            names.append(frame.f_code.co_qualname[:72])
                        matched = matched or frame.f_code is worker_code
                        frame = frame.f_back
                    if not matched:
                        continue
                    info["workers"].append({"code": names})
                info["worker_sample_missing"] = (
                    "unknown; departed or deeper worker frames may not match"
                )
                return info

            manager = fields.get("db_status_manager")
            actual_class = type(app)
            constructor = inspect.getattr_static(actual_class, "__init__", None)
            constructor_manager = (
                constructor.__globals__.get("DBStatusManager")
                if type(constructor) is FunctionType
                else None
            )
            manager_method = inspect.getattr_static(manager, "update_db_sizes", None)
            call_next = inspect.getattr_static(actual_class, "call_next", None)
            callbacks = {
                name: inspect.getattr_static(actual_class, name, None)
                for name in (
                    "_schedule_footer_status_updates",
                    "update_db_sizes",
                    "_start_deferred_audio_service_initialization",
                )
            }
            timer_module = sys.modules.get("textual.timer")
            timer_class = (
                vars(timer_module).get("Timer")
                if type(timer_module) is ModuleType
                else None
            )
            event_module = sys.modules.get("textual.events")
            event_class = (
                vars(event_module).get("Callback")
                if type(event_module) is ModuleType
                else None
            )
            records, seen, omitted = [], 0, 0
            db_timer_wait = {"missing": "unknown; second DB timer not retained"}
            periodic = inspect.getattr_static(manager, "_update_timer", None)
            timers = fields.get("_timers")
            retained = [periodic]
            if type(timers) is WeakSet:
                retained.extend(timer for timer in timers if timer is not periodic)
            for timer in retained:
                if timer_class is None or type(timer) is not timer_class:
                    continue
                callback, bound = callback_identity(
                    inspect.getattr_static(timer, "_callback", None)
                )
                roles = [
                    name for name, function in callbacks.items() if callback is function
                ]
                if timer is not periodic and not roles:
                    continue
                if bound and "update_db_sizes" in roles:
                    try:
                        db_timer_wait = timer_wait_state(
                            inspect.getattr_static(timer, "_task", None)
                        )
                    except Exception as probe_error:  # noqa: BLE001 -- Keep the original observation.
                        db_timer_wait = {
                            "missing": "unknown",
                            "error": type(probe_error).__name__[:80],
                        }
                seen += 1
                if len(records) >= 6:
                    omitted += 1
                    continue
                active = inspect.getattr_static(timer, "_active", None)
                records.append(
                    {
                        "kind": "periodic" if timer is periodic else "timer",
                        "roles": roles,
                        "app_bound": bound,
                        "code": code_identity(callback),
                        "interval": inspect.getattr_static(timer, "_interval", None),
                        "repeat": inspect.getattr_static(timer, "_repeat", None),
                        "active": inspect.getattr_static(active, "_value", "unknown"),
                        "task": native_task_state(
                            inspect.getattr_static(timer, "_task", None)
                        ),
                    }
                )
            pending = fields.get("_next_callbacks")
            if type(pending) is list:
                for message in pending:
                    if event_class is None or type(message) is not event_class:
                        continue
                    callback, bound = callback_identity(
                        inspect.getattr_static(message, "callback", None)
                    )
                    roles = [
                        name
                        for name, function in callbacks.items()
                        if callback is function
                    ]
                    if not roles:
                        continue
                    seen += 1
                    if len(records) >= 6:
                        omitted += 1
                        continue
                    records.append(
                        {
                            "kind": "queued",
                            "roles": roles,
                            "app_bound": bound,
                            "code": code_identity(callback),
                        }
                    )
            pump = fields.get("_task")
            coroutine = pump._coro if type(pump) is asyncio.Task else None
            chain = []
            for _ in range(6):
                if type(coroutine) is not CoroutineType:
                    break
                code = coroutine.cr_code
                chain.append(
                    [
                        code.co_filename[-80:],
                        code.co_qualname[:80],
                        None
                        if coroutine.cr_frame is None
                        else coroutine.cr_frame.f_lineno,
                    ]
                )
                coroutine = coroutine.cr_await
            database_module = sys.modules.get(DBStatusManager.__module__)
            canonical_manager = (
                vars(database_module).get("DBStatusManager")
                if type(database_module) is ModuleType
                else None
            )
            try:
                executor_snapshot = executor_state()
            except Exception as probe_error:  # noqa: BLE001 -- Keep the original observation.
                executor_snapshot = {
                    "missing": "unknown",
                    "error": type(probe_error).__name__[:80],
                }
            db_startup = {
                "missing": "unknown; completed timers or callback batches may no longer be retained",
                "app_class_matches": actual_class is TldwCli,
                "factory_class_matches": _build_test_app.__globals__.get("TldwCli")
                is actual_class,
                "manager_class_matches": (
                    "unknown" if manager is None else type(manager) is DBStatusManager
                ),
                "constructor_class_matches": (
                    "unknown"
                    if constructor_manager is None
                    else constructor_manager is DBStatusManager
                ),
                "canonical_class_matches": (
                    "unknown"
                    if canonical_manager is None
                    else canonical_manager is DBStatusManager
                ),
                "manager_points_to_app": inspect.getattr_static(manager, "app", None)
                is app,
                "manager_method_is_blocked": (
                    "unknown"
                    if manager_method is None
                    else manager_method is blocked_db_size_update
                ),
                "manager_method_is_class_binding": (
                    "unknown"
                    if manager_method is None
                    else manager_method
                    is inspect.getattr_static(type(manager), "update_db_sizes", None)
                ),
                "manager_method": code_identity(manager_method),
                "app_update_method": code_identity(
                    inspect.getattr_static(app, "update_db_sizes", None)
                ),
                "manager_closed": inspect.getattr_static(
                    manager, "_maintenance_closed", "unknown"
                ),
                "periodic_retained": timer_class is not None
                and type(periodic) is timer_class,
                "closing": [
                    fields.get(name, "unknown")
                    for name in ("_closed", "_closing", "_exit")
                ],
                "records": records,
                "matching_records_seen": seen,
                "records_omitted": omitted,
                "timer_set_known": type(timers) is WeakSet,
                "queue_known": type(pending) is list,
                "pump_task": native_task_state(pump),
                "pump_code": chain,
                "pump_more_code": True
                if type(coroutine) is CoroutineType
                else "unknown",
                "db_timer_wait": db_timer_wait,
                "executor": executor_snapshot,
            }

            def oversized():
                return (
                    len(
                        json.dumps(
                            dict(context, db_startup=db_startup), sort_keys=True
                        ).encode("utf-8")
                    )
                    > 4096
                )

            while oversized() and chain:
                chain.pop()
                db_startup["pump_more_code"] = True
            workers = db_startup["executor"].get("workers", [])
            while oversized() and workers:
                workers.pop()
                db_startup["executor"]["workers_omitted"] += 1
            while oversized() and records:
                records.pop()
                db_startup["records_omitted"] += 1
            for observed in (db_timer_wait.get("child", {}), db_timer_wait):
                codes = observed.get("code", [])
                while oversized() and codes:
                    codes.pop()
                    observed["more_code"] = True
            if oversized():
                db_startup = {
                    "missing": "unknown",
                    "truncated": "4096-byte observation limit",
                }
        except Exception as observation_error:  # noqa: BLE001 -- Keep the original diagnostic fields.
            db_startup = {
                "missing": "unknown",
                "observation_error": type(observation_error).__name__[:80],
            }
        context["db_startup"] = db_startup
        return context

    async with app.run_test(size=(120, 36)) as pilot:
        try:
            await _wait_until(lambda: app._ui_ready, pause=pilot.pause)
            await _wait_until(
                lambda: (
                    tts_started.is_set()
                    and stts_started.is_set()
                    and db_size_started.is_set()
                ),
                pause=pilot.pause,
            )
            assert app._ui_ready is True

            await asyncio.sleep(0.2)
            assert media_cleanup_started.is_set() is False
        except AssertionError as error:
            try:
                error.add_note(
                    "optional_startup_diagnostic="
                    + json.dumps(startup_failure_context(), sort_keys=True)
                )
            except Exception as diagnostic_error:  # noqa: BLE001 -- Preserve the original assertion.
                error.add_note(
                    "optional_startup_diagnostic_error="
                    + type(diagnostic_error).__name__
                )
            raise
        finally:
            release_optional_work.set()
            await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_tts_handler_initializes_on_first_use(monkeypatch) -> None:
    """TTS event paths can initialize the handler lazily after startup."""

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    initialized = asyncio.Event()

    async def initialize_tts(self) -> None:
        initialized.set()

    monkeypatch.setattr(TTSEventHandler, "initialize_tts", initialize_tts)

    app = _build_test_app()

    assert app._tts_handler is None
    handler = await app._ensure_tts_handler()

    assert initialized.is_set()
    assert handler is app._tts_handler
    assert handler._profile_service_loader == app._ensure_tts_profile_service
    assert app._tts_profile_service is None


@pytest.mark.asyncio
async def test_tts_profile_open_keeps_helper_startup_off_the_ui_loop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A slow fixed-helper open must not prevent the UI loop from advancing."""

    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_threads: list[int] = []

    def blocked_worker_open(self) -> None:
        worker_threads.append(threading.get_ident())
        worker_started.set()
        assert release_worker.wait(timeout=3)

    monkeypatch.setattr(TTSProfileRepository, "_worker_open", blocked_worker_open)
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite3")
    open_task = asyncio.create_task(repository.open())
    primary_error: BaseException | None = None
    try:
        assert await asyncio.to_thread(worker_started.wait, 3)
        await asyncio.sleep(0)
        assert open_task.done() is False
        assert len(worker_threads) == 1
        assert worker_threads[0] != threading.get_ident()
    except BaseException as error:  # noqa: BLE001 - settle test ownership before redelivering the primary signal
        primary_error = error
    finally:
        release_worker.set()
        try:
            try:
                await open_task
            except BaseException as error:  # noqa: BLE001 - opening failure must not skip repository cleanup
                if primary_error is None:
                    primary_error = error
        finally:
            try:
                await repository.close()
            except BaseException:
                if primary_error is None:
                    raise
    if primary_error is not None:
        raise primary_error


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["assertion", "open", "assertion_and_open"])
async def test_tts_ui_loop_guard_settles_owner_when_its_checks_fail(
    monkeypatch, tmp_path, failure_phase
):
    """Exercise the guard's own failure path with its real executor owner."""
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    assertion_error = AssertionError("injected scheduling assertion")
    open_error = RuntimeError("injected open failure")
    close_error = RuntimeError("injected close diagnostic")
    test_task = asyncio.current_task()
    real_sleep = asyncio.sleep
    real_open, real_close = TTSProfileRepository.open, TTSProfileRepository.close
    owners, open_tasks, closed = [], [], []

    async def observed_open(repository):
        owners.append(repository)
        open_tasks.append(asyncio.current_task())
        await real_open(repository)
        if failure_phase in {"open", "assertion_and_open"}:
            raise open_error

    async def observed_close(repository):
        try:
            await real_close(repository)
        finally:
            closed.append(repository)
        raise close_error

    async def fail_assertion(delay):
        if asyncio.current_task() is test_task and failure_phase != "open":
            raise assertion_error
        await real_sleep(delay)

    monkeypatch.setattr(TTSProfileRepository, "open", observed_open)
    monkeypatch.setattr(TTSProfileRepository, "close", observed_close)
    monkeypatch.setattr(asyncio, "sleep", fail_assertion)
    try:
        with pytest.raises(BaseException) as caught:
            await test_tts_profile_open_keeps_helper_startup_off_the_ui_loop(
                monkeypatch, tmp_path
            )
        expected = open_error if failure_phase == "open" else assertion_error
        assert caught.value is expected
        assert len(owners) == 1 and closed == owners
        assert all(task.done() for task in open_tasks)
        assert owners[0]._executor is None and owners[0]._executor_shutdown
    finally:
        # Even the expected RED against the old guard settles its test owner.
        for task in open_tasks:
            try:
                await task
            except BaseException:  # noqa: BLE001, S110 - consume injected test failure before owner teardown
                pass
        for owner in owners:
            await real_close(owner)


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_stts_handler_initializes_on_first_use(monkeypatch) -> None:
    """S/TT/S command paths can initialize the handler lazily after startup."""

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler

    initialized = asyncio.Event()

    async def initialize_stts(self) -> None:
        initialized.set()

    monkeypatch.setattr(STTSEventHandler, "initialize_stts", initialize_stts)

    app = _build_test_app()

    assert app._stts_handler is None
    handler = await app._ensure_stts_handler()

    assert initialized.is_set()
    assert handler is app._stts_handler


@pytest.mark.asyncio
async def test_windows_timer_delivery_and_stop_do_not_wait_for_default_workers(
    monkeypatch,
):
    from threading import Event as ThreadEvent
    from types import SimpleNamespace

    from textual import timer as timer_module
    from textual.app import App
    from textual.timer import Timer

    from tldw_chatbook.Utils import text_selection_crash_guard as guard

    loop = asyncio.get_running_loop()
    executor = guard.FreshContextExecutor(max_workers=8, thread_name_prefix="asyncio")
    previous = getattr(loop, "_default_executor", None)
    loop.set_default_executor(executor)
    if previous is not None:
        previous.shutdown(wait=False)
    release = ThreadEvent()
    entered = [ThreadEvent() for _ in range(8)]
    native_jobs = []
    proxy_wakes = []
    timers = []
    monkeypatch.setattr(guard, "_windows_timer_pool", None)

    def hold_worker(started):
        started.set()
        release.wait()

    async def windows_executor_sleep(seconds):
        wake = ThreadEvent()
        proxy_wakes.append(wake)
        job = executor.submit(wake.wait, seconds)
        native_jobs.append(job)
        try:
            await asyncio.wrap_future(job)
        except asyncio.CancelledError:
            cancellation = executor.submit(wake.set)
            native_jobs.append(cancellation)
            await asyncio.wrap_future(cancellation)
            raise

    try:
        held_jobs = []
        for started in entered:
            job = executor.submit(hold_worker, started)
            held_jobs.append(job)
            native_jobs.append(job)
        await _wait_until(
            lambda: all(started.is_set() for started in entered), pause=asyncio.sleep
        )
        monkeypatch.setattr(timer_module, "sleep", windows_executor_sleep)
        monkeypatch.setattr(guard, "sys", SimpleNamespace(platform="win32"))
        guard.ThreadWorkerContextGuard().on_load()
        assert getattr(loop, "_default_executor", None) is executor
        assert executor._max_workers == 8
        delivered = asyncio.Event()
        long_delivered = asyncio.Event()
        app = App()
        with app._context():
            short = Timer(app, 0.1, callback=delivered.set, repeat=0)
            timers.append(short)
            short._start()
            await _wait_until(delivered.is_set, pause=asyncio.sleep)
            assert not release.is_set()
            assert all(not job.done() for job in held_jobs)
            long = Timer(app, 86400, callback=long_delivered.set, repeat=0)
            timers.append(long)
            long._start()
            await asyncio.sleep(0)
            await asyncio.wait_for(Timer._stop_all([long]), timeout=3.0)
            assert not long_delivered.is_set()
            assert long._task is None
            assert all(not job.done() for job in held_jobs)
    finally:
        for wake in proxy_wakes:
            wake.set()
        release.set()
        await Timer._stop_all(timers)
        await asyncio.gather(
            *(asyncio.wrap_future(job) for job in native_jobs), return_exceptions=True
        )
        owned_timer_pool = guard._windows_timer_pool
        if owned_timer_pool is not None:
            owned_timer_pool.shutdown(wait=True)
            assert all(not thread.is_alive() for thread in owned_timer_pool._threads)
        assert all(job.done() for job in native_jobs)
        assert all(job.cancelled() or job.exception() is None for job in native_jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["darwin", "linux"])
async def test_non_windows_load_preserves_the_existing_timer_sleep(
    monkeypatch, platform
):
    from types import SimpleNamespace

    from textual import timer as timer_module

    from tldw_chatbook.Utils import text_selection_crash_guard as guard

    existing_sleep = timer_module.sleep
    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform=platform))
    guard.ThreadWorkerContextGuard().on_load()
    assert timer_module.sleep is existing_sleep
