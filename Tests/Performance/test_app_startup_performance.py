"""Performance guardrails for startup and import-time behavior."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


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
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
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
            "tldw_chatbook.UI.MediaWindow_v2",
            "tldw_chatbook.Utils.Splash_Screens.classic.glitch_reveal",
            "tldw_chatbook.Utils.Splash_Screens.tech.code_scroll",
        )
        print(json.dumps({"loaded": [name for name in guards if name in sys.modules]}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []


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

    fake_app = SimpleNamespace(
        set_timer=Mock(),
        _schedule_footer_status_updates=Mock(),
        _start_deferred_audio_service_initialization=Mock(),
        schedule_media_cleanup=Mock(),
        citation_artifact_ownership_coordinator=SimpleNamespace(writes_enabled=enabled),
        _reconcile_citation_artifact_ownership=reconcile,
        _create_deferred_startup_task=capture,
    )

    TldwCli._schedule_deferred_startup_work(fake_app)

    assert len(scheduled) == expected_tasks
    if scheduled:
        assert scheduled[0][1] == "deferred_citation_artifact_reconciliation"


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

    fake_app = SimpleNamespace(
        set_timer=Mock(),
        _schedule_footer_status_updates=Mock(),
        _start_deferred_audio_service_initialization=Mock(),
        schedule_media_cleanup=Mock(),
        citation_artifact_ownership_coordinator=None,
        citation_legacy_migration_service=SimpleNamespace(
            writes_enabled=enabled,
            ready=enabled,
        ),
        _migrate_legacy_citations_idle_unit=migrate,
        _create_deferred_startup_task=capture,
    )

    TldwCli._schedule_deferred_startup_work(fake_app)

    assert len(scheduled) == expected_tasks
    if scheduled:
        assert scheduled[0][1] == "deferred_legacy_citation_migration"


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
        def ready(self) -> bool:
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


@pytest.mark.asyncio
async def test_ui_ready_before_nonessential_startup_services_finish(
    monkeypatch,
) -> None:
    """Optional audio/DB/cleanup startup work should not gate initial UI readiness."""

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Utils.db_status_manager import DBStatusManager
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
    from tldw_chatbook.app import TldwCli

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
    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", test_cli_setting)

    app = _build_test_app()

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
        finally:
            release_optional_work.set()
            await pilot.pause(0.05)


@pytest.mark.asyncio
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
