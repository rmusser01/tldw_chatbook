"""
Pytest configuration for UI tests.

This file provides shared fixtures and configuration for all UI tests.
"""

import os
import shutil
import tempfile
from pathlib import Path

_TEST_CONFIG_ROOT_ENV = "TLDW_TEST_CONFIG_ROOT"
_TEST_CONFIG_OWNER_ENV = "TLDW_TEST_CONFIG_ROOT_OWNER"
_existing_test_config_root = os.environ.get(_TEST_CONFIG_ROOT_ENV)
# Per-xdist-worker sandbox subtree; see Tests/conftest.py for the rationale
# (task-1453). Needed here too for runs rooted at Tests/UI, where the root
# conftest is not loaded.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER and not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", _XDIST_WORKER):
    _XDIST_WORKER = None  # never join an unexpected id into a path
if _existing_test_config_root:
    _BOOTSTRAP_CONFIG_ROOT = Path(_existing_test_config_root)
    if _XDIST_WORKER and _BOOTSTRAP_CONFIG_ROOT.name != _XDIST_WORKER:
        _BOOTSTRAP_CONFIG_ROOT = _BOOTSTRAP_CONFIG_ROOT / _XDIST_WORKER
        _BOOTSTRAP_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ[_TEST_CONFIG_ROOT_ENV] = str(_BOOTSTRAP_CONFIG_ROOT)
    _OWNS_BOOTSTRAP_CONFIG_ROOT = False
else:
    _BOOTSTRAP_CONFIG_ROOT = Path(tempfile.mkdtemp(prefix="tldw_test_config_"))
    os.environ[_TEST_CONFIG_ROOT_ENV] = str(_BOOTSTRAP_CONFIG_ROOT)
    os.environ[_TEST_CONFIG_OWNER_ENV] = str(Path(__file__).resolve())
    _OWNS_BOOTSTRAP_CONFIG_ROOT = True
_BOOTSTRAP_CONFIG_PATH = _BOOTSTRAP_CONFIG_ROOT / "config" / "config.toml"
_BOOTSTRAP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ["TLDW_CONFIG_PATH"] = str(_BOOTSTRAP_CONFIG_PATH)

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from typing import TypeVar  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from textual.app import App  # noqa: E402
from textual.widget import Widget  # noqa: E402

# Import test utilities (fixture re-exports for this nested pytest root).
from Tests.textual_test_utils import app_pilot, widget_pilot  # noqa: F401,E402
from Tests.conftest import isolate_test_environment  # noqa: F401,E402
from Tests.textual_test_harness import (  # noqa: F401,E402
    IsolatedWidgetTestApp,
    TestApp,
    enhanced_app_pilot,
    isolated_widget_pilot,
)

# Re-export the canonical scratch_config fixture from Tests/Internal_Prompts.conftest.
# See Tests/RAG/conftest.py for the collision rationale (plain import, not
# pytest_plugins, sidesteps a duplicate-plugin-registration error when both
# test directories are collected in the same session).
from Tests.Internal_Prompts.conftest import scratch_config  # noqa: F401,E402

# Type variables
W = TypeVar("W", bound=Widget)
A = TypeVar("A", bound=App)


def pytest_sessionfinish(session, exitstatus):
    """Remove the module-load config sandbox created by this conftest."""
    if not _OWNS_BOOTSTRAP_CONFIG_ROOT:
        return
    if os.environ.get("TLDW_CONFIG_PATH") == str(_BOOTSTRAP_CONFIG_PATH):
        os.environ.pop("TLDW_CONFIG_PATH", None)
    if os.environ.get(_TEST_CONFIG_ROOT_ENV) == str(_BOOTSTRAP_CONFIG_ROOT):
        os.environ.pop(_TEST_CONFIG_ROOT_ENV, None)
        os.environ.pop(_TEST_CONFIG_OWNER_ENV, None)
    shutil.rmtree(_BOOTSTRAP_CONFIG_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for async tests."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh(monkeypatch):
    """Keep UI full-app boots off the ADR-020 catalog network seam (task-16198).

    Incident: the knowledge_entry suite went red on pristine dev with the
    egress guard's teardown error naming ``104.18.3.115:443`` and
    ``104.18.2.115:443`` — openrouter.ai's two A records. The one keyless
    path to that host is the startup catalog refresh:
    ``TldwCli._refresh_model_catalogs`` →
    ``LocalLLMProviderCatalogService.refresh_stale_configured_providers``,
    which exempts OpenRouter from the no-credentials skip ("OpenRouter's
    catalog is public") and issues a real
    ``GET https://openrouter.ai/api/v1/models``. The refresh is consent-gated
    and the per-test sandbox config defaults consent off, so a green run is
    the norm — but any leak of consented settings into the process (a shared
    ``TLDW_TEST_CONFIG_ROOT`` between concurrent sessions, config-cache
    pollution) turns every full-app UI boot into live egress, timed by the
    refresh worker racing test teardown. Tests/ProductionApp/conftest.py and
    Tests/RuntimePolicy/test_runtime_policy_full_app.py already pin this same
    seam shut; this fixture closes the remaining full-app surface (Tests/UI)
    so no settings content can re-open it.

    Only ``TldwCli._refresh_model_catalogs`` is patched: stub hosts that bind
    their own ``_refresh_model_catalogs`` instance attribute (e.g. the
    phase1 first-run schedule tests) and direct service-level tests are
    unaffected. A test that needs the real seam monkeypatches the method
    back within its own scope.
    """

    async def _offline_refresh(_app) -> None:
        return None

    monkeypatch.setattr(
        "tldw_chatbook.app.TldwCli._refresh_model_catalogs",
        _offline_refresh,
    )


@pytest.fixture(autouse=True)
def _no_tiktoken_bpe_download(monkeypatch):
    """Keep UI token counting off tiktoken's BPE download seam (TASK-21590).

    Incident: repairing the Console send harness let 16 mounted send tests
    reach `ConsoleProviderGateway.prepare_chat_request` for the first time —
    `prepare_provider_request` → `_account_categories` → `_count_wire` →
    `count_console_messages_tokens` → `token_counter.estimate_tokens` →
    `count_tokens_tiktoken` → `get_tiktoken_encoding`. On a cold cache
    `tiktoken.get_encoding` fetches its BPE blobs from
    ``openaipublic.blob.core.windows.net`` over HTTPS, so the egress guard
    recorded six blocked connects per test and failed each one at teardown.
    The tests themselves passed; only the guard saw it. The old, broken
    harness never got past the durable-acceptance gate, so it never reached
    this seam at all — which is why the failure is invisible on dev.

    ``get_tiktoken_encoding`` is the single chokepoint: every tiktoken use in
    the Console send path funnels through it (`console_cost_tracker` and
    `console_session_settings` only reach tiktoken via
    ``count_tokens_messages``), and it resolves as a module global at call
    time, so patching it here covers callers that imported
    ``count_tokens_tiktoken``/``estimate_tokens`` by name.

    ``TIKTOKEN_AVAILABLE`` is what selects the tier, and it is forced off
    here rather than only stubbing the encoding, because those are NOT the
    same accounting. With the flag left True, `estimate_tokens` still enters
    `count_tokens_tiktoken`, whose no-encoding fallback is a bare
    ``int(len(text) * 0.25)``: no CJK weighting, no headroom, and no
    non-empty floor. Measured against `gpt-4`/`openai` on this venv:

    ==========================  ====  =============  ==============
    text                        real  encoding=None  tiktoken off
    ==========================  ====  =============  ==============
    ``"hi"``                       1              0               1
    a repeated ASCII sentence     31             33              40
    repeated CJK                  50             17              84
    ==========================  ====  =============  ==============

    The middle column is a tier no install runs, it undercounts CJK by ~3x
    against the real tokenizer, and it re-introduces the zero-for-short-text
    truncation that `_chars_estimate`'s ``max(1, ...)`` exists to prevent.
    The right-hand column IS what a default install does, since tiktoken is
    not a base dependency (task-2526). The encoding stub stays as a second
    line of defence for anything that calls it directly.

    `_ESTIMATE_CACHE` is cleared on both sides of the test: it is
    process-global and keyed by ``(model, provider, len, hash(text))`` with
    no tokenizer identity in the key, so without this a value computed under
    the real tokenizer elsewhere in the session is served here (and one
    computed here leaks the other way). pytest-randomly shuffles the run
    order, so "Tests/Chat happens to run first" is not a defence.

    No test's LOGICAL coverage changes, only its network access, and the
    result stops depending on whether this machine happens to have a warm
    ``$TMPDIR/data-gym-cache`` (which the HOME sandbox does not redirect) or
    on what estimated the same string earlier in the session. A test that
    needs the real encoding monkeypatches it back within its own scope,
    exactly as with `_disable_model_catalog_refresh` above.
    """
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setattr(token_counter, "TIKTOKEN_AVAILABLE", False)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.token_counter.get_tiktoken_encoding",
        lambda _model: None,
    )
    token_counter.clear_estimate_cache()
    yield
    token_counter.clear_estimate_cache()


@pytest_asyncio.fixture
async def mock_app_config():
    """Provide a standard mock app configuration for tests."""
    return {
        "api_endpoints": {
            "openai": {"api_key": "test-key", "endpoint": "https://api.openai.com/v1"},
            "anthropic": {
                "api_key": "test-key",
                "endpoint": "https://api.anthropic.com/v1",
            },
        },
        "chat_defaults": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-3.5-turbo",
        },
        "ui_settings": {"theme": "dark", "font_size": 14},
    }


@pytest_asyncio.fixture
async def mock_app_instance(mock_app_config):
    """Create a mock app instance with standard configuration."""
    from unittest.mock import MagicMock, AsyncMock

    app = MagicMock()
    app.app_config = mock_app_config
    app.current_chat_is_ephemeral = False
    app.loguru_logger = MagicMock()

    # Mock common methods
    app.notify = MagicMock()
    app.push_screen = AsyncMock()
    app.pop_screen = AsyncMock()
    app.run_worker = AsyncMock()
    app.call_from_thread = MagicMock()
    app.post_message = MagicMock()

    # Mock query methods
    app.query = MagicMock()
    app.query_one = MagicMock()

    return app


@pytest_asyncio.fixture
async def ui_test_app():
    """
    Fixture that creates a test app for UI testing.

    Usage:
        async def test_my_ui(ui_test_app):
            async with ui_test_app() as pilot:
                # Test UI interactions
    """
    created_apps = []

    @asynccontextmanager
    async def _create_app():
        from tldw_chatbook.app import TldwCli

        # Create app with test configuration
        app = TldwCli()
        created_apps.append(app)

        # Override with test config
        app.app_config = {
            "api_endpoints": {"openai": {"api_key": "test"}},
            "chat_defaults": {"temperature": 0.7},
        }

        async with app.run_test() as pilot:
            yield pilot

    yield _create_app

    # Cleanup
    for app in created_apps:
        if hasattr(app, "_driver"):
            try:
                await app.exit()
            except Exception:
                pass


@pytest.fixture
def assert_tooltip():
    """Fixture providing tooltip assertion helper."""

    def _assert_tooltip(widget, expected_tooltip):
        """Assert widget has expected tooltip."""
        assert hasattr(widget, "tooltip"), f"Widget {widget} has no tooltip attribute"
        assert widget.tooltip == expected_tooltip, (
            f"Expected tooltip '{expected_tooltip}', got '{widget.tooltip}'"
        )

    return _assert_tooltip


@pytest.fixture
def assert_widget_state():
    """Fixture providing widget state assertion helpers."""

    class WidgetStateAssertions:
        @staticmethod
        def is_visible(widget):
            """Assert widget is visible."""
            assert widget.styles.display != "none", f"Widget {widget} is not visible"

        @staticmethod
        def is_hidden(widget):
            """Assert widget is hidden."""
            assert widget.styles.display == "none", (
                f"Widget {widget} is visible but should be hidden"
            )

        @staticmethod
        def is_enabled(widget):
            """Assert widget is enabled."""
            if hasattr(widget, "disabled"):
                assert not widget.disabled, f"Widget {widget} is disabled"

        @staticmethod
        def is_disabled(widget):
            """Assert widget is disabled."""
            if hasattr(widget, "disabled"):
                assert widget.disabled, f"Widget {widget} is enabled"

        @staticmethod
        def has_class(widget, class_name):
            """Assert widget has CSS class."""
            assert class_name in widget.classes, (
                f"Widget {widget} missing class '{class_name}'. Has: {list(widget.classes)}"
            )

        @staticmethod
        def not_has_class(widget, class_name):
            """Assert widget does not have CSS class."""
            assert class_name not in widget.classes, (
                f"Widget {widget} should not have class '{class_name}'"
            )

    return WidgetStateAssertions()


@pytest.fixture
def wait_for_condition():
    """Fixture providing async wait helper."""

    async def _wait_for(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while True:
            if condition_func():
                return True

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s")

            await asyncio.sleep(interval)

    return _wait_for


# Markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "ui: mark test as a UI test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_display: mark test as requiring display"
    )


# Shared test data
SAMPLE_CHAT_MESSAGES = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "Can you help me with Python?"},
    {
        "role": "assistant",
        "content": "Of course! What would you like to know about Python?",
    },
]

SAMPLE_CHARACTER = {
    "name": "Test Character",
    "description": "A helpful test character",
    "personality": "Friendly and knowledgeable",
    "scenario": "Testing environment",
}

SAMPLE_NOTE = {
    "title": "Test Note",
    "content": "This is a test note content.",
    "tags": ["test", "sample"],
    "created_at": "2024-01-01T00:00:00",
}


# Notes-specific fixtures


@pytest.fixture
def mock_notes_service():
    """Create a mock notes service with common methods."""
    from unittest.mock import Mock

    service = Mock()

    # Setup default return values
    service.list_notes = Mock(
        return_value=[
            {
                "id": 1,
                "title": "Note 1",
                "content": "Content 1",
                "version": 1,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "keywords": "",
            },
            {
                "id": 2,
                "title": "Note 2",
                "content": "Content 2",
                "version": 1,
                "created_at": "2024-01-02T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
                "keywords": "test",
            },
        ]
    )

    service.get_note_by_id = Mock(
        return_value={
            "id": 1,
            "title": "Test Note",
            "content": "Test content for the note",
            "version": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
    )

    service.add_note = Mock(return_value=3)  # Returns new note ID
    service.update_note = Mock(return_value=True)  # Returns success
    service.delete_note = Mock(return_value=True)  # Returns success

    return service


@pytest.fixture
def mock_app_with_notes(mock_notes_service):
    """Create a mock app instance with notes service."""
    from unittest.mock import Mock

    app = Mock()
    app.notes_service = mock_notes_service
    app.notify = Mock()
    app.push_screen = Mock()
    app.pop_screen = Mock()
    app.screen_stack = []

    # Add query methods
    app.query_one = Mock()
    app.query = Mock(return_value=[])

    return app


@pytest.fixture
def tmp_media_db(tmp_path):
    """A real, on-disk, per-test `MediaDatabase` (repo convention: real
    SQLite, never a mock, for DB tests -- Task 8, meeting speaker rename)."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(str(tmp_path / "media_test.sqlite"), "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def meeting_folder_media_item(tmp_path, tmp_media_db):
    """Build a meeting folder (`meeting.json` + `transcript.jsonl`) and the
    Library `Media` row that points at its `mixed.wav`, exactly as a real
    finished meeting recording leaves them (Task 8's Interfaces section).

    Returns a factory: ``factory(names: dict, segments: list[tuple[str,
    str]]) -> (media_id, folder)``, where each segment tuple is
    ``(speaker_id, text)``.
    """
    import json as _json

    def _factory(*, names: dict, segments: list[tuple[str, str]]):
        folder = tmp_path / f"meeting-{len(segments)}-{id(segments)}"
        folder.mkdir()
        (folder / "mixed.wav").write_bytes(b"")
        (folder / "meeting.json").write_text(_json.dumps({"speaker_names": dict(names)}))
        lines = []
        for seq, (speaker_id, text) in enumerate(segments):
            lines.append(
                _json.dumps(
                    {
                        "seq": seq,
                        "t_audio_start": float(seq),
                        "t_audio_end": float(seq) + 1.0,
                        "t_wall_start": 0.0,
                        "t_wall_end": 0.0,
                        "label": "others",
                        "text": text,
                        "speaker_id": speaker_id,
                    }
                )
            )
        (folder / "transcript.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
        media_id, _uuid, _msg = tmp_media_db.add_media_with_keywords(
            url=str(folder / "mixed.wav"),
            title="Test Meeting",
            media_type="audio",
            content="placeholder",
            overwrite=False,
        )
        assert media_id is not None, _msg
        return media_id, folder

    return _factory


@pytest.fixture
def sample_notes_data():
    """Provide sample notes data for tests."""
    return [
        {
            "id": 1,
            "title": "Daily Notes",
            "content": "Today I learned about Textual testing.",
            "version": 2,
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T14:30:00",
            "keywords": "daily, learning",
        },
        {
            "id": 2,
            "title": "Project Ideas",
            "content": "Build a better notes app with Textual.",
            "version": 1,
            "created_at": "2024-01-14T09:00:00",
            "updated_at": "2024-01-14T09:00:00",
            "keywords": "project, ideas",
        },
        {
            "id": 3,
            "title": "Meeting Notes",
            "content": "Discussed the new UI refactoring approach.",
            "version": 3,
            "created_at": "2024-01-13T15:00:00",
            "updated_at": "2024-01-15T16:00:00",
            "keywords": "meeting, refactoring",
        },
    ]
