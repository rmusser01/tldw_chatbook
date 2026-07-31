"""
Root conftest.py for shared test fixtures and configuration.
This file provides common fixtures used across the test suite.
"""

import os
import shutil
import tempfile
from pathlib import Path

_TEST_CONFIG_ROOT_ENV = "TLDW_TEST_CONFIG_ROOT"
_TEST_CONFIG_OWNER_ENV = "TLDW_TEST_CONFIG_ROOT_OWNER"
_SANDBOXED_ENV_NAMES = (
    "HOME",
    "USERPROFILE",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "TLDW_CONFIG_PATH",
    _TEST_CONFIG_ROOT_ENV,
    _TEST_CONFIG_OWNER_ENV,
)
_PREVIOUS_TEST_ENV = {name: os.environ.get(name) for name in _SANDBOXED_ENV_NAMES}
_existing_test_config_root = os.environ.get(_TEST_CONFIG_ROOT_ENV)
# Under pytest-xdist the controller creates the sandbox root and workers
# inherit it via the env; give each worker its own subtree so concurrent
# workers never share a config/home/data dir (task-1453). The name guard keeps
# re-entrant loads (Tests/UI/conftest.py, subprocess children) from suffixing
# twice — the suffixed path is republished to the env below. The controller
# owns the unsuffixed root and its sessionfinish rmtree removes the worker
# subtrees with it.
# The worker id is only ever xdist's own "gw<N>"; the strict pattern makes the
# env-derived path join traversal-proof without pulling app-level path
# validation into this pre-sys.path bootstrap (an id that fails the pattern is
# ignored, falling back to the shared root).
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER and not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", _XDIST_WORKER):
    _XDIST_WORKER = None
if _existing_test_config_root:
    _BOOTSTRAP_CONFIG_ROOT = Path(_existing_test_config_root)
    if _XDIST_WORKER and _BOOTSTRAP_CONFIG_ROOT.name != _XDIST_WORKER:
        _BOOTSTRAP_CONFIG_ROOT = _BOOTSTRAP_CONFIG_ROOT / _XDIST_WORKER
        _BOOTSTRAP_CONFIG_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _OWNS_BOOTSTRAP_CONFIG_ROOT = False
else:
    _BOOTSTRAP_CONFIG_ROOT = Path(tempfile.mkdtemp(prefix="tldw_test_config_"))
    _OWNS_BOOTSTRAP_CONFIG_ROOT = True
_BOOTSTRAP_CONFIG_ROOT = _BOOTSTRAP_CONFIG_ROOT.resolve(strict=True)
os.environ[_TEST_CONFIG_ROOT_ENV] = str(_BOOTSTRAP_CONFIG_ROOT)
if _OWNS_BOOTSTRAP_CONFIG_ROOT:
    os.environ[_TEST_CONFIG_OWNER_ENV] = str(Path(__file__).resolve())
_BOOTSTRAP_DATA_HOME = _BOOTSTRAP_CONFIG_ROOT / "data"
_BOOTSTRAP_CONFIG_PATH = _BOOTSTRAP_CONFIG_ROOT / "config" / "config.toml"
_BOOTSTRAP_HOME = _BOOTSTRAP_CONFIG_ROOT / "home"
_BOOTSTRAP_DATA_HOME.mkdir(parents=True, mode=0o700, exist_ok=True)
_BOOTSTRAP_CONFIG_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
_BOOTSTRAP_HOME.mkdir(parents=True, mode=0o700, exist_ok=True)
os.environ["HOME"] = str(_BOOTSTRAP_HOME)
os.environ["USERPROFILE"] = str(_BOOTSTRAP_HOME)
os.environ["XDG_DATA_HOME"] = str(_BOOTSTRAP_DATA_HOME)
os.environ["XDG_CONFIG_HOME"] = str(_BOOTSTRAP_CONFIG_PATH.parent)
os.environ["TLDW_CONFIG_PATH"] = str(_BOOTSTRAP_CONFIG_PATH)

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from loguru import logger  # noqa: E402
import asyncio  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import gc  # noqa: E402
from typing import Iterator  # noqa: E402
import warnings  # noqa: E402

# Add project root to Python path for consistent imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hypothesis: no per-example deadline (TASK-1260).
#
# Several property tests do real work per example -- `test_safe_paths_always_validate`
# creates a TemporaryDirectory plus up to four directories, and the DB property
# suites open SQLite connections. Hypothesis' default deadline is 200ms per
# example, which a loaded machine crosses on work that is not actually slow:
# this repo routinely runs 10+ concurrent pytest processes from parallel agents.
#
# The resulting failure is indistinguishable from a real regression at the moment
# it appears, and attributing one instance cost five runs across two worktrees.
# A deadline that fails a property which *holds* is measuring the machine, not
# the code -- so it is disabled rather than merely raised. Do not "tighten this
# back up" as an apparent improvement; timing belongs in benchmarks, not in
# correctness properties.
try:  # pragma: no cover - hypothesis is a test-only dependency
    from hypothesis import HealthCheck, settings as _hypothesis_settings

    # Example counts are env-scaled (task-1452): 'dev' keeps routine runs fast,
    # CI sets TLDW_HYPOTHESIS_PROFILE=ci, and the scheduled deep run uses
    # 'thorough' so the reduced dev depth has a compensating control.
    # Hypothesis binds settings.default at DECORATION time, so this profile
    # must be active whenever a test module is imported. Property modules that
    # need extra health-check suppressions register a child profile with
    # parent=settings.default and MUST restore this one at end of module —
    # a leaked load_profile() silently reconfigures every later-imported
    # module's unannotated @given tests (the pre-task-1452 state).
    _HYPOTHESIS_SCALES = {
        "dev": {"max_examples": 25, "stateful_step_count": 20},
        "ci": {"max_examples": 50, "stateful_step_count": 30},
        "thorough": {"max_examples": 300, "stateful_step_count": 100},
    }
    _pt_profile = os.environ.get("TLDW_HYPOTHESIS_PROFILE", "dev")
    if _pt_profile not in _HYPOTHESIS_SCALES:
        # A typo'd profile silently running at dev depth is exactly the kind of
        # quiet configuration rot this suite has been burned by — warn loudly
        # (pytest surfaces it in the warnings summary) but do not brick every
        # local run over it.
        warnings.warn(
            f"Unknown TLDW_HYPOTHESIS_PROFILE={_pt_profile!r}; expected one of "
            f"{sorted(_HYPOTHESIS_SCALES)} — falling back to 'dev' scale",
            UserWarning,
            stacklevel=1,
        )
        _pt_profile = "dev"
    _pt_scale = _HYPOTHESIS_SCALES[_pt_profile]

    _hypothesis_settings.register_profile(
        "tldw",
        deadline=None,
        # `too_slow` fires for the same reason the deadline does: machine load,
        # not a slow strategy.
        suppress_health_check=[HealthCheck.too_slow],
        **_pt_scale,
    )
    _hypothesis_settings.load_profile("tldw")
except ImportError:
    pass

# Protect against stdout/stderr being closed during testing
# This can happen with certain test runners or when tests manipulate file descriptors
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout.fileno()
    except (ValueError, OSError):
        # stdout is closed or invalid, replace with a safe alternative
        import io

        sys.stdout = io.StringIO()

if hasattr(sys.stderr, "fileno"):
    try:
        sys.stderr.fileno()
    except (ValueError, OSError):
        # stderr is closed or invalid, replace with a safe alternative
        import io

        sys.stderr = io.StringIO()


# ========== Path and File System Fixtures ==========


@pytest.fixture
def isolated_temp_dir():
    """Create an isolated temporary directory that's always cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="tldw_test_")
    temp_path = Path(temp_dir)
    yield temp_path
    # Ensure cleanup even if test fails
    if temp_path.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_file(isolated_temp_dir):
    """Create a temporary file within an isolated directory."""

    def _create_temp_file(name="test_file", suffix=".txt", content=""):
        file_path = isolated_temp_dir / f"{name}{suffix}"
        file_path.write_text(content)
        return file_path

    return _create_temp_file


# ========== Database Fixtures ==========


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def temp_db_path(isolated_temp_dir):
    """Provide a path for a temporary database file."""
    return isolated_temp_dir / "test_database.db"


# ========== Cleanup and Isolation Fixtures ==========


@pytest.fixture(autouse=True)
def restore_sys_path():
    """Automatically restore sys.path after each test."""
    original_path = sys.path.copy()
    yield
    sys.path[:] = original_path


@pytest.fixture(autouse=True)
def cleanup_loguru_handlers():
    """Automatically cleanup loguru handlers after each test to prevent file descriptor leaks."""
    from loguru import logger

    # Store the initial handler IDs
    initial_handlers = list(logger._core.handlers.keys())

    yield

    # Remove any handlers added during the test
    current_handlers = list(logger._core.handlers.keys())
    for handler_id in current_handlers:
        if handler_id not in initial_handlers:
            try:
                logger.remove(handler_id)
            except (ValueError, KeyError):
                # Handler might already be removed
                pass


# Full gc passes after EVERY test cost real wall-clock at suite scale: the old
# version of this fixture ran TWO gc.collect() per test, ~23,000 full-heap
# collections per run with torch/transformers-sized heaps (2026-07-30 audit,
# driver #3). The FD-leak incidents that motivated it (loguru handlers, fds
# under Textual's redirected streams) are guarded by cleanup_loguru_handlers
# above and the fd_leak_sentinel below; periodic collection plus the
# requires_cleanup marker covers the rest. TLDW_TEST_GC_EVERY=1 restores
# per-test collection as an escape hatch.
def _env_int(name: str, default: int) -> int:
    """Parse an integer environment knob without letting a typo abort the run.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or malformed.

    Returns:
        The parsed integer, or ``default`` (with a UserWarning) when the value
        is not a valid integer — a malformed escape-hatch value must degrade to
        the default, not kill conftest import for the whole suite.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not an integer; using default {default}",
            UserWarning,
            stacklevel=2,
        )
        return default


_GC_EVERY = max(1, _env_int("TLDW_TEST_GC_EVERY", 25))
_gc_test_counter = 0

# Directories whose tests mount Textual apps (run_test() call-site census from
# the 2026-07-30 audit). A Textual App is a reference CYCLE that only
# gc.collect() reclaims, and an uncollected app from the previous test
# interferes with the next app-mounting test — with periodic-only collection
# the victim rotates with heap state (task-1468: a 10-test batch failed a
# DIFFERENT UI test on consecutive runs, and passed 10/10 with
# TLDW_TEST_GC_EVERY=1). These dirs keep per-test collection; everything else
# stays periodic, which is where the task-1454 win lives.
_APP_MOUNTING_DIR_PARTS = (
    f"{os.sep}Tests{os.sep}UI{os.sep}",
    f"{os.sep}Tests{os.sep}Widgets{os.sep}",
    f"{os.sep}Tests{os.sep}Watchlists{os.sep}",
    f"{os.sep}Tests{os.sep}Skills{os.sep}",
    f"{os.sep}Tests{os.sep}Library{os.sep}",
    f"{os.sep}Tests{os.sep}Event_Handlers{os.sep}",
    f"{os.sep}Tests{os.sep}integration{os.sep}",
    f"{os.sep}Tests{os.sep}Chat{os.sep}",
)


@pytest.fixture(autouse=True)
def cleanup_file_descriptors(request: pytest.FixtureRequest) -> Iterator[None]:
    """Garbage-collect leaked file objects periodically (or per-test on request).

    Tests that genuinely need a collection pass right after they run (e.g. they
    open many files and assert on fd state) mark themselves
    ``@pytest.mark.requires_cleanup``. Tests in app-mounting directories
    (``_APP_MOUNTING_DIR_PARTS``) always collect, so a torn-down Textual app's
    reference cycle never lingers into the next app's lifetime.

    Args:
        request: The pytest fixture request, used to read the test's markers
            and path.

    Yields:
        None. Collection (if due) happens in teardown, after the test body.
    """
    yield

    global _gc_test_counter
    _gc_test_counter += 1
    _node_path = str(getattr(request.node, "path", "") or "")
    if (
        request.node.get_closest_marker("requires_cleanup")
        or any(part in _node_path for part in _APP_MOUNTING_DIR_PARTS)
        or _gc_test_counter % _GC_EVERY == 0
    ):
        # Suppress ResourceWarnings emitted for unclosed files reclaimed here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()


@pytest.fixture(scope="session", autouse=True)
def fd_leak_sentinel() -> Iterator[None]:
    """Warn when the session's open-fd count grows past a leak threshold.

    Replacement leak detection for the per-test gc.collect() this file used to
    do: cheap (two directory listings per session), and unlike the gc pass it
    produces an actionable signal instead of silently papering over leaks.
    Warn-only for now; threshold via TLDW_TEST_FD_GROWTH_LIMIT. The signal is
    a UserWarning (never in default ignore filters, unlike ResourceWarning)
    plus a stderr line, so it survives any warning-filter configuration.

    Yields:
        None. The fd count comparison happens at session teardown.
    """
    fd_dir = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"

    def _count_fds():
        try:
            return len(os.listdir(fd_dir))
        except OSError:
            return None

    limit = _env_int("TLDW_TEST_FD_GROWTH_LIMIT", 200)
    start = _count_fds()
    yield
    if start is None:
        return
    end = _count_fds()
    if end is not None and end - start > limit:
        message = (
            f"open file descriptors grew by {end - start} over the test session "
            f"(start={start}, end={end}, limit={limit}) — possible fd leak; "
            "bisect with TLDW_TEST_GC_EVERY=1 and mark offending tests "
            "@pytest.mark.requires_cleanup"
        )
        # UserWarning (not ResourceWarning): ResourceWarning sits in Python's
        # default ignore filters, so the sentinel's only signal could vanish
        # under filter configurations that don't re-enable it. The stderr echo
        # survives even -W ignore.
        warnings.warn(message, UserWarning, stacklevel=0)
        print(f"[fd_leak_sentinel] {message}", file=sys.stderr)


# ========== Async Cleanup Fixtures ==========


@pytest.fixture
def cleanup_async_tasks():
    """Cleanup any pending async tasks after async tests.

    Note: This fixture should be explicitly used by async tests that need cleanup,
    not applied automatically to all tests.
    """
    import sys

    yield

    # Only cleanup if we're in an async context with a running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, nothing to clean up
        return

    # Get all tasks in the current loop
    if sys.version_info >= (3, 9):
        # Use current_task() to exclude the cleanup task itself
        current = asyncio.current_task(loop)
        tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if task != current and not task.done()
        ]
    else:
        # Fallback for older Python versions
        try:
            current = (
                asyncio.current_task()
                if hasattr(asyncio, "current_task")
                else asyncio.Task.current_task()
            )
            tasks = [
                task
                for task in asyncio.all_tasks(loop)
                if task != current and not task.done()
            ]
        except RuntimeError:
            return

    # Cancel and cleanup tasks, specifically looking for RichLogProcessor
    for task in tasks:
        # Special handling for RichLogProcessor tasks
        if task.get_name() == "RichLogProcessor":
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, RuntimeError):
                pass
        else:
            task.cancel()

    # Don't wait for cancellation as it might cause issues
    # The event loop will handle cleanup when it shuts down


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case.

    This fixture is recognized by pytest-asyncio and helps ensure
    each async test gets a fresh event loop.
    """
    loop = asyncio.new_event_loop()
    yield loop
    # Cleanup
    try:
        _cancel_all_tasks(loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    except RuntimeError:
        pass


def _cancel_all_tasks(loop):
    """Cancel all tasks in the given event loop."""
    import sys

    # Get all tasks for this loop - API changed in Python 3.9
    if sys.version_info >= (3, 9):
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    else:
        # For Python < 3.9
        tasks = [task for task in asyncio.Task.all_tasks(loop) if not task.done()]

    if not tasks:
        return

    for task in tasks:
        # Special handling for RichLogProcessor to ensure clean shutdown
        if hasattr(task, "get_name") and task.get_name() == "RichLogProcessor":
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, RuntimeError):
                pass
        else:
            task.cancel()

    # Give tasks a chance to cleanup
    try:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    except RuntimeError:
        # Loop might be closed
        pass


@pytest.fixture
def clean_environment():
    """Provide a clean environment and restore it after test."""
    original_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(original_env)


# ========== Test Markers ==========


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external resources"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may use files/network"
    )
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")
    config.addinivalue_line(
        "markers", "requires_cleanup: Tests that need special cleanup"
    )
    config.addinivalue_line("markers", "asyncio: Async tests using asyncio")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Restore caller config variables and remove only an owned sandbox.

    In xdist workers the pre-suffix snapshot points at the controller's SHARED
    sandbox, so restoring it here would aim HOME/XDG/TLDW_* back at shared
    directories for the late-shutdown window (atexit hooks) — exactly the
    isolation this sandboxing exists to provide. Workers are about to exit and
    own nothing; skip both the restore and the (already owner-gated) cleanup.
    """
    if _XDIST_WORKER:
        return
    for name, previous in _PREVIOUS_TEST_ENV.items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    if _OWNS_BOOTSTRAP_CONFIG_ROOT:
        shutil.rmtree(_BOOTSTRAP_CONFIG_ROOT, ignore_errors=True)


# ========== Async Support ==========


@pytest.fixture
def anyio_backend():
    """Backend for anyio async tests."""
    return "asyncio"


# ========== Test Environment Isolation ==========


def _close_database_instance(db_instance):
    """Best-effort close for a database object cached by application config."""
    close_db = getattr(db_instance, "close", None)
    if not callable(close_db):
        return
    try:
        close_db()
    except Exception as exc:
        logger.warning(f"Failed to close cached test database: {exc}")


def _reset_config_database_instances(config_module):
    """Close and clear config.py's lazy database singletons."""
    for db_name in ("chachanotes_db", "prompts_db", "media_db"):
        _close_database_instance(getattr(config_module, db_name, None))
        setattr(config_module, db_name, None)


def _shutdown_prompts_interop_if_loaded():
    """Reset the prompt singleton without importing its module into every test."""
    prompts_interop = sys.modules.get("tldw_chatbook.Prompt_Management.Prompts_Interop")
    if prompts_interop is not None and prompts_interop.is_initialized():
        prompts_interop.shutdown_interop()


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch, tmp_path):
    """Automatically isolate test environment to prevent production data access.

    This fixture:
    - Sets TLDW_TEST_MODE environment variable
    - Redirects all data directories to a temporary location
    - Prevents database initialization during import
    """
    # Set test mode
    monkeypatch.setenv("TLDW_TEST_MODE", "1")

    # Create a unique test data directory
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    # Common paths that need isolation
    monkeypatch.setenv("XDG_DATA_HOME", str(test_data_dir))
    test_config_dir = test_data_dir / "config"
    test_config_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(test_config_dir))
    test_home_dir = test_data_dir / "home"
    test_home_dir.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(test_home_dir))
    monkeypatch.setenv(
        "TLDW_CONFIG_PATH",
        str(test_config_dir / "config.toml"),
    )

    # Clear lazy database and prompt singletons before each test so no
    # connection survives after its per-test temporary directory is removed.
    # ``get_user_data_dir`` resolves HOME at call time, so the environment
    # patches above are the path-isolation authority.
    try:
        from tldw_chatbook import config

        _reset_config_database_instances(config)
        _shutdown_prompts_interop_if_loaded()
    except ImportError:
        config = None

    # NOTE (task-519): this used to also try to
    # `monkeypatch.setattr(config, "get_data_dir", ...)`, but `config` has no
    # `get_data_dir` attribute -- that patch was a silent no-op. It's removed
    # rather than fixed because it's no longer needed: `get_user_data_dir()`'s
    # default-dir fallback now resolves HOME/XDG_DATA_HOME at CALL time (see
    # `config._default_base_data_dir`), so the HOME/XDG_DATA_HOME env patches
    # above are sufficient on their own.

    # Pre-arm SP2b's first-run-import once-flag so the RAG ingestion module's
    # real (no-longer-pytest-gated, see task-519) first-run wiring never fires
    # organically inside an unrelated test and creates a real
    # "imported_settings" RAG profile under the (now-isolated, but still
    # real-filesystem) data dir. Tests that specifically want to exercise
    # `_maybe_run_first_run_import` reset this flag themselves (see
    # `Tests/RAG/test_first_run_import.py`).
    #
    # This must NOT `import tldw_chatbook.RAG_Search.ingestion_indexing` itself
    # (review finding on task-519/PR #845): that would drag the RAG stack into
    # every single test in the suite, autouse, even ones that never touch RAG.
    # Instead only arm the flag if the module is ALREADY in sys.modules (i.e.
    # some earlier-collected test already imported it) -- the first-run wiring
    # lives INSIDE that module, so if it isn't imported it cannot fire. If a
    # test imports it later in its own body, task-519's call-time HOME
    # resolution already isolates any first-run write under the HOME/
    # XDG_DATA_HOME patched above, so this pre-arm is belt-and-braces for
    # modules that happen to already be loaded, not a correctness requirement.
    ii = sys.modules.get("tldw_chatbook.RAG_Search.ingestion_indexing")
    if ii is not None:
        monkeypatch.setattr(ii, "_first_run_import_attempted", True, raising=False)

    yield test_data_dir

    if config is not None:
        _shutdown_prompts_interop_if_loaded()
        _reset_config_database_instances(config)


# ========== Test Data Fixtures ==========


@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing."""
    return """
    This is a sample text for testing purposes.
    It contains multiple lines and paragraphs.
    
    This is the second paragraph with some **markdown** formatting.
    It also includes [links](http://example.com) and other elements.
    """


@pytest.fixture
def sample_json_data():
    """Provide sample JSON data for testing."""
    return {
        "title": "Test Document",
        "content": "Test content",
        "metadata": {
            "author": "Test Author",
            "date": "2025-01-01",
            "tags": ["test", "sample"],
        },
    }


# ========== App Cleanup Fixtures ==========


@pytest_asyncio.fixture
async def app_with_cleanup():
    """Create a TldwCli app instance with proper cleanup.

    This fixture ensures the RichLogHandler is properly stopped
    before the event loop closes, preventing the "Task was destroyed
    but it is pending!" error.
    """
    from tldw_chatbook.app import TldwCli

    app = TldwCli()

    yield app

    # Ensure proper cleanup
    try:
        # Stop RichLogHandler if it exists
        if hasattr(app, "_rich_log_handler") and app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            # Note: RichLogHandler still uses standard logging for Textual integration
            import logging

            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()

        # Call shutdown methods
        if hasattr(app, "on_shutdown_request"):
            await app.on_shutdown_request()

        if hasattr(app, "on_unmount"):
            await app.on_unmount()

    except Exception as e:
        # Log but don't fail the test
        logger.debug(f"Error during app cleanup: {e}")


# ========== Performance and Timing Fixtures ==========


@pytest.fixture
def benchmark_timer():
    """Simple timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time

    return Timer


# ========== Skill Trust Service Fixtures ==========
#
# Promoted from ``Tests/Skills/conftest.py`` (task-7 of the skills-script-
# execution SDD plan): ``Tests/Library/test_skill_script_grant_panel.py``
# needs ``trust_service_with_skill`` too, and pytest fixture discovery only
# walks a test file's own directory and its ANCESTORS -- a sibling
# directory's ``conftest.py`` is never visible. Living here instead of being
# duplicated means both ``Tests/Skills/`` and ``Tests/Library/`` share the
# exact same fixture (and its future edits), rather than two copies quietly
# drifting apart.


@pytest.fixture
def make_trust_service(tmp_path):
    """Return a factory that builds `SkillTrustService` instances sharing one store.

    Args:
        tmp_path: Pytest-provided temporary directory fixture.

    Returns:
        A zero-argument callable that constructs a new `SkillTrustService`
        bound to the same on-disk `skills_dir`/`trust_dir`, so repeated calls
        simulate a fresh process re-reading persisted state.
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore,
        SkillTrustStore,
        default_trust_store_dir,
    )

    # Derived from the real accessors/objects rather than re-spelled
    # "skills"/"trust" literals (TASK-866): `LocalSkillsService` computes
    # its own `skills_dir` from `_SKILLS_DIRNAME`, and
    # `default_trust_store_dir()` is the same function `app.py` calls to
    # build the live `SkillTrustStore`. If either constant's name ever
    # changed, a hardcoded literal here would silently keep matching
    # nothing -- the exact class of drift this task closes.
    skills_dir = LocalSkillsService(store_dir=tmp_path).skills_dir
    trust_dir = default_trust_store_dir(tmp_path)
    skills_dir.mkdir(exist_ok=True, parents=True)
    trust_dir.mkdir(exist_ok=True, parents=True)

    def _make() -> "SkillTrustService":
        marker_path = trust_dir / "marker.json"
        return SkillTrustService(
            skills_dir=skills_dir,
            trust_store=SkillTrustStore(
                store_dir=trust_dir,
                marker_store=FileSkillTrustGenerationMarkerStore(
                    marker_path, store_dir=marker_path.parent
                ),
            ),
        )

    return _make


@pytest.fixture
def trust_service_with_skill(make_trust_service):
    """Return a `SkillTrustService` with one on-disk demo skill (with a script).

    Args:
        make_trust_service: Factory fixture for building trust-service instances.

    Returns:
        A `(service, skill_name)` tuple where `skill_name` names a skill
        directory containing a `SKILL.md` and a `scripts/hello.py`.
    """
    service = make_trust_service()
    name = "demo-skill"
    skill_dir = service.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text(
        "print('hello')", encoding="utf-8"
    )
    return service, name


# ========== Pytest Configuration ==========


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="Run tests requiring optional dependencies",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-optional"):
        # The marker actually used by the suite is `optional` (registered in
        # pyproject); the old gate keyed on `optional_deps`, which no test has
        # ever carried, so it selected nothing (task-1457).
        skip_optional = pytest.mark.skip(reason="Need --run-optional option to run")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_optional)
