"""Regression tests for collection-time application path isolation."""

from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Prompt_Management import Prompts_Interop
from Tests.textual_test_utils import _close_app_database_instances


class _ClosingDatabase:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def close_connection(self) -> None:
        self.closed = True


_COLLECTION_CONFIG_DB = _ClosingDatabase()
_COLLECTION_PROMPT_DB = _ClosingDatabase()


@pytest.fixture(scope="module", autouse=True)
def _seed_collection_time_database_singletons():
    """Simulate DBs cached by modules imported before per-test fixtures."""
    config.chachanotes_db = _COLLECTION_CONFIG_DB
    Prompts_Interop._db_instance = _COLLECTION_PROMPT_DB
    Prompts_Interop._db_path_global = "collection-time.db"
    yield
    config.chachanotes_db = None
    Prompts_Interop._db_instance = None
    Prompts_Interop._db_path_global = None


def test_data_paths_follow_per_test_root_after_collection_import(
    isolate_test_environment: Path,
) -> None:
    """Import-time defaults must not escape the active pytest sandbox."""
    user_data_dir = config.get_user_data_dir()

    assert user_data_dir.is_relative_to(isolate_test_environment)


def test_collection_time_database_singletons_are_closed_and_cleared() -> None:
    """Fixture setup must discard handles created before a test starts."""
    assert _COLLECTION_CONFIG_DB.closed is True
    assert _COLLECTION_PROMPT_DB.closed is True
    assert config.chachanotes_db is None
    assert Prompts_Interop.is_initialized() is False


def test_full_app_database_cleanup_closes_every_handle_and_reports_failures() -> None:
    """Full-app pilot cleanup must be observable rather than best effort."""

    class _FailingDatabase(_ClosingDatabase):
        def close(self) -> None:
            self.closed = True
            raise RuntimeError("close failed")

    class _AppWithDatabases:
        chachanotes_db = _ClosingDatabase()
        prompts_db = _FailingDatabase()
        media_db = _ClosingDatabase()

    app = _AppWithDatabases()

    with pytest.raises(RuntimeError, match="prompts_db"):
        _close_app_database_instances(app)

    assert app.chachanotes_db.closed is True
    assert app.prompts_db.closed is True
    assert app.media_db.closed is True
