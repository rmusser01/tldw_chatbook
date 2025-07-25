# tests/conftest.py
import pytest
import tempfile
import os
import shutil
from pathlib import Path
import sys

# Import the MediaDatabase from tldw_chatbook instead of tldw_Server_API
try:
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
except ImportError:
    # Fallback to a basic database class if import fails
    class Database:
        def __init__(self, *args, **kwargs): 
            self.db_path = kwargs.get('db_path', ':memory:')
            self.client_id = kwargs.get('client_id', 'test_client')
        def close_connection(self): pass
        def get_sync_log_entries(self, *args, **kwargs): return []
        def execute_query(self, *args, **kwargs):
             class MockCursor:
                  rowcount = 0
                  def fetchone(self): return None
                  def fetchall(self): return []
                  def execute(self, *a, **k): pass
             return MockCursor()
        def transaction(self):
             class MockTransaction:
                  def __enter__(self): return None
                  def __exit__(self, *args): pass
             return MockTransaction()


# --- Database Fixtures ---

@pytest.fixture(scope="function")
def temp_db_path():
    """Creates a temporary directory and returns a unique DB path within it."""
    temp_dir = tempfile.mkdtemp()
    db_file = Path(temp_dir) / "test_db.sqlite"
    yield str(db_file) # Provide the path to the test function
    # Teardown: remove the directory after the test function finishes
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def memory_db_factory():
    """Factory fixture to create in-memory Database instances."""
    created_dbs = []
    def _create_db(client_id="test_client"):
        db = Database(db_path=":memory:", client_id=client_id)
        created_dbs.append(db)
        return db
    yield _create_db
    # Teardown: close connections for all created in-memory DBs
    for db in created_dbs:
        try:
            db.close_connection()
        except: # Ignore errors during cleanup
            pass

@pytest.fixture(scope="function")
def file_db(temp_db_path):
    """Creates a file-based Database instance using a temporary path."""
    db = Database(db_path=temp_db_path, client_id="file_client")
    yield db
    db.close_connection() # Ensure connection is closed

# --- Sync Engine State Fixtures ---

@pytest.fixture(scope="function")
def temp_state_file():
    """Provides a path to a temporary file for sync state."""
    # Use NamedTemporaryFile which handles deletion automatically
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".json") as tf:
        state_path = tf.name
    yield state_path # Provide the path
    # Teardown: Ensure the file is deleted even if the test fails mid-write
    if os.path.exists(state_path):
         os.remove(state_path)