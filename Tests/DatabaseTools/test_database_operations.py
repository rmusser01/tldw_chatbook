# test_database_operations.py
# Unit tests for database tools operations

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import json

# Import the class we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow


class TestDatabaseOperations:
    """Test database operation methods in ToolsSettingsWindow."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        
        # Create a simple test database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO test_table (data) VALUES ('test1'), ('test2'), ('test3')")
        conn.execute("PRAGMA user_version = 7")  # Set schema version
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_window(self):
        """Create a mock ToolsSettingsWindow instance."""
        mock = Mock(spec=ToolsSettingsWindow)
        mock.app_instance = Mock()
        mock.app_instance.notify = Mock()
        mock.config_data = {
            "database": {
                "chachanotes_db_path": "~/test/chachanotes.db",
                "media_db_path": "~/test/media.db",
                "prompts_db_path": "~/test/prompts.db",
                "evals_db_path": "~/test/evals.db",
                "rag_db_path": "~/test/rag.db",
                "subscriptions_db_path": "~/test/subscriptions.db"
            }
        }
        mock.query_one = Mock()
        mock.run_worker = Mock()
        mock.call_from_thread = Mock()
        return mock
    
    def test_get_database_path(self, mock_window):
        """Test _get_database_path returns correct paths."""
        # Call the actual method
        result = ToolsSettingsWindow._get_database_path(
            mock_window, 
            "chachanotes", 
            mock_window.config_data["database"]
        )
        
        assert result == Path("~/test/chachanotes.db").expanduser()
        
        # Test unknown database
        result = ToolsSettingsWindow._get_database_path(
            mock_window,
            "unknown",
            mock_window.config_data["database"]
        )
        assert result is None
    
    def test_get_schema_version(self, mock_window, temp_db_path):
        """Test _get_schema_version returns correct version."""
        version = ToolsSettingsWindow._get_schema_version(mock_window, temp_db_path)
        assert version == 7
        
        # Test non-existent database
        non_existent = Path("/non/existent/db.db")
        version = ToolsSettingsWindow._get_schema_version(mock_window, non_existent)
        assert version is None
    
    def test_format_file_size(self, mock_window):
        """Test _format_file_size returns human-readable sizes."""
        # Test various sizes
        assert ToolsSettingsWindow._format_file_size(mock_window, 512) == "512.0 B"
        assert ToolsSettingsWindow._format_file_size(mock_window, 1024) == "1.0 KB"
        assert ToolsSettingsWindow._format_file_size(mock_window, 1024 * 1024) == "1.0 MB"
        assert ToolsSettingsWindow._format_file_size(mock_window, 1024 * 1024 * 1024) == "1.0 GB"
        assert ToolsSettingsWindow._format_file_size(mock_window, 1024 * 1024 * 1024 * 1024) == "1.0 TB"
    
    @patch('sqlite3.connect')
    def test_vacuum_single_worker(self, mock_connect, mock_window, temp_db_path):
        """Test _vacuum_single_worker executes vacuum."""
        # Setup mock connection
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock the database path
        with patch.object(ToolsSettingsWindow, '_get_database_path', return_value=temp_db_path):
            # Call the worker
            ToolsSettingsWindow._vacuum_single_worker(mock_window, "chachanotes")
        
        # Verify vacuum was called
        mock_conn.execute.assert_called_with("VACUUM")
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
        
        # Verify notification
        assert mock_window.call_from_thread.called
    
    @patch('shutil.copy2')
    @patch('json.dump')
    def test_backup_single_worker(self, mock_json_dump, mock_copy, mock_window, temp_db_path):
        """Test _backup_single_worker creates backup."""
        # Mock the database path
        with patch.object(ToolsSettingsWindow, '_get_database_path', return_value=temp_db_path):
            with patch.object(ToolsSettingsWindow, '_get_schema_version', return_value=7):
                # Call the worker
                ToolsSettingsWindow._backup_single_worker(mock_window, "chachanotes")
        
        # Verify file was copied
        assert mock_copy.called
        
        # Verify metadata was written
        assert mock_json_dump.called
        metadata = mock_json_dump.call_args[0][0]
        assert metadata["database"] == "chachanotes"
        assert metadata["schema_version"] == 7
        assert "backup_time" in metadata
        
        # Verify notification
        assert mock_window.call_from_thread.called
    
    @patch('shutil.copy2')
    def test_restore_single_worker(self, mock_copy, mock_window, temp_db_path):
        """Test _restore_single_worker restores database."""
        backup_path = temp_db_path.parent / "backup.db"
        
        # Create a backup file
        shutil.copy2(temp_db_path, backup_path)
        
        # Mock the database path
        with patch.object(ToolsSettingsWindow, '_get_database_path', return_value=temp_db_path):
            # Call the worker
            ToolsSettingsWindow._restore_single_worker(mock_window, "chachanotes", backup_path)
        
        # Verify restore happened
        assert mock_copy.call_count == 2  # One for pre-restore backup, one for restore
        
        # Verify notification
        assert mock_window.call_from_thread.called
    
    @patch('sqlite3.connect')
    def test_check_single_worker_success(self, mock_connect, mock_window):
        """Test _check_single_worker with passing integrity check."""
        # Setup mock connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("ok",)
        mock_conn.execute.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock the database path
        with patch.object(ToolsSettingsWindow, '_get_database_path', return_value=Path("/test/db.db")):
            # Call the worker
            ToolsSettingsWindow._check_single_worker(mock_window, "chachanotes")
        
        # Verify integrity check was called
        mock_conn.execute.assert_called_with("PRAGMA integrity_check")
        
        # Verify success notification
        call_args = mock_window.call_from_thread.call_args[0]
        assert "integrity check passed" in call_args[1]
    
    @patch('sqlite3.connect')
    def test_check_single_worker_failure(self, mock_connect, mock_window):
        """Test _check_single_worker with failing integrity check."""
        # Setup mock connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("corruption at page 5",)
        mock_conn.execute.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock the database path
        with patch.object(ToolsSettingsWindow, '_get_database_path', return_value=Path("/test/db.db")):
            # Call the worker
            ToolsSettingsWindow._check_single_worker(mock_window, "chachanotes")
        
        # Verify error notification
        call_args = mock_window.call_from_thread.call_args[0]
        assert "integrity issues" in call_args[1]
    
    def test_update_last_backup_status(self, mock_window):
        """Test _update_last_backup_status updates UI."""
        mock_static = Mock()
        mock_window.query_one.return_value = mock_static
        
        # Call the method
        ToolsSettingsWindow._update_last_backup_status(
            mock_window, 
            "chachanotes", 
            "20240101_120000"
        )
        
        # Verify widget was queried
        mock_window.query_one.assert_called_with("#db-backup-chachanotes", Static)
        
        # Verify update was called with formatted time
        mock_static.update.assert_called_once()
        update_text = mock_static.update.call_args[0][0]
        assert "Last Backup: 2024-01-01 12:00" in update_text
    
    @pytest.mark.asyncio
    async def test_vacuum_single_database(self, mock_window):
        """Test async vacuum_single_database method."""
        # Call the method
        await ToolsSettingsWindow._vacuum_single_database(mock_window, "chachanotes")
        
        # Verify notification
        mock_window.app_instance.notify.assert_called_with(
            "Starting vacuum operation for chachanotes database...",
            severity="information"
        )
        
        # Verify worker was started
        mock_window.run_worker.assert_called_once()
        assert mock_window.run_worker.call_args[0][1] == "chachanotes"
        assert mock_window.run_worker.call_args[1]["name"] == "vacuum_chachanotes_worker"
    
    @pytest.mark.asyncio
    async def test_backup_single_database(self, mock_window):
        """Test async backup_single_database method."""
        # Call the method
        await ToolsSettingsWindow._backup_single_database(mock_window, "media")
        
        # Verify notification
        mock_window.app_instance.notify.assert_called_with(
            "Starting backup for media database...",
            severity="information"
        )
        
        # Verify worker was started
        mock_window.run_worker.assert_called_once()
        assert mock_window.run_worker.call_args[0][1] == "media"
        assert mock_window.run_worker.call_args[1]["name"] == "backup_media_worker"
    
    @pytest.mark.asyncio
    async def test_check_single_database(self, mock_window):
        """Test async check_single_database method."""
        # Call the method
        await ToolsSettingsWindow._check_single_database(mock_window, "prompts")
        
        # Verify notification
        mock_window.app_instance.notify.assert_called_with(
            "Checking prompts database integrity...",
            severity="information"
        )
        
        # Verify worker was started
        mock_window.run_worker.assert_called_once()
        assert mock_window.run_worker.call_args[0][1] == "prompts"
        assert mock_window.run_worker.call_args[1]["name"] == "check_prompts_worker"


class TestDatabaseRestore:
    """Test database restore functionality."""
    
    @pytest.mark.asyncio
    @patch('tldw_chatbook.UI.Tools_Settings_Window.FilePickerDialog')
    async def test_restore_single_database_no_file(self, mock_dialog_class, mock_window):
        """Test restore when no file is selected."""
        # Mock dialog returning None
        mock_dialog = Mock()
        mock_dialog_class.return_value = mock_dialog
        mock_window.app_instance.push_screen = Mock(return_value=None)
        
        # Call the method
        await ToolsSettingsWindow._restore_single_database(mock_window, "chachanotes")
        
        # Verify dialog was shown
        assert mock_window.app_instance.push_screen.called
        
        # Verify no further action taken
        assert not mock_window.run_worker.called
    
    @pytest.mark.asyncio
    @patch('tldw_chatbook.UI.Tools_Settings_Window.FilePickerDialog')
    async def test_restore_single_database_with_file(self, mock_dialog_class, mock_window):
        """Test restore when file is selected."""
        # Mock dialog returning a file path
        mock_dialog = Mock()
        mock_dialog_class.return_value = mock_dialog
        mock_window.app_instance.push_screen = Mock(return_value="/backup/test.db")
        
        # Mock perform_database_restore
        with patch.object(ToolsSettingsWindow, '_perform_database_restore') as mock_perform:
            # Call the method
            await ToolsSettingsWindow._restore_single_database(mock_window, "chachanotes")
        
        # Verify restore was initiated
        mock_perform.assert_called_once()
        assert mock_perform.call_args[0][1] == "chachanotes"
        assert str(mock_perform.call_args[0][2]) == "/backup/test.db"
    
    @pytest.mark.asyncio
    async def test_perform_database_restore_invalid_metadata(self, mock_window, temp_db_path):
        """Test restore with mismatched metadata."""
        # Create metadata file
        metadata_path = temp_db_path.with_suffix('.json')
        metadata = {"database": "prompts"}  # Wrong database type
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Call the method
        await ToolsSettingsWindow._perform_database_restore(
            mock_window, 
            "chachanotes", 
            temp_db_path
        )
        
        # Verify error notification
        mock_window.app_instance.notify.assert_called_with(
            "This backup is for prompts database, not chachanotes",
            severity="error"
        )
        
        # Verify worker not started
        assert not mock_window.run_worker.called