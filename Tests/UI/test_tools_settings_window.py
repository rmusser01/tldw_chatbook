import pytest
import pytest_asyncio
import toml
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
import tempfile
import shutil
import sqlite3
from datetime import datetime

from textual.widgets import Button, TextArea, Label, Static
from textual.app import App
try:
    from textual.app import AppTest
except ImportError:
    # AppTest not available in Textual 3.3.0, create a mock
    AppTest = None

from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
# Import DEFAULT_CONFIG_PATH to be monkeypatched, and the function that uses it
import tldw_chatbook.config

# Import test utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from db_test_utilities import TestDatabaseSchema, DatabasePopulator
from test_utilities import TestDataFactory


# Helper to create a dummy config file for testing
def create_dummy_config(config_path: Path, content: dict):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        toml.dump(content, f)


@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    """Provides a temporary path for config.toml."""
    return tmp_path / "config.toml"


@pytest.fixture(autouse=True)
def mock_config_path(monkeypatch, temp_config_path: Path):
    """Monkeypatches DEFAULT_CONFIG_PATH and related functions to use a temporary path."""
    # Ensure a default config exists at the temp path before tests run
    default_initial_content = {"initial_setting": "default_value"}
    create_dummy_config(temp_config_path, default_initial_content)

    monkeypatch.setattr(tldw_chatbook.config, 'DEFAULT_CONFIG_PATH', temp_config_path)

    # If load_cli_config_and_ensure_existence has its own reference to the original path (e.g. via default arg)
    # it might need to be mocked or reloaded. However, direct setattr should be effective for module-level constants.
    # For this setup, we assume that when ToolsSettingsWindow calls load_cli_config_and_ensure_existence,
    # it will see the monkeypatched DEFAULT_CONFIG_PATH.


@pytest.fixture
def mock_app_instance():
    """Fixture to create a mock TldwCli app instance."""
    app = MagicMock(spec=App)
    # Mock the notify method, which is used by ToolsSettingsWindow
    app.notify = MagicMock()
    return app


@pytest_asyncio.fixture
async def settings_window(mock_app_instance, temp_config_path: Path) -> ToolsSettingsWindow:
    """
    Fixture to create ToolsSettingsWindow, mount it within a test app,
    and ensure it uses the temporary config path.
    """
    # The mock_config_path fixture (autouse=True) ensures that DEFAULT_CONFIG_PATH
    # is already patched when load_cli_config_and_ensure_existence is called within ToolsSettingsWindow.

    # Create a fresh config for each test that uses this fixture,
    # or rely on the one from mock_config_path if that's intended as a common base.
    # For clarity, let's give it a distinct initial state for window creation.
    initial_window_config = {"window_init": "true"}
    create_dummy_config(temp_config_path, initial_window_config)

    window = ToolsSettingsWindow(app_instance=mock_app_instance)

    # Mount the window in a test app environment
    if AppTest is None:
        pytest.skip("AppTest not available in this version of Textual")
        
    async with AppTest(app=mock_app_instance, driver_class=None) as pilot:  # Using AppTest for proper mounting
        mock_app_instance.mount(window)  # Mount the window onto our mock app
        await pilot.pause()  # Allow compose to run
        yield window  # The window is now composed and ready


@pytest.mark.asyncio
async def test_tab_renaming(settings_window: ToolsSettingsWindow):
    """Test if the 'API Keys' tab has been correctly renamed."""
    nav_button = settings_window.query_one("#ts-nav-config-file-settings", Button)
    assert nav_button.label.plain == "Configuration File Settings"

    content_area = settings_window.query_one("#ts-view-config-file-settings")
    assert content_area is not None
    # Check that the TextArea is inside this content area and not the static text
    assert isinstance(content_area.query_one("#config-text-area", TextArea), TextArea)


@pytest.mark.asyncio
async def test_load_config_values(settings_window: ToolsSettingsWindow, temp_config_path: Path):
    """Test if configuration values are loaded and displayed correctly."""
    expected_config_content = {"general": {"model": "gpt-4"}, "api_keys": {"openai": "sk-..."}}
    create_dummy_config(temp_config_path, expected_config_content)

    # Force reload within the window or re-initialize to pick up new config
    # The settings_window is already initialized. We need to trigger its internal load.
    # The simplest way is to simulate a "Reload" click if available and makes sense,
    # or directly call a method if one exists, or update the TextArea.text
    # For now, let's assume the compose correctly loads it due to the patched DEFAULT_CONFIG_PATH
    # If compose has already run, we might need to trigger an update.
    # Let's update the text area directly after ensuring the config file is written.

    # The window's compose method calls load_cli_config_and_ensure_existence().
    # The autouse fixture mock_config_path should ensure this used temp_config_path.
    # The settings_window fixture also writes initial_window_config.
    # So, for this test, we write *again* to temp_config_path and then make the window reload.

    config_text_area = settings_window.query_one("#config-text-area", TextArea)

    # To ensure it loads the *expected_config_content* and not initial_window_config:
    reloaded_config = tldw_chatbook.config.load_cli_config_and_ensure_existence(force_reload=True)
    config_text_area.text = toml.dumps(reloaded_config)  # Manually set text after explicit load

    assert config_text_area.text.strip() != ""
    loaded_text_area_config = toml.loads(config_text_area.text)
    assert loaded_text_area_config == expected_config_content


@pytest.mark.asyncio
async def test_save_config_values(settings_window: ToolsSettingsWindow, temp_config_path: Path, mock_app_instance):
    """Test if configuration values can be saved correctly."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    save_button = settings_window.query_one("#save-config-button", Button)

    new_config_dict = {"user": {"name": "test_user", "theme": "blue"}}
    config_text_area.text = toml.dumps(new_config_dict)

    # Simulate button press by calling the handler
    await settings_window.on_button_pressed(Button.Pressed(save_button))

    mock_app_instance.notify.assert_called_with("Configuration saved successfully.")

    with open(temp_config_path, "r") as f:
        saved_content_on_disk = toml.load(f)

    assert saved_content_on_disk == new_config_dict


@pytest.mark.asyncio
async def test_reload_config_values(settings_window: ToolsSettingsWindow, temp_config_path: Path, mock_app_instance):
    """Test if configuration values can be reloaded correctly."""
    # 1. Setup initial config on disk
    original_disk_config = {"settings": {"feature_x": True, "version": 1}}
    create_dummy_config(temp_config_path, original_disk_config)

    # 2. Ensure window's TextArea reflects this initial config
    # (Simulate a reload or assume it's loaded it - let's simulate reload for clarity)
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    reload_button = settings_window.query_one("#reload-config-button", Button)

    # Press reload to make sure it's showing original_disk_config
    await settings_window.on_button_pressed(Button.Pressed(reload_button))
    mock_app_instance.notify.assert_called_with("Configuration reloaded.")
    assert toml.loads(config_text_area.text) == original_disk_config

    # 3. Modify the TextArea to simulate user changes (these are not saved yet)
    user_modified_text_dict = {"settings": {"feature_x": False, "version": 2}}
    config_text_area.text = toml.dumps(user_modified_text_dict)
    assert toml.loads(config_text_area.text) == user_modified_text_dict  # Verify change in TextArea

    # 4. Simulate reload button press again
    await settings_window.on_button_pressed(Button.Pressed(reload_button))
    mock_app_instance.notify.assert_called_with("Configuration reloaded.")  # Called again

    # 5. Verify TextArea content is reverted to original_disk_config (ignoring user_modified_text_dict)
    assert toml.loads(config_text_area.text) == original_disk_config


@pytest.mark.asyncio
async def test_save_invalid_toml_format(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test saving invalid TOML data reports an error."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    save_button = settings_window.query_one("#save-config-button", Button)

    invalid_toml_text = "this is not valid toml { text = blah"
    config_text_area.text = invalid_toml_text

    await settings_window.on_button_pressed(Button.Pressed(save_button))

    mock_app_instance.notify.assert_called_with("Error: Invalid TOML format.", severity="error")


# Test for save I/O error (conceptual - requires mocking 'open')
@pytest.mark.skip(reason="Complex to mock built-in open reliably for this specific write operation only")
@pytest.mark.asyncio
async def test_save_io_error(settings_window: ToolsSettingsWindow, mock_app_instance, monkeypatch):
    """Test saving config when an IOError occurs."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    save_button = settings_window.query_one("#save-config-button", Button)

    config_text_area.text = toml.dumps({"good": "data"})

    # Mock 'open' within the tldw_chatbook.UI.Tools_Settings_Window context or globally
    # to raise IOError only for the specific write operation.
    # This is tricky because 'open' is a builtin and patching it requires care.

    # For example, using a more specific patch target if 'open' is imported like 'from io import open':
    # with monkeypatch.context() as m:
    # m.setattr("tldw_chatbook.UI.Tools_Settings_Window.open", MagicMock(side_effect=IOError("Disk full")))
    # await settings_window.on_button_pressed(Button.Pressed(save_button))

    # Or if it uses the global 'open':
    # with patch('builtins.open', MagicMock(side_effect=IOError("Cannot write"))):
    # await settings_window.on_button_pressed(Button.Pressed(save_button))

    # This test is skipped because such mocking is highly dependent on exact 'open' usage
    # and can be fragile. A more robust way might involve filesystem-level mocks if available.

    # mock_app_instance.notify.assert_called_with("Error: Could not write to configuration file.", severity="error")
    pass


# ===========================================
# Database Tools Tests
# ===========================================

@pytest.fixture
def test_db_dir(tmp_path):
    """Create a directory with test databases."""
    db_dir = tmp_path / "databases"
    db_dir.mkdir()
    
    # Create test databases with sample data
    databases = {
        'ChaChaNotes.db': TestDatabaseSchema.CONVERSATIONS_SCHEMA + TestDatabaseSchema.MESSAGES_SCHEMA,
        'Client_Media_DB.db': """
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT
            );
            INSERT INTO media (title, content) VALUES ('Test Media', 'Content');
        """,
        'Prompts_DB.db': """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY,
                name TEXT,
                content TEXT
            );
            INSERT INTO prompts (name, content) VALUES ('Test Prompt', 'Content');
        """,
        'Evals_DB.db': """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                name TEXT,
                score REAL
            );
        """,
        'RAG_Indexing_DB.db': """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                content TEXT,
                vector BLOB
            );
        """,
        'Subscriptions_DB.db': """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY,
                name TEXT,
                url TEXT
            );
        """
    }
    
    db_paths = {}
    for db_name, schema in databases.items():
        db_path = db_dir / db_name
        conn = sqlite3.connect(str(db_path))
        conn.executescript(schema)
        # Set a schema version
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        db_paths[db_name.replace('.db', '')] = str(db_path)
    
    return db_dir, db_paths


@pytest.fixture
def mock_database_path_lookup(test_db_dir, monkeypatch):
    """Mock the database path lookup functions."""
    db_dir, db_paths = test_db_dir
    
    def mock_get_db_path(db_name):
        return db_paths.get(db_name, str(db_dir / f"{db_name}.db"))
    
    # Mock the app instance's database path method
    monkeypatch.setattr(
        "tldw_chatbook.UI.Tools_Settings_Window.ToolsSettingsWindow._get_database_path",
        mock_get_db_path
    )
    
    return db_paths


@pytest.mark.asyncio
async def test_database_tools_composition(settings_window: ToolsSettingsWindow):
    """Test that database tools section is properly composed."""
    # Check that Database Tools tab exists
    nav_button = settings_window.query_one("#ts-nav-database-tools", Button)
    assert nav_button is not None
    assert nav_button.label.plain == "Database Tools"
    
    # Check that the content area exists
    content_area = settings_window.query_one("#ts-view-database-tools")
    assert content_area is not None
    
    # Check for individual database sections
    database_names = ["ChaChaNotes", "Media", "Prompts", "Evals", "RAG", "Subscriptions"]
    for db_name in database_names:
        # Each database should have its own section
        db_section = content_area.query(f".db-section-{db_name.lower()}")
        assert len(db_section) > 0, f"Database section for {db_name} not found"


@pytest.mark.asyncio
async def test_individual_database_vacuum(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test vacuum operation on individual databases."""
    # Find the vacuum button for ChaChaNotes
    vacuum_button = settings_window.query_one("#vacuum-chachanotes", Button)
    assert vacuum_button is not None
    
    # Simulate button press
    await settings_window.on_button_pressed(Button.Pressed(vacuum_button))
    
    # Check that notification was called
    mock_app_instance.notify.assert_called()
    # Should notify about starting vacuum
    calls = mock_app_instance.notify.call_args_list
    assert any("Starting vacuum" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_individual_database_backup(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup, tmp_path):
    """Test backup operation on individual databases."""
    # Mock the backup directory
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    
    with patch("pathlib.Path.home", return_value=tmp_path):
        # Find the backup button for Media database
        backup_button = settings_window.query_one("#backup-media", Button)
        assert backup_button is not None
        
        # Simulate button press
        await settings_window.on_button_pressed(Button.Pressed(backup_button))
        
        # Check that a worker was started
        mock_app_instance.run_worker.assert_called()


@pytest.mark.asyncio
async def test_database_restore_with_file_picker(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test restore operation with file picker dialog."""
    # Find the restore button for Prompts database
    restore_button = settings_window.query_one("#restore-prompts", Button)
    assert restore_button is not None
    
    # Mock the file picker dialog
    with patch("tldw_chatbook.UI.Tools_Settings_Window.ToolsSettingsWindow.push_screen") as mock_push_screen:
        # Simulate button press
        await settings_window.on_button_pressed(Button.Pressed(restore_button))
        
        # Verify file picker was pushed
        mock_push_screen.assert_called_once()
        # The first argument should be the FilePickerDialog instance
        args = mock_push_screen.call_args[0]
        assert len(args) > 0


@pytest.mark.asyncio
async def test_database_integrity_check(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test database integrity check operation."""
    # Find the check button for RAG database
    check_button = settings_window.query_one("#check-rag", Button)
    assert check_button is not None
    
    # Simulate button press
    await settings_window.on_button_pressed(Button.Pressed(check_button))
    
    # Check that notification was called
    mock_app_instance.notify.assert_called()
    # Should notify about checking integrity
    calls = mock_app_instance.notify.call_args_list
    assert any("Checking" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_all_databases_operations(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test operations on all databases at once."""
    # Find the "All Databases" section
    all_db_section = settings_window.query_one(".db-section-all")
    assert all_db_section is not None
    
    # Test vacuum all
    vacuum_all_button = settings_window.query_one("#vacuum-all", Button)
    assert vacuum_all_button is not None
    
    await settings_window.on_button_pressed(Button.Pressed(vacuum_all_button))
    
    # Should have multiple notifications (one per database)
    assert mock_app_instance.notify.call_count >= 6  # At least 6 databases


@pytest.mark.asyncio
async def test_database_status_display(settings_window: ToolsSettingsWindow, mock_database_path_lookup):
    """Test that database status information is displayed correctly."""
    # Check each database status container
    database_names = ["chachanotes", "media", "prompts", "evals", "rag", "subscriptions"]
    
    for db_name in database_names:
        status_container = settings_window.query_one(f"#db-status-{db_name}")
        assert status_container is not None
        
        # Should contain schema version and file size
        status_text = status_container.query_one(Static)
        assert "Schema" in status_text.renderable or "Version" in status_text.renderable


@pytest.mark.asyncio
async def test_create_chatbook_button(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test that chatbook creation button exists and works."""
    # Find the create chatbook button
    create_button = settings_window.query_one("#create-chatbook", Button)
    assert create_button is not None
    assert "Create Chatbook" in create_button.label.plain
    
    # Mock the chatbook creation window
    with patch("tldw_chatbook.UI.Tools_Settings_Window.ChatbookCreationWindow") as mock_window:
        await settings_window.on_button_pressed(Button.Pressed(create_button))
        
        # Should push the chatbook creation screen
        mock_app_instance.push_screen.assert_called_once()


@pytest.mark.asyncio
async def test_import_chatbook_button(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test that chatbook import button exists and works."""
    # Find the import chatbook button
    import_button = settings_window.query_one("#import-chatbook", Button)
    assert import_button is not None
    assert "Import Chatbook" in import_button.label.plain
    
    # Mock file picker for import
    with patch("tldw_chatbook.UI.Tools_Settings_Window.ToolsSettingsWindow.push_screen") as mock_push_screen:
        await settings_window.on_button_pressed(Button.Pressed(import_button))
        
        # Should push the file picker
        mock_push_screen.assert_called_once()


@pytest.mark.asyncio
async def test_database_error_handling(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test error handling for database operations."""
    # Mock a database operation to fail
    with patch("sqlite3.connect", side_effect=sqlite3.Error("Database is locked")):
        # Try to vacuum a database
        vacuum_button = settings_window.query_one("#vacuum-chachanotes", Button)
        await settings_window.on_button_pressed(Button.Pressed(vacuum_button))
        
        # Should show error notification
        mock_app_instance.notify.assert_called()
        calls = mock_app_instance.notify.call_args_list
        assert any("error" in str(call).lower() for call in calls)

