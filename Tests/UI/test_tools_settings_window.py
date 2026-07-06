import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call

import pytest
import pytest_asyncio
import toml

from textual.widgets import Button, Checkbox, Input, Select, TextArea, Label, Static
from textual.app import App
try:
    from textual.app import AppTest
except ImportError:
    # AppTest not available in Textual 3.3.0, create a mock
    AppTest = None

from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
from tldw_chatbook.UI.Outputs_Panel import OutputsPanel
from tldw_chatbook.UI.Sharing_Panel import SharingPanel
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


class _ToolsSettingsHostApp(App):
    """Minimal real App that hosts a ToolsSettingsWindow as its own app_instance."""

    def __init__(self):
        super().__init__()
        self.notify = MagicMock()
        self.push_screen = MagicMock()
        self.unified_mcp_service = None
        self.current_runtime_backend = "local"
        self.server_sharing_scope_service = None
        self.server_outputs_scope_service = None

    def get_authoritative_runtime_source(self):
        return self.current_runtime_backend

    def compose(self):
        yield ToolsSettingsWindow(app_instance=self)


@asynccontextmanager
async def mount_settings_window(config_dict: dict, temp_config_path: Path, monkeypatch):
    """Write config_dict to temp_config_path, patch DEFAULT_CONFIG_PATH, and yield a live-mounted ToolsSettingsWindow driven by a real pilot."""
    create_dummy_config(temp_config_path, config_dict)
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", temp_config_path)

    app = _ToolsSettingsHostApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ToolsSettingsWindow)
        yield window, pilot


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


@pytest.mark.asyncio
async def test_tools_settings_window_exposes_unified_mcp_view():
    class ToolsSettingsHostApp(App):
        def __init__(self):
            super().__init__()
            self.notify = MagicMock()
            self.unified_mcp_service = None

        def compose(self):
            yield ToolsSettingsWindow(app_instance=self)

    app = ToolsSettingsHostApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ToolsSettingsWindow)
        nav_button = window.query_one("#ts-nav-unified-mcp", Button)

        assert nav_button.label.plain == "Unified MCP"

        await window.on_button_pressed(Button.Pressed(nav_button))

        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-unified-mcp"


@pytest.mark.asyncio
async def test_tools_settings_window_exposes_sharing_view():
    class ToolsSettingsHostApp(App):
        def __init__(self):
            super().__init__()
            self.notify = MagicMock()
            self.unified_mcp_service = None
            self.current_runtime_backend = "server"
            self.server_sharing_scope_service = MagicMock()

        def get_authoritative_runtime_source(self):
            return self.current_runtime_backend

        def compose(self):
            yield ToolsSettingsWindow(app_instance=self)

    app = ToolsSettingsHostApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ToolsSettingsWindow)
        nav_button = window.query_one("#ts-nav-sharing", Button)

        assert nav_button.label.plain == "Sharing"

        await window.on_button_pressed(Button.Pressed(nav_button))

        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-sharing"
        assert window.query_one("#sharing-panel", SharingPanel) is not None


@pytest.mark.asyncio
async def test_tools_settings_window_exposes_outputs_view():
    class ToolsSettingsHostApp(App):
        def __init__(self):
            super().__init__()
            self.notify = MagicMock()
            self.unified_mcp_service = None
            self.current_runtime_backend = "server"
            self.server_outputs_scope_service = MagicMock()
            self.server_sharing_scope_service = MagicMock()

        def get_authoritative_runtime_source(self):
            return self.current_runtime_backend

        def compose(self):
            yield ToolsSettingsWindow(app_instance=self)

    app = ToolsSettingsHostApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ToolsSettingsWindow)
        nav_button = window.query_one("#ts-nav-outputs", Button)

        assert nav_button.label.plain == "Outputs"

        await window.on_button_pressed(Button.Pressed(nav_button))

        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-outputs"
        assert window.query_one("#outputs-panel", OutputsPanel) is not None


class SharingPanelHostApp(App):
    def __init__(self, *, runtime_backend: str, scope_service: MagicMock):
        super().__init__()
        self.notify = MagicMock()
        self.current_runtime_backend = runtime_backend
        self.server_sharing_scope_service = scope_service

    def get_authoritative_runtime_source(self):
        return self.current_runtime_backend

    def compose(self):
        yield SharingPanel(self, id="sharing-panel")


class OutputsPanelHostApp(App):
    def __init__(self, *, runtime_backend: str, scope_service: MagicMock):
        super().__init__()
        self.notify = MagicMock()
        self.current_runtime_backend = runtime_backend
        self.server_outputs_scope_service = scope_service

    def get_authoritative_runtime_source(self):
        return self.current_runtime_backend

    def compose(self):
        yield OutputsPanel(self, id="outputs-panel")


@pytest.mark.asyncio
async def test_sharing_panel_rejects_local_mode_with_explicit_guidance():
    scope_service = MagicMock()
    app = SharingPanelHostApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(SharingPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#sharing-disabled", Static).display is True
        assert panel.query_one("#sharing-main").display is False
        assert panel.query_one("#sharing-create-workspace-share-btn", Button).disabled is True


@pytest.mark.asyncio
async def test_sharing_panel_routes_server_workspace_share_and_token_operations():
    scope_service = MagicMock()
    scope_service.share_workspace = AsyncMock(return_value={"id": "server:share:7", "access_level": "view_chat"})
    scope_service.list_workspace_shares = AsyncMock(return_value={"shares": [{"id": "server:share:7"}], "total": 1})
    scope_service.create_share_token = AsyncMock(return_value={"id": "server:share_token:5", "raw_token": "raw-token"})
    scope_service.list_share_tokens = AsyncMock(return_value={"tokens": [{"id": "server:share_token:5"}], "total": 1})
    scope_service.list_shared_with_me = AsyncMock(return_value={"items": [{"id": "server:share:9"}], "total": 1})
    app = SharingPanelHostApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(SharingPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#sharing-workspace-id", Input).value = "ws-1"
        panel.query_one("#sharing-scope-type", Select).value = "team"
        panel.query_one("#sharing-scope-id", Input).value = "11"
        panel.query_one("#sharing-access-level", Select).value = "view_chat"
        panel.query_one("#sharing-allow-clone", Checkbox).value = True
        await panel.create_workspace_share()
        await panel.list_workspace_shares()

        panel.query_one("#sharing-resource-type", Select).value = "workspace"
        panel.query_one("#sharing-resource-id", Input).value = "ws-1"
        panel.query_one("#sharing-token-password", Input).value = "passphrase"
        panel.query_one("#sharing-token-max-uses", Input).value = "10"
        await panel.create_share_token()
        await panel.list_share_tokens()
        await panel.list_shared_with_me()

        scope_service.share_workspace.assert_awaited_once_with(
            mode="server",
            workspace_id="ws-1",
            share_scope_type="team",
            share_scope_id=11,
            access_level="view_chat",
            allow_clone=True,
        )
        scope_service.list_workspace_shares.assert_awaited_once_with(
            mode="server",
            workspace_id="ws-1",
            include_revoked=False,
        )
        scope_service.create_share_token.assert_awaited_once_with(
            mode="server",
            resource_type="workspace",
            resource_id="ws-1",
            access_level="view_chat",
            allow_clone=True,
            password="passphrase",
            max_uses=10,
            expires_at=None,
        )
        scope_service.list_share_tokens.assert_awaited_once_with(mode="server")
        scope_service.list_shared_with_me.assert_awaited_once_with(mode="server")
        rendered_status = str(panel.query_one("#sharing-status", Static).render())
        assert "server:share:9" in rendered_status


@pytest.mark.asyncio
async def test_outputs_panel_rejects_local_mode_with_explicit_guidance():
    scope_service = MagicMock()
    app = OutputsPanelHostApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(OutputsPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#outputs-disabled", Static).display is True
        assert panel.query_one("#outputs-main").display is False
        assert panel.query_one("#outputs-list-templates-btn", Button).disabled is True
        assert panel.query_one("#outputs-list-artifacts-btn", Button).disabled is True


@pytest.mark.asyncio
async def test_outputs_panel_routes_server_template_and_artifact_operations():
    scope_service = MagicMock()
    scope_service.list_output_templates = AsyncMock(
        return_value={"items": [{"id": "server:output_template:7", "name": "Weekly Briefing"}], "total": 1}
    )
    scope_service.create_output_template = AsyncMock(
        return_value={"id": "server:output_template:7", "name": "Weekly Briefing"}
    )
    scope_service.preview_output_template = AsyncMock(
        return_value={"entity_kind": "output_template_preview", "rendered": "# Preview"}
    )
    scope_service.list_outputs = AsyncMock(
        return_value={"items": [{"id": "server:output:11", "title": "Weekly Briefing"}], "total": 1, "page": 1, "size": 10}
    )
    scope_service.create_output = AsyncMock(
        return_value={"id": "server:output:11", "entity_kind": "output_render_result", "title": "Weekly Briefing"}
    )
    scope_service.delete_output = AsyncMock(
        return_value={"entity_kind": "output_delete", "success": True, "output_id": 11}
    )
    app = OutputsPanelHostApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(OutputsPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#outputs-template-query", Input).value = "brief"
        panel.query_one("#outputs-template-limit", Input).value = "25"
        panel.query_one("#outputs-template-offset", Input).value = "5"
        panel.query_one("#outputs-template-name", Input).value = "Weekly Briefing"
        panel.query_one("#outputs-template-type", Select).value = "briefing_markdown"
        panel.query_one("#outputs-template-format", Select).value = "md"
        panel.query_one("#outputs-template-description", Input).value = "Render a weekly markdown briefing"
        panel.query_one("#outputs-template-body", TextArea).text = "# {{ job.name }}"
        panel.query_one("#outputs-template-default", Checkbox).value = True
        panel.query_one("#outputs-preview-template-id", Input).value = "7"
        panel.query_one("#outputs-preview-item-ids", Input).value = "1,2"
        panel.query_one("#outputs-preview-limit", Input).value = "10"

        await panel.list_output_templates()
        await panel.create_output_template()
        await panel.preview_output_template()

        panel.query_one("#outputs-artifact-page", Input).value = "1"
        panel.query_one("#outputs-artifact-size", Input).value = "10"
        panel.query_one("#outputs-artifact-run-id", Input).value = "77"
        panel.query_one("#outputs-artifact-workspace-tag", Input).value = "workspace:demo"
        panel.query_one("#outputs-create-template-id", Input).value = "7"
        panel.query_one("#outputs-create-item-ids", Input).value = "1,2"
        panel.query_one("#outputs-create-title", Input).value = "Weekly Briefing"
        panel.query_one("#outputs-create-workspace-tag", Input).value = "workspace:demo"
        panel.query_one("#outputs-create-ingest", Checkbox).value = True
        panel.query_one("#outputs-delete-output-id", Input).value = "11"
        panel.query_one("#outputs-delete-hard", Checkbox).value = True
        panel.query_one("#outputs-delete-file", Checkbox).value = True

        await panel.list_outputs()
        await panel.create_output()
        await panel.delete_output()

        scope_service.list_output_templates.assert_awaited_once_with(
            mode="server",
            q="brief",
            limit=25,
            offset=5,
        )
        scope_service.create_output_template.assert_awaited_once_with(
            mode="server",
            name="Weekly Briefing",
            type="briefing_markdown",
            format="md",
            body="# {{ job.name }}",
            description="Render a weekly markdown briefing",
            is_default=True,
        )
        scope_service.preview_output_template.assert_awaited_once_with(
            mode="server",
            template_id=7,
            item_ids=[1, 2],
            limit=10,
        )
        scope_service.list_outputs.assert_awaited_once_with(
            mode="server",
            page=1,
            size=10,
            run_id=77,
            workspace_tag="workspace:demo",
        )
        scope_service.create_output.assert_awaited_once_with(
            mode="server",
            template_id=7,
            item_ids=[1, 2],
            title="Weekly Briefing",
            workspace_tag="workspace:demo",
            ingest_to_media_db=True,
        )
        scope_service.delete_output.assert_awaited_once_with(
            mode="server",
            output_id=11,
            hard=True,
            delete_file=True,
        )
        rendered_status = str(panel.query_one("#outputs-status", Static).render())
        assert "server:output:11" in rendered_status or "output_delete" in rendered_status


@pytest.mark.asyncio
async def test_chat_api_key_field_prefilled_for_config_key(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {"openai": {"api_key": "test-configured-key"}},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        field = window.query_one("#general-chat-api-key", Input)
        assert field.password is True
        assert field.value == "test-configured-key"
        assert field.disabled is False


@pytest.mark.asyncio
async def test_chat_api_key_field_disabled_for_keyless_provider(monkeypatch, temp_config_path):
    config = {
        "providers": {"Ollama": ["llama3"], "OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "Ollama", "model": "llama3"},
        "api_settings": {},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        field = window.query_one("#general-chat-api-key", Input)
        assert field.disabled is True
        assert "No API key needed" in field.placeholder


@pytest.mark.asyncio
async def test_chat_api_key_field_reloads_on_provider_change(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {"openai": {"api_key": "test-configured-key"}},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        field = window.query_one("#general-chat-api-key", Input)
        assert field.value == "test-configured-key"

        # Switch to a keyless provider -> field disables and clears
        window.query_one("#general-chat-provider", Select).value = "Ollama"
        await pilot.pause()
        assert field.disabled is True
        assert field.value == ""


@pytest.mark.asyncio
async def test_chat_api_key_save_writes_config_and_updates_live_config(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        window.app_instance.app_config = {"api_settings": {}}
        window.query_one("#general-chat-api-key", Input).value = "test-brand-new-key"

        saved = window._save_chat_api_key()
        assert saved is True

        # Written to the on-disk config under the normalized provider key
        written = toml.load(temp_config_path)
        assert written["api_settings"]["openai"]["api_key"] == "test-brand-new-key"

        # Live app config updated in place (no restart needed)
        assert window.app_instance.app_config["api_settings"]["openai"]["api_key"] == "test-brand-new-key"


@pytest.mark.asyncio
async def test_chat_api_key_save_skips_blank(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        window.app_instance.app_config = {"api_settings": {}}
        window.query_one("#general-chat-api-key", Input).value = "   "
        assert window._save_chat_api_key() is False
        written = toml.load(temp_config_path)
        assert written.get("api_settings", {}).get("openai", {}).get("api_key") is None


@pytest.mark.asyncio
async def test_chat_api_key_field_clears_when_provider_blanked(monkeypatch, temp_config_path):
    """Blanking the provider must clear the field, not leave the prior key visible."""
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {"openai": {"api_key": "test-configured-key"}},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        field = window.query_one("#general-chat-api-key", Input)
        assert field.value == "test-configured-key"

        # The provider Select disallows a blank value in normal use, so drive the
        # defensive handler branch directly with a synthetic BLANK change event.
        select = window.query_one("#general-chat-provider", Select)
        window._on_chat_provider_changed(Select.Changed(select, Select.BLANK))
        assert field.value == ""
        assert field.disabled is True
        assert "Select a provider" in field.placeholder


@pytest.mark.asyncio
async def test_chat_api_key_save_pushes_decrypted_key_to_live_config_when_encrypted(monkeypatch, temp_config_path):
    """With config encryption on, the live app_config must receive the DECRYPTED
    key, never the on-disk ciphertext (which chat would send verbatim and fail)."""
    # A session password unlocks the field and enables encrypt-on-write.
    monkeypatch.setattr(tldw_chatbook.config, "_ENCRYPTION_PASSWORD", "test-master-pw")
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {},
        "encryption": {"enabled": True},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        window.app_instance.app_config = {"api_settings": {}}
        window.query_one("#general-chat-api-key", Input).value = "test-secret-live-key"

        assert window._save_chat_api_key() is True

        # On disk the key is encrypted...
        enc_mod = tldw_chatbook.config.get_encryption_module()
        written_key = toml.load(temp_config_path)["api_settings"]["openai"]["api_key"]
        assert enc_mod.is_encrypted(written_key)
        assert written_key != "test-secret-live-key"

        # ...but the live config the send path reads holds decrypted plaintext.
        live_key = window.app_instance.app_config["api_settings"]["openai"]["api_key"]
        assert live_key == "test-secret-live-key"
        assert not enc_mod.is_encrypted(live_key)
