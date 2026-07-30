import ast
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import toml

from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from Tests.UI.test_screen_navigation import _build_test_app
import tldw_chatbook.app as app_module
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
from tldw_chatbook.UI.Outputs_Panel import OutputsPanel
from tldw_chatbook.UI.Sharing_Panel import SharingPanel
from tldw_chatbook.UI.Screens.tools_settings_screen import ToolsSettingsScreen
# Import DEFAULT_CONFIG_PATH to be monkeypatched, and the function that uses it
import tldw_chatbook.config

# Import test utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from db_test_utilities import TestDatabaseSchema


# Helper to create a dummy config file for testing
def create_dummy_config(config_path: Path, content: dict):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        toml.dump(content, f)


def _build_full_tools_app(
    *,
    runtime_backend: str = "local",
    sharing_scope_service=None,
    outputs_scope_service=None,
):
    """Build the production application with deterministic panel collaborators."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app.notify = MagicMock()
    state = RuntimeSourceState(
        active_source=runtime_backend,
        server_configured=runtime_backend == "server",
    )
    app.runtime_policy.state = state
    app._publish_runtime_policy_projection(state)
    app.server_sharing_scope_service = sharing_scope_service
    app.server_outputs_scope_service = outputs_scope_service
    return app


@asynccontextmanager
async def _mounted_tools_window(app):
    async with app.run_test() as pilot:
        app.push_screen(ToolsSettingsScreen(app))
        await pilot.pause()
        yield app.screen.query_one(ToolsSettingsWindow), pilot


@asynccontextmanager
async def mount_settings_window(config_dict: dict, temp_config_path: Path, monkeypatch):
    """Context manager that mounts a live ToolsSettingsWindow with both config-path patches applied.

    Writes config_dict to temp_config_path, patches DEFAULT_CONFIG_PATH, and sets
    the TLDW_CONFIG_PATH environment variable. Both are required: app code resolves
    the effective config path via config._get_effective_config_path(), which prefers
    the TLDW_CONFIG_PATH environment variable. Tests/conftest.py autouse fixture
    sets TLDW_CONFIG_PATH per-test to a separate bootstrap file; patching only
    DEFAULT_CONFIG_PATH meant the widget read a config the test never wrote.

    Args:
        config_dict: Dictionary of configuration to write to the temporary config file.
        temp_config_path: Path where the temporary config.toml file will be written.
        monkeypatch: pytest monkeypatch fixture for patching DEFAULT_CONFIG_PATH and environment.

    Yields:
        Tuple of the production ToolsSettingsWindow and full-app pilot.
    """
    create_dummy_config(temp_config_path, config_dict)
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", temp_config_path)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(temp_config_path))

    app = _build_full_tools_app()
    async with _mounted_tools_window(app) as (window, pilot):
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

    # `_get_effective_config_path()` (the function every read/write path in
    # this module actually calls -- `load_cli_config_and_ensure_existence`,
    # `get_provider_readiness`, `save_setting_to_cli_config`, etc.) checks the
    # `TLDW_CONFIG_PATH` environment variable FIRST and only falls back to
    # `DEFAULT_CONFIG_PATH` when it is unset. Tests/conftest.py's autouse
    # `isolate_test_environment` fixture always sets `TLDW_CONFIG_PATH` (to a
    # sandbox path under a DIFFERENT `tmp_path`-derived directory) for every
    # test in the suite, so the `DEFAULT_CONFIG_PATH` patch above is silently
    # ineffective for any code that resolves its path through that function --
    # it keeps reading/writing the global sandbox file, never `temp_config_path`.
    # Re-pointing the env var here is what actually redirects those calls.
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(temp_config_path))
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


@pytest.fixture
def mock_app_instance():
    """Provide the full production application used by mounted window tests."""
    return _build_full_tools_app()


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

    async with _mounted_tools_window(mock_app_instance) as (window, _pilot):
        yield window




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
    assert loaded_text_area_config["general"]["model"] == "gpt-4"
    assert loaded_text_area_config["api_keys"] == {"openai": "sk-..."}


@pytest.mark.asyncio
async def test_save_config_values(settings_window: ToolsSettingsWindow, temp_config_path: Path, mock_app_instance):
    """Test if configuration values can be saved correctly."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    save_button = settings_window.query_one("#save-config-button", Button)

    new_config_dict = {"user": {"name": "test_user", "theme": "blue"}}
    config_text_area.text = toml.dumps(new_config_dict)

    # Simulate button press by calling the handler
    await settings_window.on_button_pressed(Button.Pressed(save_button))

    mock_app_instance.notify.assert_called_with(
        "Configuration saved successfully!",
        severity="successful",
    )

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
    assert toml.loads(config_text_area.text)["settings"] == original_disk_config[
        "settings"
    ]

    # 3. Modify the TextArea to simulate user changes (these are not saved yet)
    user_modified_text_dict = {"settings": {"feature_x": False, "version": 2}}
    config_text_area.text = toml.dumps(user_modified_text_dict)
    assert toml.loads(config_text_area.text) == user_modified_text_dict  # Verify change in TextArea

    # 4. Simulate reload button press again
    await settings_window.on_button_pressed(Button.Pressed(reload_button))
    mock_app_instance.notify.assert_called_with("Configuration reloaded.")  # Called again

    # 5. Verify TextArea content is reverted to original_disk_config (ignoring user_modified_text_dict)
    assert toml.loads(config_text_area.text)["settings"] == original_disk_config[
        "settings"
    ]


@pytest.mark.asyncio
async def test_save_invalid_toml_format(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test saving invalid TOML data reports an error."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)
    save_button = settings_window.query_one("#save-config-button", Button)

    invalid_toml_text = "this is not valid toml { text = blah"
    config_text_area.text = invalid_toml_text

    await settings_window.on_button_pressed(Button.Pressed(save_button))

    message = mock_app_instance.notify.call_args.args[0]
    assert message.startswith("Error: Invalid TOML format:")
    assert mock_app_instance.notify.call_args.kwargs == {"severity": "error"}


# Test for save I/O error (conceptual - requires mocking 'open')
@pytest.mark.skip(reason="Complex to mock built-in open reliably for this specific write operation only")
@pytest.mark.asyncio
async def test_save_io_error(settings_window: ToolsSettingsWindow, mock_app_instance, monkeypatch):
    """Test saving config when an IOError occurs."""
    config_text_area = settings_window.query_one("#config-text-area", TextArea)

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
# TASK-962: raw-TOML save must resolve the effective config path and
# write atomically, matching TASK-851's three encryption entry points.
# ===========================================

@pytest.mark.asyncio
async def test_save_raw_toml_config_writes_effective_path_not_default_decoy(
    monkeypatch, tmp_path
):
    """Regression test for TASK-962 AC#3.

    _save_raw_toml_config must resolve its write target through
    config._get_effective_config_path() (which honors a TLDW_CONFIG_PATH
    profile override), not the hardcoded DEFAULT_CONFIG_PATH literal.
    Simulates a user running with an active profile: TLDW_CONFIG_PATH
    points at the profile file, DEFAULT_CONFIG_PATH is a distinct decoy
    that a correct fix must never touch.
    """
    profile_path = tmp_path / "profile" / "config.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    decoy_path = tmp_path / "default_home" / "config.toml"
    create_dummy_config(profile_path, {"initial": "value"})

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile_path))
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", decoy_path)

    app = _build_full_tools_app()
    async with _mounted_tools_window(app) as (window, _pilot):

        config_text_area = window.query_one("#config-text-area", TextArea)
        new_config_dict = {"user": {"name": "profile_user"}}
        config_text_area.text = toml.dumps(new_config_dict)

        await window._save_raw_toml_config()

    # The change landed in the file config._get_effective_config_path()
    # resolves to (derived via that accessor, not a re-spelled literal).
    assert tldw_chatbook.config._get_effective_config_path() == (
        tldw_chatbook.config.lexical_path(profile_path)
    )
    saved = toml.load(profile_path)
    assert saved == new_config_dict

    # The decoy DEFAULT_CONFIG_PATH was never created or touched.
    assert not decoy_path.exists()


@pytest.mark.asyncio
async def test_save_raw_toml_config_is_atomic_on_serialization_failure(
    monkeypatch, temp_config_path
):
    """Regression test for TASK-962 AC#4.

    A failure partway through serializing the new config (crash, kill -9
    equivalent) must never truncate the on-disk config file: the write must
    go through a write-temp-then-replace helper, not a plain
    open(path, 'w') + toml.dump (which truncates on open, before any new
    content -- or a raised exception -- ever reaches the file).
    """
    initial_config = {"initial": "value", "user": {"name": "before"}}
    async with mount_settings_window(
        initial_config, temp_config_path, monkeypatch
    ) as (window, pilot):
        original_bytes = temp_config_path.read_bytes()

        config_text_area = window.query_one("#config-text-area", TextArea)
        config_text_area.text = toml.dumps({"user": {"name": "after-crash"}})

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated crash while serializing config")

        # Patch both the string-returning and file-writing serializers --
        # which one a given implementation calls is deliberately treated as
        # an implementation detail here (a plain open(path, "w")+toml.dump
        # writer uses toml.dump; the atomic write-temp-then-replace helper
        # used today calls toml.dumps first). Not toml.loads -- parsing the
        # user's edited text must still succeed so the failure is isolated
        # to the write step itself.
        monkeypatch.setattr(tldw_chatbook.config.toml, "dumps", _boom)
        monkeypatch.setattr(tldw_chatbook.config.toml, "dump", _boom)

        await window._save_raw_toml_config()

    # The file must be byte-for-byte unchanged: a plain open(path, "w")
    # would have already truncated it before toml.dumps ever ran.
    assert temp_config_path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_save_raw_toml_config_roundtrips_with_no_profile_override(
    monkeypatch, tmp_path
):
    """Regression test for TASK-962 AC#5.

    With no TLDW_CONFIG_PATH override active, saving raw TOML config must
    still round-trip correctly through DEFAULT_CONFIG_PATH -- the fix for
    AC#1/AC#2 must not regress the no-profile case.
    """
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_path = tmp_path / "config.toml"
    create_dummy_config(config_path, {"initial": "value"})
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", config_path)

    app = _build_full_tools_app()
    async with _mounted_tools_window(app) as (window, _pilot):

        config_text_area = window.query_one("#config-text-area", TextArea)
        new_config_dict = {"user": {"name": "default_user"}}
        config_text_area.text = toml.dumps(new_config_dict)

        await window._save_raw_toml_config()

    saved = toml.load(config_path)
    assert saved == new_config_dict


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
    """The production window exposes every current database action contract."""
    nav_button = settings_window.query_one("#ts-nav-db-tools", Button)
    assert nav_button.label.plain == "Database Tools"

    content_area = settings_window.query_one("#ts-view-db-tools")
    assert content_area is not None
    for db_name in (
        "chachanotes",
        "media",
        "prompts",
        "evals",
        "rag",
        "subscriptions",
    ):
        assert content_area.query_one(f"#db-vacuum-{db_name}", Button)
        assert content_area.query_one(f"#db-backup-{db_name}", Button)
        assert content_area.query_one(f"#db-last-backup-{db_name}", Static)
        assert content_area.query_one(f"#db-restore-{db_name}", Button)
        assert content_area.query_one(f"#db-check-{db_name}", Button)

    assert content_area.query_one("#db-vacuum-all", Button)
    assert content_area.query_one("#db-backup-all", Button)
    assert content_area.query_one("#db-check-integrity", Button)
    widget_ids = [node.id for node in content_area.walk_children() if node.id]
    assert len(widget_ids) == len(set(widget_ids))


@pytest.mark.asyncio
async def test_create_chatbook_button(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test that chatbook creation button exists and works."""
    # Find the create chatbook button
    create_button = settings_window.query_one("#db-create-chatbook", Button)
    assert create_button is not None
    assert "Create Chatbook" in create_button.label.plain

    # Mock the chatbook creation window
    with (
        patch("tldw_chatbook.UI.ChatbookCreationWindow.ChatbookCreationWindow"),
        patch.object(
            mock_app_instance,
            "push_screen",
            new=AsyncMock(return_value=None),
        ) as mock_push_screen,
    ):
        await settings_window.on_button_pressed(Button.Pressed(create_button))

        # Should push the chatbook creation screen
        mock_push_screen.assert_called_once()


@pytest.mark.asyncio
async def test_import_chatbook_button(settings_window: ToolsSettingsWindow, mock_app_instance):
    """Test that chatbook import button exists and works."""
    # Find the import chatbook button
    import_button = settings_window.query_one("#db-import-chatbook", Button)
    assert import_button is not None
    assert "Import Chatbook" in import_button.label.plain

    # Mock file picker for import
    with patch.object(
        mock_app_instance,
        "push_screen",
        new=AsyncMock(return_value=None),
    ) as mock_push_screen:
        await settings_window.on_button_pressed(Button.Pressed(import_button))

        # Should push the file picker
        mock_push_screen.assert_called_once()


@pytest.mark.asyncio
async def test_database_error_handling(settings_window: ToolsSettingsWindow, mock_app_instance, mock_database_path_lookup):
    """Test error handling for database operations."""
    # Mock a database operation to fail
    with patch("sqlite3.connect", side_effect=sqlite3.Error("Database is locked")):
        # Try to vacuum a database
        vacuum_button = settings_window.query_one("#db-vacuum-chachanotes", Button)
        await settings_window.on_button_pressed(Button.Pressed(vacuum_button))

        # Should show error notification
        mock_app_instance.notify.assert_called()
        calls = mock_app_instance.notify.call_args_list
        assert any("error" in str(call).lower() for call in calls)


@pytest.mark.asyncio
async def test_tools_settings_window_no_longer_exposes_unified_mcp_view():
    """MCP Hub Phase 6 Task 5: the legacy `UnifiedMCPPanel` embed (nav
    button + pane) is fully retired from Tools & Settings -- MCP management
    now lives entirely in the MCP Hub screen/workbench. The window must
    still compose and its other nav destinations must still work; only the
    "Unified MCP" entry point is gone.
    """
    app = _build_full_tools_app()
    async with _mounted_tools_window(app) as (window, _pilot):

        assert not window.query("#ts-nav-unified-mcp")
        assert not window.query("#ts-view-unified-mcp")
        assert not window.query("#unified-mcp-panel")

        # The window still composes and other navigation still works --
        # deleting the Unified MCP embed must not have taken the rest of
        # the window down with it.
        nav_button = window.query_one("#ts-nav-appearance", Button)
        await window.on_button_pressed(Button.Pressed(nav_button))
        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-appearance"


_RETIRED_MCP_MODULE_NAMES = {"unified_mcp_panel", "unified_mcp_sections"}
_RETIRED_MCP_SYMBOL_NAMES = {
    "UnifiedMCPPanel",
    "render_unified_mcp_section",
    "LAYOUT_MODE_FULL",
    "LAYOUT_MODE_COMPACT_WORKBENCH",
}


def _mcp_retirement_offense(py_file: Path) -> str | None:
    """One-line description of a real (non-comment, non-docstring) reference
    to a retired MCP module/symbol in `py_file`, or `None` if it's clean.

    AST-based rather than a plain substring search deliberately: this repo's
    surviving MCP Hub modules/tests are FULL of historical prose comments
    and docstrings explaining what `unified_mcp_panel.py`/`unified_mcp_
    sections.py` used to do and why a given piece of redaction/shim logic
    still exists -- a substring grep would flag every one of those as a
    false positive. Parsing the source and only inspecting `import`/`from
    ... import` statements and bare `Name` references (never `Constant`
    string literals, which is what a comment or docstring becomes) catches
    a REAL importer/reference while staying silent on prose that merely
    names the retired module for historical context.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _RETIRED_MCP_MODULE_NAMES or any(
                    part in _RETIRED_MCP_MODULE_NAMES for part in alias.name.split(".")
                ):
                    return f"import {alias.name} (line {node.lineno})"
        elif isinstance(node, ast.ImportFrom):
            module_parts = (node.module or "").split(".")
            if any(part in _RETIRED_MCP_MODULE_NAMES for part in module_parts):
                return f"from {node.module} import ... (line {node.lineno})"
            for alias in node.names:
                if alias.name in _RETIRED_MCP_SYMBOL_NAMES:
                    return f"from {node.module} import {alias.name} (line {node.lineno})"
        elif isinstance(node, ast.Name) and node.id in _RETIRED_MCP_SYMBOL_NAMES:
            return f"reference to {node.id} (line {node.lineno})"
        elif isinstance(node, ast.Attribute) and node.attr in _RETIRED_MCP_SYMBOL_NAMES:
            return f"attribute reference to {node.attr} (line {node.lineno})"
    return None


def test_unified_mcp_panel_modules_have_zero_importers_repo_wide():
    """Grep-gate (MCP Hub Phase 6 Task 5): `unified_mcp_panel.py` and
    `unified_mcp_sections.py` (plus their `UnifiedMCPPanel`/
    `render_unified_mcp_section`/`LAYOUT_MODE_*` symbols) are deleted along
    with their own test file -- this asserts no other module in the tree
    still imports or references them, so a stray import can't silently
    resurrect a dependency on files that no longer exist (an `ImportError`
    at collection/runtime instead of a clear, fast, repo-wide check here).
    See `_mcp_retirement_offense()` for why this is AST-based rather than a
    plain substring search.
    """
    project_root = Path(__file__).resolve().parents[2]
    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    for search_root in (project_root / "tldw_chatbook", project_root / "Tests"):
        for py_file in search_root.rglob("*.py"):
            if py_file.resolve() == this_file:
                continue
            offense = _mcp_retirement_offense(py_file)
            if offense is not None:
                offenders.append(f"{py_file.relative_to(project_root)}: {offense}")
    assert offenders == [], f"stray references to retired MCP modules: {offenders}"
    assert not (project_root / "tldw_chatbook" / "UI" / "MCP_Modules" / "unified_mcp_panel.py").exists()
    assert not (project_root / "tldw_chatbook" / "UI" / "MCP_Modules" / "unified_mcp_sections.py").exists()
    assert not (project_root / "Tests" / "UI" / "test_unified_mcp_panel.py").exists()


@pytest.mark.asyncio
async def test_tools_settings_window_exposes_sharing_view():
    app = _build_full_tools_app(
        runtime_backend="server",
        sharing_scope_service=MagicMock(),
    )
    async with _mounted_tools_window(app) as (window, _pilot):
        nav_button = window.query_one("#ts-nav-sharing", Button)

        assert nav_button.label.plain == "Sharing"

        await window.on_button_pressed(Button.Pressed(nav_button))

        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-sharing"
        assert window.query_one("#sharing-panel", SharingPanel) is not None


@pytest.mark.asyncio
async def test_tools_settings_window_exposes_outputs_view():
    app = _build_full_tools_app(
        runtime_backend="server",
        sharing_scope_service=MagicMock(),
        outputs_scope_service=MagicMock(),
    )
    async with _mounted_tools_window(app) as (window, _pilot):
        nav_button = window.query_one("#ts-nav-outputs", Button)

        assert nav_button.label.plain == "Outputs"

        await window.on_button_pressed(Button.Pressed(nav_button))

        content_switcher = window.query_one("#tools-settings-content-pane")
        assert content_switcher.current == "ts-view-outputs"
        assert window.query_one("#outputs-panel", OutputsPanel) is not None


@pytest.mark.asyncio
async def test_sharing_panel_rejects_local_mode_with_explicit_guidance():
    scope_service = MagicMock()
    app = _build_full_tools_app(
        runtime_backend="local",
        sharing_scope_service=scope_service,
    )

    async with _mounted_tools_window(app) as (window, pilot):
        panel = window.query_one(SharingPanel)
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
    app = _build_full_tools_app(
        runtime_backend="server",
        sharing_scope_service=scope_service,
    )

    async with _mounted_tools_window(app) as (window, pilot):
        panel = window.query_one(SharingPanel)
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
    app = _build_full_tools_app(
        runtime_backend="local",
        outputs_scope_service=scope_service,
    )

    async with _mounted_tools_window(app) as (window, pilot):
        panel = window.query_one(OutputsPanel)
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
    app = _build_full_tools_app(
        runtime_backend="server",
        outputs_scope_service=scope_service,
    )

    async with _mounted_tools_window(app) as (window, pilot):
        panel = window.query_one(OutputsPanel)
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


# ---------------------------------------------------------------------------
# TASK-899: DB maintenance panel must resolve real, profile-aware database
# paths (not hardcoded, profile-unaware literals) and must fail loudly
# instead of silently doing nothing when a path can't be resolved.
# ---------------------------------------------------------------------------

_ALL_MAINTENANCE_DB_NAMES = [
    "chachanotes",
    "media",
    "prompts",
    "evals",
    "rag",
    "subscriptions",
]


def _notify_calls_with_severity(mock_notify, severity: str):
    return [
        c for c in mock_notify.call_args_list if c.kwargs.get("severity") == severity
    ]


def test_db_path_resolvers_cover_exactly_the_known_databases():
    """Guard against a resolver silently disappearing (or a stale one being
    left behind) from the single source-of-truth map."""
    assert set(ToolsSettingsWindow._DB_PATH_RESOLVERS.keys()) == set(
        _ALL_MAINTENANCE_DB_NAMES
    )


def test_import_chatbook_paths_reuse_the_single_source_of_truth():
    """AC: 'The duplicated, disagreeing per-key defaults inside the file are
    gone.' _import_chatbook() used to hardcode its own second copy of the
    per-database default paths, disagreeing with _get_database_path()'s copy
    on the very same keys (TASK-899).

    Dev-reconciliation note: dev factored the Chatbook importer's key
    contract (``ChaChaNotes``/``Prompts``/``Media``, distinct from this
    window's own ``chachanotes``/``prompts``/``media`` names) into a
    dedicated ``Chatbooks.database_paths.get_chatbook_database_paths()``
    helper that itself calls the very same canonical ``config.py``
    resolvers ``_DB_PATH_RESOLVERS`` wraps -- see
    ``Tests/Chatbooks/test_chatbook_database_paths.py::
    test_chatbook_database_paths_use_canonical_runtime_getters``. Every
    other chatbook-facing surface in the app (the creation/import wizards,
    the export management window) already routes through this same helper
    (``test_chatbook_surfaces_do_not_embed_database_defaults``), so
    ``_get_chatbook_import_database_paths`` participating in that
    established, single-source-of-truth convention -- rather than
    reaching into ``_DB_PATH_RESOLVERS`` directly and re-mapping key names
    itself -- is the codebase-wide pattern, not a second, disagreeing
    source of truth. Only the no-hardcoded-literal-defaults assertion
    below is still load-bearing; which specific canonical entry point is
    used to reach config.py is not."""
    import inspect

    source = inspect.getsource(ToolsSettingsWindow._import_chatbook)

    disagreeing_literals = [
        "tldw_cli_media_v2.db",
        "tldw_cli_prompts.db",
        "tldw_media_db.db",
        "tldw_prompts_db.db",
        "tldw_evals_db.db",
        "tldw_rag_db.db",
    ]
    for literal in disagreeing_literals:
        assert literal not in source, (
            f"stale duplicate default {literal!r} still hardcoded in _import_chatbook"
        )

    assert (
        "_get_database_path" in source
        or "_DB_PATH_RESOLVERS" in source
        or "_get_chatbook_import_database_paths" in source
        or "get_chatbook_database_paths" in source
    )

    # And that helper must itself resolve to the same canonical resolvers,
    # not a fresh, independent copy of the per-database defaults.
    helper_source = inspect.getsource(
        ToolsSettingsWindow._get_chatbook_import_database_paths
    )
    for literal in disagreeing_literals:
        assert literal not in helper_source, (
            f"stale duplicate default {literal!r} hardcoded in "
            "_get_chatbook_import_database_paths"
        )
    assert "get_chatbook_database_paths" in helper_source


def test_no_bare_call_from_thread_calls_in_tools_settings_window():
    """Guard against this bug class recurring anywhere in the file.

    ToolsSettingsWindow extends Container, and Container (like Widget in
    general) has no ``call_from_thread`` of its own -- only App does. A bare
    ``self.call_from_thread(...)`` inside a ``@work(thread=True)`` worker
    therefore raises AttributeError instead of reaching the UI, silently
    swallowing both success and error notifications. This was found twice in
    this file (the four single-db maintenance workers, then
    _import_chatbook_worker) -- always use ``self.app.call_from_thread(...)``.
    """
    import inspect
    import re

    import tldw_chatbook.UI.Tools_Settings_Window as module
    from textual.app import App
    from textual.containers import Container

    source = inspect.getsource(module)
    bare_calls = re.findall(r"self\.call_from_thread\(", source)
    assert not bare_calls, (
        f"found {len(bare_calls)} bare 'self.call_from_thread(' call(s) in "
        "Tools_Settings_Window.py -- use 'self.app.call_from_thread(' instead"
    )

    # Documents WHY the bare form is wrong: Container (ToolsSettingsWindow's
    # base) genuinely has no call_from_thread of its own -- only App does. If
    # this ever stops holding (e.g. Textual adds it to Widget), the source
    # scan above is still the operative guard.
    assert not hasattr(Container, "call_from_thread")
    assert hasattr(App, "call_from_thread")
    assert not hasattr(ToolsSettingsWindow, "call_from_thread")


@pytest.mark.asyncio
async def test_get_database_path_resolves_via_config_resolvers_and_honours_profile(
    monkeypatch, temp_config_path
):
    """_get_database_path must delegate to config.py's real, profile-aware
    resolvers for every known database, and an unknown name must resolve to
    None rather than a hardcoded guess (TASK-899)."""
    from tldw_chatbook.config import (
        get_chachanotes_db_path,
        get_media_db_path,
        get_prompts_db_path,
        get_evals_db_path,
        get_rag_indexing_db_path,
        get_subscriptions_db_path,
        get_user_folder_name,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (window, pilot):
        expected = {
            "chachanotes": get_chachanotes_db_path(),
            "media": get_media_db_path(),
            "prompts": get_prompts_db_path(),
            "evals": get_evals_db_path(),
            "rag": get_rag_indexing_db_path(),
            "subscriptions": get_subscriptions_db_path(),
        }
        profile = get_user_folder_name()
        for db_name, expected_path in expected.items():
            resolved = window._get_database_path(db_name, {})
            assert resolved == expected_path, db_name
            # Every real database must live under the configured profile
            # directory, not directly under ~/.local/share/tldw_cli.
            assert resolved.parent.name == profile, resolved

        # An unknown database name must resolve to None, not a hardcoded guess.
        assert window._get_database_path("not-a-real-database", {}) is None


def test_evals_db_path_matches_orchestrator_resolution():
    """config.get_evals_db_path() must agree exactly with where
    EvaluationOrchestrator actually opens the Evals DB, or the maintenance
    panel would operate on a different file than the app uses (TASK-899)."""
    from tldw_chatbook.config import get_evals_db_path
    from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

    orchestrator = EvaluationOrchestrator(client_id="test_evals_path_agreement")
    orchestrator_path = Path(orchestrator.db.db_path)

    resolved_path = get_evals_db_path()

    assert resolved_path == orchestrator_path
    assert resolved_path.name == "evals.db"


def test_rag_indexing_db_path_matches_ingestion_module_resolution():
    """config.get_rag_indexing_db_path() must agree exactly with where
    ingestion_indexing._default_indexing_db() actually opens the RAG
    indexing-state database (TASK-899)."""
    from tldw_chatbook.config import get_rag_indexing_db_path
    from tldw_chatbook.RAG_Search.ingestion_indexing import _default_indexing_db

    indexing_db = _default_indexing_db()
    assert indexing_db is not None

    resolved_path = get_rag_indexing_db_path()

    assert Path(indexing_db.db_path) == resolved_path
    assert resolved_path.name == "rag_indexing.db"


@pytest.mark.parametrize(
    "db_name",
    [name for name in _ALL_MAINTENANCE_DB_NAMES if name != "evals"],
)
def test_backup_then_restore_round_trips_at_the_resolved_path(
    db_name, monkeypatch, tmp_path
):
    """Direct workers must back up and restore the resolver-selected path."""
    import tldw_chatbook.UI.Tools_Settings_Window as tools_settings_module

    data_dir = tmp_path / "data"
    db_path = tmp_path / "live" / f"{db_name}.db"
    monkeypatch.setattr(
        tools_settings_module,
        "get_user_data_dir",
        lambda: data_dir,
    )

    notify = MagicMock()

    def call_from_thread(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    window = SimpleNamespace(
        config_data={"database": {}},
        app=SimpleNamespace(call_from_thread=call_from_thread),
        app_instance=SimpleNamespace(notify=notify),
    )
    window._get_database_path = lambda _name, _config: db_path
    window._validate_maintenance_path = MethodType(
        ToolsSettingsWindow._validate_maintenance_path,
        window,
    )
    window._get_schema_version = MethodType(
        ToolsSettingsWindow._get_schema_version,
        window,
    )
    window._update_last_backup_status = lambda *_args: None
    window._update_database_sizes = lambda: None

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("CREATE TABLE marker (value TEXT)")
        conn.execute("INSERT INTO marker VALUES ('original')")
        conn.commit()

    ToolsSettingsWindow._backup_single_worker.__wrapped__(window, db_name)

    assert _notify_calls_with_severity(notify, "success"), (
        f"backup did not report success for {db_name}: {notify.call_args_list}"
    )
    notify.reset_mock()
    backup_dir = data_dir / "backups" / db_name
    backup_files = sorted(backup_dir.glob(f"{db_name}_backup_*.db"))
    assert backup_files, f"no backup file was written for {db_name} at {backup_dir}"
    backup_path = backup_files[-1]

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("DELETE FROM marker")
        conn.execute("INSERT INTO marker VALUES ('corrupted')")
        conn.commit()

    ToolsSettingsWindow._restore_single_worker.__wrapped__(
        window,
        db_name,
        backup_path,
    )

    assert _notify_calls_with_severity(notify, "success"), (
        f"restore did not report success for {db_name}: {notify.call_args_list}"
    )
    with closing(sqlite3.connect(str(db_path))) as restored_conn:
        value = restored_conn.execute("SELECT value FROM marker").fetchone()[0]
    assert value == "original"


@pytest.mark.asyncio
async def test_restore_refuses_live_evals_database_without_partial_replacement(
    monkeypatch, temp_config_path
):
    """The production app keeps the evaluations database open.

    A live restore must fail closed and leave the current file untouched;
    replacing a database while an application-owned connection is active
    would make the in-memory and on-disk state diverge.
    """
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        _pilot,
    ):
        db_path = window._get_database_path("evals", {})
        assert db_path is not None

        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES ('live')")
            conn.commit()

        backup_worker = window._backup_single_worker("evals")
        await backup_worker.wait()
        backup_dir = tldw_chatbook.config.get_user_data_dir() / "backups" / "evals"
        backup_path = sorted(backup_dir.glob("evals_backup_*.db"))[-1]

        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("UPDATE marker SET value = 'current'")
            conn.commit()

        window.app_instance.notify.reset_mock()
        restore_worker = window._restore_single_worker("evals", backup_path)
        await restore_worker.wait()

        assert not _notify_calls_with_severity(
            window.app_instance.notify,
            "success",
        )
        errors = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert len(errors) == 1
        assert "live restore is unavailable" in errors[0].args[0]
        with closing(sqlite3.connect(str(db_path))) as conn:
            value = conn.execute("SELECT value FROM marker").fetchone()[0]
        assert value == "current"


@pytest.mark.asyncio
async def test_unresolvable_database_fails_loudly_instead_of_silently_succeeding(
    monkeypatch, temp_config_path
):
    """A database name with no resolver must produce an error notification
    from every maintenance worker -- never a silent no-op and never a false
    'success' (TASK-899)."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (window, pilot):
        assert window._get_database_path("not-a-real-database", {}) is None

        for worker_name, extra_args in (
            ("_vacuum_single_worker", ()),
            ("_backup_single_worker", ()),
            ("_check_single_worker", ()),
            ("_restore_single_worker", (Path("/nonexistent/backup.db"),)),
        ):
            window.app_instance.notify.reset_mock()
            worker = getattr(window, worker_name)("not-a-real-database", *extra_args)
            await worker.wait()

            calls = window.app_instance.notify.call_args_list
            assert calls, f"{worker_name} produced no notification at all"
            assert _notify_calls_with_severity(window.app_instance.notify, "error"), (
                f"{worker_name} did not report an error for an unresolvable database: {calls}"
            )
            assert not _notify_calls_with_severity(window.app_instance.notify, "success"), (
                f"{worker_name} falsely reported success for an unresolvable database: {calls}"
            )


@pytest.mark.asyncio
async def test_missing_database_file_fails_loudly_instead_of_silently_succeeding(
    monkeypatch, temp_config_path
):
    """A resolvable path whose file doesn't exist yet (e.g. RAG never used in
    this profile) must not be reported as a silent success either -- it must
    say something, and that something must not be 'success' (TASK-899)."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (window, pilot):
        db_path = window._get_database_path("rag", {})
        assert db_path is not None
        assert not db_path.exists()

        for worker_name in (
            "_vacuum_single_worker",
            "_backup_single_worker",
            "_check_single_worker",
        ):
            window.app_instance.notify.reset_mock()
            worker = getattr(window, worker_name)("rag")
            await worker.wait()

            calls = window.app_instance.notify.call_args_list
            assert calls, f"{worker_name} produced no notification for a missing database"
            assert not _notify_calls_with_severity(window.app_instance.notify, "success"), (
                f"{worker_name} falsely reported success for a missing database file: {calls}"
            )


@pytest.mark.asyncio
async def test_restore_creates_missing_target_directory_for_a_custom_db_path(
    monkeypatch, temp_config_path, tmp_path
):
    """A configured custom database path is a legitimate restore target even
    when its directory has never been created yet -- DB/base_db.py creates a
    database's parent directory as a side effect of opening it, and restore
    must behave consistently rather than refusing outright (TASK-899 finding
    4 fix). Regression guard for the since-fixed bug where
    _restore_single_worker treated a merely-missing directory the same as an
    unresolvable/phantom path."""
    custom_db_path = tmp_path / "not_created_yet" / "custom_chachanotes.db"
    assert not custom_db_path.parent.exists()

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (window, pilot):
        # Stand in for a user-configured custom database path by overriding
        # the resolver map directly (an instance-level shadow of the class
        # attribute, so it can't leak to other tests) rather than the config
        # file: this test app's TLDW_CONFIG_PATH (set by the autouse
        # isolate_test_environment fixture) always wins over the
        # monkeypatched DEFAULT_CONFIG_PATH that mount_settings_window
        # writes to, so a config-file-based override wouldn't actually be
        # read here. This still exercises exactly the code
        # _restore_single_worker calls.
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)
        window._DB_PATH_RESOLVERS["chachanotes"] = lambda: custom_db_path

        resolved = window._get_database_path("chachanotes", {})
        assert resolved == custom_db_path
        assert not resolved.parent.exists()

        # A standalone backup file the "user" is restoring from, independent
        # of the app's own backup machinery.
        backup_path = tmp_path / "external_backup.db"
        with closing(sqlite3.connect(str(backup_path))) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES ('restored')")
            conn.commit()

        worker = window._restore_single_worker("chachanotes", backup_path)
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        assert _notify_calls_with_severity(window.app_instance.notify, "success"), (
            f"restore to a not-yet-created custom directory must succeed: {calls}"
        )
        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            f"restore to a not-yet-created custom directory must not error: {calls}"
        )

        assert resolved.parent.exists()
        with closing(sqlite3.connect(str(resolved))) as restored_conn:
            value = restored_conn.execute("SELECT value FROM marker").fetchone()[0]
        assert value == "restored"


@pytest.mark.asyncio
async def test_restore_refuses_a_dangerous_backup_path_via_path_validation(
    monkeypatch, temp_config_path
):
    """The user-selected backup_path must be routed through
    Utils/path_validation.py before it reaches shutil.copy2 -- a path
    containing a dangerous pattern must be refused with a clear,
    actionable error naming the offending path, never silently ignored and
    never an unhandled exception out of the worker thread (TASK-899 finding
    1).

    The dangerous-named file is created for real so that, without the
    validation call, the restore would otherwise succeed (proving this test
    exercises path validation itself, not an incidental FileNotFoundError
    from shutil.copy2 -- a real file at the same path would only produce
    that error if it were missing)."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (window, pilot):
        db_path = window._get_database_path("chachanotes", {})
        assert db_path is not None
        assert not db_path.exists()

        dangerous_backup_path = db_path.parent / "evil;rm -rf.db"
        dangerous_backup_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(dangerous_backup_path))) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES ('should not be restored')")
            conn.commit()

        worker = window._restore_single_worker("chachanotes", dangerous_backup_path)
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        assert calls, "no notification at all for a rejected backup path"
        error_calls = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert error_calls, f"dangerous backup path was not refused: {calls}"
        assert not _notify_calls_with_severity(window.app_instance.notify, "success"), (
            f"dangerous backup path falsely reported success: {calls}"
        )
        # The error must specifically be path-validation's rejection (not a
        # generic failure), and must name the offending path so the user can
        # tell which file-picker selection was refused.
        assert any("dangerous pattern" in str(c) for c in error_calls), error_calls
        assert any(str(dangerous_backup_path) in str(c) for c in error_calls), error_calls

        # No partial/failed write occurred against the live database -- the
        # rejected source file's content must never have reached it.
        assert not db_path.exists()


# ---------------------------------------------------------------------------
# TASK-927: the bulk ("all databases") maintenance workers -- vacuum, backup,
# integrity check -- and the conversation/notes/characters export workers
# carried their own separate copies of the same hardcoded, profile-unaware
# literals TASK-899 removed from the single-database workers and the
# Database Config settings form. These tests prove the bulk workers and the
# form both now go through the same _DB_PATH_RESOLVERS resolvers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vacuum_worker_operates_on_resolved_paths_not_literals(
    monkeypatch, temp_config_path
):
    """'Vacuum All Databases' (_vacuum_worker) must vacuum the ChaChaNotes
    database at the same profile-aware path _get_database_path resolves,
    not a hardcoded ~/.local/share/tldw_cli/<literal>.db path with no
    profile segment (TASK-927). Proven by creating a real, padded (then
    trimmed) database ONLY at the resolved path and asserting the worker
    actually shrinks that specific file -- if the worker instead targeted
    the old hardcoded literal path, this file would never be touched and
    would not shrink."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        resolved_path = window._get_database_path("chachanotes", {})
        assert resolved_path is not None
        old_literal_path = (
            Path.home()
            / ".local"
            / "share"
            / "tldw_cli"
            / "tldw_chatbook_ChaChaNotes.db"
        )
        # Sanity: the resolved path is genuinely profile-scoped, not the
        # bare literal the pre-fix bulk worker hardcoded.
        assert resolved_path != old_literal_path

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        db = CharactersRAGDB(str(resolved_path), "test_setup")
        db.close_connection()
        with closing(sqlite3.connect(str(resolved_path))) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS pad (value TEXT)")
            conn.executemany(
                "INSERT INTO pad (value) VALUES (?)",
                [("x" * 5000,) for _ in range(200)],
            )
            conn.commit()
            conn.execute("DELETE FROM pad")
            conn.commit()
        size_before = resolved_path.stat().st_size
        assert not old_literal_path.exists()

        worker = window._vacuum_worker()
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        assert _notify_calls_with_severity(window.app_instance.notify, "success"), (
            f"vacuum did not report success: {calls}"
        )
        size_after = resolved_path.stat().st_size
        assert size_after < size_before, (
            f"vacuum did not shrink the database at the real resolved path "
            f"({size_before} -> {size_after}); the worker may be targeting "
            f"a different (e.g. hardcoded-literal) path"
        )
        # The old, non-profile-scoped literal location must never have been
        # created/touched by this operation.
        assert not old_literal_path.exists()


@pytest.mark.asyncio
async def test_vacuum_all_fails_loudly_for_an_unresolvable_database(
    monkeypatch, temp_config_path
):
    """The bulk 'Vacuum All Databases' worker must report an unresolvable
    database loudly -- never silently drop it from the run while reporting
    overall success (TASK-927, extending TASK-899's fail-loudly guarantee
    from the single-database workers to the bulk one)."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)

        def _boom():
            raise RuntimeError("simulated resolver failure")

        window._DB_PATH_RESOLVERS["media"] = _boom

        worker = window._vacuum_worker()
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        error_calls = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert error_calls, f"unresolvable Media database was not reported: {calls}"
        assert any("Media" in str(c) for c in error_calls), error_calls


@pytest.mark.asyncio
async def test_integrity_all_fails_loudly_for_an_unresolvable_database(
    monkeypatch, temp_config_path
):
    """The bulk integrity-check worker must include an unresolvable database
    in its results as a failure, never omit it while reporting an overall
    'OK' (TASK-927)."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)

        def _boom():
            raise RuntimeError("simulated resolver failure")

        window._DB_PATH_RESOLVERS["prompts"] = _boom

        worker = window._integrity_worker()
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        error_calls = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert error_calls, f"unresolvable Prompts database was not reported: {calls}"
        assert any("Prompts" in str(c) for c in error_calls), error_calls
        assert any("UNRESOLVED" in str(c) for c in error_calls), error_calls


@pytest.mark.asyncio
async def test_backup_all_fails_loudly_for_an_unresolvable_database(
    monkeypatch, temp_config_path
):
    """'Backup All Databases' must refuse and report loudly -- never start
    copying a partial set and claim success -- when one of the triad can't
    be resolved (TASK-927).

    Dev-reconciliation note: originally patched "prompts", but dev's
    ``_backup_worker`` resolves Prompts via a direct ``get_prompts_db_path()``
    call (never a hardcoded literal there, so TASK-927 never touched it),
    not through ``_DB_PATH_RESOLVERS`` -- patching "prompts" would no
    longer exercise anything. ChaChaNotes/Media are the two the
    reconciliation actually routed through ``_get_database_path``, so this
    now patches "media". Also: dev's ``_backup_databases()`` orchestrator
    reports a generic "Database backup failed." on any legacy-phase
    exception rather than naming the specific database (a deliberate
    simplification from dev's independent rework), so this no longer
    asserts the database name appears in the notification -- only that an
    error is reported and success/partial-backup never happen.
    """
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)

        def _boom():
            raise RuntimeError("simulated resolver failure")

        window._DB_PATH_RESOLVERS["media"] = _boom

        await window._backup_databases()

        calls = window.app_instance.notify.call_args_list
        error_calls = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert error_calls, f"unresolvable Media database was not reported: {calls}"
        assert not _notify_calls_with_severity(window.app_instance.notify, "success"), (
            f"backup falsely reported success despite an unresolvable database: {calls}"
        )

        backup_root = tldw_chatbook.config.get_user_data_dir() / "backups"
        # No partial backup should have been started at all.
        if backup_root.exists():
            assert not list(backup_root.iterdir()), (
                "a partial backup directory was created despite the "
                "unresolvable database"
            )


@pytest.mark.asyncio
async def test_backup_all_produces_a_file_for_the_database_at_its_resolved_path(
    monkeypatch, temp_config_path
):
    """AC (TASK-927): 'A bulk backup produces files for the databases that
    actually exist.' Also proves the bulk backup worker targets the real
    resolved (profile-aware) path, not a hardcoded literal: the database is
    created only at the resolved path, and the backed-up content is
    verified against it."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        resolved_path = window._get_database_path("chachanotes", {})
        assert resolved_path is not None
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(resolved_path))) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES ('bulk-backup-original')")
            conn.commit()

        await window._backup_databases()

        calls = window.app_instance.notify.call_args_list
        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            f"legacy backup phase reported an error: {calls}"
        )

        backup_root = tldw_chatbook.config.get_user_data_dir() / "backups"
        backup_files = list(backup_root.glob("*/tldw_chatbook_ChaChaNotes_*.db"))
        assert backup_files, (
            f"no ChaChaNotes backup file was produced under {backup_root}"
        )

        with closing(sqlite3.connect(str(backup_files[0]))) as conn:
            value = conn.execute("SELECT value FROM marker").fetchone()[0]
        assert value == "bulk-backup-original"


@pytest.mark.asyncio
async def test_export_conversations_fails_loudly_for_unresolvable_chachanotes(
    monkeypatch, temp_config_path
):
    """The conversation-export worker was found (TASK-927 audit) to build
    its own ChaChaNotes path independently. It must now fail loudly when
    that path can't be resolved, matching the single-database workers,
    instead of raising an unhandled exception or silently doing nothing."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)

        def _boom():
            raise RuntimeError("simulated resolver failure")

        window._DB_PATH_RESOLVERS["chachanotes"] = _boom

        worker = window.run_worker(window._export_conversations_worker, thread=True)
        await worker.wait()

        calls = window.app_instance.notify.call_args_list
        assert calls, "no notification at all for an unresolvable export database"
        error_calls = _notify_calls_with_severity(window.app_instance.notify, "error")
        assert error_calls, f"unresolvable ChaChaNotes database was not reported: {calls}"
        assert not _notify_calls_with_severity(window.app_instance.notify, "success"), (
            f"export falsely reported success despite an unresolvable database: {calls}"
        )
        # Must be the deliberate "cannot resolve" guard, not an unhandled
        # AttributeError from calling .exists() on a None path that
        # happens to be caught by the outer generic exception handler --
        # that would report an error for the wrong reason and would not
        # catch a regression that replaces the guard with, say, a
        # different unguarded None dereference.
        assert any("no resolvable path" in str(c) for c in error_calls), error_calls
        assert not any("NoneType" in str(c) for c in error_calls), error_calls


@pytest.mark.asyncio
async def test_compose_database_config_form_shows_resolved_default_not_hardcoded_literal(
    monkeypatch, temp_config_path
):
    """The Database Config form's path Inputs must display the actual
    resolved (profile-aware) path the app will use, not the old hardcoded,
    profile-unaware literal (TASK-927). Saving the form unmodified right
    after opening it must therefore persist the real path, not a value that
    would make even the fixed resolver disagree with the app on the next
    read."""
    from tldw_chatbook.config import (
        get_chachanotes_db_path,
        get_prompts_db_path,
        get_media_db_path,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        chachanotes_input = window.query_one("#config-db-chachanotes-path", Input)
        prompts_input = window.query_one("#config-db-prompts-path", Input)
        media_input = window.query_one("#config-db-media-path", Input)

        assert chachanotes_input.value == str(get_chachanotes_db_path())
        assert prompts_input.value == str(get_prompts_db_path())
        assert media_input.value == str(get_media_db_path())

        # The old, wrong, pre-fix literals must be gone.
        assert (
            chachanotes_input.value
            != "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
        )
        assert prompts_input.value != "~/.local/share/tldw_cli/tldw_cli_prompts.db"
        assert media_input.value != "~/.local/share/tldw_cli/tldw_cli_media_v2.db"


@pytest.mark.asyncio
async def test_reset_database_config_form_shows_resolved_default_not_hardcoded_literal(
    monkeypatch, temp_config_path
):
    """'Reset Section' on the Database Config form must repopulate the same
    resolved (profile-aware) defaults as the initial composition, not the
    old hardcoded literals (TASK-927)."""
    from tldw_chatbook.config import (
        get_chachanotes_db_path,
        get_prompts_db_path,
        get_media_db_path,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        chachanotes_input = window.query_one("#config-db-chachanotes-path", Input)
        prompts_input = window.query_one("#config-db-prompts-path", Input)
        media_input = window.query_one("#config-db-media-path", Input)

        # Dirty the fields first, as a user editing them would.
        chachanotes_input.value = "/tmp/not-the-real-path.db"
        prompts_input.value = "/tmp/not-the-real-path-2.db"
        media_input.value = "/tmp/not-the-real-path-3.db"

        await window._reset_database_config_form()

        assert chachanotes_input.value == str(get_chachanotes_db_path())
        assert prompts_input.value == str(get_prompts_db_path())
        assert media_input.value == str(get_media_db_path())
        assert (
            chachanotes_input.value
            != "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
        )
        assert prompts_input.value != "~/.local/share/tldw_cli/tldw_cli_prompts.db"
        assert media_input.value != "~/.local/share/tldw_cli/tldw_cli_media_v2.db"


@pytest.mark.asyncio
async def test_resolved_db_path_display_preserves_an_explicit_custom_override(
    monkeypatch, temp_config_path
):
    """Fixing the wrong-default display must not break a genuine
    already-configured custom override: _resolved_db_path_display (used by
    both _compose_database_config_form and _reset_database_config_form)
    must show that override unchanged, not silently replace it with the
    computed profile default (TASK-927).

    The per-test TLDW_CONFIG_PATH env var set by the isolate_test_environment
    autouse fixture always wins over a config-file-based override in this
    test app (see test_restore_creates_missing_target_directory_for_a_custom_db_path
    above for the same constraint), so the override is simulated the same
    way that test does: shadowing _DB_PATH_RESOLVERS at the instance level.
    This still exercises the exact code _resolved_db_path_display calls."""
    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        custom_path = Path("/tmp/a-real-custom-chachanotes-override.db")
        window._DB_PATH_RESOLVERS = dict(window._DB_PATH_RESOLVERS)
        window._DB_PATH_RESOLVERS["chachanotes"] = lambda: custom_path

        assert window._resolved_db_path_display("chachanotes") == str(custom_path)


@pytest.mark.asyncio
async def test_reset_discards_a_configured_custom_override_while_compose_reflects_it(
    monkeypatch, temp_config_path
):
    """Coordinator follow-up (TASK-927): before this fix, Reset called the
    same override-aware _resolved_db_path_display as Compose, so for a user
    who had genuinely customized a database path -- the one case someone
    actually reaches for the "Reset" button -- clicking it did nothing at
    all. Reset must discard the override and restore the pure
    profile-aware default; Compose must keep reflecting the override (what
    the app will actually use). The two must differ whenever an override
    is configured.

    A real override is simulated by monkeypatching config.get_cli_setting
    itself (special-cased to the one key under test, delegating to the
    real implementation otherwise) rather than the config file, because the
    per-test TLDW_CONFIG_PATH env var always wins over a config-file-based
    override in this test app (see
    test_resolved_db_path_display_preserves_an_explicit_custom_override
    above for the same constraint). get_chachanotes_db_path's internal call
    to get_cli_setting resolves via config.py's own module globals at call
    time, so this reaches the real resolver logic -- unlike shadowing
    _DB_PATH_RESOLVERS, which would replace that logic instead of
    exercising it."""
    import tldw_chatbook.config as config_module

    custom_override = "/tmp/a-genuinely-custom-chachanotes-override-927.db"
    real_get_cli_setting = config_module.get_cli_setting

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "database" and key == "chachanotes_db_path":
            return custom_override
        return real_get_cli_setting(section, key, default)

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

        # NOTE: previously computed as
        # ``str(Path(custom_override).expanduser().resolve())``. The
        # private-storage-boundary rework (dev, ADR-029) made custom-path
        # resolution deliberately lexical -- it preserves the user's
        # spelling (e.g. an unresolved ``/tmp`` symlink) instead of
        # following symlinks, so a later no-follow boundary owns link
        # validation. Delegating to the real resolver (as
        # ``expected_default`` below already does) keeps this test honest
        # against that single source of truth instead of re-deriving a
        # now-stale expectation.
        expected_override = str(config_module.get_chachanotes_db_path())
        expected_default = str(
            config_module.get_chachanotes_db_path(ignore_override=True)
        )
        assert expected_override != expected_default, (
            "test setup bug: override and default must differ to prove anything"
        )

        # Compose's mechanism: reflects the currently-effective (override) value.
        compose_value = window._resolved_db_path_display("chachanotes")
        assert compose_value == expected_override

        # Reset's mechanism: discards the override, shows the pure default.
        reset_value = window._resolved_db_path_display(
            "chachanotes", ignore_override=True
        )
        assert reset_value == expected_default
        assert reset_value != compose_value

        # Exercise the real Reset code path end-to-end too, not just the
        # helper: dirty the field the way a user's stale view would look,
        # then confirm _reset_database_config_form() lands on the default,
        # never the override.
        chachanotes_input = window.query_one("#config-db-chachanotes-path", Input)
        chachanotes_input.value = "/tmp/whatever-the-user-was-looking-at.db"
        await window._reset_database_config_form()
        assert chachanotes_input.value == expected_default
        assert chachanotes_input.value != expected_override


def test_compose_and_reset_database_config_form_reuse_the_resolver_helper():
    """Source-scan guard: both functions must go through
    _resolved_db_path_display (and therefore _DB_PATH_RESOLVERS) rather than
    a hardcoded literal or some second, parallel mechanism (TASK-927).
    Reset must specifically pass ignore_override=True so it discards a
    configured custom override instead of reflecting it back unchanged
    (TASK-927 follow-up)."""
    import inspect

    compose_source = inspect.getsource(
        ToolsSettingsWindow._compose_database_config_form
    )
    reset_source = inspect.getsource(ToolsSettingsWindow._reset_database_config_form)

    disagreeing_literals = [
        "tldw_chatbook_ChaChaNotes.db",
        "tldw_cli_prompts.db",
        "tldw_cli_media_v2.db",
    ]
    for source, label in ((compose_source, "compose"), (reset_source, "reset")):
        assert "_resolved_db_path_display" in source, (
            f"_{label}_database_config_form no longer reuses "
            "_resolved_db_path_display"
        )
        for literal in disagreeing_literals:
            assert literal not in source, (
                f"stale hardcoded literal {literal!r} still present in "
                f"{label} form"
            )

    # Strip comment-only lines first: a stale explanatory comment mentioning
    # "ignore_override=True" (e.g. left behind by a careless revert) must
    # not make this check pass when the actual call no longer passes it --
    # confirmed to matter empirically while revert-checking this test.
    def _code_lines(source: str) -> str:
        return "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )

    reset_code = _code_lines(reset_source)
    compose_code = _code_lines(compose_source)

    assert "ignore_override=True" in reset_code, (
        "_reset_database_config_form must pass ignore_override=True so it "
        "discards a configured custom override instead of reflecting it "
        "back unchanged"
    )
    assert "ignore_override=True" not in compose_code, (
        "_compose_database_config_form must keep reflecting the "
        "currently-effective (override-aware) value, not the pure default"
    )


def _read_raw_effective_database_section() -> dict:
    """Read the ``[database]`` table straight off the on-disk effective
    config file, bypassing ``load_cli_config_and_ensure_existence``'s
    deep-merge with ``DEFAULT_CONFIG_FROM_TOML``.

    That merge means ``get_cli_setting("database", "prompts_db_path", None)``
    can NEVER observe "key absent from the file" for a *_db_path key --
    DEFAULT_CONFIG_FROM_TOML always defines a literal for it, so the merged
    view always has a value. Reading the raw file is the only way to prove
    Save genuinely did or did not write/remove a key on disk (TASK-927
    follow-up)."""
    import tomllib

    effective_path = tldw_chatbook.config._get_effective_config_path()
    if not effective_path.exists():
        return {}
    with open(effective_path, "rb") as f:
        return tomllib.load(f).get("database", {})


@pytest.mark.asyncio
async def test_save_database_config_form_without_editing_does_not_pin_an_override(
    monkeypatch, temp_config_path
):
    """_compose_database_config_form shows the fully resolved, profile-aware
    path (TASK-927), so an untouched Input already looks like a custom
    path. If Save always wrote it verbatim, merely opening Settings and
    pressing Save -- without editing anything -- would permanently pin the
    current profile's resolved path as an explicit override, silently
    reintroducing the profile-isolation failure TASK-860 fixed, from the
    opposite direction (TASK-927 follow-up)."""
    from tldw_chatbook.config import (
        get_chachanotes_db_path,
        get_prompts_db_path,
        get_media_db_path,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        await window._save_database_config_form()

        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            window.app_instance.notify.call_args_list
        )

        raw_database_section = _read_raw_effective_database_section()
        for key in ("chachanotes_db_path", "prompts_db_path", "media_db_path"):
            assert key not in raw_database_section, (
                f"no-edit Save must not write a {key} override to the config "
                f"file (raw section: {raw_database_section})"
            )

        # The resolver landing on the profile-aware default is what
        # actually matters -- not merely "no key was written".
        assert get_chachanotes_db_path() == get_chachanotes_db_path(
            ignore_override=True
        )
        assert get_prompts_db_path() == get_prompts_db_path(ignore_override=True)
        assert get_media_db_path() == get_media_db_path(ignore_override=True)


@pytest.mark.asyncio
async def test_reset_then_save_discards_a_previously_configured_override(
    monkeypatch, temp_config_path, tmp_path
):
    """Pressing Reset then Save must actually discard a previously
    configured custom override, not silently re-pin the resolved default
    as a brand-new override (TASK-927 follow-up). Uses the real
    save_setting_to_cli_config/get_cli_setting round trip (not a
    _DB_PATH_RESOLVERS shadow) because both the write under test and the
    read-back here go through the same TLDW_CONFIG_PATH-controlled
    effective config path -- unlike the compose-time config_dict seeding
    in mount_settings_window, this is not shadowed by that env var."""
    from tldw_chatbook.config import (
        save_setting_to_cli_config,
        get_chachanotes_db_path,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        custom_override = str(tmp_path / "genuinely-custom-chachanotes-927.db")
        assert save_setting_to_cli_config(
            "database", "chachanotes_db_path", custom_override
        )
        assert get_chachanotes_db_path() == Path(custom_override).expanduser().resolve()
        assert (
            _read_raw_effective_database_section().get("chachanotes_db_path")
            == custom_override
        ), "test setup bug: override was not actually written to disk"

        await window._reset_database_config_form()
        await window._save_database_config_form()

        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            window.app_instance.notify.call_args_list
        )
        assert (
            "chachanotes_db_path" not in _read_raw_effective_database_section()
        ), (
            "Reset then Save must clear the previously configured override, "
            "not reinstate it"
        )
        assert get_chachanotes_db_path() == get_chachanotes_db_path(
            ignore_override=True
        )


@pytest.mark.asyncio
async def test_save_database_config_form_persists_a_genuine_custom_path(
    monkeypatch, temp_config_path, tmp_path
):
    """A real, deliberately different path entered by the user must still
    be persisted as an explicit override -- the no-edit-Save fix must not
    become over-eager and swallow legitimate customization (TASK-927
    follow-up)."""
    from tldw_chatbook.config import get_prompts_db_path

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        custom_path = tmp_path / "a-genuinely-custom-prompts-927.db"
        prompts_input = window.query_one("#config-db-prompts-path", Input)
        prompts_input.value = str(custom_path)

        await window._save_database_config_form()

        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            window.app_instance.notify.call_args_list
        )
        stored = _read_raw_effective_database_section().get("prompts_db_path")
        assert stored == str(custom_path), (
            "a genuine custom path must be persisted verbatim on disk, got "
            f"{stored!r}"
        )
        assert get_prompts_db_path() == custom_path.expanduser().resolve()


@pytest.mark.asyncio
async def test_save_without_editing_then_switching_profile_still_moves_resolved_path(
    monkeypatch, temp_config_path
):
    """The assertion that actually captures why the bug matters. A fix that
    merely avoids writing an override-shaped key, but still writes
    *something* that happens to pin the current profile's resolved path,
    would pass a narrower 'no key written' check yet still break profile
    switching. Prove the resolved prompts path genuinely moves when the
    profile changes, after a no-edit Save (TASK-927 follow-up)."""
    from tldw_chatbook.config import (
        save_setting_to_cli_config,
        get_prompts_db_path,
        get_user_folder_name,
    )

    async with mount_settings_window({}, temp_config_path, monkeypatch) as (
        window,
        pilot,
    ):
        original_profile = get_user_folder_name()
        original_path = get_prompts_db_path()
        assert original_path.parent.name == original_profile

        await window._save_database_config_form()
        assert not _notify_calls_with_severity(window.app_instance.notify, "error"), (
            window.app_instance.notify.call_args_list
        )

        assert save_setting_to_cli_config(
            "general", "users_name", "second_profile_927"
        )
        assert get_user_folder_name() == "second_profile_927"
        assert get_user_folder_name() != original_profile

        new_path = get_prompts_db_path()
        assert new_path != original_path, (
            "switching profile after a no-edit Save must still move the "
            "resolved prompts db path -- an override was left pinned"
        )
        assert new_path.parent.name == "second_profile_927"
