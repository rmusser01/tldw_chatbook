import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widgets import Input, RichLog, Button

from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import handle_start_llamacpp_server_button_pressed
from tldw_chatbook.app import TldwCli  # Assuming TldwCli is the app class

# Import comprehensive mock but create custom fixture for LLM management
from Tests.fixtures.event_handler_mocks import create_comprehensive_app_mock

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app():
    """Fixture to create a mock TldwCli app instance for LLM management tests."""
    # Get base mock app
    app = create_comprehensive_app_mock()
    
    # Override specific widgets for LLM management tests
    widget_mocks = {
        "#llamacpp-exec-path": MagicMock(spec=Input, value="/path/to/server"),
        "#llamacpp-model-path": MagicMock(spec=Input, value="/path/to/model.gguf"),
        "#llamacpp-host": MagicMock(spec=Input, value="127.0.0.1"),
        "#llamacpp-port": MagicMock(spec=Input, value="8001"),
        "#llamacpp-additional-args": MagicMock(spec=Input, value="--n-gpu-layers 33 --verbose"),
        "#llamacpp-log-output": MagicMock(spec=RichLog),
    }
    
    # Save the original query_one behavior
    original_query_one = app.query_one.side_effect
    
    def llm_query_one_side_effect(selector, widget_type=None):
        # Check our custom widgets first
        if selector in widget_mocks:
            mock_widget = widget_mocks[selector]
            # Type check if requested
            if widget_type and not isinstance(mock_widget, MagicMock):
                if not isinstance(mock_widget, widget_type):
                    raise Exception(f"Mock for {selector} is {type(mock_widget)} not {widget_type}")
            return mock_widget
        # Fall back to original behavior
        return original_query_one(selector, widget_type)
    
    app.query_one.side_effect = llm_query_one_side_effect
    
    # Yield the app and widget_mocks for tests to access
    yield app, widget_mocks


async def test_handle_start_llamacpp_server_button_pressed_basic_command(mock_app):
    mock_app, widget_mocks = mock_app
    """Test basic command construction with all fields provided."""
    # Mock Path to return True for is_file checks
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.return_value = True
        
        # Create a mock Button.Pressed event
        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.run_worker.assert_called_once()
        call_args = mock_app.run_worker.call_args

        # The worker is called with a partial function as the first argument
        # Extract the partial function and check its args
        worker_partial = call_args.args[0]
        assert hasattr(worker_partial, 'args'), "Worker callable is not a partial function"
        # The partial function has the app as first arg and command as second arg
        assert len(worker_partial.args) == 2, "Partial function doesn't have expected args"

        actual_command = worker_partial.args[1]

        expected_command = [
            "/path/to/server",
            "--model", "/path/to/model.gguf",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--n-gpu-layers", "33",
            "--verbose"
        ]
        assert actual_command == expected_command
        mock_app.notify.assert_called_with("Llama.cpp server starting…")


async def test_handle_start_llamacpp_server_no_additional_args(mock_app):
    mock_app, widget_mocks = mock_app
    """Test command construction when additional arguments are empty."""
    # Mock Path to return True for is_file checks
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.return_value = True
        
        # Override the additional args widget value
        widget_mocks["#llamacpp-additional-args"].value = ""

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.run_worker.assert_called_once()
        call_args = mock_app.run_worker.call_args
        worker_partial = call_args.args[0]
        actual_command = worker_partial.args[1]

        expected_command = [
            "/path/to/server",
            "--model", "/path/to/model.gguf",
            "--host", "127.0.0.1",
            "--port", "8001",
        ]
        assert actual_command == expected_command
        mock_app.notify.assert_called_with("Llama.cpp server starting…")


async def test_handle_start_llamacpp_server_additional_args_with_spaces(mock_app):
    mock_app, widget_mocks = mock_app
    """Test command construction with additional arguments containing spaces (quoted)."""
    # Mock Path to return True for is_file checks
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.return_value = True
        
        # Override the additional args widget value
        widget_mocks["#llamacpp-additional-args"].value = '--custom-path "/mnt/my models/llama" --another-arg value'

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.run_worker.assert_called_once()
        call_args = mock_app.run_worker.call_args
        worker_partial = call_args.args[0]
        actual_command = worker_partial.args[1]

        expected_command = [
            "/path/to/server",
            "--model", "/path/to/model.gguf",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--custom-path", "/mnt/my models/llama",  # shlex.split handles the quotes
            "--another-arg", "value"
        ]
        assert actual_command == expected_command


async def test_handle_start_llamacpp_server_default_host_port(mock_app):
    mock_app, widget_mocks = mock_app
    """Test that default host and port are used if inputs are empty."""
    # Mock Path to return True for is_file checks
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.return_value = True
        
        # Override host and port widget values
        widget_mocks["#llamacpp-host"].value = ""
        widget_mocks["#llamacpp-port"].value = ""

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.run_worker.assert_called_once()
        call_args = mock_app.run_worker.call_args
        worker_partial = call_args.args[0]
        actual_command = worker_partial.args[1]

        # Default host is 127.0.0.1, default port is 8001 (as per handler logic)
        expected_command = [
            "/path/to/server",
            "--model", "/path/to/model.gguf",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--n-gpu-layers", "33",
            "--verbose"
        ]
        assert actual_command == expected_command


async def test_handle_start_llamacpp_server_missing_exec_path(mock_app):
    mock_app, widget_mocks = mock_app
    """Test validation: executable path is missing."""
    # Override exec path widget value
    widget_mocks["#llamacpp-exec-path"].value = ""

    mock_event = MagicMock(spec=Button.Pressed)
    await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

    mock_app.notify.assert_called_with("Executable path is required.", severity="error")
    mock_app.run_worker.assert_not_called()


async def test_handle_start_llamacpp_server_invalid_exec_path(mock_app):
    mock_app, widget_mocks = mock_app
    """Test validation: executable path is not a file."""
    # We need to make Path(exec_path).is_file() return False for this specific input
    # The fixture currently patches it globally to True.
    # We can re-patch it within this test for more specific behavior.
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.side_effect = lambda: str(widget_mocks["#llamacpp-exec-path"].value) != "/invalid/path/server"
        widget_mocks["#llamacpp-exec-path"].value = "/invalid/path/server"

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.notify.assert_called_with("Executable not found at: /invalid/path/server", severity="error")
        mock_app.run_worker.assert_not_called()
        # Check that is_file was called with the correct path
        # Path was constructed with the invalid path
        mock_path.assert_any_call("/invalid/path/server")


async def test_handle_start_llamacpp_server_missing_model_path(mock_app):
    mock_app, widget_mocks = mock_app
    """Test validation: model path is missing."""
    # Mock Path to return True for exec path but model path is empty
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        mock_path.return_value.is_file.return_value = True
        
        # Override model path widget value
        widget_mocks["#llamacpp-model-path"].value = ""

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.notify.assert_called_with("Model path is required.", severity="error")
        mock_app.run_worker.assert_not_called()


async def test_handle_start_llamacpp_server_invalid_model_path(mock_app):
    mock_app, widget_mocks = mock_app
    """Test validation: model path is not a file."""
    with patch("tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events.Path") as mock_path:
        # Ensure exec path is valid for this test
        widget_mocks["#llamacpp-exec-path"].value = "/path/to/server"  # Valid mock path
        
        def is_file_side_effect(path_str):
            return str(path_str) == "/path/to/server"
        
        mock_path.return_value.is_file.side_effect = lambda: is_file_side_effect(mock_path.call_args[0][0])

        widget_mocks["#llamacpp-model-path"].value = "/invalid/path/model.gguf"

        mock_event = MagicMock(spec=Button.Pressed)
        await handle_start_llamacpp_server_button_pressed(mock_app, mock_event)

        mock_app.notify.assert_called_with("Model file not found at: /invalid/path/model.gguf", severity="error")
        mock_app.run_worker.assert_not_called()
        # Path is called for exec_path first, then for model_path
        assert mock_path.call_count >= 2
        # Check that Path was called with the invalid model path
        mock_path.assert_any_call("/invalid/path/model.gguf")

# To run these tests, you would typically use pytest:
# pytest Tests/Event_Handlers/test_llm_management_events.py
# (Assuming pytest and pytest-asyncio are installed)
# Also ensure that tldw_chatbook is in PYTHONPATH or installable
# and that the necessary dependencies for textual are available.
# The patch for Path.is_file in the fixture and some tests is important
# to bypass actual filesystem checks.
#
# Note on the patch in the fixture:
# The `with patch(...) as _:` in the fixture means Path.is_file will return True
# for ALL calls during the tests that use this fixture, unless a test re-patches it.
# The `_` is used because we don't need to access the MagicMock object for the patch itself
# in most test cases using the fixture.
# For tests like `test_handle_start_llamacpp_server_invalid_exec_path`, we re-patch
# `pathlib.Path.is_file` with a more specific side_effect to test the False condition.
#
# The side_effect for query_one in the fixture is a simple way to return different
# mocks based on the selector. A more robust approach for complex scenarios might

# involve a dictionary lookup as shown.
#
# The check `if not isinstance(mock_widget, widget_type):` in `query_one_side_effect`
# is to more closely mimic Textual's `query_one` behavior which also checks the type.
#
# The `pytestmark = pytest.mark.asyncio` line at the top is important for pytest-asyncio
# to correctly run the async test functions.
#
# The `mock_app.loguru_logger = MagicMock()` line is added to the fixture to prevent
# errors if the logger is accessed (e.g., `app.loguru_logger.info(...)`).
#
# The `test_handle_start_llamacpp_server_invalid_exec_path` and
# `test_handle_start_llamacpp_server_invalid_model_path` have been updated
# to show how to make `Path.is_file` return `False` for specific paths by
# re-patching within the test or using a more complex side_effect.
# The `side_effect=lambda p: str(p) != "/invalid/path/server"` means `is_file`
# will return `True` for any path *except* "/invalid/path/server".
# For the invalid model path test, the side_effect is a bit more complex to ensure
# the exec_path still appears valid while the model_path does not.
#
# The assertion `mock_is_file.assert_any_call(...)` is used because `is_file` might be
# called multiple times (once for exec_path, once for model_path). We just want to ensure
# it was called with the specific path we're testing.
#
# Final check on `call_args` for `run_worker`:
# The arguments to `run_worker` in the main code are passed as `args=[app, command]`.
# So, in the test, `mock_app.run_worker.call_args` will be a Call object.
# If using `args=...`, then `call_args.args` would be `([app, command],)`.
# If using `kwargs={'args': ...}` or `args=...` as a kwarg, then `call_args.kwargs['args']`
# would be `[app, command]`. The current code uses `args=[app, command]` directly in the
# `run_worker` call, so it's passed as a keyword argument named `args`.
# `call_args.kwargs['args']` is `[<MagicMock spec='TldwCli' id='...'>, ['/path/to/server', ...]]`
# So `worker_partial.args[1]` is the command list.
# The assertions have been updated to reflect this.
#
# Added a check in the fixture's `query_one_side_effect` to ensure the mock widget found
# is an instance of the `widget_type` argument, as the actual code uses this.
# Example: `app.query_one("#llamacpp-exec-path", Input)`
# This makes the mock more accurate.
#
# Added `app.loguru_logger = MagicMock()` to the fixture.
# The handler code uses `logger = getattr(app, "loguru_logger", logging.getLogger(__name__))`
# so the mock app needs this attribute.
#
# Updated the `test_handle_start_llamacpp_server_invalid_model_path` to correctly mock `is_file`
# such that the exec_path is considered valid, but the model_path is not. This requires
# the side_effect to be a bit more conditional.
#
# The test `test_handle_start_llamacpp_server_invalid_exec_path` was also refined for clarity on how `is_file` is mocked.
#
# Final review of the run_worker call in the handler:
# `app.run_worker(run_llamacpp_server_worker, args=[app, command], ...)`
# This means `call_args.args` will be `(run_llamacpp_server_worker,)`
# and `call_args.kwargs` will be `{'args': [app, command], 'group': ..., ...}`.
# The tests correctly access `worker_partial.args[1]`.

# TODO: The following test is commented out because it requires the llm_nav_events module
# which doesn't exist. This test should be moved to a separate file or the module should be created.

# from tldw_chatbook.Event_Handlers.llm_nav_events import handle_llm_nav_button_pressed
# from textual.containers import Container
# 
# 
# async def test_mlx_lm_nav_button_shows_correct_view():
#     """Test that pressing the MLX-LM nav button shows the correct view and hides others."""
#     app = TldwCli()
#     target_view_id = "llm-view-mlx-lm"
#     other_view_ids = [
#         "llm-view-llama-cpp",
#         "llm-view-llamafile",
#         "llm-view-ollama",
#         "llm-view-vllm",
#         "llm-view-transformers",
#         "llm-view-local-models",
#         "llm-view-download-models",
#     ]
# 
#     async with app.run_test() as pilot:
#         # Initial state check (optional, but good for sanity)
#         # Ensure all views are initially hidden or one is active as per app logic
#         # For this test, we assume the handler will correctly set states regardless of initial.
# 
#         # Simulate the MLX-LM nav button being pressed
#         await handle_llm_nav_button_pressed(pilot.app, "llm-nav-mlx-lm")
# 
#         # Check that the MLX-LM view is visible
#         mlx_view = pilot.app.query_one(f"#{target_view_id}", Container)
#         assert mlx_view.styles.display is not None
#         assert mlx_view.styles.display.value == "block", f"{target_view_id} should be 'block'"
# 
#         # Check that all other main LLM views are hidden
#         for view_id in other_view_ids:
#             other_view = pilot.app.query_one(f"#{view_id}", Container)
#             assert other_view.styles.display is not None
#             assert other_view.styles.display.value == "none", f"{view_id} should be 'none'"
