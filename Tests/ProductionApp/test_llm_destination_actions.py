from __future__ import annotations

import asyncio
import ast
import io
import inspect
import logging
from pathlib import Path
import subprocess
import textwrap
import threading
import time

import pytest
from textual.widgets import Button, Input, RichLog, TextArea

import tldw_chatbook.app as app_module
from tldw_chatbook.Local_Inference import ollama_model_mgmt
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_LLM
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_ollama as ollama_events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_transformers as transformers_events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_vllm as vllm_events,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    clear_server_process,
    current_server_claim,
    current_llm_destination,
    publish_server_process,
    release_server_claim,
    reserve_server_launch,
    retain_cancelled_server_process,
    run_server_subprocess,
    stop_server_process,
    terminate_process_bounded,
)
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen, MODELS_RAIL_SECTIONS
from tldw_chatbook.Event_Handlers.worker_handlers.misc_worker_handler import (
    MiscWorkerHandler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_WINDOW_PATH = PROJECT_ROOT / "tldw_chatbook" / "UI" / "LLM_Management_Window.py"
PROHIBITED_LLM_TEST_PATHS = (
    PROJECT_ROOT / "Tests" / "LLM_Management" / "test_llm_management_events.py",
    PROJECT_ROOT / "Tests" / "UI" / "test_llm_runtime_browse_tooltips.py",
)
LLM_EVENT_PATHS = tuple(
    sorted(
        (
            PROJECT_ROOT / "tldw_chatbook" / "Event_Handlers" / "LLM_Management_Events"
        ).glob("*.py")
    )
)
REMOVED_TRANSFORMERS_ACTION_IDS = {
    "transformers-browse-script-button",
    "transformers-start-server-button",
    "transformers-stop-server-button",
}
ROOT_LOG_CALLBACKS = {
    "_update_llamacpp_log",
    "_update_llamafile_log",
    "_update_mlx_log",
    "_update_model_download_log",
    "_update_transformers_log",
    "_update_vllm_log",
}
EXPECTED_BROWSE_TOOLTIPS = {
    "llamacpp-browse-exec-button": "Choose the llama.cpp server executable.",
    "llamacpp-browse-model-button": "Choose a GGUF model file for llama.cpp.",
    "llamafile-browse-exec-button": "Choose the llamafile executable.",
    "llamafile-browse-model-button": (
        "Choose an optional external GGUF model for llamafile."
    ),
    "vllm-browse-python-button": ("Choose the Python interpreter used to launch vLLM."),
    "vllm-browse-model-button": (
        "Choose a local model directory for vLLM, or type a Hugging Face repo ID."
    ),
    "onnx-browse-python-button": (
        "Choose the Python interpreter used to launch the ONNX server."
    ),
    "onnx-browse-script-button": "Choose the ONNX server script to run.",
    "onnx-browse-model-button": ("Choose the ONNX model file or directory to load."),
    "transformers-browse-models-dir-button": (
        "Choose the local Transformers models root directory."
    ),
    "mlx-browse-model-button": (
        "Choose a local MLX model path, or type a Hugging Face repo ID."
    ),
    "ollama-browse-exec-button": "Choose the Ollama executable.",
    "ollama-browse-modelfile-button": (
        "Choose the Modelfile used to create an Ollama model."
    ),
}


class _DeterministicOllamaProcess:
    """Record bounded shutdown while forcing the kill fallback."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else -9

    def terminate(self) -> None:
        self.calls.append("terminate")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if self.calls.count(("wait", timeout)) == 1:
            raise subprocess.TimeoutExpired("ollama serve", timeout)
        self._running = False
        return -9

    def kill(self) -> None:
        self.calls.append("kill")


class _PersistentOllamaProcess:
    """Remain live across both bounded shutdown waits."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.calls.append("terminate")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        raise subprocess.TimeoutExpired("ollama serve", timeout)

    def kill(self) -> None:
        self.calls.append("kill")


class _TerminateRaisesProcess:
    """Require the kill path even when terminate itself raises."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else -9

    def terminate(self) -> None:
        self.calls.append("terminate")
        raise OSError("private terminate failure")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        self._running = False
        return -9


class _SecondTerminationStopsProcess:
    """Stop only when final cleanup retries the bounded termination sequence."""

    def __init__(self) -> None:
        self.running = True
        self.terminate_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return None if self.running else -9

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_calls >= 2:
            self.running = False

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.running:
            raise subprocess.TimeoutExpired("provider", timeout)
        return -9


class _LifecycleProcess:
    """Controllable process identity for mounted lifecycle generations."""

    def __init__(self, pid: int, running: bool = True) -> None:
        self.pid = pid
        self.running = running

    def poll(self) -> int | None:
        return None if self.running else 0


class _StopProcess:
    """Exercise graceful, kill-fallback, and stubborn bounded Stop outcomes."""

    def __init__(self, pid: int, mode: str) -> None:
        self.pid = pid
        self.mode = mode
        self.running = True
        self.calls: list[object] = []

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.calls.append("terminate")
        if self.mode == "graceful":
            self.running = False

    def kill(self) -> None:
        self.calls.append("kill")
        if self.mode == "kill":
            self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        time.sleep(0.02)
        if self.running:
            raise subprocess.TimeoutExpired("provider", timeout)
        return 0


class _BlockingStopProcess:
    """Pause one bounded Stop so the production destination can be replaced."""

    pid = 4545

    def __init__(self) -> None:
        self.running = True
        self.wait_started = threading.Event()
        self.release_wait = threading.Event()

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.running = False

    def kill(self) -> None:
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_started.set()
        if not self.release_wait.wait(timeout=5):
            raise subprocess.TimeoutExpired("provider", timeout)
        return 0


class _CompletedOllamaProcess:
    """Complete immediately while exposing deterministic worker output."""

    pid = 4242
    returncode = 0

    def __init__(self) -> None:
        self.stdout = io.StringIO("ollama-ready\n")

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


class _TransformersProcess:
    """Deterministic subprocess result for direct worker execution."""

    pid = 5252

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _TimeoutTransformersProcess:
    """Force timeout cleanup and expose secret output that must be discarded."""

    pid = 6262
    returncode = None

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if not self.killed:
            raise subprocess.TimeoutExpired("PRIVATE_COMMAND", timeout)
        self.returncode = -9
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.calls.append("kill")
        self.killed = True

    def terminate(self) -> None:
        self.calls.append("terminate")


class _SubprocessScenario:
    """Narrow deterministic subprocess adapter for worker functions."""

    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL
    STDOUT = subprocess.STDOUT
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, processes: list[object]) -> None:
        self.processes = processes
        self.commands: list[list[str]] = []
        self.popen_kwargs: list[dict[str, object]] = []

    def Popen(self, command, **kwargs):
        self.commands.append(list(command))
        self.popen_kwargs.append(dict(kwargs))
        return self.processes.pop(0)


class _FailingSubprocessScenario:
    """Fail launch with a private payload that must never reach the UI."""

    def __init__(self, private_message: str) -> None:
        self.private_message = private_message
        self.commands: list[list[str]] = []

    def Popen(self, command, **kwargs):
        self.commands.append(list(command))
        raise OSError(self.private_message)


def _direct_compose_button_ids() -> set[str]:
    module = ast.parse(
        LLM_WINDOW_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_WINDOW_PATH),
    )
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMManagementWindow"
    )
    compose = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "compose"
    )
    ids: set[str] = set()
    for node in ast.walk(compose):
        if not isinstance(node, ast.Call):
            continue
        callable_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "")
        )
        if callable_name != "Button":
            continue
        id_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "id"),
            None,
        )
        if (
            id_keyword is not None
            and isinstance(id_keyword.value, ast.Constant)
            and isinstance(id_keyword.value.value, str)
        ):
            ids.add(id_keyword.value.value)
    return ids


def _direct_compose_button_keywords(button_id: str) -> dict[str, ast.expr]:
    module = ast.parse(
        LLM_WINDOW_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_WINDOW_PATH),
    )
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        callable_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "")
        )
        if callable_name != "Button":
            continue
        keywords = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
        }
        id_value = keywords.get("id")
        if isinstance(id_value, ast.Constant) and id_value.value == button_id:
            return keywords
    raise AssertionError(f"Button {button_id!r} is not composed directly")


def test_server_lifecycle_is_app_owned_and_root_worker_handler_is_retired() -> None:
    app_source = inspect.getsource(TldwCli)
    window_source = LLM_WINDOW_PATH.read_text(encoding="utf-8")
    worker_init_source = inspect.getsource(TldwCli._init_worker_handlers)

    assert "_llm_server_launch_claims" in app_source
    assert "_ollama_launch_reserved" not in window_source
    assert "_ollama_launch_cancel_event" not in window_source
    assert "ServerWorkerHandler" not in worker_init_source
    assert "ServerWorkerHandler" not in (
        PROJECT_ROOT / "tldw_chatbook" / "app.py"
    ).read_text(encoding="utf-8")
    assert not (
        PROJECT_ROOT
        / "tldw_chatbook"
        / "Event_Handlers"
        / "worker_handlers"
        / "server_worker_handler.py"
    ).exists()

    for button_id in (
        "llamacpp-stop-server-button",
        "llamafile-stop-server-button",
        "vllm-stop-server-button",
        "onnx-stop-server-button",
        "mlx-stop-server-button",
        "ollama-stop-service-button",
    ):
        disabled = _direct_compose_button_keywords(button_id).get("disabled")
        assert isinstance(disabled, ast.Constant)
        assert disabled.value is True


def test_bounded_termination_kills_and_reaps_when_terminate_raises() -> None:
    process = _TerminateRaisesProcess()

    assert terminate_process_bounded(process, timeout=0.25) is True
    assert process.calls == ["terminate", "kill", ("wait", 0.25)]


def test_live_llm_handlers_do_not_persist_sensitive_diagnostics() -> None:
    forbidden_source = (
        "exception=True",
        "Executing:",
        "Command:",
        "quoted_command",
        "Raw data:",
        "WORKER STDOUT",
        "WORKER STDERR",
    )
    violations: list[tuple[str, int, str]] = []
    for path in LLM_EVENT_PATHS:
        source = path.read_text(encoding="utf-8")
        for marker in forbidden_source:
            if marker in source:
                violations.append((path.name, 0, marker))
        tree = ast.parse(source, filename=str(path))
        for handler in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.name
        ):
            for joined_string in (
                node for node in ast.walk(handler) if isinstance(node, ast.JoinedStr)
            ):
                if any(
                    isinstance(node, ast.Name) and node.id == handler.name
                    for node in ast.walk(joined_string)
                ):
                    violations.append(
                        (
                            path.name,
                            joined_string.lineno,
                            f"exception payload {handler.name}",
                        )
                    )
        for worker_name in (
            "run_llamafile_server_worker",
            "run_llamacpp_server_worker",
            "run_vllm_server_worker",
            "run_onnx_server_worker",
            "run_mlx_lm_server_worker",
            "run_ollama_service_worker",
        ):
            worker = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == worker_name
                ),
                None,
            )
            if worker is not None:
                arguments = {argument.arg for argument in worker.args.args}
                assert "claim" in arguments
                assert "window" not in arguments

    assert violations == []


def test_ollama_api_failures_and_metrics_exclude_private_payloads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    private_url = "file://PRIVATE_CREDENTIAL"
    _, invalid_error = ollama_model_mgmt._ollama_request(
        "GET",
        private_url,
        "/api/tags",
    )
    assert invalid_error == "Invalid Ollama server URL."
    assert "PRIVATE_CREDENTIAL" not in caplog.text

    class RaisingSession:
        headers: dict[str, str]

        def __init__(self) -> None:
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, *args, **kwargs):
            raise ollama_model_mgmt.requests.exceptions.RequestException(
                "PRIVATE_REQUEST_EXCEPTION"
            )

    monkeypatch.setattr(ollama_model_mgmt.requests, "Session", RaisingSession)
    _, request_error = ollama_model_mgmt._ollama_request(
        "GET",
        "https://user:PRIVATE_PASSWORD@example.invalid",
        "/api/tags",
    )
    assert request_error == "Ollama request failed (category=request_exception)."

    metric_calls: list[tuple[str, object]] = []
    histogram_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        ollama_model_mgmt,
        "log_counter",
        lambda name, *args, **kwargs: metric_calls.append((name, kwargs.get("labels"))),
    )
    monkeypatch.setattr(
        ollama_model_mgmt,
        "log_histogram",
        lambda name, value, *args, **kwargs: histogram_calls.append((name, value)),
    )
    monkeypatch.setattr(
        ollama_model_mgmt,
        "_ollama_request",
        lambda *args, **kwargs: ({"status": "success"}, None),
    )
    ollama_model_mgmt.ollama_delete_model(
        "https://example.invalid",
        "PRIVATE_MODEL_ID",
    )
    ollama_model_mgmt.ollama_pull_model(
        "https://example.invalid",
        "PRIVATE_MODEL_ID",
    )
    ollama_model_mgmt.ollama_model_info(
        "https://example.invalid",
        "PRIVATE_INFO_MODEL",
    )
    ollama_model_mgmt.ollama_copy_model(
        "https://example.invalid",
        "PRIVATE_SOURCE_MODEL",
        "PRIVATE_DESTINATION_MODEL",
    )
    ollama_model_mgmt.ollama_create_model(
        "https://example.invalid",
        "PRIVATE_CREATED_MODEL",
        "PRIVATE_MODELFILE_PATH",
    )
    ollama_model_mgmt.ollama_push_model(
        "https://example.invalid",
        "PRIVATE_PUSH_MODEL",
    )
    ollama_model_mgmt.ollama_generate_embeddings(
        "https://example.invalid",
        "PRIVATE_EMBEDDING_MODEL",
        "PRIVATE_EMBEDDING_PROMPT",
    )
    monkeypatch.setattr(
        ollama_model_mgmt,
        "_ollama_request",
        lambda *args, **kwargs: (
            {"models": [{"size": "PRIVATE_RAW_API_METRIC_VALUE"}]},
            None,
        ),
    )
    ollama_model_mgmt.ollama_list_running_models("https://example.invalid")

    diagnostics = caplog.text + repr(metric_calls) + repr(histogram_calls)
    for private_value in (
        "PRIVATE_CREDENTIAL",
        "PRIVATE_PASSWORD",
        "PRIVATE_REQUEST_EXCEPTION",
        "PRIVATE_MODEL_ID",
        "PRIVATE_INFO_MODEL",
        "PRIVATE_SOURCE_MODEL",
        "PRIVATE_DESTINATION_MODEL",
        "PRIVATE_CREATED_MODEL",
        "PRIVATE_MODELFILE_PATH",
        "PRIVATE_PUSH_MODEL",
        "PRIVATE_EMBEDDING_MODEL",
        "PRIVATE_EMBEDDING_PROMPT",
        "PRIVATE_RAW_API_METRIC_VALUE",
    ):
        assert private_value not in diagnostics


def test_ollama_success_payloads_are_bounded_and_redacted() -> None:
    rendered = ollama_events._format_ollama_success_payload(
        {
            "family": "llama",
            "api_key": "PRIVATE_API_KEY",
            "details": {"format": "gguf"},
        }
    )

    assert '"family": "llama"' in rendered
    assert '"format": "gguf"' in rendered
    assert "PRIVATE_API_KEY" not in rendered
    assert "REDACTED" in rendered
    assert len(rendered) <= ollama_events.MAX_OLLAMA_SUCCESS_OUTPUT_CHARS

    names = ollama_events._safe_ollama_model_names(
        [
            {"name": "org/model:latest"},
            {"name": "token=PRIVATE_MODEL_TOKEN"},
        ]
    )
    assert names is not None
    assert names[0] == "org/model:latest"
    assert "PRIVATE_MODEL_TOKEN" not in names[1]


def test_transformers_download_has_no_root_worker_completion_owner() -> None:
    source = inspect.getsource(MiscWorkerHandler)

    assert "transformers_download" not in MiscWorkerHandler.HANDLED_GROUPS
    assert "_handle_transformers_download" not in source


def test_llm_destination_action_census_is_complete_and_removed_controls_are_absent() -> (
    None
):
    button_ids = _direct_compose_button_ids()
    navigation_ids = {
        button_id for button_id in button_ids if button_id.startswith("nav-")
    }
    action_ids = button_ids - navigation_ids

    assert navigation_ids == set()
    assert {
        view_key
        for _section, entries in MODELS_RAIL_SECTIONS
        for view_key, _label in entries
    } == {
        "llama-cpp",
        "llamafile",
        "ollama",
        "vllm",
        "onnx",
        "transformers",
        "mlx-lm",
        "local-models",
        "download-models",
    }
    assert REMOVED_TRANSFORMERS_ACTION_IDS.isdisjoint(action_ids)
    assert action_ids == set(LLMManagementWindow.ACTION_HANDLERS)


def test_llm_destination_action_contract_uses_window_for_ui_lookups() -> None:
    violations: list[tuple[str, str]] = []
    for callback in LLMManagementWindow.ACTION_HANDLERS.values():
        parameters = tuple(inspect.signature(callback).parameters)
        if parameters[:3] != ("window", "app", "event"):
            violations.append((callback.__name__, f"signature={parameters!r}"))
            continue

        callback_tree = ast.parse(textwrap.dedent(inspect.getsource(callback)))
        for node in ast.walk(callback_tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"query", "query_one"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "app"
            ):
                violations.append(
                    (callback.__name__, f"app.{node.func.attr}@{node.lineno}")
                )

    assert violations == []


def test_llm_transitive_worker_paths_never_query_ui_through_app_root() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in LLM_EVENT_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr in {"query", "query_one"}
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"app", "app_instance"}
                ):
                    violations.append((path.name, node.lineno, ast.unparse(node.func)))
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "LogWidgetManager"
                    and node.func.attr.startswith("update_")
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in {"app", "app_instance"}
                ):
                    violations.append((path.name, node.lineno, ast.unparse(node)))
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ROOT_LOG_CALLBACKS
                and isinstance(node.value, ast.Name)
                and node.value.id in {"app", "app_instance"}
            ):
                violations.append((path.name, node.lineno, ast.unparse(node)))

    assert violations == []


def test_destination_registry_is_a_direct_merge_and_workers_are_thread_callables() -> (
    None
):
    module = ast.parse(
        LLM_WINDOW_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_WINDOW_PATH),
    )
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMManagementWindow"
    )
    action_assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ACTION_HANDLERS"
    )

    assert not any(
        isinstance(node, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp))
        for node in ast.walk(action_assignment.value)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "action_handlers"
        for node in ast.walk(class_node)
    )
    assert "browse-models-dir-button" not in LLMManagementWindow.ACTION_HANDLERS
    assert "start-model-download-button" not in LLMManagementWindow.ACTION_HANDLERS
    assert all(not path.exists() for path in PROHIBITED_LLM_TEST_PATHS)
    assert not inspect.iscoroutinefunction(vllm_events.run_vllm_server_worker)
    assert not inspect.iscoroutinefunction(
        transformers_events.run_transformers_model_download_worker
    )
    assert tuple(
        inspect.signature(
            transformers_events.run_transformers_model_download_worker
        ).parameters
    )[:1] == ("app_instance",)


async def _wait_for_llm_screen(app: TldwCli, pilot) -> LLMScreen:
    for _ in range(200):
        if getattr(app, "_initial_screen_pushed", False) and isinstance(
            app.screen, LLMScreen
        ):
            screen = app.screen
            if (
                screen.llm_window is not None
                and screen.llm_window.is_mounted
                and screen.llm_window.active_view
            ):
                return screen
        await pilot.pause(0.01)
    raise AssertionError(
        "production TldwCli did not mount its registered LLMScreen and Models body"
    )


async def _wait_for_llm_window(
    screen: LLMScreen,
    pilot,
    *,
    previous: LLMManagementWindow | None = None,
) -> LLMManagementWindow:
    for _ in range(200):
        window = screen.llm_window
        if (
            window is not None
            and window is not previous
            and window.is_mounted
            and window.active_view
        ):
            return window
        await pilot.pause(0.01)
    raise AssertionError("production Models body did not mount")


def _rich_log_text(log: RichLog) -> str:
    return "\n".join(line.text for line in log.lines)


@pytest.mark.asyncio
async def test_production_llm_lifecycle_generations_survive_window_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app._initial_tab_value = TAB_LLM

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _wait_for_llm_screen(app, pilot)
            old_window = screen.query_one(LLMManagementWindow)
            for _provider, (_start_id, stop_id) in old_window.SERVER_CONTROLS.items():
                assert old_window.query_one(f"#{stop_id}", Button).disabled is True

            first_claim = reserve_server_launch(app, "ollama")
            assert first_claim is not None
            old_window._sync_process_controls("ollama")
            assert (
                old_window.query_one("#ollama-start-service-button", Button).disabled
                is True
            )

            screen.refresh(recompose=True)
            new_window = await _wait_for_llm_window(
                screen,
                pilot,
                previous=old_window,
            )
            assert new_window is not old_window
            assert current_server_claim(app, "ollama") is first_claim
            assert (
                new_window.query_one("#ollama-start-service-button", Button).disabled
                is True
            )
            assert (
                new_window.query_one("#ollama-stop-service-button", Button).disabled
                is False
            )

            assert release_server_claim(app, "ollama", first_claim) is True
            second_claim = reserve_server_launch(app, "ollama")
            assert second_claim is not None
            second_process = _LifecycleProcess(2002)
            assert (
                publish_server_process(app, "ollama", second_claim, second_process)
                is True
            )

            stale_process = _LifecycleProcess(1001)
            assert (
                publish_server_process(app, "ollama", first_claim, stale_process)
                is False
            )
            stale_process.running = False
            assert (
                clear_server_process(app, "ollama", first_claim, stale_process) is False
            )
            assert current_server_claim(app, "ollama") is second_claim
            assert app.ollama_server_process is second_process

            second_process.running = False
            assert (
                clear_server_process(app, "ollama", second_claim, second_process)
                is True
            )
            cancelled_claim = reserve_server_launch(app, "ollama")
            assert cancelled_claim is not None
            cancelled_claim.cancel_event.set()
            stubborn_process = _LifecycleProcess(3003)
            assert (
                publish_server_process(app, "ollama", cancelled_claim, stubborn_process)
                is False
            )
            assert (
                retain_cancelled_server_process(
                    app, "ollama", cancelled_claim, stubborn_process
                )
                is True
            )
            assert app.ollama_server_process is stubborn_process
            stubborn_process.running = False
            assert (
                clear_server_process(app, "ollama", cancelled_claim, stubborn_process)
                is True
            )

            unclaimed_process = _DeterministicOllamaProcess()
            app.ollama_server_process = unclaimed_process
            assert await stop_server_process(app, "ollama", "Ollama service") is True
            assert app.ollama_server_process is None

            orphan_claim = reserve_server_launch(app, "vllm")
            assert orphan_claim is not None
            orphan_process = _DeterministicOllamaProcess()
            orphan_scenario = _SubprocessScenario([orphan_process])
            real_call_from_thread = app.call_from_thread
            publish_attempts = 0

            def fail_first_publish(callback, *args, **kwargs):
                nonlocal publish_attempts
                if callback is publish_server_process and publish_attempts == 0:
                    publish_attempts += 1
                    raise RuntimeError("private publish failure")
                return callback(*args, **kwargs)

            monkeypatch.setattr(app, "call_from_thread", fail_first_publish)
            orphan_result = await asyncio.to_thread(
                run_server_subprocess,
                app,
                "vllm",
                ["python", "--token", "PRIVATE_SENTINEL"],
                orphan_claim,
                orphan_scenario,
            )
            monkeypatch.setattr(app, "call_from_thread", real_call_from_thread)
            assert orphan_result == "vllm server failed (category=RuntimeError)"
            assert "PRIVATE_SENTINEL" not in orphan_result
            assert orphan_process.poll() is not None
            assert current_server_claim(app, "vllm") is None

            settlement_claim = reserve_server_launch(app, "vllm")
            assert settlement_claim is not None
            settlement_process = _DeterministicOllamaProcess()
            settlement_scenario = _SubprocessScenario([settlement_process])

            def reject_all_marshalling(callback, *args, **kwargs):
                raise RuntimeError("PRIVATE_MARSHALLING_FAILURE")

            monkeypatch.setattr(app, "call_from_thread", reject_all_marshalling)
            settlement_result = await asyncio.to_thread(
                run_server_subprocess,
                app,
                "vllm",
                ["python", "--token", "PRIVATE_SETTLEMENT_TOKEN"],
                settlement_claim,
                settlement_scenario,
            )
            monkeypatch.setattr(app, "call_from_thread", real_call_from_thread)
            assert settlement_result == "vllm server failed (category=RuntimeError)"
            assert settlement_process.poll() is not None
            assert current_server_claim(app, "vllm") is None
            assert app.vllm_server_process is None

            retry_claim = reserve_server_launch(app, "vllm")
            assert retry_claim is not None
            retry_process = _SecondTerminationStopsProcess()
            retry_scenario = _SubprocessScenario([retry_process])

            def cancel_before_publish(callback, *args, **kwargs):
                if callback is publish_server_process:
                    retry_claim.cancel_event.set()
                return callback(*args, **kwargs)

            monkeypatch.setattr(app, "call_from_thread", cancel_before_publish)
            retry_result = await asyncio.to_thread(
                run_server_subprocess,
                app,
                "vllm",
                ["python", "--token", "PRIVATE_RETRY_TOKEN"],
                retry_claim,
                retry_scenario,
            )
            monkeypatch.setattr(app, "call_from_thread", real_call_from_thread)
            assert retry_result == "vllm launch cancelled"
            assert retry_process.poll() is not None
            assert retry_process.terminate_calls == 2
            assert retry_process.wait_calls == 3
            assert current_server_claim(app, "vllm") is None
            assert app.vllm_server_process is None

            for mode in ("graceful", "kill", "persistent"):
                for index, provider in enumerate(new_window.SERVER_CONTROLS):
                    claim = reserve_server_launch(app, provider)
                    assert claim is not None
                    process = _StopProcess(4000 + index, mode)
                    assert publish_server_process(app, provider, claim, process) is True
                    new_window._sync_process_controls(provider)
                    stop_id = new_window.SERVER_CONTROLS[provider][1]
                    stop_event = Button.Pressed(
                        new_window.query_one(f"#{stop_id}", Button)
                    )
                    pulse = asyncio.create_task(asyncio.sleep(0.005))
                    await new_window.on_button_pressed(stop_event)
                    assert pulse.done(), f"{provider} Stop blocked the event loop"
                    if mode == "persistent":
                        assert current_server_claim(app, provider) is claim
                        assert (
                            getattr(
                                app,
                                {
                                    "llamacpp": "llamacpp_server_process",
                                    "llamafile": "llamafile_server_process",
                                    "vllm": "vllm_server_process",
                                    "onnx": "onnx_server_process",
                                    "mlx": "mlx_server_process",
                                    "ollama": "ollama_server_process",
                                }[provider],
                            )
                            is process
                        )
                        start_id = new_window.SERVER_CONTROLS[provider][0]
                        assert (
                            new_window.query_one(f"#{start_id}", Button).disabled
                            is True
                        )
                        assert (
                            new_window.query_one(f"#{stop_id}", Button).disabled
                            is False
                        )
                        process.running = False
                        assert (
                            clear_server_process(app, provider, claim, process) is True
                        )
                    else:
                        assert current_server_claim(app, provider) is None
                    expected = ["terminate", ("wait", 5.0)]
                    if mode != "graceful":
                        expected.extend(["kill", ("wait", 5.0)])
                    assert process.calls == expected
    finally:
        try:
            if app._rich_log_handler:
                await app._rich_log_handler.stop_processor()
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            await app.on_shutdown_request()
            await app.on_unmount()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_production_llm_async_results_require_current_owner_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app._initial_tab_value = TAB_LLM

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _wait_for_llm_screen(app, pilot)
            window = screen.query_one(LLMManagementWindow)

            from tldw_chatbook.Widgets.delete_confirmation_dialog import (
                create_delete_confirmation,
            )

            dialog = create_delete_confirmation(
                item_type="Model",
                item_name="mounted-owner-check",
                permanent=True,
            )
            await app.push_screen(dialog)
            await pilot.pause()
            assert current_llm_destination(app) is window
            await app.pop_screen()
            await pilot.pause()

            stop_claim = reserve_server_launch(app, "ollama")
            assert stop_claim is not None
            stop_process = _BlockingStopProcess()
            assert (
                publish_server_process(app, "ollama", stop_claim, stop_process) is True
            )
            stop_event = Button.Pressed(
                window.query_one("#ollama-stop-service-button", Button)
            )
            stop_task = asyncio.create_task(window.on_button_pressed(stop_event))
            assert await asyncio.to_thread(stop_process.wait_started.wait, 5)

            screen.refresh(recompose=True)
            replacement = await _wait_for_llm_window(
                screen,
                pilot,
                previous=window,
            )
            assert replacement is not window
            stale_stop_syncs: list[str] = []
            monkeypatch.setattr(
                window,
                "_sync_process_controls",
                lambda provider: stale_stop_syncs.append(provider),
            )
            stop_process.release_wait.set()
            await stop_task
            await pilot.pause()
            assert stale_stop_syncs == []
            assert (
                replacement.query_one("#ollama-start-service-button", Button).disabled
                is False
            )
            assert (
                replacement.query_one("#ollama-stop-service-button", Button).disabled
                is True
            )

            models_dir = tmp_path / "models"
            models_dir.mkdir()
            replacement.query_one("#transformers-models-dir-path", Input).value = str(
                models_dir
            )
            scan_started = threading.Event()
            release_scan = threading.Event()

            def blocking_scan(models_path: Path) -> list[str]:
                assert models_path == models_dir.resolve()
                scan_started.set()
                assert release_scan.wait(timeout=5)
                return ["stale/transformer-result"]

            monkeypatch.setattr(
                transformers_events,
                "scan_transformers_local_models",
                blocking_scan,
            )
            stale_models_log = replacement.query_one(
                "#transformers-local-models-list",
                RichLog,
            )
            transformers_event = Button.Pressed(
                replacement.query_one("#transformers-list-local-models-button", Button)
            )
            transformers_task = asyncio.create_task(
                replacement.on_button_pressed(transformers_event)
            )
            assert await asyncio.to_thread(scan_started.wait, 5)

            screen.refresh(recompose=True)
            current_window = await _wait_for_llm_window(
                screen,
                pilot,
                previous=replacement,
            )
            assert current_window is not replacement
            release_scan.set()
            await transformers_task
            await pilot.pause()
            assert "stale/transformer-result" not in _rich_log_text(stale_models_log)
            assert "stale/transformer-result" not in _rich_log_text(
                current_window.query_one("#transformers-local-models-list", RichLog)
            )

            current_window.query_one(
                "#ollama-server-url", Input
            ).value = "http://127.0.0.1:11434"
            assert current_llm_destination(app) is current_window
            await pilot.click("#lab-models-row-ollama")
            await pilot.pause()
            first_list_started = threading.Event()
            release_first_list = threading.Event()
            call_lock = threading.Lock()
            list_call_count = 0

            def sequenced_model_list(*, base_url: str):
                nonlocal list_call_count
                assert base_url == "http://127.0.0.1:11434"
                with call_lock:
                    list_call_count += 1
                    call_number = list_call_count
                if call_number == 1:
                    first_list_started.set()
                    assert release_first_list.wait(timeout=5)
                    return {"models": [{"name": "older/model"}]}, None
                return {"models": [{"name": "newer/model"}]}, None

            monkeypatch.setattr(
                ollama_events,
                "ollama_list_local_models",
                sequenced_model_list,
            )
            list_button = current_window.query_one("#ollama-list-models-button", Button)
            first_list_task = asyncio.create_task(
                current_window.on_button_pressed(Button.Pressed(list_button))
            )
            assert await asyncio.to_thread(first_list_started.wait, 5)
            await current_window.on_button_pressed(Button.Pressed(list_button))
            assert (
                current_window._async_presentation_generations["ollama-combined-output"]
                == 2
            )
            release_first_list.set()
            await first_list_task
            await pilot.pause()
            list_output = _rich_log_text(
                current_window.query_one("#ollama-combined-output", RichLog)
            )
            assert "newer/model" in list_output
            assert "older/model" not in list_output
    finally:
        try:
            if app._rich_log_handler:
                await app._rich_log_handler.stop_processor()
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            await app.on_shutdown_request()
            await app.on_unmount()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_production_llm_duplicate_starts_are_reserved_for_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app._initial_tab_value = TAB_LLM
    executable = tmp_path / "server"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_text("model", encoding="utf-8")
    script = tmp_path / "server.py"
    script.write_text("pass\n", encoding="utf-8")

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _wait_for_llm_screen(app, pilot)
            window = screen.query_one(LLMManagementWindow)
            window.query_one("#llamacpp-exec-path", Input).value = str(executable)
            window.query_one("#llamacpp-model-path", Input).value = str(model)
            window.query_one("#llamafile-exec-path", Input).value = str(executable)
            window.query_one("#llamafile-model-path", Input).value = str(model)
            window.query_one("#vllm-python-path", Input).value = "python"
            window.query_one("#vllm-model-path", Input).value = "org/model"
            window.query_one("#onnx-python-path", Input).value = "python"
            window.query_one("#onnx-script-path", Input).value = str(script)
            window.query_one("#mlx-model-path", Input).value = "org-model"
            window.query_one("#ollama-exec-path", Input).value = str(executable)
            for text_area_id in (
                "llamafile-additional-args",
                "vllm-additional-args",
                "onnx-additional-args",
                "mlx-additional-args",
            ):
                window.query_one(f"#{text_area_id}", TextArea).text = ""

            launches: list[tuple[object, dict[str, object]]] = []
            monkeypatch.setattr(
                app,
                "run_worker",
                lambda work, **kwargs: launches.append((work, kwargs)),
            )
            starts = {
                "llamacpp": "llamacpp-start-server-button",
                "llamafile": "llamafile-start-server-button",
                "vllm": "vllm-start-server-button",
                "onnx": "onnx-start-server-button",
                "mlx": "mlx-start-server-button",
                "ollama": "ollama-start-service-button",
            }
            for provider, button_id in starts.items():
                button = window.query_one(f"#{button_id}", Button)
                await window.on_button_pressed(Button.Pressed(button))
                await window.on_button_pressed(Button.Pressed(button))
                assert current_server_claim(app, provider) is not None

            assert len(launches) == len(starts)
            screen.refresh(recompose=True)
            remounted = await _wait_for_llm_window(
                screen,
                pilot,
                previous=window,
            )
            assert remounted is not window
            for provider, (start_id, stop_id) in remounted.SERVER_CONTROLS.items():
                assert remounted.query_one(f"#{start_id}", Button).disabled is True
                assert remounted.query_one(f"#{stop_id}", Button).disabled is False
    finally:
        try:
            if app._rich_log_handler:
                await app._rich_log_handler.stop_processor()
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            await app.on_shutdown_request()
            await app.on_unmount()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_production_llm_destination_owns_navigation_actions_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        get_cli_setting_without_splash,
    )
    app = TldwCli()
    app.app_config["_first_run"] = False
    app._initial_tab_value = TAB_LLM

    try:
        async with app.run_test(size=(140, 48)) as pilot:
            screen = await _wait_for_llm_screen(app, pilot)
            window = screen.query_one(LLMManagementWindow)

            await pilot.click("#lab-models-row-transformers")
            await pilot.pause()
            assert window.active_view == "transformers"
            assert not hasattr(app, "llm_active_view")

            for button_id, expected_tooltip in EXPECTED_BROWSE_TOOLTIPS.items():
                button = window.query_one(f"#{button_id}", Button)
                assert str(button.tooltip) == expected_tooltip

            await pilot.click("#lab-models-row-ollama")
            await pilot.pause()
            assert window.active_view == "ollama"

            ollama_executable = tmp_path / "ollama"
            ollama_executable.write_text("#!/bin/sh\n", encoding="utf-8")
            ollama_executable.chmod(0o700)
            window.query_one("#ollama-exec-path", Input).value = str(ollama_executable)
            worker_launches: list[tuple[object, dict[str, object]]] = []
            real_run_worker = app.run_worker

            def record_worker_launch(work, **kwargs):
                worker_launches.append((work, kwargs))
                return None

            monkeypatch.setattr(app, "run_worker", record_worker_launch)
            ollama_start_event = Button.Pressed(
                window.query_one("#ollama-start-service-button", Button)
            )
            await window.on_button_pressed(ollama_start_event)
            duplicate_start_event = Button.Pressed(
                window.query_one("#ollama-start-service-button", Button)
            )
            await window.on_button_pressed(duplicate_start_event)
            assert len(worker_launches) == 1
            assert callable(worker_launches[0][0])
            assert worker_launches[0][1]["thread"] is True
            assert worker_launches[0][1]["group"] == "ollama_serve"
            assert ollama_start_event._stop_propagation is True
            assert duplicate_start_event._stop_propagation is True
            ollama_start_button = window.query_one(
                "#ollama-start-service-button", Button
            )
            ollama_stop_button = window.query_one("#ollama-stop-service-button", Button)
            claim = current_server_claim(app, "ollama")
            assert claim is not None
            assert ollama_start_button.disabled is True
            assert ollama_stop_button.disabled is False

            old_window = window
            screen.refresh(recompose=True)
            window = await _wait_for_llm_window(
                screen,
                pilot,
                previous=old_window,
            )
            assert window is not old_window
            ollama_start_button = window.query_one(
                "#ollama-start-service-button", Button
            )
            ollama_stop_button = window.query_one("#ollama-stop-service-button", Button)
            assert ollama_start_button.disabled is True
            assert ollama_stop_button.disabled is False

            completed_ollama_process = _CompletedOllamaProcess()
            ollama_scenario = _SubprocessScenario([completed_ollama_process])
            monkeypatch.setattr(ollama_events, "subprocess", ollama_scenario)
            worker_result = await asyncio.to_thread(worker_launches[0][0])
            await pilot.pause()
            monkeypatch.setattr(ollama_events, "subprocess", subprocess)
            assert worker_result == "ollama server exited (code=0)"
            assert app.ollama_server_process is None
            assert current_server_claim(app, "ollama") is None
            assert ollama_start_button.disabled is False
            assert ollama_stop_button.disabled is True
            window.query_one("#ollama-exec-path", Input).value = str(ollama_executable)
            assert "ollama-ready" not in _rich_log_text(
                window.query_one("#ollama-log-output", RichLog)
            )

            await pilot.click("#lab-models-row-ollama")
            await pilot.pause()
            monkeypatch.setattr(
                ollama_events,
                "ollama_model_info",
                lambda **kwargs: (
                    {
                        "family": "llama",
                        "format": "gguf",
                        "api_key": "PRIVATE_SUCCESS_SECRET",
                    },
                    None,
                ),
            )
            window.query_one("#ollama-show-model-name", Input).value = "llama3"
            show_event = Button.Pressed(
                window.query_one("#ollama-show-model-button", Button)
            )
            await window.on_button_pressed(show_event)
            await pilot.pause()
            model_info_output = _rich_log_text(
                window.query_one("#ollama-combined-output", RichLog)
            )
            assert '"family": "llama"' in model_info_output
            assert '"format": "gguf"' in model_info_output
            assert "PRIVATE_SUCCESS_SECRET" not in model_info_output

            monkeypatch.setattr(
                ollama_events,
                "ollama_generate_embeddings",
                lambda **kwargs: ({"embedding": [0.125, -0.5]}, None),
            )
            window.query_one("#ollama-embeddings-model-name", Input).value = "llama3"
            window.query_one("#ollama-embeddings-prompt", Input).value = "prompt"
            embedding_event = Button.Pressed(
                window.query_one("#ollama-embeddings-button", Button)
            )
            await window.on_button_pressed(embedding_event)
            await pilot.pause()
            embedding_output = _rich_log_text(
                window.query_one("#ollama-combined-output", RichLog)
            )
            assert "0.125" in embedding_output
            assert "-0.5" in embedding_output

            private_launch_failure = "/private/secret --token credential-value"
            lifecycle_notifications: list[str] = []
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, *args, **kwargs: lifecycle_notifications.append(
                    str(message)
                ),
            )
            launch_failure = _FailingSubprocessScenario(private_launch_failure)
            monkeypatch.setattr(ollama_events, "subprocess", launch_failure)
            failure_event = Button.Pressed(ollama_start_button)
            await window.on_button_pressed(failure_event)
            assert len(worker_launches) == 2
            failure_result = await asyncio.to_thread(worker_launches[1][0])
            await pilot.pause()
            assert failure_result == "ollama server failed (category=OSError)"
            assert current_server_claim(app, "ollama") is None
            assert ollama_start_button.disabled is False
            assert ollama_stop_button.disabled is True
            lifecycle_status = "\n".join(lifecycle_notifications)
            assert "ollama server failed (category=OSError)" in lifecycle_status
            assert private_launch_failure not in lifecycle_status

            nonzero_process = _CompletedOllamaProcess()
            nonzero_process.returncode = 7
            monkeypatch.setattr(
                ollama_events,
                "subprocess",
                _SubprocessScenario([nonzero_process]),
            )
            nonzero_event = Button.Pressed(ollama_start_button)
            await window.on_button_pressed(nonzero_event)
            assert len(worker_launches) == 3
            nonzero_result = await asyncio.to_thread(worker_launches[2][0])
            await pilot.pause()
            monkeypatch.setattr(ollama_events, "subprocess", subprocess)
            assert nonzero_result == "ollama server exited (code=7)"
            assert current_server_claim(app, "ollama") is None
            assert ollama_start_button.disabled is False
            assert ollama_stop_button.disabled is True
            lifecycle_status = "\n".join(lifecycle_notifications)
            assert "ollama server exited (code=7)" in lifecycle_status
            assert private_launch_failure not in lifecycle_status
            monkeypatch.setattr(app, "run_worker", real_run_worker)

            await pilot.click("#lab-models-row-transformers")
            await pilot.pause()
            assert window.active_view == "transformers"

            list_calls = 0
            original_list = LLMManagementWindow.ACTION_HANDLERS[
                "transformers-list-local-models-button"
            ]

            async def count_list(window_arg, app_arg, event):
                nonlocal list_calls
                list_calls += 1
                await original_list(window_arg, app_arg, event)

            monkeypatch.setitem(
                LLMManagementWindow.ACTION_HANDLERS,
                "transformers-list-local-models-button",
                count_list,
            )
            await pilot.click("#transformers-list-local-models-button")
            await pilot.pause()
            assert list_calls == 1

            models_dir = tmp_path / "models"
            models_dir.mkdir()
            window.query_one("#transformers-models-dir-path", Input).value = str(
                models_dir
            )
            scanned_paths: list[Path] = []

            def slow_local_model_scan(models_path: Path) -> list[str]:
                scanned_paths.append(models_path)
                time.sleep(0.03)
                return ["org/model"]

            monkeypatch.setattr(
                transformers_events,
                "scan_transformers_local_models",
                slow_local_model_scan,
                raising=False,
            )
            list_event = Button.Pressed(
                window.query_one("#transformers-list-local-models-button", Button)
            )
            pulse = asyncio.create_task(asyncio.sleep(0.005))
            await window.on_button_pressed(list_event)
            await pilot.pause()
            assert list_calls == 2
            assert pulse.done(), "local-model scan blocked Textual's event loop"
            assert scanned_paths == [models_dir.resolve()]
            assert list_event._stop_propagation is True
            assert "org/model" in _rich_log_text(
                window.query_one("#transformers-local-models-list", RichLog)
            )
            assert "Local model scan complete." in _rich_log_text(
                window.query_one("#transformers-log-output", RichLog)
            )

            download_calls = 0
            original_download = LLMManagementWindow.ACTION_HANDLERS[
                "transformers-download-model-button"
            ]

            async def count_download(window_arg, app_arg, event):
                nonlocal download_calls
                download_calls += 1
                await original_download(window_arg, app_arg, event)

            monkeypatch.setitem(
                LLMManagementWindow.ACTION_HANDLERS,
                "transformers-download-model-button",
                count_download,
            )
            download_event = Button.Pressed(
                window.query_one("#transformers-download-model-button", Button)
            )
            await window.on_button_pressed(download_event)
            await pilot.pause()
            assert download_calls == 1

            unsafe_launches: list[object] = []
            monkeypatch.setattr(
                app,
                "run_worker",
                lambda work, **kwargs: unsafe_launches.append(work),
            )
            for unsafe_repo_id in ("--token", "../outside", r"org\\outside"):
                window.query_one(
                    "#transformers-download-repo-id", Input
                ).value = unsafe_repo_id
                unsafe_event = Button.Pressed(
                    window.query_one("#transformers-download-model-button", Button)
                )
                await window.on_button_pressed(unsafe_event)
            assert unsafe_launches == []
            assert not (tmp_path / "outside").exists()

            window.query_one(
                "#transformers-download-repo-id", Input
            ).value = "org/model"
            window.query_one("#transformers-download-revision", Input).value = "--token"
            unsafe_revision_event = Button.Pressed(
                window.query_one("#transformers-download-model-button", Button)
            )
            await window.on_button_pressed(unsafe_revision_event)
            assert unsafe_launches == []
            window.query_one("#transformers-download-revision", Input).value = ""
            transformer_launches: list[tuple[object, dict[str, object]]] = []

            def record_transformer_launch(work, **kwargs):
                transformer_launches.append((work, kwargs))
                return None

            monkeypatch.setattr(app, "run_worker", record_transformer_launch)
            valid_download_event = Button.Pressed(
                window.query_one("#transformers-download-model-button", Button)
            )
            await window.on_button_pressed(valid_download_event)
            assert download_calls == 6
            assert len(transformer_launches) == 1
            transformer_work = transformer_launches[0][0]
            if inspect.iscoroutine(transformer_work):
                transformer_work.close()
            assert callable(transformer_work)
            assert transformer_launches[0][1]["thread"] is True
            assert valid_download_event._stop_propagation is True

            timeout_transformers_process = _TimeoutTransformersProcess()
            transformers_scenario = _SubprocessScenario(
                [
                    _TransformersProcess(0, "download-complete", ""),
                    _TransformersProcess(7, "", "download-failed"),
                    timeout_transformers_process,
                ]
            )
            monkeypatch.setattr(
                transformers_events,
                "subprocess",
                transformers_scenario,
            )
            real_call_from_thread = app.call_from_thread
            monkeypatch.setattr(
                app,
                "call_from_thread",
                lambda callback, *args, **kwargs: callback(*args, **kwargs),
            )
            success_result = await asyncio.to_thread(transformer_work)
            failure_result = await asyncio.to_thread(transformer_work)
            transformers_output = _rich_log_text(
                window.query_one("#transformers-log-output", RichLog)
            )
            old_transformers_window = window
            screen.refresh(recompose=True)
            window = await _wait_for_llm_window(
                screen,
                pilot,
                previous=old_transformers_window,
            )
            assert window is not old_transformers_window
            timeout_result = await asyncio.to_thread(transformer_work)
            await pilot.pause()
            await pilot.click("#lab-models-row-transformers")
            await pilot.pause()
            remounted_transformers_output = _rich_log_text(
                window.query_one("#transformers-log-output", RichLog)
            )
            monkeypatch.setattr(app, "call_from_thread", real_call_from_thread)
            monkeypatch.setattr(transformers_events, "subprocess", subprocess)
            monkeypatch.setattr(app, "run_worker", real_run_worker)
            assert "completed successfully" in success_result
            assert "failed with code: 7" in failure_result
            assert "timed out" in timeout_result
            assert "timed out" in remounted_transformers_output
            assert timeout_transformers_process.calls == [
                ("wait", 600),
                "terminate",
                ("wait", 5),
                "kill",
                ("wait", 5),
            ]
            assert all(
                kwargs["stdout"] is subprocess.DEVNULL
                and kwargs["stderr"] is subprocess.DEVNULL
                for kwargs in transformers_scenario.popen_kwargs
            )
            assert "download-complete" not in transformers_output
            assert "download-failed" not in transformers_output
            assert "PRIVATE_STDOUT" not in transformers_output
            assert "PRIVATE_STDERR" not in transformers_output

            secret = "/private/secret/model-id command --token value"
            recovery_notifications: list[str] = []
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, *args, **kwargs: recovery_notifications.append(
                    str(message)
                ),
            )

            def fail_registered_action(window_arg, app_arg, event):
                raise RuntimeError(secret)

            monkeypatch.setitem(
                LLMManagementWindow.ACTION_HANDLERS,
                "transformers-download-model-button",
                fail_registered_action,
            )
            fault_event = Button.Pressed(
                window.query_one("#transformers-download-model-button", Button)
            )
            await window.on_button_pressed(fault_event)
            await pilot.pause()
            recovery = "\n".join(recovery_notifications)
            assert fault_event._stop_propagation is True
            assert "transformers-download-model-button" in recovery
            assert "RuntimeError" in recovery
            assert secret not in recovery
            assert len(recovery.rsplit("\n", 1)[-1]) <= 200

            unknown = Button("Unknown", id="llm-unknown-action")
            await window.mount(unknown)
            unknown_event = Button.Pressed(unknown)
            await window.on_button_pressed(unknown_event)
            assert unknown_event._stop_propagation is True
    finally:
        try:
            if app._rich_log_handler:
                await app._rich_log_handler.stop_processor()
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            await app.on_shutdown_request()
            await app.on_unmount()
        except Exception:
            pass
