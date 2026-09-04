from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import time

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_vllm as vllm_events,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmIssue,
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmReadinessState,
    build_vllm_command,
    client_api_url,
    run_vllm_preflight,
    semantic_fingerprint,
    validate_raw_arguments,
)
from tldw_chatbook.UI.LLM_Management import vllm_setup
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedSelectDirectory
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow


def local_draft(**changes: object) -> VllmLaunchDraft:
    values = {
        "mode": VllmMode.LOCAL,
        "python_environment": "python",
        "model_source": VllmModelSource.HUGGING_FACE,
        "model_value": "org/model",
    }
    values.update(changes)
    return VllmLaunchDraft(**values)


def passing_preflight(
    draft: VllmLaunchDraft, *, cli_path: Path
):
    cli_path.parent.mkdir(parents=True, exist_ok=True)
    cli_path.touch()
    cli_path.chmod(0o755)
    python_path = (
        Path(draft.python_environment)
        if Path(draft.python_environment).parent != Path(".")
        else cli_path.with_name("python")
    )
    python_path.touch()
    python_path.chmod(0o755)

    def probe(argv, **kwargs):
        version = "Python 3.12.0" if argv[0] == str(python_path) else "vLLM 0.9.0"
        return type("Result", (), {"returncode": 0, "stdout": version})()

    return run_vllm_preflight(
        draft,
        4,
        which=lambda name: str(python_path) if name == draft.python_environment else None,
        run=probe,
        port_available=lambda host, port: True,
    )


def test_preflight_rejects_oversize_or_unclassified_probe_output(tmp_path):
    cli_path = tmp_path / "bin/vllm"
    python_path = tmp_path / "bin/python"
    cli_path.parent.mkdir()
    cli_path.touch()
    python_path.touch()
    cli_path.chmod(0o755)
    python_path.chmod(0o755)

    def noisy_run(argv, **kwargs):
        return type("Result", (), {"returncode": 0, "stdout": "CANARY_SECRET_" + "x" * 2048})()

    result = run_vllm_preflight(
        local_draft(python_environment=str(python_path)),
        4,
        which=lambda _: None,
        run=noisy_run,
        port_available=lambda host, port: True,
    )
    assert result.python_version is None
    assert result.vllm_version is None
    assert all("CANARY_SECRET" not in issue.detail for issue in result.issues)


def test_default_probe_kills_and_reaps_child_at_output_byte_ceiling(tmp_path):
    pid_path = tmp_path / "noisy-child.pid"
    executable = tmp_path / "noisy-probe.py"
    executable.write_text(
        "#!" + os.sys.executable + "\n"
        "import os, sys\n"
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n"
        "    sys.stdout.buffer.write(b'X' * 64)\n"
        "    sys.stdout.buffer.flush()\n"
    )
    executable.chmod(0o755)

    started = time.monotonic()
    succeeded, version = vllm_setup._run_probe(subprocess.run, [str(executable)])
    elapsed = time.monotonic() - started

    assert succeeded is False
    assert version is None
    assert elapsed < 2
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_bare_python_requires_sibling_vllm_not_path_lookup(tmp_path):
    python_path = tmp_path / "venv/bin/python"
    unrelated_vllm = tmp_path / "other/vllm"
    python_path.parent.mkdir(parents=True)
    unrelated_vllm.parent.mkdir()
    python_path.touch()
    unrelated_vllm.touch()
    python_path.chmod(0o755)
    unrelated_vllm.chmod(0o755)
    result = run_vllm_preflight(
        local_draft(),
        4,
        which=lambda name: str(python_path) if name == "python" else str(unrelated_vllm),
        run=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "Python 3.12.0"})(),
        port_available=lambda host, port: True,
    )
    assert VllmIssue("vllm_cli_unavailable", "python_environment") in result.issues


@pytest.mark.asyncio
async def test_local_directory_picker_returns_selected_directory_through_callback(tmp_path):
    selected = tmp_path / "model"
    selected.mkdir()
    received = {}

    class Input:
        value = ""

    input_widget = Input()
    window = type(
        "Window",
        (),
        {"is_mounted": True, "query_one": lambda self, selector, kind: input_widget},
    )()

    class App:
        screen_stack = [type("Screen", (), {"llm_window": window})()]

        async def push_screen(self, picker, callback):
            received["picker"] = picker
            received["callback"] = callback

    app = App()
    await vllm_events.handle_vllm_local_directory_browse_requested(window, app, object())
    assert isinstance(received["picker"], EnhancedSelectDirectory)
    await received["callback"](selected)
    assert input_widget.value == str(selected)


def test_explicit_environment_rejects_unrelated_global_vllm_cli(tmp_path):
    python_path = tmp_path / "venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    result = run_vllm_preflight(
        local_draft(python_environment=str(python_path)),
        4,
        which=lambda _: "/usr/local/bin/vllm",
        run=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "0.9.0\n"})(),
        port_available=lambda host, port: True,
    )
    assert VllmIssue("vllm_cli_unavailable", "python_environment") in result.issues


def test_preflight_rejects_a_non_executable_vllm_cli(tmp_path):
    cli_path = tmp_path / "vllm"
    cli_path.touch()
    result = run_vllm_preflight(
        local_draft(),
        4,
        which=lambda _: str(cli_path),
        run=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "0.9.0\n"})(),
        port_available=lambda host, port: True,
    )
    assert VllmIssue("vllm_cli_unavailable", "python_environment") in result.issues


def test_local_command_uses_public_cli_and_one_served_alias(tmp_path):
    draft = local_draft(python_environment=str(tmp_path / "venv/bin/python"))
    result = passing_preflight(draft, cli_path=tmp_path / "venv/bin/vllm")
    command = build_vllm_command(draft, result)
    assert command[:3] == (str(result.cli_path), "serve", "org/model")
    assert command.count("--served-model-name") == 1
    assert command[command.index("--served-model-name") + 1] == "chatbook-vllm"
    assert "vllm.entrypoints" not in " ".join(command)


@pytest.mark.parametrize(
    "raw,flag",
    [
        ("--host 0.0.0.0", "--host"),
        ("--port=9000", "--port"),
        ("--model other", "--model"),
        ("--served-model-name other", "--served-model-name"),
        ("--api-key secret", "--api-key"),
        ("--hf-token secret", "--hf-token"),
    ],
)
def test_raw_arguments_cannot_override_managed_or_secret_flags(raw, flag):
    errors = validate_raw_arguments(raw)
    assert errors == (VllmIssue("arguments_conflict", "raw_arguments", flag),)


@pytest.mark.parametrize("model", ["org/model", "meta-llama/Llama-3.1-8B-Instruct"])
def test_preflight_accepts_hugging_face_repository_ids(model, tmp_path):
    result = passing_preflight(
        local_draft(model_value=model), cli_path=tmp_path / "vllm"
    )
    assert result.issues == ()


def test_preflight_validates_selected_local_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = passing_preflight(
        local_draft(
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value=str(model_dir),
        ),
        cli_path=tmp_path / "vllm",
    )
    assert result.issues == ()


def test_preflight_reports_missing_local_model_directory(tmp_path):
    result = passing_preflight(
        local_draft(
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value=str(tmp_path / "missing"),
        ),
        cli_path=tmp_path / "vllm",
    )
    assert VllmIssue("invalid_model_directory", "model_value") in result.issues


@pytest.mark.parametrize(
    ("bind_address", "expected"),
    [("0.0.0.0", "http://127.0.0.1:8000/v1"), ("::", "http://[::1]:8000/v1")],
)
def test_wildcard_binds_use_loopback_client_urls(bind_address, expected):
    assert client_api_url(bind_address, 8000) == expected


def test_ipv6_wildcard_availability_checks_the_requested_bind(monkeypatch):
    """Substituting ::1 misses conflicts created by platform dual-stack policy."""

    calls = []

    class SocketDouble:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def setsockopt(self, *args):
            calls.append(("setsockopt", args))

        def bind(self, address):
            calls.append(("bind", address))

    def open_socket(family, kind):
        calls.append(("socket", (family, kind)))
        return SocketDouble()

    monkeypatch.setattr(vllm_setup.socket, "socket", open_socket)

    assert vllm_setup.is_port_available("::", 8000)
    assert ("bind", ("::", 8000)) in calls
    assert ("bind", ("::1", 8000)) not in calls


def test_defaults_are_real_and_safe_values():
    draft = local_draft()
    assert draft.bind_address == "127.0.0.1"
    assert draft.port == 8000
    assert draft.trust_remote_code is False


@pytest.mark.parametrize(
    "change",
    [
        {"port": 0},
        {"tensor_parallel_size": 0},
        {"maximum_model_length": 0},
        {"gpu_memory_utilization": 0.0},
        {"gpu_memory_utilization": 1.1},
        {"tensor_parallel_size": "two"},
        {"gpu_memory_utilization": "0.9"},
    ],
)
def test_preflight_rejects_out_of_bounds_structured_values(change, tmp_path):
    result = passing_preflight(local_draft(**change), cli_path=tmp_path / "vllm")
    assert result.issues


def test_semantic_fingerprint_changes_for_every_launch_field_except_profile_name():
    draft = local_draft()
    baseline = semantic_fingerprint(draft)
    for field, value in {
        "mode": VllmMode.EXISTING,
        "python_environment": "/tmp/python",
        "model_source": VllmModelSource.LOCAL_DIRECTORY,
        "model_value": "/tmp/model",
        "bind_address": "0.0.0.0",
        "port": 9000,
        "existing_server_url": "https://example.test/v1",
        "dtype": "float16",
        "tensor_parallel_size": 2,
        "maximum_model_length": 4096,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "raw_arguments": "--enable-prefix-caching",
    }.items():
        assert semantic_fingerprint(replace(draft, **{field: value})) != baseline


def test_command_rejects_stale_or_failed_preflight(tmp_path):
    draft = local_draft()
    failed = run_vllm_preflight(
        draft,
        1,
        which=lambda _: None,
        run=lambda *args, **kwargs: None,
        port_available=lambda host, port: True,
    )
    with pytest.raises(ValueError, match="successful current preflight"):
        build_vllm_command(draft, failed)
    successful = passing_preflight(draft, cli_path=tmp_path / "vllm")
    with pytest.raises(ValueError, match="matching fingerprint"):
        build_vllm_command(replace(draft, port=8001), successful)
    with pytest.raises(ValueError, match="current generation"):
        build_vllm_command(draft, successful, current_generation=5)


def test_legacy_vllm_buttons_are_not_registered():
    assert vllm_events.VLLM_BUTTON_HANDLERS == {}


def test_same_draft_old_success_cannot_launch_after_newer_check_failed(tmp_path):
    draft = local_draft()
    old_success = passing_preflight(draft, cli_path=tmp_path / "vllm")

    class View:
        preflight = old_success

        def apply_state(self, **kwargs):
            self.state = kwargs["state"]

    view = View()
    window = type(
        "Window",
        (),
        {"_vllm_preflight_generation": 5, "query_one": lambda self, selector: view},
    )()

    class App:
        def __init__(self):
            self.workers = []

        def run_worker(self, *args, **kwargs):
            self.workers.append((args, kwargs))

    event = type("Event", (), {"draft": draft})()
    app = App()
    vllm_events.handle_vllm_setup_start_requested(window, app, event)
    assert app.workers == []
    assert view.state is VllmReadinessState.NEEDS_ATTENTION


def test_lifecycle_sync_projects_vllm_without_legacy_button_queries():
    projections = []

    class View:
        def project_lifecycle(self, **kwargs):
            projections.append(kwargs)

    window = LLMManagementWindow.__new__(LLMManagementWindow)
    window.query_one = lambda selector, kind: View()
    window._server_active = lambda provider: provider == "vllm"
    window.app_instance = type("App", (), {"notify": lambda *args, **kwargs: None})()
    window._handle_server_process_state_change("vllm", "process exited")
    assert projections == [{"active": True, "status": "process exited"}]


@pytest.mark.asyncio
async def test_stop_request_settles_the_owned_server_without_opening_a_picker(monkeypatch):
    draft = local_draft()
    states = []

    class View:
        preflight = None

        def __init__(self) -> None:
            self.view_draft = draft

        @property
        def draft(self):
            return self.view_draft

        def apply_state(self, **kwargs):
            states.append(kwargs["state"])

    view = View()
    window = type("Window", (), {"query_one": lambda self, selector: view})()
    calls = []

    async def fake_stop(app, provider, display_name):
        calls.append((provider, display_name))
        return True

    monkeypatch.setattr(vllm_events, "stop_server_process", fake_stop)
    await vllm_events.handle_vllm_setup_stop_requested(window, object(), object())

    assert calls == [("vllm", "vLLM server")]
    assert states == [VllmReadinessState.STOPPING, VllmReadinessState.NOT_CONFIGURED]


@pytest.mark.asyncio
async def test_failed_stop_keeps_recovery_state(monkeypatch):
    draft = local_draft()
    states = []

    class View:
        preflight = None

        def __init__(self) -> None:
            self.draft = draft

        def apply_state(self, **kwargs):
            states.append(kwargs["state"])

    view = View()
    window = type("Window", (), {"query_one": lambda self, selector: view})()

    async def failed_stop(app, provider, display_name):
        return False

    monkeypatch.setattr(vllm_events, "stop_server_process", failed_stop)
    await vllm_events.handle_vllm_setup_stop_requested(window, object(), object())
    assert states == [VllmReadinessState.STOPPING, VllmReadinessState.NEEDS_ATTENTION]
