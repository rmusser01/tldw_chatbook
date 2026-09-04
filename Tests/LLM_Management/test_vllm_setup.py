from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    return run_vllm_preflight(
        draft,
        4,
        which=lambda name: str(cli_path) if name == "vllm" else None,
        run=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "0.9.0\n"})(),
        port_available=lambda host, port: True,
    )


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

    monkeypatch.setattr(vllm_events, "stop_server_process", fake_stop)
    await vllm_events.handle_vllm_setup_stop_requested(window, object(), object())

    assert calls == [("vllm", "vLLM server")]
    assert states == [VllmReadinessState.STOPPING, VllmReadinessState.NOT_CONFIGURED]
