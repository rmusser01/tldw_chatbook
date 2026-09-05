from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management.snapshot_admission import (
    compatibility_matches,
    finalize_launch,
    prepare_launch,
    revalidate_files,
)
from tldw_chatbook.LLM_Management.snapshot_models import (
    CatalogPage,
    CompatibilityEvidence,
    ManagerView,
    ReadinessObservation,
    SaveResult,
    SlotObservation,
    SlotReceipt,
    SnapshotError,
    SnapshotRecord,
    WorkingFile,
)

# Argument aliases, environment names, and domains below are fixtures from
# common/arg.cpp and tools/server/README.md at llama.cpp 427291b5b34cd914a31b3fd3b61a68f6184f4b9f.


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _launch_files(tmp_path: Path) -> tuple[Path, Path]:
    executable = _write(tmp_path / "llama-server", b"pinned runtime")
    model = _write(tmp_path / "model.gguf", b"model bytes")
    return executable, model


def _explicit_command(executable: Path, model: Path, *extra: str) -> tuple[str, ...]:
    return (
        str(executable),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--flash-attn",
        "off",
        "--fit",
        "off",
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
        "--no-mmproj",
        *extra,
    )


def _claim() -> ServerLaunchClaim:
    return ServerLaunchClaim(provider="llamacpp", authority="External GGUF")


def _ready(model: Path, *runtime_values: tuple[str, str]) -> ReadinessObservation:
    return ReadinessObservation(
        slots=(
            SlotObservation(
                slot_id=0,
                busy=False,
                tokens=7,
                context_size=4096,
                observed_at=12.5,
            ),
        ),
        build_info="427291b5b34c",
        model_path=str(model.resolve()),
        runtime_values=tuple(runtime_values),
    )


def _finalized(tmp_path: Path, *extra: str):
    executable, model = _launch_files(tmp_path)
    prepared = prepare_launch(
        _explicit_command(executable, model, *extra),
        {},
        _claim(),
        "launch-1",
    )
    return finalize_launch(prepared, _ready(model))


def test_last_host_and_port_arguments_win(tmp_path: Path) -> None:
    """A stale form endpoint must not override later additional arguments."""
    executable, model = _launch_files(tmp_path)
    command = _explicit_command(
        executable,
        model,
        "--host",
        "192.0.2.2",
        "--host=127.0.0.2",
        "--port",
        "9000",
        "--port=8123",
    )

    descriptor = prepare_launch(command, {}, _claim(), "launch-last-wins")

    assert descriptor.disabled_reason is None
    assert descriptor.base_url == "http://127.0.0.2:8123"


def test_equals_syntax_and_pinned_aliases_are_canonicalized(tmp_path: Path) -> None:
    """Pinned aliases must produce one semantic launch identity."""
    executable, model = _launch_files(tmp_path)
    command = (
        str(executable),
        "-m=" + str(model),
        "--host=127.0.0.1",
        "--port=8081",
        "-c=2048",
        "-np=2",
        "-fa=off",
        "-fit=off",
        "-dev=none",
        "-ngl=0",
        "--no-mmproj",
        "-ctk=q8_0",
        "-ctv=f16",
        "-cb",
        "--context-shift",
    )

    prepared = prepare_launch(command, {}, _claim(), "launch-aliases")
    finalized = finalize_launch(
        prepared,
        ReadinessObservation(
            slots=(
                SlotObservation(
                    slot_id=0,
                    busy=False,
                    tokens=None,
                    context_size=1024,
                    observed_at=1.0,
                ),
                SlotObservation(
                    slot_id=1,
                    busy=False,
                    tokens=None,
                    context_size=1024,
                    observed_at=1.0,
                ),
            ),
            build_info="427291b5b34c",
            model_path=str(model.resolve()),
            runtime_values=(),
        ),
    )

    assert finalized.disabled_reason is None
    settings = dict(finalized.compatibility.state_settings)
    assert settings["ctx-size"] == "2048"
    assert settings["parallel"] == "2"
    assert settings["flash-attn"] == "off"
    assert settings["fit"] == "off"
    assert settings["device"] == "none"
    assert settings["gpu-layers"] == "0"
    assert settings["cache-type-k"] == "q8_0"
    assert settings["cache-type-v"] == "f16"
    assert settings["cont-batching"] == "on"
    assert settings["context-shift"] == "on"
    assert settings["effective-slot-contexts"] == "0:1024,1:1024"
    assert set(settings) == {
        "batch-size",
        "cache-type-k",
        "cache-type-v",
        "cont-batching",
        "context-shift",
        "ctx-size",
        "device",
        "effective-slot-contexts",
        "fit",
        "fit-ctx",
        "fit-target",
        "flash-attn",
        "gpu-layers",
        "image-max-tokens",
        "image-min-tokens",
        "keep",
        "kv-offload",
        "main-gpu",
        "mmproj-auto",
        "mmproj-device",
        "mmproj-offload",
        "mtmd-batch-max-tokens",
        "parallel",
        "rope-freq-base",
        "rope-freq-scale",
        "rope-scale",
        "rope-scaling",
        "split-mode",
        "swa-full",
        "tensor-split",
        "ubatch-size",
        "yarn-attn-factor",
        "yarn-beta-fast",
        "yarn-beta-slow",
        "yarn-ext-factor",
        "yarn-orig-ctx",
    }
    assert settings["swa-full"] == "off"
    assert settings["rope-freq-base"] == "@model"
    assert settings["image-max-tokens"] == "@model"


def test_cli_values_override_recognized_environment_values(tmp_path: Path) -> None:
    """The child environment is fallback state, never authority over argv."""
    executable, model = _launch_files(tmp_path)
    env = {
        "LLAMA_ARG_HOST": "192.0.2.40",
        "LLAMA_ARG_PORT": "9000",
        "LLAMA_ARG_CTX_SIZE": "2048",
        "LLAMA_ARG_N_PARALLEL": "2",
        "LLAMA_ARG_FLASH_ATTN": "on",
        "LLAMA_ARG_FIT": "on",
        "LLAMA_ARG_DEVICE": "Metal0",
        "LLAMA_ARG_N_GPU_LAYERS": "99",
        "LLAMA_ARG_MMPROJ_AUTO": "true",
    }

    descriptor = prepare_launch(
        _explicit_command(executable, model), env, _claim(), "launch-cli-env"
    )

    assert descriptor.disabled_reason is None
    assert descriptor.base_url == "http://127.0.0.1:8080"
    with pytest.raises(TypeError):
        descriptor.child_env["LLAMA_ARG_PORT"] = "7777"


def test_recognized_environment_fallback_is_captured(tmp_path: Path) -> None:
    """A launch configured only through upstream env names remains identifiable."""
    executable, model = _launch_files(tmp_path)
    env = {
        "LLAMA_ARG_MODEL": str(model),
        "LLAMA_ARG_HOST": "127.0.0.3",
        "LLAMA_ARG_PORT": "8124",
        "LLAMA_ARG_CTX_SIZE": "4096",
        "LLAMA_ARG_N_PARALLEL": "1",
        "LLAMA_ARG_FLASH_ATTN": "off",
        "LLAMA_ARG_FIT": "off",
        "LLAMA_ARG_DEVICE": "none",
        "LLAMA_ARG_N_GPU_LAYERS": "0",
        "LLAMA_ARG_MMPROJ_AUTO": "false",
        "LLAMA_ARG_CONT_BATCHING": "enabled",
        "LLAMA_ARG_CONTEXT_SHIFT": "0",
        "LLAMA_ARG_KV_OFFLOAD": "on",
        "MTMD_BACKEND_DEVICE": "none",
    }

    prepared = prepare_launch((str(executable),), env, _claim(), "launch-env")
    finalized = finalize_launch(prepared, _ready(model))

    assert finalized.disabled_reason is None
    assert finalized.base_url == "http://127.0.0.3:8124"
    settings = dict(finalized.compatibility.state_settings)
    assert settings["cont-batching"] == "on"
    assert settings["context-shift"] == "off"
    assert settings["kv-offload"] == "on"
    assert settings["mmproj-device"] == "none"


def test_unknown_argument_or_llama_environment_disables_manager(
    tmp_path: Path,
) -> None:
    """Unknown upstream state cannot be misclassified as an unset known option."""
    executable, model = _launch_files(tmp_path)

    unknown_arg = prepare_launch(
        _explicit_command(executable, model, "--future-kv-layout", "x"),
        {},
        _claim(),
        "launch-unknown-arg",
    )
    unknown_env = prepare_launch(
        _explicit_command(executable, model),
        {"LLAMA_ARG_FUTURE_KV_LAYOUT": "x"},
        _claim(),
        "launch-unknown-env",
    )

    assert "unknown" in unknown_arg.disabled_reason.lower()
    assert "LLAMA_ARG_FUTURE_KV_LAYOUT" in unknown_env.disabled_reason


def test_known_non_state_controls_are_consumed_without_affecting_identity(
    tmp_path: Path,
) -> None:
    """A non-state value must not masquerade as the next state-affecting flag."""
    executable, model = _launch_files(tmp_path)
    first = prepare_launch(
        _explicit_command(
            executable,
            model,
            "--threads",
            "7",
            "--temperature=0.25",
            "--metrics",
            "--verbosity",
            "4",
        ),
        {},
        _claim(),
        "launch-controls",
    )
    second = prepare_launch(
        _explicit_command(executable, model),
        {},
        _claim(),
        "launch-controls-2",
    )

    finalized_first = finalize_launch(first, _ready(model))
    finalized_second = finalize_launch(second, _ready(model))
    assert finalized_first.disabled_reason is None
    assert compatibility_matches(
        finalized_first.compatibility, finalized_second.compatibility
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--slot-save-path", "/tmp/foreign"),
        ("--slots",),
        ("--no-slots",),
    ],
)
def test_conflicting_owned_slot_flags_fail_snapshot_admission(
    arguments: tuple[str, ...], tmp_path: Path
) -> None:
    """User-owned slot management flags must not collide with app ownership."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, *arguments),
        {},
        _claim(),
        "launch-owned-conflict",
    )

    assert "slot" in descriptor.disabled_reason.lower()


@pytest.mark.parametrize(
    ("arguments", "reason_fragment"),
    [
        (("--lora", "adapter.gguf"), "lora"),
        (("--lora-scaled", "adapter.gguf:0.5"), "lora"),
        (("--control-vector", "control.gguf"), "control"),
        (("--override-kv", "a=str:b"), "metadata"),
        (("--spec-draft-model", "draft.gguf"), "speculative"),
        (("--rpc", "10.0.0.2:5000"), "rpc"),
        (("--models-dir", "/models"), "router"),
        (("--api-prefix", "/custom"), "prefix"),
        (("--ssl-key-file", "key.pem"), "tls"),
        (("--mmproj-url", "https://example.invalid/mmproj.gguf"), "projector"),
        (("--video-fps", "4"), "video"),
    ],
)
def test_unrepresentable_modes_fail_closed_with_specific_reason(
    arguments: tuple[str, ...], reason_fragment: str, tmp_path: Path
) -> None:
    """A special runtime mode must not be silently stripped from compatibility."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, *arguments),
        {},
        _claim(),
        "launch-unsupported",
    )

    assert reason_fragment in descriptor.disabled_reason.lower()
    assert not descriptor.disabled_reason.startswith("Unknown llama.cpp")


@pytest.mark.parametrize(
    ("arguments", "environment", "reason_fragment"),
    [
        (("-md", "draft.gguf"), {}, "speculative"),
        (("--model-draft", "draft.gguf"), {}, "speculative"),
        ((), {"LLAMA_ARG_API_PREFIX": "/custom"}, "prefix"),
        ((), {"LLAMA_ARG_SSL_CERT_FILE": "cert.pem"}, "tls"),
    ],
)
def test_alias_and_environment_special_modes_keep_specific_reasons(
    arguments: tuple[str, ...],
    environment: dict[str, str],
    reason_fragment: str,
    tmp_path: Path,
) -> None:
    """Pinned aliases and env forms must not collapse into an unknown-option error."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, *arguments),
        environment,
        _claim(),
        "launch-special-alias",
    )

    assert reason_fragment in descriptor.disabled_reason.lower()


@pytest.mark.parametrize("port", ["0", "65536", "abc", "1.5", ""])
def test_invalid_port_disables_manager(port: str, tmp_path: Path) -> None:
    """The management destination must be a valid TCP port."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--port", port),
        {},
        _claim(),
        "launch-bad-port",
    )

    assert "port" in descriptor.disabled_reason.lower()


def test_ipv6_loopback_is_frozen_as_a_bracketed_numeric_url(tmp_path: Path) -> None:
    """IPv6 management requests need an unambiguous numeric authority."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--host", "::1", "--port", "8125"),
        {},
        _claim(),
        "launch-v6",
    )

    assert descriptor.disabled_reason is None
    assert descriptor.base_url == "http://[::1]:8125"


def test_non_loopback_literal_disables_manager(tmp_path: Path) -> None:
    """Snapshot credentials must never be sent to a shared-network listener."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--host", "192.0.2.10"),
        {},
        _claim(),
        "launch-public",
    )

    assert "loopback" in descriptor.disabled_reason.lower()


def test_localhost_with_any_non_loopback_resolution_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """A mixed localhost DNS answer must not create a rebinding opportunity."""
    executable, model = _launch_files(tmp_path)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.9", 0)),
        ],
    )

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--host", "localhost"),
        {},
        _claim(),
        "launch-localhost",
    )

    assert "loopback" in descriptor.disabled_reason.lower()


def test_localhost_resolution_is_frozen_to_one_numeric_address(
    tmp_path: Path, monkeypatch
) -> None:
    """Repeated requests must not resolve localhost after launch admission."""
    executable, model = _launch_files(tmp_path)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ],
    )

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--host", "localhost"),
        {},
        _claim(),
        "launch-localhost-safe",
    )

    assert descriptor.disabled_reason is None
    assert descriptor.base_url == "http://[::1]:8080"


def test_explicit_api_key_overrides_environment_and_stays_out_of_repr(
    tmp_path: Path,
) -> None:
    """Launch credentials are memory-only and argv remains authoritative."""
    executable, model = _launch_files(tmp_path)
    secret = "argv-secret-sentinel"

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--api-key", secret),
        {"LLAMA_API_KEY": "env-secret-sentinel"},
        _claim(),
        "launch-auth",
    )

    assert descriptor.bearer_token == secret
    rendered = repr(descriptor)
    assert secret not in rendered
    assert "env-secret-sentinel" not in rendered
    assert str(model) not in rendered
    assert "ServerLaunchClaim" not in rendered


def test_api_key_list_uses_one_server_accepted_key(tmp_path: Path) -> None:
    """A comma-separated server key list must not become one invalid bearer token."""
    executable, model = _launch_files(tmp_path)

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--api-key", "first-key,second-key"),
        {},
        _claim(),
        "launch-key-list",
    )

    assert descriptor.disabled_reason is None
    assert descriptor.bearer_token == "first-key"


def test_explicit_key_file_overrides_environment_api_key(tmp_path: Path) -> None:
    """An explicit argv credential source must outrank environment fallback."""
    executable, model = _launch_files(tmp_path)
    key_file = tmp_path / "server.keys"
    key_file.write_text("file-key\n", encoding="utf-8")

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--api-key-file", str(key_file)),
        {"LLAMA_API_KEY": "environment-key"},
        _claim(),
        "launch-key-file-precedence",
    )

    assert descriptor.disabled_reason is None
    assert descriptor.bearer_token == "file-key"


def test_valid_key_file_uses_first_non_comment_key_without_repr_leak(
    tmp_path: Path,
) -> None:
    """A local key file must yield one bounded bearer token without path leakage."""
    executable, model = _launch_files(tmp_path)
    key_file = tmp_path / "server.keys"
    key_file.write_text("# comment\nfirst-key\nsecond-key\n", encoding="utf-8")

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--api-key-file", str(key_file)),
        {},
        _claim(),
        "launch-key-file",
    )

    assert descriptor.disabled_reason is None
    assert descriptor.bearer_token == "first-key"
    assert str(key_file) not in repr(descriptor)
    assert "first-key" not in repr(descriptor)


@pytest.mark.parametrize("kind", ["missing", "empty", "comment-only"])
def test_invalid_or_empty_key_file_disables_manager(kind: str, tmp_path: Path) -> None:
    """A key-file launch must not silently send unauthenticated management calls."""
    executable, model = _launch_files(tmp_path)
    key_file = tmp_path / "server.keys"
    if kind == "empty":
        key_file.write_text("", encoding="utf-8")
    elif kind == "comment-only":
        key_file.write_text("# no key\n", encoding="utf-8")

    descriptor = prepare_launch(
        _explicit_command(executable, model, "--api-key-file", str(key_file)),
        {},
        _claim(),
        "launch-bad-key-file",
    )

    assert "key file" in descriptor.disabled_reason.lower()


def test_model_and_projector_identity_changes_require_readmission(
    tmp_path: Path,
) -> None:
    """Replacement bytes must invalidate the launch instead of rebinding it."""
    executable, model = _launch_files(tmp_path)
    projector = _write(tmp_path / "mmproj.gguf", b"projector bytes")
    descriptor = prepare_launch(
        _explicit_command(
            executable, model, "--mmproj", str(projector), "--mmproj-auto"
        ),
        {},
        _claim(),
        "launch-files",
    )
    assert revalidate_files(descriptor) is True

    model.write_bytes(b"replacement model bytes")
    assert revalidate_files(descriptor) is False

    descriptor = prepare_launch(
        _explicit_command(
            executable, model, "--mmproj", str(projector), "--mmproj-auto"
        ),
        {},
        _claim(),
        "launch-files-2",
    )
    projector.write_bytes(b"replacement projector bytes")
    assert revalidate_files(descriptor) is False


def test_split_model_digest_uses_ordered_shard_manifest(tmp_path: Path) -> None:
    """Selecting any split requires every numbered shard in canonical order."""
    executable = _write(tmp_path / "llama-server", b"runtime")
    _write(tmp_path / "model-00001-of-00002.gguf", b"first")
    shard_two = _write(tmp_path / "model-00002-of-00002.gguf", b"second shard")
    prepared = prepare_launch(
        _explicit_command(executable, shard_two),
        {},
        _claim(),
        "launch-split",
    )
    finalized = finalize_launch(prepared, _ready(shard_two))
    first_digest = hashlib.sha256(b"first").hexdigest()
    second_digest = hashlib.sha256(b"second shard").hexdigest()
    manifest = (f"1:5:{first_digest}\n2:12:{second_digest}\n").encode("ascii")

    assert finalized.disabled_reason is None
    assert finalized.compatibility.model_sha256 == hashlib.sha256(manifest).hexdigest()
    assert [item.path.name for item in finalized.files[1:]] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]


def test_missing_split_shard_leaves_compatibility_unavailable(tmp_path: Path) -> None:
    """An incomplete split cannot yield a model identity."""
    executable = _write(tmp_path / "llama-server", b"runtime")
    shard = _write(tmp_path / "model-00001-of-00002.gguf", b"first")

    descriptor = prepare_launch(
        _explicit_command(executable, shard), {}, _claim(), "launch-missing-shard"
    )

    assert descriptor.compatibility is None
    assert "shard" in descriptor.disabled_reason.lower()


def test_non_gguf_model_file_cannot_supply_model_identity(tmp_path: Path) -> None:
    """An arbitrary regular file must not be admitted as the loaded GGUF model."""
    executable = _write(tmp_path / "llama-server", b"runtime")
    model = _write(tmp_path / "model.bin", b"not a GGUF identity")

    descriptor = prepare_launch(
        _explicit_command(executable, model), {}, _claim(), "launch-not-gguf"
    )

    assert descriptor.compatibility is None
    assert "GGUF" in descriptor.disabled_reason


@pytest.mark.parametrize(
    "observation",
    [
        lambda model: ReadinessObservation(
            slots=(
                SlotObservation(
                    slot_id=0,
                    busy=False,
                    tokens=0,
                    context_size=None,
                    observed_at=1.0,
                ),
            ),
            build_info="build",
            model_path=str(model.resolve()),
            runtime_values=(),
        ),
        lambda model: ReadinessObservation(
            slots=(
                SlotObservation(
                    slot_id=0,
                    busy=False,
                    tokens=0,
                    context_size=4096,
                    observed_at=1.0,
                ),
            ),
            build_info="",
            model_path=str(model.resolve()),
            runtime_values=(),
        ),
        lambda model: ReadinessObservation(
            slots=(
                SlotObservation(
                    slot_id=0,
                    busy=False,
                    tokens=0,
                    context_size=4096,
                    observed_at=1.0,
                ),
            ),
            build_info="build",
            model_path=str(model.with_name("other.gguf")),
            runtime_values=(),
        ),
    ],
)
def test_missing_readiness_evidence_keeps_compatibility_unavailable(
    observation, tmp_path: Path
) -> None:
    """Finalization must fail closed when build, model, or context is unverified."""
    executable, model = _launch_files(tmp_path)
    prepared = prepare_launch(
        _explicit_command(executable, model), {}, _claim(), "launch-incomplete"
    )

    finalized = finalize_launch(prepared, observation(model))

    assert finalized.compatibility is None
    assert finalized.disabled_reason is not None


def test_auto_state_requires_a_matching_runtime_observation(tmp_path: Path) -> None:
    """An effective auto value may not be guessed from its launch spelling."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    flash_index = command.index("off", command.index("--flash-attn"))
    command[flash_index] = "auto"
    prepared = prepare_launch(tuple(command), {}, _claim(), "launch-auto")

    missing = finalize_launch(prepared, _ready(model))
    observed = finalize_launch(prepared, _ready(model, ("flash-attn", "on")))

    assert missing.compatibility is None
    assert "flash-attn" in missing.disabled_reason
    assert observed.disabled_reason is None
    assert dict(observed.compatibility.state_settings)["flash-attn"] == "on"


@pytest.mark.parametrize(
    ("option", "original", "runtime_key"),
    [
        ("--flash-attn", "off", "flash-attn"),
        ("--n-gpu-layers", "0", "gpu-layers"),
        ("--device", "none", "device"),
    ],
)
def test_unresolved_auto_runtime_values_do_not_create_compatibility(
    option: str,
    original: str,
    runtime_key: str,
    tmp_path: Path,
) -> None:
    """An observed launch-domain auto value is not effective runtime evidence."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    value_index = command.index(original, command.index(option))
    command[value_index] = "auto"
    prepared = prepare_launch(tuple(command), {}, _claim(), "launch-unresolved-auto")

    assert prepared.disabled_reason is None
    finalized = finalize_launch(prepared, _ready(model, (runtime_key, "auto")))

    assert finalized.compatibility is None
    assert runtime_key in finalized.disabled_reason


@pytest.mark.parametrize("source", ["cli", "environment"])
def test_explicit_auto_parallelism_resolves_to_observed_slot_count(
    source: str, tmp_path: Path
) -> None:
    """Pinned -1=auto forms must resolve to the verified positive slot count."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    env: dict[str, str] = {}
    if source == "cli":
        command.extend(("--parallel", "-1"))
    else:
        parallel_index = command.index("--parallel")
        del command[parallel_index : parallel_index + 2]
        env["LLAMA_ARG_N_PARALLEL"] = "-1"
    observation = ReadinessObservation(
        slots=tuple(
            SlotObservation(
                slot_id=slot_id,
                busy=False,
                tokens=7,
                context_size=2048,
                observed_at=12.5,
            )
            for slot_id in range(2)
        ),
        build_info="427291b5b34c",
        model_path=str(model.resolve()),
        runtime_values=(),
    )

    finalized = finalize_launch(
        prepare_launch(tuple(command), env, _claim(), f"launch-auto-parallel-{source}"),
        observation,
    )

    assert finalized.disabled_reason is None
    assert dict(finalized.compatibility.state_settings)["parallel"] == "2"


def test_explicit_flash_attention_value_needs_no_auto_observation(
    tmp_path: Path,
) -> None:
    """An explicit on/off value is already effective-state evidence at launch."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    flash_index = command.index("off", command.index("--flash-attn"))
    command[flash_index] = "on"

    finalized = finalize_launch(
        prepare_launch(tuple(command), {}, _claim(), "launch-flash-on"),
        _ready(model),
    )

    assert finalized.disabled_reason is None
    assert dict(finalized.compatibility.state_settings)["flash-attn"] == "on"


def test_upstream_truthy_falsey_domains_are_canonicalized(tmp_path: Path) -> None:
    """Pinned bool-like value options accept the same documented spellings."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    flash_index = command.index("off", command.index("--flash-attn"))
    fit_index = command.index("off", command.index("--fit"))
    command[flash_index] = "enabled"
    command[fit_index] = "false"

    finalized = finalize_launch(
        prepare_launch(tuple(command), {}, _claim(), "launch-bool-domains"),
        _ready(model),
    )

    assert finalized.disabled_reason is None
    settings = dict(finalized.compatibility.state_settings)
    assert settings["flash-attn"] == "on"
    assert settings["fit"] == "off"


def test_fit_on_requires_observed_values_for_every_adjustable_setting(
    tmp_path: Path,
) -> None:
    """Fit must not turn launch defaults into guessed effective compatibility state."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    fit_index = command.index("off", command.index("--fit"))
    command[fit_index] = "on"
    prepared = prepare_launch(tuple(command), {}, _claim(), "launch-fit")

    incomplete = finalize_launch(prepared, _ready(model, ("fit", "on")))
    complete_values = tuple(
        sorted(
            {
                "batch-size": "2048",
                "ctx-size": "4096",
                "fit": "on",
                "gpu-layers": "0",
                "ubatch-size": "512",
            }.items()
        )
    )
    complete = finalize_launch(prepared, _ready(model, *complete_values))

    assert incomplete.compatibility is None
    assert incomplete.disabled_reason is not None
    assert complete.disabled_reason is None


def test_invalid_observed_runtime_value_fails_closed(tmp_path: Path) -> None:
    """A whitelisted key with an invalid domain is not compatibility evidence."""
    executable, model = _launch_files(tmp_path)
    command = list(_explicit_command(executable, model))
    flash_index = command.index("off", command.index("--flash-attn"))
    command[flash_index] = "auto"
    prepared = prepare_launch(tuple(command), {}, _claim(), "launch-bad-runtime")

    finalized = finalize_launch(prepared, _ready(model, ("flash-attn", "banana")))

    assert finalized.compatibility is None
    assert "flash-attn" in finalized.disabled_reason


def test_compatibility_requires_every_evidence_field_to_match(tmp_path: Path) -> None:
    """A build or state mismatch must never be classified as restorable."""
    current = _finalized(tmp_path)
    assert current.compatibility is not None
    same = current.compatibility.model_copy()
    different = current.compatibility.model_copy(update={"build_info": "other-build"})

    assert compatibility_matches(same, current.compatibility) is True
    assert compatibility_matches(different, current.compatibility) is False


def test_boundary_models_are_frozen_strict_and_path_free(tmp_path: Path) -> None:
    """Disk/HTTP DTOs must reject coercion, extras, and credential fields."""
    evidence = CompatibilityEvidence(
        model_sha256="a" * 64,
        projector_sha256=None,
        runtime_sha256="b" * 64,
        build_info="build",
        state_settings=(("ctx-size", "4096"),),
    )
    record_values = {
        "snapshot_id": "20260904T010203Z-000001",
        "filename": "20260904T010203Z-000001.bin",
        "created_utc": "2026-09-04T01:02:03Z",
        "publication_sequence": 1,
        "source_slot": 0,
        "tokens": 4,
        "bytes": 8,
        "sha256": "c" * 64,
        "model_label": "Local model",
        "compatibility": evidence,
    }

    with pytest.raises(ValidationError):
        SlotReceipt(slot_id=True, filename="x.bin", tokens=1, bytes=1)
    with pytest.raises(ValidationError):
        SnapshotRecord(**record_values, bearer_token="secret")
    with pytest.raises(ValidationError):
        CompatibilityEvidence(
            model_sha256="A" * 64,
            projector_sha256=None,
            runtime_sha256="b" * 64,
            build_info="build",
            state_settings=(("ctx-size", "4096"),),
        )
    with pytest.raises(ValidationError):
        CompatibilityEvidence(
            model_sha256="a" * 64,
            projector_sha256=None,
            runtime_sha256="b" * 64,
            build_info="build",
            state_settings=(("parallel", "1"), ("ctx-size", "4096")),
        )

    record = SnapshotRecord(**record_values)
    with pytest.raises(ValidationError):
        record.tokens = 99
    assert "secret" not in record.model_dump_json()


def test_operation_projection_models_preserve_bounded_typed_state(
    tmp_path: Path,
) -> None:
    """The later store/service tasks receive the exact immutable shared projections."""
    evidence = CompatibilityEvidence(
        model_sha256="a" * 64,
        projector_sha256=None,
        runtime_sha256="b" * 64,
        build_info="build",
        state_settings=(("ctx-size", "4096"),),
    )
    record = SnapshotRecord(
        snapshot_id="20260904T010203Z-000001",
        filename="20260904T010203Z-000001.bin",
        created_utc="2026-09-04T01:02:03Z",
        publication_sequence=1,
        source_slot=0,
        tokens=4,
        bytes=8,
        sha256="c" * 64,
        model_label="Local model",
        compatibility=evidence,
    )
    working = WorkingFile(
        launch_id="launch-1",
        operation_id="operation-1",
        path=tmp_path / "private.bin",
        source_record=record,
    )
    result = SaveResult(record=record, removed_ids=("old",), cleanup_failed_ids=())
    catalog = CatalogPage(
        records=(record,),
        next_offset=None,
        stored_bytes=8,
        residual_bytes=0,
        scan_complete=True,
    )
    view = ManagerView(
        launch_id="launch-1",
        status="idle",
        operation_id=None,
        started_at=None,
        slots=(),
        catalog=catalog,
        disabled_reason=None,
        message=None,
    )
    error = SnapshotError("restore_integrity_failed", submission_possible=False)

    assert working.source_record is record
    assert result.removed_ids == ("old",)
    assert view.catalog.records == (record,)
    assert str(error) == "restore_integrity_failed"
    assert error.code == "restore_integrity_failed"
    assert error.submission_possible is False
    assert str(working.path) not in repr(working)
