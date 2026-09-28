"""Pure launch validation and privacy-safe llama.cpp launch summaries."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field

from tldw_chatbook.LLM_Management.llamacpp_connection import local_launch_url
from tldw_chatbook.LLM_Management.llamacpp_profiles import (
    LlamaCppTuning,
    build_tuning_arguments,
)

INVALID_TUNING_MESSAGE = (
    "Invalid tuning or conflicting expert arguments. Use one value per "
    "tuning option; model, alias, host and port belong to the launch controls."
)
_INVALID_ENDPOINT_MESSAGE = (
    "Use localhost or a numeric IP address and a port between 1 and 65535."
)
_INVALID_QUOTING_MESSAGE = (
    "Expert arguments have incomplete quoting. Close each quote and try again."
)
_SNAPSHOT_CONFLICT_MESSAGE = (
    "Snapshot launch options belong to Chatbook. Remove slot options from expert "
    "arguments and the launch environment, or disable prompt-cache snapshots."
)
_EXPERT_LIMIT_MESSAGE = "Expert arguments are too long. Shorten them and try again."
_MAX_EXPERT_TEXT_LENGTH = 16384
_MAX_TYPED_ARGS_LENGTH = 1024
_SNAPSHOT_LINE_PREFIX = "Snapshot launch options: "


class LlamaCppLaunchValidationError(ValueError):
    """A bounded recovery message with no original input or runtime payload."""


@dataclass(frozen=True, slots=True)
class LlamaCppLaunchOptions:
    """Validated process arguments paired with an independently safe projection."""

    host: str
    port: str
    additional_args: tuple[str, ...] = field(repr=False)
    preview: str


_GGUF_PRIMARY_SOURCE_ARGUMENTS = frozenset(
    {
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-dr",
        "--docker-repo",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
        "--models-dir",
        "--models-preset",
        "--embd-gemma-default",
        "--fim-qwen-1.5b-default",
        "--fim-qwen-3b-default",
        "--fim-qwen-7b-default",
        "--fim-qwen-7b-spec",
        "--fim-qwen-14b-spec",
        "--fim-qwen-30b-default",
        "--gpt-oss-20b-default",
        "--gpt-oss-120b-default",
        "--vision-gemma-4b-default",
        "--vision-gemma-12b-default",
    }
)


def validate_gguf_additional_args(arguments: tuple[str, ...]) -> None:
    """Reject primary source selectors while preserving accepted arguments exactly."""
    if any(
        argument.partition("=")[0] in _GGUF_PRIMARY_SOURCE_ARGUMENTS
        for argument in arguments
    ):
        raise ValueError("additional arguments cannot select a model source")


def prepare_launch_options(
    host: str,
    port: str,
    expert_text: str,
    tuning: LlamaCppTuning,
    *,
    snapshots_enabled: bool,
    environment: Mapping[str, str],
) -> LlamaCppLaunchOptions:
    """Validate shared Preview/Start values without any external side effects.

    Args:
        host: Explicit numeric bind address or localhost; blank uses loopback.
        port: Explicit port; blank uses 8080.
        expert_text: Session-local, quote-aware expert arguments.
        tuning: Validated structured tuning; omitted values use runtime defaults.
        snapshots_enabled: Whether the snapshot owner reserves slot options.
        environment: Launch environment used only for snapshot ownership checks.

    Returns:
        Validated argv values and a redacted, non-executable display summary.

    Raises:
        LlamaCppLaunchValidationError: A fixed recovery message for invalid input.
    """
    try:
        host = host.strip() or "127.0.0.1"
        port = port.strip() or "8080"
        # Scoped IPv6 identifiers can contain arbitrary user text. They are not
        # safe typed display values or supported snapshot bind addresses.
        if len(host) > 45 or "%" in host or len(port) > 16:
            raise ValueError
        local_launch_url(host, port)
        port = str(int(port))
    except (AttributeError, TypeError, ValueError):
        raise LlamaCppLaunchValidationError(_INVALID_ENDPOINT_MESSAGE) from None
    if type(expert_text) is not str or len(expert_text) > _MAX_EXPERT_TEXT_LENGTH:
        raise LlamaCppLaunchValidationError(_EXPERT_LIMIT_MESSAGE)
    try:
        raw_args = tuple(shlex.split(expert_text))
    except ValueError:
        raise LlamaCppLaunchValidationError(_INVALID_QUOTING_MESSAGE) from None
    try:
        validate_gguf_additional_args(raw_args)
        additional_args = build_tuning_arguments(tuning, raw_args)
        typed_args = build_tuning_arguments(tuning, ())
        if sum(map(len, typed_args)) > _MAX_TYPED_ARGS_LENGTH:
            raise ValueError
    except (TypeError, ValueError):
        raise LlamaCppLaunchValidationError(INVALID_TUNING_MESSAGE) from None
    if snapshots_enabled:
        from tldw_chatbook.LLM_Management.snapshot_admission import (
            has_owned_slot_options,
        )

        if has_owned_slot_options(("<executable>", *additional_args), environment):
            raise LlamaCppLaunchValidationError(_SNAPSHOT_CONFLICT_MESSAGE)
    command = (
        f"<executable> --model <model> --alias chatbook-llamacpp "
        f"--host {host} --port {port}"
    )
    if typed_args:
        command += " " + " ".join(typed_args)
    snapshot_options = (
        "conditional on Start admission: --slots --slot-save-path <private directory>."
        if snapshots_enabled
        else "disabled."
    )
    preview = "\n".join(
        (
            "Launch settings (redacted; not a shell command)",
            command,
            "Omitted tuning flags use runtime defaults.",
            "Expert options: <hidden>." if raw_args else "Expert options: none.",
            _SNAPSHOT_LINE_PREFIX + snapshot_options,
            "Executable and source files are checked at Start.",
        )
    )
    return LlamaCppLaunchOptions(host, port, additional_args, preview)


def finalize_launch_preview(preview: str, *, snapshots_active: bool) -> str:
    """Freeze actual snapshot flag admission without exposing paths or errors."""
    snapshot_options = (
        "added: --slots --slot-save-path <private directory>; "
        "snapshot readiness still requires verification."
        if snapshots_active
        else "disabled for this launch."
    )
    return "\n".join(
        _SNAPSHOT_LINE_PREFIX + snapshot_options
        if line.startswith(_SNAPSHOT_LINE_PREFIX)
        else line
        for line in preview.splitlines()
    )
