"""Task-owned observer: records errors before Console replaces their detail."""

import threading
from pathlib import Path
from unittest.mock import patch

_REASONS = {
    "Hosted Chat stream JSON is malformed.": "stream_json_malformed",
    "Hosted Chat stream read failed.": "stream_read_failed",
    "Hosted Chat stream ended before a clean terminal record.": "stream_terminal_missing",
    "Hosted Chat stream terminated before required metadata.": "stream_metadata_missing",
}


def install_provider_failure_observer(stack, evidence):
    """Record bounded exception metadata without changing native error handling."""
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Chat import console_provider_gateway as gateway
    from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError

    native_copy = gateway.safe_provider_error_copy
    lock = threading.Lock()
    sequence = 0

    def observed_copy(provider, exc, *, known_credentials=()):
        nonlocal sequence
        try:
            actor = current_run_actor()
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                code = tb.tb_frame.f_code
                frames.append(
                    {
                        "file": Path(code.co_filename).name,
                        "function": code.co_name,
                        "line": tb.tb_lineno,
                    }
                )
                tb = tb.tb_next
            reason = "unclassified"
            if (
                isinstance(exc, HostedChatProtocolError)
                and exc.args
                and type(exc.args[0]) is str
            ):
                reason = _REASONS.get(exc.args[0], reason)
            status = getattr(exc, "status_code", None)
            # No exception text, source lines, locals, prompts, headers or bodies.
            record = {
                "seam": "console_error_before_projection",
                "run_id": actor.run_id if actor else None,
                "actor": actor.kind if actor else None,
                "exception_class": type(exc).__name__,
                "original_status": status if type(status) is int else None,
                "reason_code": reason,
                "frames": frames[-16:],
            }
            with lock:
                sequence += 1
                evidence.write(f"provider-error-{sequence:02d}.json", record)
        except Exception:  # noqa: BLE001, S110 - passive evidence cannot replace native errors
            # An observer failure must never replace the native provider failure.
            pass
        return native_copy(provider, exc, known_credentials=known_credentials)

    stack.enter_context(
        patch.object(gateway, "safe_provider_error_copy", observed_copy)
    )
