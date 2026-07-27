"""Identity-safe lifecycle primitives for local LLM server subprocesses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import subprocess
import threading
from typing import Any


SERVER_PROCESS_ATTRS = {
    "llamacpp": "llamacpp_server_process",
    "llamafile": "llamafile_server_process",
    "vllm": "vllm_server_process",
    "onnx": "onnx_server_process",
    "mlx": "mlx_server_process",
    "ollama": "ollama_server_process",
}


@dataclass(eq=False)
class ServerLaunchClaim:
    """An identity-bearing reservation for one provider launch generation."""

    provider: str
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _validate_provider(provider: str) -> str:
    try:
        return SERVER_PROCESS_ATTRS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported local server provider: {provider}") from exc


def _lock(app: Any):
    return app._llm_server_lifecycle_lock


def process_is_running(process: Any) -> bool:
    """Return whether a process handle is known to still be live."""

    if process is None:
        return False
    try:
        return process.poll() is None
    except Exception:
        return True


def reserve_server_launch(app: Any, provider: str) -> ServerLaunchClaim | None:
    """Atomically reserve a provider launch unless it is already active."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if app._llm_server_launch_claims.get(provider) is not None:
            return None
        process = getattr(app, process_attr, None)
        if process_is_running(process):
            return None
        if process is not None:
            setattr(app, process_attr, None)
        claim = ServerLaunchClaim(provider)
        app._llm_server_launch_claims[provider] = claim
        return claim


def current_server_claim(app: Any, provider: str) -> ServerLaunchClaim | None:
    """Return the current launch claim for a provider."""

    _validate_provider(provider)
    with _lock(app):
        return app._llm_server_launch_claims.get(provider)


def claim_is_current(app: Any, provider: str, claim: ServerLaunchClaim) -> bool:
    """Return whether ``claim`` is still the provider's current generation."""

    _validate_provider(provider)
    return claim.provider == provider and current_server_claim(app, provider) is claim


def publish_server_process(
    app: Any,
    provider: str,
    claim: ServerLaunchClaim,
    process: Any,
) -> bool:
    """Publish a process only for the current provider generation."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if (
            claim.provider != provider
            or app._llm_server_launch_claims.get(provider) is not claim
            or claim.cancel_event.is_set()
        ):
            return False
        existing = getattr(app, process_attr, None)
        if (
            existing is not None
            and existing is not process
            and process_is_running(existing)
        ):
            return False
        setattr(app, process_attr, process)
        return True


def retain_cancelled_server_process(
    app: Any,
    provider: str,
    claim: ServerLaunchClaim,
    process: Any,
) -> bool:
    """Retain a stubborn cancelled process only for its current generation."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if (
            claim.provider != provider
            or app._llm_server_launch_claims.get(provider) is not claim
        ):
            return False
        existing = getattr(app, process_attr, None)
        if (
            existing is not None
            and existing is not process
            and process_is_running(existing)
        ):
            return False
        setattr(app, process_attr, process)
        return True


def release_server_claim(
    app: Any,
    provider: str,
    claim: ServerLaunchClaim,
) -> bool:
    """Release a current claim that has no published live process."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if app._llm_server_launch_claims.get(provider) is not claim:
            return False
        if process_is_running(getattr(app, process_attr, None)):
            return False
        setattr(app, process_attr, None)
        del app._llm_server_launch_claims[provider]
        return True


def clear_server_process(
    app: Any,
    provider: str,
    claim: ServerLaunchClaim,
    process: Any,
) -> bool:
    """Clear only the current claim's exact process after confirmed exit."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if (
            app._llm_server_launch_claims.get(provider) is not claim
            or getattr(app, process_attr, None) is not process
            or process_is_running(process)
        ):
            return False
        setattr(app, process_attr, None)
        del app._llm_server_launch_claims[provider]
        return True


def clear_unclaimed_process(app: Any, provider: str, process: Any) -> bool:
    """Clear an exact exited legacy handle only when no generation owns it."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        if (
            app._llm_server_launch_claims.get(provider) is not None
            or getattr(app, process_attr, None) is not process
            or process_is_running(process)
        ):
            return False
        setattr(app, process_attr, None)
        return True


def server_is_active(app: Any, provider: str) -> bool:
    """Return whether a launch is reserved or a process is live."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        return app._llm_server_launch_claims.get(
            provider
        ) is not None or process_is_running(getattr(app, process_attr, None))


def server_process(app: Any, provider: str) -> Any:
    """Return the app-owned process handle for a provider."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        return getattr(app, process_attr, None)


def server_lifecycle_snapshot(
    app: Any,
    provider: str,
) -> tuple[ServerLaunchClaim | None, Any]:
    """Return one provider's claim and process from the same lock epoch."""

    process_attr = _validate_provider(provider)
    with _lock(app):
        return (
            app._llm_server_launch_claims.get(provider),
            getattr(app, process_attr, None),
        )


def current_llm_destination(app: Any) -> Any | None:
    """Return the currently mounted Models destination, if it is active."""

    try:
        for screen in reversed(app.screen_stack):
            window = getattr(screen, "llm_window", None)
            if window is not None and window.is_mounted:
                return window
        return None
    except Exception:
        return None


def sync_current_llm_destination(
    app: Any,
    provider: str,
    status: str | None = None,
) -> None:
    """Refresh only the current mounted Models destination."""

    window = current_llm_destination(app)
    if window is not None:
        window._handle_server_process_state_change(provider, status)


async def stop_server_process(
    app: Any,
    provider: str,
    label: str,
) -> bool:
    """Cancel or stop one provider without blocking Textual's event loop."""

    claim, process = server_lifecycle_snapshot(app, provider)
    if claim is not None:
        claim.cancel_event.set()
    if process is None:
        sync_current_llm_destination(app, provider)
        if claim is not None:
            app.notify(
                f"{label} startup cancellation requested.",
                severity="information",
            )
        else:
            app.notify(f"{label} is not running.", severity="warning")
        return False
    pid = getattr(process, "pid", "unknown")
    if process_is_running(process):
        stopped = await asyncio.to_thread(terminate_process_bounded, process)
    else:
        stopped = True
    if stopped and claim is not None:
        clear_server_process(app, provider, claim, process)
        app.notify(f"{label} stopped.", severity="information")
    elif stopped:
        clear_unclaimed_process(app, provider, process)
        app.notify(f"{label} stopped.", severity="information")
    elif not stopped:
        app.notify(
            f"{label} process {pid} did not stop; retry Stop.",
            severity="error",
        )
    sync_current_llm_destination(app, provider)
    return stopped


def terminate_process_bounded(process: Any, timeout: float = 5.0) -> bool:
    """Terminate, then kill and reap if needed; never wait without a bound."""

    if not process_is_running(process):
        return True
    try:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass
    if process_is_running(process):
        try:
            process.kill()
            process.wait(timeout=timeout)
        except Exception:
            pass
    return not process_is_running(process)


def run_server_subprocess(
    app: Any,
    provider: str,
    command: list[str],
    claim: ServerLaunchClaim,
    subprocess_module: Any,
    *,
    cwd: Any = None,
) -> str:
    """Run one claimed server while discarding potentially sensitive output."""

    def notify_state_change(status: str | None = None) -> None:
        try:
            app.call_from_thread(
                sync_current_llm_destination,
                app,
                provider,
                status,
            )
        except Exception:
            pass

    def settle_lifecycle(
        callback: Any,
        *args: Any,
        status: str | None = None,
    ) -> bool:
        """Settle app state and current presentation in one main-thread turn."""

        def settle_on_main_thread() -> bool:
            settled = bool(callback(*args))
            if settled:
                sync_current_llm_destination(app, provider, status)
            return settled

        try:
            return bool(app.call_from_thread(settle_on_main_thread))
        except Exception:
            try:
                return bool(callback(*args))
            except Exception:
                return False

    process = None
    retained = False
    published = False
    final_status = None
    try:
        if claim.cancel_event.is_set() or not claim_is_current(app, provider, claim):
            return f"{provider} launch cancelled"
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "text": True,
        }
        if cwd is not None:
            kwargs["cwd"] = cwd
        process = subprocess_module.Popen(command, **kwargs)
        published = bool(
            app.call_from_thread(
                publish_server_process,
                app,
                provider,
                claim,
                process,
            )
        )
        notify_state_change()
        if not published or claim.cancel_event.is_set():
            if not terminate_process_bounded(process):
                retained = bool(
                    app.call_from_thread(
                        retain_cancelled_server_process,
                        app,
                        provider,
                        claim,
                        process,
                    )
                )
                notify_state_change()
            return f"{provider} launch cancelled"
        return_code = process.wait()
        if return_code and not claim.cancel_event.is_set():
            final_status = f"{provider} server exited (code={return_code})"
        return f"{provider} server exited (code={return_code})"
    except Exception as exc:
        exception_category = type(exc).__name__
        result = f"{provider} server failed (category={exception_category})"
        if not claim.cancel_event.is_set():
            final_status = result
        return result
    finally:
        if process is not None and not published and process_is_running(process):
            if terminate_process_bounded(process):
                retained = False
            else:
                retained = settle_lifecycle(
                    retain_cancelled_server_process,
                    app,
                    provider,
                    claim,
                    process,
                )
        if retained and not process_is_running(process):
            retained = False
        if not retained:
            if process is None or not published:
                settle_lifecycle(
                    release_server_claim,
                    app,
                    provider,
                    claim,
                    status=final_status,
                )
            else:
                settle_lifecycle(
                    clear_server_process,
                    app,
                    provider,
                    claim,
                    process,
                    status=final_status,
                )
