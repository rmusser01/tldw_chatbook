"""Privacy contracts for Evals screen worker diagnostics."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens import evals_screen as evals_screen_module
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen


class _DiagnosticRecorder:
    """Capture diagnostic API use without installing a persistent sink."""

    def __init__(self) -> None:
        self.opt_calls: list[dict] = []
        self.warnings: list[str] = []

    def opt(self, **kwargs):
        self.opt_calls.append(dict(kwargs))
        return self

    def warning(self, message, *args) -> None:
        self.warnings.append(str(message).format(*args))


def _bare_screen(notifications: list[tuple[str, dict]]) -> EvalsScreen:
    """Build only the receiver state required by the production workers."""
    screen = object.__new__(EvalsScreen)
    screen.app_instance = SimpleNamespace(
        notify=lambda message, **kwargs: notifications.append((str(message), kwargs))
    )
    screen._current_app_config = lambda: {"private": "APP_CONFIG_SECRET"}
    screen._sample_bench_client_factory = None
    screen._view_model = object()
    screen._set_sample_bench_running_ui = lambda: None
    screen._reset_sample_bench_running_ui = lambda: None
    screen._set_bench_run_running_ui = lambda: None
    screen._reset_bench_run_running_ui = lambda: None
    screen._on_sample_bench_progress = lambda _done, _total: None
    screen._on_bench_run_progress = lambda _done, _total: None
    screen._sample_bench_running = False
    screen._sample_bench_cancel_token = None
    screen._bench_run_running = False
    screen._bench_run_cancel_token = None
    screen._bench_run_task_id = "PRIVATE_DATASET_NAME"
    return screen


@pytest.mark.asyncio
async def test_sample_bench_failure_diagnostic_omits_traceback_and_user_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "PRIVATE_DATASET_NAME"
    notifications: list[tuple[str, dict]] = []
    recorder = _DiagnosticRecorder()
    screen = _bare_screen(notifications)

    async def fail_sample(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(evals_screen_module, "logger", recorder)
    monkeypatch.setattr(
        evals_screen_module.sample_bench,
        "create_and_run_sample_bench",
        fail_sample,
    )

    await EvalsScreen._create_sample_bench_worker(screen)

    assert recorder.opt_calls == []
    assert recorder.warnings == [
        "Sample bench creation failed (exception_category=RuntimeError)."
    ]
    assert all(secret not in warning for warning in recorder.warnings)
    assert notifications == [
        (
            f"Could not create the sample bench: {secret}",
            {"severity": "error", "markup": False},
        )
    ]


@pytest.mark.asyncio
async def test_existing_bench_failure_diagnostic_omits_traceback_and_user_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "PRIVATE_DATASET_NAME"
    notifications: list[tuple[str, dict]] = []
    recorder = _DiagnosticRecorder()
    screen = _bare_screen(notifications)

    async def fail_bench(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(evals_screen_module, "logger", recorder)
    monkeypatch.setattr(
        evals_screen_module.sample_bench,
        "run_existing_bench",
        fail_bench,
    )

    await EvalsScreen._run_bench_worker(screen)

    assert recorder.opt_calls == []
    assert recorder.warnings == ["Bench run failed (exception_category=RuntimeError)."]
    assert all(secret not in warning for warning in recorder.warnings)
    assert notifications == [
        (
            f"Could not run the bench: {secret}",
            {"severity": "error", "markup": False},
        )
    ]
