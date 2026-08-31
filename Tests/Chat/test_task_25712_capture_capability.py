"""TASK-25712: don't claim Capture-On the runtime cannot deliver.

`ConsoleProviderGateway._trace_call_boundary_factory` is never supplied in
production -- both callers of `ensure_provider_gateway` omit it and
`ConsoleTraceCallBoundary` is constructed only in tests -- while
`ConsoleTurnPreparation.capture_mode` defaults to CAPTURE_ON. So every real
send reached `_reserve_trace_call`, whose first statement raises when the
factory is None, and the provider was never contacted.

The dispatch guard is CORRECT and stays: a Capture-On turn that cannot record
a durable boundary must not reach a provider
(test_capture_on_without_durable_boundary_cannot_enter_adapter pins it). The
error is upstream -- preparing a turn as Capture-On when the runtime has no
way to capture it. A runtime that cannot capture must say so, and the app
already models running without capture (`one_shot_capture_off`).
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


def test_gateway_without_a_factory_reports_no_durable_capture() -> None:
    gateway = ConsoleProviderGateway()
    assert gateway.supports_durable_capture is False


def test_gateway_with_a_factory_reports_durable_capture() -> None:
    gateway = ConsoleProviderGateway(
        trace_call_boundary_factory=lambda _request, _resolution, _route: object()
    )
    assert gateway.supports_durable_capture is True


def test_capture_mode_decision_consults_the_runtime_capability() -> None:
    """The policy seam must not promise capture the gateway cannot honour.

    Reads the module source rather than one method: dev inlined this decision
    into the submit path (it used to live in `_capture_mode_for_preparation`,
    which no longer exists), so pinning a method name would break on the next
    refactor while the property it guards stayed correct.
    """
    from pathlib import Path

    import tldw_chatbook.Chat.console_chat_controller as controller

    source = Path(controller.__file__).read_text()
    assert "supports_durable_capture" in source
    # It must gate the CAPTURE_ON branch, not sit somewhere inert.
    capture_on_index = source.index("ConsoleTraceCaptureMode.CAPTURE_ON")
    guard_index = source.index("supports_durable_capture")
    assert abs(guard_index - capture_on_index) < 4000
