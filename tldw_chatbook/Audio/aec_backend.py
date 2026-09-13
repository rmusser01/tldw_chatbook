"""Lazy optional loader for the narrow native WebRTC AEC3 processor."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol

PROCESSING_SAMPLE_RATE = 48_000
PROCESSING_CHANNELS = 1


class AecProcessor(Protocol):
    """Application-facing subset of the native AEC processor."""

    def analyze_render(self, pcm16: bytes, *, delay_ms: int) -> None: ...

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes: ...

    def reset(self) -> None: ...

    def metrics(self) -> dict[str, float]: ...


def create_aec_processor() -> AecProcessor | None:
    """Create the optional 48 kHz mono native processor, failing closed.

    The companion is imported only here so importing the application or duplex
    contracts never initializes native audio/DSP code. Absence or an unusable
    binary is represented as ``None`` and must select honest half duplex.
    ADR-098 keeps this isolated-child loader independent of the app-side
    ``Utils.optional_deps`` configuration and dependency graph.
    """

    try:
        native = import_module("tldw_voice_aec")
        processor_type = native.AecProcessor
        return processor_type(
            sample_rate=PROCESSING_SAMPLE_RATE,
            channels=PROCESSING_CHANNELS,
        )
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
