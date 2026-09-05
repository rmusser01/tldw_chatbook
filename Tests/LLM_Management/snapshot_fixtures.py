"""Real admission/publication fixtures; only the llama-server writer is bypassed."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management.snapshot_admission import (
    finalize_launch,
    prepare_launch,
)
from tldw_chatbook.LLM_Management.snapshot_models import (
    CompatibilityEvidence,
    ReadinessObservation,
    SlotObservation,
    SlotReceipt,
    SnapshotRecord,
)

if TYPE_CHECKING:
    from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore


def test_evidence() -> CompatibilityEvidence:
    """Parse a pinned minimal CPU launch with every canonical effective setting."""
    with TemporaryDirectory() as directory:
        runtime = Path(directory) / "llama-server"
        model = Path(directory) / "model.gguf"
        runtime.write_bytes(b"pinned runtime")
        model.write_bytes(b"model bytes")
        prepared = prepare_launch(
            (
                str(runtime),
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
            ),
            {},
            ServerLaunchClaim(provider="llamacpp", authority="External GGUF"),
            "test-launch-a",
        )
        finalized = finalize_launch(
            prepared,
            ReadinessObservation(
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
                runtime_values=(),
            ),
        )
        assert finalized.compatibility is not None, finalized.disabled_reason
        return finalized.compatibility


def commit_test_snapshot(
    store: SnapshotStore, *, payload: bytes, slot_id: int
) -> SnapshotRecord:
    """Run the real reserve/receipt/commit path around supplied server bytes."""
    working = store.reserve_save("test-launch-a", slot_id)
    working.path.write_bytes(payload)
    receipt = SlotReceipt(
        slot_id=slot_id, filename=working.path.name, tokens=7, bytes=len(payload)
    )
    return store.commit_save(working, receipt, test_evidence(), "Test model", 10).record
