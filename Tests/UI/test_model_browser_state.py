"""Pure state-mapping tests for the managed model browser."""

from pathlib import Path

from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDiskUsage,
    ArtifactRef,
    InstalledArtifact,
    ProvenanceClass,
)


def _report(destination: Path) -> PreflightReport:
    reference = ArtifactRef("parakeet-v2", "rev1", "int8")
    return PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=reference,
                source_url="https://example.test/model",
                repository="publisher/parakeet-v2",
                revision="immutable-revision",
                license_id="CC-BY-4.0",
                license_url="https://example.test/license",
                precision="int8",
                total_bytes=1234,
                file_count=4,
                already_installed=False,
                provenance=(ProvenanceClass.CHATBOOK_CURATED,),
            ),
        ),
        download_bytes=1234,
        already_staged_bytes=100,
        staging_overhead_bytes=256,
        retained_bytes=0,
        destination=destination,
        free_bytes=10_000,
        required_bytes=1490,
        sufficient_space=True,
        gating_errors=(),
    )


def test_plan_rows_and_totals_preserve_every_consent_field(tmp_path: Path) -> None:
    """The render model contains every field required for informed consent."""
    from tldw_chatbook.UI.Screens.model_browser_state import plan_rows, plan_totals

    report = _report(tmp_path / "models")
    row = plan_rows(report)[0]
    totals = plan_totals(report)

    assert row.repository == "publisher/parakeet-v2"
    assert row.revision == "immutable-revision"
    assert row.license_id == "CC-BY-4.0"
    assert row.license_url == "https://example.test/license"
    assert row.precision == "int8"
    assert row.file_count == 4
    assert row.total_bytes == 1234
    assert row.provenance == "Curated by Chatbook"
    assert totals.download_bytes == 1234
    assert totals.already_staged_bytes == 100
    assert totals.staging_overhead_bytes == 256
    assert totals.destination == tmp_path / "models"
    assert totals.free_bytes == 10_000
    assert totals.required_bytes == 1490
    assert totals.sufficient_space is True


def test_provenance_labels_are_precise_and_never_imply_safety() -> None:
    """Trust copy states evidence without claiming malware safety."""
    from tldw_chatbook.UI.Screens.model_browser_state import provenance_label

    labels = (
        provenance_label((ProvenanceClass.CHATBOOK_CURATED,)),
        provenance_label((ProvenanceClass.INTEGRITY_VERIFIED,)),
        provenance_label((ProvenanceClass.LOCAL_INTEGRITY_RECORDED,)),
    )
    assert len(set(labels)) == 3
    for label in labels:
        assert "safe" not in label.lower()
        assert "malware" not in label.lower()
        assert "trusted" not in label.lower()


def test_inventory_keeps_broken_and_unmanaged_rows_visible(tmp_path: Path) -> None:
    """Corrupt manifests and legacy files remain visible and actionable."""
    from tldw_chatbook.UI.Screens.model_browser_state import (
        UnmanagedRow,
        inventory_rows,
    )

    broken = InstalledArtifact(
        path=tmp_path / "broken-model",
        descriptor=None,
        ready=False,
        active=False,
        error="readiness: unreadable",
    )
    usage = ArtifactDiskUsage(installed_bytes=100, staging_bytes=25, free_bytes=500)
    rows = inventory_rows(
        (broken,),
        usage,
        (UnmanagedRow(path=tmp_path / "legacy.gguf", size_bytes=40),),
    )

    assert rows[0].is_broken is True
    assert "Repair" in rows[0].action_hint
    assert rows[1].is_unmanaged is True
    assert rows[1].provenance == "Unmanaged — integrity unknown"
    assert rows[1].installed_store_bytes == 100
    assert rows[1].staging_store_bytes == 25


def test_install_failure_messages_are_typed_labeled_and_sanitized() -> None:
    """Raw acquisition details never reach user notifications."""
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionBusyError,
        CatalogError,
        InsufficientSpaceError,
        PreflightNotGrantableError,
        TransferError,
    )
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    marker = "SECRET-GATING-DETAIL"
    errors = (
        AcquisitionBusyError(marker),
        CatalogError(marker),
        InsufficientSpaceError(marker),
        PreflightNotGrantableError(marker),
        TransferError(marker, retryable=True),
    )
    messages = tuple(
        install_failure_message(error, model_label="Whisper") for error in errors
    )

    assert all(marker not in message for message in messages)
    assert all(message for message in messages)
    assert "Whisper" in messages[0]
    assert "Whisper" in messages[1]
