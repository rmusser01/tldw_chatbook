"""Pure state-mapping tests for the managed model browser."""

from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDiskUsage,
    ArtifactRef,
    ArtifactRole,
    InstalledArtifact,
    ProvenanceClass,
)

from Tests.Model_Artifacts.test_acquisition_types import make_descriptor


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


def test_variant_guidance_recognizes_a_bounded_quantization_token() -> None:
    """Removing the exact-token parser must lose useful variant guidance."""
    from tldw_chatbook.UI.Screens import model_browser_state

    guidance_factory = getattr(model_browser_state, "variant_guidance", None)
    if guidance_factory is None:
        pytest.fail("variant_guidance is not implemented")

    guidance = guidance_factory(
        "publisher/model.Q4_K_M.gguf",
        total_bytes=4_294_967_296,
        file_count=1,
        source_index=7,
    )

    assert guidance.filename == "publisher/model.Q4_K_M.gguf"
    assert guidance.quantization == "Q4_K_M"
    assert guidance.summary == (
        "4-bit quantization · for the same model, typically smaller than "
        "higher-bit variants, with a greater fidelity trade-off."
    )
    assert guidance.total_bytes == 4_294_967_296
    assert guidance.file_count == 1
    assert guidance.source_index == 7


@pytest.mark.parametrize(
    ("filename", "expected_token", "expected_summary"),
    (
        (
            "model-IQ2_XXS.gguf",
            "IQ2_XXS",
            "2-bit importance-matrix quantization · for the same model, "
            "typically very compact, with a substantial fidelity trade-off.",
        ),
        (
            "model-q5_k_m.gguf",
            "Q5_K_M",
            "5-bit quantization · for the same model, typically a middle "
            "ground between 4-bit size and higher-bit fidelity.",
        ),
        (
            "model.Q8_0.gguf",
            "Q8_0",
            "8-bit quantization · for the same model, typically large, with "
            "a smaller fidelity trade-off than lower-bit variants.",
        ),
        (
            "model-BF16.gguf",
            "BF16",
            "High-precision weights · typically larger than quantized "
            "variants of the same model.",
        ),
    ),
)
def test_variant_guidance_covers_supported_quantization_families(
    filename: str,
    expected_token: str,
    expected_summary: str,
) -> None:
    """Removing a supported family must not silently downgrade it to unknown."""
    from tldw_chatbook.UI.Screens.model_browser_state import variant_guidance

    guidance = variant_guidance(
        filename,
        total_bytes=1,
        file_count=1,
        source_index=0,
    )

    assert guidance.quantization == expected_token
    assert guidance.summary == expected_summary


def test_variant_guidance_recognizes_a_token_at_a_nested_path_basename() -> None:
    """A provider path separator must not hide an exact basename token."""
    from tldw_chatbook.UI.Screens.model_browser_state import variant_guidance

    guidance = variant_guidance(
        "nested/Q4_K_M.gguf",
        total_bytes=1,
        file_count=1,
        source_index=0,
    )

    assert guidance.quantization == "Q4_K_M"


def test_variant_guidance_does_not_merge_conflicting_shard_tokens() -> None:
    """A mixed file set must not be presented as one known quantization."""
    from tldw_chatbook.UI.Screens.model_browser_state import variant_guidance

    guidance = variant_guidance(
        "model-Q4_K_M-00001-of-00002.gguf",
        total_bytes=2,
        file_count=2,
        source_index=0,
        filenames=("model-Q5_K_M-00002-of-00002.gguf",),
    )

    assert guidance.quantization is None
    assert guidance.summary == "No recognized quantization token in the filename."


@pytest.mark.parametrize(
    "filename",
    (
        "model.gguf",
        "model-Q4KM.gguf",
        "model-Q40.gguf",
        "model-Q4_K_MEDIUM.gguf",
        "model-F160.gguf",
    ),
)
def test_variant_guidance_never_guesses_an_unrecognized_filename(
    filename: str,
) -> None:
    """A loose substring match must not fabricate a quantization fact."""
    from tldw_chatbook.UI.Screens.model_browser_state import variant_guidance

    guidance = variant_guidance(
        filename,
        total_bytes=2,
        file_count=1,
        source_index=0,
    )

    assert guidance.quantization is None
    assert guidance.summary == "No recognized quantization token in the filename."


def test_variant_guidance_filter_matches_filename_or_quantization_locally() -> None:
    """Dropping either local match field must hide a valid candidate search hit."""
    from tldw_chatbook.UI.Screens import model_browser_state
    from tldw_chatbook.UI.Screens.model_browser_state import VariantGuidance

    filter_rows = getattr(model_browser_state, "filter_variant_guidance", None)
    if filter_rows is None:
        pytest.fail("filter_variant_guidance is not implemented")
    rows = (
        VariantGuidance("alpha-Q4_K_M.gguf", 40, 1, 0, "Q4_K_M", "four"),
        VariantGuidance("beta-Q8_0.gguf", 80, 1, 1, "Q8_0", "eight"),
        VariantGuidance("experimental.gguf", 60, 2, 2, None, "unknown"),
    )

    assert tuple(row.source_index for row in filter_rows(rows, "q8_0")) == (1,)
    assert tuple(row.source_index for row in filter_rows(rows, "EXPERIMENT")) == (2,)
    assert tuple(row.source_index for row in filter_rows(rows, "  ")) == (0, 1, 2)


def test_variant_guidance_sort_orders_are_deterministic() -> None:
    """Removing any supported order must not silently fall back to source order."""
    from tldw_chatbook.UI.Screens import model_browser_state
    from tldw_chatbook.UI.Screens.model_browser_state import VariantGuidance

    sort_rows = getattr(model_browser_state, "sort_variant_guidance", None)
    if sort_rows is None:
        pytest.fail("sort_variant_guidance is not implemented")
    rows = (
        VariantGuidance("unknown.gguf", 60, 1, 4, None, "unknown"),
        VariantGuidance("q8.gguf", 80, 1, 3, "Q8_0", "eight"),
        VariantGuidance("iq2.gguf", 20, 1, 2, "IQ2_XXS", "two"),
        VariantGuidance("bf16.gguf", 160, 1, 1, "BF16", "precision"),
        VariantGuidance("q4.gguf", 40, 1, 0, "Q4_K_M", "four"),
    )

    assert tuple(row.source_index for row in sort_rows(rows, "source")) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert tuple(row.source_index for row in sort_rows(rows, "size-asc")) == (
        2,
        0,
        4,
        3,
        1,
    )
    assert tuple(row.source_index for row in sort_rows(rows, "size-desc")) == (
        1,
        3,
        4,
        0,
        2,
    )
    assert tuple(row.source_index for row in sort_rows(rows, "quantization")) == (
        2,
        0,
        3,
        1,
        4,
    )


def test_variant_guidance_sort_rejects_an_unknown_order() -> None:
    """A typo must fail closed instead of producing an unexplained ordering."""
    from tldw_chatbook.UI.Screens.model_browser_state import (
        VariantGuidance,
        sort_variant_guidance,
    )

    rows = (VariantGuidance("q4.gguf", 40, 1, 0, "Q4_K_M", "four"),)

    with pytest.raises(ValueError, match="Unsupported variant sort order"):
        sort_variant_guidance(rows, "smallish")


# ---------------------------------------------------------------------------
# format_mib -- the one byte formatter every plan/progress/inventory caller
# shares (TASK-596 delta port). Before this existed, plan_panel.py,
# install_progress.py, and model_installed_view.py each reimplemented MiB
# formatting independently and disagreed: install_progress.py additionally
# switched to KiB/B below 1 MiB, so the same byte count rendered
# differently in the install plan than in the progress display. Every
# expected string below is hand-verified against the formula
# ``size_bytes / (1024 * 1024)`` rounded to one decimal place, not
# re-derived from the implementation being tested.
# ---------------------------------------------------------------------------


def test_format_mib_renders_zero_bytes() -> None:
    from tldw_chatbook.UI.Screens.model_browser_state import format_mib

    assert format_mib(0) == "0.0 MiB"


def test_format_mib_renders_a_sub_mib_value() -> None:
    from tldw_chatbook.UI.Screens.model_browser_state import format_mib

    # 512_000 / (1024*1024) = 0.48828125 -- hand-computed, not re-derived
    # from the division this test checks. This is also the case that used
    # to render as "500.0 KiB" under install_progress.py's old sub-MiB
    # branch instead of "0.5 MiB"; format_mib always renders MiB.
    assert format_mib(512_000) == "0.5 MiB"


def test_format_mib_renders_exactly_one_mib() -> None:
    from tldw_chatbook.UI.Screens.model_browser_state import format_mib

    assert format_mib(1_048_576) == "1.0 MiB"


def test_format_mib_renders_exactly_one_gib() -> None:
    from tldw_chatbook.UI.Screens.model_browser_state import format_mib

    # 1_073_741_824 bytes == exactly 1024 MiB (1 GiB) -- a round number
    # chosen so the expected string is verifiable by hand.
    assert format_mib(1_073_741_824) == "1024.0 MiB"


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
    assert rows[1].reference is None
    assert rows[1].provenance == "Unmanaged — integrity unknown"
    assert rows[1].action_hint == "Outside Chatbook · integrity unknown"
    assert rows[1].installed_store_bytes == 100
    assert rows[1].staging_store_bytes == 25


def test_unmanaged_gguf_row_offers_import_without_managed_reference(
    tmp_path: Path,
) -> None:
    """An outside GGUF remains usable and gains only an optional Import action."""
    from tldw_chatbook.UI.Screens.model_browser_state import (
        UnmanagedRow,
        inventory_rows,
    )

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    usage = ArtifactDiskUsage(0, 0, 0)

    row = inventory_rows(
        (),
        usage,
        (UnmanagedRow(source, source.stat().st_size),),
    )[0]

    assert row.is_unmanaged is True
    assert row.reference is None
    assert row.action_hint == "Outside Chatbook · integrity unknown"


def test_inventory_labels_dependencies_and_never_allows_activation(
    tmp_path: Path,
) -> None:
    """A managed dependency remains readable without a root-only action."""
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows

    descriptor = replace(make_descriptor(), role=ArtifactRole.DEPENDENCY)
    row = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "remote-model",
                descriptor=descriptor,
                ready=True,
                active=True,
                error=None,
            ),
        ),
        None,
        (),
    )[0]

    assert row.activation_allowed is False
    assert row.action_hint == "Managed dependency"


def test_inventory_keeps_repair_copy_ahead_of_dependency_label(
    tmp_path: Path,
) -> None:
    """A broken dependency remains repairable instead of looking installed."""
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows

    descriptor = replace(make_descriptor(), role=ArtifactRole.DEPENDENCY)
    row = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "broken-remote-model",
                descriptor=descriptor,
                ready=True,
                active=True,
                error="manifest mismatch",
            ),
        ),
        None,
        (),
    )[0]

    assert row.activation_allowed is False
    assert row.action_hint == "Needs repair — Repair"


def test_inventory_marks_valid_unready_root_as_activation_required(
    tmp_path: Path,
) -> None:
    """Missing readiness is the action to activate, not a broken root."""
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows

    row = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "installed-root",
                descriptor=make_descriptor(),
                ready=False,
                active=False,
                error=None,
            ),
        ),
        None,
        (),
    )[0]

    assert row.activation_allowed is True
    assert row.ready is False
    assert row.action_hint == "Installed · activation required"


def test_inventory_keeps_assigned_consumer_readiness_copy(tmp_path: Path) -> None:
    """An assigned ready model retains the existing activation state copy.

    This fails if the unassigned policy suppresses activation for any other
    consumer.
    """
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows

    descriptor = make_descriptor()
    row = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "assigned-model",
                descriptor=descriptor,
                ready=True,
                active=False,
                error=None,
            ),
        ),
        None,
        (),
    )[0]

    assert row.activation_allowed is True
    assert row.action_hint == "Ready"


def test_install_failure_messages_are_typed_labeled_and_sanitized() -> None:
    """Raw acquisition details never reach user notifications.

    TASK-596 delta port: every one of install_failure_message's branches
    (there are nine: eight typed exceptions plus the untyped fallback) gets
    its own case here, each asserting the mapped text IS present -- not
    merely that the message is truthy/nonempty, which would pass even if
    two different exception types collapsed onto the same generic string
    -- AND that the injected raw marker is absent, exercised per branch
    rather than once for the whole batch.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionBusyError,
        CatalogError,
        ConsentMismatchError,
        GatedRepositoryError,
        InsufficientSpaceError,
        PreflightNotGrantableError,
        TransferError,
    )
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    marker = "SECRET-GATING-DETAIL-9c31af"

    def _check(exc: BaseException, expected_text: str) -> None:
        message = install_failure_message(exc, model_label="Whisper")
        assert expected_text in message
        assert marker not in message

    _check(
        InsufficientSpaceError(marker),
        "Not enough free disk space for this install.",
    )
    _check(
        GatedRepositoryError(marker),
        "Configure HUGGINGFACE_API_KEY (or HF_TOKEN) and retry.",
    )
    _check(
        AcquisitionBusyError(marker),
        "Another Whisper install is already in progress. Try again shortly.",
    )
    _check(
        ConsentMismatchError(marker),
        "The install plan changed. Retry Install to review the current plan.",
    )
    _check(
        PreflightNotGrantableError(marker),
        "This install plan cannot proceed. Retry Install to review the current plan.",
    )
    _check(
        CatalogError(marker),
        "The Whisper download source is misconfigured.",
    )
    _check(
        TransferError(marker, retryable=True),
        "The download was interrupted. Retry Install to resume.",
    )
    _check(
        TransferError(marker, retryable=False),
        "The download failed and cannot be retried automatically.",
    )
    # The untyped fallback: any exception not one of the eight typed cases
    # above still gets a safe, labeled message rather than leaking str(exc).
    _check(
        RuntimeError(marker),
        "Whisper install failed. See the application log for details.",
    )
