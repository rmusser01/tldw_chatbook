"""Read-only acquisition plan rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

from tldw_chatbook.UI.Screens.model_browser_state import plan_rows, plan_totals

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport


def _mib(size_bytes: int) -> str:
    """Format a byte count as mebibytes."""
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


class ModelPlanPanel(Static):
    """Render an immutable managed-model acquisition plan."""

    def __init__(
        self,
        report: PreflightReport,
        *,
        model_label: str,
        selected_file_details: tuple[tuple[str, int, str, str], ...] = (),
        id: str | None = None,
    ) -> None:
        """Build the panel from preflight state only.

        Args:
            report: Immutable acquisition preflight report.
            model_label: User-visible model name.
            selected_file_details: Optional upstream path, byte count, digest,
                and pinned source URL values for only the selected candidate.
            id: Optional Textual widget id.
        """
        rows = plan_rows(report)
        totals = plan_totals(report)
        lines = [f"Install {model_label}?", ""]
        for row in rows:
            installed = " (already installed)" if row.already_installed else ""
            license_label = (
                "Unknown / not declared"
                if row.license_id == "NOASSERTION"
                else row.license_id
            )
            lines.extend(
                (
                    f"Source: {row.repository}{installed}",
                    f"Revision: {row.revision}",
                    f"License: {license_label}",
                    f"Source review page: {row.license_url}",
                    f"Precision: {row.precision}",
                    f"Contents: {row.file_count} files, {_mib(row.total_bytes)}",
                    f"Provenance: {row.provenance}",
                    "",
                )
            )
        if selected_file_details:
            lines.append("Selected upstream files:")
            for path, size_bytes, sha256, source_url in selected_file_details:
                lines.extend(
                    (
                        f"Path: {path}",
                        f"Bytes: {size_bytes}",
                        f"SHA-256: {sha256}",
                        f"Pinned source URL: {source_url}",
                        "",
                    )
                )
        lines.extend(
            (
                f"Download: {_mib(totals.download_bytes)}",
                f"Already staged: {_mib(totals.already_staged_bytes)}",
                f"Staging overhead: {_mib(totals.staging_overhead_bytes)}",
                f"Destination: {totals.destination}",
                f"Free space: {_mib(totals.free_bytes)}",
            )
        )
        if totals.sufficient_space:
            lines.append("Enough free space is available for this install.")
        else:
            lines.append(
                f"Not enough free space: this install needs "
                f"{_mib(totals.required_bytes)} free."
            )
        if totals.gating_errors:
            lines.extend(("", *totals.gating_errors))
        lines.extend(
            (
                "",
                "Every declared file is checked against pinned sizes and "
                "SHA-256 digests before installation completes.",
            )
        )
        super().__init__(
            "\n".join(lines),
            markup=False,
            id=id,
            classes="model-plan-panel",
        )
