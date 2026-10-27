"""Read-only acquisition plan rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from tldw_chatbook.UI.Screens.model_browser_state import (
    PlanRow,
    display_license,
    format_free_space,
    format_mib,
    format_mib_progress,
    plan_rows,
    plan_totals,
)

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport


def _home_relative(path: Path) -> str:
    """Show a path under the home directory as ``~/...``."""
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(path)


def _store_root(destination: Path) -> Path:
    """The managed store root above ``<store>/artifacts/<id>/<rev>/<variant>``.

    Falls back to the destination itself for any other layout, so the
    summary never names a directory the install does not use.
    """
    parents = destination.parents
    if len(parents) >= 4 and parents[2].name == "artifacts":
        return parents[3]
    return destination


def _component_heading(row: PlanRow, *, is_root: bool, model_label: str) -> str:
    size = format_mib(row.total_bytes)
    if row.role:
        return f"{row.role[:1].upper()}{row.role[1:]} ({size}) — {row.purpose}."
    name = model_label if is_root else row.reference.artifact_id
    return f"{name} ({size})"


def _component_role(row: PlanRow, *, is_root: bool, model_label: str) -> str:
    if row.role:
        return row.role
    return model_label if is_root else row.reference.artifact_id


def plan_summary_text(report: PreflightReport, *, model_label: str) -> str:
    """The decision facts, first: size, place, licences, space, resume.

    TASK-34100.14 ([voice-speech-18]): what a newcomer needs to decide leads
    the review; source, revisions, provenance, staging and hashes are in
    :func:`plan_details_text`. Usage terms stay here -- a licence id such as
    ``other`` is not consent on its own (UAT, PR #2825).

    Args:
        report: Immutable acquisition preflight report.
        model_label: User-visible model name.

    Returns:
        Plain text (rendered with markup off).
    """
    rows = plan_rows(report)
    totals = plan_totals(report)
    lines = [f"Install {model_label}?", ""]
    if totals.download_bytes:
        lines.append(f"Download: {format_mib(totals.download_bytes)}")
    else:
        lines.append("Download: nothing — every file is already here.")
    destination = _home_relative(_store_root(totals.destination))
    lines.append(
        f"Installs to: {destination} ({format_free_space(totals.free_bytes)})"
    )
    if len(rows) == 1:
        lines.append(f"Licence: {display_license(rows[0].license_id)}")
    else:
        lines.append(
            "Licences: "
            + " · ".join(
                f"{display_license(row.license_id)} "
                f"({_component_role(row, is_root=row.reference == report.root, model_label=model_label)})"
                for row in rows
            )
        )
    for row in rows:
        if not row.usage_notice:
            continue
        if len(rows) == 1:
            lines.append(f"Terms: {row.usage_notice}")
        else:
            role = _component_role(
                row, is_root=row.reference == report.root, model_label=model_label
            )
            lines.append(f"Terms ({role}): {row.usage_notice}")
    if totals.already_staged_bytes:
        lines.extend(
            (
                "",
                "A previous download stopped at "
                + format_mib_progress(
                    totals.already_staged_bytes,
                    totals.already_staged_bytes + totals.download_bytes,
                )
                + ". Resume download picks up from there.",
            )
        )
    lines.append("")
    if totals.sufficient_space:
        lines.append("Enough free space is available for this install.")
    else:
        lines.append(
            f"Not enough free space: this install needs "
            f"{format_mib(totals.required_bytes)} free."
        )
    if totals.gating_errors:
        lines.extend(("", *totals.gating_errors))
    lines.extend(
        (
            "",
            "Every file is checked against its recorded size and SHA-256 "
            "before install.",
        )
    )
    return "\n".join(lines)


def plan_details_text(
    report: PreflightReport,
    *,
    model_label: str,
    selected_file_details: tuple[tuple[str, int, str, str], ...] = (),
) -> str:
    """Per-component provenance and the staging figures, for "Details".

    Args:
        report: Immutable acquisition preflight report.
        model_label: User-visible model name.
        selected_file_details: Optional upstream path, byte count, digest,
            and pinned source URL values for only the selected candidate.

    Returns:
        Plain text (rendered with markup off).
    """
    rows = plan_rows(report)
    totals = plan_totals(report)
    lines: list[str] = []
    for row in rows:
        is_root = row.reference == report.root
        installed = " (already installed)" if row.already_installed else ""
        files = "1 file" if row.file_count == 1 else f"{row.file_count} files"
        lines.extend(
            (
                _component_heading(row, is_root=is_root, model_label=model_label),
                f"Source: {row.repository}{installed}",
                f"Revision: {row.revision}",
                f"Licence: {display_license(row.license_id)}",
                *((f"Licence text: {row.license_url}",) if row.license_url else ()),
                f"Precision: {row.precision}",
                f"Contents: {files}, {format_mib(row.total_bytes)}",
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
            f"Destination: {totals.destination}",
            f"Already staged: {format_mib(totals.already_staged_bytes)}",
            f"Staging overhead: {format_mib(totals.staging_overhead_bytes)}",
        )
    )
    return "\n".join(lines)


class ModelPlanPanel(Vertical):
    """Render an immutable managed-model acquisition plan.

    The decision facts lead (:func:`plan_summary_text`); provenance, hashes
    and staging sit in a collapsed "Details" section shared by every host
    (the first-run wizard, Lab ▸ Models, Library).
    """

    BUNDLED_CSS = """
    ModelPlanPanel {
        height: auto;
    }

    ModelPlanPanel .model-plan-details {
        margin-top: 1;
    }
    """

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
        self._summary = plan_summary_text(report, model_label=model_label)
        self._details = plan_details_text(
            report,
            model_label=model_label,
            selected_file_details=selected_file_details,
        )
        super().__init__(id=id, classes="model-plan-panel")

    def compose(self) -> ComposeResult:
        """Compose the summary and the collapsed details."""
        yield Static(self._summary, markup=False, classes="model-plan-summary")
        with Collapsible(title="Details", collapsed=True, classes="model-plan-details"):
            yield Static(
                self._details, markup=False, classes="model-plan-details-text"
            )
