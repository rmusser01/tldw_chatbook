"""Literal, bounded untrusted plugin descriptions; no Markdown/image execution."""

import re
import unicodedata

_TERMINAL = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|$)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]"
)


def display_text(value: object, *, limit: int = 2000) -> str:
    text = _TERMINAL.sub("", str(value))
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"}
    )
    return text if len(text) <= limit else text[: limit - 12] + " [truncated]"


def installed_details(row) -> str:
    activation = "Enabled" if row.effective_active else "Disabled"
    selected = set(row.selected_components)
    ready = set(row.ready_components)
    lines = [
        display_text(row.alias, limit=256),
        "Installation on this device: Installed",
        f"Activation: {activation} · {row.override} (global default: {'enabled' if row.activation_default else 'disabled'})",
        f"Trust: {'reviewed' if row.trusted else 'needs review'}",
        f"{len(ready)} of {len(selected)} selected components eligible by recorded metadata",
        f"Installation ID: {row.installation_id}",
        f"Revision: {row.revision_digest}",
        f"Format: {row.dialect or 'unknown'}",
        f"Source ID: {row.source_id or 'direct selected package'}",
        "Origin: " + display_text(row.origin),
        f"Ref: {display_text(row.ref or 'local selected bytes')}\nCommit: {row.commit or 'not recorded'}",
        "\nComponents (recorded metadata; package bytes not reverified):",
    ]
    for item in row.components:
        state = (
            "eligible (recorded)"
            if item.component_id in ready
            else (
                "excluded"
                if item.component_id not in selected
                else (
                    "disabled"
                    if not row.effective_active
                    else (
                        "needs trust"
                        if not row.trusted
                        else "needs configuration / blocked"
                    )
                )
            )
        )
        evidence = (
            ", ".join(f"{entry.level} {entry.observed_at}" for entry in item.evidence)
            or "not authenticated; execution verifies retained bytes"
        )
        lines.extend(
            (
                f"{display_text(item.component_id, limit=256)} · {item.support} · {state}",
                "  Evidence: " + display_text(evidence),
            )
        )
        if item.dependencies:
            lines.append("  Requires: " + display_text(", ".join(item.dependencies)))
        if item.blockers:
            lines.append(
                "  Inspection constraints: " + display_text(", ".join(item.blockers))
            )
    return "\n".join(lines)


def source_details(row) -> str:
    locator = row.locator
    return "\n".join(
        (
            "Marketplace source · untrusted display metadata",
            f"Source ID: {row.source_id}",
            "Origin: " + display_text(locator.url if locator else "unavailable"),
            "Ref: " + display_text(locator.ref if locator else "unknown"),
            "Catalog: "
            + display_text(row.catalog_path or "select on refresh if ambiguous"),
            "Last success: " + display_text(row.fetched_at or "never"),
            "Cache: "
            + (
                "last-good (stale)"
                if row.cached and row.refresh_error
                else "available" if row.cached else "not loaded"
            ),
            "Status: " + display_text(row.refresh_error or "ready"),
            "Refresh is explicit. Discovery does not install or activate plugins.",
        )
    )


def entry_details(row) -> str:
    return "\n".join(
        (
            display_text(row.name, limit=256),
            "Discovery entry · compatibility unknown until inspection",
            f"Source ID: {row.source_id}\nEntry ID: {row.identity}",
            f"Catalog support: {row.support}",
            "Package: "
            + display_text(row.local_path or row.package_locator or "unavailable"),
            f"Entry digest: {row.digest}",
            "Diagnostics: "
            + display_text(", ".join(item.code for item in row.diagnostics) or "none"),
            "Preview acquires selected source bytes. It does not install, trust or enable them.",
        )
    )
