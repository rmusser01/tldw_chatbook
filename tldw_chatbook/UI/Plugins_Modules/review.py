"""Nonsecret in-session drafts; original owner reviews remain opaque objects."""

from dataclasses import dataclass

from .details import display_text


@dataclass
class PluginReviewDraft:
    workspace_id: str | None
    installation_id: str | None = None
    producer: object | None = None
    selection: tuple[str, ...] | None = None
    dialect: str | None = None
    review: object | None = None
    action: str | None = None
    focus_id: str = "plugin-review-close"
    configuration: str = ""
    test_data: str = "{}"
    job_id: str | None = None
    root_impact: tuple[dict, ...] = ()
    root_before_refs: tuple[object, ...] = ()


@dataclass(frozen=True, eq=False)
class PluginConnectionDraft:
    """Reviewed display inputs; the existing MCP owner validates the actual save."""

    token: str
    installation_id: str
    workspace_id: str | None
    component_id: str
    inspection: object
    data_root: object | None = None
    credential_reference: str | None = None
    credential_generation: int | None = None
    session_isolation: str = "separate"
    protocol_version: str = "2026-07-28"


def review_text(review):
    inspection = review.inspection
    lines = [
        f"Installation: {review.installation_id}",
        f"Source: {inspection.source_identity}",
        f"Revision: {inspection.effective_digest}",
        f"Workspace: {review.workspace_id or 'global default'}",
        f"Interpretation: {inspection.dialect}",
        f"Action: {review.kind} · intent: {review.intent or 'unchanged / disabled install'}",
        "Commands and stdio servers run with your host user privileges.",
        "Trust, setup and normal per-call permissions remain separate gates.",
    ]
    if review.kind == "configure":
        lines.append("Exact reviewed mappings: " + review.mappings_json)
    if review.kind in {"activate", "revoke", "uninstall"}:
        import json

        state = json.loads(review.authority_json)
        lines.append(
            "Reviewed workspace overrides: "
            + str(
                [
                    row
                    for row in state["activation"]
                    if row["installation_id"] == review.installation_id
                ]
            )
        )
        lines.append("Saved roots are retained; cleanup requires its own review.")
    for key, component in inspection.inventory.items():
        lines.append(
            f"{'Selected' if key in review.selection else 'Excluded'}: {key} · {component.support}"
        )
        lines.append(f"Dependencies: {', '.join(component.dependencies) or 'none'}")
        lines.append(
            f"Blockers: {', '.join(component.activation_blockers) or 'none recorded'}"
        )
        lines.append(component.definition_json)
    return display_text("\n".join(lines), limit=65536)


async def open_owned_plugin(
    app, installation_id: str, component_id: str | None = None
) -> None:
    """Capture current workspace and open the exact typed owner, never parse labels."""
    import asyncio

    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    registry = app.workspace_registry_service
    workspace = (
        await asyncio.to_thread(registry.get_active_workspace) if registry else None
    )
    await app.push_screen(
        PluginReviewModal(
            app.plugin_source_service,
            installation_id=installation_id,
            workspace_id=workspace.workspace_id if workspace else None,
            component_id=component_id,
        )
    )
