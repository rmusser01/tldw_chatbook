"""Capture native Data Tables inputs and wait for the exact published owner."""

from __future__ import annotations

import asyncio

from tldw_chatbook.tldw_api.workspace_data_tables_client import (
    WorkspaceDataTablesClient,
)

from .source_readiness import versioned_media_sources
from .studio_generation import StudioGeneratedOutput, StudioRetryUnavailable
from .studio_models import ResearchOutputKind, ResearchWorkProductKind

MAX_TABLE_SOURCES = 20
_TABLE_TASKS = {
    ResearchOutputKind.DATA_TABLE: (
        "Extract comparable evidence into a data table.",
        (),
    ),
    ResearchWorkProductKind.LITERATURE_MATRIX: (
        "Build a literature matrix with one row per study. Compare methods and findings and preserve limitations.",
        (
            "Source",
            "Method",
            "Sample",
            "Findings",
            "Limitations",
            "Contradictions",
        ),
    ),
    ResearchWorkProductKind.CORPUS_GAP_FINDER: (
        "Identify gaps in the selected corpus. Distinguish missing coverage from contradictory evidence "
        "and propose a concrete next research step for each gap.",
        (
            "Gap",
            "Type",
            "Evidence",
            "Sources",
            "Importance",
            "Follow-up",
        ),
    ),
}


async def generate_native_table(
    owner: WorkspaceDataTablesClient,
    request,
    cancel_event: asyncio.Event,
) -> StudioGeneratedOutput:
    if cancel_event.is_set():
        return StudioGeneratedOutput("cancelled")
    options, route = request.generation, request.processing_route
    task, default_columns = _TABLE_TASKS[request.output_kind]
    if default_columns:
        task += " In the source column cite the source number and title for each material finding."
    if options.focus:
        task += f" Focus on {options.focus}."
    payload = {
        "operation_id": request.operation_id,
        "provider": route.provider,
        "model": route.model,
        "name": options.title or request.output_kind.value.replace("_", " ").title(),
        "prompt": task,
        "sources": versioned_media_sources(
            request.source_snapshot, maximum=MAX_TABLE_SOURCES
        ),
        "column_hints": [
            {"name": name} for name in options.table_columns or default_columns
        ],
        "max_rows": options.table_max_rows,
    }
    receipt = await owner.launch(payload)
    status, publication = receipt.status, receipt.publication_status
    version = receipt.published_version
    for _ in range(300):
        stopping = cancel_event.is_set()
        if stopping and status not in {
            "cancelled",
            "completed",
            "failed",
            "dead_letter",
            "quarantined",
        }:
            status = (await owner.cancel(receipt)).status
        if status == "cancelled":
            raise StudioRetryUnavailable(
                "The native Data Tables job was cancelled. Start a new generation."
            )
        if stopping:
            return StudioGeneratedOutput("cancelled")
        if status in {"failed", "dead_letter", "quarantined"}:
            raise ValueError(
                "The native Data Tables job failed. Start a new generation after checking Server Jobs."
            )
        if status == "completed" and publication == "ready":
            if type(version) is not int or version < 1:
                raise ValueError(
                    "The Server did not confirm the published native Data Table revision"
                )
            detail = await owner.detail(receipt.table.uuid)
            if (
                not detail
                or detail.table.version != version
                or detail.table.status != "ready"
            ):
                raise ValueError(
                    "The published Data Table changed before Studio could link it"
                )
            return StudioGeneratedOutput(
                "complete", native_owner=(receipt.table.uuid, version)
            )
        if status == "completed" and publication in {"conflict", "unavailable"}:
            raise ValueError(
                "The Server could not publish the captured Data Table. Check the native job."
            )
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=1)
            continue
        except TimeoutError:
            state = await owner.job(receipt)
        status = state.status
        result = state.result or {}
        publication = result.get("publication_status")
        version = result.get("table_version")
    raise ValueError("The Data Table job is still running. Retry to reconnect to it.")
