"""Use the Server's structured Slides generator and native presentation owner."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from tldw_chatbook.tldw_api.exceptions import (
    APIConnectionError,
    APIResponseError,
    TLDWAPIError,
)
from tldw_chatbook.tldw_api.slides_schemas import GenerateFromMediaRequest

from .source_readiness import versioned_media_sources
from .studio_generation import StudioGeneratedOutput

SLIDES_UNCONFIRMED = (
    "Studio could not confirm whether Server Slides saved the presentation. "
    "Check the native Slides library before starting a new generation."
)


class SlidesSubmissionUnconfirmed(ValueError):
    """The structured Slides API could have committed without returning a receipt."""


async def slides_available(client, route, sources) -> bool:
    if (
        not callable(getattr(client, "get_slides_health", None))
        or not route.provider
        or route.model.casefold() in {"", "auto"}
    ):
        return False
    try:
        versioned_media_sources(sources, maximum=1)
    except ValueError:
        return False
    try:
        health, modes, provider = await asyncio.gather(
            client.get_slides_health(),
            client._request("GET", "/api/v1/slides/capabilities"),
            client.get_llm_provider(route.provider),
        )
    except (APIConnectionError, APIResponseError, ValidationError):
        return False
    structured = (
        (modes.get("generation_modes") or {}).get("structured_slides")
        if isinstance(modes, dict)
        else None
    )
    return (
        health.status == "ok"
        and isinstance(structured, dict)
        and modes.get("schema_version") == 1
        and structured.get("enabled") is True
        and structured.get("transport") == "existing_source_endpoints"
        and provider.name == route.provider
        and provider.is_configured
        and provider.availability == "enabled"
        and route.model in provider.models
    )


async def _check_source(client, workspace_id: str, expected) -> None:
    associations, state = await asyncio.gather(
        client.list_workspace_sources(workspace_id),
        client.get_workspace_source_status(workspace_id),
    )
    if (
        [row["id"] for row in associations if row.get("selected") is True]
        != [expected["source_id"]]
        or not any(
            (row["id"], row["media_id"], row["version"])
            == (
                expected["source_id"],
                expected["catalog_item_id"],
                expected["association_version"],
            )
            for row in associations
        )
        or not any(
            (row["id"], row["media_id"], row.get("catalog_item_version"))
            == (
                expected["source_id"],
                expected["catalog_item_id"],
                expected["catalog_item_version"],
            )
            for row in state["sources"]
        )
    ):
        raise ValueError(
            "The selected Slides source changed. Refresh its Workspace and Media revisions."
        )


async def generate_native_slides(
    client, request, cancel_event
) -> StudioGeneratedOutput:
    if cancel_event.is_set():
        return StudioGeneratedOutput("cancelled")
    [source] = versioned_media_sources(request.source_snapshot, maximum=1)
    await _check_source(client, request.workspace.workspace_id, source)
    if cancel_event.is_set():
        return StudioGeneratedOutput("cancelled")
    style_id, scope = request.generation.slides_visual_style or (None, None)
    route = request.processing_route
    payload = GenerateFromMediaRequest(
        media_id=source["catalog_item_id"],
        title_hint=request.generation.title or None,
        visual_style_id=style_id,
        visual_style_scope=scope,
        provider=route.provider,
        model=route.model,
    )
    generation = asyncio.create_task(client.generate_presentation_from_media(payload))
    stop = asyncio.create_task(cancel_event.wait())
    try:
        await asyncio.wait((generation, stop), return_when=asyncio.FIRST_COMPLETED)
        if not generation.done():
            raise SlidesSubmissionUnconfirmed(SLIDES_UNCONFIRMED)
        response = generation.result()
    except (APIConnectionError, ValidationError) as exc:
        raise SlidesSubmissionUnconfirmed(SLIDES_UNCONFIRMED) from exc
    except APIResponseError as exc:
        if exc.status_code >= 500:
            raise SlidesSubmissionUnconfirmed(SLIDES_UNCONFIRMED) from exc
        raise
    finally:
        generation.cancel()
        stop.cancel()
        await asyncio.gather(generation, stop, return_exceptions=True)
    try:
        await _check_source(client, request.workspace.workspace_id, source)
    except (TLDWAPIError, ValueError, KeyError) as exc:
        raise SlidesSubmissionUnconfirmed(SLIDES_UNCONFIRMED) from exc
    if (
        response.deleted
        or response.version < 1
        or not response.slides
        or response.source_type != "media"
        or str(response.source_ref) != str(source["catalog_item_id"])
        or (
            style_id
            and (response.visual_style_id, response.visual_style_scope)
            != (style_id, scope)
        )
    ):
        raise SlidesSubmissionUnconfirmed(SLIDES_UNCONFIRMED)
    return StudioGeneratedOutput(
        "complete", native_owner=(response.id, response.version)
    )
