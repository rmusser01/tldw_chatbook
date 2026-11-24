"""Check qualified Server Study persistence through the ordinary HTTP client."""

from dataclasses import replace
import json

import httpx
import pytest

from Tests.Research_Workspace.test_server_studio import server_request
from Tests.Research_Workspace.test_studio_models import SERVER
from tldw_chatbook.Research_Workspace.contracts import ResearchCapability
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_generation import (
    GeneratedFlashcard,
    GeneratedQuizQuestion,
    StudioCitation,
    StudioGeneratedOutput,
)
from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_server_study_discovers_saves_replays_and_reads_both_native_kinds():
    ref = replace(SERVER, principal_id="jwt-sub:42")
    address = "https://research-study.invalid"
    endpoint = f"/api/v1/workspaces/{ref.workspace_id}/study-outputs"
    records = {}
    requests = []
    supports_study = False

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/artifacts"):
            return httpx.Response(200, json=[])
        assert path.startswith(endpoint)
        assert request.headers["X-TLDW-Expected-User-ID"] == "42"
        if not supports_study:
            return httpx.Response(404, json={"detail": "Not Found"})
        if request.method == "POST":
            fields = json.loads(request.content)
            requests.append(fields)
            kind = fields["output_kind"]
            record = records.setdefault(
                kind,
                {
                    "workspace_id": ref.workspace_id,
                    "owner_id": 71,
                    "version": 5 if kind == "quiz" else 1,
                    "name": fields["title"],
                    "created_at": "2026-11-24T14:00:00+00:00",
                    "generation": {
                        key: value
                        for key, value in fields.items()
                        if key not in {"title", "items"}
                    },
                },
            )
            return httpx.Response(200, json=record)
        if path != endpoint:
            return httpx.Response(200, json=records[path.split("/")[-2]])
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200,
            json={
                "contract": "research-study.v1",
                "limit": limit,
                "offset": offset,
                "total": len(records),
                "items": list(records.values())[offset : offset + limit],
            },
        )

    client = TLDWAPIClient(address, token="study-fixture")
    async with httpx.AsyncClient(
        base_url=address, transport=httpx.MockTransport(respond)
    ) as http:
        client._client = http
        studio = ServerStudioAdapter(lambda _ref: client)
        ready = ResearchCapability(True, "available", "Chat ready", "server")
        assert not await studio.study_available(ref)
        assert (await studio.list_outputs(ref)).items == ()

        supports_study = True
        caps = studio.capabilities(
            ready, writable=True, study_supported=await studio.study_available(ref)
        )
        assert caps["create_deck"].available and caps["create_quiz"].available
        for kind in ("flashcards", "quiz"):
            base_request = server_request(kind, operation_id=f"op-{kind}")
            source = replace(
                base_request.source_snapshot[0],
                ref=ref,
                catalog_item_id="19",
                association_version=1,
            )
            turn = replace(
                base_request,
                workspace=ref,
                source_snapshot=(source,),
                generation=replace(base_request.generation, title=f"Study {kind}"),
            )
            output = StudioGeneratedOutput(
                "complete",
                cards=(
                    (
                        GeneratedFlashcard(
                            front="Prompt", back="Answer", citations=("S1",)
                        ),
                    )
                    if kind == "flashcards"
                    else ()
                ),
                questions=(
                    (
                        GeneratedQuizQuestion(
                            question="Prompt",
                            options=("One", "Two"),
                            correct_index=1,
                            citations=("S1",),
                        ),
                    )
                    if kind == "quiz"
                    else ()
                ),
                citations=(StudioCitation("S1", source),),
            )
            saved = await studio.save(turn, output)
            assert saved.owner_id == "71"
            assert saved.workspace == ref
            assert saved.source_snapshot == (source,)
            assert requests[-1]["items"][0]["citations"] == [source.source_id]
            restarted = ServerStudioAdapter(lambda _ref: client)
            assert await restarted.save(turn, output) == saved
            assert (await restarted.resolve_output(saved))["name"] == f"Study {kind}"
            records[kind]["workspace_id"] = "elsewhere"
            with pytest.raises(ValueError, match="different workspace"):
                await restarted.resolve_output(saved)
            records[kind]["workspace_id"] = ref.workspace_id

        first = await studio.list_outputs(ref, limit=1)
        second = await studio.list_outputs(ref, limit=1, offset=1)
        assert first.total == second.total == 2
        assert first.has_more and not second.has_more
        assert first.items[0].owner_kind is not second.items[0].owner_kind
        assert len(requests) == 4
