import asyncio
from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_studio_models import LOCAL, ROUTE, SECOND, SOURCE
from tldw_chatbook.Research_Workspace.egress_consent import ResearchEgressConsentService
from tldw_chatbook.Research_Workspace.studio_generation import (
    ResearchStudioGenerator,
    StudioEvidence,
)
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchStudioRequest,
    WorkspaceOutputRef,
)
from tldw_chatbook.Research_Workspace.studio_service import ResearchStudioService

pytestmark = pytest.mark.asyncio


def request(kind="report", route=ROUTE):
    return ResearchStudioRequest(
        "output-1",
        LOCAL,
        kind,
        (SOURCE, SECOND),
        ResearchGenerationConfig(title="Observatory results", item_count=2),
        route,
        7,
    )


class _GenerationFixture:
    def __init__(self):
        self.sources = (SOURCE, SECOND)
        self.approved = ()
        self.prompts = []
        self.answer = "North station results [S1]."
        self.calls = []
        self.evidence = (
            StudioEvidence(SOURCE, "Observation", "North station evidence"),
            StudioEvidence(
                replace(SECOND, source_id="other", catalog_item_id="999"),
                "Wrong item",
                "Excluded source evidence",
            ),
        )

    async def current_sources(self, ref):
        assert ref == LOCAL
        return self.sources

    async def read_evidence(self, requested):
        self.calls.append(requested.operation_id)
        return self.evidence

    async def approvals(self, ref):
        assert ref == LOCAL
        return self.approved

    async def stream(self, requested, messages):
        self.prompts.append(messages)
        yield self.answer

    def generator(self):
        return ResearchStudioGenerator(
            self.current_sources, self.read_evidence, self.stream, self.approvals
        )


async def test_studio_only_passes_captured_ready_evidence_to_the_provider():
    fixture = _GenerationFixture()
    chunks = []
    output = await fixture.generator().generate(
        request(), on_chunk=chunks.append, cancel_event=asyncio.Event()
    )
    assert output.status == "complete"
    assert output.text == fixture.answer == "".join(chunks)
    assert [row["role"] for row in fixture.prompts[0]] == ["system", "user"]
    assert "North station evidence" in repr(fixture.prompts)
    assert "Excluded source evidence" not in repr(fixture.prompts)
    assert [(citation.marker, citation.source) for citation in output.citations] == [
        ("S1", SOURCE)
    ]


async def test_remote_studio_requires_route_consent_and_refuses_changed_source_versions():
    fixture = _GenerationFixture()
    remote = replace(
        ROUTE,
        remote=True,
        endpoint_class="public_network",
        endpoint_fingerprint="a" * 64,
    )
    operation = request(route=remote)
    generator = fixture.generator()
    with pytest.raises(ValueError, match="consent"):
        await generator.generate(
            operation, on_chunk=lambda _: None, cancel_event=asyncio.Event()
        )
    assert not fixture.calls
    consent = ResearchEgressConsentService().preflight(operation)
    fixture.approved = (consent.fingerprint,)
    fixture.sources = (replace(SOURCE, catalog_item_version=12), SECOND)
    with pytest.raises(ValueError, match="sources changed"):
        await generator.generate(
            operation, on_chunk=lambda _: None, cancel_event=asyncio.Event()
        )
    assert not fixture.prompts


async def test_cancel_and_invalid_structured_output_do_not_return_saveable_work():
    fixture = _GenerationFixture()
    generator = fixture.generator()
    stop = asyncio.Event()
    stop.set()
    cancelled = await generator.generate(
        request(), on_chunk=lambda _: None, cancel_event=stop
    )
    assert cancelled.status == "cancelled" and not cancelled.text
    assert not fixture.calls

    fixture.answer = '{"items":[{"question":"Where?","options":["North","South"],"correct_index":8}]}'
    invalid = await generator.generate(
        request("quiz"), on_chunk=lambda _: None, cancel_event=asyncio.Event()
    )
    assert invalid.status == "failed" and not invalid.questions and not invalid.text
    fixture.answer = (
        '{"items":[{"front":"Location?","back":"North station","citations":["S1"]}]}'
    )
    valid = await generator.generate(
        request("flashcards"), on_chunk=lambda _: None, cancel_event=asyncio.Event()
    )
    assert valid.status == "complete"
    assert [card.front for card in valid.cards] == ["Location?"]
    assert valid.citations[0].source == SOURCE


async def test_retry_reuses_generated_content_after_a_canonical_write_failure():
    fixture = _GenerationFixture()
    operation = request()
    bodies = []

    class Owner:
        async def save(self, requested, generated, **_target):
            bodies.append(generated.text)
            if len(bodies) == 1:
                raise OSError("canonical owner temporarily unavailable")
            return WorkspaceOutputRef(
                LOCAL,
                requested.output_kind,
                "local_chatbook",
                "created-id",
                1,
                requested.source_snapshot,
                requested.generation.fingerprint,
                "complete",
            )

    service = ResearchStudioService(fixture.generator(), Owner())
    with pytest.raises(OSError):
        await service.generate(operation, lambda _chunk: None, asyncio.Event())
    result = await service.retry(LOCAL, operation.operation_id)
    assert result.owner_id == "created-id"
    assert bodies == [fixture.answer, fixture.answer]
    assert len(fixture.prompts) == 1
