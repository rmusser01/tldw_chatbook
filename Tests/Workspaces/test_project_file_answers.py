"""Local Files answer tests at the physical provider request boundary."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Workspaces.project_index.models import (
    BindingSearchResult,
    ProjectIndexHit,
    ProjectSearchResult,
)


QUERY = "What controls retention?"
HIT = ProjectIndexHit("notes/retention.md", 4, 7, "Archive the files after six days.")
NAME_ONLY = ProjectIndexHit("retention.txt", 1, 1, "")
LOCAL = {"api_url": "http://localhost:8080", "model": "local-model"}
HOSTED = {
    "api_base_url": "https://api.moonshot.ai/v1",
    "api_key": "test-moonshot-credential",
    "model": "moonshot-v1-8k",
}


def _search(*hits):
    return BindingSearchResult(ProjectSearchResult("ready", hits, None, False))


def _validate(_search_result, hits):
    return tuple(Path(hit.relative_path) for hit in hits)


@pytest.fixture
def wire(monkeypatch):
    from tldw_chatbook.Workspaces.project_index import file_answers

    class Wire:
        requests: list[httpx.Request]
        body = "The files are archived after six days [S1]."
        redirect = False
        ollama_details = {"model_info": {"general.architecture": "llama"}}

        def respond(self, request):
            self.requests.append(request)
            if request.url.path == "/api/show":
                return httpx.Response(200, json=self.ollama_details)
            if self.redirect:
                return httpx.Response(
                    307, headers={"Location": "https://other.example/"}
                )
            return httpx.Response(
                200,
                json={
                    "model": json.loads(request.content)["model"],
                    "choices": [
                        {"message": {"role": "assistant", "content": self.body}},
                    ],
                },
            )

    wire = Wire()
    wire.requests = []
    original_client = httpx.AsyncClient

    def client(**options):
        assert options["trust_env"] is False
        return original_client(transport=httpx.MockTransport(wire.respond), **options)

    monkeypatch.setattr(file_answers.httpx, "AsyncClient", client)
    return wire


async def _ask(
    settings,
    *,
    search=None,
    approved=False,
    validate=_validate,
    is_current=lambda: True,
    destination=None,
):
    from tldw_chatbook.Workspaces.project_index.file_answers import (
        answer_files,
        list_answer_destinations,
    )

    if destination is None:
        (destination,) = list_answer_destinations(settings)
    return await answer_files(
        query=QUERY,
        search=search if search is not None else _search(NAME_ONLY, HIT),
        destination=destination,
        approved=approved,
        app_config=settings,
        validate=validate,
        is_current=is_current,
    )


@pytest.mark.asyncio
async def test_local_answer_cites_submitted_relative_path_and_lines(wire):
    answer = await _ask({"api_settings": {"llama_cpp": LOCAL}})

    assert answer.status == "ready"
    assert answer.citation_status == "validated"
    (reference,) = answer.evidence_bundle.references
    assert reference.evidence_id == "S1"
    assert reference.source_type == "workspace_file"
    assert reference.metadata == {
        "relative_path": "notes/retention.md",
        "line_start": 4,
        "line_end": 4,
        "hit_index": 1,
    }
    assert HIT.relative_path not in reference.source_id
    (request,) = wire.requests
    assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == "local-model"
    prompt = payload["messages"][1]["content"]
    assert QUERY in prompt and HIT.excerpt in prompt and reference.title in prompt
    assert NAME_ONLY.relative_path not in prompt


@pytest.mark.asyncio
async def test_file_citation_covers_only_the_submitted_lines(wire):
    excerpt = "a" * 3998 + "\nnext line\nfinal line"
    hit = ProjectIndexHit("notes/long.txt", 11, 13, excerpt)
    answer = await _ask(
        {"api_settings": {"llama_cpp": LOCAL}}, search=_search(NAME_ONLY, hit)
    )

    (reference,) = answer.evidence_bundle.references
    assert reference.metadata["hit_index"] == 1
    assert reference.metadata["line_start"] == 11
    assert reference.metadata["line_end"] == 12
    assert reference.snippet.endswith("\nn")
    (request,) = wire.requests
    assert "lines 11–12" in json.loads(request.content)["messages"][1]["content"]


@pytest.mark.asyncio
async def test_filename_only_evidence_makes_no_generation_request(wire):
    answer = await _ask(
        {"api_settings": {"llama_cpp": LOCAL}},
        search=_search(NAME_ONLY),
    )
    assert answer.status == "no_evidence"
    assert not wire.requests


@pytest.mark.asyncio
async def test_hosted_confirmation_pins_visible_model_url_and_evidence(wire):
    from tldw_chatbook.Workspaces.project_index.file_answers import (
        list_answer_destinations,
    )

    settings = {"api_settings": {"moonshot": HOSTED}}
    (destination,) = list_answer_destinations(settings)
    assert destination.external
    assert "https://api.moonshot.ai/v1/chat/completions" in destination.label
    assert "moonshot-v1-8k" in destination.label
    assert HOSTED["api_key"] not in repr(destination)

    assert (await _ask(settings)).status == "failed"
    assert not wire.requests
    answer = await _ask(settings, approved=True)
    assert answer.citation_status == "validated"
    (request,) = wire.requests
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.moonshot.ai/v1/chat/completions"
    assert payload["model"] == "moonshot-v1-8k"
    assert HIT.excerpt in payload["messages"][1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed", ["target", "model", "credential", "evidence", "request"]
)
async def test_changed_approval_is_refused_before_transport(wire, changed):
    config = {"api_settings": {"moonshot": dict(HOSTED)}}
    current = True

    def validate(search, hits):
        nonlocal current
        provider = config["api_settings"]["moonshot"]
        if changed == "target":
            provider["api_base_url"] = "https://api.moonshot.cn/v1"
        elif changed == "model":
            provider["model"] = "moonshot-v1-32k"
        elif changed == "credential":
            provider["api_key"] = "changed-moonshot-credential"
        elif changed == "request":
            current = False
        elif changed == "evidence":
            return None
        return _validate(search, hits)

    answer = await _ask(
        lambda: config, approved=True, validate=validate, is_current=lambda: current
    )
    assert answer.status == "failed"
    assert not wire.requests


@pytest.mark.asyncio
async def test_evidence_revoked_during_completion_cannot_publish_answer(wire):
    def validate(search, hits):
        return None if wire.requests else _validate(search, hits)

    answer = await _ask({"api_settings": {"llama_cpp": LOCAL}}, validate=validate)
    assert len(wire.requests) == 1
    assert answer.status == "failed"
    assert answer.text == ""
    assert answer.evidence_bundle is None


@pytest.mark.asyncio
async def test_local_request_does_not_follow_redirect(wire):
    wire.redirect = True
    answer = await _ask({"api_settings": {"llama_cpp": LOCAL}})
    assert answer.status == "failed"
    assert len(wire.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cloud_alias", [False, True])
async def test_ollama_checks_installed_model_before_sending_excerpts(wire, cloud_alias):
    if cloud_alias:
        wire.ollama_details = {
            **wire.ollama_details,
            "remote_host": "https://ollama.com",
            "remote_model": "gemma4:31b",
        }
    settings = {
        "api_settings": {
            "ollama": {
                "api_url": "http://localhost:11434/api/chat",
                "model": "gemma4",
            }
        }
    }

    answer = await _ask(settings)
    model_check, *generation = wire.requests
    assert str(model_check.url) == "http://127.0.0.1:11434/api/show"
    assert json.loads(model_check.content) == {"model": "gemma4"}
    if cloud_alias:
        assert answer.status == "failed"
        assert not generation
    else:
        assert answer.status == "ready"
        (request,) = generation
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert HIT.excerpt in json.loads(request.content)["messages"][1]["content"]


def test_remote_self_hosted_endpoint_cannot_be_selected_as_local():
    from tldw_chatbook.Workspaces.project_index.file_answers import (
        list_answer_destinations,
    )

    settings = {
        "api_settings": {"llama_cpp": {**LOCAL, "api_url": "https://remote.example"}}
    }
    assert not list_answer_destinations(settings)


@pytest.mark.parametrize(
    ("provider", "configured", "target"),
    [
        (
            "llama_cpp",
            "localhost:8080/completion",
            "http://127.0.0.1:8080/v1/chat/completions",
        ),
        (
            "llama_cpp",
            "http://localhost:8080/v1",
            "http://127.0.0.1:8080/v1/chat/completions",
        ),
        (
            "ollama",
            "http://localhost:11434",
            "http://127.0.0.1:11434/v1/chat/completions",
        ),
        (
            "ollama",
            "http://localhost:11434/v1/chat/completions",
            "http://127.0.0.1:11434/v1/chat/completions",
        ),
        (
            "ollama",
            "http://localhost:11434/api/chat",
            "http://127.0.0.1:11434/v1/chat/completions",
        ),
    ],
)
def test_local_settings_endpoint_forms_resolve_to_a_chat_url(
    provider, configured, target
):
    from tldw_chatbook.Workspaces.project_index.file_answers import (
        list_answer_destinations,
    )

    settings = {
        "api_settings": {provider: {"api_url": configured, "model": "local-model"}}
    }
    (destination,) = list_answer_destinations(settings)
    assert destination.url == target
    assert not destination.external


@pytest.mark.asyncio
async def test_unknown_file_citation_is_not_shown_as_an_answer(wire):
    wire.body = "The files are archived after six days [S2]."
    answer = await _ask({"api_settings": {"llama_cpp": LOCAL}})
    assert answer.status != "ready"
    assert "[S2]" not in answer.text
