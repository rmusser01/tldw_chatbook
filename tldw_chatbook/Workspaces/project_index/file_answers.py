"""Transient cited answers over a single Local Files search."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from tldw_chatbook.Chat.citation_evidence_models import (
    EVIDENCE_SNIPPET_CHAR_LIMIT,
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.provider_readiness import (
    configured_workspace_base_url,
    resolve_provider_credential,
)
from tldw_chatbook.Chat.provider_setup_persistence import (
    resolve_remembered_provider_model,
)
from tldw_chatbook.LLM_Calls.moonshot import (
    MoonshotResolution,
    build_moonshot_chat_payload,
    resolve_moonshot_request,
)
from tldw_chatbook.Library.library_rag_answer_service import (
    ANSWER_STATUS_FAILED,
    ANSWER_STATUS_READY,
    LibraryRagAnswer,
    build_answer_system_prompt,
    generate_evidence_answer,
)
from tldw_chatbook.Provider_Contract import ChatConfigurationError
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    resolve_provider_endpoint,
)
from tldw_chatbook.config import provider_settings_for_key

from .models import BindingSearchResult, ProjectIndexHit


_PROVIDER_LABELS = {
    "llama_cpp": "llama.cpp",
    "ollama": "Ollama",
    "moonshot": "Moonshot",
}
_CHAT_ROUTE = "/chat/completions"
_LOCAL_CHAT_ROUTE = f"/v1{_CHAT_ROUTE}"
_NO_EVIDENCE = "The matching files do not support an answer to that."
_SYSTEM_PROMPT = build_answer_system_prompt(
    source_description="the selected folder",
    source_name="the file evidence",
    no_evidence_text=_NO_EVIDENCE,
)
_Settings = Mapping[str, Any] | Callable[[], Mapping[str, Any]]
_ValidateHits = Callable[
    [BindingSearchResult, tuple[ProjectIndexHit, ...]], tuple[Path, ...] | None
]


@dataclass(frozen=True, slots=True)
class FileAnswerDestination:
    """Provider, explicit model, and exact request URL shown before dispatch."""

    provider: str
    model: str
    url: str
    _credential_digest: bytes = field(repr=False)

    @property
    def external(self) -> bool:
        return self.provider == "moonshot"

    @property
    def label(self) -> str:
        return f"{_PROVIDER_LABELS[self.provider]} · {self.model} · {self.url}"


def _local_request_url(provider: str, value: str) -> str | None:
    endpoint = resolve_provider_endpoint(provider, value)
    if endpoint.chat_url is None or endpoint.errors:
        return None
    target = endpoint.chat_url
    if provider == "ollama" and endpoint.normalized_input.endswith("/api/chat"):
        # The native Ollama URL is sometimes entered in Settings. Use its
        # OpenAI-compatible sibling for the shared answer request format.
        target = endpoint.normalized_input.removesuffix("/api/chat") + _LOCAL_CHAT_ROUTE
    url = httpx.URL(target)
    if url.host == "localhost":
        # Use a literal loopback IP so a hosts-file or DNS override cannot
        # turn a choice marked local into an external disclosure.
        url = url.copy_with(host="127.0.0.1")
    return str(url) if url.host in {"127.0.0.1", "::1"} else None


def _resolve_destination(
    config: Mapping[str, Any], provider: str
) -> tuple[FileAnswerDestination, str, MoonshotResolution | None] | None:
    if provider not in _PROVIDER_LABELS:
        return None
    try:
        model = resolve_remembered_provider_model(config, provider)
        moonshot = None
        if provider == "moonshot":
            moonshot = resolve_moonshot_request(app_config=config, explicit_model=model)
            model, api_key = moonshot.model, moonshot.api_key
            url = httpx.URL(f"{moonshot.base_url}{_CHAT_ROUTE}")
            if url.scheme != "https":
                return None
            target = str(url)
        else:
            api_settings = config.get("api_settings")
            if not model or not isinstance(api_settings, Mapping):
                return None
            settings = provider_settings_for_key(api_settings, provider)
            configured_url = configured_workspace_base_url(settings)
            if (
                not configured_url
                or (target := _local_request_url(provider, configured_url)) is None
            ):
                return None
            api_key, _, _ = resolve_provider_credential(
                provider, settings, environ=os.environ
            )
    except (ValueError, ChatConfigurationError, httpx.InvalidURL):
        return None

    key = api_key or ""
    return (
        FileAnswerDestination(provider, model, target, sha256(key.encode()).digest()),
        key,
        moonshot,
    )


def list_answer_destinations(
    app_config: _Settings,
) -> tuple[FileAnswerDestination, ...]:
    """Offer configured loopback providers first and one HTTPS hosted provider."""
    config = app_config() if callable(app_config) else app_config
    return tuple(
        resolved[0]
        for provider in _PROVIDER_LABELS
        if (resolved := _resolve_destination(config, provider)) is not None
    )


class _AnswerUnavailable(RuntimeError):
    """A code-owned refusal safe to show without provider or file details."""


async def _check_ollama_model(
    client: httpx.AsyncClient, destination: FileAnswerDestination, api_key: str
) -> None:
    # The local daemon also serves cloud-backed models, including aliased ones.
    # Ask only for model metadata before entrusting the daemon with any text.
    response = await client.post(
        f"{destination.url.removesuffix(_LOCAL_CHAT_ROUTE)}/api/show",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={"model": destination.model},
        timeout=10.0,
    )
    response.raise_for_status()
    details = response.json()
    if (
        not isinstance(details, dict)
        or not details.get("model_info")
        or details.get("remote_model")
        or details.get("remote_host")
    ):
        raise _AnswerUnavailable("Choose an installed local Ollama model.")


async def answer_files(
    *,
    query: str,
    search: BindingSearchResult,
    destination: FileAnswerDestination,
    approved: bool,
    app_config: _Settings,
    validate: _ValidateHits,
    is_current: Callable[[], bool],
) -> LibraryRagAnswer:
    """Answer from bounded excerpts while the request and file receipts are live."""
    selected = tuple(
        (index, hit)
        for index, hit in enumerate(search.result.hits)
        if hit.excerpt.strip()
    )
    hits = tuple(hit for _, hit in selected)

    def build_bundle() -> EvidenceBundle:
        bundle_id = uuid4().hex
        references = []
        for citation, (hit_index, hit) in enumerate(selected, 1):
            text = hit.excerpt.strip()
            offset = hit.excerpt[: len(hit.excerpt) - len(hit.excerpt.lstrip())].count("\n")
            line_start = hit.line_start + offset
            submitted = text[:EVIDENCE_SNIPPET_CHAR_LIMIT].rstrip()
            line_end = min(hit.line_end, line_start + submitted.count("\n"))
            references.append(
                EvidenceReference(
                    evidence_id=f"S{citation}",
                    source_id=f"{bundle_id}-{citation}",
                    source_type="workspace_file",
                    title=f"{hit.relative_path} · lines {line_start}–{line_end}",
                    snippet=text,
                    authority_label="Selected folder",
                    metadata={
                        "relative_path": hit.relative_path,
                        "line_start": line_start,
                        "line_end": line_end,
                        "hit_index": hit_index,
                    },
                )
            )
        return EvidenceBundle(
            bundle_id=bundle_id,
            query=query,
            source="Local Files",
            references=tuple(references),
        )

    async def check_evidence() -> None:
        if (
            not is_current()
            or await asyncio.to_thread(validate, search, hits) is None
            or not is_current()
        ):
            raise _AnswerUnavailable(
                "The files or search changed. Run the search again."
            )

    def current_destination() -> (
        tuple[FileAnswerDestination, str, MoonshotResolution | None]
    ):
        config = app_config() if callable(app_config) else app_config
        resolved = _resolve_destination(config, destination.provider)
        if resolved is None or resolved[0] != destination:
            raise _AnswerUnavailable("The answer destination changed. Select it again.")
        return resolved

    async def chat(
        *, messages_payload, system_message, max_tokens, temp, **_options
    ) -> dict[str, Any]:
        if destination.external and approved is not True:
            raise _AnswerUnavailable("Confirm sending this question and file excerpts.")

        try:
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False
            ) as client:
                if destination.provider == "ollama" and is_current():
                    target, api_key, _ = current_destination()
                    await _check_ollama_model(client, target, api_key)
                # Run bounded native file validation off the event loop, then
                # resolve current settings immediately before sending evidence.
                await check_evidence()
                target, api_key, moonshot = current_destination()
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                if moonshot is not None:
                    payload = build_moonshot_chat_payload(
                        resolution=moonshot,
                        messages_payload=messages_payload,
                        system_message=system_message,
                        streaming=False,
                        max_tokens=max_tokens,
                        temperature=temp,
                    )
                else:
                    payload = {
                        "model": target.model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            *messages_payload,
                        ],
                        "stream": False,
                        "max_tokens": max_tokens,
                        "temperature": temp,
                    }
                if not is_current():
                    raise _AnswerUnavailable("The search changed. Run it again.")
                response = await client.post(
                    target.url,
                    headers=headers,
                    json=payload,
                    timeout=moonshot.timeout if moonshot else 300.0,
                )
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw, dict):
                    raise _AnswerUnavailable(
                        "The answer provider returned an invalid response."
                    )
                await check_evidence()
                return raw
        except _AnswerUnavailable:
            raise
        except Exception:
            # Provider or filesystem exceptions can include private data.
            raise _AnswerUnavailable(
                "Could not get an answer from the selected provider."
            ) from None

    answer = await generate_evidence_answer(
        query=query,
        build_bundle=build_bundle,
        coverage_note="",
        provider=destination.provider,
        model=destination.model,
        chat=chat,
        system_prompt=_SYSTEM_PROMPT,
        no_evidence_text=_NO_EVIDENCE,
    )
    if answer.status == ANSWER_STATUS_READY and answer.citation_status != "validated":
        return replace(
            answer,
            status=ANSWER_STATUS_FAILED,
            text="",
            error="The answer did not cite the submitted file evidence.",
            evidence_bundle=None,
        )
    if answer.status == ANSWER_STATUS_FAILED:
        return replace(answer, evidence_bundle=None)
    return answer
