"""Bounded metadata requests to Hugging Face's fixed public API origin."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx

_API_MODELS_URL = "https://huggingface.co/api/models"
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_SEARCH_RESULTS = 50
_MAX_QUERY_CHARACTERS = 256
_MAX_REPOSITORY_CHARACTERS = 96
_MAX_COUNTER = (2**63) - 1
_MAX_LAST_MODIFIED_CHARACTERS = 64
_REPOSITORY_COMPONENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


@dataclass(frozen=True)
class RemoteModelSummary:
    """Safe metadata used to present a remote repository search result."""

    repository: str
    private: bool
    gated: Literal["none", "auto", "manual"]
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None


class RemoteDiscoveryError(RuntimeError):
    """A stable, display-safe failure from remote metadata discovery."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details


class HuggingFaceRemoteAdapter:
    """Perform narrow, non-redirecting Hugging Face metadata searches."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._client_factory = client_factory

    async def search(
        self, query: str, *, token: str | None = None
    ) -> tuple[RemoteModelSummary, ...]:
        """Search the fixed Hugging Face models endpoint.

        Args:
            query: Free-text search input, limited after whitespace trimming.
            token: Optional Hugging Face token for this fixed-origin request.

        Returns:
            At most fifty validated remote model summaries.

        Raises:
            RemoteDiscoveryError: Input is invalid or remote metadata is unusable.
        """
        normalized_query = _validated_query(query)
        headers = _authorization_header(token)
        try:
            async with self._client_factory() as client:
                async with client.stream(
                    "GET",
                    _API_MODELS_URL,
                    params={"search": normalized_query, "limit": str(_MAX_SEARCH_RESULTS)},
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    _raise_for_status(response.status_code)
                    payload = await _read_bounded_json(response)
        except RemoteDiscoveryError:
            raise
        except httpx.TimeoutException as error:
            raise RemoteDiscoveryError("network_error", retryable=True) from error
        except httpx.HTTPError as error:
            raise RemoteDiscoveryError("network_error", retryable=True) from error

        return _parse_search_results(payload)


def is_exact_repository(value: str) -> bool:
    """Return whether ``value`` is one bounded portable owner/repository pair."""
    if not isinstance(value, str) or len(value) > _MAX_REPOSITORY_CHARACTERS:
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(
        _REPOSITORY_COMPONENT_RE.fullmatch(part)
        and "--" not in part
        and ".." not in part
        for part in parts
    )


def _validated_query(query: str) -> str:
    if not isinstance(query, str):
        raise RemoteDiscoveryError("invalid_query")
    normalized = query.strip()
    if not normalized or len(normalized) > _MAX_QUERY_CHARACTERS:
        raise RemoteDiscoveryError("invalid_query")
    return normalized


def _authorization_header(token: str | None) -> dict[str, str]:
    if token is None:
        return {}
    if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
        raise RemoteDiscoveryError("invalid_token")
    return {"Authorization": f"Bearer {token}"}


def _raise_for_status(status_code: int) -> None:
    if status_code == 401:
        raise RemoteDiscoveryError("authentication_required")
    if status_code == 403:
        raise RemoteDiscoveryError("access_forbidden")
    if status_code == 404:
        raise RemoteDiscoveryError("repository_not_found")
    if status_code == 429:
        raise RemoteDiscoveryError("rate_limited", retryable=True)
    if 400 <= status_code < 500:
        raise RemoteDiscoveryError("remote_error")
    if status_code >= 500:
        raise RemoteDiscoveryError("remote_error", retryable=True)


async def _read_bounded_json(response: httpx.Response) -> object:
    """Read one decoded metadata JSON body without exceeding its byte budget."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > _MAX_METADATA_BYTES:
            raise RemoteDiscoveryError("response_too_large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteDiscoveryError("invalid_response") from error


def _parse_search_results(payload: object) -> tuple[RemoteModelSummary, ...]:
    if not isinstance(payload, list) or len(payload) > _MAX_SEARCH_RESULTS:
        raise RemoteDiscoveryError("invalid_response")
    return tuple(_parse_search_result(item) for item in payload)


def _parse_search_result(item: object) -> RemoteModelSummary:
    if not isinstance(item, dict):
        raise RemoteDiscoveryError("invalid_response")
    repository = item.get("modelId")
    private = item.get("private")
    gated = item.get("gated")
    if not is_exact_repository(repository) or type(private) is not bool:
        raise RemoteDiscoveryError("invalid_response")
    normalized_gated = _normalized_gated(gated)
    if normalized_gated is None:
        raise RemoteDiscoveryError("invalid_response")
    return RemoteModelSummary(
        repository=repository,
        private=private,
        gated=normalized_gated,
        downloads=_optional_counter(item.get("downloads")),
        likes=_optional_counter(item.get("likes")),
        last_modified=_optional_last_modified(item.get("lastModified")),
    )


def _normalized_gated(value: object) -> Literal["none", "auto", "manual"] | None:
    if value is False:
        return "none"
    if value == "auto":
        return "auto"
    if value == "manual":
        return "manual"
    return None


def _optional_counter(value: object) -> int | None:
    if type(value) is int and 0 <= value <= _MAX_COUNTER:
        return value
    return None


def _optional_last_modified(value: object) -> str | None:
    if isinstance(value, str) and len(value) <= _MAX_LAST_MODIFIED_CHARACTERS:
        return value
    return None
