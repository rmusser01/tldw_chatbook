"""Tests for bounded Hugging Face metadata search (TASK-596.1)."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import traceback

import httpx
import pytest

from tldw_chatbook.Model_Artifacts.remote_huggingface import (
    HuggingFaceRemoteAdapter,
    RemoteGGUFCandidate,
    RemoteGGUFFile,
    RemoteDiscoveryError,
    RemoteModelSummary,
    ResolvedRemoteModel,
    build_remote_catalog,
    is_exact_repository,
)
from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService
from tldw_chatbook.Model_Artifacts.service import (
    ModelArtifactService,
    ProvenanceClass,
)


def _client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[[], httpx.AsyncClient]:
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _model_info(
    siblings: list[object],
    *,
    commit: str = "a" * 40,
    card_data: object = None,
) -> dict[str, object]:
    """Return a complete minimal Hugging Face repository-info response."""
    return {"sha": commit, "siblings": siblings, "cardData": card_data}


def _lfs_file(path: str, *, size: int = 123, digest: str = "b" * 64) -> dict[str, object]:
    """Return one complete LFS-backed GGUF sibling response entry."""
    return {"rfilename": path, "lfs": {"size": size, "sha256": digest}}


@pytest.mark.asyncio
async def test_search_trims_query_and_uses_fixed_bounded_request() -> None:
    """Catches a wrong endpoint, untrimmed query, or unbounded search."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    assert await adapter.search("  whisper  ", token="secret") == ()
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url).split("?")[0] == "https://huggingface.co/api/models"
    assert requests[0].url.params["search"] == "whisper"
    assert requests[0].url.params["limit"] == "50"
    assert requests[0].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_search_disables_redirects() -> None:
    """Catches a metadata request that could forward credentials via redirects."""
    recorded: list[bool] = []

    class TrackingClient(httpx.AsyncClient):
        def stream(self, method: str, url: str | httpx.URL, **kwargs: object):
            recorded.append(bool(kwargs["follow_redirects"]))
            return super().stream(method, url, **kwargs)

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: TrackingClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
        )
    )

    assert await adapter.search("whisper", token="secret") == ()
    assert recorded == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "code", "retryable"),
    [
        (401, "authentication_required", False),
        (403, "access_forbidden", False),
        (404, "repository_not_found", False),
        (429, "rate_limited", True),
    ],
)
async def test_search_sanitizes_expected_http_errors(
    status_code: int, code: str, retryable: bool
) -> None:
    """Catches raw upstream error propagation or a wrong retry policy."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(status_code, text="upstream secret detail")
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert raised.value.details == ()
    assert "upstream secret detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_search_sanitizes_timeout() -> None:
    """Catches timeout leakage instead of a recoverable discovery failure."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream secret", request=request)

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "network_error",
        True,
        (),
    )
    assert "upstream secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_search_timeout_traceback_has_no_upstream_secret_or_cause() -> None:
    """Catches chained HTTPX errors that expose credentials or upstream text."""
    secret = "upstream-secret-and-token"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    adapter = HuggingFaceRemoteAdapter(client_factory=_client_factory(handler))

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper", token=secret)

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"metadata":"json-secret-marker",}',
        b'{"metadata":"utf8-secret-marker"}\xff',
    ],
    ids=["malformed-json", "invalid-utf8"],
)
async def test_search_parser_failures_have_no_cause_or_upstream_marker(
    body: bytes,
) -> None:
    """Catches parser exception chains exposing an upstream response body."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, content=body))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "secret-marker" not in rendered


@pytest.mark.asyncio
async def test_search_rejects_redirect_status_as_remote_error() -> None:
    """Catches a redirect response being parsed as trusted search metadata."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(302, json=[]))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "remote_error",
        False,
        (),
    )


@pytest.mark.parametrize(
    ("code", "details"),
    [
        ("unexpected", ()),
        ("invalid_response", ("x" * 553,)),
        ("invalid_response", ("line\nbreak",)),
        ("invalid_response", tuple("warning" for _ in range(21))),
        ("invalid_response", ["warning"]),
    ],
)
def test_remote_discovery_error_rejects_unbounded_or_unsanitized_values(
    code: str, details: object
) -> None:
    """Catches public error values that could retain arbitrary upstream content."""
    with pytest.raises(ValueError):
        RemoteDiscoveryError(code, details=details)  # type: ignore[arg-type]


def test_remote_discovery_error_retains_bounded_display_safe_warnings() -> None:
    """Catches rejection of the bounded warning capacity needed by Task 2."""
    details = tuple("model.gguf missing 00001" for _ in range(20))

    error = RemoteDiscoveryError("no_eligible_gguf", details=details)

    assert error.details == details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [b"not json", b"[" + (b" " * (2 * 1024 * 1024)) + b"]"],
)
async def test_search_rejects_malformed_or_oversized_response(body: bytes) -> None:
    """Catches decoding before response-size enforcement or raw parse errors."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, content=body))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("whisper")

    assert raised.value.code in {"invalid_response", "response_too_large"}
    assert raised.value.retryable is False
    assert raised.value.details == ()


@pytest.mark.asyncio
async def test_search_parses_valid_bounded_metadata() -> None:
    """Catches loss of valid private/gated metadata or malformed optional fields."""
    models = [
        {
            "modelId": "acme/private",
            "private": True,
            "gated": False,
            "downloads": 42,
            "likes": 3,
            "lastModified": "2026-08-01T00:00:00Z",
        },
        {
            "modelId": "acme/auto",
            "private": False,
            "gated": "auto",
            "downloads": -1,
            "likes": "3",
            "lastModified": "x" * 65,
        },
        {
            "modelId": "acme/manual",
            "private": False,
            "gated": "manual",
            "downloads": 2**63 - 1,
            "likes": 0,
            "lastModified": "updated",
        },
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, json=models))
    )

    assert await adapter.search("models") == (
        RemoteModelSummary(
            repository="acme/private",
            private=True,
            gated="none",
            downloads=42,
            likes=3,
            last_modified="2026-08-01T00:00:00Z",
        ),
        RemoteModelSummary(
            repository="acme/auto", private=False, gated="auto"
        ),
        RemoteModelSummary(
            repository="acme/manual",
            private=False,
            gated="manual",
            downloads=2**63 - 1,
            likes=0,
            last_modified="updated",
        ),
    )


@pytest.mark.asyncio
async def test_search_caps_results_at_fifty() -> None:
    """Catches a server response exceeding the declared result limit."""
    models = [
        {"modelId": f"owner/model-{index}", "private": False, "gated": False}
        for index in range(51)
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: httpx.Response(200, json=models))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search("models")

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "invalid_response",
        False,
        (),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("owner/repository", True),
        ("a/b", True),
        ("owner/repository/extra", False),
        ("owner", False),
        (" owner/repository", False),
        ("owner/repository ", False),
        ("owner/" + ("a" * 91), False),
        ("owner/repo?query", False),
        ("owner/repo--name", False),
        ("owner/repo..name", False),
        ("owner/repo-", False),
    ],
)
def test_is_exact_repository_requires_one_bounded_portable_pair(
    value: str, expected: bool
) -> None:
    """Catches unsafe, ambiguous, or oversized exact repository identifiers."""
    assert is_exact_repository(value) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "x" * 257])
async def test_search_rejects_empty_or_oversized_trimmed_query(query: str) -> None:
    """Catches unbounded or blank user input reaching the remote endpoint."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(lambda _: pytest.fail("request was sent"))
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.search(query)

    assert (raised.value.code, raised.value.retryable, raised.value.details) == (
        "invalid_query",
        False,
        (),
    )


@pytest.mark.asyncio
async def test_resolve_uses_pinned_model_info_request_with_lfs_blobs() -> None:
    """Catches a mutable, unpinned, or credential-leaking resolution request."""
    requests: list[httpx.Request] = []
    redirects: list[bool] = []

    class TrackingClient(httpx.AsyncClient):
        def stream(self, method: str, url: str | httpx.URL, **kwargs: object):
            redirects.append(bool(kwargs["follow_redirects"]))
            return super().stream(method, url, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_model_info([_lfs_file("model.gguf")]))

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: TrackingClient(transport=httpx.MockTransport(handler))
    )

    resolved = await adapter.resolve("acme/model", token="secret")

    assert resolved.repository == "acme/model"
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url).split("?")[0] == (
        "https://huggingface.co/api/models/acme/model"
    )
    assert requests[0].url.params == httpx.QueryParams({"blobs": "true"})
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert redirects == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", ["main", "A" * 40, "a" * 39, "a" * 41])
async def test_resolve_rejects_non_immutable_commit(commit: str) -> None:
    """Catches a branch, tag, or malformed revision becoming an artifact revision."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info([], commit=commit))
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert raised.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_resolve_rejects_repository_with_over_2048_files() -> None:
    """Catches silently inspecting only part of an unbounded repository listing."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(
                200,
                json=_model_info([_lfs_file(f"model-{index}.gguf") for index in range(2049)]),
            )
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert raised.value.code == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("card_data", "expected"),
    [
        ({"license": "apache-2.0"}, "apache-2.0"),
        (None, "NOASSERTION"),
        ({}, "NOASSERTION"),
        ("not-a-mapping", "NOASSERTION"),
        ({"license": None}, "NOASSERTION"),
        ({"license": ""}, "NOASSERTION"),
        ({"license": 12}, "NOASSERTION"),
        ({"license": "x" * 129}, "NOASSERTION"),
    ],
)
async def test_resolve_uses_only_bounded_card_data_license(
    card_data: object, expected: str
) -> None:
    """Catches license inference from non-authoritative or malformed card fields."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(
                200,
                json=_model_info([_lfs_file("model.gguf")], card_data=card_data),
            )
        )
    )

    assert (await adapter.resolve("acme/model")).license_id == expected


@pytest.mark.asyncio
async def test_resolve_groups_complete_shards_and_keeps_single_files_sorted() -> None:
    """Catches shard members becoming selectable singles or unstable file ordering."""
    siblings = [
        _lfs_file("z.gguf", size=9, digest="9" * 64),
        _lfs_file("nested/pack-00003-of-00003.gguf", size=3, digest="3" * 64),
        _lfs_file("nested/pack-00001-of-00003.gguf", size=1, digest="1" * 64),
        _lfs_file("nested/pack-00002-of-00003.gguf", size=2, digest="2" * 64),
        _lfs_file("a.gguf", size=4, digest="4" * 64),
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    resolved = await adapter.resolve("acme/model")

    assert [(candidate.label, candidate.total_bytes) for candidate in resolved.candidates] == [
        ("acme/model · a.gguf", 4),
        ("acme/model · nested/pack", 6),
        ("acme/model · z.gguf", 9),
    ]
    assert [item.upstream_path for item in resolved.candidates[1].files] == [
        "nested/pack-00001-of-00003.gguf",
        "nested/pack-00002-of-00003.gguf",
        "nested/pack-00003-of-00003.gguf",
    ]


@pytest.mark.asyncio
async def test_resolve_rejects_incomplete_shards_without_reintroducing_members() -> None:
    """Catches rejected shard members reappearing as independently installable files."""
    siblings = [
        _lfs_file("nested/pack-00001-of-00003.gguf"),
        _lfs_file("nested/pack-00003-of-00003.gguf"),
        _lfs_file("single.gguf"),
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    resolved = await adapter.resolve("acme/model")

    assert [candidate.label for candidate in resolved.candidates] == [
        "acme/model · single.gguf"
    ]
    assert resolved.warnings == ("acme/model · nested/pack missing 00002",)


@pytest.mark.asyncio
async def test_resolve_carries_incomplete_shard_warning_when_nothing_is_eligible() -> None:
    """Catches loss of actionable incomplete-shard context on empty resolution."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(
                200,
                json=_model_info([_lfs_file("pack-00001-of-00002.gguf")]),
            )
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert raised.value.code == "no_eligible_gguf"
    assert raised.value.details == ("acme/model · pack missing 00002",)


@pytest.mark.asyncio
async def test_resolve_keeps_all_missing_indexes_for_a_maximum_label_warning() -> None:
    """Catches truncating required missing shard indexes to fit an error-detail cap."""
    repository = ("o" * 47) + "/" + ("r" * 48)
    stem = "s" * 61
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(
                200,
                json=_model_info(
                    [{"rfilename": f"{stem}-00001-of-00064.gguf", "lfs": {}}]
                ),
            )
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve(repository)

    warning = raised.value.details[0]
    assert len(warning) == 552
    assert warning.startswith(f"{repository} · {stem} missing 00001")
    assert warning.endswith("00064")


@pytest.mark.asyncio
async def test_resolve_rejects_malformed_or_oversized_shard_sets() -> None:
    """Catches invalid shard cardinalities returning an installable candidate."""
    siblings = [
        _lfs_file("bad-00000-of-00002.gguf"),
        _lfs_file("large-00001-of-00065.gguf"),
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert raised.value.code == "no_eligible_gguf"


@pytest.mark.asyncio
async def test_resolve_rejects_huge_shard_count_without_expanding_warning() -> None:
    """Catches expanding an attacker-declared shard count into an unbounded warning."""
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(
                200,
                json=_model_info([_lfs_file("model-00001-of-99999.gguf")]),
            )
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert (raised.value.code, raised.value.details) == ("no_eligible_gguf", ())


@pytest.mark.asyncio
async def test_resolve_retains_only_twenty_valid_incomplete_shard_warnings() -> None:
    """Catches retaining more than twenty bounded incomplete-shard warnings."""
    siblings = [
        _lfs_file(f"set-{index:02d}-00001-of-00002.gguf") for index in range(21)
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    with pytest.raises(RemoteDiscoveryError) as raised:
        await adapter.resolve("acme/model")

    assert raised.value.code == "no_eligible_gguf"
    assert raised.value.details == tuple(
        f"acme/model · set-{index:02d} missing 00002" for index in range(20)
    )


@pytest.mark.asyncio
async def test_resolve_ignores_gguf_without_complete_lfs_metadata() -> None:
    """Catches unverified GGUF payload metadata becoming an acquisition candidate."""
    siblings = [
        {"rfilename": "missing.gguf", "lfs": {"size": 4}},
        _lfs_file("valid.gguf", size=5),
    ]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    assert [item.label for item in (await adapter.resolve("acme/model")).candidates] == [
        "acme/model · valid.gguf"
    ]


@pytest.mark.asyncio
async def test_resolve_records_total_before_deterministic_candidate_cap() -> None:
    """Catches a display cap that hides the true candidate count or changes ordering."""
    siblings = [_lfs_file(f"{index:03d}.gguf") for index in range(101, -1, -1)]
    adapter = HuggingFaceRemoteAdapter(
        client_factory=_client_factory(
            lambda _: httpx.Response(200, json=_model_info(siblings))
        )
    )

    resolved = await adapter.resolve("acme/model")

    assert resolved.total_candidate_count == 102
    assert len(resolved.candidates) == 100
    assert resolved.candidates[0].files[0].upstream_path == "000.gguf"
    assert resolved.candidates[-1].files[0].upstream_path == "099.gguf"


def test_build_remote_catalog_maps_a_candidate_to_one_inert_pinned_artifact() -> None:
    """Catches mutable IDs/URLs, unsafe names, or compatibility claims for remote bytes."""
    resolved = ResolvedRemoteModel(
        repository="acme/model",
        commit="a" * 40,
        license_id="apache-2.0",
        review_url="https://huggingface.co/acme/model/tree/" + ("a" * 40),
        candidates=(),
        total_candidate_count=0,
        warnings=(),
    )
    candidate = RemoteGGUFCandidate(
        label="acme/model · nested/pack",
        files=(
            RemoteGGUFFile("nested/pack-00001-of-00002.gguf", 11, "1" * 64),
            RemoteGGUFFile("nested/pack-00002-of-00002.gguf", 12, "2" * 64),
        ),
        total_bytes=23,
    )

    catalog = build_remote_catalog(resolved, candidate)
    artifact = catalog.artifact

    assert artifact.reference.artifact_id == (
        "hf-gguf-0cac08cf6bec99fb43ebc68340f029996d72b111ec52945b773fbae8d6005e05"
    )
    assert (artifact.reference.revision, artifact.reference.variant, artifact.precision) == (
        "a" * 40,
        "not-declared",
        "not-declared",
    )
    assert (artifact.consumer, artifact.model_family, artifact.runtime_name) == (
        "unassigned",
        "unassigned",
        "unassigned",
    )
    assert artifact.runtime_version_constraint == "none"
    assert artifact.supported_os == ("unassigned",)
    assert artifact.supported_architectures == ("unassigned",)
    assert [item.path for item in artifact.files] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]
    assert [(item.size_bytes, item.sha256) for item in artifact.files] == [
        (11, "1" * 64),
        (12, "2" * 64),
    ]
    assert artifact.expected_installed_bytes == 23
    assert artifact.license_id == "apache-2.0"
    assert artifact.license_url == resolved.review_url
    assert artifact.source_url == (
        "https://huggingface.co/acme/model/resolve/" + ("a" * 40)
        + "/nested/pack-00001-of-00002.gguf"
    )
    assert catalog.sources[artifact.reference] == {
        "model-00001-of-00002.gguf": artifact.source_url,
        "model-00002-of-00002.gguf": (
            "https://huggingface.co/acme/model/resolve/" + ("a" * 40)
            + "/nested/pack-00002-of-00002.gguf"
        ),
    }
    assert artifact.dependencies == ()
    assert artifact.usage_notice == (
        "Runtime compatibility has not been verified. Configuration is required."
    )
    assert catalog.descriptor(artifact.reference) is artifact


@pytest.mark.asyncio
async def test_remote_gguf_flows_through_managed_install_without_activation(
    tmp_path, monkeypatch
) -> None:
    """Catches remote bytes bypassing managed integrity or becoming active.

    A regression that skips the pinned source map, omits the integrity
    manifest, assigns a consumer, or ignores ``activate=False`` fails this
    real resolve-to-provision flow. Network transport and egress checks are
    isolated at their external boundary; metadata parsing, catalog mapping,
    preflight, consent, fetch, verification, and installation stay real.
    """
    payload = b"GGUF\x00managed-test"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    commit = "c" * 40
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if request.method == "GET" and request.url.path == "/api/models/acme/model":
            return httpx.Response(
                200,
                json=_model_info(
                    [_lfs_file("tiny-model.gguf", size=len(payload), digest=payload_sha256)],
                    commit=commit,
                    card_data=None,
                ),
            )
        if request.url.path == f"/acme/model/resolve/{commit}/tiny-model.gguf":
            if request.method == "HEAD":
                return httpx.Response(200, headers={"etag": '"tiny-v1"'})
            if request.method == "GET":
                return httpx.Response(200, content=payload, headers={"etag": '"tiny-v1"'})
        pytest.fail(f"unexpected network request: {request.method} {request.url}")

    async def allow_mocked_egress(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.acquisition.check_url_or_raise_async",
        allow_mocked_egress,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.fetch.check_url_or_raise_async",
        allow_mocked_egress,
    )
    client_factory = _client_factory(handler)

    resolved = await HuggingFaceRemoteAdapter(client_factory=client_factory).resolve(
        "acme/model"
    )
    catalog = build_remote_catalog(resolved, resolved.candidates[0])
    core = ModelArtifactService(tmp_path / "managed")
    acquisition = ArtifactAcquisitionService(
        core,
        client_factory=client_factory,
        free_bytes_probe=lambda _path: 10**12,
    )

    report = await acquisition.preflight(
        catalog.artifact.reference, catalog, sources=catalog.sources
    )
    assert report.gating_errors == ()
    provisioned = await acquisition.provision(
        report.root,
        report.grant(),
        catalog,
        sources=catalog.sources,
        activate=False,
    )

    artifact = catalog.artifact
    assert provisioned == artifact.reference
    installed_payload = core.artifact_path(artifact.reference) / "model.gguf"
    manifest_path = core.artifact_path(artifact.reference) / "manifest.json"
    assert installed_payload.read_bytes() == payload
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["descriptor"]["reference"] == artifact.reference.to_dict()
    assert manifest["descriptor"]["files"] == [
        {"path": "model.gguf", "size_bytes": len(payload), "sha256": payload_sha256}
    ]
    assert artifact.provenance == (ProvenanceClass.LOCAL_INTEGRITY_RECORDED,)
    assert artifact.consumer == "unassigned"
    assert not core.active_path(artifact.reference.artifact_id).exists()
    assert [(method, httpx.URL(url).path) for method, url in requests] == [
        ("GET", "/api/models/acme/model"),
        ("HEAD", f"/acme/model/resolve/{commit}/tiny-model.gguf"),
        ("GET", f"/acme/model/resolve/{commit}/tiny-model.gguf"),
    ]
