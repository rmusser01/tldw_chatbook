from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.UI.LLM_Management.vllm_connection import (
    VllmActivityEvent,
    VllmConnectionOwner,
    VllmOperationToken,
    VllmProbeRequest,
    VllmProbeResult,
    probe_vllm_target,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmConnectionTarget,
    VllmIssue,
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmReadinessState,
    changed_launch_field_labels,
    launch_snapshot_from_draft,
)

pytestmark = pytest.mark.loopback_network


def local_draft(**changes: object) -> VllmLaunchDraft:
    values = {
        "mode": VllmMode.LOCAL,
        "python_environment": "python",
        "model_source": VllmModelSource.HUGGING_FACE,
        "model_value": "org/model",
    }
    values.update(changes)
    return VllmLaunchDraft(**values)


class _LoopbackVllmServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        self.health_status = 503
        self.models: object = []
        self.models_raw: bytes | None = None
        self.delay_seconds = 0.0
        self.required_authorization: str | None = None
        self.seen_paths: list[str] = []
        self.seen_authorization: list[str | None] = []
        super().__init__(("127.0.0.1", 0), _LoopbackVllmHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"

    def handle_error(self, request, client_address) -> None:
        # Read-timeout tests intentionally close before the delayed response.
        return


class _LoopbackVllmHandler(BaseHTTPRequestHandler):
    server: _LoopbackVllmServer

    def do_GET(self) -> None:
        self.server.seen_paths.append(self.path)
        authorization = self.headers.get("authorization")
        self.server.seen_authorization.append(authorization)
        if self.server.delay_seconds:
            time.sleep(self.server.delay_seconds)
        if (
            self.server.required_authorization is not None
            and authorization != self.server.required_authorization
        ):
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/health":
            self.send_response(self.server.health_status)
            self.end_headers()
            return
        if self.path == "/v1/models":
            body = self.server.models_raw
            if body is None:
                body = json.dumps({"data": self.server.models}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def loopback_vllm():
    server = _LoopbackVllmServer()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def probe_request(
    url: str,
    *,
    token: VllmOperationToken | None = None,
    expected_model_id: str | None = "chatbook-vllm",
    cancelled: Event | None = None,
    process_alive=None,
    timeout: float = 0.5,
) -> VllmProbeRequest:
    return VllmProbeRequest(
        token=token
        or VllmConnectionOwner().begin(local_draft(), runtime_owner="chatbook"),
        api_url=url,
        expected_model_id=expected_model_id,
        cancellation_requested=(cancelled.is_set if cancelled is not None else None),
        process_alive=process_alive,
        connect_timeout_seconds=timeout,
        read_timeout_seconds=timeout,
        total_timeout_seconds=timeout,
    )


def ready_result(token: VllmOperationToken) -> VllmProbeResult:
    return VllmProbeResult(
        token=token,
        state=VllmReadinessState.READY,
        target=VllmConnectionTarget(
            provider_key="vllm",
            api_url="http://127.0.0.1:8000/v1/chat/completions",
            model_id="chatbook-vllm",
            runtime_owner=token.runtime_owner,
            generation=token.generation,
            credential_source="none",
        ),
        issue=None,
        activity=(VllmActivityEvent("ready", "under_1s"),),
    )


def bind_local_claim(
    owner: VllmConnectionOwner, token: VllmOperationToken
) -> ServerLaunchClaim:
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(token, claim)
    return claim


@pytest.mark.asyncio
async def test_ready_requires_health_and_exact_models_identity(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "chatbook-vllm"}]

    result = await probe_vllm_target(probe_request(loopback_vllm.url))

    assert result.state is VllmReadinessState.READY
    assert result.target is not None
    assert result.target.model_id == "chatbook-vllm"
    assert result.target.api_url.endswith("/v1/chat/completions")
    assert loopback_vllm.seen_paths == ["/health", "/v1/models"]


def test_older_generation_cannot_replace_newer_owner_state():
    owner = VllmConnectionOwner()
    old = owner.begin(local_draft(), runtime_owner="chatbook")
    current = owner.begin(replace(local_draft(), port=8001), runtime_owner="chatbook")

    assert owner.settle(old, ready_result(old)) is False
    assert owner.snapshot().generation == current.generation
    assert owner.snapshot().target is None


def test_live_claim_retry_keeps_exact_launch_after_non_network_draft_edit():
    """Catch a Retry binding an old process to the newly edited draft."""

    owner = VllmConnectionOwner()
    launched_draft = local_draft(port=8000)
    edited_draft = replace(launched_draft, port=8001)
    launched = owner.begin(launched_draft, runtime_owner="chatbook")
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(launched, claim)

    owner.invalidate("target_changed")
    retry = owner.begin_claim_retry(claim)

    assert retry is not None
    assert retry.fingerprint == launched.fingerprint
    assert retry.fingerprint != owner.begin(
        edited_draft, runtime_owner="chatbook"
    ).fingerprint

    # Restore the exact live-claim retry after the comparison generation.
    retry = owner.begin_claim_retry(claim)
    assert retry is not None
    snapshot = owner.snapshot()
    assert snapshot.launch_snapshot is not None
    assert snapshot.launch_snapshot.client_api_url == "http://127.0.0.1:8000/v1"
    assert owner.settle(retry, ready_result(retry))
    assert owner.snapshot().target is not None
    assert owner.snapshot().target.api_url == (
        "http://127.0.0.1:8000/v1/chat/completions"
    )


def test_cancelled_live_claim_cannot_begin_retry_generation():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="chatbook")
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(token, claim)
    claim.cancel_event.set()

    assert owner.begin_claim_retry(claim) is None


def test_launch_snapshot_is_immutable_exact_and_changed_labels_are_allowlisted():
    """A restart comparison must retain launch truth without copying values."""

    draft = local_draft(
        python_environment="/private/PATH_CANARY/bin/python",
        model_source=VllmModelSource.LOCAL_DIRECTORY,
        model_value="/private/MODEL_CANARY",
        raw_arguments="--adapter COMMAND_CANARY",
    )
    snapshot = launch_snapshot_from_draft(
        draft,
        generation=4,
        profile_id="d34adf77-02ce-4a28-8bcd-85c66bb193ec",
        profile_name="Local GPU",
    )

    assert snapshot.generation == 4
    assert snapshot.profile_id == "d34adf77-02ce-4a28-8bcd-85c66bb193ec"
    assert snapshot.environment_display == "/private/PATH_CANARY/bin/python"
    assert snapshot.model_source_kind is VllmModelSource.LOCAL_DIRECTORY
    assert snapshot.model_source_display == "/private/MODEL_CANARY"
    assert snapshot.redacted_argument_summary == "Custom launch arguments"
    with pytest.raises(FrozenInstanceError):
        snapshot.port = 9000  # type: ignore[misc]

    changed = changed_launch_field_labels(
        snapshot,
        replace(
            draft,
            model_value="/private/OTHER_MODEL_CANARY",
            port=8001,
            raw_arguments="--other-secret-shaped-value",
        ),
    )
    assert changed == ("Model", "Port", "Advanced arguments")
    rendered = repr(changed)
    assert "PATH_CANARY" not in rendered
    assert "MODEL_CANARY" not in rendered
    assert "COMMAND_CANARY" not in rendered


def test_ready_result_requires_a_target():
    token = VllmConnectionOwner().begin(local_draft(), runtime_owner="chatbook")

    with pytest.raises(ValueError, match="ready result"):
        VllmProbeResult(
            token=token,
            state=VllmReadinessState.READY,
            target=None,
            issue=None,
        )


def test_chatbook_owned_ready_result_requires_exact_served_alias():
    token = VllmConnectionOwner().begin(local_draft(), runtime_owner="chatbook")

    with pytest.raises(ValueError, match="target"):
        VllmProbeResult(
            token=token,
            state=VllmReadinessState.READY,
            target=VllmConnectionTarget(
                provider_key="vllm",
                api_url="http://127.0.0.1:8000/v1/chat/completions",
                model_id="other-model",
                runtime_owner="chatbook",
                generation=token.generation,
                credential_source="none",
            ),
            issue=None,
        )


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://user:CREDENTIAL_CANARY@127.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions?api_key=CREDENTIAL_CANARY",
        "http://127.0.0.1:8000/v1",
    ],
)
def test_ready_result_requires_canonical_credential_free_target(unsafe_url):
    token = VllmConnectionOwner().begin(local_draft(), runtime_owner="chatbook")

    with pytest.raises(ValueError, match="target"):
        VllmProbeResult(
            token=token,
            state=VllmReadinessState.READY,
            target=VllmConnectionTarget(
                provider_key="vllm",
                api_url=unsafe_url,
                model_id="chatbook-vllm",
                runtime_owner="chatbook",
                generation=token.generation,
                credential_source="none",
            ),
            issue=None,
        )


def test_owner_settlement_revalidates_a_mutated_result_fail_closed():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="chatbook")
    result = ready_result(token)
    assert result.target is not None
    object.__setattr__(
        result,
        "target",
        replace(
            result.target,
            api_url=(
                "http://user:CREDENTIAL_CANARY@127.0.0.1:8000/"
                "v1/chat/completions"
            ),
        ),
    )

    assert owner.settle(token, result) is False
    assert owner.snapshot().target is None


def test_owned_settlement_rejects_canonical_target_for_different_launch_endpoint():
    """Catch a safe canonical target being attributed to the wrong process."""

    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(port=8000), runtime_owner="chatbook")
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(token, claim)
    result = ready_result(token)
    assert result.target is not None
    object.__setattr__(
        result,
        "target",
        replace(
            result.target,
            api_url="http://127.0.0.1:8001/v1/chat/completions",
        ),
    )

    assert owner.settle(token, result) is False
    snapshot = owner.snapshot()
    assert snapshot.state is VllmReadinessState.CHECKING
    assert snapshot.target is None


def test_owned_settlement_requires_token_bound_to_the_launch_claim():
    owner = VllmConnectionOwner()
    launched = owner.begin(local_draft(), runtime_owner="chatbook")
    bind_local_claim(owner, launched)
    unbound_check = owner.begin(local_draft(), runtime_owner="chatbook")

    assert owner.settle(unbound_check, ready_result(unbound_check)) is False
    assert owner.snapshot().target is None


def test_external_settlement_accepts_its_canonical_probed_target_without_claim():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="external")
    result = VllmProbeResult(
        token=token,
        state=VllmReadinessState.READY,
        target=VllmConnectionTarget(
            provider_key="vllm",
            api_url="https://models.example.test/v1/chat/completions",
            model_id="organization/model",
            runtime_owner="external",
            generation=token.generation,
            credential_source="configured",
        ),
        issue=None,
        activity=(VllmActivityEvent("ready", "under_1s"),),
    )

    assert owner.settle(token, result)
    assert owner.snapshot().target == result.target


def test_owner_settlement_rejects_wrong_target_type_without_raising():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="chatbook")
    result = ready_result(token)
    object.__setattr__(result, "target", object())

    assert owner.settle(token, result) is False
    assert owner.snapshot().target is None


@pytest.mark.asyncio
async def test_probe_timeout_is_bounded_and_sanitized(loopback_vllm):
    loopback_vllm.delay_seconds = 0.2
    started = time.monotonic()

    result = await probe_vllm_target(probe_request(loopback_vllm.url, timeout=0.03))

    assert time.monotonic() - started < 0.5
    assert result.issue == VllmIssue("health_timeout", "connection")
    assert result.target is None


@pytest.mark.asyncio
async def test_auth_required_never_echoes_response_or_credential(loopback_vllm):
    loopback_vllm.required_authorization = "Bearer CREDENTIAL_CANARY"

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url), credential_resolver=lambda: (None, "none")
    )

    assert result.issue == VllmIssue("credential_required", "connection")
    assert "CREDENTIAL_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_configured_authorization_is_used_without_entering_result(loopback_vllm):
    loopback_vllm.required_authorization = "Bearer CREDENTIAL_CANARY"
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "chatbook-vllm"}]

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url),
        credential_resolver=lambda: ("CREDENTIAL_CANARY", "configured"),
    )

    assert result.target is not None
    assert result.target.credential_source == "configured"
    assert loopback_vllm.seen_authorization == [
        "Bearer CREDENTIAL_CANARY",
        "Bearer CREDENTIAL_CANARY",
    ]
    assert "CREDENTIAL_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_healthy_api_without_exact_model_is_not_ready(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "other-model"}]

    result = await probe_vllm_target(probe_request(loopback_vllm.url))

    assert result.state is VllmReadinessState.NEEDS_ATTENTION
    assert result.issue == VllmIssue("model_missing", "model")
    assert result.target is None


@pytest.mark.asyncio
async def test_malformed_models_json_is_bounded_failure(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models_raw = b'{"data": [RESPONSE_CANARY'

    result = await probe_vllm_target(probe_request(loopback_vllm.url))

    assert result.issue == VllmIssue("invalid_models_response", "connection")
    assert "RESPONSE_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_invalid_endpoint_is_a_sanitized_failure():
    result = await probe_vllm_target(probe_request("file:///private/PATH_CANARY"))

    assert result.issue == VllmIssue("invalid_endpoint", "connection")
    assert "PATH_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_oversized_models_response_is_rejected_without_retention(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models_raw = b'{"data":"' + (b"RESPONSE_CANARY" * 5000) + b'"}'

    result = await probe_vllm_target(probe_request(loopback_vllm.url))

    assert result.issue == VllmIssue("invalid_models_response", "connection")
    assert "RESPONSE_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_process_exit_prevents_any_http_probe(loopback_vllm):
    result = await probe_vllm_target(
        probe_request(loopback_vllm.url, process_alive=lambda: False)
    )

    assert result.issue == VllmIssue("process_exited", "process")
    assert loopback_vllm.seen_paths == []


@pytest.mark.asyncio
async def test_process_exit_during_probe_prevents_ready_publication(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "chatbook-vllm"}]
    liveness = iter((True, True, False))

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url, process_alive=lambda: next(liveness))
    )

    assert result.issue == VllmIssue("process_exited", "process")
    assert result.target is None


@pytest.mark.asyncio
async def test_cancellation_prevents_any_http_probe(loopback_vllm):
    cancelled = Event()
    cancelled.set()

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url, cancelled=cancelled)
    )

    assert result.issue == VllmIssue("cancelled", "connection")
    assert loopback_vllm.seen_paths == []


@pytest.mark.parametrize(
    "rejected",
    [
        "/private/model",
        "../model",
        "~/model",
        r"C:\\models\\model",
        r"owner\\model",
        "model.gguf",
        " owner/model ",
        "owner//../model",
        "bad\N{RIGHT-TO-LEFT OVERRIDE}model",
        "x" * 121,
    ],
)
@pytest.mark.asyncio
async def test_existing_server_rejects_path_like_or_noncanonical_model_ids(
    loopback_vllm, rejected
):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": rejected}]
    token = VllmConnectionOwner().begin(local_draft(), runtime_owner="external")

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url, token=token, expected_model_id=None)
    )

    assert result.issue == VllmIssue("model_missing", "model")
    assert rejected not in repr(result)


@pytest.mark.asyncio
async def test_existing_server_accepts_namespace_model_id(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "organization/model"}]
    token = VllmConnectionOwner().begin(local_draft(), runtime_owner="external")

    result = await probe_vllm_target(
        probe_request(loopback_vllm.url, token=token, expected_model_id=None)
    )

    assert result.target is not None
    assert result.target.model_id == "organization/model"


def test_owner_keeps_only_current_operation_bounded_allowlisted_activity():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="chatbook")
    for _ in range(40):
        result = VllmProbeResult(
            token=token,
            state=VllmReadinessState.LOADING_MODEL,
            target=None,
            issue=None,
            activity=(VllmActivityEvent("health_checking", "under_1s"),),
        )
        assert owner.settle(token, result)

    snapshot = owner.snapshot()
    assert len(snapshot.activity) == 32
    assert {event.code for event in snapshot.activity} == {
        "health_checking",
    }


def test_owner_snapshot_excludes_launch_privacy_canaries(caplog):
    owner = VllmConnectionOwner()
    draft = local_draft(
        python_environment="/private/PATH_CANARY/bin/python",
        model_source=VllmModelSource.LOCAL_DIRECTORY,
        model_value="/private/MODEL_SOURCE_CANARY",
        raw_arguments="--flag COMMAND_CANARY",
    )
    token = owner.begin(draft, runtime_owner="chatbook")
    bind_local_claim(owner, token)
    assert owner.settle(token, ready_result(token))

    visible = repr(owner.snapshot()) + repr(owner.snapshot().activity) + caplog.text
    for canary in (
        "PATH_CANARY",
        "MODEL_SOURCE_CANARY",
        "COMMAND_CANARY",
        "CREDENTIAL_CANARY",
        "RESPONSE_CANARY",
    ):
        assert canary not in visible


def test_invalidate_advances_generation_and_clears_ready_target():
    owner = VllmConnectionOwner()
    token = owner.begin(local_draft(), runtime_owner="chatbook")
    bind_local_claim(owner, token)
    assert owner.settle(token, ready_result(token))

    generation = owner.invalidate("target_changed")

    snapshot = owner.snapshot()
    assert generation == token.generation + 1
    assert snapshot.generation == generation
    assert snapshot.target is None
    assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
    assert snapshot.activity[-1].code == "target_changed"
