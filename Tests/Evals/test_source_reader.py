"""Reader boundaries: exact evidence, isolated requests and owned timeouts."""

import asyncio
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


def api():
    try:
        return importlib.import_module("tldw_chatbook.Evals.source_reader.reader")
    except ModuleNotFoundError:
        pytest.fail("The source reader has not been implemented", pytrace=False)


def packets(text="α🙂 The launch is Friday. Budget is $40."):
    return (
        SimpleNamespace(
            packet_id="p1", source_id="media:one", revision="1", start=100, text=text
        ),
    )


def envelope(quote="The launch is Friday.", packet_id="p1"):
    return json.dumps(
        {
            "findings": [
                {
                    "statement": "Launch date",
                    "evidence": [{"packet_id": packet_id, "quote": quote}],
                }
            ]
        }
    )


def test_exact_quote_uses_source_unicode_offsets():
    result = api().validate_findings(envelope(), packets())
    evidence = result["findings"][0]["evidence"][0]
    assert (evidence["start"], evidence["end"]) == (103, 124)
    assert evidence["source_id"] == "media:one"
    assert result["claim_support"] == "unchecked"


@pytest.mark.parametrize(
    "raw",
    [envelope("launch on Monday"), envelope(packet_id="foreign"), envelope("yes")],
)
def test_unlocatable_or_ambiguous_evidence_never_becomes_a_finding(raw):
    result = api().validate_findings(raw, packets("yes yes"))
    assert result["status"] == "invalid_worker_output"
    assert result["findings"] == []


@pytest.mark.parametrize(
    "raw",
    [
        '{"findings":[],"findings":[]}',
        '{"findings":[],"extra":1}',
        '```json\n{"findings":[]}\n```',
        '{"findings": NaN}',
        '{"findings":[' + "[" * 30 + "0" + "]" * 30 + "]}",
    ],
)
def test_malformed_json_is_not_repaired(raw):
    assert api().validate_findings(raw, packets())["status"] == "invalid_worker_output"


def test_escaped_lone_surrogate_is_rejected_without_crashing_result_fitting():
    raw = json.loads(envelope())
    raw["findings"][0]["statement"] = "\ud800"
    result = api().validate_findings(json.dumps(raw), packets())
    assert result["status"] == "invalid_worker_output"
    assert result["findings"] == []


@pytest.mark.asyncio
async def test_deadline_cannot_exceed_experiment_ceiling():
    session = api().ReaderSession(object())
    with pytest.raises(ValueError, match="invalid_deadline"):
        await session.complete(object(), deadline=61)
    assert session.pending is None


def test_empty_findings_and_short_source_are_distinct_valid_results():
    assert (
        api().validate_findings('{"findings":[]}', packets())["status"]
        == "no_evidence_found"
    )
    assert api().validate_findings(envelope("42"), packets("42"))["accepted"] == 1


def test_partial_rejection_is_visible_and_keeps_good_evidence():
    raw = json.loads(envelope())
    raw["findings"].append(json.loads(envelope(packet_id="bad"))["findings"][0])
    result = api().validate_findings(json.dumps(raw), packets())
    assert (result["status"], result["accepted"], result["rejected"]) == (
        "partial",
        1,
        1,
    )


def test_output_fits_both_byte_and_character_limits_without_slicing_json():
    result = api().validate_findings(envelope(), packets(), max_chars=300)
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 300
    assert result["status"] == "result_budget_too_small"


def resolution():
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

    return ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="test-model",
        ready=True,
        execution_key="openai",
        temperature=0.2,
    )


def test_request_contains_only_question_packets_and_reader_instruction():
    request = api().build_reader_request(
        resolution(), "When is launch?", packets(), context_limit=32000
    )
    assert request.sensitive is True
    assert [x["role"] for x in request.messages] == ["system", "user"]
    payload = json.loads(request.messages[1]["content"])
    assert payload["question"] == "When is launch?"
    assert payload["packets"][0]["text"] == packets()[0].text
    assert request.max_output_tokens == 2000


def test_overflow_or_invalid_question_stops_request_construction():
    with pytest.raises(ValueError):
        api().build_reader_request(
            resolution(), "question", packets("long" * 1000), context_limit=100
        )
    with pytest.raises(ValueError):
        api().build_reader_request(resolution(), " ", packets(), context_limit=32000)


@pytest.mark.parametrize("caps", [{"context_window": 100}, {"max_input_tokens": 100}])
def test_known_model_limits_override_looser_operator_caps(monkeypatch, caps):
    from tldw_chatbook import model_capabilities

    monkeypatch.setattr(
        model_capabilities,
        "get_model_capabilities",
        lambda: SimpleNamespace(get_model_capabilities=lambda *args: caps),
    )
    with pytest.raises(ValueError, match="input_budget_exceeded"):
        api().build_reader_request(
            resolution(), "question", packets(), context_limit=32000
        )


def test_known_output_cap_reduces_reader_reservation(monkeypatch):
    from tldw_chatbook import model_capabilities

    monkeypatch.setattr(
        model_capabilities,
        "get_model_capabilities",
        lambda: SimpleNamespace(
            get_model_capabilities=lambda *args: {"output_token_limit": 500}
        ),
    )
    request = api().build_reader_request(
        resolution(), "question", packets(), context_limit=32000
    )
    assert request.max_output_tokens == 500


@pytest.mark.asyncio
async def test_deadline_retains_one_request_and_observes_late_usage():
    started, release = asyncio.Event(), asyncio.Event()
    from tldw_chatbook.Chat.console_provider_gateway import AuxiliaryCompletionResult

    class Gateway:
        async def complete_auxiliary(self, request):
            started.set()
            await release.wait()
            return AuxiliaryCompletionResult(
                provider="openai",
                model="test-model",
                text="PRIVATE LATE CONTENT",
                usage=None,
            )

    session = api().ReaderSession(Gateway())
    task = asyncio.create_task(session.complete(object(), deadline=0.02))
    await started.wait()
    assert (await task)["status"] == "timeout"
    assert session.pending is not None and not session.pending.done()
    assert (await session.complete(object()))["status"] == "stopped"
    release.set()
    await session.drain()
    assert session.late_outcome["status"] == "late_completion"
    assert "PRIVATE LATE CONTENT" not in repr(session.late_outcome)


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_owned_provider_task():
    started, release = asyncio.Event(), asyncio.Event()

    class Gateway:
        async def complete_auxiliary(self, request):
            started.set()
            await release.wait()
            raise RuntimeError("PRIVATE PROVIDER ERROR")

    session = api().ReaderSession(Gateway())
    task = asyncio.create_task(session.complete(object()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.pending is not None and not session.pending.cancelled()
    release.set()
    await session.drain()
    assert "PRIVATE PROVIDER ERROR" not in repr(session.late_outcome)


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_worker_only_text_never_reaches_main_request_over_real_http():
    from dataclasses import replace

    from tldw_chatbook.Chat.console_provider_gateway import (
        AuxiliaryCompletionRequest,
        ConsoleProviderGateway,
    )

    wire = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            wire.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            reply = envelope() if len(wire) == 1 else "The launch is Friday."
            body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = ConsoleProviderGateway()
    resolved = replace(
        resolution(),
        provider="llama_cpp",
        execution_key="llama_cpp",
        base_url=f"http://127.0.0.1:{server.server_port}",
    )
    source = packets(
        "WORKER_ONLY_CANARY. The launch is Friday. Irrelevant private detail."
    )
    session = api().ReaderSession(gateway)
    try:
        request = api().build_reader_request(
            resolved, "When?", source, context_limit=32000
        )
        outcome = await session.complete(request)
        assert outcome["status"] == "ok"
        fitted = api().validate_findings(outcome["completion"].text, source)
        main = AuxiliaryCompletionRequest(
            resolution=resolved,
            messages=({"role": "user", "content": json.dumps(fitted)},),
            response_format=None,
            max_output_tokens=100,
        )
        await gateway.complete_auxiliary(main)
        assert len(wire) == 2
        assert "WORKER_ONLY_CANARY" in json.dumps(wire[0]["messages"])
        assert "WORKER_ONLY_CANARY" not in json.dumps(wire[1]["messages"])
        assert "The launch is Friday." in json.dumps(wire[1]["messages"])
        assert all(not {"tools", "tool_choice", "stop"} & body.keys() for body in wire)
    finally:
        await session.drain()
        await gateway.aclose()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(1)


@pytest.mark.asyncio
async def test_real_auxiliary_thread_survives_deadline_with_accounted_late_result():
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    started, release = threading.Event(), threading.Event()

    def adapter(**kwargs):
        started.set()
        assert release.wait(3)
        return {
            "choices": [{"message": {"content": "SOURCE_CANARY"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 2},
        }

    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
    session = api().ReaderSession(gateway)
    request = api().build_reader_request(
        resolution(), "When?", packets(), context_limit=32000
    )
    task = asyncio.create_task(session.complete(request, deadline=0.03))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert (await task)["status"] == "timeout"
        release.set()
        await session.drain()
        assert session.late_outcome["usage"].output == 2
        assert "SOURCE_CANARY" not in repr(session.late_outcome)
    finally:
        release.set()
        await session.drain()
        await gateway.aclose()
