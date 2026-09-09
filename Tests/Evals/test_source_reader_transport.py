"""Qualify the runnable experiment at the actual OpenAI-compatible HTTP edge."""

import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from loguru import logger


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.parametrize("response_mode", ["ok", "error", "missing_output_usage"])
@pytest.mark.parametrize("provider", ["openai", "deepseek"])
async def test_runner_real_transport_is_isolated_and_does_not_retry(
    tmp_path, monkeypatch, response_mode, provider
):
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment

    received, logs = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            data = json.loads(payload["messages"][-1]["content"])
            response_text = "Answer from fixture evidence."
            if "packets" in data:
                packet = data["packets"][0]
                response_text = json.dumps(
                    {
                        "findings": [
                            {
                                "statement": "Candidate fixture fact",
                                "evidence": [
                                    {
                                        "packet_id": packet["packet_id"],
                                        "quote": packet["text"].split(".")[0] + ".",
                                    }
                                ],
                            }
                        ]
                    }
                )
            usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            if provider == "deepseek":
                usage.update(prompt_cache_hit_tokens=40, prompt_cache_miss_tokens=60)
            if response_mode == "missing_output_usage":
                usage = {"prompt_tokens": 100}
            body = json.dumps(
                {
                    "choices": [{"message": {"content": response_text}}],
                    "usage": usage,
                }
            ).encode()
            self.send_response(503 if response_mode == "error" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    prepared = tmp_path / "prepared"
    prepare(prepared)
    monkeypatch.setenv("SOURCE_READER_TEST_KEY", "SOURCE_READER_SECRET_CANARY")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sink = logger.add(lambda message: logs.append(str(message)), level="DEBUG")
    try:
        attempts = await run_experiment(
            Namespace(
                provider=provider,
                prepared=prepared,
                output=tmp_path / "run",
                main_model="deepseek-v4-pro" if provider == "deepseek" else "gpt-4o",
                worker_model="deepseek-v4-flash"
                if provider == "deepseek"
                else "gpt-4o-mini",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="SOURCE_READER_TEST_KEY",
                request_cap=100,
                token_cap=3000000,
                spend_cap_usd=100.0,
                context_limit=32768,
                input_limit=24000,
                output_limit=512,
                split="development",
            )
        )
    finally:
        logger.remove(sink)
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(received) == (24 if response_mode == "ok" else 1)
    assert all(path == "/v1/chat/completions" for path, _ in received)
    for _, payload in received:
        assert [row["role"] for row in payload["messages"]] == ["system", "user"]
        assert not {"tools", "tool_choice", "stop"}.intersection(payload)
        assert payload["stream"] is False
        if provider == "deepseek":
            assert payload["thinking"] == {"type": "disabled"}
    if response_mode == "error":
        assert any(row["status"] == "provider_error" for row in attempts)
    elif response_mode == "missing_output_usage":
        assert any(row["status"] == "unknown_usage" for row in attempts)
    else:
        assert all(
            row["status"] == "ok" for row in attempts if row["arm"] != "retrieval"
        )
        assert all(
            row["cost_usd"] is not None for row in attempts if row["arm"] != "retrieval"
        )
        if provider == "deepseek":
            calls = [call for row in attempts for call in row["calls"]]
            assert all(call["usage"]["provider"] == "deepseek" for call in calls)
            assert all(call["usage"]["cache_read"] == 40 for call in calls)
    captured = "".join(logs)
    assert "SOURCE_READER_SECRET_CANARY" not in captured
    assert "practice garden opens" not in captured
