"""Focused evidence for the opt-in bulk-reader comparison tool."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_bulk_reader.py"
TEST_PYTHON = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python")


def _load_evaluator():
    assert SCRIPT_PATH.is_file(), "bulk-reader evaluator has not been implemented"
    spec = importlib.util.spec_from_file_location("bulk_reader_evaluator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fence(name: str, args: dict) -> str:
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _corpus(path: Path, *, source_path: str = "notes/policy.txt") -> Path:
    payload = {
        "schema_version": 1,
        "id": "bulk-reader-test-corpus",
        "description": "A minimal pinned synthetic comparison corpus.",
        "rubric": {
            "method": "manual",
            "criteria": ["fact coverage", "citation accuracy", "absence honesty"],
        },
        "cases": [
            {
                "id": "renewal-exception",
                "category": "repository",
                "question": "What is the renewal period and its exception?",
                "sources": {
                    source_path: (
                        "Standard renewal is 30 days.\n"
                        "Exception: lunar accounts renew after 45 days.\n"
                    )
                },
                "expected_facts": [
                    "Standard renewal is 30 days.",
                    "Lunar accounts renew after 45 days.",
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _RecordingGateway:
    """Deterministic provider boundary; the runtime, tools and DB stay real."""

    def __init__(
        self,
        *,
        ignore_delegation: bool = False,
        fail_direct: bool = False,
        ad_hoc_delegation: bool = False,
        omit_usage_field: str | None = None,
        loop_direct: bool = False,
        oversized_final: bool = False,
        direct_skips_read: bool = False,
        worker_skips_read: bool = False,
        parent_rereads_after_worker: bool = False,
        length_on_final: bool = False,
    ):
        self.ignore_delegation = ignore_delegation
        self.fail_direct = fail_direct
        self.ad_hoc_delegation = ad_hoc_delegation
        self.omit_usage_field = omit_usage_field
        self.loop_direct = loop_direct
        self.oversized_final = oversized_final
        self.direct_skips_read = direct_skips_read
        self.worker_skips_read = worker_skips_read
        self.parent_rereads_after_worker = parent_rereads_after_worker
        self.length_on_final = length_on_final
        self.calls: list[dict] = []

    async def stream_chat(self, resolution, messages, *, signals=None, **_kwargs):
        from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT
        from tldw_chatbook.Chat.console_provider_gateway import (
            ProviderToolCalls,
            ProviderTurnMetadata,
        )

        rows = list(messages)
        system = str(rows[0].get("content", "")) if rows else ""
        is_worker = system.startswith(SUBAGENT_SYSTEM_PROMPT)
        has_read_result = any(
            str(row.get("content", "")).startswith("Tool result for fs_read:")
            for row in rows
        )
        has_spawn_result = any(
            str(row.get("content", "")).startswith("Tool result for spawn_subagent:")
            for row in rows
        )
        offered_spawn = "spawn_subagent" in system
        self.calls.append(
            {
                "model": resolution.model,
                "is_worker": is_worker,
                "messages": rows,
            }
        )

        if self.fail_direct and not offered_spawn and not is_worker:
            raise RuntimeError("synthetic provider failure")
        if is_worker and self.worker_skips_read:
            reply = "Worker answered without opening a source."
        elif is_worker and not has_read_result:
            reply = _fence("fs_read", {"path": "notes/policy.txt"})
        elif is_worker:
            reply = "Worker: 30 days; lunar exception is 45 days (policy.txt:1-2)."
        elif offered_spawn and not has_spawn_result:
            if self.ignore_delegation:
                reply = "I chose not to delegate and answered directly."
            else:
                spawn_args = {
                    "task": (
                        "Answer the supplied question using only these paths: "
                        "notes/policy.txt"
                    )
                }
                if not self.ad_hoc_delegation:
                    spawn_args["agent"] = "bulk-reader"
                reply = _fence(
                    "spawn_subagent",
                    spawn_args,
                )
        elif offered_spawn and self.parent_rereads_after_worker and not has_read_result:
            reply = _fence("fs_read", {"path": "notes/policy.txt"})
        elif offered_spawn:
            reply = "Delegated answer: 30 days, except lunar accounts use 45."
        elif self.direct_skips_read:
            reply = "Direct answer without opening a source."
        elif not has_read_result or self.loop_direct:
            reply = _fence(
                "fs_read",
                {"path": "notes/policy.txt", "offset": len(self.calls)},
            )
        else:
            reply = "Direct answer: 30 days, except lunar accounts use 45."

        if self.oversized_final and not reply.startswith("```tool_call"):
            reply = "x" * 20_000

        yield reply
        usage = (
            {
                "prompt_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens": 4,
            }
            if resolution.model == "worker-test"
            else {
                "prompt_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens": 6,
            }
        )
        usage.pop(self.omit_usage_field, None)
        if signals is not None:
            signals.record_usage_payload(usage)
        yield ProviderToolCalls(
            (),
            metadata=ProviderTurnMetadata(
                finish_reason=(
                    "length"
                    if self.length_on_final and not reply.startswith("```tool_call")
                    else "stop"
                ),
                usage=usage,
            ),
        )
        if signals is not None:
            signals.close_usage_call()


class _MetadataCapturingGateway:
    """Test-only bridge from deterministic sentinels to the evaluator recorder."""

    def __init__(self, gateway, recorder):
        self.gateway = gateway
        self.recorder = recorder

    async def stream_chat(self, *args, **kwargs):
        recorded = False
        try:
            async for item in self.gateway.stream_chat(*args, **kwargs):
                metadata = getattr(item, "metadata", None)
                if metadata is not None:
                    self.recorder.record(metadata)
                    recorded = True
                yield item
        finally:
            if not recorded:
                self.recorder.record(None)

    def __getattr__(self, name):
        return getattr(self.gateway, name)


def _pricing_catalog(*, worker_cache_rate=0.5, provider="openai"):
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    return PricingCatalog(
        config={
            "models": {
                f"{provider}:main-test": {
                    "input_per_mtok": 2.0,
                    "output_per_mtok": 8.0,
                    "cache_read_per_mtok": 0.25,
                    "cache_write_per_mtok": 2.5,
                    "as_of": "2026-09-07",
                },
                f"{provider}:worker-test": {
                    "input_per_mtok": 1.0,
                    "output_per_mtok": 4.0,
                    "cache_read_per_mtok": worker_cache_rate,
                    "cache_write_per_mtok": 1.25,
                    "as_of": "2026-09-07",
                },
            }
        }
    )


def _resolution():
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

    return ConsoleProviderResolution(
        provider="OpenAI",
        base_url="https://example.invalid/v1",
        model="main-test",
        ready=True,
        readiness_key="openai",
        execution_key="openai",
        streaming=True,
        max_tokens=2_048,
        request_timeout=60.0,
        request_retries=0,
        request_retry_delay=0.0,
    )


def _run_evaluation(tmp_path: Path, gateway, *, pricing_catalog=None):
    evaluator = _load_evaluator()
    metadata = evaluator._ProviderMetadataRecorder()
    recording_gateway = _MetadataCapturingGateway(gateway, metadata)
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = _corpus(tmp_path / "corpus.json")
    output = tmp_path / "report.json"
    report = asyncio.run(
        evaluator.evaluate_comparison(
            corpus_path=corpus,
            output_path=output,
            provider="OpenAI",
            main_model="main-test",
            worker_model="worker-test",
            gateway=recording_gateway,
            resolution=_resolution(),
            pricing_catalog=pricing_catalog or _pricing_catalog(),
            provider_metadata=metadata,
        )
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    return evaluator, report


def test_real_runtime_comparison_records_delegation_reads_models_and_worker_cost(
    tmp_path,
):
    gateway = _RecordingGateway()
    evaluator, report = _run_evaluation(tmp_path, gateway)

    assert report["status"] == "complete"
    assert report["quality_review"] == {
        "status": "pending",
        "rubric": {
            "method": "manual",
            "criteria": [
                "fact coverage",
                "citation accuracy",
                "absence honesty",
            ],
        },
    }
    case = report["cases"][0]
    direct = case["direct"]
    delegated = case["delegated"]
    assert direct["status"] == delegated["status"] == "completed"
    assert direct["delegation_occurred"] is False
    assert delegated["delegation_occurred"] is True
    assert delegated["child_runs"][0]["agent_definition"] == "bulk-reader"
    assert delegated["child_runs"][0]["status"] == "done"
    assert delegated["worker_output"].startswith("Worker: 30 days")

    assert {call["model"] for call in direct["calls"]} == {"main-test"}
    assert {call["model"] for call in delegated["calls"]} == {
        "main-test",
        "worker-test",
    }
    worker_calls = [
        call for call in delegated["calls"] if call["model"] == "worker-test"
    ]
    assert len(worker_calls) == 2
    assert worker_calls[0]["usage"] == {
        "uncached_input": 15,
        "cache_read": 5,
        "cache_write": 0,
        "output": 4,
        "audio_input": 0,
        "audio_output": 0,
        "transcription_seconds": 0.0,
        "provider": "OpenAI",
        "model": "worker-test",
        "partial": False,
    }
    assert all(call["cost_status"] == "known" for call in delegated["calls"])
    assert delegated["total_cost_usd"] == round(
        sum(call["cost"]["total"] for call in delegated["calls"]), 6
    )

    assert direct["tool_reads"] == [
        {"run_kind": "primary", "tool": "fs_read", "path": "notes/policy.txt"}
    ]
    assert delegated["tool_reads"] == [
        {
            "run_kind": "subagent",
            "tool": "fs_read",
            "path": "notes/policy.txt",
        }
    ]
    assert case["sources"][0]["sha256"]
    assert report["limits"] == evaluator.REPORT_LIMITS
    assert report["execution_order"] == ["direct", "delegated"]
    direct_input = gateway.calls[0]["messages"][1]["content"]
    delegated_input = gateway.calls[2]["messages"][1]["content"]
    assert direct_input == delegated_input
    assert "Standard renewal is 30 days" not in direct_input


def test_missing_or_unknown_per_call_accounting_keeps_total_cost_unknown(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(),
        pricing_catalog=_pricing_catalog(worker_cache_rate=None),
    )

    delegated = report["cases"][0]["delegated"]
    worker_calls = [
        call for call in delegated["calls"] if call["model"] == "worker-test"
    ]
    assert {call["cost_status"] for call in worker_calls} == {"unknown"}
    assert all(call["cost"] is None for call in worker_calls)
    assert delegated["total_cost_usd"] is None
    assert delegated["cost_status"] == "unknown"


@pytest.mark.parametrize("omitted", ["prompt_tokens", "completion_tokens"])
def test_incomplete_raw_usage_is_partial_and_never_priced(tmp_path, omitted):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(omit_usage_field=omitted),
    )

    calls = report["cases"][0]["direct"]["calls"]
    assert calls
    assert all(call["usage"]["partial"] is True for call in calls)
    assert all(call["cost_status"] == "unknown" for call in calls)
    assert report["cases"][0]["direct"]["total_cost_usd"] is None


def test_failed_and_nondelegating_arms_are_not_labeled_successful(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(ignore_delegation=True, fail_direct=True),
    )

    case = report["cases"][0]
    assert report["status"] == "incomplete"
    assert case["direct"]["status"] == "failed"
    assert case["direct"]["runtime_status"] == "error"
    assert case["direct"]["calls"][0]["status"] == "failed"
    assert case["direct"]["calls"][0]["usage"] is None
    assert case["delegated"]["runtime_status"] == "done"
    assert case["delegated"]["status"] == "non_delegating"
    assert case["delegated"]["delegation_occurred"] is False
    assert case["delegated"]["child_runs"] == []


def test_ad_hoc_child_does_not_count_as_named_bulk_reader_delegation(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(ad_hoc_delegation=True),
    )

    delegated = report["cases"][0]["delegated"]
    assert delegated["runtime_status"] == "done"
    assert delegated["status"] == "non_delegating"
    assert delegated["delegation_occurred"] is False
    assert delegated["child_runs"][0]["agent_definition"] is None
    assert {call["model"] for call in delegated["calls"]} == {"main-test"}


def test_direct_answer_without_content_read_is_incomplete(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(direct_skips_read=True),
    )

    direct = report["cases"][0]["direct"]
    assert direct["runtime_status"] == "done"
    assert direct["status"] == "incomplete"
    assert direct["status_reasons"] == ["direct_content_read_missing"]
    assert direct["tool_reads"] == []


def test_parent_reread_does_not_hide_named_worker_without_content_read(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(
            worker_skips_read=True,
            parent_rereads_after_worker=True,
        ),
    )

    delegated = report["cases"][0]["delegated"]
    assert delegated["runtime_status"] == "done"
    assert delegated["delegation_occurred"] is True
    assert delegated["status"] == "incomplete"
    assert delegated["status_reasons"] == ["bulk_reader_content_read_missing"]
    assert delegated["tool_reads"] == [
        {"run_kind": "primary", "tool": "fs_read", "path": "notes/policy.txt"}
    ]


def test_short_provider_length_finish_marks_arms_incomplete_with_known_cost(tmp_path):
    _evaluator, report = _run_evaluation(
        tmp_path,
        _RecordingGateway(length_on_final=True),
    )

    for arm_name in ("direct", "delegated"):
        arm = report["cases"][0][arm_name]
        assert arm["status"] == "incomplete"
        assert "provider_output_token_limit" in arm["status_reasons"]
        limited = [call for call in arm["calls"] if call["provider_output_limited"]]
        assert limited
        assert all(call["finish_reason"] == "length" for call in limited)
        assert all(call["status"] == "incomplete" for call in limited)
        assert all(call["cost_status"] == "known" for call in limited)
        assert all(call["cost"] is not None for call in limited)
        assert all(call["output_truncated"] is False for call in limited)


def test_real_console_gateway_fence_path_exposes_length_to_call_recorder():
    evaluator = _load_evaluator()
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn

    usage = {"prompt_tokens": 12, "completion_tokens": 3}

    def deterministic_provider(**_kwargs):
        text = "A short provider-limited answer."
        turn = HostedChatTurn(
            text=text,
            tool_calls=(),
            assistant_message={"role": "assistant", "content": text},
            finish_reason="length",
            usage=usage,
        )

        class DeterministicStream:
            def __init__(self):
                self.items = iter(
                    [
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": "length",
                                }
                            ],
                            "usage": usage,
                        }
                    ]
                )

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.items)

            @property
            def terminal_turn(self):
                return turn

            @property
            def provider_continuation(self):
                return None

            def close(self) -> None:
                return None

        return DeterministicStream()

    async def exercise():
        metadata = evaluator._ProviderMetadataRecorder()
        gateway = ConsoleProviderGateway(
            chat_api_call_fn=metadata.wrap_chat_api_call(deterministic_provider),
            environ={},
        )
        resolution = ConsoleProviderResolution(
            provider="moonshot",
            base_url="https://example.invalid/v1",
            model="main-test",
            ready=True,
            readiness_key="moonshot",
            execution_key="moonshot",
            api_key="secret",
            streaming=True,
            max_tokens=2_048,
            request_timeout=60.0,
            request_retries=0,
            request_retry_delay=0.0,
        )

        class Store:
            def append_stream_chunk(self, _message_id, _chunk) -> None:
                return None

            def reset_stream_content(self, _message_id) -> None:
                return None

        signals = ConsoleProviderStreamSignals()
        adapter = _StreamingModelAdapter(
            store=Store(),
            provider_gateway=gateway,
            resolution=resolution,
            assistant_message_id="assistant",
            should_cancel=lambda: False,
            loop=asyncio.get_running_loop(),
            native_tools=False,
            provider_stream_signals=signals,
        )
        recorded = evaluator._RecordingChatCall(
            adapter,
            resolution,
            signals,
            _pricing_catalog(provider="moonshot"),
            metadata,
        )
        try:
            response = await asyncio.to_thread(
                recorded,
                messages_payload=[
                    {"role": "system", "content": "Read before answering."},
                    {"role": "user", "content": "Question"},
                ],
                model="main-test",
                api_endpoint="moonshot",
            )
        finally:
            await gateway.aclose()
        return response, recorded.calls

    response, calls = asyncio.run(exercise())
    assert response["choices"][0]["message"]["content"] == (
        "A short provider-limited answer."
    )
    assert calls[0]["finish_reason"] == "length"
    assert calls[0]["provider_output_limited"] is True
    assert calls[0]["status"] == "incomplete"
    assert calls[0]["cost_status"] == "known"


def test_model_turn_and_recorded_output_caps_make_arms_incomplete(tmp_path):
    evaluator, looped = _run_evaluation(
        tmp_path / "looped",
        _RecordingGateway(loop_direct=True),
    )
    direct = looped["cases"][0]["direct"]
    assert len(direct["calls"]) == evaluator.MAX_PROVIDER_CALLS_PER_ARM
    assert direct["runtime_status"] == "stuck"
    assert direct["status"] == "incomplete"

    evaluator, oversized = _run_evaluation(
        tmp_path / "oversized",
        _RecordingGateway(oversized_final=True),
    )
    direct = oversized["cases"][0]["direct"]
    assert len(direct["answer"]) == evaluator.MAX_RECORDED_OUTPUT_CHARS
    assert direct["calls"][-1]["output_truncated"] is True
    assert direct["status"] == "incomplete"


def test_process_agent_overrides_are_restored(tmp_path, monkeypatch):
    names = {
        "TLDW_AGENTS_MAX_LIVE_SUBAGENTS": "7",
        "TLDW_AGENTS_RUN_LOG_ENABLED": "true",
        "TLDW_AGENTS_RUN_LOG_EVICT_ENABLED": "true",
    }
    for name, value in names.items():
        monkeypatch.setenv(name, value)

    _run_evaluation(tmp_path, _RecordingGateway())

    assert {name: __import__("os").environ[name] for name in names} == names


def test_checked_in_corpus_is_pinned_and_loadable():
    evaluator = _load_evaluator()
    corpus, digest = evaluator._load_corpus(evaluator.DEFAULT_CORPUS)

    assert corpus["id"] == "bulk-reader-pilot-v1"
    assert len(corpus["cases"]) == 4
    assert len(digest) == 64


@pytest.mark.parametrize("unsafe_path", ["../secret.txt", "/tmp/secret.txt"])
def test_corpus_refuses_paths_outside_materialized_workspace(tmp_path, unsafe_path):
    evaluator = _load_evaluator()
    corpus = _corpus(tmp_path / "corpus.json", source_path=unsafe_path)
    gateway = _RecordingGateway()

    with pytest.raises(ValueError, match="relative confined path"):
        asyncio.run(
            evaluator.evaluate_comparison(
                corpus_path=corpus,
                output_path=tmp_path / "report.json",
                provider="OpenAI",
                main_model="main-test",
                worker_model="worker-test",
                gateway=gateway,
                resolution=_resolution(),
                pricing_catalog=_pricing_catalog(),
                provider_metadata=evaluator._ProviderMetadataRecorder(),
            )
        )
    assert gateway.calls == []


def test_materializer_refuses_symlink_ancestors(tmp_path):
    evaluator = _load_evaluator()
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "notes").symlink_to(outside, target_is_directory=True)
    case = json.loads(_corpus(tmp_path / "corpus.json").read_text())["cases"][0]

    with pytest.raises(ValueError, match="symlink"):
        evaluator.materialize_case(case, root)
    assert list(outside.iterdir()) == []


def test_existing_output_refuses_before_provider_calls(tmp_path):
    evaluator = _load_evaluator()
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    gateway = _RecordingGateway()

    with pytest.raises(FileExistsError, match="already exists"):
        asyncio.run(
            evaluator.evaluate_comparison(
                corpus_path=_corpus(tmp_path / "corpus.json"),
                output_path=output,
                provider="OpenAI",
                main_model="main-test",
                worker_model="worker-test",
                gateway=gateway,
                resolution=_resolution(),
                pricing_catalog=_pricing_catalog(),
                provider_metadata=evaluator._ProviderMetadataRecorder(),
            )
        )
    assert output.read_text(encoding="utf-8") == "keep"
    assert gateway.calls == []


def test_cli_help_and_billable_refusal_do_not_import_application(tmp_path):
    probe = (
        "import runpy,sys; "
        f"sys.argv=[{str(SCRIPT_PATH)!r},*sys.argv[1:]]; "
        "code=0; "
        "\ntry: runpy.run_path(sys.argv[0],run_name='__main__')"
        "\nexcept SystemExit as exc: code=int(exc.code or 0)"
        "\nprint('APP_IMPORTED='+str(any(k=='tldw_chatbook' or "
        "k.startswith('tldw_chatbook.') for k in sys.modules)))"
        "\nraise SystemExit(code)"
    )
    help_run = subprocess.run(
        [str(TEST_PYTHON), "-c", probe, "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "--confirm-billable" in help_run.stdout
    assert "--corpus" not in help_run.stdout
    assert "APP_IMPORTED=False" in help_run.stdout

    refusal = subprocess.run(
        [
            str(TEST_PYTHON),
            "-c",
            probe,
            "--provider",
            "Moonshot",
            "--main-model",
            "main-test",
            "--worker-model",
            "worker-test",
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refusal.returncode == 2
    assert "Refusing provider calls without --confirm-billable" in refusal.stderr
    assert "APP_IMPORTED=False" in refusal.stdout

    unsupported = subprocess.run(
        [
            str(TEST_PYTHON),
            "-c",
            probe,
            "--provider",
            "OpenAI",
            "--main-model",
            "main-test",
            "--worker-model",
            "worker-test",
            "--output",
            str(tmp_path / "report.json"),
            "--confirm-billable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert unsupported.returncode == 2
    assert "invalid choice" in unsupported.stderr
    assert "APP_IMPORTED=False" in unsupported.stdout
