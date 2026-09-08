"""Compare direct reading with the named bulk-reader on a synthetic corpus.

Live execution is intentionally limited to Moonshot and ZAI because those are
the Console gateway routes whose timeout and retry policy can be pinned by this
tool. Provider calls require ``--confirm-billable``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPOSITORY_ROOT / "Docs/Examples/agents/bulk-reader/corpus.json"
SUPPORTED_LIVE_PROVIDERS = ("Moonshot", "ZAI")
PROVIDER_REQUEST_TIMEOUT_SECONDS = 60.0
PROVIDER_REQUEST_RETRIES = 0
MAX_PROVIDER_CALLS_PER_ARM = 8
MAX_WALL_SECONDS_PER_ARM = 120.0
MAX_TOOL_CALL_SECONDS = 30.0
MAX_TOTAL_TOKENS_PER_ARM = 100_000
MAX_OUTPUT_TOKENS_PER_CALL = 2_048
MAX_RECORDED_OUTPUT_CHARS = 16_000
REPORT_LIMITS = {
    "supported_live_providers": list(SUPPORTED_LIVE_PROVIDERS),
    "provider_request_timeout_seconds": PROVIDER_REQUEST_TIMEOUT_SECONDS,
    "provider_request_retries": PROVIDER_REQUEST_RETRIES,
    "max_provider_calls_per_arm": MAX_PROVIDER_CALLS_PER_ARM,
    "max_wall_seconds_per_arm": MAX_WALL_SECONDS_PER_ARM,
    "max_tool_call_seconds": MAX_TOOL_CALL_SECONDS,
    "max_total_tokens_per_arm": MAX_TOTAL_TOKENS_PER_ARM,
    "max_output_tokens_per_call": MAX_OUTPUT_TOKENS_PER_CALL,
    "max_recorded_output_chars": MAX_RECORDED_OUTPUT_CHARS,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the inert CLI parser without importing application modules."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        required=True,
        choices=SUPPORTED_LIVE_PROVIDERS,
        help="Live pilot route with evaluator-controlled transport policy.",
    )
    parser.add_argument("--main-model", required=True)
    parser.add_argument("--worker-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-billable",
        action="store_true",
        help="Confirm the comparison may make charged provider calls.",
    )
    return parser


def validate_live_request(args: argparse.Namespace) -> Path:
    """Refuse unapproved calls and existing outputs before app imports."""

    if not args.confirm_billable:
        raise ValueError(
            "Refusing provider calls without --confirm-billable; the comparison "
            "may make charged requests."
        )
    if args.provider not in SUPPORTED_LIVE_PROVIDERS:
        raise ValueError(
            "Live bulk-reader evaluation supports only Moonshot and ZAI because "
            "their existing Console routes expose bounded timeout/retry policy."
        )
    if not str(args.main_model).strip() or not str(args.worker_model).strip():
        raise ValueError("Both --main-model and --worker-model must be non-empty.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {output.parent}")
    return output


def _load_corpus(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        corpus = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bulk-reader corpus must be valid UTF-8 JSON.") from exc
    if not isinstance(corpus, dict):
        raise TypeError("Bulk-reader corpus must be a JSON object.")
    if not isinstance(corpus.get("id"), str) or not corpus["id"].strip():
        raise ValueError("Bulk-reader corpus requires a non-empty id.")
    if type(corpus.get("schema_version")) is not int:
        raise ValueError("Bulk-reader corpus requires an integer schema_version.")
    if not isinstance(corpus.get("rubric"), dict):
        raise TypeError("Bulk-reader corpus requires a manual grading rubric.")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Bulk-reader corpus requires at least one case.")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("Every bulk-reader case must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("Bulk-reader case ids must be non-empty and unique.")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case_id} requires a question.")
        if not isinstance(case.get("expected_facts"), list):
            raise TypeError(f"Case {case_id} requires expected_facts.")
        if not isinstance(case.get("sources"), dict) or not case["sources"]:
            raise ValueError(f"Case {case_id} requires sources.")
        for source_path, content in case["sources"].items():
            if not isinstance(content, str):
                raise TypeError(f"Case {case_id} sources require string content.")
            _confined_relative_path(source_path)
    return corpus, hashlib.sha256(raw).hexdigest()


def _confined_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Corpus sources require a relative confined path.")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
        or value != candidate.as_posix()
    ):
        raise ValueError(f"Corpus source is not a relative confined path: {value!r}")
    return candidate


def materialize_case(case: Mapping[str, Any], workspace_root: Path) -> list[dict]:
    """Write one validated case beneath a fresh, confined workspace."""

    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    materialized: list[dict] = []
    for source_path, content in case["sources"].items():
        relative = _confined_relative_path(source_path)
        target = workspace_root.joinpath(*relative.parts)
        current = workspace_root
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Corpus path traverses a symlink: {relative}")
            current.mkdir(exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"Corpus path targets a symlink: {relative}")
        resolved_parent = target.parent.resolve()
        if workspace_root not in (resolved_parent, *resolved_parent.parents):
            raise ValueError(f"Corpus source escaped its workspace: {relative}")
        target.write_text(content, encoding="utf-8")
        materialized.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "bytes": len(content.encode("utf-8")),
                "lines": len(content.splitlines()),
            }
        )
    return materialized


@contextmanager
def _scoped_agent_environment():
    overrides = {
        "TLDW_AGENTS_MAX_LIVE_SUBAGENTS": "1",
        "TLDW_AGENTS_RUN_LOG_ENABLED": "false",
        "TLDW_AGENTS_RUN_LOG_EVICT_ENABLED": "false",
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    return dataclasses.asdict(usage) if usage is not None else None


def _usage_counts_complete(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if "prompt_tokens" in payload or "completion_tokens" in payload:
        keys = ("prompt_tokens", "completion_tokens")
    elif "input_tokens" in payload or "output_tokens" in payload:
        keys = ("input_tokens", "output_tokens")
    else:
        return False
    return all(type(payload.get(key)) is int and payload[key] >= 0 for key in keys)


def _normalize_call_usage(
    payload: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    failed: bool,
):
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    usage = ProviderUsage.from_provider_payload(
        payload, provider=provider, model=model, partial=failed
    )
    if usage is None and isinstance(payload, Mapping):
        completion = payload.get("completion_tokens")
        if type(completion) is int and completion >= 0:
            details = payload.get("prompt_tokens_details")
            cached = (
                details.get("cached_tokens", 0)
                if isinstance(details, Mapping)
                and type(details.get("cached_tokens", 0)) is int
                else 0
            )
            usage = ProviderUsage(
                cache_read=max(cached, 0),
                output=completion,
                provider=provider,
                model=model,
                partial=True,
            )
    if usage is not None and (failed or not _usage_counts_complete(payload)):
        usage = dataclasses.replace(usage, partial=True)
    return usage


def _price_call(usage: Any, pricing_catalog: Any) -> tuple[str, dict | None]:
    if usage is None or usage.partial:
        return "unknown", None
    pricing = pricing_catalog.get_pricing(usage.provider, usage.model)
    if pricing is None:
        return "unknown", None
    missing_used_rate = (
        (usage.cache_read > 0 and pricing.cache_read_per_mtok is None)
        or (usage.cache_write > 0 and pricing.cache_write_per_mtok is None)
        or (usage.audio_input > 0 and pricing.audio_in_per_mtok is None)
        or (usage.audio_output > 0 and pricing.audio_out_per_mtok is None)
        or (
            usage.transcription_seconds > 0 and pricing.transcription_per_minute is None
        )
    )
    if missing_used_rate:
        return "unknown", None
    cost = pricing_catalog.cost_for_usage(usage)
    return (
        ("known", dataclasses.asdict(cost)) if cost is not None else ("unknown", None)
    )


class _RecordingChatCall:
    def __init__(self, adapter: Any, resolution: Any, signals: Any, pricing: Any):
        self.adapter = adapter
        self.resolution = resolution
        self.signals = signals
        self.pricing = pricing
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict:
        model = str(kwargs.get("model") or self.resolution.model or "")
        messages = kwargs.get("messages_payload") or []
        agent_role = "worker" if self.adapter._is_subagent(messages) else "main"
        index = len(self.calls) + 1
        if index > MAX_PROVIDER_CALLS_PER_ARM:
            record = {
                "index": index,
                "agent_role": agent_role,
                "provider": self.resolution.provider,
                "model": model,
                "status": "failed",
                "latency_ms": 0,
                "usage": None,
                "cost_status": "unknown",
                "cost": None,
                "failure": "provider_call_limit_exceeded",
                "output_truncated": False,
            }
            self.calls.append(record)
            raise RuntimeError("provider call limit exceeded")

        usage_before = len(self.signals.usage_payloads())
        started = time.perf_counter()
        response: dict | None = None
        failure: BaseException | None = None
        try:
            response = self.adapter.chat_call(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- record every provider failure
            failure = exc
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        payloads = self.signals.usage_payloads()[usage_before:]
        raw_usage = payloads[-1] if payloads else None
        usage = _normalize_call_usage(
            raw_usage,
            provider=self.resolution.provider,
            model=model,
            failed=failure is not None,
        )

        output_truncated = False
        if response is not None:
            try:
                message = response["choices"][0]["message"]
                content = str(message.get("content") or "")
                if len(content) > MAX_RECORDED_OUTPUT_CHARS:
                    message["content"] = content[:MAX_RECORDED_OUTPUT_CHARS]
                    output_truncated = True
            except (KeyError, IndexError, TypeError):
                pass
        cost_status, cost = _price_call(usage, self.pricing)
        self.calls.append(
            {
                "index": index,
                "agent_role": agent_role,
                "provider": self.resolution.provider,
                "model": model,
                "status": "failed" if failure is not None else "completed",
                "latency_ms": elapsed_ms,
                "usage": _usage_dict(usage),
                "cost_status": cost_status,
                "cost": cost,
                "failure": (
                    f"{type(failure).__name__}: {str(failure)[:400]}"
                    if failure is not None
                    else None
                ),
                "output_truncated": output_truncated,
            }
        )
        if failure is not None:
            raise failure
        assert response is not None
        return response


def _arm_prompt(case: Mapping[str, Any]) -> str:
    paths = list(case["sources"])
    return (
        f"Case: {case['id']}\n"
        f"Question: {case['question']}\n"
        "Requested workspace-relative paths:\n"
        + "\n".join(f"- {path}" for path in paths)
        + "\nRead the files with the available tools. Do not assume their contents."
    )


def _tool_reads(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    reads: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item.get("created_at", "")):
        kind = str(row.get("agent_kind") or "")
        for step in row.get("steps") or []:
            tool = step.get("tool_name")
            if step.get("kind") != "tool_call" or tool not in {
                "fs_list",
                "fs_read",
                "fs_glob",
                "fs_grep",
            }:
                continue
            args = step.get("args") or {}
            reads.append(
                {
                    "run_kind": kind,
                    "tool": str(tool),
                    "path": str(args.get("path") or "."),
                }
            )
    return reads


def _arm_cost(calls: Sequence[Mapping[str, Any]]) -> tuple[str, float | None]:
    if not calls or any(call.get("cost_status") != "known" for call in calls):
        return "unknown", None
    return "known", round(sum(call["cost"]["total"] for call in calls), 6)


def _run_arm(
    *,
    case: Mapping[str, Any],
    arm: str,
    gateway: Any,
    resolution: Any,
    main_model: str,
    worker_model: str,
    loop: asyncio.AbstractEventLoop,
    pricing_catalog: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_presets import BULK_READER_PRESET
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.local_tool_provider import (
        LocalToolProvider,
        _default_specs,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    with tempfile.TemporaryDirectory(prefix="tldw-bulk-reader-") as temp_name:
        scratch = Path(temp_name)
        workspace = scratch / "workspace"
        source_metadata = materialize_case(case, workspace)
        db = AgentRunsDB(scratch / "runs.sqlite", client_id="bulk-reader-evaluation")
        try:
            preset = dataclasses.replace(BULK_READER_PRESET, model=worker_model)
            db.create_agent_definition(preset)
            specs = [
                spec
                for spec in _default_specs(workspace)
                if spec.name in preset.tool_allowlist
            ]
            provider = LocalToolProvider(
                workspace_root=workspace,
                specs=specs,
                resolve_state=lambda _hub: EffectiveToolState(
                    state="allow", origin="tool_override"
                ),
                allow_write=False,
            )
            registry = ToolCatalogRegistry()
            registry.register_provider(provider)
            signals = ConsoleProviderStreamSignals()

            class _Store:
                def append_stream_chunk(self, _message_id: str, _chunk: str) -> None:
                    return None

                def reset_stream_content(self, _message_id: str) -> None:
                    return None

            adapter = _StreamingModelAdapter(
                store=_Store(),
                provider_gateway=gateway,
                resolution=resolution,
                assistant_message_id=f"{case['id']}-{arm}",
                should_cancel=lambda: False,
                loop=loop,
                native_tools=False,
                provider_stream_signals=signals,
            )
            recorded = _RecordingChatCall(adapter, resolution, signals, pricing_catalog)
            allowed = tuple(preset.tool_allowlist)
            delegated = arm == "delegated"
            if delegated:
                allowed = (*allowed, "spawn_subagent")
            budget = RunBudget(
                max_steps=3 * MAX_PROVIDER_CALLS_PER_ARM,
                max_wall_seconds=MAX_WALL_SECONDS_PER_ARM,
                max_subagents=1 if delegated else 0,
                max_active_tools=len(preset.tool_allowlist),
                max_model_turns=MAX_PROVIDER_CALLS_PER_ARM,
                max_total_tokens=MAX_TOTAL_TOKENS_PER_ARM,
                max_tool_call_seconds=MAX_TOOL_CALL_SECONDS,
            )
            system_prompt = (
                "Use the confined read tools to answer the requested question with "
                "short quotations and file/line references."
                if not delegated
                else "Delegate the entire reading task to the named bulk-reader agent. "
                "Use its result, and verify source passages directly when needed."
            )
            service = AgentService(
                db=db,
                registry=registry,
                chat_call=recorded,
            )
            conversation_id = f"bulk-reader-{case['id']}-{arm}-{uuid4().hex}"
            run_id, outcome = service.run_turn(
                conversation_id=conversation_id,
                messages=[{"role": "user", "content": _arm_prompt(case)}],
                config=AgentConfig(
                    model=main_model,
                    system_prompt=system_prompt,
                    allowed_tools=allowed,
                    budget=budget,
                    native_tools=False,
                ),
                api_endpoint=str(resolution.execution_key or resolution.provider),
            )
            rows = db.list_runs(conversation_id)
        finally:
            db.close()

    primary = next(row for row in rows if row["id"] == run_id)
    children = [row for row in rows if row.get("parent_run_id") == run_id]
    named_child = any(row.get("agent_definition") == preset.name for row in children)
    worker_model_reached_provider = any(
        call.get("agent_role") == "worker" and call.get("model") == worker_model
        for call in recorded.calls
    )
    delegation_occurred = named_child and worker_model_reached_provider
    child_failed = any(row.get("status") != "done" for row in children)
    truncated = any(call["output_truncated"] for call in recorded.calls)
    if outcome.status == "error" or child_failed:
        status = "failed"
    elif outcome.status != "done" or truncated:
        status = "incomplete"
    elif delegated and not delegation_occurred:
        status = "non_delegating"
    else:
        status = "completed"
    cost_status, total_cost = _arm_cost(recorded.calls)
    return (
        {
            "status": status,
            "runtime_status": outcome.status,
            "run_id": run_id,
            "answer": outcome.final_text,
            "delegation_occurred": delegation_occurred,
            "calls": recorded.calls,
            "tool_reads": _tool_reads(rows),
            "child_runs": [
                {
                    "run_id": row["id"],
                    "status": row["status"],
                    "agent_definition": row.get("agent_definition"),
                    "result": row.get("result") or "",
                }
                for row in children
            ],
            "worker_output": "\n".join(
                str(row.get("result") or "") for row in children
            ),
            "cost_status": cost_status,
            "total_cost_usd": total_cost,
            "primary_error": next(
                (
                    str(step.get("summary") or "")
                    for step in primary.get("steps") or []
                    if step.get("kind") == "error"
                ),
                None,
            ),
        },
        source_metadata,
    )


def _run_comparison_sync(
    *,
    corpus: Mapping[str, Any],
    corpus_sha256: str,
    provider: str,
    main_model: str,
    worker_model: str,
    gateway: Any,
    resolution: Any,
    pricing_catalog: Any,
    loop: asyncio.AbstractEventLoop,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with _scoped_agent_environment():
        for case in corpus["cases"]:
            direct, direct_sources = _run_arm(
                case=case,
                arm="direct",
                gateway=gateway,
                resolution=resolution,
                main_model=main_model,
                worker_model=worker_model,
                loop=loop,
                pricing_catalog=pricing_catalog,
            )
            delegated, delegated_sources = _run_arm(
                case=case,
                arm="delegated",
                gateway=gateway,
                resolution=resolution,
                main_model=main_model,
                worker_model=worker_model,
                loop=loop,
                pricing_catalog=pricing_catalog,
            )
            if direct_sources != delegated_sources:
                raise RuntimeError("Materialized corpus differed between arms.")
            cases.append(
                {
                    "id": case["id"],
                    "category": case.get("category", ""),
                    "question": case["question"],
                    "expected_facts": case["expected_facts"],
                    "sources": direct_sources,
                    "direct": direct,
                    "delegated": delegated,
                }
            )
    complete = all(
        case[arm]["status"] == "completed"
        for case in cases
        for arm in ("direct", "delegated")
    )
    return {
        "schema_version": 1,
        "experiment": "bulk-reader-direct-vs-delegated",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if complete else "incomplete",
        "execution_order": ["direct", "delegated"],
        "corpus": {
            "id": corpus["id"],
            "schema_version": corpus["schema_version"],
            "sha256": corpus_sha256,
        },
        "provider": {
            "requested": provider,
            "resolved": resolution.provider,
            "execution_key": resolution.execution_key,
            "base_url": resolution.base_url,
            "transport_policy": {
                "request_timeout_seconds": resolution.request_timeout,
                "request_retries": resolution.request_retries,
                "request_retry_delay_seconds": resolution.request_retry_delay,
            },
        },
        "models": {"main": main_model, "worker": worker_model},
        "limits": REPORT_LIMITS,
        "quality_review": {"status": "pending", "rubric": corpus["rubric"]},
        "cases": cases,
    }


async def evaluate_comparison(
    *,
    corpus_path: Path,
    output_path: Path,
    provider: str,
    main_model: str,
    worker_model: str,
    gateway: Any,
    resolution: Any,
    pricing_catalog: Any,
) -> dict[str, Any]:
    """Run both arms through one loop and exclusively create a JSON report."""

    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    corpus, corpus_sha256 = _load_corpus(corpus_path)
    loop = asyncio.get_running_loop()
    report = await asyncio.to_thread(
        _run_comparison_sync,
        corpus=corpus,
        corpus_sha256=corpus_sha256,
        provider=provider,
        main_model=main_model,
        worker_model=worker_model,
        gateway=gateway,
        resolution=resolution,
        pricing_catalog=pricing_catalog,
        loop=loop,
    )
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


async def run_live(args: argparse.Namespace) -> int:
    """Resolve the configured provider, run the comparison, and close it."""

    output = validate_live_request(args)
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    config = load_settings(force_reload=True)
    gateway = ConsoleProviderGateway(config_provider=lambda: config)
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=args.provider,
                explicit_model=args.main_model,
                max_tokens=MAX_OUTPUT_TOKENS_PER_CALL,
                streaming=True,
            )
        )
        if not resolution.ready:
            raise RuntimeError(resolution.visible_copy or "Provider is not ready.")
        if resolution.execution_key not in {"moonshot", "zai"}:
            raise RuntimeError("Resolved provider is outside the live pilot allowlist.")
        resolution = dataclasses.replace(
            resolution,
            request_timeout=PROVIDER_REQUEST_TIMEOUT_SECONDS,
            request_retries=PROVIDER_REQUEST_RETRIES,
            request_retry_delay=0.0,
        )
        await evaluate_comparison(
            corpus_path=DEFAULT_CORPUS,
            output_path=output,
            provider=args.provider,
            main_model=args.main_model,
            worker_model=args.worker_model,
            gateway=gateway,
            resolution=resolution,
            pricing_catalog=PricingCatalog(config=config.get("pricing", {})),
        )
    finally:
        with contextlib.suppress(Exception):
            await gateway.aclose()
    print(f"Wrote pending-manual-review comparison report: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_live(args))
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
