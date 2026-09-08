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
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from tldw_chatbook.Agents.bulk_reader_corpus import BulkReaderCase, BulkReaderCorpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPOSITORY_ROOT / "Docs/Examples/agents/bulk-reader/corpus.json"
SUPPORTED_LIVE_PROVIDERS = ("Moonshot", "ZAI")
PROVIDER_REQUEST_TIMEOUT_SECONDS = 60.0
PROVIDER_REQUEST_RETRIES = 0
MAX_PROVIDER_CALLS_PER_ARM = 8
# One primary plus the single child admitted by the runtime budget.
MAX_RUN_RECORDS_PER_ARM = 2
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
    "max_run_records_per_arm": MAX_RUN_RECORDS_PER_ARM,
    "max_wall_seconds_per_arm": MAX_WALL_SECONDS_PER_ARM,
    "max_tool_call_seconds": MAX_TOOL_CALL_SECONDS,
    "max_total_tokens_per_arm": MAX_TOTAL_TOKENS_PER_ARM,
    "max_output_tokens_per_call": MAX_OUTPUT_TOKENS_PER_CALL,
    "max_recorded_output_chars": MAX_RECORDED_OUTPUT_CHARS,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the inert CLI parser without importing application modules.

    Returns:
        The parser for provider, model, output, and billable-consent arguments.
    """

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
    """Validate live inputs, refusing unapproved calls before app imports.

    Args:
        args: Parsed CLI arguments. Validated provider/model values replace the
            corresponding fields before execution.

    Returns:
        The validated absolute path for a new report in an existing directory.

    Raises:
        ValueError: If consent is missing or an input is invalid or unsupported.
        FileExistsError: If the report path already exists.
        OSError: If the output directory cannot be inspected.
    """

    if not args.confirm_billable:
        raise ValueError(
            "Refusing provider calls without --confirm-billable; the comparison "
            "may make charged requests."
        )
    args.provider, args.main_model, args.worker_model = _validate_model_selection(
        args.provider, args.main_model, args.worker_model
    )
    if args.provider not in SUPPORTED_LIVE_PROVIDERS:
        raise ValueError(
            "Live bulk-reader evaluation supports only Moonshot and ZAI because "
            "their existing Console routes expose bounded timeout/retry policy."
        )
    return _validate_output_path(args.output)


def _validate_model_selection(
    provider: str, main_model: str, worker_model: str
) -> tuple[str, str, str]:
    from tldw_chatbook.Utils.input_validation import validate_navigation_context_text

    return (
        validate_navigation_context_text(provider, name="provider", max_length=128),
        validate_navigation_context_text(main_model, name="main model", max_length=256),
        validate_navigation_context_text(
            worker_model, name="worker model", max_length=256
        ),
    )


def _validate_output_path(value: Path) -> Path:
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    output = validate_path_simple(value, probe_existing=False).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {output.parent}")
    return output


def _load_corpus(path: Path) -> tuple[BulkReaderCorpus, str]:
    from tldw_chatbook.Agents.bulk_reader_corpus import BulkReaderCorpus
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    path = validate_path_simple(path, require_exists=True)
    raw = path.read_bytes()
    try:
        corpus = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bulk-reader corpus must be valid UTF-8 JSON.") from exc
    return BulkReaderCorpus.model_validate(corpus), hashlib.sha256(raw).hexdigest()


def materialize_case(case: BulkReaderCase, workspace_root: Path) -> list[dict]:
    """Write one validated case beneath a fresh, confined workspace.

    Args:
        case: Validated synthetic case with canonical relative source paths.
        workspace_root: Evaluator-owned scratch directory to populate.

    Returns:
        Source path, SHA-256, byte count, and line count for each written file.

    Raises:
        ValueError: If a path is unsafe, a symlink, or outside the workspace.
        OSError: If the workspace or sources cannot be created.
    """

    from tldw_chatbook.Utils.path_validation import validate_path, validate_path_simple

    workspace_root = validate_path_simple(
        workspace_root, probe_existing=False
    ).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    materialized: list[dict] = []
    for source_path, content in case.sources.items():
        relative = PurePosixPath(source_path)
        target = workspace_root.joinpath(*relative.parts)
        current = workspace_root
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Corpus path traverses a symlink: {relative}")
            current = validate_path(
                current, workspace_root, allow_hidden=True, redact_paths=True
            )
            current.mkdir(exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"Corpus path targets a symlink: {relative}")
        target = validate_path(
            target, workspace_root, allow_hidden=True, redact_paths=True
        )
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


def _provider_metadata(response: Any) -> Any:
    from tldw_chatbook.Chat.console_provider_gateway import _provider_turn_metadata

    try:
        return _provider_turn_metadata(response)
    except Exception:  # noqa: BLE001 -- metadata loss must not fail a provider call
        return None


class TerminalMetadataIterator(Iterator[Any]):
    """Forward a provider stream and record its terminal metadata once."""

    def __init__(self, response: Iterator[Any], recorder: Any) -> None:
        """Initialize stream observation.

        Args:
            response: Provider-owned iterator, closed when this wrapper closes.
            recorder: Receiver for terminal metadata or an unknown outcome.
        """
        self._response = response
        self.recorder = recorder
        self.finished = False

    def __next__(self) -> Any:
        try:
            return next(self._response)
        except StopIteration:
            self._finish(_provider_metadata(self._response))
            raise
        except BaseException:
            self._finish(None)
            raise

    def _finish(self, metadata: Any) -> None:
        if self.finished:
            return
        self.finished = True
        self.recorder.record(metadata)

    def close(self) -> None:
        """Close the underlying stream and record unknown metadata if unfinished."""

        close = getattr(self._response, "close", None)
        if callable(close):
            close()
        self._finish(None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class ProviderMetadataRecorder:
    """Capture typed terminal metadata without changing provider responses."""

    def __init__(self) -> None:
        self._records: list[Any] = []
        self._lock = threading.Lock()

    def record(self, metadata: Any) -> None:
        """Append one provider outcome under the recorder lock.

        Args:
            metadata: Typed terminal metadata, or None when unavailable.
        """

        with self._lock:
            self._records.append(metadata)

    def count(self) -> int:
        """Return the number of observed provider outcomes.

        Returns:
            A snapshot count suitable for a later latest_since lookup.
        """

        with self._lock:
            return len(self._records)

    def latest_since(self, index: int) -> Any:
        """Read the latest outcome recorded after an earlier count.

        Args:
            index: Count captured before the provider call.

        Returns:
            The latest metadata, or None if absent or no new call completed.
        """

        with self._lock:
            return self._records[-1] if len(self._records) > index else None

    def wrap_chat_api_call(self, call: Callable[..., Any]) -> Callable[..., Any]:
        """Observe a provider call while preserving its response protocol.

        Args:
            call: Synchronous provider entry point accepting keyword arguments.

        Returns:
            A wrapper preserving mappings and proxying streamed responses.
        """

        def recording_call(**kwargs: Any) -> Any:
            response = call(**kwargs)
            if isinstance(response, Mapping):
                self.record(_provider_metadata(response))
                return response
            if isinstance(response, Iterator):
                return TerminalMetadataIterator(response, self)
            self.record(None)
            return response

        return recording_call


class RecordingChatCall:
    """Enforce the evaluator call cap and retain per-call usage and outcomes."""

    def __init__(
        self,
        adapter: Any,
        resolution: Any,
        signals: Any,
        pricing: Any,
        provider_metadata: ProviderMetadataRecorder,
    ) -> None:
        """Bind the runtime adapter and accounting sources.

        Args:
            adapter: Console adapter whose chat_call dispatches model requests.
            resolution: Default provider/model resolution for the primary agent.
            signals: Per-call Console usage signals.
            pricing: Catalog of rates keyed by the actual provider and model.
            provider_metadata: Recorder of terminal finish reasons and usage.
        """

        self.adapter = adapter
        self.resolution = resolution
        self.signals = signals
        self.pricing = pricing
        self.provider_metadata = provider_metadata
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
                "finish_reason": None,
                "provider_output_limited": False,
            }
            self.calls.append(record)
            raise RuntimeError("provider call limit exceeded")

        usage_before = len(self.signals.usage_payloads())
        metadata_before = self.provider_metadata.count()
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
        terminal_metadata = self.provider_metadata.latest_since(metadata_before)
        finish_reason = (
            str(terminal_metadata.finish_reason)
            if terminal_metadata is not None
            else None
        )
        provider_output_limited = finish_reason == "length"

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
                "status": (
                    "failed"
                    if failure is not None
                    else "incomplete"
                    if provider_output_limited
                    else "completed"
                ),
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
                "finish_reason": finish_reason,
                "provider_output_limited": provider_output_limited,
            }
        )
        if failure is not None:
            raise failure
        assert response is not None
        return response


def _arm_prompt(case: BulkReaderCase) -> str:
    paths = list(case.sources)
    return (
        f"Case: {case.id}\n"
        f"Question: {case.question}\n"
        "Requested workspace-relative paths:\n"
        + "\n".join(f"- {path}" for path in paths)
        + "\nRead the files with the available tools. Do not assume their contents."
    )


def _tool_reads(
    rows: Sequence[Mapping[str, Any]], invoked_reads: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    kinds = {row["id"]: str(row.get("agent_kind") or "") for row in rows}
    return [
        {
            "run_kind": kinds.get(read["run_id"], "unknown"),
            "tool": read["tool"],
            "path": read["path"],
        }
        for read in invoked_reads
    ]


def _arm_cost(calls: Sequence[Mapping[str, Any]]) -> tuple[str, float | None]:
    if not calls or any(call.get("cost_status") != "known" for call in calls):
        return "unknown", None
    return "known", round(sum(call["cost"]["total"] for call in calls), 6)


def _run_arm(
    *,
    case: BulkReaderCase,
    arm: str,
    gateway: Any,
    resolution: Any,
    main_model: str,
    worker_model: str,
    loop: asyncio.AbstractEventLoop,
    pricing_catalog: Any,
    provider_metadata: ProviderMetadataRecorder,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_presets import BULK_READER_PRESET
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.local_tool_provider import (
        LocalToolProvider,
        _default_specs,
    )
    from tldw_chatbook.Agents.run_context import current_run_id
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

    with tempfile.TemporaryDirectory(prefix="tldw-bulk-reader-") as temp_name:
        scratch = Path(temp_name)
        workspace = scratch / "workspace"
        source_metadata = materialize_case(case, workspace)
        db = AgentRunsDB(scratch / "runs.sqlite", client_id="bulk-reader-evaluation")
        try:
            preset = dataclasses.replace(BULK_READER_PRESET, model=worker_model)
            db.create_agent_definition(preset)
            workspace_executor = WorkspaceToolExecutor(workspace)
            specs = [
                spec
                for spec in _default_specs(
                    workspace, workspace_executor=workspace_executor
                )
                if spec.name in preset.tool_allowlist
            ]
            provider = LocalToolProvider(
                workspace_root=workspace,
                workspace_executor=workspace_executor,
                specs=specs,
                resolve_state=lambda _hub: EffectiveToolState(
                    state="allow", origin="tool_override"
                ),
                allow_write=False,
            )
            successful_content_read_runs: set[str] = set()
            invoked_reads: list[dict[str, str]] = []
            content_read_lock = threading.Lock()
            invoke_local_tool = provider.invoke

            def record_local_tool_result(tool_id: str, args: dict):
                tool_name = tool_id.split(":", 1)[-1]
                read_run_id = current_run_id()
                # Capture synthetic read paths at invocation; current run-step
                # persistence intentionally omits tool arguments with capture off.
                with content_read_lock:
                    invoked_reads.append(
                        {
                            "run_id": read_run_id,
                            "tool": tool_name,
                            "path": str(args.get("path") or "."),
                        }
                    )
                result = invoke_local_tool(tool_id, args)
                if result.ok and tool_name in {"fs_read", "fs_grep"}:
                    with content_read_lock:
                        successful_content_read_runs.add(read_run_id)
                return result

            provider.invoke = record_local_tool_result
            registry = ToolCatalogRegistry()
            registry.register_provider(provider)
            signals = ConsoleProviderStreamSignals()

            store = ConsoleChatStore()
            session = store.ensure_session()
            assistant = store.append_message(
                session.id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            adapter = _StreamingModelAdapter(
                store=store,
                provider_gateway=gateway,
                resolution=resolution,
                assistant_message_id=assistant.id,
                should_cancel=lambda: False,
                loop=loop,
                native_tools=False,
                provider_stream_signals=signals,
            )
            recorded = RecordingChatCall(
                adapter,
                resolution,
                signals,
                pricing_catalog,
                provider_metadata,
            )
            allowed = tuple(preset.tool_allowlist)
            delegated = arm == "delegated"
            if delegated:
                allowed = (*allowed, "spawn_subagent")
            budget = RunBudget(
                max_steps=3 * MAX_PROVIDER_CALLS_PER_ARM,
                max_wall_seconds=MAX_WALL_SECONDS_PER_ARM,
                max_subagents=1 if delegated else 0,
                max_model_turns=MAX_PROVIDER_CALLS_PER_ARM,
                max_model_retries=0,
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
            conversation_id = f"bulk-reader-{case.id}-{arm}-{uuid4().hex}"
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
            # Both helpers reuse this thread's held connection. Keep the
            # primary and hydrated history in one snapshot without changing
            # the shared DB's read/write policy for other callers.
            with db.transaction():
                primary = db.get_run(run_id)
                rows = db.list_runs(conversation_id, limit=MAX_RUN_RECORDS_PER_ARM + 1)
        finally:
            db.close()

    if primary is None:
        raise RuntimeError("Evaluator primary run is missing from scratch history.")
    history_complete = len(rows) <= MAX_RUN_RECORDS_PER_ARM
    children = [row for row in rows if row.get("parent_run_id") == run_id]
    named_children = [
        row for row in children if row.get("agent_definition") == preset.name
    ]
    named_child = bool(named_children)
    worker_model_reached_provider = any(
        call.get("agent_role") == "worker" and call.get("model") == worker_model
        for call in recorded.calls
    )
    delegation_occurred = named_child and worker_model_reached_provider
    child_failed = any(row.get("status") != "done" for row in children)
    truncated = any(call["output_truncated"] for call in recorded.calls)
    provider_output_limited = any(
        call["provider_output_limited"] for call in recorded.calls
    )
    content_read = (
        any(row["id"] in successful_content_read_runs for row in named_children)
        if delegated
        else run_id in successful_content_read_runs
    )
    status_reasons: list[str] = []
    if not history_complete:
        status_reasons.append("run_history_limit_exceeded")
    if outcome.status == "error":
        status_reasons.append("runtime_error")
    if child_failed:
        status_reasons.append("child_run_failed")
    if outcome.status not in {"done", "error"}:
        status_reasons.append(f"runtime_{outcome.status}")
    if truncated:
        status_reasons.append("retained_output_limit")
    if provider_output_limited:
        status_reasons.append("provider_output_token_limit")
    if delegated and not delegation_occurred:
        status_reasons.append("named_bulk_reader_delegation_missing")
    if not content_read:
        status_reasons.append(
            "bulk_reader_content_read_missing"
            if delegated
            else "direct_content_read_missing"
        )
    if outcome.status == "error" or child_failed:
        status = "failed"
    elif delegated and not delegation_occurred:
        status = "non_delegating"
    elif (
        outcome.status != "done"
        or truncated
        or provider_output_limited
        or not content_read
        or not history_complete
    ):
        status = "incomplete"
    else:
        status = "completed"
    cost_status, total_cost = _arm_cost(recorded.calls)
    return (
        {
            "status": status,
            "status_reasons": status_reasons,
            "runtime_status": outcome.status,
            "run_id": run_id,
            "answer": outcome.final_text,
            "delegation_occurred": delegation_occurred,
            "calls": recorded.calls,
            "tool_reads": _tool_reads(rows, invoked_reads),
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
    corpus: BulkReaderCorpus,
    corpus_sha256: str,
    provider: str,
    main_model: str,
    worker_model: str,
    gateway: Any,
    resolution: Any,
    pricing_catalog: Any,
    loop: asyncio.AbstractEventLoop,
    provider_metadata: ProviderMetadataRecorder,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with _scoped_agent_environment():
        for case in corpus.cases:
            direct, direct_sources = _run_arm(
                case=case,
                arm="direct",
                gateway=gateway,
                resolution=resolution,
                main_model=main_model,
                worker_model=worker_model,
                loop=loop,
                pricing_catalog=pricing_catalog,
                provider_metadata=provider_metadata,
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
                provider_metadata=provider_metadata,
            )
            if direct_sources != delegated_sources:
                raise RuntimeError("Materialized corpus differed between arms.")
            cases.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "question": case.question,
                    "expected_facts": case.expected_facts,
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
            "id": corpus.id,
            "schema_version": corpus.schema_version,
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
        "quality_review": {"status": "pending", "rubric": corpus.rubric},
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
    provider_metadata: ProviderMetadataRecorder,
) -> dict[str, Any]:
    """Run both arms through one loop and exclusively create a JSON report.

    Args:
        corpus_path: UTF-8 JSON synthetic corpus matching schema version 1.
        output_path: New report path whose parent directory already exists.
        provider: Provider label to retain in report provenance.
        main_model: Model used by the primary agent in both arms.
        worker_model: Same-provider model assigned to the named reader.
        gateway: Configured Console gateway, or a deterministic test boundary.
        resolution: Ready provider resolution with transport limits already set.
        pricing_catalog: Model-specific rates; absent rates remain unknown.
        provider_metadata: Per-call recorder wrapping the gateway's provider seam.

    Returns:
        The saved report with source provenance, per-arm outcomes, provider
        calls, usage, costs, limits, and a pending manual quality review.

    Raises:
        ValueError: If the corpus, model selection, or paths are invalid.
        FileExistsError: If output already exists, including a creation race.
        OSError: If a corpus, scratch workspace, or report operation fails.
        RuntimeError: If evaluator setup or runtime bookkeeping fails. Ordinary
            provider failures are retained as unsuccessful arm outcomes.
    """

    provider, main_model, worker_model = _validate_model_selection(
        provider, main_model, worker_model
    )
    output_path = _validate_output_path(output_path)
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
        provider_metadata=provider_metadata,
    )
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


async def run_live(args: argparse.Namespace) -> int:
    """Resolve the configured provider, run the comparison, and close it.

    Args:
        args: Parsed provider/model/output options and explicit billable consent.

    Returns:
        Zero after saving the comparison, including incomplete arm outcomes.

    Raises:
        ValueError: If consent, input validation, or corpus validation fails.
        OSError: If a filesystem operation fails or the report already exists.
        RuntimeError: If provider readiness or evaluator setup fails.
    """

    output = validate_live_request(args)
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    config = load_settings(force_reload=True)
    # load_settings keeps unprojected sections, including pricing, in the
    # raw TOML snapshot. Direct configuration dictionaries remain supported.
    pricing_config = config.get("pricing") or config.get(
        "COMPREHENSIVE_CONFIG_RAW", {}
    ).get("pricing", {})
    provider_metadata = ProviderMetadataRecorder()
    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        chat_api_call_fn=provider_metadata.wrap_chat_api_call(chat_api_call),
    )
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
            pricing_catalog=PricingCatalog(config=pricing_config),
            provider_metadata=provider_metadata,
        )
    finally:
        with contextlib.suppress(Exception):
            await gateway.aclose()
    print(f"Wrote pending-manual-review comparison report: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with explicit consent for potentially charged requests.

    Args:
        argv: Command-line arguments, or None to use the process arguments.

    Returns:
        Zero after a report is saved; execution outcomes remain in the report.

    Raises:
        SystemExit: Zero for help or two for invalid arguments/runtime setup.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_live(args))
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
