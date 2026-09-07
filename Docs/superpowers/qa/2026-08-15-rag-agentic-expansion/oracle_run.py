#!/usr/bin/env python3
"""TASK-16174 Phase E — the answer-level oracle run.

Runs the REAL Console agent loop (`Agents/agent_service.AgentService` ->
`Chat.Chat_Functions.chat_api_call` -> api.anthropic.com) over a REAL,
isolated installation of the retrieval stack (the RAG-eval harness's
`build_eval_runtime`), once per question per arm:

* **tool-OFF** -- the shipped default posture. `[tools]
  expand_document_enabled` is absent from the scratch config, so
  `BuiltinToolProvider` never registers the tool, and the run's
  `allowed_tools` is `("search_library_rag",)`.
* **tool-ON** -- the user has switched the gate on. The scratch config is
  rewritten with `expand_document_enabled = true`, settings are force-
  reloaded, a fresh provider is built (which is what actually registers
  the tool), and `allowed_tools` gains `"expand_document"`.

Everything else is byte-identical between the arms: same corpus, same
index, same retrieval route, same system prompt, same model, same
temperature, same questions, same order.

**Scoring is mechanical (spec Phase E): no LLM grader.** Each question
carries a fact-oracle regex verified to appear only inside its target
document's body (see `questions.toml`). A question scores 1 when the
oracle matches the run's final answer text, 0 otherwise. An OFF >= ON
result is a FINDING, reported as it lands.

**The retrieval route is `plain`.** `runtime.service.config.search.
default_search_mode = "plain"` sends the tool's `mode="rag"` request down
`LibraryLocalRagSearchService._search_keyword`, the Library's four-seam
scope-aware keyword path -- the shipped route for a BM25/plain RAG
profile and the one that emits LABEL-ONLY media/conversation rows
(`Matched media · document`, `Matched conversation · 1 message`), which is
the regime this whole arc exists for. Pre-registered in `questions.toml`.

**Isolation** (`backlog/docs/lessons-live-verification.md`). Every app
import happens after a scratch `HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/
`TLDW_CONFIG_PATH`/`TLDW_TEST_MODE` are set, so nothing resolves the real
profile; `XDG_CACHE_HOME` is deliberately NOT redirected and
`config.get_model_cache_dir` is repointed at the harness's
`model_cache_dir()` so the embedding model is read (read-only) from the
real HuggingFace cache instead of being downloaded -- downloads are
blocked outright by forcing `huggingface_hub.constants.HF_HUB_OFFLINE`.
The real `~/.config/tldw_cli/config.toml` is sha256'd before and after and
the two are compared. The scratch DBs the harness writes are the SAME
files `expand_document` reads, via `[database] media_db_path /
chachanotes_db_path / prompts_db_path` overrides in the scratch config --
the tool resolves its handles through `config.get_*_db_lazy()`, so
without those overrides it would read the developer's own library.

**Credentials.** The API key is read from the git-excluded repo-root
`anthropic-api-key.txt` at call time into a local variable, passed to
`chat_api_call(api_key=...)`, and never printed, logged, or written to any
config file.

Usage::

    python oracle_run.py --verify-oracles      # corpus checks only, no imports of the app
    python oracle_run.py --dry-run             # build the runtime + probe retrieval; NO model calls
    python oracle_run.py --live --confirm-billable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
QUESTIONS_PATH = HERE / "questions.toml"
# Qodo PR-1712: an absolute machine-specific path is both a poor
# secret-handling pattern and non-portable -- anyone re-running this probe on
# another checkout gets a confusing FileNotFoundError rather than a usable
# message. Resolution order: an explicit env var, then the git-excluded
# repo-root file located RELATIVE to this script.
API_KEY_ENV = "ANTHROPIC_API_KEY"
# parents[4], not [3]: this file sits at
# Docs/superpowers/qa/<arc>/oracle_run.py, so [3] is Docs/ (Qodo PR-1795
# finding 5 -- an off-by-one in this PR's own fix, caught by review).
API_KEY_PATH = Path(__file__).resolve().parents[4] / "anthropic-api-key.txt"

#: Cheapest capable model with native tool-calls. Pricing (Anthropic first-party,
#: cached 2026-06-24): $1.00 / MTok input, $5.00 / MTok output; cache writes
#: 1.25x input, cache reads 0.1x input.
MODEL = "claude-haiku-4-5"
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00

#: Identical in both arms and deliberately tool-agnostic: it must not name
#: `expand_document`, or the ON arm would be measuring an instruction rather
#: than the tool's own description-borne policy.
SYSTEM_PROMPT = (
    "You answer questions using the user's local Library. Use the tools "
    "available to you to retrieve evidence before you answer. Library search "
    "is keyword-based: search with a few distinctive terms rather than a "
    "whole sentence, and retry with different terms if a search returns "
    "nothing. Base your answer only on evidence you actually retrieved -- if "
    "you cannot find the fact, say plainly that you could not find it. "
    "Answer in one or two sentences."
)

TOP_K = 10


# --------------------------------------------------------------------------
# stdlib-only helpers (safe to run before any app import)
# --------------------------------------------------------------------------


def _load_toml(path: Path) -> dict:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_questions() -> list[dict]:
    return list(_load_toml(QUESTIONS_PATH)["question"])


def real_home() -> Path:
    """The user's real home, immune to the scratch `$HOME` this script sets."""
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # pragma: no cover - non-POSIX
        return Path(os.path.expanduser("~"))


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def verify_oracles() -> int:
    """Mechanically re-check every oracle against the corpus and the questions.

    Prints one line per question and returns a process exit code. This is the
    check whose recorded output lives in `questions.toml`'s header; it imports
    nothing from the application.
    """
    corpus = _load_toml(WORKTREE / "Tests/RAG_Eval/fixtures/corpus.toml")["doc"]
    ok = True
    for question in load_questions():
        pattern = re.compile(question["oracle"], re.IGNORECASE)
        content_hits = [d["slug"] for d in corpus if pattern.search(d["content"])]
        title_hits = [d["slug"] for d in corpus if pattern.search(d["title"])]
        in_question = bool(pattern.search(question["question"]))
        good = (
            content_hits == [question["slug"]] and not title_hits and not in_question
        )
        ok = ok and good
        print(
            f"{'OK ' if good else 'BAD'} {question['slug']:32s} "
            f"/{question['oracle']}/ content={content_hits} title={title_hits} "
            f"in_question={in_question}"
        )
    print("ALL OK" if ok else "PROBLEMS")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# scratch profile
# --------------------------------------------------------------------------


def _config_text(*, media_db: Path, chacha_db: Path, prompts_db: Path, gate_on: bool) -> str:
    """The scratch `config.toml`.

    `[database]` is what makes `expand_document` (which resolves its handles
    through `config.get_media_db_lazy()` / `get_chachanotes_db_lazy()` /
    `get_prompts_db_lazy()`) read the very databases the harness wrote.
    `[tools]` is the real registration gate `BuiltinToolProvider` consults.
    """
    return (
        "[database]\n"
        f'media_db_path = "{media_db}"\n'
        f'chachanotes_db_path = "{chacha_db}"\n'
        f'prompts_db_path = "{prompts_db}"\n'
        "\n"
        "[tools]\n"
        f"expand_document_enabled = {'true' if gate_on else 'false'}\n"
    )


def prepare_scratch(scratch: Path, *, gate_on: bool) -> dict[str, Path]:
    """Create the scratch profile and set the isolation env. Call BEFORE imports."""
    home = scratch / "home"
    for directory in (home, home / ".config", home / ".local" / "share", scratch / "run"):
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)

    paths = {
        "scratch": scratch,
        "home": home,
        "config": scratch / "config.toml",
        "eval": scratch / "eval",
        "media_db": scratch / "eval" / "eval_media.db",
        "chacha_db": scratch / "eval" / "eval_chachanotes.db",
        "prompts_db": scratch / "eval" / "eval_prompts.db",
        "runs_db": scratch / "run" / "agent_runs.db",
    }
    paths["eval"].mkdir(parents=True, exist_ok=True)
    write_scratch_config(paths, gate_on=gate_on)

    os.environ["HOME"] = str(home)
    os.environ["XDG_CONFIG_HOME"] = str(home / ".config")
    os.environ["XDG_DATA_HOME"] = str(home / ".local" / "share")
    os.environ["TLDW_CONFIG_PATH"] = str(paths["config"])
    os.environ["TLDW_TEST_MODE"] = "1"
    # Read by `Tests/RAG_Eval/harness/environment.py` at import to latch
    # HF_HUB_OFFLINE before huggingface_hub.constants is evaluated.
    os.environ["RAG_EVAL"] = "1"
    # NOT set: XDG_CACHE_HOME / HF_HOME / HF_HUB_CACHE. The harness's
    # `model_cache_dir()` must keep resolving the developer's real
    # ~/.cache/huggingface/hub (read-only) or the embedding model is a cold
    # miss -- and downloads are blocked, so it would abort rather than fetch.
    return paths


def write_scratch_config(paths: dict[str, Path], *, gate_on: bool) -> None:
    paths["config"].write_text(
        _config_text(
            media_db=paths["media_db"],
            chacha_db=paths["chacha_db"],
            prompts_db=paths["prompts_db"],
            gate_on=gate_on,
        ),
        encoding="utf-8",
    )
    os.chmod(paths["config"], 0o600)


# --------------------------------------------------------------------------
# app-facing stages (import the application; must run after prepare_scratch)
# --------------------------------------------------------------------------


def import_app(paths: dict[str, Path]) -> Any:
    """Import the app, assert provenance and isolation, and block downloads."""
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))

    import tldw_chatbook

    module_file = Path(tldw_chatbook.__file__).resolve()
    if WORKTREE not in module_file.parents:
        raise SystemExit(f"tldw_chatbook resolved OUTSIDE the worktree: {module_file}")
    print(f"[provenance] tldw_chatbook -> {module_file}")

    from tldw_chatbook import config as app_config

    from Tests.RAG_Eval.harness.environment import model_cache_dir

    from huggingface_hub import constants as hf_constants

    hf_constants.HF_HUB_OFFLINE = True
    cache_dir = model_cache_dir()
    app_config.get_model_cache_dir = lambda: cache_dir  # type: ignore[assignment]
    print(f"[isolation] model cache (read-only) -> {cache_dir}")
    print(f"[isolation] user data dir -> {app_config.get_user_data_dir()}")
    for label, getter, expected in (
        ("media", app_config.get_media_db_path, paths["media_db"]),
        ("chachanotes", app_config.get_chachanotes_db_path, paths["chacha_db"]),
        ("prompts", app_config.get_prompts_db_path, paths["prompts_db"]),
    ):
        resolved = Path(getter())
        if resolved != expected:
            raise SystemExit(
                f"{label} db override did not take: {resolved} != {expected}"
            )
    print("[isolation] all three DB paths resolve to the scratch eval DBs")
    return app_config


def build_runtime(paths: dict[str, Path]) -> Any:
    from Tests.RAG_Eval.harness.goldenset import CORPUS_PATH, load_corpus
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    corpus = load_corpus(CORPUS_PATH)
    started = time.monotonic()
    runtime = build_eval_runtime(corpus, paths["eval"])
    runtime.service.config.search.default_search_mode = "plain"
    print(
        f"[runtime] {len(corpus)} corpus docs, indexed="
        f"{runtime.index_summary['indexed']} in {time.monotonic() - started:.1f}s; "
        f"route=plain (four-seam keyword path)"
    )
    return runtime


def build_rag_provider(runtime: Any) -> Any:
    from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    return LibraryRagToolProvider(LibraryLocalRagSearchService(runtime.app))


def probe_retrieval(runtime: Any, rag_provider: Any, questions: list[dict]) -> list[dict]:
    """Run the retrieval tool for every question and inspect the projected rows.

    This is the pre-flight that costs nothing: it establishes that the target
    document is retrievable at all, that its row is LABEL-ONLY (the regime
    under measurement), that the row carries the identity `expand_document`
    requires (TASK-16174 Task 3b), and that the tool can actually fetch it
    through the config-resolved DB handles. It also records TASK-3b's two
    pre-registered suspects: (a) a chunked row expanding from the document
    HEAD because `chunk_start` is absent from the payload, and (b) a semantic
    `source_id` that is a vector-store point id and so resolves `not_found`.
    """
    import asyncio

    from tldw_chatbook.Tools.document_expansion_tool import ExpandDocumentTool

    tool = ExpandDocumentTool()
    rows_report: list[dict] = []
    for question in questions:
        expected_type, expected_id = runtime.slug_to_source[question["slug"]]
        result = rag_provider.invoke(
            "search_library_rag", {"query": question["probe_query"], "top_k": TOP_K}
        )
        record: dict[str, Any] = {
            "id": question["id"],
            "probe_query": question["probe_query"],
            "slug": question["slug"],
            "expected": [expected_type, expected_id],
            "ok": bool(result.ok),
        }
        if not result.ok:
            record["error"] = result.error
            rows_report.append(record)
            continue
        payload = json.loads(result.content)
        record["returned"] = payload.get("returned")
        record["rows"] = [
            {
                "rank": index,
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "source_type": row.get("source_type"),
                "source_id": row.get("source_id"),
                "chunk_id": row.get("chunk_id", ""),
                "expand_hint": row.get("expand_hint"),
            }
            for index, row in enumerate(payload.get("results") or (), start=1)
        ]
        hit = next(
            (
                row
                for row in record["rows"]
                if row["source_type"] == expected_type and row["source_id"] == expected_id
            ),
            None,
        )
        record["target_rank"] = hit["rank"] if hit else None
        record["target_hint"] = hit["expand_hint"] if hit else None
        record["target_snippet"] = hit["snippet"] if hit else None
        # Suspect (a): a projected row with a chunk_id expands from offset 0.
        record["chunked_rows"] = [r["rank"] for r in record["rows"] if r["chunk_id"]]
        if hit:
            expansion = asyncio.run(
                tool.execute(source_type=hit["source_type"], source_id=hit["source_id"])
            )
            record["expand_status"] = expansion["status"]
            record["expand_total_size"] = expansion["total_size"]
            record["expand_truncated"] = expansion["truncated"]
            record["oracle_in_expansion"] = bool(
                re.search(question["oracle"], expansion["text"], re.IGNORECASE)
            )
        rows_report.append(record)
    return rows_report


# --------------------------------------------------------------------------
# the live arms
# --------------------------------------------------------------------------


class SpendRecorder:
    """Wrap `chat_api_call`, inject the credential, and total the spend."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0

    def __call__(self, **kwargs: Any) -> Any:
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        kwargs.setdefault("api_key", self._api_key)
        kwargs.setdefault("temp", 0.0)
        kwargs.setdefault("max_tokens", 1024)
        response = chat_api_call(**kwargs)
        self.calls += 1
        usage = (response or {}).get("usage") or {}
        if isinstance(usage, dict):
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
            self.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)
            self.cache_read_tokens += int(usage.get("cache_read_input_tokens") or 0)
        return response

    @property
    def usd(self) -> float:
        return (
            self.input_tokens * PRICE_IN_PER_MTOK
            + self.cache_write_tokens * PRICE_IN_PER_MTOK * 1.25
            + self.cache_read_tokens * PRICE_IN_PER_MTOK * 0.1
            + self.output_tokens * PRICE_OUT_PER_MTOK
        ) / 1_000_000

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usd": round(self.usd, 6),
        }


def build_arm(paths: dict[str, Path], rag_provider: Any, chat_call: Any, arm: str) -> tuple[Any, Any, list, list[str]]:
    """Assemble one arm exactly as `console_chat_controller` does.

    Returns `(service, config, approval_rounds, offered_tool_names)`.
    """
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.builtin_tool_gate import build_builtin_gate
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    gate = build_builtin_gate(None)
    builtin = BuiltinToolProvider(gate=gate)
    offered = sorted(entry.name for entry in builtin.list_catalog())

    approval_rounds: list = []

    def request_approvals(pending: list) -> dict[str, str]:
        approval_rounds.append([row.llm_name for row in pending])
        decisions: dict[str, str] = {}
        for row in pending:
            decisions[row.llm_name] = "approve_session"
            call_id = str(getattr(row, "call_id", "") or "")
            if call_id:
                decisions[call_id] = "approve_session"
        return decisions

    registry = ToolCatalogRegistry()
    registry.register_provider(builtin)
    registry.register_provider(rag_provider)

    service = AgentService(
        db=AgentRunsDB(paths["runs_db"], client_id=f"rag-16174-{arm}"),
        registry=registry,
        chat_call=chat_call,
        review_tool_calls=build_tool_review_hook(gate, builtin, None, request_approvals),
    )
    allowed = ("search_library_rag",)
    if arm == "on":
        allowed = ("search_library_rag", "expand_document")
    config = AgentConfig(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed,
        budget=RunBudget(
            max_steps=14,
            max_model_turns=8,
            max_subagents=0,
            max_wall_seconds=420.0,
        ),
        native_tools=True,
    )
    return service, config, approval_rounds, offered


def run_arm(
    arm: str,
    paths: dict[str, Path],
    rag_provider: Any,
    questions: list[dict],
    api_key: str,
) -> dict[str, Any]:
    from tldw_chatbook.Agents.agent_models import STEP_TOOL_CALL, STEP_TOOL_RESULT

    recorder = SpendRecorder(api_key)
    service, config, approval_rounds, offered = build_arm(paths, rag_provider, recorder, arm)
    print(f"[arm {arm}] builtin catalog offers: {offered}")
    print(f"[arm {arm}] allowed_tools: {list(config.allowed_tools)}")

    results = []
    for question in questions:
        started = time.monotonic()
        run_id, outcome = service.run_turn(
            conversation_id=f"rag-16174-{arm}-{question['id']}",
            messages=[{"role": "user", "content": question["question"]}],
            config=config,
            api_endpoint="anthropic",
        )
        answer = outcome.final_text or ""
        hit = bool(re.search(question["oracle"], answer, re.IGNORECASE))
        calls = [
            {"name": step.tool_name, "args": step.args}
            for step in outcome.steps
            if step.kind == STEP_TOOL_CALL
        ]
        tool_results = [
            {"name": step.tool_name, "result": (step.result or "")[:400]}
            for step in outcome.steps
            if step.kind == STEP_TOOL_RESULT
        ]
        queries = [
            str((call.get("args") or {}).get("query") or "")
            for call in calls
            if call["name"] == "search_library_rag"
        ]
        results.append(
            {
                "id": question["id"],
                "queries": queries,
                "slug": question["slug"],
                "oracle": question["oracle"],
                "hit": hit,
                "status": outcome.status,
                "answer": answer,
                "tool_calls": calls,
                "tool_results": tool_results,
                "seconds": round(time.monotonic() - started, 1),
                "run_id": run_id,
            }
        )
        marker = "HIT " if hit else "miss"
        names = ",".join(call["name"] for call in calls) or "-"
        print(f"  [{arm}] {question['id']:24s} {marker} tools={names} status={outcome.status}")
    return {
        "arm": arm,
        "offered_tools": offered,
        "allowed_tools": list(config.allowed_tools),
        "approval_rounds": approval_rounds,
        "results": results,
        "spend": recorder.snapshot(),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def render_table(questions: list[dict], off: dict, on: dict) -> str:
    by_id_off = {row["id"]: row for row in off["results"]}
    by_id_on = {row["id"]: row for row in on["results"]}
    lines = [
        "| question | target (label-only row) | oracle | tool-OFF | tool-ON | ON tool calls |",
        "|---|---|---|---|---|---|",
    ]
    for question in questions:
        left = by_id_off[question["id"]]
        right = by_id_on[question["id"]]
        calls = ", ".join(call["name"] for call in right["tool_calls"]) or "-"
        lines.append(
            f"| `{question['id']}` | `{question['slug']}` ({question['source_type']}) "
            f"| `/{question['oracle']}/` "
            f"| {'HIT' if left['hit'] else 'miss'} "
            f"| {'HIT' if right['hit'] else 'miss'} | {calls} |"
        )
    off_score = sum(1 for row in off["results"] if row["hit"])
    on_score = sum(1 for row in on["results"] if row["hit"])
    total = len(questions)
    lines.append(
        f"| **TOTAL** | | | **{off_score}/{total}** | **{on_score}/{total}** | |"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-oracles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-billable", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions (smoke test)")
    parser.add_argument("--scratch", default="")
    parser.add_argument("--out", default=str(HERE / "run-artifacts.json"))
    args = parser.parse_args()

    if args.verify_oracles:
        return verify_oracles()
    if not (args.dry_run or args.live):
        parser.error("pass --verify-oracles, --dry-run, or --live")
    if args.live and not args.confirm_billable:
        parser.error("--live spends real money; pass --confirm-billable too")

    api_key = ""
    if args.live:
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            if not API_KEY_PATH.exists():
                raise SystemExit(
                    f"no credential: set ${API_KEY_ENV} or place a key at "
                    f"{API_KEY_PATH}"
                )
            api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()
        if not api_key:
            raise SystemExit("the repo-root API key file is empty")

    real_config = real_home() / ".config" / "tldw_cli" / "config.toml"
    before = sha256_of(real_config)
    print(f"[isolation] real config {real_config} sha256(before)={before}")

    import tempfile

    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="rag16174-oracle-"))
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[isolation] scratch profile -> {scratch}")
    paths = prepare_scratch(scratch, gate_on=False)

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
        print(f"[limit] running the first {len(questions)} question(s) only")
    app_config = import_app(paths)
    runtime = build_runtime(paths)
    artifacts: dict[str, Any] = {
        "model": MODEL,
        "route": "plain (four-seam keyword path)",
        "questions": questions,
        "scratch": str(scratch),
        "pid": os.getpid(),
    }
    try:
        rag_provider = build_rag_provider(runtime)
        artifacts["retrieval_probe"] = probe_retrieval(runtime, rag_provider, questions)
        for record in artifacts["retrieval_probe"]:
            print(
                f"  [probe] {record['id']:24s} rank={record.get('target_rank')} "
                f"hint={(record.get('target_hint') or {}).get('reason')} "
                f"expand={record.get('expand_status')} "
                f"oracle_in_expansion={record.get('oracle_in_expansion')} "
                f"chunked_rows={record.get('chunked_rows')}"
            )

        if args.live:
            from tldw_chatbook.config import get_cli_setting, load_settings

            gate_off = get_cli_setting("tools", "expand_document_enabled", False)
            print(f"[gate] before OFF arm: expand_document_enabled={gate_off!r}")
            artifacts["off"] = run_arm("off", paths, rag_provider, questions, api_key)

            write_scratch_config(paths, gate_on=True)
            app_config._invalidate_config_caches()
            load_settings(force_reload=True)
            gate_on = get_cli_setting("tools", "expand_document_enabled", False)
            print(f"[gate] before ON arm: expand_document_enabled={gate_on!r}")
            if not gate_on:
                raise SystemExit("the [tools] gate did not flip on; aborting the ON arm")
            artifacts["on"] = run_arm("on", paths, rag_provider, questions, api_key)

            table = render_table(questions, artifacts["off"], artifacts["on"])
            artifacts["table"] = table
            total = artifacts["off"]["spend"]["usd"] + artifacts["on"]["spend"]["usd"]
            artifacts["total_usd"] = round(total, 6)
            print("\n" + table)
            print(f"\nspend OFF: {artifacts['off']['spend']}")
            print(f"spend ON:  {artifacts['on']['spend']}")
            print(f"TOTAL USD: {total:.4f}")
    finally:
        runtime.close()

    after = sha256_of(real_config)
    artifacts["real_config_sha256"] = {"before": before, "after": after}
    print(f"[isolation] real config sha256(after)={after} unchanged={before == after}")

    Path(args.out).write_text(json.dumps(artifacts, indent=2, default=str), encoding="utf-8")
    print(f"[artifacts] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
