"""Test-only synthetic qualification tooling; production imports stay deferred."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

FIXTURE_VERSION = "task31245-rebuilt-v1"
MANIFEST_PATH = Path(__file__).with_name("character_qualification_queries.json")


def reserve_root(path: Path) -> Path:
    """Reserve a new explicit disposable directory."""
    if not path.is_absolute():
        raise ValueError("An absolute disposable root is required")
    resolved = path.resolve()
    temporary_roots = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
    if not any(
        resolved != base and resolved.is_relative_to(base) for base in temporary_roots
    ):
        raise ValueError("The destination must be below an OS temporary directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Never reuse an existing qualification destination")
    # No parent creation: the operator explicitly selects an existing container.
    resolved.mkdir(mode=0o700)
    (resolved / ".qualification-owned").write_text(FIXTURE_VERSION)
    return resolved


def isolated_environment(root: Path) -> dict[str, str]:
    """Describe an offline private process environment before app imports."""
    if (root / ".qualification-owned").read_text() != FIXTURE_VERSION:
        raise ValueError("Root was not reserved by this fixture version")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "LANG", "TERM", "COLORTERM"} or name.startswith("LC_")
    }
    paths = {
        "HOME": root / "home",
        "USERPROFILE": root / "home",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_CACHE_HOME": root / "cache",
        "HF_HOME": root / "cache" / "huggingface",
        "TIKTOKEN_CACHE_DIR": root / "cache" / "tiktoken",
        "TMPDIR": root / "tmp",
    }
    for path in paths.values():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment.update({name: str(path) for name, path in paths.items()})
    environment.update(
        {
            "TLDW_CONFIG_PATH": str(root / "config" / "config.toml"),
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def verify_source(repository: Path, expected_head: str) -> str:
    """Refuse an unfrozen or different source revision."""

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    if head != expected_head:
        raise ValueError("Source head differs from the frozen expected head")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Source must be clean, including untracked files")
    return head


def load_manifest() -> list[dict[str, Any]]:
    """Load the independently declared 30-query oracle."""
    return json.loads(MANIFEST_PATH.read_text())


def build_corpus(
    root: Path, *, conversations: int, messages_per_chat: int
) -> dict[str, Any]:
    """Build selected-branch synthetic chats using the installed production APIs."""
    if (root / ".qualification-owned").read_text() != FIXTURE_VERSION:
        raise ValueError("Unreserved fixture root")
    if type(conversations) is not int or not 30 <= conversations <= 10000:
        raise ValueError("Conversation count must be 30..10000")
    if type(messages_per_chat) is not int or not 3 <= messages_per_chat <= 25:
        raise ValueError("Message count must be 3..25")
    path = root / "corpus.sqlite"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Never overwrite a corpus")
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    receipt: dict[str, Any] = {
        "fixture_version": FIXTURE_VERSION,
        "status": "failed",
        "conversations": conversations,
        "eligible_messages": conversations * messages_per_chat,
    }
    database = None
    try:
        started = time.perf_counter()
        database = CharactersRAGDB(path, client_id="character-qualification")
        _seed(database, conversations, messages_per_chat)
        receipt["seed_seconds"] = time.perf_counter() - started
        service = CharacterConversationNavigationService(database)
        started = time.perf_counter()
        status = service.ensure_keyword_index().value
        receipt["build_seconds"] = time.perf_counter() - started
        receipt["index_status"] = status
        if status != "ready":
            raise RuntimeError("Synthetic Keyword generation was not ready")
        connection = database.get_connection()
        counts = [
            int(connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0])
            for table in (
                "conversations",
                "messages",
                "character_conversation_search_documents",
            )
        ]
        receipt["counts"] = counts
        if counts != [
            conversations,
            conversations * messages_per_chat + 4,
            conversations,
        ]:
            raise RuntimeError("Fixture counts do not match the prescribed corpus")
        eligible = sum(
            len(row[0].split("\n\n"))
            for row in connection.execute(
                "SELECT body FROM character_conversation_search_documents"
            ).fetchall()
        )
        if eligible != conversations * messages_per_chat:
            raise RuntimeError("Selected-branch message count differs")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("Corpus integrity failed")
        receipt["integrity"] = "ok"
        receipt["status"] = "built"
    except BaseException as error:
        receipt["exception_type"] = type(error).__name__
        raise
    finally:
        if database is not None:
            database.close()
            receipt["registered_handles_after_cleanup"] = (
                database.registered_connection_count()
            )
        if path.exists():
            receipt["corpus_digest"] = file_digest(path)
        (root / "build-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )
    return receipt


def file_digest(path: Path) -> str:
    """Hash a fixture without loading the full database into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _seed(database: Any, conversations: int, messages_per_chat: int) -> None:
    """Create one selected chain per chat and six independently tested exclusions."""
    from tldw_chatbook.Chat.thinking_blocks import (
        DisplayableThinkingBlock,
        ThinkingEnvelope,
        dump_thinking_blocks_json,
    )

    card_id = database.add_character_card(
        {
            "name": "SYNTHETIC Fixture character",
            "description": "Disposable qualification data; no provider use.",
        }
    )
    authority = database.get_local_authority_id()
    queries = load_manifest()
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    for ordinal in range(conversations):
        conversation_id = f"perf-{ordinal:05d}"
        case = queries[ordinal] if ordinal < 30 else None
        title_token = case["query"] if case and case["category"] == "title" else ""
        document_token = case["document"] if case else "ordinaryfixturetext"
        with database.transaction(immediate=True) as connection:
            database.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": card_id,
                    "assistant_kind": "character",
                    "assistant_id": str(card_id),
                    "assistant_authority_id": authority,
                    "title": f"Fixture {ordinal:05d} {title_token}".rstrip(),
                }
            )
            parent = None
            for position in range(messages_per_chat):
                message_id = f"{conversation_id}-m{position:02d}"
                role = "user" if position % 2 == 0 else "assistant"
                content = f"Visible message {position} for Fixture {ordinal:05d}"
                if position == 0:
                    content += f" {document_token}"
                database.add_message(
                    {
                        "id": message_id,
                        "conversation_id": conversation_id,
                        "parent_message_id": parent,
                        "role": role,
                        "sender": role,
                        "content": content,
                        "timestamp": (
                            epoch
                            + timedelta(seconds=ordinal, microseconds=position * 1000)
                        ).isoformat(),
                    }
                )
                parent = message_id
                if ordinal == 0 and position == 1:
                    thinking = dump_thinking_blocks_json(
                        ThinkingEnvelope(
                            blocks=(
                                DisplayableThinkingBlock(
                                    block_id="fixture-thinking",
                                    round_ordinal=0,
                                    provider="local",
                                    model="synthetic",
                                    protocol="openai_chat",
                                    source_format="start_anchored_think",
                                    status="complete",
                                    text="THINKING_CANARY",
                                ),
                            )
                        )
                    )
                    database.update_message_with_attachments(
                        message_id,
                        {"thinking_blocks_json": thinking},
                        expected_version=1,
                        attachments=(
                            {
                                "position": 1,
                                "data": b"ATTACHMENT_CANARY",
                                "mime_type": "text/plain",
                                "display_name": "ATTACHMENT_CANARY",
                            },
                        ),
                        preserve_descendants=True,
                    )
                    for role, canary in (
                        ("system", "SYSTEM_CANARY"),
                        ("tool", "TOOL_CANARY"),
                    ):
                        excluded_id = f"fixture-{role}"
                        database.add_message(
                            {
                                "id": excluded_id,
                                "conversation_id": conversation_id,
                                "parent_message_id": parent,
                                "role": role,
                                "sender": role,
                                "content": canary,
                                "timestamp": (
                                    epoch
                                    + timedelta(
                                        microseconds=1500 if role == "system" else 1700
                                    )
                                ).isoformat(),
                            }
                        )
                        parent = excluded_id
            if ordinal == 0:
                for excluded_id, text in (
                    ("fixture-branch", "NON_SELECTED_CANARY"),
                    ("fixture-deleted", "DELETED_CANARY"),
                ):
                    database.add_message(
                        {
                            "id": excluded_id,
                            "conversation_id": conversation_id,
                            "parent_message_id": "perf-00000-m00",
                            "role": "assistant",
                            "sender": "assistant",
                            "content": text,
                            "timestamp": (
                                epoch
                                + timedelta(
                                    microseconds=500
                                    if excluded_id == "fixture-branch"
                                    else 700
                                )
                            ).isoformat(),
                        }
                    )
                database.soft_delete_message("fixture-deleted", expected_version=1)
            database.set_conversation_active_leaf(conversation_id, parent)
            # Fixture-only date normalization, with all production triggers intact.
            # No raw message writes or semantic guard bypasses.
            stamp = (epoch + timedelta(seconds=ordinal)).isoformat()
            connection.execute(
                "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
                (stamp, stamp, conversation_id),
            )


async def measure_keyword(
    root: Path,
    corpus: Path,
    build_receipt: Path,
    *,
    expected_head: str,
    repetitions: int = 10,
    warmups: int = 5,
) -> dict[str, Any]:
    """Measure a source-verified copy, without modifying the original corpus."""
    verify_disposable_source(corpus, build_receipt)
    if (root / ".qualification-owned").read_text() != FIXTURE_VERSION:
        raise ValueError("Unreserved measurement root")
    source = json.loads(build_receipt.read_text())
    digest = file_digest(corpus)
    if source["status"] != "built" or source["fixture_version"] != FIXTURE_VERSION:
        raise ValueError("A built current-version source receipt is required")
    if digest != source["corpus_digest"]:
        raise ValueError("Corpus digest differs from its build receipt")
    source_bound = source.get("head") == expected_head
    if source.get("head") is not None and not source_bound:
        raise ValueError("Corpus belongs to a different source head")
    if not source_bound and source["counts"] != [30, 94, 30]:
        raise ValueError("Scale evidence requires an explicit matching source head")
    if (
        type(repetitions) is not int
        or not 1 <= repetitions <= 10
        or type(warmups) is not int
        or not 0 <= warmups <= 5
    ):
        raise ValueError("Invalid bounded repetition count")
    destination = root / "measurement.sqlite"
    _copy_checkpointed(corpus, destination)
    queries = load_manifest()
    evidence: dict[str, Any] = {
        "status": "failed",
        "source_head_bound": source_bound,
        "head": expected_head,
        "fixture_version": FIXTURE_VERSION,
        "corpus_digest": digest,
        "manifest_digest": hashlib.sha256(
            json.dumps(queries, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "queries": queries,
        "conversations": source["conversations"],
        "eligible_messages": source["eligible_messages"],
        "measurements_per_query": repetitions,
        "warmups_per_query": warmups,
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "machine": platform.machine(),
        },
        "mode": "Keyword; no embedding model",
        "model_digest": None,
        "limitations": "Standalone Keyword worker and event-loop responsiveness only; not paint, native, Windows, human or Meaning evidence. Tiny runs are smoke only.",
        "limits_ms": {"warm_p95": 300.0, "event_loop_max": 50.0},
        "correctness_failures": [],
        "timings": [],
        "event_loop_intervals_ns": [],
    }
    measuring = False
    ended_ns = 0
    previous_ns = 0
    failure = None
    stop_requested = threading.Event()

    def measure() -> None:
        nonlocal measuring, ended_ns, previous_ns
        from tldw_chatbook.Character_Chat.character_conversation_navigation import (
            CharacterConversationNavigationService,
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        database = CharactersRAGDB(
            destination, client_id="character-keyword-measurement"
        )
        try:
            service = CharacterConversationNavigationService(database)
            evidence["index_status"] = service.keyword_index_status().value
            if evidence["index_status"] != "ready":
                raise RuntimeError("Copied generation is not ready")
            previous_ns = time.perf_counter_ns()
            measuring = True
            for query in queries:
                for repetition in range(-warmups, repetitions):
                    if stop_requested.is_set():
                        return
                    started = time.perf_counter_ns()
                    page = service.keyword_search(query["query"])
                    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                    actual = [row.target.conversation_id for row in page.rows]
                    if actual != query["expected"]:
                        evidence["correctness_failures"].append(
                            {
                                "id": query["id"],
                                "repetition": repetition,
                                "actual": actual,
                                "expected": query["expected"],
                            }
                        )
                    if repetition >= 0:
                        evidence["timings"].append(
                            {
                                "query": query["id"],
                                "repetition": repetition,
                                "ms": elapsed_ms,
                            }
                        )
            ended_ns = time.perf_counter_ns()
            measuring = False
            for canary in (
                "SYSTEM_CANARY",
                "TOOL_CANARY",
                "NON_SELECTED_CANARY",
                "DELETED_CANARY",
                "THINKING_CANARY",
                "ATTACHMENT_CANARY",
            ):
                if service.keyword_search(canary).rows:
                    evidence["correctness_failures"].append({"excluded_canary": canary})
            if (
                database.get_connection().execute("PRAGMA quick_check").fetchone()[0]
                != "ok"
            ):
                raise RuntimeError("Copied corpus integrity failed")
            evidence["integrity"] = "ok"
        finally:
            ended_ns = ended_ns or time.perf_counter_ns()
            measuring = False
            database.close()
            evidence["registered_handles_after_cleanup"] = (
                database.registered_connection_count()
            )

    async def sentinel() -> None:
        nonlocal previous_ns
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter_ns()
            if measuring:
                evidence["event_loop_intervals_ns"].append(now - previous_ns)
                previous_ns = now

    watcher = asyncio.create_task(sentinel())
    worker = asyncio.create_task(asyncio.to_thread(measure))
    try:
        await asyncio.shield(worker)
    except BaseException as error:  # noqa: BLE001 - drain exact worker, record, then rethrow
        failure = error
        evidence["exception_type"] = type(error).__name__
        stop_requested.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception as worker_error:  # noqa: BLE001 - retain drained failure before rethrowing original
                evidence["worker_exception_type"] = type(worker_error).__name__
                break  # The worker's exception is retrieved below.
        if not worker.cancelled() and worker.exception() is not None:
            evidence["worker_exception_type"] = type(worker.exception()).__name__
    finally:
        if previous_ns and ended_ns >= previous_ns:
            evidence["event_loop_intervals_ns"].append(ended_ns - previous_ns)
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        evidence["event_loop_max_gap_ms"] = (
            max(evidence["event_loop_intervals_ns"], default=0) / 1e6
        )
        ordered = sorted(row["ms"] for row in evidence["timings"])
        # Nearest-rank P95 over the complete retained sample population.
        evidence["warm_p95_ms"] = (
            ordered[(95 * len(ordered) + 99) // 100 - 1] if ordered else None
        )
        evidence["source_unchanged"] = file_digest(corpus) == digest
        evidence["owned_database_descriptors_after_cleanup"] = owned_descriptors(root)
        evidence["failures"] = []
        if evidence["correctness_failures"] or len(ordered) != 30 * repetitions:
            evidence["failures"].append("Query correctness or completeness failed")
        if (
            not ordered
            or evidence["warm_p95_ms"] > 300.0
            or evidence["event_loop_max_gap_ms"] > 50.0
        ):
            evidence["failures"].append("Keyword or event-loop latency limit exceeded")
        if (
            not evidence["source_unchanged"]
            or evidence.get("registered_handles_after_cleanup") != 0
            or evidence["owned_database_descriptors_after_cleanup"] != []
        ):
            evidence["failures"].append(
                "Integrity or terminal ownership failed/unavailable"
            )
        if source_bound:
            try:
                verify_source(Path(__file__).resolve().parents[2], expected_head)
                evidence["source_clean_and_exact_after"] = True
            except ValueError:
                evidence["source_clean_and_exact_after"] = False
                evidence["failures"].append(
                    "Source is no longer clean and exact at measurement completion"
                )
        if failure is None and not evidence["failures"]:
            evidence["status"] = (
                "passed"
                if source["counts"] == [10000, 250004, 10000]
                and repetitions == 10
                and warmups == 5
                else "smoke"
            )
        (root / "keyword-receipt.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True)
        )
    if failure is not None:
        raise failure
    if evidence["failures"]:
        raise RuntimeError(
            "Keyword qualification failed; retained receipt contains all samples"
        )
    return evidence


def _copy_checkpointed(source: Path, destination: Path) -> None:
    """Backup an immutable, checkpointed source into a new owned file."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Never overwrite a measurement database")
    wal = Path(str(source) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source corpus must be checkpointed before copying")
    with (
        closing(
            sqlite3.connect(
                source.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
            )
        ) as original,
        closing(sqlite3.connect(destination)) as copied,
    ):
        original.backup(copied)


def verify_disposable_source(corpus: Path, receipt: Path) -> None:
    """Refuse non-fixture source paths before reading database or receipt bytes."""
    temporary_roots = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
    for path in (corpus, receipt):
        resolved = path.resolve()
        if not path.is_absolute() or not any(
            resolved.is_relative_to(base) for base in temporary_roots
        ):
            raise ValueError("Source must be in disposable fixture storage")
        if path.is_symlink():
            raise ValueError("Source must not be a substituted link")
        if (resolved.parent / ".qualification-owned").read_text() != FIXTURE_VERSION:
            raise ValueError("Source must be in a reserved disposable fixture")


def owned_descriptors(root: Path) -> list[str] | None:
    """Read this process's owned regular-file descriptors; never close them."""
    if sys.platform not in {"darwin", "linux"}:
        return None  # Missing observer is not a retirement pass.
    result = []
    directory = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
    for raw in os.listdir(directory):
        try:
            if sys.platform == "darwin":
                import fcntl

                target = os.fsdecode(
                    fcntl.fcntl(int(raw), 50, bytes(1024)).split(b"\0", 1)[0]
                )
            else:
                target = os.readlink(f"{directory}/{raw}")
            path = Path(target)
            if path.is_relative_to(root.resolve()) and path.name.endswith(
                (
                    ".sqlite",
                    ".sqlite-wal",
                    ".sqlite-shm",
                    ".db",
                    ".db-wal",
                    ".db-shm",
                    ".sqlite3",
                    ".sqlite3-wal",
                    ".sqlite3-shm",
                )
            ):
                result.append(path.name)
        except (OSError, ValueError):
            continue
    return sorted(result)


def _bootstrap(root: Path) -> None:
    """Establish private paths, no credential inheritance and denied network."""
    if any(
        name == "tldw_chatbook" or name.startswith("tldw_chatbook.")
        for name in sys.modules
    ):
        raise RuntimeError("Production was imported before the qualification guard")
    from Tests import network_guard, real_profile_guard

    real_profile_guard.install()
    environment = isolated_environment(root)
    os.environ.clear()
    os.environ.update(environment)
    Path(os.environ["TLDW_CONFIG_PATH"]).write_text(
        Path(__file__).with_name("character_qualification_config.toml").read_text()
    )
    network_guard.install()


def _final_source_guard(root: Path, repository: Path, expected_head: str) -> None:
    """Revalidate source at CLI settlement and invalidate any retained receipts."""
    failure = None
    try:
        verify_source(repository, expected_head)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        failure = error
    for name in (
        "build-receipt.json",
        "keyword-receipt.json",
        "native-receipt.json",
        "native-return.json",
        "ui-evidence/ui-latency-evidence.json",
    ):
        path = root / name
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        receipt["source_clean_and_exact_after"] = failure is None
        if failure is not None:
            receipt["status"] = "failed"
            receipt.setdefault("failures", []).append(
                "Source is no longer clean and exact at CLI completion"
            )
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if failure is not None:
        raise failure


def main() -> None:
    """Explicit CLI entry: no production import until source/path guards pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("build", "keyword", "ui", "prepare-native", "native")
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="New absolute directory below the OS temporary directory",
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--size", choices=("tiny", "scale"))
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    args = parser.parse_args()
    head = verify_source(Path(__file__).resolve().parents[2], args.expected_head)
    if args.mode == "build" and args.size is None:
        parser.error("build requires --size tiny or scale")
    if args.mode not in {"build", "prepare-native"} and (
        args.corpus is None or args.source_receipt is None
    ):
        parser.error("measurement requires --corpus and --source-receipt")
    if args.mode not in {"build", "prepare-native"}:
        verify_disposable_source(args.corpus, args.source_receipt)
    root = reserve_root(args.root)
    _bootstrap(root)
    from loguru import logger

    logger.remove()
    logger.add(root / "qualification.log", level="INFO")
    try:
        _execute(args, root, head)
    finally:
        prior_failure = sys.exception()
        try:
            _final_source_guard(root, Path(__file__).resolve().parents[2], head)
        except (ValueError, OSError, subprocess.CalledProcessError):
            if prior_failure is None:
                raise
    print(f"Retained qualification artifacts: {root}")


def _execute(args: argparse.Namespace, root: Path, head: str) -> None:
    """Execute one guarded mode; the caller owns the final source receipt fence."""
    if args.mode == "build":
        receipt = build_corpus(
            root,
            conversations=10000 if args.size == "scale" else 30,
            messages_per_chat=25 if args.size == "scale" else 3,
        )
        receipt["head"] = head
        verify_source(Path(__file__).resolve().parents[2], head)
        (root / "build-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )
    elif args.mode == "keyword":
        asyncio.run(
            measure_keyword(
                root,
                args.corpus.resolve(),
                args.source_receipt.resolve(),
                expected_head=head,
            )
        )
    elif args.mode == "ui":
        source = json.loads(args.source_receipt.read_text())
        if (
            source.get("head") != head
            or source.get("fixture_version") != FIXTURE_VERSION
            or source.get("corpus_digest") != file_digest(args.corpus)
        ):
            raise ValueError(
                "UI qualification requires a verified current-head corpus receipt"
            )
        from Tests.Benchmarks.console_character_switcher_latency import run

        os.environ["TLDW_TASK5_UI_LATENCY_GUARDED"] = "1"
        asyncio.run(
            run(
                corpus=args.corpus,
                manifest_evidence=args.source_receipt,
                output_dir=root / "ui-evidence",
                profile_root=root,
                expected_head=head,
            )
        )
    elif args.mode == "prepare-native":
        from Tests.Benchmarks.character_native_fixture import build_native

        receipt = build_native(root)
        verify_source(Path(__file__).resolve().parents[2], head)
        receipt["head"] = head
        (root / "native-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )
    else:
        from Tests.Benchmarks.character_native_fixture import launch_native

        launch_native(root, args.corpus, args.source_receipt, expected_head=head)


if __name__ == "__main__":
    main()
