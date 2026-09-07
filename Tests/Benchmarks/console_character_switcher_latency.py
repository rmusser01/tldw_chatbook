"""Opt-in, guarded production-owner UI latency probe; never seeds a corpus.

The launcher must supply a fresh, offline disposable profile before importing
production modules. This measures Textual compositor paint, not native output.
Run only when other native/benchmark/test activity has stopped. Example entry:
``python -m Tests.Benchmarks.console_character_switcher_latency --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sqlite3
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUSY_LIMIT_MS = 100.0
LOOP_LIMIT_MS = 50.0
READY_TIMEOUT = 60.0
OPERATION_TIMEOUT = 15.0


@dataclass
class PaintWindow:
    """One event-post lifetime, fed only by actual current-screen refreshes."""

    operation: str
    query: str
    posted_ns: int
    screen: object | None = field(default=None, repr=False)
    busy_ns: int | None = None
    busy_text: str = ""
    settled_ns: int | None = None
    settled_text: str = ""
    gaps_ns: list[int] = field(default_factory=list)
    observer_ns: list[int] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, text: str, now_ns: int, *, screen: object) -> None:
        """Record the first matching frame, never merely a status assignment."""
        if self.screen is not None and screen is not self.screen:
            return
        self.screen = screen
        labels = (
            ("Opening…", "Finishing…")
            if self.operation == "activation"
            else ("Searching local chats…", "Loading local chats…")
        )
        if self.busy_ns is None and any(label in text for label in labels):
            self.busy_ns = now_ns
            self.busy_text = text

    def summary(self) -> dict[str, Any]:
        """Serialize raw observations and separate busy from total latency."""
        return {
            "operation": self.operation,
            "query": self.query,
            "posted_ns": self.posted_ns,
            "busy_ns": self.busy_ns,
            "busy_ms": (
                (self.busy_ns - self.posted_ns) / 1e6
                if self.busy_ns is not None
                else None
            ),
            "settled_ns": self.settled_ns,
            "settled_ms": (
                (self.settled_ns - self.posted_ns) / 1e6
                if self.settled_ns is not None
                else None
            ),
            "event_loop_intervals_ns": self.gaps_ns,
            "event_loop_max_gap_ms": max(self.gaps_ns, default=0) / 1e6,
            "observer_durations_ns": self.observer_ns,
            "calls": self.calls,
        }

    def failures(self) -> list[str]:
        """Keep missing paint and either maximum-threshold breach as failures."""
        result = []
        busy = self.summary()["busy_ms"]
        if busy is None:
            result.append("No busy frame observed")
        elif busy > BUSY_LIMIT_MS:
            result.append(f"Busy paint {busy:.3f}ms > {BUSY_LIMIT_MS}ms")
        gap = max(self.gaps_ns, default=0) / 1e6
        if gap > LOOP_LIMIT_MS:
            result.append(f"Event-loop interval {gap:.3f}ms > {LOOP_LIMIT_MS}ms")
        return result


def _accepted_modal_closed(app: Any, modal: Any, chat: Any) -> bool:
    """Check the actual removal boundary before recording activation settled."""
    return (
        app.screen is chat
        and modal not in app.screen_stack
        and not app.is_mounted(modal)
    )


def _observe_commit_waiter(original: Any, current_window: Any) -> Any:
    """Observe the real acknowledgement without substituting its owner."""

    async def observed(*args: Any, **kwargs: Any) -> None:
        window = current_window()
        try:
            await original(*args, **kwargs)
        except BaseException as error:
            if window is not None:
                window.calls.append(
                    {
                        "kind": "commit_waiter_error",
                        "exception_type": type(error).__name__,
                        "at_ns": time.perf_counter_ns(),
                    }
                )
            raise
        if window is not None:
            window.calls.append(
                {"kind": "commit_started", "at_ns": time.perf_counter_ns()}
            )

    return observed


async def _wait(predicate: Any, label: str, timeout: float = OPERATION_TIMEOUT) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)
    except TimeoutError as error:
        raise TimeoutError(f"Timed out waiting for {label}") from error


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _descriptors() -> dict[str, Any]:
    """Read descriptor types/basenames only; never close or collect owners."""
    import fcntl

    types: dict[str, int] = {}
    names: dict[str, int] = {}
    for raw in os.listdir("/dev/fd"):
        try:
            mode = os.fstat(int(raw)).st_mode
            kind = (
                "regular"
                if stat.S_ISREG(mode)
                else "socket"
                if stat.S_ISSOCK(mode)
                else "fifo"
                if stat.S_ISFIFO(mode)
                else "other"
            )
            types[kind] = types.get(kind, 0) + 1
            if kind == "regular" and platform.system() == "Darwin":
                value = fcntl.fcntl(int(raw), 50, bytes(1024))
                name = os.path.basename(value.split(b"\0", 1)[0].decode())
                names[name] = names.get(name, 0) + 1
        except OSError:
            continue
    return {"types": types, "regular_basenames": names}


def _copy_corpus(source: Path, destination: Path) -> None:
    """Copy a checkpointed ready corpus without opening it for mutation."""
    assert source.is_file() and not destination.exists()
    wal = Path(str(source) + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0, "Source must be checkpointed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)
    writer = sqlite3.connect(destination)
    try:
        reader.backup(writer)
    finally:
        writer.close()
        reader.close()
    assert _inspect_copy(destination) == (10000, 250004, 10000)


def _inspect_copy(destination: Path) -> tuple[int, int, int]:
    """Check copied schema integrity and return the three scale counts."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(destination, client_id="task5-ui-copy-validation")
    try:
        connection = database.get_connection()
        assert [row[0] for row in connection.execute("PRAGMA quick_check")] == ["ok"]
        return (
            connection.execute("SELECT count(*) FROM conversations").fetchone()[0],
            connection.execute("SELECT count(*) FROM messages").fetchone()[0],
            connection.execute(
                "SELECT count(*) FROM character_conversation_search_documents"
            ).fetchone()[0],
        )
    finally:
        # This entire validation runs on one worker; close only its own handle.
        database.close()


def _keyword_evidence(database: Any) -> dict[str, Any]:
    """Read one real database snapshot; never start maintenance from a probe."""
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )

    service = CharacterConversationNavigationService(database)
    with database.transaction() as connection:
        return {
            "database_path": str(database.db_path),
            "status": str(service.keyword_index_status()),
            "revision": connection.execute(
                "SELECT data_revision FROM character_conversation_search_revision"
            ).fetchone()[0],
            "generations": [
                dict(row)
                for row in connection.execute(
                    "SELECT generation_id,data_authority_id,source_revision,status "
                    "FROM character_conversation_search_generations ORDER BY rowid"
                )
            ],
            "dirty_conversations": connection.execute(
                "SELECT count(*) FROM character_conversation_search_dirty"
            ).fetchone()[0],
            "counts": [
                connection.execute("SELECT count(*) FROM conversations").fetchone()[0],
                connection.execute("SELECT count(*) FROM messages").fetchone()[0],
                connection.execute(
                    "SELECT count(*) FROM character_conversation_search_documents"
                ).fetchone()[0],
            ],
        }


def _finalize_evidence(
    evidence: dict[str, Any], failure: BaseException | None, corpus_digest: str
) -> BaseException | None:
    """Classify final source integrity before the durable receipt is written."""
    evidence["corpus_sha256_after"] = corpus_digest
    evidence["source_unchanged"] = evidence["corpus_sha256_before"] == corpus_digest
    if not evidence["source_unchanged"]:
        evidence["failures"].append("Controller corpus changed")
        if failure is None:
            failure = AssertionError("Controller corpus changed")
    evidence["status"] = "failed" if failure is not None else "passed"
    if failure is not None:
        evidence["error"] = f"{type(failure).__name__}: {failure}"
    return failure


async def run(
    *,
    corpus: Path,
    manifest_evidence: Path,
    output_dir: Path,
    profile_root: Path,
    expected_head: str,
) -> dict[str, Any]:
    """Exercise real app callbacks against one private ready-corpus copy.

    Args:
        corpus: Controller-owned checkpointed scale-v2 database, read-only.
        manifest_evidence: Successful versioned standalone benchmark receipt.
        output_dir: New evidence directory inside the disposable profile.
        profile_root: Guarded profile containing all configured mutable paths.
        expected_head: Exact approved source revision for this measurement.

    Returns:
        JSON-compatible receipt, also written on failure after cleanup.

    Raises:
        AssertionError: Guard, corpus, exact-acceptance or timing gate fails.
        TimeoutError: A real UI operation does not settle within its bound.
    """
    assert os.environ.get("TLDW_TASK5_UI_LATENCY_GUARDED") == "1"
    corpus, manifest_evidence = corpus.resolve(), manifest_evidence.resolve()
    output_dir, profile_root = output_dir.resolve(), profile_root.resolve()
    assert output_dir.is_relative_to(profile_root) and not output_dir.exists()
    assert not corpus.is_relative_to(profile_root), "Never reuse the source profile"
    source_root = Path(__file__).resolve().parents[2]
    head = (
        await asyncio.to_thread(
            subprocess.check_output,
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        )
    ).strip()
    assert head == expected_head
    source_evidence = json.loads(manifest_evidence.read_text())
    assert source_evidence["status"] == "passed"
    assert source_evidence["conversations"] == 10000
    assert source_evidence["eligible_messages"] == 250000
    queries = source_evidence["queries"]
    assert len(queries) == 30 and len({item["id"] for item in queries}) == 30
    assert (
        hashlib.sha256(
            json.dumps(queries, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        == source_evidence["manifest_digest"]
    )
    output_dir.mkdir(parents=True)
    evidence: dict[str, Any] = {
        "head": head,
        "pid": os.getpid(),
        "fixture_version": source_evidence["fixture_version"],
        "corpus_sha256_before": _digest(corpus),
        "manifest_evidence_sha256": _digest(manifest_evidence),
        "manifest_digest": source_evidence["manifest_digest"],
        "standalone_host": source_evidence["host"],
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "machine": platform.machine(),
        },
        "descriptors_before_app": _descriptors(),
        "samples": [],
        "failures": [],
        "viewports": [],
        "limits_ms": {"busy_max": BUSY_LIMIT_MS, "event_loop_max": LOOP_LIMIT_MS},
        "limitations": (
            "Headless production-owner compositor paint; not native input/font or "
            "portable-machine qualification. Synthetic Input.value/Enter injection. "
            "One real app resized between viewports; no fake loader/activation results. "
            "Setup/backup/boot, geometry preparation and activation query preparation "
            "are separate from the 60 manifest-query timing samples."
        ),
    }
    app = None
    database = None
    watcher = None
    active: PaintWindow | None = None
    previous_tick = 0
    failure: BaseException | None = None
    try:
        # All production imports occur after the launcher establishes guards.
        import keyring
        from textual.widgets import Input

        from tldw_chatbook import config
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
        from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
            ConsoleSessionSwitcherModal,
        )

        assert type(keyring.get_keyring()).__module__ == "keyring.backends.null"
        destination = Path(config.get_chachanotes_db_path()).resolve()
        assert destination.is_relative_to(profile_root)
        await asyncio.to_thread(_copy_corpus, corpus, destination)
        evidence["copied_database"] = str(destination)

        class ObservedApp(TldwCli):
            def _display(self, screen: Any, renderable: Any) -> None:
                super()._display(screen, renderable)
                window = active
                if (
                    window is None
                    or renderable is None
                    or self._batch_count
                    or not isinstance(screen, ConsoleSessionSwitcherModal)
                    or self.screen is not screen
                    or screen._mode is not SwitcherMode.CHARACTER_CHATS
                ):
                    return
                observed_ns = time.perf_counter_ns()
                frame = "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                )
                window.observe(frame, observed_ns, screen=screen)
                if (
                    window.operation != "activation"
                    and not screen._query_pending
                    and screen._rendered_query == window.query
                    and window.settled_ns is None
                ):
                    window.settled_ns = observed_ns
                    window.settled_text = frame
                window.observer_ns.append(time.perf_counter_ns() - observed_ns)

        async def sentinel() -> None:
            nonlocal previous_tick
            while True:
                await asyncio.sleep(0.005)
                now = time.perf_counter_ns()
                if active is not None:
                    active.gaps_ns.append(now - previous_tick)
                previous_tick = now

        def begin(operation: str, query: str, screen: Any) -> PaintWindow:
            nonlocal active, previous_tick
            previous_tick = time.perf_counter_ns()
            active = PaintWindow(operation, query, previous_tick, screen=screen)
            return active

        def finish(window: PaintWindow, label: str, *, qualify: bool = True) -> None:
            nonlocal active
            now = time.perf_counter_ns()
            window.gaps_ns.append(now - previous_tick)
            active = None
            row = {**window.summary(), "id": label, "size": list(app.size)}
            if qualify:
                row["failures"] = window.failures()
                evidence["failures"].extend(
                    f"{label}: {value}" for value in row["failures"]
                )
            evidence["samples"].append(row)
            (output_dir / f"{label}-busy.txt").write_text(window.busy_text)
            (output_dir / f"{label}-settled.txt").write_text(window.settled_text)

        async def settled_modal(query: str) -> Any:
            await _wait(
                lambda: (
                    isinstance(app.screen, ConsoleSessionSwitcherModal)
                    and not app.screen._query_pending
                    and app.screen._rendered_query == query
                ),
                "switcher query",
            )
            return app.screen

        app = ObservedApp()
        started = time.perf_counter_ns()
        watcher = asyncio.create_task(sentinel(), name="task5-ui-latency-observer")
        async with asyncio.timeout(240), app.run_test(size=(52, 20)) as pilot:
            await _wait(
                lambda: (
                    getattr(app, "_ui_ready", False)
                    and isinstance(app.screen, ChatScreen)
                ),
                "Console ready",
                READY_TIMEOUT,
            )
            evidence["boot_ms"] = (time.perf_counter_ns() - started) / 1e6
            database = app.chachanotes_db
            evidence["index_after_boot"] = await asyncio.to_thread(
                _keyword_evidence, database
            )
            assert (
                Path(evidence["index_after_boot"]["database_path"]).resolve()
                == destination
            )
            chat = app.screen
            workspace = chat._workspace
            original_page = chat._character_context.keyword_page
            original_activate = workspace.activate_character_conversation
            original_commit_waiter = (
                workspace.wait_until_character_conversation_commit_started
            )

            async def observed_page(**kwargs: Any) -> Any:
                window = active
                call = {"kind": "keyword_page", "entered_ns": time.perf_counter_ns()}
                if window is not None:
                    window.calls.append(call)
                try:
                    return await original_page(**kwargs)
                finally:
                    call["returned_ns"] = time.perf_counter_ns()

            async def observed_activate(
                request: Any, cancellation: Any = None, **kwargs: Any
            ) -> Any:
                window = active
                call = {"kind": "activation", "entered_ns": time.perf_counter_ns()}
                if window is not None:
                    window.calls.append(call)
                try:
                    result = await original_activate(request, cancellation, **kwargs)
                    call.update(
                        result=str(result.kind), target=result.target.conversation_id
                    )
                    return result
                finally:
                    call["returned_ns"] = time.perf_counter_ns()

            # Observation-only wrappers delegate every real input/result unchanged.
            chat._character_context.keyword_page = observed_page
            workspace.activate_character_conversation = observed_activate
            workspace.wait_until_character_conversation_commit_started = (
                _observe_commit_waiter(original_commit_waiter, lambda: active)
            )

            for size in ((52, 20), (120, 50)):
                if tuple(app.size) != size:
                    await pilot.resize_terminal(*size)
                prefix = f"{size[0]}x{size[1]}"
                await chat.action_open_console_session_switcher(
                    initial_mode=SwitcherMode.CHARACTER_CHATS
                )
                modal = await settled_modal("")
                await pilot.pause()
                blank = "\n".join(
                    strip.text for strip in modal._compositor.render_strips()
                )
                assert "Type a Keyword" in blank and not modal._entries
                (output_dir / f"{prefix}-blank.txt").write_text(blank)

                # Normal startup can add unrelated built-in cards, advancing
                # the global revision. This genuine first query lets the real
                # installed loader maintain its existing generation. Preparation
                # is measured separately, never hidden in warm query timings.
                preparation = begin("preparation", "Fixture", modal)
                try:
                    modal.query_one("#console-switcher-query", Input).value = "Fixture"
                    await _wait(
                        lambda preparation=preparation: (
                            preparation.settled_ns is not None
                        ),
                        "first Keyword preparation",
                        READY_TIMEOUT,
                    )
                finally:
                    finish(preparation, f"{prefix}-preparation")
                index = await asyncio.to_thread(_keyword_evidence, database)
                evidence.setdefault("index_after_preparation", []).append(
                    {"size": size, **index}
                )
                assert index["status"] == "ready", index
                assert index["counts"] == [10000, 250004, 10000], index
                assert [row["generation_id"] for row in index["generations"]] == [
                    row["generation_id"]
                    for row in evidence["index_after_boot"]["generations"]
                ], index
                await pilot.pause()
                assert len(modal._entries) == 50
                assert [entry.target.conversation_id for entry in modal._entries] == [
                    f"perf-{number:05d}" for number in range(9999, 9949, -1)
                ]
                results = modal.query_one("#console-switcher-results")
                visible = [
                    widget
                    for widget in modal.query(".console-switcher-result")
                    if widget.region.overlaps(results.content_region)
                ]
                if size == (52, 20):
                    assert len(visible) == 4
                evidence["viewports"].append(
                    {
                        "size": size,
                        "fetched": 50,
                        "visible_rows": len(visible),
                        "blank": "prompt; no browse",
                    }
                )
                (output_dir / f"{prefix}-geometry.svg").write_text(
                    app.export_screenshot()
                )

                for item in queries:
                    window = begin("search", item["query"], modal)
                    try:
                        modal.query_one("#console-switcher-query", Input).value = item[
                            "query"
                        ]
                        await _wait(
                            lambda window=window: window.settled_ns is not None,
                            item["id"],
                        )
                        actual_ids = [
                            entry.target.conversation_id for entry in modal._entries
                        ]
                        assert actual_ids == item["expected"], (item["id"], actual_ids)
                    finally:
                        finish(window, f"{prefix}-{item['id']}")
                await pilot.press("escape")
                await _wait(lambda: app.screen is chat, "close query visit")

                targets = [
                    next(item for item in queries if item["category"] == category)
                    for category in ("title", "body", "unicode-long")
                ]
                for index, item in enumerate([*targets, targets[0]]):
                    await chat.action_open_console_session_switcher(
                        initial_mode=SwitcherMode.CHARACTER_CHATS,
                        initial_character_query=item["query"],
                    )
                    modal = await settled_modal(item["query"])
                    await pilot.pause()
                    target = modal._entries[0].target
                    assert target.conversation_id == item["expected"][0]
                    window = begin("activation", item["query"], modal)
                    try:
                        await pilot.press("enter")
                        await _wait(
                            lambda window=window: any(
                                call.get("returned_ns")
                                for call in window.calls
                                if call["kind"] == "activation"
                            ),
                            "typed activation completion",
                        )
                        result = next(
                            call
                            for call in window.calls
                            if call["kind"] == "activation"
                        )
                        assert result["result"] == "opened", result
                        await _wait(
                            lambda modal=modal: _accepted_modal_closed(
                                app, modal, chat
                            ),
                            "accepted modal close",
                        )
                        assert workspace._character_conversation_target_visible(target)
                        store = chat._console_chat_store
                        session = next(
                            row
                            for row in store.sessions()
                            if row.id == store.active_session_id
                        )
                        assert (
                            session.persisted_conversation_id == target.conversation_id
                        )
                        window.settled_ns = time.perf_counter_ns()
                        window.settled_text = "\n".join(
                            strip.text for strip in chat._compositor.render_strips()
                        )
                    finally:
                        finish(window, f"{prefix}-activation-{index + 1}")
                evidence.setdefault("descriptors_after_viewport", []).append(
                    _descriptors()
                )
            assert (
                len(
                    [row for row in evidence["samples"] if row["operation"] == "search"]
                )
                == 60
            )
            assert (
                len(
                    [
                        row
                        for row in evidence["samples"]
                        if row["operation"] == "activation"
                    ]
                )
                == 8
            )
        if evidence["failures"]:
            raise AssertionError(evidence["failures"])
        evidence["status"] = "passed"
    except BaseException as error:  # noqa: BLE001 - write terminal receipt, then re-raise unchanged
        failure = error
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        active = None
        if watcher is not None:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        failure = _finalize_evidence(evidence, failure, _digest(corpus))
        evidence["descriptors_after_run_test_unmount"] = _descriptors()
        evidence["registered_handles_after_run_test_unmount"] = (
            database.registered_connection_count() if database is not None else None
        )
        evidence["app_exception"] = repr(getattr(app, "_exception", None))
        (output_dir / "ui-latency-evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
    if failure is not None:
        raise failure
    return evidence


def main() -> None:
    """Enter only through an independently established disposable guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest-evidence", "output-dir", "profile-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    options = vars(parser.parse_args())
    result = asyncio.run(run(**options))
    print(json.dumps({"status": result["status"], "samples": len(result["samples"])}))


if __name__ == "__main__":
    main()
