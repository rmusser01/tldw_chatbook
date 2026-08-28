from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from typing import Any

import pytest
from pydantic import BaseModel

from tldw_chatbook.Agents.run_log import DEFAULT_MAX_RECORD_BYTES
from tldw_chatbook.Agents.run_log_format import iter_records
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_raw_cli import (
    LocalCommandResumeRecord,
    LOCAL_COMMAND_RUN_LOG_DIR,
    local_command_resume_marker,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    inject_resume_agent_markers,
)
from tldw_chatbook.Chat.console_chat_models import (
    MAX_RAW_CLI_DISPLAY_FIELD_BYTES,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Tools.raw_cli_executor import (
    MAX_RAW_COMMAND_BYTES,
    MAX_RAW_PREVIEW_BYTES,
    RawCliResult,
)
from tldw_chatbook.UI.Console_Modules import raw_cli as raw_cli_module
from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash


class _Runtime:
    permitted = True
    armed = True

    def __init__(
        self,
        initial_directory: Path,
        *,
        db: AgentRunsDB | None = None,
        expected_conversation_id: str | None = None,
        record_output: str = "[stdout]\nok\n",
    ) -> None:
        self._db = db
        self._initial_directory = initial_directory
        self._expected_conversation_id = expected_conversation_id
        self._record_output = record_output
        self.calls = 0

    def execute(self, request: Any, _on_event: Any, **callbacks: Any) -> RawCliResult:
        self.calls += 1
        if self._db is not None and self._expected_conversation_id is not None:
            runs = self._db.list_runs(
                self._expected_conversation_id,
                agent_kind="local_command",
            )
            assert len(runs) == 1
            assert self._db.list_runs(
                request.console_session_id,
                agent_kind="local_command",
            ) == []
        callbacks["on_registered"]()
        callbacks["on_started"](1.0)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell=request.shell,
            initial_directory=self._initial_directory,
            elapsed_seconds=0.01,
            stdout_preview="ok\n",
            stderr_preview="",
            record_output=self._record_output,
            exit_code=0,
            terminal_state="exited",
            truncated=False,
            cleanup_proven=True,
        )


def _stash() -> ConsoleDraftStash:
    return ConsoleDraftStash(
        segments=[],
        text="! printf secret-command",
        has_paste=False,
        raw_cli_prefix_typed=True,
    )


def _resume_record() -> dict[str, Any]:
    return {
        "id": "local-run-1",
        "agent_kind": "local_command",
        "status": "done",
        "steps": [
            {
                "index": 0,
                "kind": "tool_call",
                "tool_name": "raw_cli",
                "args": {
                    "command": "printf secret-command",
                    "shell": "auto",
                    "cwd": "/private/tmp",
                    "invocation_id": "shared-invocation",
                },
            },
            {
                "index": 1,
                "kind": "tool_result",
                "tool_name": "raw_cli",
                "result": "unused potentially large durable output",
                "args": {
                    "invocation_id": "shared-invocation",
                    "shell": "/bin/zsh",
                    "cwd": "/private/tmp",
                    "stdout_preview": "ok\n",
                    "stderr_preview": "",
                    "elapsed_seconds": 0.25,
                    "exit_code": 0,
                    "terminal_state": "exited",
                    "truncated": False,
                    "cleanup_proven": True,
                },
                "status": "done",
                "tool_outcome": "success",
            },
        ],
    }


def _set_nested(record: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    target: Any = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_local_command_resume_marker_reconstructs_valid_bounded_record() -> None:
    marker = local_command_resume_marker(_resume_record())

    assert marker is not None
    assert marker.id == "raw-cli-run-local-run-1"
    assert marker.raw_cli_presentation is not None
    assert marker.raw_cli_presentation.invocation_id == "shared-invocation"
    assert marker.raw_cli_presentation.command == "printf secret-command"
    assert marker.raw_cli_presentation.lifecycle_state == "exited"
    assert marker.tool_output_full == "stdout:\nok\n\n\nstderr:\n(no output)"


def test_local_command_resume_record_uses_strict_pydantic_boundary() -> None:
    assert issubclass(LocalCommandResumeRecord, BaseModel)
    assert LocalCommandResumeRecord.model_config["strict"] is True
    assert LocalCommandResumeRecord.model_config["frozen"] is True


def test_raw_cli_display_field_limit_is_shared() -> None:
    assert MAX_RAW_CLI_DISPLAY_FIELD_BYTES == 4096


def test_local_command_resume_marker_uses_run_identity_for_unique_message_ids() -> None:
    first = _resume_record()
    second = copy.deepcopy(first)
    second["id"] = "local-run-2"

    first_marker = local_command_resume_marker(first)
    second_marker = local_command_resume_marker(second)

    assert first_marker is not None
    assert second_marker is not None
    assert first_marker.id != second_marker.id


def test_local_command_resume_marker_does_not_read_full_step_result() -> None:
    class _PoisonResult(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "result":
                raise AssertionError("resume must not read the full durable output")
            return super().get(key, default)

    record = _resume_record()
    record["steps"][1] = _PoisonResult(record["steps"][1])

    assert local_command_resume_marker(record) is not None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("id",), ""),
        (("id",), "bad\ud800id"),
        (("steps",), "not-a-step-sequence"),
        (("steps",), [42, {}]),
        (("steps", 0, "args"), "not-a-mapping"),
        (("steps", 1, "args"), ["not-a-mapping"]),
        (("steps", 0, "args", "invocation_id"), "bad\ud800id"),
        (("steps", 0, "args", "invocation_id"), "i" * 129),
        (("steps", 1, "args", "invocation_id"), "different"),
        (("steps", 0, "args", "command"), "bad\ud800command"),
        (("steps", 0, "args", "command"), "x" * (MAX_RAW_COMMAND_BYTES + 1)),
        (
            ("steps", 0, "args", "shell"),
            "s" * (MAX_RAW_CLI_DISPLAY_FIELD_BYTES + 1),
        ),
        (("steps", 1, "args", "cwd"), "bad\ud800cwd"),
        (("steps", 1, "args", "stdout_preview"), "bad\ud800preview"),
        (("steps", 1, "args", "stdout_preview"), "x" * (MAX_RAW_PREVIEW_BYTES + 1)),
        (("steps", 1, "args", "elapsed_seconds"), "0.25"),
        (("steps", 1, "args", "elapsed_seconds"), True),
        (("steps", 1, "args", "exit_code"), "0"),
        (("steps", 1, "args", "exit_code"), False),
        (("steps", 1, "args", "terminal_state"), "unknown"),
        (("steps", 1, "args", "terminal_state"), True),
        (("steps", 1, "args", "truncated"), 0),
        (("steps", 1, "args", "cleanup_proven"), 1),
        (("steps", 1, "status"), True),
        (("status",), "running"),
    ],
    ids=[
        "blank-run-id",
        "surrogate-run-id",
        "scalar-steps",
        "non-mapping-step",
        "call-args-not-mapping",
        "result-args-not-mapping",
        "surrogate-invocation",
        "oversize-invocation",
        "mismatched-invocation",
        "surrogate-command",
        "oversize-command",
        "oversize-shell",
        "surrogate-cwd",
        "surrogate-preview",
        "oversize-preview",
        "string-elapsed",
        "bool-elapsed",
        "string-exit",
        "bool-exit",
        "unknown-terminal-state",
        "bool-terminal-state",
        "numeric-truncated",
        "numeric-cleanup",
        "bool-step-status",
        "nonterminal-run-status",
    ],
)
def test_local_command_resume_marker_drops_malformed_rows(
    path: tuple[Any, ...],
    value: Any,
) -> None:
    record = _resume_record()
    _set_nested(record, path, value)

    assert local_command_resume_marker(record) is None


def test_local_command_resume_marker_drops_scalar_record() -> None:
    assert local_command_resume_marker(42) is None  # type: ignore[arg-type]


def test_local_command_resume_marker_rejects_cumulative_preview_overflow() -> None:
    record = _resume_record()
    result_args = record["steps"][1]["args"]
    result_args["stdout_preview"] = "x" * MAX_RAW_PREVIEW_BYTES
    result_args["stderr_preview"] = "y"

    assert local_command_resume_marker(record) is None


def test_local_command_resume_marker_requires_terminal_cleanup_proof() -> None:
    record = _resume_record()
    del record["steps"][1]["args"]["cleanup_proven"]

    assert local_command_resume_marker(record) is None


def _controller(
    tmp_path: Path,
    runtime: _Runtime,
    *,
    agent_runs_db: Any,
    persist_session_if_needed: Any = lambda _session_id: "durable-conversation-1",
    active_leaf_anchor: Any = lambda _session_id: "native-leaf-1",
    persisted_leaf_anchor: Any = (
        lambda _session_id, _leaf_id: "assistant-leaf-1"
    ),
    run_log_access: Any = lambda: None,
    errors: list[str] | None = None,
    updates: list[str] | None = None,
    restore_stash: Any = lambda _session_id, _stash: True,
) -> tuple[ConsoleRawCliController, list[Any]]:
    workers: list[Any] = []
    errors = errors if errors is not None else []
    updates = updates if updates is not None else []

    def update_marker(_session_id: str, _marker_id: str, **fields: Any) -> None:
        updates.append(fields["raw_cli_presentation"].lifecycle_state)

    controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: runtime,
        active_session_id=lambda: "native-session-1",
        persist_session_if_needed=persist_session_if_needed,
        active_leaf_anchor=active_leaf_anchor,
        persisted_leaf_anchor=persisted_leaf_anchor,
        selected_local_root=lambda _session_id: tmp_path,
        private_scratch_root=lambda _session_id: tmp_path,
        refusal_stash_bank={},
        accepts_raw_cli_refusal_callbacks=lambda: True,
        restore_stash=restore_stash,
        append_local_error=lambda _session_id, message: errors.append(message),
        append_store_marker=lambda _session_id, **_fields: None,
        update_store_marker=update_marker,
        agent_runs_db=agent_runs_db,
        run_log_access=run_log_access,
        start_worker=lambda work, **_kwargs: workers.append(work),
        marshal_to_ui=lambda callback, *args: callback(*args),
    )
    return controller, workers


def test_local_command_first_persists_real_session_under_durable_identity(
    tmp_path: Path,
) -> None:
    chat_db = CharactersRAGDB(tmp_path / "chat.db", client_id="raw-cli-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
    session = store.create_session(session_id="native-session-1", title="Unsaved")
    runs_db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    assert session.persisted_conversation_id is None

    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: runs_db,
        persist_session_if_needed=store.persist_session_if_needed,
        active_leaf_anchor=lambda _session_id: None,
        persisted_leaf_anchor=lambda _session_id, _leaf_id: None,
        run_log_access=lambda: tmp_path / "app-data",
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 1
    assert session.persisted_conversation_id is not None
    (run,) = runs_db.list_runs(
        session.persisted_conversation_id,
        agent_kind="local_command",
    )
    assert run["assistant_message_id"] is None
    assert runs_db.list_runs(session.id, agent_kind="local_command") == []

    blocks = ConsoleAgentBridge(
        agent_runs_db=runs_db,
        store=None,
        provider_gateway=None,
    ).resume_marker_messages(session.persisted_conversation_id)
    later_transcript = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="later question"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="later answer",
            status="complete",
            persisted_message_id="later-assistant",
        ),
    ]

    resumed = inject_resume_agent_markers(later_transcript, blocks)

    assert resumed[0].raw_cli_presentation is not None
    assert [message.content for message in resumed[1:]] == [
        "later question",
        "later answer",
    ]


def test_first_session_persistence_is_serialized_to_one_identity() -> None:
    class RacingPersistence:
        def __init__(self) -> None:
            self.calls = 0
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()
            self._lock = threading.Lock()

        def create_conversation(self, **_kwargs: Any) -> str:
            with self._lock:
                self.calls += 1
                call_number = self.calls
            if call_number == 1:
                self.first_entered.set()
                assert self.release_first.wait(timeout=2)
            else:
                self.second_entered.set()
            return f"conversation-{call_number}"

    persistence = RacingPersistence()
    store = ConsoleChatStore(persistence=persistence)  # type: ignore[arg-type]
    session = store.create_session(session_id="new-session", title="Unsaved")
    results: list[str | None] = []
    errors: list[BaseException] = []

    def persist() -> None:
        try:
            results.append(store.persist_session_if_needed(session.id))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=persist)
    second = threading.Thread(target=persist)
    first.start()
    assert persistence.first_entered.wait(timeout=2)
    second.start()
    persistence.second_entered.wait(timeout=1)
    persistence.release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert persistence.calls == 1
    assert results == ["conversation-1", "conversation-1"]
    assert session.persisted_conversation_id == "conversation-1"


def test_local_command_refuses_temporary_real_session_without_any_local_write(
    tmp_path: Path,
) -> None:
    chat_db = CharactersRAGDB(tmp_path / "chat.db", client_id="raw-cli-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
    session = store.create_session(
        session_id="native-session-1",
        title="Temporary",
        ephemeral=True,
    )
    runs_db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []
    restored: list[tuple[str | None, ConsoleDraftStash]] = []
    log_accesses = 0

    def restore(session_id: str | None, stash: ConsoleDraftStash) -> bool:
        restored.append((session_id, stash))
        return True

    def run_log_access() -> Path:
        nonlocal log_accesses
        log_accesses += 1
        return tmp_path / "app-data"

    stash = _stash()
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: runs_db,
        persist_session_if_needed=store.persist_session_if_needed,
        run_log_access=run_log_access,
        errors=errors,
        restore_stash=restore,
    )

    assert controller.start_user_command(stash)
    workers.pop()()

    assert runtime.calls == 0
    assert runs_db.list_runs(session.id, agent_kind="local_command") == []
    assert log_accesses == 0
    assert not (tmp_path / "app-data").exists()
    assert session.persisted_conversation_id is None
    assert chat_db.get_conversation_by_id(session.id) is None
    assert restored == [(session.id, stash)]
    assert errors == [
        "Raw CLI could not persist this command locally. The exact draft was restored."
    ]


def test_local_command_refuses_when_session_persistence_is_unavailable(
    tmp_path: Path,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(session_id="native-session-1", title="Unsaved")
    runs_db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    restored: list[ConsoleDraftStash] = []
    stash = _stash()
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: runs_db,
        persist_session_if_needed=store.persist_session_if_needed,
        restore_stash=lambda _session_id, item: restored.append(item) or True,
    )

    assert controller.start_user_command(stash)
    workers.pop()()

    assert runtime.calls == 0
    assert runs_db.list_runs(session.id, agent_kind="local_command") == []
    assert restored == [stash]


def test_local_command_creates_run_before_execution_and_persists_result(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent-runs.db"
    db = AgentRunsDB(db_path)
    runtime = _Runtime(
        tmp_path,
        db=db,
        expected_conversation_id="durable-conversation-1",
    )
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path / "app-data",
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    reopened = AgentRunsDB(db_path)
    runs = reopened.list_runs(
        "durable-conversation-1",
        agent_kind="local_command",
    )
    assert len(runs) == 1
    assert reopened.list_runs("native-session-1", agent_kind="local_command") == []
    run = runs[0]
    assert run["task"] == "Local command"
    assert run["assistant_message_id"] == "assistant-leaf-1"
    assert run["status"] == "done"
    assert run["result"] is None

    steps = AgentRunsDB(db_path).get_run(run["id"])["steps"]
    assert [step["kind"] for step in steps] == ["tool_call", "tool_result"]
    assert steps[0]["args"]["command"] == "printf secret-command"
    assert steps[1]["result"] == "[stdout]\nok\n"
    assert steps[1]["args"]["exit_code"] == 0

    metadata = json.dumps(
        {key: value for key, value in run.items() if key != "steps"},
        sort_keys=True,
    )
    assert "secret-command" not in metadata
    assert str(tmp_path) not in metadata
    assert "[stdout]" not in metadata

    run_dir = tmp_path / "app-data" / f".{LOCAL_COMMAND_RUN_LOG_DIR}" / run["id"]
    records = []
    raw_log_bytes = b""
    for segment in sorted(run_dir.glob("logs.*.txt")):
        payload = segment.read_bytes()
        raw_log_bytes += payload
        records.extend(iter_records(payload))
    assert [record.type for record in records] == ["tool_call", "tool_result"]
    assert records[0].content == "printf secret-command"
    assert records[1].content == "[stdout]\nok\n"
    assert b"secret-command" in raw_log_bytes
    assert b"[stdout]\nok\n" in raw_log_bytes

    manifest = (run_dir / "MANIFEST").read_bytes()
    assert b"secret-command" not in manifest
    assert str(tmp_path).encode("utf-8") not in manifest
    assert b"[stdout]" not in manifest


def test_local_command_separates_durable_and_private_output_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_limit = DEFAULT_MAX_RECORD_BYTES + 64 * 1024
    escape_heavy_output = "\\" * (DEFAULT_MAX_RECORD_BYTES + 32 * 1024)
    monkeypatch.setattr(
        raw_cli_module,
        "configured_max_record_bytes",
        lambda: private_limit,
    )
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path, record_output=escape_heavy_output)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path / "app-data",
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    (run,) = db.list_runs(
        "durable-conversation-1",
        agent_kind="local_command",
    )
    durable_result = db.get_run(run["id"])["steps"][1]["result"]
    assert len(durable_result.encode("utf-8")) == DEFAULT_MAX_RECORD_BYTES
    assert durable_result == escape_heavy_output[:DEFAULT_MAX_RECORD_BYTES]

    run_dir = tmp_path / "app-data" / f".{LOCAL_COMMAND_RUN_LOG_DIR}" / run["id"]
    records = [
        record
        for segment in sorted(run_dir.glob("logs.*.txt"))
        for record in iter_records(segment.read_bytes())
    ]
    assert records[-1].content == escape_heavy_output
    assert records[-1].truncated_from == 0

    (resume_record,) = db.local_command_resume_records("durable-conversation-1")
    assert local_command_resume_marker(resume_record) is not None


@pytest.mark.parametrize("db_failure", ["missing", "raises"])
def test_local_command_fails_closed_when_run_db_is_unavailable(
    tmp_path: Path,
    db_failure: str,
) -> None:
    runtime = _Runtime(tmp_path)
    errors: list[str] = []

    def db_access() -> None:
        if db_failure == "raises":
            raise RuntimeError("db detail must stay private")
        return None

    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=db_access,
        errors=errors,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 0
    assert errors == [
        "Raw CLI could not persist this command locally. The exact draft was restored."
    ]
    assert len(errors[0].encode("utf-8")) <= 128
    assert "secret-command" not in errors[0]
    assert str(tmp_path) not in errors[0]


def test_local_command_terminalizes_created_row_when_initial_step_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []

    def fail_initial_step(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("step detail must stay private")

    monkeypatch.setattr(db, "append_steps", fail_initial_step)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        errors=errors,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    (run,) = db.list_runs("durable-conversation-1", agent_kind="local_command")
    assert run["status"] == "error"
    assert runtime.calls == 0
    assert errors == [
        "Raw CLI could not persist this command locally. The exact draft was restored."
    ]


def test_local_command_closes_writer_when_initial_log_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []
    updates: list[str] = []

    class FailingWriter:
        instance: "FailingWriter | None" = None

        def __init__(self, **_kwargs: Any) -> None:
            self.close_calls = 0
            self.is_active = True
            type(self).instance = self

        def bind(self, _run_id: str) -> None:
            return None

        def append(self, **_kwargs: Any) -> None:
            raise RuntimeError("initial writer detail must stay private")

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(raw_cli_module, "RunLogWriter", FailingWriter)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path,
        errors=errors,
        updates=updates,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 1
    assert FailingWriter.instance is not None
    assert FailingWriter.instance.close_calls == 1
    assert updates[-1] == "exited"
    assert errors == ["Raw CLI persistence was incomplete."]


@pytest.mark.parametrize("failure_stage", ["bind", "append"])
def test_local_command_rejects_inactive_initial_writer_without_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []
    updates: list[str] = []

    class InactiveWriter:
        instance: "InactiveWriter | None" = None

        def __init__(self, **_kwargs: Any) -> None:
            self.append_calls = 0
            self.close_calls = 0
            self.is_active = True
            type(self).instance = self

        def bind(self, _run_id: str) -> None:
            if failure_stage == "bind":
                self.is_active = False

        def append(self, **_kwargs: Any) -> int | None:
            self.append_calls += 1
            if failure_stage == "append":
                self.is_active = False
                return None
            return 1

        def write_manifest(self, _manifest: dict[str, Any]) -> None:
            raise AssertionError("inactive initial writer must not be retained")

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(raw_cli_module, "RunLogWriter", InactiveWriter)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path,
        errors=errors,
        updates=updates,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 1
    assert InactiveWriter.instance is not None
    assert InactiveWriter.instance.append_calls == (failure_stage == "append")
    assert InactiveWriter.instance.close_calls == 1
    assert updates[-1] == "exited"
    assert errors == ["Raw CLI persistence was incomplete."]


def test_local_command_reports_terminal_append_deactivation_without_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []
    updates: list[str] = []

    class DeactivatingWriter:
        instance: "DeactivatingWriter | None" = None

        def __init__(self, **_kwargs: Any) -> None:
            self.append_calls = 0
            self.manifest_calls = 0
            self.close_calls = 0
            self.is_active = True
            type(self).instance = self

        def bind(self, _run_id: str) -> None:
            return None

        def append(self, **_kwargs: Any) -> int | None:
            self.append_calls += 1
            if self.append_calls == 2:
                self.is_active = False
                return None
            return self.append_calls

        def write_manifest(self, _manifest: dict[str, Any]) -> None:
            self.manifest_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(raw_cli_module, "RunLogWriter", DeactivatingWriter)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path,
        errors=errors,
        updates=updates,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 1
    assert DeactivatingWriter.instance is not None
    assert DeactivatingWriter.instance.append_calls == 2
    assert DeactivatingWriter.instance.manifest_calls == 1
    assert DeactivatingWriter.instance.close_calls == 1
    assert updates[-1] == "exited"
    assert errors == ["Raw CLI persistence was incomplete."]


def test_local_command_terminal_persistence_failures_never_block_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    runtime = _Runtime(tmp_path)
    errors: list[str] = []
    updates: list[str] = []
    terminal_attempts = 0

    def fail_terminal_write(*_args: Any, **_kwargs: Any) -> None:
        nonlocal terminal_attempts
        terminal_attempts += 1
        raise RuntimeError("terminal db detail must stay private")

    monkeypatch.setattr(db, "set_terminal_with_step", fail_terminal_write)

    class FailingWriter:
        instance: "FailingWriter | None" = None

        def __init__(self, **_kwargs: Any) -> None:
            self.append_calls = 0
            self.manifest_calls = 0
            self.close_calls = 0
            self.is_active = True
            type(self).instance = self

        def bind(self, _run_id: str) -> None:
            return None

        def append(self, **_kwargs: Any) -> int:
            self.append_calls += 1
            if self.append_calls > 1:
                raise RuntimeError("terminal append detail must stay private")
            return self.append_calls

        def write_manifest(self, _manifest: dict[str, Any]) -> None:
            self.manifest_calls += 1
            raise RuntimeError("manifest detail must stay private")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close detail must stay private")

    monkeypatch.setattr(raw_cli_module, "RunLogWriter", FailingWriter)
    controller, workers = _controller(
        tmp_path,
        runtime,
        agent_runs_db=lambda: db,
        run_log_access=lambda: tmp_path,
        errors=errors,
        updates=updates,
    )

    assert controller.start_user_command(_stash())
    workers.pop()()

    assert runtime.calls == 1
    assert terminal_attempts == 1
    assert FailingWriter.instance is not None
    assert FailingWriter.instance.append_calls == 2
    assert FailingWriter.instance.manifest_calls == 1
    assert FailingWriter.instance.close_calls == 1
    assert updates[-1] == "exited"
    assert errors == ["Raw CLI persistence was incomplete."]
    assert "secret-command" not in errors[0]
    assert str(tmp_path) not in errors[0]
