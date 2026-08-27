"""Direct-user raw CLI routing before every model-send seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_command_grammar import CommandParse, KIND_NOT_COMMAND
from tldw_chatbook.Tools.raw_cli_executor import RawCliResult
from tldw_chatbook.UI.Console_Modules.raw_cli import (
    ConsoleRawCliController,
    restore_refused_raw_cli_stash,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ConsoleComposerBar,
    ConsoleDraftStash,
    classify_console_raw_draft,
)
from textual.events import Key


def _stash(
    text: str,
    *,
    trusted: bool = True,
    has_paste: bool = False,
) -> ConsoleDraftStash:
    return ConsoleDraftStash(
        segments=[],
        text=text,
        has_paste=has_paste,
        raw_cli_prefix_typed=trusted,
    )


def _physical_raw_stash(command: str) -> ConsoleDraftStash:
    composer = ConsoleComposerBar()
    assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
    assert composer.handle_console_key(Key("space", " ")) is True
    composer.insert_pasted_text(command)
    stash = composer.stash_draft_for_send()
    assert stash is not None
    return stash


class _Runtime:
    def __init__(self, *, permitted: bool = True, armed: bool = True) -> None:
        self.permitted = permitted
        self.armed = armed
        self.requests: list[Any] = []

    def execute(self, request: Any, _on_event: Any) -> RawCliResult:
        self.requests.append(request)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.01,
            stdout_preview="ok\n",
            stderr_preview="",
            record_output="ok\n",
            exit_code=0,
            terminal_state="exited",
            truncated=False,
            cleanup_proven=True,
        )


def _controller(
    tmp_path: Path,
    runtime: _Runtime,
    *,
    selected_root: Path | None = None,
    anchor: str | None = "persisted-leaf",
    active_session_id: Any = lambda: "session-1",
    restore_stash: Any | None = None,
    append_local_error: Any | None = None,
    marshal_to_ui: Any | None = None,
):
    restored: list[tuple[str | None, ConsoleDraftStash]] = []
    errors: list[tuple[str | None, str]] = []
    workers: list[tuple[Any, dict[str, Any]]] = []
    selected_reads: list[str] = []
    scratch_reads: list[str] = []

    def read_selected(session_id: str) -> Path | None:
        selected_reads.append(session_id)
        return selected_root

    def read_scratch(session_id: str) -> Path:
        scratch_reads.append(session_id)
        return tmp_path

    def start_worker(work: Any, **kwargs: Any) -> object:
        workers.append((work, kwargs))
        return object()

    if restore_stash is None:
        def default_restore_stash(
            session_id: str | None, stash: ConsoleDraftStash
        ) -> bool:
            restored.append((session_id, stash))
            return True

        restore_stash = default_restore_stash
    if append_local_error is None:
        def default_append_local_error(session_id: str | None, text: str) -> None:
            errors.append((session_id, text))

        append_local_error = default_append_local_error
    if marshal_to_ui is None:
        def default_marshal_to_ui(callback: Any, *args: Any) -> None:
            callback(*args)

        marshal_to_ui = default_marshal_to_ui

    controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: runtime,
        active_session_id=active_session_id,
        persisted_leaf_anchor=lambda session_id: anchor,
        selected_local_root=read_selected,
        private_scratch_root=read_scratch,
        restore_stash=restore_stash,
        append_local_error=append_local_error,
        append_store_marker=lambda *args, **kwargs: None,
        update_store_marker=lambda *args, **kwargs: None,
        agent_runs_db=lambda: None,
        run_log_access=lambda: None,
        start_worker=start_worker,
        marshal_to_ui=marshal_to_ui,
    )
    return controller, restored, errors, workers, selected_reads, scratch_reads


def test_raw_cli_controller_captures_one_submission_snapshot_and_named_worker(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    runtime = _Runtime()
    (
        controller,
        restored,
        errors,
        workers,
        selected_reads,
        scratch_reads,
    ) = _controller(tmp_path, runtime, selected_root=selected)
    stash = _stash("! printf ok")

    assert controller.start_user_command(stash) is True
    assert restored == []
    assert errors == []
    assert selected_reads == ["session-1"]
    assert scratch_reads == []
    assert len(workers) == 1
    work, options = workers[0]
    assert options["thread"] is True
    assert options["exclusive"] is False
    assert options["name"].startswith("console-raw-cli-")

    work()

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.caller == "user"
    assert request.command == "printf ok"
    assert request.shell == "auto"
    assert request.initial_directory == selected.resolve()
    assert request.timeout_seconds == 300.0
    assert request.console_session_id == "session-1"
    assert request.transcript_anchor_id == "persisted-leaf"


def test_raw_cli_controller_falls_back_to_private_scratch_once(tmp_path: Path) -> None:
    runtime = _Runtime()
    controller, _restored, _errors, workers, selected_reads, scratch_reads = (
        _controller(tmp_path, runtime)
    )

    assert controller.start_user_command(_stash("! pwd")) is True
    workers[0][0]()

    assert selected_reads == ["session-1"]
    assert scratch_reads == ["session-1"]
    assert runtime.requests[0].initial_directory == tmp_path.resolve()


def test_runtime_admission_refusal_marshals_restore_to_ui(tmp_path: Path) -> None:
    class RefusingRuntime(_Runtime):
        def execute(self, request: Any, _on_event: Any) -> RawCliResult:
            self.requests.append(request)
            return RawCliResult(
                invocation_id=request.invocation_id,
                caller=request.caller,
                resolved_shell=request.shell,
                initial_directory=request.initial_directory,
                elapsed_seconds=0.0,
                stdout_preview="",
                stderr_preview="",
                record_output="",
                exit_code=None,
                terminal_state="refused",
                truncated=False,
                cleanup_proven=True,
            )

    runtime = RefusingRuntime()
    marshalled: list[tuple[Any, tuple[Any, ...]]] = []
    controller, restored, errors, workers, _selected, _scratch = _controller(
        tmp_path,
        runtime,
        marshal_to_ui=lambda callback, *args: marshalled.append((callback, args)),
    )
    stash = _stash("! pwd")

    assert controller.start_user_command(stash) is True
    assert restored == []
    workers[0][0]()

    assert restored == []
    assert errors == []
    assert len(marshalled) == 1

    callback, args = marshalled.pop()
    callback(*args)

    assert restored == [("session-1", stash)]
    assert "authority changed" in errors[0][1].lower()


def test_prelaunch_containment_unavailable_restores_draft_on_ui_thread(
    tmp_path: Path,
) -> None:
    class ContainmentRefusingRuntime(_Runtime):
        def execute(self, request: Any, _on_event: Any) -> RawCliResult:
            self.requests.append(request)
            return RawCliResult(
                invocation_id=request.invocation_id,
                caller=request.caller,
                resolved_shell=request.shell,
                initial_directory=request.initial_directory,
                elapsed_seconds=0.0,
                stdout_preview="",
                stderr_preview="",
                record_output="",
                exit_code=None,
                terminal_state="containment_unavailable",
                truncated=False,
                cleanup_proven=True,
            )

    marshalled: list[tuple[Any, tuple[Any, ...]]] = []
    runtime = ContainmentRefusingRuntime()
    controller, restored, errors, workers, _selected, _scratch = _controller(
        tmp_path,
        runtime,
        marshal_to_ui=lambda callback, *args: marshalled.append((callback, args)),
    )
    stash = _stash("! pwd")

    assert controller.start_user_command(stash) is True
    workers[0][0]()

    assert restored == []
    assert errors == []
    callback, args = marshalled.pop()
    callback(*args)
    assert restored == [("session-1", stash)]
    assert "did not launch" in errors[0][1].lower()
    assert "restored" in errors[0][1].lower()


def test_postlaunch_terminal_failure_does_not_restore_the_sent_draft(
    tmp_path: Path,
) -> None:
    class TimedOutRuntime(_Runtime):
        def execute(self, request: Any, _on_event: Any) -> RawCliResult:
            self.requests.append(request)
            return RawCliResult(
                invocation_id=request.invocation_id,
                caller=request.caller,
                resolved_shell=request.shell,
                initial_directory=request.initial_directory,
                elapsed_seconds=3.0,
                stdout_preview="partial",
                stderr_preview="",
                record_output="partial",
                exit_code=None,
                terminal_state="timed_out",
                truncated=False,
                cleanup_proven=True,
            )

    runtime = TimedOutRuntime()
    controller, restored, errors, workers, _selected, _scratch = _controller(
        tmp_path, runtime
    )

    assert controller.start_user_command(_stash("! sleep 10")) is True
    workers[0][0]()

    assert restored == []
    assert errors == []


@pytest.mark.parametrize(
    ("text", "permitted", "armed", "copy"),
    [
        ("! ", True, True, "empty"),
        ("! echo no", False, False, "Locked"),
        ("! echo no", True, False, "not armed"),
    ],
)
def test_raw_cli_local_refusals_restore_the_exact_trusted_stash(
    tmp_path: Path,
    text: str,
    permitted: bool,
    armed: bool,
    copy: str,
) -> None:
    runtime = _Runtime(permitted=permitted, armed=armed)
    controller, restored, errors, workers, selected_reads, scratch_reads = _controller(
        tmp_path, runtime
    )
    stash = _stash(text)

    assert controller.start_user_command(stash) is False

    assert restored == [("session-1", stash)]
    assert restored[0][1] is stash
    assert restored[0][1].raw_cli_prefix_typed is True
    assert copy.lower() in errors[0][1].lower()
    assert workers == []
    assert runtime.requests == []
    assert selected_reads == []
    assert scratch_reads == []


class _Composer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.restored: list[ConsoleDraftStash | None] = []

    def draft_text(self) -> str:
        return self.text

    def has_paste_segments(self) -> bool:
        return False

    def restore_stashed_draft(self, stash: ConsoleDraftStash | None) -> None:
        self.restored.append(stash)

    def stash_raw_cli_draft_for_send(self) -> ConsoleDraftStash | None:
        return None


def test_refusal_restore_requires_active_and_visible_origin_ownership() -> None:
    stash = _stash("! pwd")
    composer = _Composer("session b")

    assert restore_refused_raw_cli_stash(
        None,
        stash,
        composer=composer,
        active_session_id="session-b",
        visible_session_id="session-b",
    ) is False
    assert composer.restored == []


def _route_screen(stash: ConsoleDraftStash):
    composer = _Composer(stash.text)
    raw_calls: list[ConsoleDraftStash] = []
    parsed: list[str] = []
    dispatched: list[tuple[str, ConsoleDraftStash | None]] = []

    async def dispatch(draft: str, *, stash: ConsoleDraftStash | None = None) -> bool:
        dispatched.append((draft, stash))
        return True

    screen = SimpleNamespace(
        _console_pending_send_stash=stash,
        _raw_cli=SimpleNamespace(start_user_command=raw_calls.append),
        _console_composer_or_none=lambda: composer,
        query_one=lambda *_args, **_kwargs: composer,
        _console_pending_image_attachment=lambda: None,
        _focus_console_composer_if_needed=lambda **_kwargs: None,
        _dismiss_console_guidance=lambda: None,
        _console_command_registry=SimpleNamespace(
            parse=lambda text: (
                parsed.append(text) or CommandParse(kind=KIND_NOT_COMMAND)
            )
        ),
        _dispatch_console_draft_send=dispatch,
    )
    return screen, composer, raw_calls, parsed, dispatched


@pytest.mark.asyncio
async def test_trusted_raw_stash_is_intercepted_before_slash_attachment_and_queue() -> (
    None
):
    stash = _stash("! /slash-looking-command")
    screen, _composer, raw_calls, _parsed, _dispatched = _route_screen(stash)
    staged_attachment = object()
    screen._staged_attachments = [staged_attachment]

    def forbidden() -> None:
        raise AssertionError("raw command reached a model-send seam")

    screen.query_one = lambda *_args, **_kwargs: forbidden()
    screen._console_composer_or_none = forbidden
    screen._console_pending_image_attachment = forbidden
    screen._console_command_registry.parse = lambda _text: forbidden()
    screen._dispatch_console_draft_send = lambda *_args, **_kwargs: forbidden()

    assert await ChatScreen._send_console_message_from_visible_action(screen) is False
    assert raw_calls == [stash]
    assert screen._console_pending_send_stash is None
    assert screen._staged_attachments == [staged_attachment]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stash", "expected", "expected_parse"),
    [
        (_stash(r"\! echo escaped", trusted=False), "! echo escaped", True),
        (
            _stash("! echo pasted", trusted=False, has_paste=True),
            "! echo pasted",
            False,
        ),
    ],
)
async def test_escaped_and_untrusted_bang_prefixes_follow_ordinary_chat(
    stash: ConsoleDraftStash,
    expected: str,
    expected_parse: bool,
) -> None:
    screen, _composer, raw_calls, parsed, dispatched = _route_screen(stash)

    assert await ChatScreen._send_console_message_from_visible_action(screen) is True

    assert raw_calls == []
    assert parsed == ([expected] if expected_parse else [])
    assert len(dispatched) == 1
    dispatched_text, dispatched_stash = dispatched[0]
    assert dispatched_text == expected
    assert dispatched_stash is not None
    assert dispatched_stash.text == expected
    assert dispatched_stash.raw_cli_prefix_typed is False
    assert (dispatched_stash is not stash) is expected_parse
    if expected_parse:
        assert "".join(segment.text for segment in dispatched_stash.segments) == expected


@pytest.mark.asyncio
async def test_workbench_send_consumes_existing_physical_raw_provenance() -> None:
    composer = ConsoleComposerBar()
    assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
    assert composer.handle_console_key(Key("space", " ")) is True
    composer.insert_pasted_text("pwd")
    raw_calls: list[ConsoleDraftStash] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("alternate Send reached a model-send seam")

    screen = SimpleNamespace(
        _console_pending_send_stash=None,
        _raw_cli=SimpleNamespace(start_user_command=raw_calls.append),
        _console_composer_or_none=lambda: composer,
        query_one=forbidden,
        _console_pending_image_attachment=forbidden,
        _console_command_registry=SimpleNamespace(parse=forbidden),
        _dispatch_console_draft_send=forbidden,
    )

    assert await ChatScreen._send_console_message_from_visible_action(screen) is False
    assert len(raw_calls) == 1
    assert raw_calls[0].text == "! pwd"
    assert raw_calls[0].raw_cli_prefix_typed is True
    assert composer.draft_text() == ""


def test_hidden_session_refusal_banks_exact_stash_until_reconciled(
    tmp_path: Path,
) -> None:
    class RefusingRuntime(_Runtime):
        def execute(self, request: Any, _on_event: Any) -> RawCliResult:
            self.requests.append(request)
            return RawCliResult(
                invocation_id=request.invocation_id,
                caller=request.caller,
                resolved_shell=request.shell,
                initial_directory=request.initial_directory,
                elapsed_seconds=0.0,
                stdout_preview="",
                stderr_preview="",
                record_output="",
                exit_code=None,
                terminal_state="refused",
                truncated=False,
                cleanup_proven=True,
            )

    marshalled: list[tuple[Any, tuple[Any, ...]]] = []
    runtime = RefusingRuntime()
    controller, _restored, errors, workers, _selected, _scratch = _controller(
        tmp_path,
        runtime,
        active_session_id=lambda: "session-a",
        restore_stash=lambda _session_id, _stash: False,
        marshal_to_ui=lambda callback, *args: marshalled.append((callback, args)),
    )
    stash = _physical_raw_stash("pwd")

    assert controller.start_user_command(stash) is True
    workers[0][0]()
    callback, args = marshalled.pop()
    callback(*args)

    reconciled = ConsoleComposerBar()
    reconciled.load_draft("newer a")
    assert controller.restore_banked_stashes("session-a", reconciled) == 1
    restored = reconciled.stash_draft_for_send()

    assert restored is not None
    assert restored.text == "! pwdnewer a"
    assert restored.raw_cli_prefix_typed is True
    assert restored.has_paste is True
    assert restored.segments[: len(stash.segments)] == stash.segments
    classified = classify_console_raw_draft(restored)
    assert classified.kind == "raw"
    assert classified.text == "pwdnewer a"
    assert errors[0][0] == "session-a"


def test_hidden_session_refusal_bank_restores_multiple_stashes_in_order(
    tmp_path: Path,
) -> None:
    runtime = _Runtime(permitted=False, armed=False)
    controller, _restored, _errors, _workers, _selected, _scratch = _controller(
        tmp_path,
        runtime,
        active_session_id=lambda: "session-a",
        restore_stash=lambda _session_id, _stash: False,
    )
    first = _physical_raw_stash("first")
    second = _physical_raw_stash("second")

    assert controller.start_user_command(first) is False
    assert controller.start_user_command(second) is False

    reconciled = ConsoleComposerBar()
    reconciled.load_draft("tail")
    assert controller.restore_banked_stashes("session-a", reconciled) == 2
    assert reconciled.draft_text() == "! first! secondtail"
    assert controller.restore_banked_stashes("session-a", reconciled) == 0
