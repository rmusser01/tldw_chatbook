"""Direct-user raw CLI routing before every model-send seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_command_grammar import CommandParse, KIND_NOT_COMMAND
from tldw_chatbook.Tools.raw_cli_executor import RawCliResult
from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash


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
):
    restored: list[ConsoleDraftStash] = []
    errors: list[str] = []
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

    controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: runtime,
        active_session_id=lambda: "session-1",
        persisted_leaf_anchor=lambda session_id: anchor,
        selected_local_root=read_selected,
        private_scratch_root=read_scratch,
        restore_stash=restored.append,
        append_local_error=errors.append,
        append_store_marker=lambda *args, **kwargs: None,
        update_store_marker=lambda *args, **kwargs: None,
        agent_runs_db=lambda: None,
        run_log_access=lambda: None,
        start_worker=start_worker,
        marshal_to_ui=lambda callback, *args: callback(*args),
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
    controller, restored, errors, workers, _selected, _scratch = _controller(
        tmp_path, runtime
    )
    stash = _stash("! pwd")

    assert controller.start_user_command(stash) is True
    assert restored == []
    workers[0][0]()

    assert restored == [stash]
    assert "authority changed" in errors[0].lower()


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

    assert restored == [stash]
    assert restored[0] is stash
    assert restored[0].raw_cli_prefix_typed is True
    assert copy.lower() in errors[0].lower()
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
    assert dispatched == [(expected, stash)]
    assert stash.text == expected
