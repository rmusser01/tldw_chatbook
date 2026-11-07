"""Native first-send publication must not admit stale direct RawCLI custody."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.test_console_first_send_survivor_native import _SCRIPT

pytestmark = pytest.mark.bootstrap_profile


def _raw_publication_script():
    script = _SCRIPT.replace(
        "sys.argv[2] == 'admission'", "sys.argv[2] == 'raw_publication'"
    )
    start = script.index(
        "    if sys.argv[2] == 'raw_publication':",
        script.index("    assert member.host_returned"),
    )
    end = script.index("    await runtime.dispose()", start)
    script = (
        script[:start]
        + r"""
    async with asyncio.timeout(10):
        while not publication_entered.is_set():
            await asyncio.sleep(.01)
    assert runtime._first_send_publishing
    raw = app.raw_cli_runtime
    assert raw.permitted is True and raw.arm().armed
    raw_source = runtime.raw_cli_source
    accepted_during_publication = False
    call = None
    try:
        try:
            call = raw_source.capture(raw, session.id, 'fix2-during-publication')
        except RuntimeError as error:
            assert 'being saved' in str(error)
        else:
            accepted_during_publication = True
            assert call in raw_source.pending and call.owner.is_admitted
            assert call.config_generation == before[0]
            call.cancel_unstarted()
            assert call.retired.is_set() and call not in raw_source.pending
        print('FIX2_RAW_ADMITTED_DURING_SAVE', accepted_during_publication, flush=True)
        from types import SimpleNamespace
        from Tests.UI.test_console_raw_cli_send import _bind_visible_send_contract, _send_visible_action
        from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
        from textual.events import Key
        composer = ConsoleComposerBar()
        assert composer.handle_console_key(Key('exclamation_mark', '!'))
        assert composer.handle_console_key(Key('space', ' '))
        composer.insert_pasted_text('printf preserved')
        draft = composer.capture_draft_for_send()
        raw_starts, notices = [], []
        screen = SimpleNamespace(
            _console_composer_or_none=lambda: composer,
            _raw_cli=SimpleNamespace(start_user_command=raw_starts.append),
            app_instance=SimpleNamespace(notify=lambda *args, **kwargs: notices.append(args)),
        )
        _bind_visible_send_contract(screen, draft)
        screen._console_runtime = lambda: runtime
        assert await _send_visible_action(screen) is False
        assert composer.draft_text() == draft.text
        retained = composer.capture_draft_for_send()
        assert retained.raw_cli_prefix_typed and retained.segments == draft.segments
        assert not raw_starts and not raw_source.pending
        assert not runtime.execution_capacity.snapshot().executions
        assert len(notices) == 1 and 'being saved' in notices[0][0]
        print('FIX2_ACTUAL_PUBLICATION_VISIBLE_DRAFT_PRESERVED', flush=True)
    finally:
        publication_release.set()
    await asyncio.shield(runtime._first_send_task)
    threading.setprofile_all_threads(None)
"""
        + script[end:]
    )
    script = script.replace(
        "    print('retired and reopened')",
        "    print('FIX2_ORIGINAL_CREATORS_RETIRED', flush=True)\n"
        "    assert not accepted_during_publication, 'RawCLI captured during native first-send publication'\n"
        "    print('retired and reopened')",
    )
    script = script.replace(
        "[splash_screen]\\nenabled=false",
        "[console]\\nraw_cli_permitted=true\\n[splash_screen]\\nenabled=false",
    )
    return script


def test_native_raw_cli_refuses_capture_during_first_send_publication(tmp_path):
    _run(
        tmp_path,
        "fix2_raw",
        "raw_publication",
        script=_raw_publication_script(),
        timeout=55,
    )


def _raw_lifetime_script():
    setup = r"""
    import inspect
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation
    from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

    raw = app.raw_cli_runtime
    assert raw.permitted and raw.arm().armed
    raw_calls, raw_workers, raw_errors = [], [], []
    finish_entered, finish_release = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def marshal(callback, *args):
        async def deliver():
            return callback(*args)
        return asyncio.run_coroutine_threadsafe(deliver(), loop).result(15)

    def start_worker(work, **kwargs):
        call = next(iter(runtime.raw_cli_source.pending))
        raw_calls.append(call)
        worker = threading.Thread(target=work)
        raw_workers.append(worker)
        worker.start()

    raw_controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: raw,
        native_source=runtime.raw_cli_source,
        active_session_id=lambda: session.id,
        persist_session_if_needed=store.persist_session_if_needed,
        active_leaf_anchor=store.active_leaf,
        persisted_leaf_anchor=store.persisted_message_id_for_session_node,
        selected_local_root=lambda _sid: None,
        private_scratch_root=lambda sid: runtime.scratch_spaces.snapshot(sid).root,
        refusal_stash_bank=runtime.raw_cli_refusal_stash_bank,
        accepts_raw_cli_refusal_callbacks=lambda: runtime.accepts_raw_cli_refusal_callbacks,
        restore_stash=lambda sid, stash: raw_errors.append(('restore', stash)) or True,
        append_local_error=lambda sid, text: raw_errors.append(('error', text)),
        append_store_marker=store.append_message,
        update_store_marker=store.update_tool_marker,
        agent_runs_db=lambda: runtime._agent_runs_db,
        run_log_access=lambda: Path(os.environ['XDG_DATA_HOME']) / 'raw-logs',
        start_worker=start_worker,
        marshal_to_ui=marshal,
    )
    publication_code = runtime._persist_first_send_when_quiescent.__func__.__code__
    lines, first = inspect.getsourcelines(runtime._persist_first_send_when_quiescent)
    election_line = first + next(i for i, line in enumerate(lines) if 'not self._first_send_sources' in line)
    injected = []

    def observe_election(code, line):
        if (code is publication_code and line == election_line and not injected
                and not runtime._ordinary_turns and not runtime._turn_custody):
            injected.append(True)
            assert not runtime._first_send_publishing
            assert raw_controller.start_user_command(ConsoleDraftStash(
                segments=[], text='! printf native-raw-output', has_paste=False,
                raw_cli_prefix_typed=True,
            ))

    monitoring = sys.monitoring
    monitor_id = next(i for i in range(6) if monitoring.get_tool(i) is None)
    monitoring.use_tool_id(monitor_id, 'first-send-raw-election')
    monitoring.register_callback(monitor_id, monitoring.events.LINE, observe_election)
    monitoring.set_local_events(monitor_id, publication_code, monitoring.events.LINE)
    def cleanup_election():
        monitoring.set_local_events(monitor_id, publication_code, 0)
        monitoring.register_callback(monitor_id, monitoring.events.LINE, None)
        monitoring.free_tool_id(monitor_id)
    globals()['_task11_cleanup_election'] = cleanup_election

    command_code = ConsoleRawCliController._execute_command.__code__
    command_lines, command_first = inspect.getsourcelines(ConsoleRawCliController._execute_command)
    run_log_failure_line = command_first + next(
        i for i, line in enumerate(command_lines)
        if 'report_persistence_error("Raw CLI persistence was incomplete.")' in line
    )
    persistence_reports = []
    finish_code = OwnedOperation.finish.__code__
    def observe_finish(frame, event, _arg):
        if (event == 'call' and frame.f_code.co_name == 'report_persistence_error'
                and frame.f_back.f_code is command_code):
            caller = frame.f_back
            persistence_reports.append((caller.f_lineno, caller.f_locals['writer'],
                caller.f_locals['candidate_writer'].is_active))
        if (event == 'return' and frame.f_code is finish_code and raw_calls
                and frame.f_locals.get('self') is raw_calls[0].operation):
            finish_entered.set()
            assert finish_release.wait(20)
    threading.setprofile_all_threads(observe_finish)
"""
    check = r"""
    async with asyncio.timeout(15):
        while not finish_entered.is_set():
            await asyncio.sleep(.01)
    cleanup_election()
    globals().pop('_task11_cleanup_election', None)
    call = raw_calls[0]
    try:
        assert injected
        assert raw_errors in ([], [('error', 'Raw CLI persistence was incomplete.')]), raw_errors
        assert persistence_reports in ([], [(run_log_failure_line, None, False)])
        print('FIX2_RUN_LOG_BIND_DIAGNOSTIC', raw_errors, persistence_reports, flush=True)
        assert call in runtime.raw_cli_source.pending
        assert runtime._first_send_sources == 1
        assert not call._operation_finished.is_set() and not call.retired.is_set()
        assert not runtime._first_send_task.done() and not runtime._first_send_publishing
        assert config.current_config_identity() == before
        assert call.required_proven and call.physical_proven and call.host_proven
        markers = [m for m in store.messages_for_session(session.id) if m.raw_cli_presentation]
        assert len(markers) == 1
        marker = markers[0]
        assert 'native-raw-output' in marker.tool_output_full
        assert marker.raw_cli_presentation.exit_code == 0
        assert marker.raw_cli_presentation.cleanup_proven is True
        assert marker.raw_cli_presentation.lifecycle_state == 'exited'
        row = runtime._agent_runs_db.get_run(call.run_id)
        assert row['agent_kind'] == 'local_command' and row['status'] == 'done'
        assert row['steps'][1]['args']['cleanup_proven'] is True
        print('FIX2_ACTUAL_RAW_OUTPUT_TERMINAL_HELD_FINISH_RETURN', flush=True)
    finally:
        finish_release.set()
    await asyncio.to_thread(raw_workers[0].join, 10)
    assert not raw_workers[0].is_alive()
    assert call.retired.is_set() and call not in runtime.raw_cli_source.pending
    assert runtime._first_send_sources == 0
    await asyncio.shield(runtime._first_send_task)
    threading.setprofile_all_threads(None)
"""
    script = _SCRIPT.replace(
        "    turn_id = runtime.accept_received_intent",
        setup + "\n    turn_id = runtime.accept_received_intent",
    ).replace("    await runtime.dispose()", check + "\n    await runtime.dispose()")
    script = script.replace(
        "    assert not runtime.execution_capacity.snapshot().executions",
        "    assert not runtime._ordinary_turns",
    )
    script = script.replace(
        "except BaseException:\n    (Path(os.environ['XDG_DATA_HOME'])",
        "except BaseException:\n    globals().get('_task11_cleanup_election', lambda: None)()\n    (Path(os.environ['XDG_DATA_HOME'])",
    )
    return script.replace(
        "[splash_screen]\\nenabled=false",
        "[console]\\nraw_cli_permitted=true\\n[splash_screen]\\nenabled=false",
    )


def test_native_raw_cli_output_and_terminal_hold_publication_through_finish_return(
    tmp_path,
):
    _run(
        tmp_path, "fix2_raw", "raw_lifetime", script=_raw_lifetime_script(), timeout=65
    )
