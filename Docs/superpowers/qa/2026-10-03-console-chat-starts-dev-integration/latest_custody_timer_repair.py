from pathlib import Path
p=Path('tldw_chatbook/Chat/console_runtime.py');s=p.read_text();old='''    def has_custodied_turns(self) -> bool:
        """Whether an accepted turn is in custody, even before its run starts.

        Returns:
            True until every accepted turn's task has finished.
        """
        return bool(self._turn_custody)''';new='''    def has_custodied_turns(self, session_id: str | None = None) -> bool:
        """Whether accepted work is in custody, even before its run starts.

        Args:
            session_id: Limit the check to one chat, or check all chats when unset.

        Returns:
            True until the selected accepted turn tasks have finished.
        """
        if session_id is None:
            return bool(self._turn_custody)
        return any(record.session_id == session_id for record in self._turn_custody.values())''';assert old in s;s=s.replace(old,new,1);p.write_text(s)
p=Path('tldw_chatbook/UI/Screens/chat_screen.py');s=p.read_text();old='''                and controller.run_state_for(session_id).status
                in CONSOLE_ACTIVE_RUN_STATUSES
            )
        )''';new='''                and (
                    controller.run_state_for(session_id).status
                    in CONSOLE_ACTIVE_RUN_STATUSES
                    or self._console_runtime().has_custodied_turns(session_id)
                )
            )
        )''';start=s.index('    def _on_console_composer_draft_changed');end=s.index('\n    @on(',start);part=s[start:end];assert part.count(old)==1;part=part.replace(old,new,1);s=s[:start]+part+s[end:];p.write_text(s)
p=Path('Tests/UI/test_console_cost_chip_screen.py');s=p.read_text();anchor='\n\n@pytest.mark.asyncio\nasync def test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current():';new='''

@pytest.mark.asyncio
async def test_runtime_custody_cancels_before_validating_and_leaves_other_chat_idle(monkeypatch):
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)
    release = asyncio.Event()
    entered = asyncio.Event()

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        controller = console._ensure_console_chat_controller()
        runtime = console._console_runtime()
        source_id = store.active_session_id
        assert runtime.has_custodied_turns() is False
        assert runtime.has_custodied_turns(source_id) is False

        async def hold_before_controller(record, **kwargs):
            entered.set()
            await release.wait()
            return None

        monkeypatch.setattr(runtime, "_run_custodied_turn", hold_before_controller)
        refresh = console._console_draft_spend_refresh
        callback = Mock(wraps=refresh.refresh)
        monkeypatch.setattr(type(refresh), "refresh", callback)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("runtime admitted before validating")
        await pilot.pause(0.01)
        assert refresh.timer is not None
        assert await console._dispatch_console_draft_send(
            composer.draft_text(), session_id=source_id
        ) is True
        await asyncio.wait_for(entered.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await pilot.pause(0.3)
        assert controller.run_state_for(source_id).status is ConsoleRunStatus.IDLE
        assert runtime.has_custodied_turns() is True
        assert runtime.has_custodied_turns(source_id) is True
        assert refresh.timer is None
        callback.assert_not_called()
        turn_id = next(iter(runtime._turn_custody))

        other = store.create_session(title="Other idle chat", settings=store.session_settings(source_id))
        await console._session._activate_native_console_session(other.id)
        assert runtime.has_custodied_turns(other.id) is False
        assert runtime.has_custodied_turns() is True
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("other chat forecast")
        await pilot.pause(0.01)
        assert refresh.timer is not None
        for _ in range(100):
            await pilot.pause(0.01)
            if callback.call_count:
                break
        callback.assert_called_once()
        assert refresh.timer is None

        release.set()
        await runtime.wait_for_turn(turn_id)
        await pilot.pause()
        assert runtime.has_custodied_turns() is False
        assert runtime.has_custodied_turns(source_id) is False
''';assert anchor in s;s=s.replace(anchor,new+anchor,1);p.write_text(s)
