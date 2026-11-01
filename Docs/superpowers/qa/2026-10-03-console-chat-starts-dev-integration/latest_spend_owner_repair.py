from pathlib import Path
p=Path('tldw_chatbook/UI/Screens/chat_screen.py');s=p.read_text()
start=s.index('    async def _dispatch_console_draft_send(');end=s.index('\n    def _note_console_follow_intent',start);part=s[start:end]
old='''        from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

        async with send_diagnostic_scope('''
new='''        from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

        self._console_draft_spend_refresh.stop()
        async with send_diagnostic_scope('''
assert part.count(old)==1;part=part.replace(old,new,1);s=s[:start]+part+s[end:];p.write_text(s)
p=Path('Tests/UI/test_console_spend_projection.py');s=p.read_text();anchor='\n\n';pos=s.index('\n\ndef ',s.index('import pytest'));s=s[:pos]+'\n\n# The UI autouse app consumer retains its collection-selected private profile.\npytestmark = pytest.mark.bootstrap_profile'+s[pos:];p.write_text(s)
p=Path('Tests/UI/test_console_cost_chip_screen.py');s=p.read_text();anchor='\n\n@pytest.mark.asyncio\nasync def test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current():'
new='''

@pytest.mark.asyncio
async def test_refused_send_cancels_idle_refresh_and_next_idle_edit_rearms(monkeypatch):
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        refresh = console._console_draft_spend_refresh
        callback = Mock(wraps=refresh.refresh)
        monkeypatch.setattr(type(refresh), "refresh", callback)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: "test send refused"
        )
        composer.load_draft("kept refused draft")
        await pilot.pause(0.01)
        assert refresh.timer is not None

        assert await console._dispatch_console_draft_send(
            composer.draft_text(), session_id=console._console_visible_send_session_id()
        ) is False
        assert composer.draft_text() == "kept refused draft"
        assert refresh.timer is None
        callback.assert_not_called()

        composer.load_draft("edited after refusal")
        await pilot.pause(0.01)
        assert refresh.timer is not None
        for _ in range(100):
            await pilot.pause(0.01)
            if callback.call_count:
                break
        callback.assert_called_once()
        assert refresh.timer is None
'''
assert anchor in s;s=s.replace(anchor,new+anchor,1);p.write_text(s)
