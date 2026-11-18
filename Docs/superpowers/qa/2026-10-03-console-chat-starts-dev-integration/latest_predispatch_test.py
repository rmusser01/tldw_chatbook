from pathlib import Path
p=Path('Tests/UI/test_console_cost_chip_screen.py');s=p.read_text()
old='async def test_predispatch_echo_stays_in_context_but_not_current():';new='async def test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current():'
assert old in s;s=s.replace(old,new,1)
start=s.index(new);end=s.index('\n\n@pytest',start);part=s[start:end]
old='''        assert (
            console._last_console_context_control_state.request_tokens == before_tokens
        )'''
new='''        # Unaccepted owners stay visible but cannot enter request or billed history.
        assert before_tokens > 0
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        controller = console._ensure_console_chat_controller()
        assert controller.run_state_for(session_id).status is ConsoleRunStatus.VALIDATING
        assert any(
            row.role is ConsoleMessageRole.USER and row.content == "slow readiness request"
            for row in store.messages_for_session(session_id)
        )
        assert console._last_console_context_control_state.request_tokens == 0'''
assert old in part;part=part.replace(old,new,1);s=s[:start]+part+s[end:];p.write_text(s)
