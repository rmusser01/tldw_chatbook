from pathlib import Path
p=Path('Tests/UI/test_console_runtime_ownership.py')
s=p.read_text()
anchor='@pytest.mark.asyncio\nasync def test_second_console_visit_reuses_the_runtime(tmp_path):'
addition='''def _build_manually_mounted_console_app(**kwargs):
    """Build the app for a test that supplies its own initial content screen."""
    return _build_test_app(**kwargs)


@pytest.mark.asyncio
async def test_manual_console_fixture_owns_startup_before_any_mount(tmp_path):
    """A deferred startup callback cannot add a competing retained Console."""
    app = _build_manually_mounted_console_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    # Check the ownership boundary before run_test can schedule startup.
    assert getattr(app, "_initial_screen_pushed", False)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app.current_tab = "chat"
        await _wait_for_selector(chat, pilot, "#console-native-composer")
        generation = chat._console_runtime_attachment_generation
        # Pin a late startup invocation explicitly, rather than relying on speed.
        await app._push_initial_screen()
        await pilot.pause()
        assert app.screen_stack == [app.screen_stack[0], chat]
        assert app.screen is chat
        assert app.console_runtime.view is chat
        assert app.console_runtime._attached_generation == generation
        assert "chat" not in getattr(app, "_reusable_screen_instances", {})


@pytest.mark.asyncio
async def test_public_startup_console_keeps_its_claim_across_navigation(tmp_path):
    """The shipping single retained Console reconciles and resumes delivery."""
    app = _build_test_app(config_overrides={"splash_screen": {"enabled": False}})
    persist_seeded_config(app, "splash_screen")
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await app._initial_screen_setup_task
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        await _wait_for_selector(chat, pilot, "#console-native-composer")
        for _ in range(50):
            if chat._console_attach_reconciled:
                break
            await pilot.pause(0.1)
        assert chat._console_attach_reconciled
        assert app.screen_stack == [app.screen_stack[0], chat]
        runtime = app.console_runtime
        controller = chat._ensure_console_chat_controller()
        store, bridge = runtime.chat_store, runtime.agent_bridge
        generation = chat._console_runtime_attachment_generation
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack
        assert chat.is_mounted  # Installed startup screens suspend under current dev.
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await _wait_for_selector(chat, pilot, "#console-native-composer")
        for _ in range(50):
            if chat._console_attach_reconciled:
                break
            await pilot.pause(0.1)
        assert app.screen is chat
        assert chat._console_attach_reconciled
        assert app.console_runtime is runtime
        assert (runtime.chat_controller, runtime.chat_store, runtime.agent_bridge) == (
            controller, store, bridge
        )
        assert runtime.view is chat
        assert runtime._attached_generation == generation
        assert controller.notify_run_outcome.__self__ is chat
        assert chat._console_transcript_sync_timer is None

        from tldw_chatbook.Chat.console_fleet_wake import _WakeDelivery

        session_id = store.active_session_id
        conversation_id = controller._agent_conversation_id(session_id)
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        controller.fleet_wake._active[conversation_id] = _WakeDelivery(session_id)
        try:
            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await _wait_for_selector(chat, pilot, "#console-native-composer")
            for _ in range(50):
                if chat._console_attach_reconciled:
                    break
                await pilot.pause(0.1)
            assert app.screen is chat and chat._console_attach_reconciled
            assert runtime.view is chat
            assert runtime._attached_generation == generation
            assert chat._console_transcript_sync_timer is not None
        finally:
            controller.fleet_wake._active.pop(conversation_id, None)


'''
assert anchor in s
s=s.replace(anchor,addition+anchor,1)
p.write_text(s)
