from __future__ import annotations

import asyncio
import io
import json
import logging
import threading

import pytest
from loguru import logger

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_LIBRARY, TAB_PERSONAS
from tldw_chatbook.UI.CCP_Modules import ccp_character_handler
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_ROW_BROWSE_PROMPTS,
    LibraryScreen,
)
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


REMOVED_ROOT_NAMES = (
    "ccp_active_view",
    "ccp_api_provider_value",
    "current_editing_character_id",
    "current_editing_character_data",
    "conv_char_sidebar_left_collapsed",
    "conv_char_sidebar_right_collapsed",
    "current_conv_char_tab_conversation_id",
    "current_ccp_character_details",
    "current_prompt_id",
    "current_prompt_uuid",
    "current_prompt_name",
    "current_prompt_author",
    "current_prompt_details",
    "current_prompt_system",
    "current_prompt_user",
    "current_prompt_keywords_str",
    "current_prompt_version",
    "current_ccp_character_image",
    "_conv_char_search_timer",
    "_ccp_conversation_search_generation",
)


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    return app


async def _wait_for_screen(app: TldwCli, pilot, screen_type, canonical_tab: str):
    for _ in range(300):
        if isinstance(app.screen, screen_type) and app.current_tab == canonical_tab:
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"production TldwCli did not finish routing to {screen_type.__name__}"
    )


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_real_personas_and_library_own_character_and_prompt_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = _production_app(monkeypatch)
    diagnostic_output = io.StringIO()
    diagnostic_sink: int | None = None
    character_name = "TASK-651 imported character"
    prompt_name = "TASK-651 imported prompt"
    private_system_body = "TASK_651_PRIVATE_SYSTEM_BODY"
    private_user_body = "TASK_651_PRIVATE_USER_BODY"
    private_character_error = "TASK_651_PRIVATE_CHARACTER_IMPORT_ERROR"
    character_path = tmp_path / "task-651-character.json"
    prompt_path = tmp_path / "task-651-prompt.json"
    stale_character_name = "TASK-651 stale completion character"
    stale_character_path = tmp_path / "task-651-stale-character.json"
    failed_prompt_path = tmp_path / "task-651-failed-prompt.json"
    character_path.write_text(
        json.dumps(
            {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    "name": character_name,
                    "description": "A TASK-651 ownership sentinel.",
                    "personality": "Careful.",
                    "scenario": "Production application import.",
                    "first_mes": "Hello.",
                    "mes_example": "",
                },
            }
        ),
        encoding="utf-8",
    )
    prompt_path.write_text(
        json.dumps(
            {
                "name": prompt_name,
                "author": "TASK-651",
                "details": "Ownership sentinel",
                "system_prompt": private_system_body,
                "user_prompt": private_user_body,
                "keywords": ["task-651", "ownership"],
            }
        ),
        encoding="utf-8",
    )
    stale_character_path.write_text(
        json.dumps(
            {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    "name": stale_character_name,
                    "description": "Durable import after owner navigation.",
                    "personality": "Patient.",
                    "scenario": "The original screen is unmounted.",
                    "first_mes": "Hello later.",
                    "mes_example": "",
                },
            }
        ),
        encoding="utf-8",
    )
    failed_prompt_path.write_text(
        json.dumps(
            {
                "name": "TASK-651 bounded failure prompt",
                "system_prompt": private_system_body,
                "user_prompt": private_user_body,
            }
        ),
        encoding="utf-8",
    )

    try:
        async with app.run_test(size=(150, 48)) as pilot:
            # TldwCli configures Loguru during mount and removes earlier sinks,
            # so attach the privacy sentinel only after the real app is live.
            diagnostic_sink = logger.add(diagnostic_output, format="{message}")
            app.post_message(NavigateToScreen("ccp"))
            personas = await _wait_for_screen(app, pilot, PersonasScreen, TAB_PERSONAS)
            assert app.current_tab == TAB_PERSONAS
            assert personas.state.active_mode == "characters"

            character_worker = personas._start_character_import(str(character_path))
            assert character_worker.node is app
            await character_worker.wait()
            await pilot.pause()
            assert personas.state.selected_entity_kind == "character"
            assert personas.state.selected_entity_name == character_name

            with monkeypatch.context() as character_failure:

                def fail_character_import(_path: str):
                    raise RuntimeError(private_character_error)

                character_failure.setattr(
                    ccp_character_handler,
                    "import_character_card",
                    fail_character_import,
                )
                failed_character_worker = personas._start_character_import(
                    str(character_path)
                )
                assert failed_character_worker.node is app
                await failed_character_worker.wait()

            app.post_message(NavigateToScreen("prompts"))
            library = await _wait_for_screen(app, pilot, LibraryScreen, TAB_LIBRARY)
            assert app.current_tab == TAB_LIBRARY
            assert library._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS

            library._library_prompts_import_path = str(prompt_path)
            prompt_worker = library._start_library_prompts_import()
            assert prompt_worker is not None
            assert prompt_worker.node is app
            await prompt_worker.wait()
            await pilot.pause()
            assert library._library_prompts_import_status.startswith("1 imported")
            assert all(not hasattr(app, name) for name in REMOVED_ROOT_NAMES)

            async def fail_prompt_save(**kwargs):
                raise RuntimeError(
                    f"{kwargs['system_prompt']}::{kwargs['user_prompt']}"
                )

            monkeypatch.setattr(
                app.prompt_scope_service,
                "save_prompt",
                fail_prompt_save,
            )
            library._library_prompts_import_path = str(failed_prompt_path)
            failed_prompt_worker = library._start_library_prompts_import()
            assert failed_prompt_worker is not None
            assert failed_prompt_worker.node is app
            await failed_prompt_worker.wait()
            assert "1 failed" in library._library_prompts_import_status

            app.post_message(NavigateToScreen("ccp"))
            stale_personas = await _wait_for_screen(
                app, pilot, PersonasScreen, TAB_PERSONAS
            )
            refresh_calls: list[str] = []

            async def record_stale_refresh() -> None:
                refresh_calls.append("refreshed")

            monkeypatch.setattr(
                stale_personas.character_handler,
                "refresh_character_list",
                record_stale_refresh,
            )
            import_started = threading.Event()
            release_import = threading.Event()
            real_import = ccp_character_handler.import_character_card

            def delayed_real_import(path: str):
                import_started.set()
                if not release_import.wait(timeout=5):
                    raise TimeoutError("TASK-651 import release timed out")
                return real_import(path)

            monkeypatch.setattr(
                ccp_character_handler,
                "import_character_card",
                delayed_real_import,
            )
            import_worker = stale_personas._start_character_import(
                str(stale_character_path)
            )
            assert import_worker.node is app
            assert await asyncio.to_thread(import_started.wait, 5)
            app.post_message(NavigateToScreen("prompts"))
            await _wait_for_screen(app, pilot, LibraryScreen, TAB_LIBRARY)
            # Textual leaves ``is_mounted`` true after pruning; a closed,
            # detached message pump is the completed-unmount invariant.
            assert stale_personas._closed
            assert stale_personas._parent is None
            assert not import_worker.is_finished
            release_import.set()
            await import_worker.wait()
            assert refresh_calls == []

            app.post_message(NavigateToScreen("ccp"))
            fresh_personas = await _wait_for_screen(
                app, pilot, PersonasScreen, TAB_PERSONAS
            )
            for _ in range(300):
                if any(
                    record.get("name") == stale_character_name
                    for record in fresh_personas._characters
                ):
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError(
                    "fresh Personas owner did not load the durably imported character"
                )

        rendered_logs = diagnostic_output.getvalue()
        assert "Library prompt save failed" in rendered_logs
        assert (
            "Character import failed (file_type=.json, category=RuntimeError)."
            in rendered_logs
        )
        assert "RuntimeError" in rendered_logs
        assert private_character_error not in rendered_logs
        assert private_system_body not in rendered_logs
        assert private_user_body not in rendered_logs
    finally:
        if diagnostic_sink is not None:
            try:
                logger.remove(diagnostic_sink)
            except ValueError:
                pass
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_real_library_prompt_batch_survives_owner_unmount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = _production_app(monkeypatch)
    prompt_names = (
        "TASK-651 durable prompt one",
        "TASK-651 durable prompt two",
    )
    prompt_path = tmp_path / "task-651-durable-prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "system_prompt": f"System for {name}",
                    "user_prompt": f"User for {name}",
                }
                for name in prompt_names
            ]
        ),
        encoding="utf-8",
    )
    save_started = threading.Event()
    release_save = threading.Event()

    try:
        async with app.run_test(size=(150, 48)) as pilot:
            app.post_message(NavigateToScreen("prompts"))
            library = await _wait_for_screen(app, pilot, LibraryScreen, TAB_LIBRARY)
            real_save_prompt = app.prompt_scope_service.save_prompt
            saved_names: list[str] = []

            async def delayed_save_prompt(**kwargs):
                saved_names.append(kwargs["name"])
                if len(saved_names) == 1:
                    save_started.set()
                    if not await asyncio.to_thread(release_save.wait, 5):
                        raise TimeoutError("TASK-651 prompt import release timed out")
                return await real_save_prompt(**kwargs)

            monkeypatch.setattr(
                app.prompt_scope_service,
                "save_prompt",
                delayed_save_prompt,
            )
            library._library_prompts_import_path = str(prompt_path)
            import_worker = library._start_library_prompts_import()
            assert import_worker is not None
            assert import_worker.node is app
            assert await asyncio.to_thread(save_started.wait, 5)

            # Repeated starts preserve a single durable slot and must not
            # cancel or replace the in-flight batch.
            assert library._start_library_prompts_import() is import_worker

            app.post_message(NavigateToScreen("ccp"))
            await _wait_for_screen(app, pilot, PersonasScreen, TAB_PERSONAS)
            assert library._closed
            assert library._parent is None
            assert not import_worker.is_finished

            release_save.set()
            await import_worker.wait()
            assert saved_names == list(prompt_names)

            app.post_message(NavigateToScreen("prompts"))
            fresh_library = await _wait_for_screen(
                app, pilot, LibraryScreen, TAB_LIBRARY
            )
            for _ in range(300):
                visible_names = {
                    row.name
                    for row in fresh_library._build_library_prompts_state().rows
                }
                if set(prompt_names) <= visible_names:
                    break
                await pilot.pause(0.01)
            else:
                raise AssertionError(
                    "fresh Library owner did not load the completed prompt batch"
                )
    finally:
        release_save.set()
        await _close_production_app(app)
