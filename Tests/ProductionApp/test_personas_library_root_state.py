# ruff: noqa: E402

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import threading

import pytest
from loguru import logger

# Exercise the full production app in its supported "optional transcription
# backend absent" configuration. The installed parakeet-mlx wheel aborts the
# interpreter while importing MLX in this test runner, before Textual can mount.
_MISSING_MODULE = object()
_previous_parakeet_mlx = sys.modules.get("parakeet_mlx", _MISSING_MODULE)
sys.modules["parakeet_mlx"] = None

try:
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
finally:
    if _previous_parakeet_mlx is _MISSING_MODULE:
        sys.modules.pop("parakeet_mlx", None)
    else:
        sys.modules["parakeet_mlx"] = _previous_parakeet_mlx


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


async def _wait_for_screen(app: TldwCli, pilot, screen_type):
    for _ in range(300):
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"production TldwCli did not mount {screen_type.__name__}")


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
            personas = await _wait_for_screen(app, pilot, PersonasScreen)
            assert app.current_tab == TAB_PERSONAS
            assert personas.state.active_mode == "characters"

            await personas._import_character_from_path(str(character_path))
            await pilot.pause()
            assert personas.state.selected_entity_kind == "character"
            assert personas.state.selected_entity_name == character_name

            app.post_message(NavigateToScreen("prompts"))
            library = await _wait_for_screen(app, pilot, LibraryScreen)
            assert app.current_tab == TAB_LIBRARY
            assert library._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS

            await library._run_library_prompts_import(str(prompt_path))
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
            await library._run_library_prompts_import(str(failed_prompt_path))
            assert "1 failed" in library._library_prompts_import_status

            app.post_message(NavigateToScreen("ccp"))
            stale_personas = await _wait_for_screen(app, pilot, PersonasScreen)
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
            import_task = asyncio.create_task(
                stale_personas._import_character_from_path(str(stale_character_path))
            )
            assert await asyncio.to_thread(import_started.wait, 5)
            app.post_message(NavigateToScreen("prompts"))
            await _wait_for_screen(app, pilot, LibraryScreen)
            release_import.set()
            await import_task
            assert refresh_calls == []

            app.post_message(NavigateToScreen("ccp"))
            fresh_personas = await _wait_for_screen(app, pilot, PersonasScreen)
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
        assert "RuntimeError" in rendered_logs
        assert private_system_body not in rendered_logs
        assert private_user_body not in rendered_logs
    finally:
        if diagnostic_sink is not None:
            try:
                logger.remove(diagnostic_sink)
            except ValueError:
                pass
        await _close_production_app(app)
