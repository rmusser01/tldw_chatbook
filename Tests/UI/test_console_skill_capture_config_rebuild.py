"""Native skill capture must not fall back to UI config IO during a rebuild."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


def test_stock_skill_capture_keeps_rebuild_out_of_its_snapshot(tmp_path):
    _run(tmp_path, "skill", "capture", timeout=240, script=r'''
import asyncio, inspect, os, sys, threading, traceback
from pathlib import Path
from Tests.windows_private_fixture_runner import user_fixture_default_owner
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()

async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring as wiring, config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules.skill import _capture_console_skill_discovery
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators

    creators = OriginalPrivateAppCreators(app_source, Path(os.environ["XDG_DATA_HOME"]).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    screen = ChatScreen(app)
    runtime.attach_view(screen)
    service = app.skills_scope_service
    local = app.local_skills_service
    capture = wiring._capture_console_skill_trust_preparation
    source_capture = wiring._capture_console_skill_trust_source
    current_code = next(code for code in source_capture.__code__.co_consts
                        if inspect.iscode(code) and code.co_name == "source_current")
    requested, held, release = threading.Event(), threading.Event(), threading.Event()
    timed_out, prepared = [], []

    def hold_rebuild():
        if not requested.wait(10):
            return
        with config._settings_rebuild_lock(), config._config_file_lock():
            config._invalidate_config_caches()
            held.set()
            if not release.wait(2):
                timed_out.extend(traceback.format_stack(sys._current_frames()[threading.main_thread().ident]))
                release.set()

    def observe(frame, event, result):
        if frame.f_code is current_code and event == "call" and not requested.is_set():
            # The original source reader has already sampled its cache.
            # Without retained locks the rebuild now invalidates that sample.
            requested.set()
            held.wait(0.05)
        elif frame.f_code is capture.__code__ and event == "return":
            prepared.append(result is not None)

    holder = threading.Thread(target=hold_rebuild)
    old_profile = sys.getprofile()
    try:
        with app._context():
            owner = _capture_console_skill_discovery(screen._skill)
            assert owner is not None and owner[0](), "original discovery owner must qualify"
            assert capture(app, service, owner[0], owner[1]) is not None
            assert app._local_skill_trust_service is None and local._trust_service is None
            holder.start()
            sys.setprofile(observe)
            try:
                await screen._skill._fetch_console_skill_context()
            finally:
                sys.setprofile(old_profile)
                release.set()
                holder.join(3)
            assert requested.is_set() and not holder.is_alive()
            assert not timed_out, "native skill fallback blocked the UI loop:\n" + "".join(timed_out[-16:])
            assert prepared and prepared[0], "stock preparation declined its raced cache snapshot"
            # A source change may refuse this attempt; a fresh attempt remains live.
            context = await screen._skill._fetch_console_skill_context()
            assert "available_skills" in context
    finally:
        sys.setprofile(old_profile)
        release.set()
        if holder.ident is not None:
            holder.join(3)
        await runtime.dispose()
        creators.retire()
    print("retired and reopened")

with user_fixture_default_owner():
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.write_text('[general]\nusers_name="skill-capture"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="fixture-not-a-real-key"\n[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n', encoding="utf-8")
    config_path.chmod(0o600)
    asyncio.run(exercise())
''')


def test_stock_skill_context_capture_keeps_rebuild_out_of_its_snapshot(tmp_path):
    """Keep the second stock capture on its qualified native-worker route."""
    _run(tmp_path, "skill", "context-capture", timeout=240, script=r'''
import asyncio, inspect, os, sys, threading, traceback
from pathlib import Path
from Tests.windows_private_fixture_runner import user_fixture_default_owner
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()

async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring as wiring, config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules.skill import _capture_console_skill_discovery, _capture_console_skill_context_read
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators

    creators = OriginalPrivateAppCreators(app_source, Path(os.environ["XDG_DATA_HOME"]).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    screen = ChatScreen(app)
    runtime.attach_view(screen)
    service = app.skills_scope_service
    local = app.local_skills_service
    with app._context():
        await screen._skill._fetch_console_skill_context()
        assert app._local_skill_trust_service is not None
    context_capture = _capture_console_skill_context_read
    capture = wiring._capture_console_skill_trust_preparation
    source_capture = wiring._capture_console_skill_trust_source
    current_code = next(code for code in source_capture.__code__.co_consts
                        if inspect.iscode(code) and code.co_name == "source_current")
    requested, held, release = threading.Event(), threading.Event(), threading.Event()
    timed_out, prepared, context_prepared = [], [], []

    def inside_context_capture(frame):
        caller = frame.f_back
        while caller is not None:
            if caller.f_code is context_capture.__code__:
                return True
            caller = caller.f_back
        return False

    def hold_rebuild():
        if not requested.wait(10):
            return
        with config._settings_rebuild_lock(), config._config_file_lock():
            config._invalidate_config_caches()
            held.set()
            if not release.wait(2):
                timed_out.extend(traceback.format_stack(sys._current_frames()[threading.main_thread().ident]))
                release.set()

    def observe(frame, event, result):
        if frame.f_code is current_code and event == "call" and not requested.is_set() and inside_context_capture(frame):
            # The original source reader has already sampled its cache.
            # Without retained locks the rebuild now invalidates that sample.
            requested.set()
            held.wait(0.05)
        elif frame.f_code is capture.__code__ and event == "return":
            prepared.append(result is not None)
        elif frame.f_code is context_capture.__code__ and event == "return":
            context_prepared.append(result is not None)

    holder = threading.Thread(target=hold_rebuild)
    old_profile = sys.getprofile()
    try:
        with app._context():
            owner = _capture_console_skill_discovery(screen._skill)
            assert owner is not None and owner[0](), "original discovery owner must qualify"
            assert context_capture(app, service, wiring, owner[2]) is not None, "warm original context reader must qualify"
            holder.start()
            sys.setprofile(observe)
            try:
                await screen._skill._fetch_console_skill_context()
            finally:
                sys.setprofile(old_profile)
                release.set()
                holder.join(3)
            assert requested.is_set() and not holder.is_alive()
            print("SECOND_CAPTURE_RESULT", context_prepared, "FIRST_CAPTURE_RESULT", prepared, flush=True)
            assert not timed_out, "second native skill capture fallback blocked the UI loop:\n" + "".join(timed_out[-20:])
            assert context_prepared and context_prepared[0], "stock context capture declined its raced cache snapshot"
            # A source change may refuse this attempt; a fresh attempt remains live.
            context = await screen._skill._fetch_console_skill_context()
            assert "available_skills" in context
    finally:
        sys.setprofile(old_profile)
        release.set()
        if holder.ident is not None:
            holder.join(3)
        await runtime.dispose()
        creators.retire()
    print("retired and reopened")

with user_fixture_default_owner():
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.write_text('[general]\nusers_name="skill-capture"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="fixture-not-a-real-key"\n[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n', encoding="utf-8")
    config_path.chmod(0o600)
    asyncio.run(exercise())
''')
