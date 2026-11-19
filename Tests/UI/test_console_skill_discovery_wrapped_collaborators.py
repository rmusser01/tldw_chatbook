"""A wrapped collaborator must not change how Console mount discovery reads skills (TASK-41642).

ADR-356 rule 4: authority comes from runtime-owned registration and current
state, not from function, code, default or closure identity. Skill mount
discovery used to compare its helpers, the controller and view methods, the
popup callback, the context services and the disabled-builtins loader with
their definition-time code. A decorator or a test patch around any of them
made discovery decline its off-loop route, so the cold trust build and the
skill read ran on the UI loop instead. Each wrapped case wraps one
collaborator with ``functools.wraps`` and drives the real discovery fetch;
``stock`` is the control. Each foreign case puts a callback on one object,
which still declines the off-loop route and never runs off the loop.
"""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, functools, inspect, os, sys, threading
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()

case = sys.argv[2]


def wrap(owner, name, calls):
    original = inspect.getattr_static(owner, name)
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            calls.append(name)
            return await original(*args, **kwargs)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)
    setattr(owner, name, wrapped)
    return lambda: setattr(owner, name, original)


async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring as wiring
    from tldw_chatbook.app_skill_wiring import SkillServiceWiringMixin
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.UI.Console_Modules import skill as skill_source
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets import compact_model_bar
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators

    # Where each read runs: the stock route builds trust and reads skills off
    # the UI loop; a declined route does both on it.
    main = threading.main_thread()
    threads = {'builder': [], 'context': []}
    monitoring = sys.monitoring
    tool = next(index for index in range(6) if monitoring.get_tool(index) is None)
    monitoring.use_tool_id(tool, 'skill-discovery-wrapped')
    watched = {
        SkillServiceWiringMixin._build_local_skill_trust_service.__code__: 'builder',
        LocalSkillsService.get_context.__code__: 'context',
    }

    def started(code, _offset):
        role = watched.get(code)
        if role is not None:
            threads[role].append(threading.current_thread() is main)

    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    for code in watched:
        monitoring.set_local_events(tool, code, monitoring.events.PY_START)

    calls, restores = [], []
    if case == 'loader':
        # The app registers this partial when it builds its skills stack, so
        # the decorator must be in place before the app exists.
        restores.append(wrap(wiring, '_read_app_disabled_builtin_skills', calls))
    creators = OriginalPrivateAppCreators(app_source, Path(os.environ['XDG_DATA_HOME']).parent)
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
    controller = screen._skill
    local = app.local_skills_service
    assert vars(app)['_local_skill_trust_service'] is None and vars(local)['_trust_service'] is None

    wrapped = {
        'trust_preparation': (wiring, '_capture_console_skill_trust_preparation'),
        'trust_service_preparation': (wiring, '_prepare_console_skill_trust_service'),
        'trust_source': (wiring, '_capture_console_skill_trust_source'),
        'plain_fields': (compact_model_bar, '_plain_fields'),
        'controller_method': (skill_source.ConsoleSkillController, '_fetch_console_skill_context'),
        'popup': (skill_source, '_sync_console_skill_popup'),
        'context_service': (LocalSkillsService, 'get_context'),
    }
    foreign_calls = []

    def foreign(*args, **kwargs):
        foreign_calls.append(threading.current_thread() is main)
        return frozenset()

    foreign_slots = {
        'instance_controller_override': (vars(controller), '_refresh_console_skill_candidates'),
        'instance_view_popup': (vars(screen), '_sync_console_command_popup'),
        'foreign_loader': (vars(local), '_builtin_disabled_loader'),
    }
    if case in wrapped:
        restores.append(wrap(*wrapped[case], calls))
    elif case in foreign_slots:
        fields, name = foreign_slots[case]
        fields[name] = foreign
    elif case == 'foreign_policy_provider':
        # A state provider the app did not register for its policy enforcer.
        policy_fields = vars(vars(local)['policy_enforcer'])
        registered = policy_fields['_state_provider']

        def foreign_provider():
            foreign_calls.append(threading.current_thread() is main)
            return registered()

        policy_fields['_state_provider'] = foreign_provider
        restores.append(lambda: policy_fields.__setitem__('_state_provider', registered))
    try:
        with app._context():
            context = await controller._fetch_console_skill_context()
    finally:
        for restore in reversed(restores):
            restore()
        for code in watched:
            monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)
        await runtime.dispose()
        creators.retire()
    assert 'available_skills' in context, context
    assert vars(app)['_local_skill_trust_service'] is not None
    assert threads['builder'] and threads['context'], threads
    if case in foreign_slots or case == 'foreign_policy_provider':
        # The off-loop read declines: the skills are read on the UI loop, and
        # a foreign callback never runs anywhere else.
        assert threads['context'] == [True] * len(threads['context']), threads
        assert all(foreign_calls), foreign_calls
        if case in {'foreign_loader', 'foreign_policy_provider'}:
            assert foreign_calls, f'{case}: the foreign callback never ran'
    else:
        # The same route as stock: trust built and skills read off the loop.
        assert not any(threads['builder']), f'{case}: trust built on the UI loop {threads}'
        assert not any(threads['context']), f'{case}: skills read on the UI loop {threads}'
        if case not in {'stock', 'popup'}:
            assert calls, f'{case}: the wrapped collaborator never ran'
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selected.write_text('[general]\nusers_name="skill-discovery-wrapped"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[api_settings.openai]\napi_key="fixture-not-a-real-key"\n[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n', encoding='utf-8')
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=120))
"""


@pytest.mark.parametrize(
    "case",
    [
        "stock",
        "trust_preparation",
        "trust_service_preparation",
        "trust_source",
        "plain_fields",
        "controller_method",
        "popup",
        "context_service",
        "loader",
    ],
)
def test_wrapped_skill_collaborator_keeps_the_off_loop_discovery(tmp_path, case):
    _run(tmp_path, "skill_discovery_wrapped", case, script=_SCRIPT, timeout=180)


@pytest.mark.parametrize(
    "case",
    [
        "instance_controller_override",
        "instance_view_popup",
        "foreign_loader",
        "foreign_policy_provider",
    ],
)
def test_foreign_skill_callback_declines_the_off_loop_discovery(tmp_path, case):
    _run(tmp_path, "skill_discovery_wrapped", case, script=_SCRIPT, timeout=180)
