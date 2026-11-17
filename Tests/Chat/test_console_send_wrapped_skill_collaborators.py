"""A wrapped skill collaborator must not change what a cold Send captures (TASK-41319).

ADR-356 rule 4: authority comes from runtime-owned registration and current
state, not from function, code, default or closure identity. A decorator or a
test patch around one of these collaborators used to make the cold Send's
configuration preparation decline, so the turn lost its native skill-trust
capture. Each case wraps one collaborator with ``functools.wraps`` (the
collaborator still runs) and drives the real received Send; the stock case is
the control.
"""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, functools, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()

case = sys.argv[2]

async def exercise():
    from tldw_chatbook import app as app_source, app_service_wiring
    import tldw_chatbook.Widgets.compact_model_bar
    from tldw_chatbook.Chat import console_configuration_capture as capture_source
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators

    # A managed skill makes the Send build the cold trust service and record it.
    seed = LocalSkillsService(
        store_dir=app_service_wiring.default_local_skills_store_dir(app_service_wiring.get_user_data_dir()),
    )
    await seed.create_skill(
        name='wrapped-managed',
        content='---\nname: wrapped-managed\ndescription: Wrapped collaborator control\n---\nBody.\n',
        trust_approved=False,
    )
    creators = OriginalPrivateAppCreators(app_source, Path(os.environ['XDG_DATA_HOME']).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    runtime = app.console_runtime
    local = app.local_skills_service
    assert vars(app)['_local_skill_trust_service'] is None
    assert vars(local)['_trust_service'] is None
    app.local_mcp_control_service.manifest_provider = lambda: {'tools': []}
    store = runtime.ensure_chat_store()
    session = store.create_session(workspace_id='global')
    store.set_session_draft(session.id, 'wrapped collaborator send')
    gateway = _RecordingGateway()
    controller = runtime.ensure_chat_controller(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True,
    )
    controller._agent_bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store, provider_gateway_factory=lambda: gateway,
        skills_service=app.skills_scope_service,
    )
    runtime.ensure_hook_permissions()

    # Wrap the collaborator where the product resolves it, as a decorator would.
    targets = {
        'stock': None,
        'trust_source': (app_service_wiring, '_capture_console_skill_trust_source'),
        'metadata': (app_service_wiring, '_console_skill_metadata_current'),
        'source_current': (app_service_wiring, '_console_skill_source_current'),
        'skill_capture': (capture_source, '_capture_skill_context_from_records'),
    }
    calls = []
    if targets[case] is not None:
        module, name = targets[case]
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        setattr(module, name, wrapped)

    result = None
    try:
        source = preparation.capture_console_received_configuration_preparation(
            app, store, controller, session_id=session.id,
        )
        assert source is not None, f'{case}: cold Send preparation declined'
        assert source.trust_source is not None, f'{case}: no skill-trust capture'
        assert source.trust_source.app is app and source.trust_source.local is local
        assert source.trust_source.scope is app.skills_scope_service
        selection = preparation.ConsoleTurnCaptureSelection(
            provider_selection=ConsoleProviderSelection(provider='llama_cpp', explicit_model='m'),
            presentation_context=None, rag_defaults={},
            tool_configuration={'agent_runtime_enabled': True, 'local_tools_enabled': False},
            skill_workspace_id=None, project_bindings_eligible=False, agent_runtime_enabled=True,
        )
        launch, revision, _ = runtime.snapshot_console_staged_evidence()
        intent = ConsoleReceivedTurnIntent(
            'wrapped-collaborator', session.id, store.session_input_snapshot(session.id),
            selection, launch, revision, runtime._attached_generation or 0, None,
        )
        turn_id = runtime.accept_received_intent(intent, _configuration_preparation=source)
        result = await asyncio.wait_for(runtime._turn_custody[turn_id].task, 30)
        winner = vars(app)['_local_skill_trust_service']
        receipt = vars(local).get('_console_native_skill_trust_publication')
    finally:
        if targets[case] is not None:
            setattr(module, name, original)
        await runtime.dispose()
        creators.retire()
    assert result.accepted, result
    assert gateway.payloads, 'the model was never called'
    # The same skill trust is built and recorded as on the stock path.
    assert winner is not None
    assert type(receipt) is preparation._NativeSkillTrustPublication, receipt
    assert receipt.source is source.trust_source
    assert receipt.initial_app is None and receipt.initial_local is None
    assert receipt.app_winner is winner and receipt.local_winner is winner
    if case != 'stock':
        assert calls, f'{case}: the wrapped collaborator never ran'
    assert not network_guard.blocked_attempts()
    assert not real_profile_guard.take_violations()
    print('retired and reopened')

with user_fixture_default_owner():
    selected = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selected.write_text('[general]\nusers_name="wrapped-collaborator"\n[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[chat_defaults]\nprovider="llama_cpp"\nmodel="m"\n', encoding='utf-8')
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=70))
"""


@pytest.mark.parametrize(
    "case", ["stock", "trust_source", "metadata", "source_current", "skill_capture"]
)
def test_wrapped_skill_collaborator_send_records_the_same_trust(tmp_path, case):
    _run(tmp_path, "wrapped_collaborator", case, script=_SCRIPT, timeout=90)
