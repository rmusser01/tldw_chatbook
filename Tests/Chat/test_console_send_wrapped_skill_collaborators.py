"""A wrapped skill collaborator must not change what a cold Send captures (TASK-41319).

ADR-356 rule 4: authority comes from runtime-owned registration and current
state, not from function, code, default or closure identity. A decorator or a
test patch around one of these collaborators used to make the cold Send's
configuration preparation decline, so the turn lost its native skill-trust
capture. Each case wraps one collaborator with ``functools.wraps`` (the
collaborator still runs) and drives the real received Send; the stock case is
the control.
"""

import functools
from types import SimpleNamespace

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
    from tldw_chatbook import app as app_source, app_service_wiring, app_skill_wiring
    from tldw_chatbook.Widgets import compact_model_bar
    from tldw_chatbook.Chat import console_configuration_capture as capture_source
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
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
    skill_contexts = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            # The skill context the admitted turn was captured with.
            tree = current_ordinary_tree()
            skill_contexts.append(tree.original_configuration.skill_context_maximum)
            async for chunk in super().stream_chat(resolution, messages, **kwargs):
                yield chunk

    gateway = Gateway()
    controller = runtime.ensure_chat_controller(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True,
    )
    controller._agent_bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store, provider_gateway_factory=lambda: gateway,
        skills_service=app.skills_scope_service,
    )
    runtime.ensure_hook_permissions()

    refusals = {
        # A slot that no longer holds the callback the app registered for its
        # skills stack is refused before anything invokes it.
        'foreign_trust_factory': (local, '_trust_service_factory'),
        'foreign_plugin_factory': (local, '_plugin_service_factory'),
        'foreign_loader': (local, '_builtin_disabled_loader'),
        # Per-instance overrides of skill methods still decline.
        'instance_catalog_override': (local, '_visible_records'),
        # Refused only by the catalog check (_skill_catalog_sources_current).
        'instance_trust_fields_override': (local, '_trust_fields_for_record'),
        'instance_trust_override': (app, 'ensure_local_skill_trust_service'),
    }
    if case in refusals:
        owner, name = refusals[case]
        foreign_calls = []

        def foreign(*args, **kwargs):
            foreign_calls.append(name)
            raise AssertionError(f'{case}: the foreign callback ran')

        assert vars(local).get('_plugin_service') is None  # its factory is consulted
        vars(owner)[name] = foreign
        try:
            source = preparation.capture_console_received_configuration_preparation(
                app, store, controller, session_id=session.id,
            )
        finally:
            await runtime.dispose()
            creators.retire()
        assert source is None, f'{case}: the cold Send preparation was accepted'
        assert not foreign_calls, foreign_calls
        assert vars(app)['_local_skill_trust_service'] is None
        assert not network_guard.blocked_attempts()
        assert not real_profile_guard.take_violations()
        print('retired and reopened')
        return

    # Wrap the collaborator where the product resolves it, as a decorator would.
    targets = {
        'stock': None,
        # skill trust: the trust-source capture and the trust builder
        'trust_source': (app_service_wiring, '_capture_console_skill_trust_source'),
        'trust_builder': (app_skill_wiring.SkillServiceWiringMixin, '_build_local_skill_trust_service'),
        # skill metadata: the plain-field reader the trust source reads slots with
        'metadata': (compact_model_bar, '_plain_fields'),
        # configuration capture: the skill-context projection
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
    # The same skill configuration as the stock path: the built-ins are
    # available and the untrusted managed skill is blocked by the new trust.
    from tldw_chatbook.Skills_Interop.builtin_skills import builtin_skill_records

    assert len(skill_contexts) == 1, skill_contexts
    captured = {
        key: sorted(row.get('name') for row in skill_contexts[0].get(key, ()))
        for key in ('available_skills', 'blocked_skills')
    }
    assert captured == {
        'available_skills': sorted(builtin_skill_records(frozenset())),
        'blocked_skills': ['wrapped-managed'],
    }, captured
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
    "case", ["stock", "trust_source", "trust_builder", "metadata", "skill_capture"]
)
def test_wrapped_skill_collaborator_send_records_the_same_trust(tmp_path, case):
    _run(tmp_path, "wrapped_collaborator", case, script=_SCRIPT, timeout=90)


@pytest.mark.parametrize(
    "case",
    [
        "foreign_trust_factory",
        "foreign_plugin_factory",
        "foreign_loader",
        "instance_catalog_override",
        "instance_trust_fields_override",
        "instance_trust_override",
    ],
)
def test_foreign_skill_callback_is_refused_before_it_runs(tmp_path, case):
    _run(tmp_path, "wrapped_collaborator", case, script=_SCRIPT, timeout=90)


def test_skill_publisher_guard_rechecks_slots_and_environment(monkeypatch):
    """The trust receipt fence still requires unchanged slots and env vars."""
    from tldw_chatbook.Chat import console_configuration_preparation as preparation

    app = SimpleNamespace(
        _local_skill_trust_service_build_lock=object(),
        _local_skills_service=object(),
        _skills_scope_service=object(),
        _local_skills_stack_inputs=(None, None),
        _local_skills_callbacks=(object(), object(), object()),
        app_config={},
    )
    local = SimpleNamespace(
        _trust_service_factory=object(),
        store_dir="store",
        skills_dir="skills",
        policy_enforcer=None,
    )
    source = SimpleNamespace(app=app, local=local)
    guard = preparation._skill_publisher_guard(SimpleNamespace(trust_source=source))
    assert guard is not None
    assert preparation._skill_publisher_guard_current(source, guard)
    for owner, name in (
        (app, "_local_skills_callbacks"),
        (app, "app_config"),
        (local, "_trust_service_factory"),
        (local, "store_dir"),
    ):
        kept = getattr(owner, name)
        setattr(owner, name, object())
        assert not preparation._skill_publisher_guard_current(source, guard), name
        setattr(owner, name, kept)
        assert preparation._skill_publisher_guard_current(source, guard)
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/changed/config.toml")
    assert not preparation._skill_publisher_guard_current(source, guard)


def _first_library_source():
    from tldw_chatbook.Chat import chat_persistence_service as persistence_source
    from tldw_chatbook.Chat import console_chat_store as store_source
    from tldw_chatbook.Chat import console_library_policy_repository as repository_source
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    # Bare instances: only their slots and types are read, nothing is invoked.
    store = object.__new__(store_source.ConsoleChatStore)
    persistence = object.__new__(persistence_source.ChatPersistenceService)
    repository = object.__new__(repository_source.ConsoleLibraryPolicyRepository)
    database = object.__new__(CharactersRAGDB)
    vars(database)["db_path"] = "policy.db"
    vars(store)["persistence"] = persistence
    vars(persistence).update(console_library_policy_repository=repository, db=database)
    vars(repository)["db"] = database
    return (
        store,
        persistence,
        repository,
        database,
        "policy.db",
        store_source,
        repository_source,
        store_source.ConsoleChatStore,
        repository_source._NATIVE_FIRST_LIBRARY_REPOSITORY,
        persistence_source,
        persistence_source.ChatPersistenceService,
    )


@pytest.mark.parametrize(
    ("owner", "name"),
    [(0, "publish_durable_turn_identity"), (0, "_session_or_raise"), (2, "read")],
)
def test_first_library_source_accepts_wrapped_methods_but_not_instance_overrides(
    monkeypatch, owner, name
):
    from tldw_chatbook.Chat import console_configuration_preparation as preparation

    source = _first_library_source()
    assert preparation._first_library_source_current(source)
    cls = type(source[owner])
    original = getattr(cls, name)
    monkeypatch.setattr(cls, name, functools.wraps(original)(lambda *a, **k: original(*a, **k)))
    assert preparation._first_library_source_current(source)
    vars(source[owner])[name] = lambda *a, **k: None
    assert not preparation._first_library_source_current(source)
    del vars(source[owner])[name]
    vars(source[1])["db"] = object()
    assert not preparation._first_library_source_current(source)


def test_first_library_coordinator_decides_from_its_holder_registration(monkeypatch):
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat import console_library_policy_coordinator as module

    record = module._NATIVE_FIRST_LIBRARY_COORDINATOR
    coordinator = object.__new__(module.ConsoleLibraryPolicyCoordinator)
    holder, repository = object(), object()
    registered = SimpleNamespace(holder=holder, conversation_id="conversation")
    session = SimpleNamespace(
        id="session", library_policy_holder=holder, persisted_conversation_id="conversation"
    )
    vars(coordinator).update(repository=repository, _holders={"session": registered})

    def current():
        return preparation._first_library_coordinator_current(
            coordinator, repository, registered, session, record
        )

    assert current()
    original = module.ConsoleLibraryPolicyCoordinator.capture_for_execution
    monkeypatch.setattr(
        module.ConsoleLibraryPolicyCoordinator,
        "capture_for_execution",
        functools.wraps(original)(lambda *a, **k: original(*a, **k)),
    )
    assert current()
    vars(coordinator)["capture_for_execution"] = lambda *a, **k: None
    assert not current()
    del vars(coordinator)["capture_for_execution"]
    session.library_policy_holder = object()
    assert not current()
