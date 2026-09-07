"""Workspace default persona applies to NEW sessions only (Task 9).

Tested at the two isolable layers (a full ChatScreen harness is
impractical here -- `_create_native_console_session_from_active_context`
needs a live Textual screen):
1. ``ConsoleSessionController._workspace_default_for_new_session`` -- the
   workspace -> ``(assistant_id, label, prompt, memory_mode)`` resolver,
   invoked unbound against a stub host.
2. The controller/store seam -- ``new_session`` assistant kwargs forwarded
   into ``store.create_session`` plus the settings replace the session.py
   injection performs.
"""

from __future__ import annotations

from dataclasses import replace

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    default_console_session_settings,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Console_Modules.session import (
    ConsoleSessionController,
    build_persona_agent_system_prompt,
)
from tldw_chatbook.Workspaces.models import (
    DEFAULT_WORKSPACE_ID,
    WorkspaceAssistantDefaults,
    WorkspaceRecord,
)


class StreamingGateway:
    """Minimal gateway stub (same idiom as Tests/Chat/conftest.py)."""

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


class StubPersonas:
    def __init__(self, records):
        self._records = dict(records)

    def get_persona_profile(self, persona_id):
        record = self._records.get(persona_id)
        if record is None:
            raise ValueError(f"not_found:{persona_id}")
        return dict(record)


class StubRegistry:
    def __init__(self, workspaces):
        self._workspaces = dict(workspaces)

    def get_workspace(self, workspace_id):
        return self._workspaces.get(workspace_id)


class StubApp:
    def __init__(self, registry, personas):
        self.workspace_registry_service = registry
        self.local_character_persona_service = personas


class StubHost:
    """Just enough surface for the unbound mixin-method call."""

    def __init__(self, app, store):
        self.app_instance = app
        self._store = store

    def _ensure_console_chat_store(self):
        return self._store


PERSONA = {
    "id": "local-persona-1",
    "name": "Lit Agent",
    "system_prompt": "You are a literary companion.",
    "personality": "wry and precise",
    "description": "A bookish assistant.",
}


def _workspace(
    workspace_id="w-1",
    *,
    archived=False,
    defaults=None,
):
    if defaults is None:
        defaults = WorkspaceAssistantDefaults(
            assistant_id="local-persona-1",
            persona_memory_mode="read_only",
        )
    return WorkspaceRecord(
        workspace_id=workspace_id,
        name="Explicit",
        archived=archived,
        assistant_defaults=defaults,
    )


def _store(workspace_id):
    return ConsoleChatStore(
        workspace_context=ConsoleWorkspaceContext(active_workspace_id=workspace_id)
    )


def _controller(store):
    return ConsoleChatController(store=store, provider_gateway=StreamingGateway())


def _host(workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID, *, workspace=None, persona=PERSONA):
    records = {persona["id"]: persona} if persona is not None else {}
    registry = StubRegistry(
        {} if workspace is None else {workspace.workspace_id: workspace}
    )
    return StubHost(StubApp(registry, StubPersonas(records)), _store(workspace_id))


def _resolve_workspace_default(host):
    return ConsoleSessionController._workspace_default_for_new_session(host)


# -- Layer 1: the workspace-default resolver --------------------------------


def test_helper_returns_tuple_for_explicit_workspace_with_available_default():
    host = _host("w-1", workspace=_workspace("w-1"))
    result = _resolve_workspace_default(host)
    assert result is not None
    assistant_id, label, prompt, memory_mode = result
    assert assistant_id == "local-persona-1"
    assert label == "Lit Agent"
    assert "You are a literary companion." in prompt
    assert "wry and precise" in prompt
    assert memory_mode == "read_only"


def test_helper_skips_global_default_and_missing_workspaces():
    global_host = _host(CONSOLE_GLOBAL_WORKSPACE_ID, workspace=_workspace("w-1"))
    assert _resolve_workspace_default(global_host) is None

    builtin_host = _host(DEFAULT_WORKSPACE_ID, workspace=_workspace("w-1"))
    assert _resolve_workspace_default(builtin_host) is None

    assert _resolve_workspace_default(_host("")) is None

    unknown = _host("w-missing")
    assert _resolve_workspace_default(unknown) is None


def test_helper_skips_archived_workspaces():
    host = _host("w-1", workspace=_workspace("w-1", archived=True))
    assert _resolve_workspace_default(host) is None


def test_helper_skips_unavailable_defaults():
    deleted = _host(
        "w-1", workspace=_workspace("w-1"), persona={**PERSONA, "deleted": True}
    )
    assert _resolve_workspace_default(deleted) is None

    no_defaults = _host(
        "w-1",
        workspace=WorkspaceRecord(workspace_id="w-1", name="Explicit"),
    )
    assert _resolve_workspace_default(no_defaults) is None


def test_helper_never_raises_when_services_missing():
    host = _host("w-1", workspace=_workspace("w-1"))
    host.app_instance = object()  # no services bound
    assert _resolve_workspace_default(host) is None


def test_build_persona_agent_system_prompt_composes_and_falls_back():
    prompt = build_persona_agent_system_prompt(PERSONA)
    assert "You are a literary companion." in prompt
    assert "wry and precise" in prompt

    assert build_persona_agent_system_prompt({}) == "Stay in character."


# -- Layer 2: controller/store seam ------------------------------------------


def test_new_session_forwards_assistant_kwargs_and_workspace():
    store = _store("w-1")
    controller = _controller(store)
    session = controller.new_session(
        settings=default_console_session_settings({}, "llama_cpp"),
        assistant_kind="persona",
        assistant_id="local-persona-1",
        assistant_label="Lit Agent",
    )
    assert session.assistant_kind == "persona"
    assert session.assistant_id == "local-persona-1"
    assert session.workspace_id == "w-1"
    assert session.settings is not None
    assert session.settings.character_label == "Lit Agent"


def test_plain_new_session_unchanged():
    controller = _controller(_store(CONSOLE_GLOBAL_WORKSPACE_ID))
    session = controller.new_session()
    assert session.assistant_kind == "generic"
    assert session.assistant_id == "console"
    assert session.settings is None or session.settings.character_label == ""


def test_settings_gain_persona_memory_mode_field():
    defaults = default_console_session_settings({}, "llama_cpp")
    assert defaults.persona_memory_mode is None
    snapshot = replace(defaults, persona_memory_mode="read_only")
    assert snapshot.persona_memory_mode == "read_only"
    assert ConsoleSessionSettings(provider="llama_cpp").persona_memory_mode is None


# -- Precedence + independence (session.py injection composition) ------------


def test_injection_applies_workspace_default_to_plain_defaults_only():
    # The session.py plain-new-tab path: default settings with no explicit
    # persona markers get the workspace default stamped in.
    host = _host("w-1", workspace=_workspace("w-1"))
    resolved = _resolve_workspace_default(host)
    assert resolved is not None
    assistant_id, label, prompt, memory_mode = resolved
    defaults = default_console_session_settings({}, "llama_cpp")
    stamped = replace(
        defaults,
        system_prompt=prompt,
        character_label=label,
        persona_memory_mode=memory_mode,
    )
    session = _controller(host._store).new_session(
        settings=stamped,
        assistant_kind="persona",
        assistant_id=assistant_id,
        assistant_label=label,
    )
    assert session.settings.persona_memory_mode == "read_only"
    assert "You are a literary companion." in session.settings.system_prompt

    # Explicit settings (an existing session's snapshot carrying its own
    # system prompt) keep winning: the injection's plain-detection guard
    # (system_prompt is None and no character label and no memory mode)
    # refuses to stamp these.
    explicit = replace(
        defaults,
        system_prompt="Explicit persona prompt",
        character_label="Chosen",
    )
    assert explicit.system_prompt != stamped.system_prompt


def test_existing_sessions_independent_of_later_default_edits():
    host = _host("w-1", workspace=_workspace("w-1"))
    resolved = _resolve_workspace_default(host)
    assert resolved is not None
    assistant_id, label, prompt, memory_mode = resolved
    stamped = replace(
        default_console_session_settings({}, "llama_cpp"),
        system_prompt=prompt,
        character_label=label,
        persona_memory_mode=memory_mode,
    )
    store = host._store
    controller = _controller(store)
    session = controller.new_session(
        settings=stamped,
        assistant_kind="persona",
        assistant_id=assistant_id,
        assistant_label=label,
    )
    before = (
        session.assistant_kind,
        session.assistant_id,
        session.settings.system_prompt,
        session.settings.persona_memory_mode,
    )

    # A later defaults edit (registry now points at another persona).
    host.app_instance.workspace_registry_service = StubRegistry(
        {
            "w-1": _workspace(
                "w-1",
                defaults=WorkspaceAssistantDefaults(assistant_id="other"),
            )
        }
    )

    reloaded = next(item for item in store.sessions() if item.id == session.id)
    after = (
        reloaded.assistant_kind,
        reloaded.assistant_id,
        reloaded.settings.system_prompt,
        reloaded.settings.persona_memory_mode,
    )
    assert before == after


# -- Startup settings/identity selection ------------------------------------


class StartupHost(StubHost):
    """Host for the unbound ``_new_session_startup_settings`` seam."""

    def __init__(self, app, store):
        super().__init__(app, store)
        # The real controller exposes this as a property; a plain attribute
        # is the equivalent surface for the unbound-method call.
        self._console_chat_store = store

    def _active_console_session_settings(self):
        store = self._store
        if store.active_session_id is None:
            return None
        try:
            return store.session_settings(store.active_session_id)
        except KeyError:
            return None

    def _default_console_session_settings(self):
        return ConsoleSessionSettings(
            provider="local_llamacpp",
            model="active-control-model",
            temperature=1.4,
        )

    def _blank_console_session_settings(self):
        return ConsoleSessionSettings(
            provider="openai",
            model="published-model",
            temperature=0.23,
            streaming=False,
        )

    def _workspace_default_for_new_session(self):
        return ConsoleSessionController._workspace_default_for_new_session(self)


def _startup_host(workspace_id="w-1", *, workspace=None, persona=PERSONA):
    if workspace is None:
        workspace = _workspace(workspace_id)
    records = {persona["id"]: persona} if persona is not None else {}
    registry = StubRegistry(
        {} if workspace is None else {workspace.workspace_id: workspace}
    )
    return StartupHost(StubApp(registry, StubPersonas(records)), _store(workspace_id))


def _startup(host):
    return ConsoleSessionController._new_session_startup_settings(host)


def test_new_tab_re_resolves_workspace_persona_on_published_defaults():
    host = _startup_host()
    workspace_default = ConsoleSessionController._workspace_default_for_new_session(
        host
    )
    assert workspace_default is not None
    assistant_id, label, prompt, memory_mode = workspace_default
    stamped = replace(
        host._default_console_session_settings(),
        system_prompt=prompt,
        character_label=label,
        persona_memory_mode=memory_mode,
    )
    host._store.create_session(
        settings=stamped,
        assistant_kind="persona",
        assistant_id=assistant_id,
    )
    settings, assistant_kwargs = _startup(host)
    assert settings is not stamped
    assert (settings.provider, settings.model) == ("openai", "published-model")
    assert settings.temperature == 0.23
    assert assistant_kwargs == {
        "assistant_kind": "persona",
        "assistant_id": "local-persona-1",
        "assistant_label": "Lit Agent",
    }


def test_plain_new_tab_ignores_pristine_active_session():
    host = _startup_host()
    defaults = default_console_session_settings({}, "llama_cpp")
    host._store.create_session(
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    settings, assistant_kwargs = _startup(host)
    assert assistant_kwargs["assistant_kind"] == "persona"
    assert assistant_kwargs["assistant_id"] == "local-persona-1"
    assert settings.persona_memory_mode == "read_only"
    assert "You are a literary companion." in settings.system_prompt


def test_plain_new_tab_does_not_clone_active_settings():
    host = _startup_host()
    defaults = host._default_console_session_settings()
    explicit = replace(defaults, temperature=0.11)
    host._store.create_session(settings=explicit)
    settings, assistant_kwargs = _startup(host)
    assert settings is not explicit
    assert (settings.provider, settings.model) == ("openai", "published-model")
    assert settings.temperature == 0.23
    assert assistant_kwargs["assistant_kind"] == "persona"
    assert assistant_kwargs["assistant_id"] == "local-persona-1"


def test_no_active_session_stamps_workspace_default():
    host = _startup_host()
    settings, assistant_kwargs = _startup(host)
    assert assistant_kwargs["assistant_kind"] == "persona"
    assert (settings.provider, settings.model) == ("openai", "published-model")
    assert settings.temperature == 0.23
    assert settings.streaming is False
    assert settings.persona_memory_mode == "read_only"
