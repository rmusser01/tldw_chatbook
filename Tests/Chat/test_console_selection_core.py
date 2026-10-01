"""Direct characterization tests for the shared console selection core.

TASK-32859 review follow-up: the resolver became the single implementation
of the selection algorithm; these pin its contract directly, independent of
the (partially pre-existing-red) console suites.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleSelectionCore,
    build_console_provider_selection_from_settings,
    resolve_console_selection_core,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleWorkspaceContext
from tldw_chatbook.Chat.console_endpoint_provenance import ConsoleEndpointProvenance
from tldw_chatbook.Chat.console_session_endpoint_policy import (
    ConsoleEphemeralEndpointPolicy,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    configured_provider_model,
    console_provider_settings,
)
from tldw_chatbook.config import ProviderSettingsError


def _settings(**overrides):
    values = {"provider": "openai", "model": None, "base_url": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_core_shape_is_frozen() -> None:
    import dataclasses

    core = resolve_console_selection_core(_settings(), app_config={})
    assert isinstance(core, ConsoleSelectionCore)
    with pytest.raises(dataclasses.FrozenInstanceError):
        core.provider = "anthropic"
    assert core.provider == "openai"
    assert core.explicit_model is None
    assert core.configured_model is None
    assert core.base_url is None


def test_provider_identity_falls_back_to_llama_cpp() -> None:
    core = resolve_console_selection_core(_settings(provider=None), app_config={})
    assert core.provider == "llama_cpp"


def test_configured_model_reads_model_api_model_default_chain() -> None:
    config = {"api_settings": {"openai": {"api_model": "gpt-x"}}}
    core = resolve_console_selection_core(_settings(), app_config=config)
    assert core.configured_model == "gpt-x"


def test_explicit_equals_configured_dedup_clears_explicit() -> None:
    config = {"api_settings": {"openai": {"model": "same-model"}}}
    core = resolve_console_selection_core(_settings(model="same-model"), app_config=config)
    assert core.explicit_model is None
    assert core.configured_model == "same-model"


def test_explicit_model_survives_when_it_differs() -> None:
    config = {"api_settings": {"openai": {"model": "configured"}}}
    core = resolve_console_selection_core(_settings(model="explicit"), app_config=config)
    assert core.explicit_model == "explicit"
    assert core.configured_model == "configured"


def test_legacy_model_blocks_the_explicit_dedup() -> None:
    config = {"api_settings": {"openai": {"model": "same-model"}}}
    core = resolve_console_selection_core(
        _settings(model="same-model"), app_config=config, legacy_model="legacy"
    )
    assert core.explicit_model == "same-model"


def test_blankish_models_are_treated_as_unset() -> None:
    config = {"api_settings": {"openai": {"model": "none"}}}
    core = resolve_console_selection_core(_settings(model=" null "), app_config=config)
    assert core.explicit_model is None
    assert core.configured_model is None


# --- TASK-33004.2: one builder of a ConsoleProviderSelection from settings ---


def _build(settings, **kwargs):
    kwargs.setdefault("app_config", {})
    kwargs.setdefault("workspace_context", ConsoleWorkspaceContext())
    return build_console_provider_selection_from_settings(settings, **kwargs)


def test_builder_endpoint_policy_owns_only_its_own_pair() -> None:
    """The screen's endpoint-policy field survives the merge (AC#1)."""
    settings = ConsoleSessionSettings(provider="openai", model="gpt-x")
    owning = ConsoleEphemeralEndpointPolicy("openai", "gpt-x", "http://lan:1")
    other = ConsoleEphemeralEndpointPolicy("openai", "gpt-y", "http://lan:1")

    owned = _build(settings, endpoint_policy=owning)
    assert owned.configured_endpoint_fallback_allowed is False
    assert owned.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION
    for selection in (_build(settings, endpoint_policy=other), _build(settings)):
        assert selection.configured_endpoint_fallback_allowed is True
        assert (
            selection.endpoint_provenance
            is ConsoleEndpointProvenance.DURABLE_CONFIGURATION
        )


def _identity_session(kind, **fields):
    values = {
        "assistant_kind": kind,
        "assistant_name": None,
        "persona_system_template": None,
        "character_name": None,
        "character_system_template": None,
        "user_display_name_override": None,
    }
    values.update(fields)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        (
            _identity_session(
                "persona",
                assistant_name="Ada",
                persona_system_template="You are {{char}}; the user is {{user}}.",
            ),
            "You are Ada; the user is Rob.",
        ),
        (
            _identity_session(
                "character",
                character_name="Kit",
                character_system_template="{{char}} greets {{user}}.",
                user_display_name_override="Sam",
            ),
            "Kit greets Sam.",
        ),
        (_identity_session("character", character_name="Kit"), "settings prompt"),
        (_identity_session("persona", persona_system_template="{{char}}"), "settings prompt"),
        (_identity_session("generic"), "settings prompt"),
        (None, "settings prompt"),
    ],
    ids=["persona", "character", "no-template", "no-name", "generic", "no-session"],
)
def test_builder_re_expands_a_named_identity_template(session, expected) -> None:
    """The screen's identity field survives: a named persona/character with a
    trusted template sends a fresh expansion, anything else keeps the
    settings prompt (task-32484)."""
    selection = _build(
        ConsoleSessionSettings(provider="openai", system_prompt="settings prompt"),
        identity_session=session,
        global_user_name=lambda: "Rob",
    )
    assert selection.system_prompt == expected


def test_builder_keeps_the_sessions_blank_max_tokens() -> None:
    """A blank session Max tokens stays blank: the stored value is the
    effective one (ADR-095 D3 resolves a blank at Apply time), which is what
    the production send path always sent. The controller copy used to refill
    it from the provider default, so background paths built a different
    selection than the send."""
    config = {
        "chat_defaults": {"provider": "openai", "max_tokens": 4096},
        "api_settings": {"openai": {"max_tokens": 4096}},
    }
    selection = _build(
        ConsoleSessionSettings(provider="openai", model="m", max_tokens=None),
        app_config=config,
    )
    assert selection.max_tokens is None


def test_core_reads_a_registry_entry_through_the_unified_lookup() -> None:
    """ADR-146: the core's provider-settings lookup is the registry-aware one."""
    config = {
        "custom_endpoints": {
            "gpu-box": {
                "display_name": "GPU box",
                "family": "llama_cpp",
                "base_url": "http://192.168.1.9:9090",
                "models": ["model-a"],
            }
        }
    }
    core = resolve_console_selection_core(
        _settings(provider="custom-ep:gpu-box"), app_config=config
    )
    assert core.provider == "custom-ep:gpu-box"
    assert core.configured_model == "model-a"


def test_core_finds_an_aliased_provider_table() -> None:
    config = {"api_settings": {"OpenAI": {"model": "gpt-alias"}}}
    core = resolve_console_selection_core(_settings(), app_config=config)
    assert core.configured_model == "gpt-alias"


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"model": "a", "api_model": "b", "default_model": "c"}, "a"),
        ({"api_model": "b", "default_model": "c"}, "b"),
        ({"default_model": "c"}, "c"),
        ({"model": " None ", "api_model": "", "default_model": " c "}, "c"),
        ({}, None),
    ],
)
def test_configured_provider_model_walks_the_one_chain(table, expected) -> None:
    assert configured_provider_model(table) == expected


def test_unified_lookup_raises_only_for_strict_callers() -> None:
    """The gateway's raising policy is kept at its call sites (strict=True);
    every other caller keeps the swallow-to-empty policy."""
    config = {"api_settings": {"qwencloud": "not-a-table"}}
    assert console_provider_settings(config, "qwencloud") == {}
    with pytest.raises(ProviderSettingsError):
        console_provider_settings(config, "qwencloud", strict=True)


def test_controller_selection_carries_the_sessions_endpoint_policy() -> None:
    """The controller's owning-session path hands the one builder the
    session's live endpoint policy, as the screen path does."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    settings = ConsoleSessionSettings(provider="vllm", model="local-model")
    session = store.create_session(title="Chat", settings=settings)
    store.adopt_session_ephemeral_endpoint(
        session.id,
        settings=settings,
        policy=ConsoleEphemeralEndpointPolicy("vllm", "local-model", "http://lan:8000"),
    )
    controller = ConsoleChatController(store=store, provider_gateway=object())

    selection = controller._provider_selection_for_session(session.id)

    assert selection.base_url == "http://lan:8000"
    assert selection.configured_endpoint_fallback_allowed is False
    assert selection.endpoint_provenance is ConsoleEndpointProvenance.EPHEMERAL_SESSION


def _constructs_selection(call, aliases: set[str]) -> bool:
    """A call of the class by name, by an import alias or as ``mod.Class``."""
    import ast

    func = call.func
    if isinstance(func, ast.Name):
        return func.id in aliases
    return isinstance(func, ast.Attribute) and func.attr == "ConsoleProviderSelection"


def _selection_construction_sites(
    sources: dict[str, str] | None = None,
) -> set[tuple[str, str]]:
    import ast

    if sources is None:
        root = Path(__file__).resolve().parents[2]
        sources = {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((root / "tldw_chatbook").rglob("*.py"))
        }
    sites: set[tuple[str, str]] = set()
    for name, source in sources.items():
        if "ConsoleProviderSelection" not in source:
            continue
        tree = ast.parse(source)
        aliases = {"ConsoleProviderSelection"} | {
            alias.asname
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "ConsoleProviderSelection" and alias.asname
        }

        def visit(node, owner):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, child.name)
                    continue
                if isinstance(child, ast.Call) and _constructs_selection(child, aliases):
                    sites.add((name, owner))
                visit(child, owner)

        visit(tree, "<module>")
    return sites


def test_the_site_scan_sees_aliased_and_qualified_constructions() -> None:
    """cubic #2947: an import alias or a module-qualified call is a site too,
    so the guard below fails closed for them."""
    sources = {
        "a.py": "from m import ConsoleProviderSelection as CPS\ndef f():\n    CPS()\n",
        "b.py": "import m\ndef g():\n    m.ConsoleProviderSelection()\n",
        "c.py": "def h():\n    ConsoleProviderSelection()\n",
    }

    assert _selection_construction_sites(sources) == {
        ("a.py", "f"),
        ("b.py", "g"),
        ("c.py", "h"),
    }


def test_only_the_builder_constructs_a_selection_from_session_settings() -> None:
    """AC#1/#2: every construction site is the builder or a listed exception
    that is not built from Console session settings (task notes give each
    reason). A new site fails here until it routes through the builder."""
    assert _selection_construction_sites() == {
        # The one builder from Console session settings.
        ("tldw_chatbook/Chat/console_chat_controller.py", "build_console_provider_selection_from_settings"),
        # Bare controller with no session settings: its own attributes.
        ("tldw_chatbook/Chat/console_chat_controller.py", "_provider_selection"),
        # Detached deep copy of an already-built selection.
        ("tldw_chatbook/Chat/console_turn_context.py", "_detached_selection"),
        # Explicit provider/model for a visual evaluation run.
        ("tldw_chatbook/Chat/console_visual_evaluation.py", "resolve_visual_evaluation_model"),
        # Sub-agent routed to an explicit provider/model/endpoint.
        ("tldw_chatbook/Chat/console_agent_bridge.py", "_resolve_routed_resolution"),
        # Side chat's typed "provider/model" string.
        ("tldw_chatbook/Chat/console_side_chat.py", "_build_selection"),
        # Personas preview: only the empty-provider sentinel for an unset
        # defaults section; a configured section routes through the builder.
        ("tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py", "_selection_from_defaults"),
    }


def test_personas_preview_builds_a_configured_section_through_the_builder() -> None:
    """Review round 1 (AC#1/#2): the Personas preview builds its selection
    from Console session settings, so it goes through the one builder. The
    PR-2668 identity fix reaches it: a hand-spelled registry id resolves to
    the dashed identity the gateway looks up, not the underscore config key
    the hand-written copy sent. An unset section stays an empty provider, so
    the preview falls back to chat_defaults (task-425) instead of trying the
    builder's llama.cpp default; the preview sends its own system prompt."""
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        PersonasPreviewController,
    )

    config = {
        "custom_endpoints": {
            "gpu-box": {
                "display_name": "GPU box",
                "family": "llama_cpp",
                "base_url": "http://192.168.1.9:9090",
                "models": ["model-a", "model-b"],
            }
        },
        "chat_defaults": {"provider": "custom-ep:GPU-Box", "model": "model-b"},
        "character_defaults": {"model": "orphan"},
    }

    chat = PersonasPreviewController._selection_from_defaults(config, "chat_defaults")
    assert chat.provider == "custom-ep:gpu-box"
    assert chat.explicit_model == "model-b"
    assert chat.system_prompt is None

    unset = PersonasPreviewController._selection_from_defaults(
        config, "character_defaults"
    )
    assert unset.provider == ""
    assert unset.explicit_model == "orphan"
