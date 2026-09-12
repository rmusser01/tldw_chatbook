from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.agent_routing import (
    AgentsRoutingConfig, RoutingError, allowlist_matches,
    load_agents_routing_config, resolve_child_params, resolve_spawn_target,
)

READY = lambda cfg, provider: None                      # readiness fake: ready
NOT_READY = lambda cfg, provider: "no API key"          # readiness fake: blocked
CFG_OFF = AgentsRoutingConfig()                         # everything default
CFG_ON = AgentsRoutingConfig(
    spawn_override_enabled=True,
    spawn_override_allowlist=("llama_cpp", "custom-ep:qwen-local/qwen3.8-*"),
)
PRESET = AgentDefinition(
    name="implementer", instructions="Implement.",
    provider="custom-ep:qwen-local", model="qwen3.8-27b",
    params=(("temperature", 0.2),),
)
APP_CFG = {
    "custom_endpoints": {"qwen-local": {
        "display_name": "Qwen Local", "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
        "params": {"top_k": 40},
    }},
    "chat_defaults": {"temperature": 0.9},
    "api_settings": {"llama_cpp": {"model": "llama-3-8b"}},
}

def test_inherit_parent_when_nothing_configured():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=CFG_OFF, readiness=READY)
    assert (t.provider, t.model, t.source) == ("moonshot", "kimi-k2", "inherit")
    assert t.base_url is None

def test_preset_routes_across_providers():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", preset=PRESET, routing=CFG_OFF, readiness=READY)
    assert t.provider == "custom-ep:qwen-local" and t.model == "qwen3.8-27b"
    assert t.base_url == "http://127.0.0.1:8080" and t.source == "preset"

def test_preset_model_only_keeps_parent_endpoint():
    legacy = AgentDefinition(name="r", instructions="i", model="qwen3.8-27b")
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="llama-3-8b", preset=legacy, routing=CFG_OFF, readiness=READY)
    assert (t.provider, t.model) == ("llama_cpp", "qwen3.8-27b")

def test_subagent_default_used_when_no_preset_or_override():
    cfg = AgentsRoutingConfig(subagent_default_provider="llama_cpp",
                              subagent_default_model="qwen3.8-27b")
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=cfg, readiness=READY)
    assert (t.provider, t.model, t.source) == ("llama_cpp", "qwen3.8-27b", "default")

def test_override_refused_when_disabled():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            override_provider="llama_cpp", routing=CFG_OFF, readiness=READY)
        assert False, "expected RoutingError"
    except RoutingError as e:
        assert e.code == "override_disabled" and e.level == "override"

def test_override_provider_not_allowlisted():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            override_provider="openai", routing=CFG_ON, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted"

def test_final_provider_guard_refuses_model_only_onto_paid_default():
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("llama_cpp",),
        subagent_default_provider="moonshot", subagent_default_model="kimi-k2")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="anthropic",
            parent_model="claude", override_model="kimi-k2-thinking",
            routing=cfg, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted"

def test_final_provider_guard_allows_model_only_on_parent_provider():
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="llama-3-8b", override_model="qwen3.8-27b",
        routing=CFG_ON, readiness=READY)
    assert (t.provider, t.model) == ("llama_cpp", "qwen3.8-27b")

def test_allowlist_glob_matching():
    al = ("llama_cpp/qwen3.8-*", "custom-ep:box")
    assert allowlist_matches(al, "llama_cpp", "Qwen3.8-27B")      # case-insensitive
    assert not allowlist_matches(al, "llama_cpp", "llama-3-8b")
    assert allowlist_matches(al, "custom-ep:box", "anything")
    assert not allowlist_matches(al, "custom-ep:other", "anything")

def test_unknown_endpoint_slug():
    bad = AgentDefinition(name="r", instructions="i", provider="custom-ep:gone")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=bad, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "unknown_endpoint_slug" and e.level == "preset"

def test_unknown_provider():
    bad = AgentDefinition(name="r", instructions="i", provider="not-a-provider")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=bad, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "unknown_provider"

def test_provider_not_ready():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=PRESET, routing=CFG_OFF, readiness=NOT_READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_ready" and "no API key" in str(e)

def test_no_model_resolved_only_when_routed_and_unconfigured():
    preset = AgentDefinition(name="r", instructions="i", provider="custom-ep:qwen-local")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=preset, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "no_model_resolved"
    # inherit path with blank parent model never raises:
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="", routing=CFG_OFF, readiness=READY)
    assert t.model == ""

def test_builtin_provider_configured_model_fills_blank():
    preset = AgentDefinition(name="r", instructions="i", provider="llama_cpp")
    t = resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
        preset=preset, routing=CFG_OFF, readiness=READY)
    assert t.model == "llama-3-8b"   # from api_settings.llama_cpp.model

def test_params_precedence_preset_over_entry_over_chat_defaults():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", preset=PRESET, routing=CFG_OFF, readiness=READY)
    merged = dict(t.params)
    assert merged["temperature"] == 0.2    # preset beats chat_defaults 0.9
    assert merged["top_k"] == 40           # entry params present
    assert "streaming" not in merged

def test_params_never_come_from_parent():
    # parent params simply are not an input; resolved params come from the
    # child's own provider stack.
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=CFG_OFF, readiness=READY)
    assert dict(t.params)["temperature"] == 0.9   # chat_defaults, not a parent value
