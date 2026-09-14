from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.agent_routing import (
    AgentsRoutingConfig, RoutingError, _strict_bool, allowlist_matches,
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

def test_model_only_override_outside_glob_refused_on_parent_provider():
    # qodo PR-2651 High: the final (provider, model) glob check must fire even
    # when the override lands on the parent's own provider -- otherwise a
    # restricted model rides in through the same-provider seam.
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("llama_cpp/qwen3.8-*",))
    try:
        resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
            parent_model="qwen3.8-27b", override_model="llama-3-8b",
            routing=cfg, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted" and e.level == "override"

def test_explicit_same_provider_override_outside_glob_refused():
    # Same hole via override_provider repeating the parent provider verbatim.
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("llama_cpp/qwen3.8-*",))
    try:
        resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
            parent_model="qwen3.8-27b", override_provider="llama_cpp",
            override_model="llama-3-8b", routing=cfg, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted" and e.level == "override"

def test_same_provider_override_inside_glob_allowed():
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("llama_cpp/qwen3.8-*",))
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="llama-3-8b", override_provider="llama_cpp",
        override_model="qwen3.8-27b", routing=cfg, readiness=READY)
    assert (t.provider, t.model, t.source) == (
        "llama_cpp", "qwen3.8-27b", "override")

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

def test_inherit_parent_never_readiness_gated():
    # The parent's own provider readiness is the parent's admission problem:
    # the Console send path already refused an unconfigured provider before
    # any turn ran. An inheriting child must spawn even when the readiness
    # probe would block -- the fleet harness drives AgentService keyless.
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=CFG_OFF, readiness=NOT_READY)
    assert (t.provider, t.model, t.source) == ("moonshot", "kimi-k2", "inherit")

def test_override_back_to_parent_provider_never_readiness_gated():
    # Same principle via the override path: routing that resolves back to the
    # parent's provider adds no new target, so readiness is not consulted.
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("moonshot",))
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", override_model="kimi-k2-thinking",
        routing=cfg, readiness=NOT_READY)
    assert (t.provider, t.model) == ("moonshot", "kimi-k2-thinking")

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

# --- _default_readiness: the production readiness gate (no readiness= fake) ---

def test_default_readiness_custom_ep_keyless_family_resolves_ready():
    # Regression-guards the custom-ep adaptation: a custom-ep id must check
    # its entry's FAMILY readiness (llama_cpp is keyless), not read as an
    # unknown api_settings section.
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", preset=PRESET, routing=CFG_OFF)
    assert (t.provider, t.model) == ("custom-ep:qwen-local", "qwen3.8-27b")
    assert t.base_url == "http://127.0.0.1:8080" and t.source == "preset"

def test_default_readiness_custom_ep_unresolvable_credential_blocks(monkeypatch):
    # Entry declares an env credential that provably cannot be set.
    monkeypatch.delenv("TLDW_TEST_AGENT_ROUTING_ENTRY_KEY", raising=False)
    cfg = {"custom_endpoints": {"locked-box": {
        "display_name": "Locked Box", "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
        "api_key_env": "TLDW_TEST_AGENT_ROUTING_ENTRY_KEY",
    }}}
    preset = AgentDefinition(name="r", instructions="i",
        provider="custom-ep:locked-box", model="qwen3.8-27b")
    try:
        resolve_spawn_target(cfg, parent_provider="m", parent_model="k",
            preset=preset, routing=CFG_OFF)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_ready" and e.level == "preset"

def test_default_readiness_keyed_builtin_without_key_blocks(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    preset = AgentDefinition(name="r", instructions="i",
        provider="openai", model="gpt-4o")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=preset, routing=CFG_OFF)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_ready" and e.level == "preset"

def test_strict_bool_passes_real_booleans_through():
    assert _strict_bool(True, key="k") is True
    assert _strict_bool(False, key="k") is False

def test_strict_bool_parses_recognized_strings_case_insensitively():
    for s in ("true", "1", "yes", "on", "TRUE", " On "):
        assert _strict_bool(s, key="k") is True
    for s in ("false", "0", "no", "off", "", "NO"):
        assert _strict_bool(s, key="k") is False

def test_strict_bool_rejects_garbage_loudly():
    # bool("false") == True is the silent-gate-open bug this prevents.
    for bad in (1, 0, "maybe", None, ["true"]):
        try:
            _strict_bool(bad, key="spawn_override_enabled")
            assert False, bad
        except ValueError as e:
            assert "must be a boolean" in str(e)

def test_load_config_garbage_bool_raises(monkeypatch):
    from tldw_chatbook.Agents import run_log
    monkeypatch.setattr(run_log, "_setting",
        lambda key, default=None:
            "maybe" if key == "spawn_override_enabled" else default)
    try:
        load_agents_routing_config()
        assert False
    except ValueError as e:
        assert "spawn_override_enabled" in str(e)

def test_inherit_to_scripted_parent_provider_skips_validity_gate():
    # Embedded/headless harnesses drive AgentService against provider ids no
    # readiness key knows; inheriting the parent's provider must not be
    # gated on provider validity (same principle as the readiness skip).
    t = resolve_spawn_target(
        {}, parent_provider="recording-provider", parent_model="scripted",
        preset=None, routing=CFG_OFF)
    assert (t.provider, t.model, t.source) == (
        "recording-provider", "scripted", "inherit")

def test_routed_unknown_provider_still_refused():
    preset = AgentDefinition(
        name="ghost", instructions="i", provider="not-a-provider", model="m")
    try:
        resolve_spawn_target(
            {}, parent_provider="moonshot", parent_model="k",
            preset=preset, routing=CFG_OFF)
        assert False
    except RoutingError as e:
        assert e.code == "unknown_provider" and e.level == "preset"

def test_spawn_admission_refusal_type_is_shared_with_service():
    # agent_service once re-declared this class locally, shadowing its own
    # agent_models import: the runtime loop's isinstance() guard then
    # counted admission refusals against the loop-level spawn budget and
    # refused every retry before the chain ledger could pause the chain
    # (regressed six Tests/Agents/test_automatic_child_scope.py cases).
    from tldw_chatbook.Agents import agent_models, agent_service
    assert (agent_service.SpawnAdmissionRefusal
            is agent_models.SpawnAdmissionRefusal)
