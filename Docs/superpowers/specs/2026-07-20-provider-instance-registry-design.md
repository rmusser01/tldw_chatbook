# Provider Instance Registry — Design Spec

**Date:** 2026-07-20  
**Status:** Approved  
**Scope:** TUI API provider configuration and selection in `tldw_chatbook`.

## Goal

Enable users to configure multiple named API instances (endpoint + keys) for a single provider type, and improve the ergonomics of swapping between them during chat.

## User Requests Addressed

1. Support 9 local API instances (llama.cpp, vLLM, Triton) with quick swapping.
2. Support 4 API keys per provider for testing across accounts.
3. Allow switching keys mid-chat without losing conversation context.
4. Provide a "duplicate current" workflow for quickly creating variants.

## Approach

Introduce a first-class `ProviderInstance` concept. Each instance is a named API endpoint with its own credentials and model defaults. Instances are grouped by `provider_type`, which determines the request/response adapter. The configuration migrates from flat `[api_settings.<provider>]` entries to `[provider_instances.<id>]` entries.

## Architecture

### Module Layout

```
Providers/
├── __init__.py
├── instances.py          # ProviderInstance dataclass + registry
├── adapters/
│   ├── __init__.py
│   ├── base.py           # ProviderAdapter protocol
│   ├── openai_compat.py  # Generic OpenAI-compatible adapter
│   ├── llamacpp.py       # llama.cpp quirks
│   ├── vllm.py           # vLLM quirks
│   └── triton.py         # Triton quirks
├── resolver.py           # Instance + key → call options
└── readiness.py          # On-demand readiness checks with caching
```

### Data Model

```python
@dataclass(frozen=True)
class ApiKey:
    label: str                   # "production", "staging", "test"
    value: str                   # The actual key
    is_default: bool = False

@dataclass(frozen=True)
class ProviderInstance:
    id: str                      # e.g., "vllm-1", "llamacpp-prod"
    provider_type: str           # "vllm", "llamacpp", "triton", "custom"
    name: str                    # User label, e.g., "vLLM Production"
    endpoint: str                # "http://localhost:8000/v1"
    api_keys: tuple[ApiKey, ...]
    model_defaults: dict[str, Any]
    extra_options: dict[str, Any]
```

### Config Schema

```toml
[provider_instances.vllm-1]
provider_type = "vllm"
name = "vLLM Production"
endpoint = "http://localhost:8000/v1"
model = "llama-3.1-70b"
temperature = 0.7
max_tokens = 4096

[provider_instances.vllm-1.keys.production]
value = "<API_KEY_HERE>"
default = true

[provider_instances.vllm-1.keys.staging]
value = "<API_KEY_HERE>"
```

## Components

| Component | Change |
|-----------|--------|
| `Providers/instances.py` | New. `ProviderInstance`, `ApiKey`, `InstanceRegistry`. |
| `Providers/adapters/` | New. Adapter protocol + implementations for OpenAI-compatible, llama.cpp, vLLM, Triton. |
| `Providers/resolver.py` | New. Resolve instance + key label to call options. |
| `Providers/readiness.py` | New. On-demand readiness checks with TTL caching. |
| `config.py` | Add `[provider_instances]` section; auto-migrate legacy `[api_settings]` on first run. |
| `UI/Screens/settings_screen.py` | Add "Provider Instances" CRUD section. |
| `UI/Chat_Window_Enhanced.py` | Two-level provider/instance/key selector; "Duplicate current" button. |

## Chat UI

- **Instance dropdown:** Primary selector, shows instance names grouped by provider type. Searchable. "Add new instance" at bottom.
- **Key dropdown:** Secondary, only visible when instance has >1 key. Shows labels with default marker.
- **Duplicate current:** Creates a new instance sharing the same endpoint, focuses settings on the new instance's keys.
- **Keyboard shortcut:** `Ctrl+P` opens instance selector directly.

## Settings UI

- **List view:** All instances with provider type, name, endpoint, key count.
- **Add instance:** Guided form (type → name → endpoint → first key → options).
- **Edit instance:** Inline editing for name, endpoint, model defaults.
- **Keys section:** Per-instance key management (add/remove/set default).
- **Readiness test:** Button to test connectivity per instance.
- **Duplicate:** Two modes — reference (share endpoint) or copy (full copy).

## Error Handling

- **Duplicate IDs:** Warn and reject on save; auto-rename only on migration.
- **Key sanitization:** `sanitize_error_for_display()` strips keys from provider errors.
- **Mid-chat failure:** Fail message with "Retry" and "Switch instance" options.
- **Provider quirks:** Warn on fallback, no silent generic fallback.
- **Model fallback:** Show system message when falling back to default model.
- **Key switching:** One-time notice; offer "Fork conversation" option.
- **Duplicate modes:** Reference (share endpoint) or copy (full copy).
- **Readiness caching:** 5 min for success, 30 s for failure; manual bypass.
- **Key storage:** Plaintext default; optional keyring with `[use_keyring = true]`.

## Migration

- On first run, legacy `[api_settings.<provider>]` entries migrate to `[provider_instances.<provider>-1]`, `<provider>-2`, etc.
- Legacy config is preserved as fallback for 2 releases.
- Corrupt or duplicate configs are skipped with warnings, never crash.

## Testing Strategy

- **Unit tests:** Dataclasses, registry, migration, resolver, readiness, sanitization.
- **Integration tests:** Chat switching, settings CRUD, duplicate current, config watching, cross-adapter calls.
- **Property tests (Hypothesis):** ID generation, key sanitization, migration validity, resolver validity, readiness TTL.
- **Performance tests:** 36 instances dropdown (<100ms), 100 instances stress test, parallel readiness checks (<2s).
- **Manual QA:** 9 real instances with 4 keys each, rapid switching, duplicate-and-edit workflow.

## Acceptance Criteria

- [ ] Users can add/edit/delete named provider instances in Settings.
- [ ] Each instance supports multiple named API keys with a default.
- [ ] Chat UI shows instances grouped by provider type.
- [ ] Users can switch instances and keys mid-chat without losing conversation.
- [ ] "Duplicate current" creates a variant sharing the endpoint.
- [ ] Legacy `[api_settings]` configs migrate automatically on first run.
- [ ] Readiness checks are on-demand and cached.
- [ ] All new logic has unit, integration, and property tests.

## Related Files

- `tldw_chatbook/config.py`
- `tldw_chatbook/Providers/` (new)
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- `tldw_chatbook/Chat/provider_readiness.py`
- `tldw_chatbook/Chat/provider_catalog.py`
