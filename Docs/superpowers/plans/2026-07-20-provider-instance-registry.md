# Provider Instance Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to configure multiple named API instances (endpoint + keys) for local/custom providers and swap between them during chat.

**Architecture:** New `Providers/` package with `ProviderInstance`/`ApiKey` dataclasses, adapter stack wrapping `LLM_Calls/`, TTL-cached readiness, and a two-level chat UI selector. Config migrates from `[api_settings]` to `[provider_instances]` for local/custom providers.

**Tech Stack:** Python 3.11+, Textual, Pydantic, SQLite, tomlkit, Hypothesis

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Providers/__init__.py` | Package exports |
| `Providers/instances.py` | `ProviderInstance`, `ApiKey`, `InstanceRegistry` |
| `Providers/adapters/__init__.py` | Adapter registry |
| `Providers/adapters/base.py` | `ProviderAdapter` protocol |
| `Providers/adapters/openai_compat.py` | Generic OpenAI-compatible adapter wrapping `LLM_API_Calls_Local` |
| `Providers/adapters/llama_cpp.py` | llama.cpp quirks |
| `Providers/adapters/vllm.py` | vLLM quirks |
| `Providers/resolver.py` | Instance + key → call options |
| `Providers/readiness.py` | TTL-cached readiness checks wrapping `settings_endpoint_probe` |
| `Providers/sanitization.py` | Key sanitization for error display |
| `Providers/bootstrap.py` | Registry construction from config, migration, attachment to app |
| `config.py` | `[provider_instances]` section, migration, config watching |
| `UI/Screens/settings_screen.py` | Provider Instances CRUD UI |
| `UI/Chat_Window_Enhanced.py` | Two-level instance/key selector |
| Tests | Unit, integration, property tests |

---

## Task Dependencies

```
Task 1 (Scaffold)
  └── Task 2 (Instances dataclasses)
        └── Task 3 (Adapters wrapping LLM_Calls)
              └── Task 4 (Resolver)
                    └── Task 5 (Readiness)
                          └── Task 6 (Sanitization)
                                └── Task 7 (Config + Migration)
                                      └── Task 8 (Bootstrap: registry → app)
                                            └── Task 9 (Settings UI CRUD)
                                                  └── Task 10 (Chat UI selector)
                                                        └── Task 11 (Config watching)
                                                              └── Task 12 (Integration tests)
                                                                    └── Task 13 (Property tests)
                                                                          └── Task 14 (Final QA)
```

---

### Task 1: Scaffold feature branch and verify baseline

**Files:**
- Branch: `feature/provider-instance-registry`

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b feature/provider-instance-registry
```

- [ ] **Step 2: Run the relevant existing tests to establish a green baseline**

```bash
pytest Tests/Chat/ Tests/UI/ -q
```

Expected: all existing tests pass.

- [ ] **Step 3: Commit the baseline check**

```bash
git commit --allow-empty -m "chore: start provider instance registry"
```

---

### Task 2: Add `ProviderInstance` and `ApiKey` dataclasses

**Files:**
- Create: `tldw_chatbook/Providers/__init__.py`
- Create: `tldw_chatbook/Providers/instances.py`
- Test: `Tests/Providers/__init__.py`
- Test: `Tests/Providers/test_instances.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Providers.instances import ApiKey, ProviderInstance


def test_api_key_creation():
    key = ApiKey(label="production", value="sk-123", is_default=True)
    assert key.label == "production"
    assert key.value == "sk-123"
    assert key.is_default is True


def test_provider_instance_creation():
    keys = (ApiKey(label="prod", value="sk-1", is_default=True),)
    instance = ProviderInstance(
        id="vllm-1",
        provider_type="vllm",
        name="vLLM Production",
        endpoint="http://localhost:8000/v1",
        api_keys=keys,
        model_defaults={"model": "llama-3.1-70b"},
        extra_options={},
    )
    assert instance.id == "vllm-1"
    assert instance.provider_type == "vllm"
    assert len(instance.api_keys) == 1
```

Run: `pytest Tests/Providers/test_instances.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement the dataclasses**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApiKey:
    label: str
    value: str
    is_default: bool = False


@dataclass(frozen=True)
class ProviderInstance:
    id: str
    provider_type: str
    name: str
    endpoint: str
    api_keys: tuple[ApiKey, ...]
    model_defaults: dict[str, Any] = field(default_factory=dict)
    extra_options: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 3: Run the test**

Run: `pytest Tests/Providers/test_instances.py -v`
Expected: PASS.

- [ ] **Step 4: Add InstanceRegistry**

```python
class InstanceRegistry:
    def __init__(self):
        self._instances: dict[str, ProviderInstance] = {}

    def add(self, instance: ProviderInstance) -> None:
        self._instances[instance.id] = instance

    def get(self, instance_id: str) -> ProviderInstance | None:
        return self._instances.get(instance_id)

    def list(self, provider_type: str | None = None) -> list[ProviderInstance]:
        if provider_type is None:
            return list(self._instances.values())
        return [i for i in self._instances.values() if i.provider_type == provider_type]

    def remove(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)

    def list_by_type(self) -> dict[str, list[ProviderInstance]]:
        result: dict[str, list[ProviderInstance]] = {}
        for instance in self._instances.values():
            result.setdefault(instance.provider_type, []).append(instance)
        return result
```

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Providers/ Tests/Providers/
git commit -m "feat(providers): add ProviderInstance and ApiKey dataclasses"
```

---

### Task 3: Add provider adapters wrapping LLM_Calls

**Files:**
- Create: `tldw_chatbook/Providers/adapters/__init__.py`
- Create: `tldw_chatbook/Providers/adapters/base.py`
- Create: `tldw_chatbook/Providers/adapters/openai_compat.py`
- Create: `tldw_chatbook/Providers/adapters/llama_cpp.py`
- Create: `tldw_chatbook/Providers/adapters/vllm.py`
- Test: `Tests/Providers/test_adapters.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch

from tldw_chatbook.Providers.adapters.openai_compat import OpenAICompatAdapter


def test_openai_compat_adapter_delegates_to_llm_calls():
    adapter = OpenAICompatAdapter()
    # Patch the adapter's local reference, not the original module
    with patch("tldw_chatbook.Providers.adapters.openai_compat._chat_with_openai_compatible_local_server") as mock:
        adapter.chat(endpoint="http://localhost:8000/v1", api_key="sk-1", messages=[])
        mock.assert_called_once()
```

Run: `pytest Tests/Providers/test_adapters.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement base protocol**

```python
from __future__ import annotations

from typing import Any, Protocol


class ProviderAdapter(Protocol):
    provider_type: str

    def build_options(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def chat(self, *, endpoint: str, api_key: str, messages: list[dict], **kwargs) -> Any:
        ...
```

- [ ] **Step 3: Implement OpenAICompatAdapter**

```python
from __future__ import annotations

from typing import Any

from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import _chat_with_openai_compatible_local_server

from .base import ProviderAdapter


class OpenAICompatAdapter(ProviderAdapter):
    provider_type = "custom"

    def build_options(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "endpoint": endpoint,
            "api_key": api_key,
        }
        if model:
            options["model"] = model
        if extra_options:
            options.update(extra_options)
        return options

    def chat(self, *, endpoint: str, api_key: str, messages: list[dict], **kwargs) -> Any:
        # Call the internal helper directly with per-instance endpoint/key.
        # The public wrappers (chat_with_custom_openai, etc.) read endpoint from
        # global settings; we bypass them to inject per-instance values.
        return _chat_with_openai_compatible_local_server(
            api_base_url=endpoint,
            model_name=kwargs.get("model"),
            input_data=messages,
            api_key=api_key,
            temp=kwargs.get("temperature"),
            streaming=kwargs.get("streaming", False),
            max_tokens=kwargs.get("max_tokens"),
            provider_name=self.provider_type,
        )
```

- [ ] **Step 4: Implement llama_cpp and vllm adapters**

```python
class LlamaCppAdapter(OpenAICompatAdapter):
    provider_type = "llama_cpp"

    def chat(self, *, endpoint: str, api_key: str, messages: list[dict], **kwargs) -> Any:
        extra = kwargs.get("extra_options") or {}
        result = _chat_with_openai_compatible_local_server(
            api_base_url=endpoint,
            model_name=kwargs.get("model"),
            input_data=messages,
            api_key=api_key,
            temp=kwargs.get("temperature"),
            streaming=kwargs.get("streaming", False),
            max_tokens=kwargs.get("max_tokens"),
            provider_name=self.provider_type,
        )
        # llama.cpp-specific post-processing could go here
        return result


class VLLMAdapter(OpenAICompatAdapter):
    provider_type = "vllm"

    def chat(self, *, endpoint: str, api_key: str, messages: list[dict], **kwargs) -> Any:
        extra = kwargs.get("extra_options") or {}
        return _chat_with_openai_compatible_local_server(
            api_base_url=endpoint,
            model_name=kwargs.get("model"),
            input_data=messages,
            api_key=api_key,
            temp=kwargs.get("temperature"),
            streaming=kwargs.get("streaming", False),
            max_tokens=kwargs.get("max_tokens"),
            provider_name=self.provider_type,
        )
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest Tests/Providers/test_adapters.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Providers/adapters/ Tests/Providers/test_adapters.py
git commit -m "feat(providers): add provider adapters wrapping LLM_Calls"
```

---

### Task 4: Add resolver

**Files:**
- Create: `tldw_chatbook/Providers/resolver.py`
- Test: `Tests/Providers/test_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Providers.instances import ApiKey, ProviderInstance
from tldw_chatbook.Providers.resolver import resolve_instance_options


def test_resolve_instance_options():
    instance = ProviderInstance(
        id="vllm-1",
        provider_type="vllm",
        name="vLLM Prod",
        endpoint="http://localhost:8000/v1",
        api_keys=(
            ApiKey(label="prod", value="sk-1", is_default=True),
            ApiKey(label="staging", value="sk-2"),
        ),
        model_defaults={"model": "llama-3.1-70b"},
        extra_options={"gpu_memory_utilization": 0.9},
    )
    options = resolve_instance_options(instance, key_label="staging")
    assert options["endpoint"] == "http://localhost:8000/v1"
    assert options["api_key"] == "sk-2"
    assert options["model"] == "llama-3.1-70b"
    assert options["gpu_memory_utilization"] == 0.9
```

Run: `pytest Tests/Providers/test_resolver.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement resolver**

```python
from __future__ import annotations

import os
from typing import Any

from .instances import ProviderInstance


def _resolve_key_value(value: str) -> str:
    if value.startswith("env:"):
        env_var = value[4:]
        result = os.environ.get(env_var, "")
        if not result:
            import logging
            logging.warning(f"Environment variable {env_var} not set for API key")
        return result
    return value


def resolve_instance_options(
    instance: ProviderInstance,
    *,
    key_label: str | None = None,
) -> dict[str, Any]:
    key = None
    if key_label:
        key = next((k for k in instance.api_keys if k.label == key_label), None)
    if key is None:
        key = next((k for k in instance.api_keys if k.is_default), instance.api_keys[0] if instance.api_keys else None)
    if key is None:
        raise ValueError(f"Instance {instance.id} has no API keys")

    options: dict[str, Any] = {
        "endpoint": instance.endpoint,
        "api_key": _resolve_key_value(key.value),
    }
    options.update(instance.model_defaults)
    options.update(instance.extra_options)
    return options
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest Tests/Providers/test_resolver.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Providers/resolver.py Tests/Providers/test_resolver.py
git commit -m "feat(providers): add instance resolver"
```

---

### Task 5: Add readiness checks wrapping settings_endpoint_probe

**Files:**
- Create: `tldw_chatbook/Providers/readiness.py`
- Test: `Tests/Providers/test_readiness.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Providers.readiness import ReadinessCache


def test_readiness_cache():
    cache = ReadinessCache(success_ttl=300, failure_ttl=10)
    cache.record_success("vllm-1")
    assert cache.is_ready("vllm-1") is True
    cache.record_failure("vllm-1")
    assert cache.is_ready("vllm-1") is False
```

Run: `pytest Tests/Providers/test_readiness.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement readiness cache with probe integration**

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.UI.Screens.settings_endpoint_probe import probe_settings_endpoint


@dataclass
class ReadinessCache:
    success_ttl: int = 300  # 5 minutes
    failure_ttl: int = 10   # 10 seconds

    def __post_init__(self):
        self._cache: dict[str, tuple[bool, float]] = {}

    def record_success(self, instance_id: str) -> None:
        self._cache[instance_id] = (True, time.monotonic())

    def record_failure(self, instance_id: str) -> None:
        self._cache[instance_id] = (False, time.monotonic())

    def is_ready(self, instance_id: str) -> bool | None:
        if instance_id not in self._cache:
            return None
        ready, timestamp = self._cache[instance_id]
        ttl = self.success_ttl if ready else self.failure_ttl
        if time.monotonic() - timestamp > ttl:
            del self._cache[instance_id]
            return None
        return ready

    async def check_instance(self, instance_id: str, endpoint: str) -> bool:
        """Check readiness using the existing endpoint probe."""
        cached = self.is_ready(instance_id)
        if cached is not None:
            return cached
        try:
            result = await probe_settings_endpoint(endpoint)
            if result.reachable:
                self.record_success(instance_id)
                return True
            self.record_failure(instance_id)
            return False
        except Exception:
            self.record_failure(instance_id)
            return False
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest Tests/Providers/test_readiness.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Providers/readiness.py Tests/Providers/test_readiness.py
git commit -m "feat(providers): add readiness cache with probe integration"
```

---

### Task 6: Add key sanitization

**Files:**
- Create: `tldw_chatbook/Providers/sanitization.py`
- Test: `Tests/Providers/test_sanitization.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Providers.sanitization import sanitize_error_for_display


def test_sanitize_error_for_display():
    error = "Invalid API key sk-abc123def456ghi789jkl012 provided"
    result = sanitize_error_for_display(error)
    assert "sk-abc123def456ghi789jkl012" not in result
    assert "[key]" in result


def test_sanitize_short_key_passes_through():
    error = "Invalid key sk-abc"
    result = sanitize_error_for_display(error)
    assert result == error  # Short keys are not sanitized (not real keys)
```

Run: `pytest Tests/Providers/test_sanitization.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement sanitization**

```python
from __future__ import annotations

import re


_KEY_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer [a-zA-Z0-9]{20,}"),
    re.compile(r"[a-fA-F0-9]{32,}"),
]


def sanitize_error_for_display(error: str) -> str:
    result = error
    for pattern in _KEY_PATTERNS:
        result = pattern.sub("[key]", result)
    return result
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest Tests/Providers/test_sanitization.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Providers/sanitization.py Tests/Providers/test_sanitization.py
git commit -m "feat(providers): add key sanitization"
```

---

### Task 7: Add config section and migration with tomlkit

**Files:**
- Modify: `pyproject.toml`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/test_config_provider_instances.py`

- [ ] **Step 1: Add tomlkit dependency**

Add to `pyproject.toml`:
```toml
tomlkit = ">=0.12.0"
```

- [ ] **Step 2: Write the failing test**

```python
from tldw_chatbook.config import load_settings


def test_provider_instances_section():
    settings = load_settings()
    assert "provider_instances" in settings


def test_migrate_legacy_providers(tmp_path):
    """Legacy [api_settings.vllm] migrates to [provider_instances.vllm-1]."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
[api_settings.vllm]
api_ip = "http://localhost:8000/v1"
api_key = "sk-test"
model = "llama-3.1-70b"
temperature = 0.7
max_tokens = 4096

[api_settings.openai]
api_key = "sk-openai"
""")

    from tldw_chatbook.config import migrate_legacy_providers
    migrated = migrate_legacy_providers(config_path)
    assert migrated is True

    import tomlkit
    with open(config_path) as f:
        doc = tomlkit.parse(f.read())

    assert "provider_instances" in doc
    assert "vllm-1" in doc["provider_instances"]
    assert doc["provider_instances"]["vllm-1"]["endpoint"] == "http://localhost:8000/v1"
    assert "vllm" not in doc.get("api_settings", {})
    assert "openai" in doc["api_settings"]  # Cloud providers preserved
```

Run: `pytest Tests/test_config_provider_instances.py -v`
Expected: FAIL.

- [ ] **Step 3: Add provider_instances section to config**

In `load_settings()`:
```python
provider_instances = get_toml_section("provider_instances")
```

Add to returned dict:
```python
"provider_instances": provider_instances,
```

- [ ] **Step 4: Implement migration with tomlkit**

```python
from tomlkit import dumps, parse


def migrate_legacy_providers(config_path: Path) -> bool:
    """Migrate [api_settings] local/custom providers to [provider_instances].

    Returns True if migration was performed.
    """
    if not config_path.exists():
        return False

    with open(config_path) as f:
        doc = parse(f.read())

    if "provider_instances" in doc:
        return False  # Already migrated

    migrated = {}
    cloud_providers = {"openai", "anthropic", "cohere", "deepseek", "google", "groq", "huggingface", "mistralai", "moonshot", "openrouter", "zai"}

    for provider, settings in doc.get("api_settings", {}).items():
        if provider in cloud_providers:
            continue
        instance_id = f"{provider}-1"
        # Handle both api_ip and api_url field names across providers
        endpoint = settings.get("api_ip") or settings.get("api_url", "")
        migrated[instance_id] = {
            "provider_type": provider,
            "name": provider.replace("_", " ").title(),
            "endpoint": endpoint,
            "model": settings.get("model", ""),
            "temperature": settings.get("temperature", 0.7),
            "max_tokens": settings.get("max_tokens", 4096),
            "keys": {
                "default": {
                    "value": settings.get("api_key", ""),
                    "default": True,
                }
            },
        }

    if not migrated:
        return False

    doc["provider_instances"] = migrated
    # Delete migrated local/custom providers from api_settings
    for provider in list(doc.get("api_settings", {}).keys()):
        if provider not in cloud_providers:
            del doc["api_settings"][provider]

    with open(config_path, "w") as f:
        f.write(dumps(doc))
    return True
```

- [ ] **Step 5: Wire migration into load_settings**

In `load_settings()`, call `migrate_legacy_providers(config_path)` before loading.

- [ ] **Step 6: Run tests and commit**

Run: `pytest Tests/test_config_provider_instances.py -v`
Expected: PASS.

```bash
git add pyproject.toml tldw_chatbook/config.py Tests/test_config_provider_instances.py
git commit -m "feat(config): add provider_instances section and tomlkit migration"
```

---

### Task 8: Bootstrap registry and attach to app

**Files:**
- Create: `tldw_chatbook/Providers/bootstrap.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Providers/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Providers.bootstrap import build_registry_from_config


def test_build_registry_from_config():
    config = {
        "provider_instances": {
            "vllm-1": {
                "provider_type": "vllm",
                "name": "vLLM Prod",
                "endpoint": "http://localhost:8000/v1",
                "model": "llama-3.1-70b",
                "temperature": 0.7,
                "max_tokens": 4096,
                "keys": {
                    "default": {"value": "sk-1", "default": True},
                },
            }
        }
    }
    registry = build_registry_from_config(config)
    instance = registry.get("vllm-1")
    assert instance is not None
    assert instance.name == "vLLM Prod"
    assert len(instance.api_keys) == 1
```

Run: `pytest Tests/Providers/test_bootstrap.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement bootstrap**

```python
from __future__ import annotations

from typing import Any

from .instances import ApiKey, InstanceRegistry, ProviderInstance


def build_registry_from_config(config: dict[str, Any]) -> InstanceRegistry:
    registry = InstanceRegistry()
    for instance_id, data in config.get("provider_instances", {}).items():
        keys = tuple(
            ApiKey(
                label=k,
                value=v.get("value", ""),
                is_default=v.get("default", False),
            )
            for k, v in data.get("keys", {}).items()
        )
        instance = ProviderInstance(
            id=instance_id,
            provider_type=data.get("provider_type", "custom"),
            name=data.get("name", instance_id),
            endpoint=data.get("endpoint", ""),
            api_keys=keys,
            model_defaults={
                "model": data.get("model", ""),
                "temperature": data.get("temperature", 0.7),
                "max_tokens": data.get("max_tokens", 4096),
            },
            extra_options=data.get("extra_options", {}),
        )
        registry.add(instance)
    return registry
```

- [ ] **Step 3: Attach to app**

In `app.py` `__init__` or `on_mount`:
```python
from tldw_chatbook.Providers.bootstrap import build_registry_from_config
from tldw_chatbook.Providers.readiness import ReadinessCache

self.provider_instance_registry = build_registry_from_config(self.app_config)
self.provider_readiness_cache = ReadinessCache()
```

- [ ] **Step 4: Add config persistence helper**

Add to `Providers/bootstrap.py`:

```python
from pathlib import Path

from tomlkit import dumps, parse

from tldw_chatbook.config import _get_effective_config_path


def save_instances_to_config(registry: InstanceRegistry) -> None:
    """Persist the current instance registry to config.toml via tomlkit."""
    config_path = _get_effective_config_path()
    with open(config_path) as f:
        doc = parse(f.read())

    instances = {}
    for instance in registry.list():
        instances[instance.id] = {
            "provider_type": instance.provider_type,
            "name": instance.name,
            "endpoint": instance.endpoint,
            "model": instance.model_defaults.get("model", ""),
            "temperature": instance.model_defaults.get("temperature", 0.7),
            "max_tokens": instance.model_defaults.get("max_tokens", 4096),
            "keys": {
                k.label: {"value": k.value, "default": k.is_default}
                for k in instance.api_keys
            },
        }
    doc["provider_instances"] = instances

    with open(config_path, "w") as f:
        f.write(dumps(doc))
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest Tests/Providers/test_bootstrap.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/Providers/bootstrap.py tldw_chatbook/app.py Tests/Providers/test_bootstrap.py
git commit -m "feat(providers): bootstrap registry and attach to app"
```

---

### Task 9: Add Settings UI for instance CRUD

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_provider_instances.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_provider_instances_section_present(settings_screen):
    section = settings_screen.query_one("#provider-instances-section")
    assert section is not None
```

Run: `pytest Tests/UI/test_settings_provider_instances.py -v`
Expected: FAIL.

- [ ] **Step 2: Add Provider Instances section to settings**

Add a new collapsible section in `settings_screen.py` compose:
```python
with Collapsible(title="Provider Instances", id="provider-instances-section"):
    yield DataTable(id="provider-instances-table")
    yield Button("Add Instance", id="add-provider-instance", variant="primary")
    yield Button("Test Selected", id="test-provider-instance")
    yield Button("Delete Selected", id="delete-provider-instance", variant="error")
```

- [ ] **Step 3: Add instance list population**

```python
def _populate_provider_instances_table(self) -> None:
    table = self.query_one("#provider-instances-table", DataTable)
    table.clear(columns=True)
    table.add_columns("Type", "Name", "Endpoint", "Keys")
    registry = self.app_instance.provider_instance_registry
    for instance in registry.list():
        table.add_row(
            instance.provider_type,
            instance.name,
            instance.endpoint,
            str(len(instance.api_keys)),
            key=instance.id,
        )
```

- [ ] **Step 4: Add CRUD handlers and config persistence**

```python
def _save_instances_to_config(self) -> None:
    """Persist the current instance registry to config.toml via tomlkit."""
    from tldw_chatbook.Providers.bootstrap import save_instances_to_config

    save_instances_to_config(self.app_instance.provider_instance_registry)
```


@on(Button.Pressed, "#add-provider-instance")
def _on_add_instance(self) -> None:
    # Open add instance form/modal
    pass

@on(Button.Pressed, "#test-provider-instance")
async def _on_test_instance(self) -> None:
    table = self.query_one("#provider-instances-table", DataTable)
    if table.row_count == 0:
        return
    row_key = table.coordinate_to_cell_key(table.cursor_row, 0).row_key
    if row_key is None or row_key.value is None:
        return
    instance_id = str(row_key.value)
    registry = self.app_instance.provider_instance_registry
    instance = registry.get(instance_id)
    if instance is None:
        return
    readiness = self.app_instance.provider_readiness_cache
    ok = await readiness.check_instance(instance.id, instance.endpoint)
    self.notify("Ready" if ok else "Unreachable", severity="information" if ok else "error")

@on(Button.Pressed, "#delete-provider-instance")
def _on_delete_instance(self) -> None:
    table = self.query_one("#provider-instances-table", DataTable)
    if table.row_count == 0:
        return
    row_key = table.coordinate_to_cell_key(table.cursor_row, 0).row_key
    if row_key is None or row_key.value is None:
        return
    instance_id = str(row_key.value)
    registry = self.app_instance.provider_instance_registry
    registry.remove(instance_id)
    self._populate_provider_instances_table()
    self._save_instances_to_config()
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest Tests/UI/test_settings_provider_instances.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_provider_instances.py
git commit -m "feat(settings): add provider instances CRUD UI"
```

---

### Task 10: Add Chat UI two-level selector with existing provider integration

**Files:**
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Test: `Tests/UI/test_chat_provider_selector.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_instance_selector_present(chat_window):
    instance_select = chat_window.query_one("#provider-instance-select")
    assert instance_select is not None
```

Run: `pytest Tests/UI/test_chat_provider_selector.py -v`
Expected: FAIL.

- [ ] **Step 2: Add instance selector alongside existing provider dropdown**

Keep the existing `#chat-api-provider` for cloud providers. Add a new section for instances:
```python
with Horizontal(id="provider-instance-section"):
    yield Select(
        [(i.name, i.id) for i in self.app_instance.provider_instance_registry.list()],
        id="provider-instance-select",
        prompt="Select instance",
    )
    yield Select(
        [],
        id="provider-key-select",
        prompt="Select key",
        disabled=True,
    )
    yield Button("Duplicate", id="duplicate-instance", compact=True)
```

- [ ] **Step 3: Add key selector population on instance change**

```python
@on(Select.Changed, "#provider-instance-select")
def _on_instance_changed(self, event: Select.Changed) -> None:
    instance_id = event.value
    if instance_id == Select.BLANK:
        return
    registry = self.app_instance.provider_instance_registry
    instance = registry.get(instance_id)
    if instance is None:
        return
    key_select = self.query_one("#provider-key-select", Select)
    key_select.set_options([(k.label, k.label) for k in instance.api_keys])
    key_select.disabled = len(instance.api_keys) <= 1
    if instance.api_keys:
        default_key = next((k for k in instance.api_keys if k.is_default), instance.api_keys[0])
        key_select.value = default_key.label
```

- [ ] **Step 4: Add duplicate current button**

```python
from dataclasses import replace


@on(Button.Pressed, "#duplicate-instance")
def _on_duplicate_instance(self) -> None:
    instance_id = self.query_one("#provider-instance-select", Select).value
    if instance_id == Select.BLANK:
        return
    registry = self.app_instance.provider_instance_registry
    instance = registry.get(instance_id)
    if instance is None:
        return
    # Find a unique ID for the copy
    base_id = f"{instance.id}-copy"
    new_id = base_id
    counter = 1
    while registry.get(new_id) is not None:
        counter += 1
        new_id = f"{base_id}-{counter}"
    new_instance = replace(instance, id=new_id, name=f"{instance.name} (Copy)")
    registry.add(new_instance)
    self.notify(f"Duplicated to {new_id}")
    self.refresh(recompose=True)
    self._save_instances_to_config()
```

- [ ] **Step 5: Add Ctrl+Shift+I shortcut**

```python
BINDINGS = [
    # ... existing bindings ...
    ("ctrl+shift+i", "focus_instance_selector", "Provider Instances"),
]
```

- [ ] **Step 6: Run tests and commit**

Run: `pytest Tests/UI/test_chat_provider_selector.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/UI/Chat_Window_Enhanced.py Tests/UI/test_chat_provider_selector.py
git commit -m "feat(chat): add two-level provider instance selector"
```

---

### Task 11: Add config watching

**Files:**
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/test_config_watching.py`

- [ ] **Step 1: Write the failing test**

```python
def test_config_watcher_detects_changes(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[general]\n")
    watcher = ConfigWatcher(config_path)
    config_path.write_text("[general]\nkey = \"value\"\n")
    assert watcher.check_for_changes() is True
```

Run: `pytest Tests/test_config_watching.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement config watcher**

```python
class ConfigWatcher:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._last_mtime = config_path.stat().st_mtime if config_path.exists() else 0

    def check_for_changes(self) -> bool:
        if not self.config_path.exists():
            return False
        current_mtime = self.config_path.stat().st_mtime
        if current_mtime > self._last_mtime:
            self._last_mtime = current_mtime
            return True
        return False
```

- [ ] **Step 3: Add periodic check in app**

In `app.py`, add a timer that checks for config changes every 2 seconds and reloads the instance registry:
```python
def _watch_config_changes(self) -> None:
    if self._config_watcher.check_for_changes():
        self.app_config = load_settings()
        self.provider_instance_registry = build_registry_from_config(self.app_config)
        self.notify("Config reloaded")
        # Note: Select widgets are not repopulated automatically in v1;
        # the user must reopen the chat screen to see new instances.
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest Tests/test_config_watching.py -v`
Expected: PASS.

```bash
git add tldw_chatbook/config.py tldw_chatbook/app.py Tests/test_config_watching.py
git commit -m "feat(config): add config file watching"
```

---

### Task 12: Integration tests

**Files:**
- Create: `Tests/Integration/test_provider_instance_flow.py`

- [ ] **Step 1: Write integration tests**

```python
async def test_full_instance_flow(chat_window, tmp_path):
    """Add instance, select it, switch key mid-chat, verify conversation preserved."""
    # Setup: add instance via registry
    registry = chat_window.app_instance.provider_instance_registry
    instance = ProviderInstance(
        id="vllm-1",
        provider_type="vllm",
        name="vLLM Test",
        endpoint="http://localhost:8000/v1",
        api_keys=(
            ApiKey(label="prod", value="sk-1", is_default=True),
            ApiKey(label="staging", value="sk-2"),
        ),
        model_defaults={"model": "test-model"},
        extra_options={},
    )
    registry.add(instance)

    # Select instance
    instance_select = chat_window.query_one("#provider-instance-select", Select)
    instance_select.value = "vllm-1"
    await chat_window.app.workers.wait_for_complete()

    # Verify key selector populated
    key_select = chat_window.query_one("#provider-key-select", Select)
    assert not key_select.disabled

    # Switch key
    key_select.value = "staging"
    await chat_window.app.workers.wait_for_complete()

    # Verify conversation preserved (no reset)
    assert chat_window.conversation_id is not None
```

- [ ] **Step 2: Run integration tests**

Run: `pytest Tests/Integration/test_provider_instance_flow.py -v`

- [ ] **Step 3: Commit**

```bash
git add Tests/Integration/test_provider_instance_flow.py
git commit -m "test(integration): add provider instance flow tests"
```

---

### Task 13: Property tests

**Files:**
- Create: `Tests/Providers/test_properties.py`

- [ ] **Step 1: Write property tests**

```python
from hypothesis import given, strategies as st

from tldw_chatbook.Providers.sanitization import sanitize_error_for_display


@given(st.text())
def test_sanitize_never_leaks_long_key(error_text):
    """Any string containing a 20+ char key-like sequence should have it sanitized."""
    result = sanitize_error_for_display(error_text)
    # If the input had a long key pattern, it should be gone or replaced
    import re
    long_keys = re.findall(r"sk-[a-zA-Z0-9]{20,}", error_text)
    for key in long_keys:
        assert key not in result
```

- [ ] **Step 2: Run property tests**

Run: `pytest Tests/Providers/test_properties.py -v`

- [ ] **Step 3: Commit**

```bash
git add Tests/Providers/test_properties.py
git commit -m "test(providers): add property tests"
```

---

### Task 14: Final QA and documentation

- [ ] **Step 1: Run full test suite**

```bash
pytest Tests/Providers/ Tests/UI/ Tests/Integration/ Tests/test_config_provider_instances.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

```bash
ruff check tldw_chatbook/Providers/ Tests/Providers/
```

Expected: no issues.

- [ ] **Step 3: Update spec status**

Change spec status from "Draft" to "Implemented".

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(spec): mark provider instance registry implemented"
```

---

## Plan Review Loop

After completing the plan document, dispatch a single `plan-document-reviewer` subagent with:
- Path to this plan: `docs/superpowers/plans/2026-07-20-provider-instance-registry.md`
- Path to the spec: `docs/superpowers/specs/2026-07-20-provider-instance-registry-design.md`

Fix any blockers and re-dispatch until approved (max 3 iterations).

## Execution Handoff

Once the plan is approved, choose execution:

1. **Subagent-Driven (recommended)** — dispatch fresh subagent per task using `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`.
