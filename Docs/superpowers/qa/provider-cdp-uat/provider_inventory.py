"""Build a redacted provider inventory for Console CDP UAT.

This helper intentionally avoids shell-sourcing credential files. Raw secrets
are kept only in local memory long enough to decide whether a provider can be
tested and to write a masked status into QA artifacts.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PLACEHOLDER_PREFIXES = ("<", "your_", "YOUR_")
PLACEHOLDER_VALUES = {"", "<API_KEY_HERE>", "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS"}

LOW_COST_MODEL_OVERRIDES = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-3-5-haiku-20241022",
    "cohere": "command-r-08-2024",
    "deepseek": "deepseek-chat",
    "google": "gemini-2.0-flash-lite",
    "groq": "llama-3.1-8b-instant",
    "mistral": "open-mistral-nemo",
    "mistralai": "open-mistral-nemo",
    "moonshot": "kimi-latest",
    "openrouter": "openai/gpt-4o-mini",
    "zai": "glm-4.5-flash",
}

KNOWN_PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "huggingface": ("HUGGINGFACE_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "mistralai": ("MISTRAL_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "zai": ("ZAI_API_KEY",),
}

HANDLER_MODEL_DEFAULTS = {
    "zai": "glm-4.5-flash",
}

ENDPOINT_CONFIG_KEYS = (
    "api_url",
    "base_url",
    "api_base",
    "api_endpoint",
    "endpoint",
    "api_base_url",
    "api_ip",
)

LEGACY_PROVIDER_CONFIG_ALIASES = {
    "aphrodite": ("aphrodite_api",),
    "oobabooga": ("ooba_api",),
    "tabbyapi": ("tabby_api",),
}

OPENAI_COMPATIBLE_LOCAL_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "llama_cpp",
        "local_llm",
        "local_llamacpp",
        "local_llamafile",
        "local_mlx_lm",
        "local_ollama",
        "local_vllm",
        "mlx_lm",
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)

SERVER_DEFAULT_MODEL_KEYS = frozenset(
    {
        "koboldcpp",
        "llama_cpp",
        "local_llm",
        "local_llamacpp",
        "local_llamafile",
    }
)

DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0
MAX_PROBE_TIMEOUT_SECONDS = 3.0
ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class ModelSelection:
    """Selected UAT model and the reason it was selected."""

    model: str
    source: str
    requires_explicit_selection: bool


@dataclass(frozen=True)
class EndpointConfig:
    """Configured endpoint value and the config key that supplied it."""

    value: str
    source: str


@dataclass(frozen=True)
class EndpointProbeResult:
    """Reachability result for a local/custom endpoint probe."""

    reachable: bool | None
    probe_url: str
    status: str


@dataclass(frozen=True)
class KeyStatus:
    """Redacted provider key state for QA output."""

    has_usable_key: bool
    source: str
    masked: str


@dataclass(frozen=True)
class RuntimeProviderDependencies:
    """Chatbook provider dependencies loaded only for inventory extraction."""

    api_call_handlers: Mapping[str, object]
    resolve_console_provider_identity: object
    providers_requiring_api_key_keys: frozenset[str]


def load_env_values(path: Path) -> dict[str, str]:
    """Parse dotenv-style key/value pairs without executing shell syntax.

    Args:
        path: Environment file to read.

    Returns:
        Parsed values with simple ``${NAME}`` references expanded from values
        already parsed or from the current process environment.
    """

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue

        value = _strip_matching_quotes(value.strip())

        def replace_reference(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in values:
                return values[name]
            return os.environ.get(name, "")

        values[key] = ENV_REFERENCE_RE.sub(replace_reference, value)
    return values


def should_use_key_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped in PLACEHOLDER_VALUES:
        return False
    if stripped.startswith("${") and stripped.endswith("}"):
        return False
    if stripped.startswith(PLACEHOLDER_PREFIXES):
        return False
    return bool(stripped)


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def choose_uat_model(
    provider_key: str,
    configured_models: Sequence[str] | None,
    provider_config: Mapping[str, object],
    *,
    allow_server_default: bool = False,
) -> ModelSelection:
    """Select the model to use for a provider UAT row.

    Args:
        provider_key: Readiness or execution provider key.
        configured_models: Provider model list from Console/config, if known.
        provider_config: Provider-specific config mapping.
        allow_server_default: Whether the target endpoint can supply a model.

    Returns:
        Model selection with source and whether CDP must explicitly select it.
    """

    normalized_key = normalize_provider_key(provider_key)
    override_key = _model_override_key(provider_key)
    if override_key in LOW_COST_MODEL_OVERRIDES:
        return ModelSelection(
            model=LOW_COST_MODEL_OVERRIDES[override_key],
            source=f"override:{override_key}",
            requires_explicit_selection=True,
        )

    configured_model = first_usable_model(configured_models or ())
    if configured_model is not None:
        return ModelSelection(
            model=configured_model,
            source=f"configured_models:{provider_key}",
            requires_explicit_selection=True,
        )

    for config_key in ("model", "api_model", "default_model"):
        model = first_usable_model((provider_config.get(config_key),))
        if model is not None:
            return ModelSelection(
                model=model,
                source=f"config:{config_key}",
                requires_explicit_selection=True,
            )

    handler_default = HANDLER_MODEL_DEFAULTS.get(provider_key) or HANDLER_MODEL_DEFAULTS.get(normalized_key)
    if handler_default:
        return ModelSelection(
            model=handler_default,
            source=f"handler_default:{provider_key}",
            requires_explicit_selection=True,
        )

    if allow_server_default:
        return ModelSelection(
            model="",
            source="server_default",
            requires_explicit_selection=False,
        )

    return ModelSelection(
        model="",
        source="explicit_model_missing",
        requires_explicit_selection=True,
    )


def extract_endpoint_config(provider_config: Mapping[str, object]) -> EndpointConfig:
    """Return the first configured endpoint and its source config key."""

    for key in ENDPOINT_CONFIG_KEYS:
        value = provider_config.get(key)
        if isinstance(value, str) and value.strip():
            return EndpointConfig(value=value.strip(), source=key)
    return EndpointConfig(value="", source="config_missing")


def classify_external_outcome(reason: str) -> str:
    if reason in {"missing_key", "endpoint_unreachable", "explicit_model_missing"}:
        return "skip"
    if reason in {"auth", "quota_or_rate_limit", "model_unavailable"}:
        return "fail_external"
    if reason in {"request_shape", "response_shape", "streaming", "console_ui"}:
        return "fail_chatbook"
    return "unknown"


def build_provider_inventory(
    *,
    env_file: Path | None = None,
    app_config: Mapping[str, object] | None = None,
    configured_models_by_provider: Mapping[str, Sequence[str]] | None = None,
    environ: Mapping[str, str] | None = None,
    probe_endpoints: bool = True,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    """Build redacted QA inventory rows from Chatbook's runtime handlers."""

    runtime = load_runtime_provider_dependencies()
    env_values = load_env_values(env_file) if env_file is not None else {}
    env = environ if environ is not None else os.environ
    config = app_config if app_config is not None else load_app_config()
    configured_models = (
        configured_models_by_provider
        if configured_models_by_provider is not None
        else load_configured_models_by_provider()
    )
    timeout = max(0.1, min(float(probe_timeout), MAX_PROBE_TIMEOUT_SECONDS))

    rows: list[dict[str, object]] = []
    handler_keys = frozenset(runtime.api_call_handlers)
    for handler_key in sorted(handler_keys):
        identity = runtime.resolve_console_provider_identity(handler_key, handler_keys=handler_keys)
        readiness_key = identity.readiness_key
        provider_config = provider_settings_for_key(config, readiness_key)
        requires_api_key = readiness_key in runtime.providers_requiring_api_key_keys
        is_local_or_custom = not requires_api_key

        key_status = resolve_key_status(
            readiness_key=readiness_key,
            provider_config=provider_config,
            env_values=env_values,
            environ=env,
            requires_api_key=requires_api_key,
        )

        endpoint_config = EndpointConfig(value="", source="")
        endpoint_probe = EndpointProbeResult(reachable=None, probe_url="", status="not_applicable")
        allow_server_default = False
        if is_local_or_custom:
            endpoint_config = extract_endpoint_config(provider_config)
            if endpoint_config.value and probe_endpoints:
                endpoint_probe = probe_endpoint(
                    readiness_key,
                    endpoint_config.value,
                    timeout=timeout,
                )
            elif endpoint_config.value:
                probe_url = endpoint_probe_url(readiness_key, endpoint_config.value)
                endpoint_probe = EndpointProbeResult(
                    reachable=None,
                    probe_url=probe_url,
                    status="not_probed",
                )
            else:
                endpoint_probe = EndpointProbeResult(
                    reachable=False,
                    probe_url="",
                    status="config_missing",
                )
            allow_server_default = provider_can_use_server_default(readiness_key)

        model_selection = choose_uat_model(
            provider_key=readiness_key,
            configured_models=configured_models_for_provider(
                readiness_key,
                identity.execution_key,
                configured_models,
            ),
            provider_config=provider_config,
            allow_server_default=allow_server_default,
        )

        initial_reason = "ready_for_cdp"
        initial_status = "pending_cdp"
        if requires_api_key and not key_status.has_usable_key:
            initial_reason = "missing_key"
            initial_status = classify_external_outcome(initial_reason)
        elif model_selection.source == "explicit_model_missing":
            initial_reason = "explicit_model_missing"
            initial_status = classify_external_outcome(initial_reason)
        elif is_local_or_custom and endpoint_probe.reachable is not True:
            initial_reason = "endpoint_unreachable"
            initial_status = classify_external_outcome(initial_reason)
        classification = classify_inventory_row(initial_reason, initial_status)

        rows.append(
            {
                "handler_key": handler_key,
                "display_key": identity.display_key,
                "readiness_key": readiness_key,
                "execution_key": identity.execution_key,
                "provider_name": provider_display_name(readiness_key),
                "requires_api_key": requires_api_key,
                "key_source": key_status.source,
                "has_usable_key": key_status.has_usable_key,
                "masked_key": key_status.masked,
                "model": model_selection.model,
                "model_source": model_selection.source,
                "requires_explicit_selection": model_selection.requires_explicit_selection,
                "endpoint": redact_url_for_output(endpoint_config.value),
                "endpoint_source": endpoint_config.source,
                "endpoint_reachable": endpoint_probe.reachable,
                "endpoint_probe_url": redact_url_for_output(endpoint_probe.probe_url),
                "endpoint_probe_status": endpoint_probe.status,
                "initial_status": initial_status,
                "initial_reason": initial_reason,
                "classification": classification,
            }
        )
    return rows


def write_json_inventory(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write inventory rows as redacted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    redacted_rows = [redact_inventory_row(row) for row in rows]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_count": len(redacted_rows),
        "providers": redacted_rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_inventory(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write inventory rows as a compact redacted Markdown table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Provider",
        "Display",
        "Readiness",
        "Execution",
        "Model",
        "Model Source",
        "Key",
        "Endpoint",
        "Endpoint Source",
        "Reachable",
        "Probe URL",
        "Probe Status",
        "Status",
        "Reason",
        "Classification",
    ]
    lines = [
        "# Provider CDP UAT Inventory",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        redacted_row = redact_inventory_row(row)
        key_label = "not_required"
        if redacted_row.get("requires_api_key"):
            key_label = f"{redacted_row.get('key_source', '')} {redacted_row.get('masked_key', '')}".strip()
        endpoint_source = redacted_row.get("endpoint_source") or ""
        reachable = redacted_row.get("endpoint_reachable")
        if reachable is None:
            reachable_label = ""
        else:
            reachable_label = "yes" if reachable else "no"
        values = [
            redacted_row.get("provider_name", ""),
            redacted_row.get("display_key", ""),
            redacted_row.get("readiness_key", ""),
            redacted_row.get("execution_key", ""),
            redacted_row.get("model", ""),
            redacted_row.get("model_source", ""),
            key_label,
            redacted_row.get("endpoint", ""),
            endpoint_source,
            reachable_label,
            redacted_row.get("endpoint_probe_url", ""),
            redacted_row.get("endpoint_probe_status", ""),
            redacted_row.get("initial_status", ""),
            redacted_row.get("initial_reason", ""),
            redacted_row.get("classification", ""),
        ]
        lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_runtime_provider_dependencies() -> RuntimeProviderDependencies:
    """Load Chatbook provider modules only when inventory extraction runs."""

    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
    from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS

    return RuntimeProviderDependencies(
        api_call_handlers=API_CALL_HANDLERS,
        resolve_console_provider_identity=resolve_console_provider_identity,
        providers_requiring_api_key_keys=frozenset(PROVIDERS_REQUIRING_API_KEY_KEYS),
    )


def load_app_config() -> Mapping[str, object]:
    """Load Chatbook config through the existing config system."""

    try:
        from tldw_chatbook.config import load_settings

        loaded = load_settings()
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def load_configured_models_by_provider() -> Mapping[str, Sequence[str]]:
    """Load Console provider model lists if the config helper is available."""

    try:
        from tldw_chatbook.config import get_cli_providers_and_models

        models = get_cli_providers_and_models()
    except Exception:
        return {}
    if not isinstance(models, Mapping):
        return {}
    return {
        str(provider): tuple(str(model) for model in provider_models if isinstance(model, str))
        for provider, provider_models in models.items()
        if isinstance(provider_models, Sequence) and not isinstance(provider_models, str)
    }


def resolve_key_status(
    *,
    readiness_key: str,
    provider_config: Mapping[str, object],
    env_values: Mapping[str, str],
    environ: Mapping[str, str],
    requires_api_key: bool,
) -> KeyStatus:
    """Resolve a provider key source without exposing the raw key."""

    if not requires_api_key:
        return KeyStatus(has_usable_key=True, source="not_required", masked="")

    configured_key = provider_config.get("api_key")
    if should_use_key_value(configured_key):
        key_value = str(configured_key).strip()
        return KeyStatus(
            has_usable_key=True,
            source=f"config:api_settings.{readiness_key}.api_key",
            masked=mask_secret(key_value),
        )

    for env_name in env_var_names_for_provider(readiness_key, provider_config):
        env_value = env_values.get(env_name)
        if should_use_key_value(env_value):
            return KeyStatus(
                has_usable_key=True,
                source=f"env_file:{env_name}",
                masked=mask_secret(env_value.strip()),
            )
        environ_value = environ.get(env_name)
        if should_use_key_value(environ_value):
            return KeyStatus(
                has_usable_key=True,
                source=f"env:{env_name}",
                masked=mask_secret(environ_value.strip()),
            )

    return KeyStatus(has_usable_key=False, source="missing", masked="")


def env_var_names_for_provider(
    readiness_key: str,
    provider_config: Mapping[str, object],
) -> tuple[str, ...]:
    """Return configured and conventional env var names for a provider."""

    names: list[str] = []
    configured_env_var = provider_config.get("api_key_env_var")
    if isinstance(configured_env_var, str) and configured_env_var.strip():
        names.append(configured_env_var.strip())
    names.extend(KNOWN_PROVIDER_ENV_KEYS.get(readiness_key, ()))
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def provider_settings_for_key(
    app_config: Mapping[str, object],
    readiness_key: str,
) -> Mapping[str, object]:
    """Return ``api_settings`` for a normalized provider readiness key."""

    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return {}

    for wanted_key in provider_config_lookup_keys(readiness_key):
        for configured_provider, configured_value in api_settings.items():
            if normalize_provider_key(configured_provider) != wanted_key:
                continue
            if isinstance(configured_value, Mapping):
                return configured_value
            return {}
    return {}


def provider_config_lookup_keys(readiness_key: str) -> tuple[str, ...]:
    """Return config keys matching a readiness key, preferring runtime aliases."""

    normalized = normalize_provider_key(readiness_key)
    aliases = tuple(normalize_provider_key(alias) for alias in LEGACY_PROVIDER_CONFIG_ALIASES.get(normalized, ()))
    keys = (*aliases, normalized)
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def provider_can_use_server_default(provider_key: str) -> bool:
    """Return whether Chatbook can omit the model for this provider."""

    return normalize_provider_key(provider_key) in SERVER_DEFAULT_MODEL_KEYS


def classify_inventory_row(initial_reason: str, initial_status: str) -> str:
    """Return the explicit QA classification for an inventory row."""

    return initial_reason or initial_status


def redact_inventory_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of a row safe for JSON/Markdown persistence."""

    redacted = dict(row)
    endpoint = redacted.get("endpoint")
    if isinstance(endpoint, str):
        redacted["endpoint"] = redact_url_for_output(endpoint)
    probe_url = redacted.get("endpoint_probe_url")
    if isinstance(probe_url, str):
        redacted["endpoint_probe_url"] = redact_url_for_output(probe_url)
    if redacted.get("masked_key"):
        redacted["masked_key"] = "***REDACTED***"
    return redacted


def redact_url_for_output(url: str) -> str:
    """Remove userinfo and query strings from endpoint URLs before persistence."""

    raw_url = str(url or "").strip()
    if not raw_url:
        return ""

    parsed = urllib.parse.urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def configured_models_for_provider(
    readiness_key: str,
    execution_key: str,
    configured_models_by_provider: Mapping[str, Sequence[str]],
) -> Sequence[str]:
    """Return configured model list matching a readiness or execution key."""

    wanted_keys = {normalize_provider_key(readiness_key), normalize_provider_key(execution_key)}
    for provider, models in configured_models_by_provider.items():
        if normalize_provider_key(provider) in wanted_keys:
            return models
    return ()


def probe_endpoint(
    provider_key: str,
    endpoint: str,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> EndpointProbeResult:
    """Probe a local/custom provider endpoint without sending chat content."""

    probe_url = endpoint_probe_url(provider_key, endpoint)
    if not probe_url:
        return EndpointProbeResult(reachable=False, probe_url="", status="config_missing")

    request = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "tldw-chatbook-provider-cdp-uat/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(getattr(response, "status", response.getcode()))
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        return EndpointProbeResult(
            reachable=status_code in {401, 403} or 200 <= status_code < 400,
            probe_url=probe_url,
            status=str(status_code),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return EndpointProbeResult(
            reachable=False,
            probe_url=probe_url,
            status=f"unreachable:{exc.__class__.__name__}",
        )

    return EndpointProbeResult(
        reachable=200 <= status_code < 400,
        probe_url=probe_url,
        status=str(status_code),
    )


def endpoint_probe_url(provider_key: str, endpoint: str) -> str:
    """Return the URL used to probe a configured local/custom endpoint."""

    endpoint = endpoint.strip()
    if not endpoint:
        return ""

    normalized_key = normalize_provider_key(provider_key)
    if normalized_key == "koboldcpp":
        return kobold_probe_url(endpoint)
    if normalized_key in OPENAI_COMPATIBLE_LOCAL_KEYS:
        return openai_compatible_models_url(endpoint)
    return endpoint


def openai_compatible_models_url(endpoint: str) -> str:
    """Normalize an OpenAI-compatible endpoint to its ``/v1/models`` URL."""

    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/completion",
    ):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path.endswith("/v1/models"):
        models_path = path
    elif path.endswith("/models"):
        models_path = path
    elif path.endswith("/v1"):
        models_path = f"{path}/models"
    else:
        models_path = f"{path}/v1/models" if path else "/v1/models"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, models_path, "", "")
    )


def kobold_probe_url(endpoint: str) -> str:
    """Return the KoboldCpp model probe URL for a configured endpoint."""

    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1/generate"):
        path = path[: -len("/generate")] + "/model"
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return endpoint


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_provider_key(provider: object) -> str:
    return str(provider or "").strip().lower().replace(" ", "_").replace("-", "_")


def _model_override_key(provider_key: str) -> str:
    raw = str(provider_key or "").strip().lower()
    if raw in LOW_COST_MODEL_OVERRIDES:
        return raw
    normalized = normalize_provider_key(raw)
    if normalized in LOW_COST_MODEL_OVERRIDES:
        return normalized
    return raw


def first_usable_model(values: Sequence[object]) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        model = value.strip()
        if not model or model.lower() == "none" or model.startswith("<"):
            continue
        return model
    return None


def provider_display_name(readiness_key: str) -> str:
    names = {
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "custom": "Custom OpenAI",
        "custom_2": "Custom OpenAI 2",
        "deepseek": "DeepSeek",
        "google": "Google",
        "groq": "Groq",
        "huggingface": "Hugging Face",
        "llama_cpp": "llama.cpp",
        "local_llamacpp": "local llama.cpp",
        "local_mlx_lm": "MLX LM",
        "local_vllm": "local vLLM",
        "mistral": "Mistral",
        "mistralai": "MistralAI",
        "moonshot": "Moonshot",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
        "vllm": "vLLM",
        "zai": "Z.ai",
    }
    return names.get(readiness_key, readiness_key.replace("_", " ").replace("-", " ").title())


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Dotenv file containing provider keys")
    parser.add_argument("--json", type=Path, help="Path for redacted JSON inventory")
    parser.add_argument("--markdown", type=Path, help="Path for redacted Markdown inventory")
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="Local endpoint probe timeout in seconds, capped at 3 seconds",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Build endpoint probe URLs without contacting local/custom endpoints",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_provider_inventory(
        env_file=args.env_file,
        probe_endpoints=not args.no_probe,
        probe_timeout=args.probe_timeout,
    )
    if args.json:
        write_json_inventory(args.json, rows)
    if args.markdown:
        write_markdown_inventory(args.markdown, rows)
    if not args.json and not args.markdown:
        json.dump({"providers": rows}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        outputs = [str(path) for path in (args.json, args.markdown) if path]
        print(f"Wrote {len(rows)} provider inventory rows to {', '.join(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
