"""Installed semantic locations for managed credentials, never archive policy."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SQLiteCredentialColumn:
    owner: str
    table: str
    column: str
    kind: Literal["config", "headers", "auth", "url"]
    identity: str = "id"


SQLITE_CREDENTIAL_COLUMNS = (
    SQLiteCredentialColumn("db.subscriptions", "subscriptions", "auth_config", "auth"),
    SQLiteCredentialColumn(
        "db.subscriptions", "subscriptions", "custom_headers", "headers"
    ),
    SQLiteCredentialColumn(
        "db.subscriptions", "subscriptions", "notification_config", "config"
    ),
    SQLiteCredentialColumn("db.subscriptions", "subscriptions", "source", "url"),
    SQLiteCredentialColumn(
        "db.subscriptions", "subscription_templates", "auth_config_template", "auth"
    ),
    SQLiteCredentialColumn("db.subscriptions", "site_configs", "config_data", "config"),
    SQLiteCredentialColumn("db.subscriptions", "url_snapshots", "headers", "headers"),
    SQLiteCredentialColumn("db.subscriptions", "url_snapshots", "url", "url"),
    SQLiteCredentialColumn(
        "research.local", "research_runs", "provider_overrides_json", "config"
    ),
)

# These containers have semantics beyond a key-name classifier: custom HTTP
# headers/cookies may carry credentials under arbitrary names, and auth.key is
# the installed subscriptions API-key value (its header name is merely a hint).
HEADER_FIELDS = frozenset({"headers", "custom_headers", "http_headers"})
AUTH_FIELDS = frozenset(
    {"auth", "auth_config", "auth_credentials", "auth_config_template"}
)
REFERENCE_FIELDS = frozenset(
    {"auth_reference", "credential_ref", "credential_reference", "keyring_ref"}
)
URL_FIELDS = frozenset({"url", "base_url", "api_url", "webhook_url", "endpoint"})
URL_SECRET_PARAMETERS = frozenset(
    {"x-amz-signature", "x-amz-credential", "x-amz-security-token", "sig", "signature"}
)
CONFIG_OWNERS = frozenset({"config", "config.history"})
JSON_CONNECTION_OWNERS = frozenset({"mcp.local", "mcp.targets", "runtime.source_state"})

# Exactly the installed image/video providers' owned keyring slots. Selection
# requires an affected staged config section; neither environment nor unrelated
# keychain entries are enumerated.
GENERATION_KEYRINGS = (
    (
        "image_generation",
        "tldw_chatbook_imagegen",
        ("swarmui", "openrouter", "novita", "together", "modelstudio", "fal", "gemini"),
    ),
    ("video_generation", "tldw_chatbook_videogen", ("minimax",)),
)
