"""Two secret-key policies, one vocabulary.

TASK-32901 (tier-2 S13 P2): ``Workspaces/models.scrub_secret_metadata`` is the
scrubber on every runtime-binding metadata write (the boundary between
caller-supplied metadata and the persisted ``metadata_json`` column), and it
is a second, weaker policy than the shared ``MCP/redaction.redact_mapping``.
The drift ran both ways: Workspaces missed ``passwd``/``authorization``/
``bearer``; the shared redactor missed ``private_key``, which Workspaces
caught.

Only internal writers reach the column today, so this is defence-in-depth
with a hole rather than a live leak. The drop-vs-redact difference between
the two is deliberate and stays: the registry drops secret keys instead of
persisting ``"***"``.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.MCP.redaction import is_secret_key, redact_mapping
from tldw_chatbook.Workspaces.models import scrub_secret_metadata

# Every key either policy is meant to treat as a secret.
SECRET_KEYS = (
    "api_key",
    "apikey",
    "API-Key",
    "authorization",
    "Authorization",
    "bearer",
    "bearer_token",
    "credential",
    "db_passwd",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_registry_metadata_scrubber_drops_every_secret_key(key):
    assert scrub_secret_metadata({key: "hunter2", "note": "keep"}) == {"note": "keep"}


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_shared_redactor_recognizes_every_secret_key(key):
    assert is_secret_key(key) is True
    assert redact_mapping({key: "hunter2"})[key] == "***"


def test_nested_binding_metadata_is_scrubbed_at_depth():
    scrubbed = scrub_secret_metadata(
        {"outer": {"authorization": "Bearer abc", "host": "example.invalid"}}
    )
    assert scrubbed == {"outer": {"host": "example.invalid"}}


def test_ordinary_keys_survive_both_policies():
    payload = {"host": "example.invalid", "port": 8080, "keyring_backend": "file"}
    assert scrub_secret_metadata(payload) == payload
    assert redact_mapping(payload) == payload
