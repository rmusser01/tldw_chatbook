"""Effective authority identity and closed profile storage for scoped MCP owners."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from tldw_chatbook.MCP.connection_ownership import (
    ConnectionAuthorityKey,
    ConnectionOwnership,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("installation_id", "other"),
        ("revision_digest", "different"),
        ("executable", "/different/python"),
        ("arguments", ("two",)),
        ("environment", (("A", "2"),)),
        ("cwd", "/different/cwd"),
        ("endpoint", "https://different.example/rpc"),
        ("configuration_digest", "changed"),
        ("credential_bindings", "changed-generation"),
    ],
)
async def test_reuse_requires_every_reviewed_authority_field(field, value):
    owner = ConnectionOwnership(
        plugin_service=SimpleNamespace(), local_service=SimpleNamespace()
    )
    key = ConnectionAuthorityKey(
        "installed",
        "revision",
        "/python",
        ("one",),
        (("A", "1"),),
        "/cwd",
        "",
        "config",
        "credential-generation-1",
        "request_independent",
    )
    a = owner.attach(key, "a")
    assert owner.attach(key, "b") == a
    assert owner.attach(replace(key, **{field: value}), "c") != a


@pytest.mark.asyncio
async def test_unknown_isolation_cannot_create_shared_authority():
    owner = ConnectionOwnership(
        plugin_service=SimpleNamespace(), local_service=SimpleNamespace()
    )
    key = ConnectionAuthorityKey(
        "installed", "revision", "/python", (), (), None, "", "config", "[]", "separate"
    )
    assert owner.attach(key, "a") != owner.attach(key, "b")
    with pytest.raises(ValueError, match="isolation|authority"):
        owner.attach(replace(key, session_isolation="server_claimed"), "c")
