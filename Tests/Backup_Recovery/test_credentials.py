"""Staged managed-credential processing never changes source/private peers."""

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
import toml

from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.credentials import (
    plan_credential_scopes,
    process_credentials,
    restore_credential_values,
    sanitize_config,
)
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem


def inventory(path, owner="config"):
    return Inventory(
        (StorageItem(owner, "profile:p:" + owner, path, "included", ()),),
        True,
        "scope",
        (),
    )


def config_file(tmp_path, content):
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700, parents=True)
    path = staging / "config.toml"
    path.write_bytes(toml.dumps(content).encode())
    path.chmod(0o600)
    return staging, path


def test_config_secret_is_removed_without_mutating_source():
    original = {"API": {"openai_api_key": "synthetic-secret-sentinel"}}
    sanitized = sanitize_config(original)
    assert "synthetic-secret-sentinel" not in repr(sanitized)
    assert original["API"]["openai_api_key"] == "synthetic-secret-sentinel"


def test_semantic_aliases_headers_arrays_and_encrypted_blobs():
    original = {
        "API": {"OPENAI_API_KEY_fallback": "secret"},
        "providers": [
            {"api_key": "secret", "api_key_env_var": "OPENAI_API_KEY", "max_tokens": 10}
        ],
        "connection": {
            "auth": {"type": "api_key", "header": "X-Owned", "key": "secret"},
            "headers": {"X-Owned": "secret"},
        },
        "encrypted_other": "enc:secret",
        "encryption": {"enabled": True, "password_verifier": "secret"},
    }
    result = sanitize_config(original)
    assert "secret" not in repr(result)
    assert result["providers"] == [
        {"api_key_env_var": "OPENAI_API_KEY", "max_tokens": 10}
    ]
    assert original["connection"]["headers"]["X-Owned"] == "secret"


@pytest.mark.parametrize("mode", ["include", "rollback"])
def test_encryption_required_before_any_value_read(tmp_path, mode):
    with pytest.raises(ValueError, match="credentials_require_encryption"):
        process_credentials(
            tmp_path / "absent",
            inventory(tmp_path / "secret"),
            mode=mode,
            encrypted=False,
        )


@pytest.mark.parametrize("owner", ["config", "config.history"])
def test_staged_current_and_history_leave_source_untouched(tmp_path, owner):
    source = tmp_path / "source.toml"
    content = {"API": {"openai_api_key": "synthetic-managed-secret"}}
    source.write_text(toml.dumps(content))
    staging, path = config_file(tmp_path, content)
    assert (
        process_credentials(
            staging, inventory(path, owner), mode="exclude", encrypted=False
        )
        == ()
    )
    assert b"synthetic-managed-secret" not in path.read_bytes()
    assert "synthetic-managed-secret" in source.read_text()


@pytest.mark.parametrize("kind", ["outside", "hardlink", "symlink"])
def test_source_or_alias_never_sanitized(tmp_path, kind):
    source = tmp_path / "source.toml"
    source.write_text('[API]\nopenai_api_key="keep"\n')
    source.chmod(0o600)
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700, parents=True)
    path = source
    if kind != "outside":
        path = staging / "config.toml"
        path.hardlink_to(source) if kind == "hardlink" else path.symlink_to(source)
    assert process_credentials(
        staging, inventory(path), mode="exclude", encrypted=False
    )
    assert '"keep"' in source.read_text()


def test_unknown_credential_history_refuses_completeness(tmp_path):
    staging, path = config_file(tmp_path, {})
    path.write_bytes(b"broken-config=secret")
    assert process_credentials(
        staging, inventory(path, "config.history"), mode="exclude", encrypted=False
    )
    assert path.read_bytes() == b"broken-config=secret"


def test_subscription_rebuild_removes_freed_secrets_preserves_ids(tmp_path):
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700, parents=True)
    path = staging / "subscriptions.db"
    store = SubscriptionsDB(path)
    store.close()
    secret = "synthetic-freed-secret-" * 500
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA secure_delete=OFF")
        db.execute(
            "INSERT INTO subscriptions(id,name,type,source,auth_config) VALUES (70,'Keep','rss','https://example.invalid',?)",
            (json.dumps({"type": "bearer", "token": secret}),),
        )
        db.execute(
            "INSERT INTO subscriptions(id,name,type,source,auth_config) VALUES (71,'Delete','rss','https://example.invalid',?)",
            (json.dumps({"token": secret}),),
        )
        db.commit()
        db.execute("DELETE FROM subscriptions WHERE id=71")
        db.execute(
            "INSERT INTO subscription_items(id,subscription_id,url,title) VALUES (91,70,'https://example.invalid/item','preserved searchable word')"
        )
        db.commit()
        before_search = db.execute(
            "SELECT rowid FROM subscription_items_fts WHERE subscription_items_fts MATCH 'searchable'"
        ).fetchall()
        assert before_search == [(91,)]
        before_shadows = {
            name: db.execute('SELECT * FROM "' + name + '"').fetchall()
            for (name,) in db.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'subscription_items_fts_%'"
            )
        }
    source = tmp_path / "source.db"
    source.write_bytes(path.read_bytes())
    source_bytes = source.read_bytes()
    assert b"synthetic-freed-secret" in path.read_bytes()
    assert (
        process_credentials(
            staging,
            inventory(path, "db.subscriptions"),
            mode="exclude",
            encrypted=False,
        )
        == ()
    )
    assert b"synthetic-freed-secret" not in path.read_bytes()
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT id,name FROM subscriptions").fetchall() == [
            (70, "Keep")
        ]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            db.execute(
                "SELECT rowid FROM subscription_items_fts WHERE subscription_items_fts MATCH 'searchable'"
            ).fetchall()
            == before_search
        )
        for name, rows in before_shadows.items():
            assert db.execute('SELECT * FROM "' + name + '"').fetchall() == rows
    assert source.read_bytes() == source_bytes


def test_setup_marker_never_reads_shared_keyring():
    from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
    from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
    from tldw_chatbook.runtime_policy.server_credentials import RECOVERY_SETUP_REQUIRED

    reads = []
    provider = SimpleNamespace(
        _purpose_from_auth_reference=RuntimeServerContextProvider._purpose_from_auth_reference,
        _purposes_for_auth_mode=RuntimeServerContextProvider._purposes_for_auth_mode,
        _get_credential_secret=lambda *args: reads.append(args) or "shared-secret",
        _legacy_api_config=dict,
    )
    target = ConfiguredServerTarget.from_dict(
        {
            "server_id": "peer",
            "base_url": "https://example.invalid",
            "auth_reference": RECOVERY_SETUP_REQUIRED,
        }
    )
    assert RuntimeServerContextProvider._resolve_auth_token(
        provider, "peer", target, allow_legacy_config=False
    ) == (None, "none")
    assert reads == []


def test_research_hidden_rowid_survives_credential_reconstruction(tmp_path):
    from Tests.Backup_Recovery.test_sqlite_validation import research_candidate

    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    _, path = research_candidate(staging)
    path.chmod(0o600)
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "UPDATE research_runs SET rowid=93,provider_overrides_json=?",
            (json.dumps({"api_key": "research-secret"}),),
        )
        db.commit()
    assert (
        process_credentials(
            staging, inventory(path, "research.local"), mode="exclude", encrypted=False
        )
        == ()
    )
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT rowid,id,query FROM research_runs").fetchall() == [
            (93, "kept", "nebula")
        ]
    assert b"research-secret" not in path.read_bytes()


def test_existing_reconfigure_replaces_setup_marker(tmp_path):
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget

    store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="https://example.invalid",
                label="Restored",
                base_url="https://example.invalid",
                auth_reference="recovery:setup_required",
            )
        ]
    )
    updated = store.upsert_legacy_config_target(
        {
            "tldw_api": {
                "base_url": "https://example.invalid",
                "api_key": "newly-entered",
            }
        }
    )
    assert updated.auth_reference == "legacy:tldw_api"


def test_implicit_config_keyring_is_retained_with_explicit_issue(
    tmp_path, keyring_capture
):
    _, _, store, _ = keyring_capture
    store.set_secret("https://example.invalid", "api_key", "implicit-secret")
    staging, path = config_file(
        tmp_path / "config", {"tldw_api": {"base_url": "https://example.invalid/"}}
    )
    issues = process_credentials(
        staging, inventory(path), mode="include", encrypted=True
    )
    assert any("manual_recovery" in issue for issue in issues)
    assert "implicit-secret" in (staging / "credential-recovery.json").read_text()
    assert restore_credential_values(staging, plan_credential_scopes(staging))


def test_staging_directory_must_be_private(tmp_path):
    staging, path = config_file(tmp_path, {"api_key": "keep"})
    staging.chmod(0o755)
    assert process_credentials(
        staging, inventory(path), mode="exclude", encrypted=False
    )
    assert "keep" in path.read_text()


@pytest.fixture
def keyring_capture(tmp_path, monkeypatch):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    backend = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=backend)
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700, parents=True)
    path = staging / "targets.json"
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "server_id": "shared-peer",
                        "base_url": "https://example.invalid",
                        "auth_reference": "keyring:api_key",
                    }
                ]
            }
        )
    )
    path.chmod(0o600)
    store.set_secret("shared-peer", "api_key", "synthetic-captured-secret")
    return staging, path, store, backend


@pytest.mark.parametrize("change", ["changed", "deleted", "same"])
def test_changed_or_deleted_keyring_value_restores_isolated_scope(
    keyring_capture, change
):
    staging, path, store, backend = keyring_capture
    assert (
        process_credentials(
            staging, inventory(path, "mcp.targets"), mode="include", encrypted=True
        )
        == ()
    )
    if change == "changed":
        store.set_secret("shared-peer", "api_key", "peer-owned-secret")
    elif change == "deleted":
        store.delete_secret("shared-peer", "api_key")
    before = dict(backend.values)
    plan = plan_credential_scopes(staging)
    assert backend.values == before
    assert "synthetic-captured-secret" not in repr(plan)
    assert restore_credential_values(staging, plan) == ()
    purpose = json.loads(path.read_text())["targets"][0]["auth_reference"].removeprefix(
        "keyring:"
    )
    assert store.get_secret("shared-peer", purpose) == "synthetic-captured-secret"
    if change == "same":
        assert purpose == "api_key" and backend.values == before
    else:
        assert purpose.startswith("recovery_")
        assert store.get_secret("shared-peer", "api_key") == (
            "peer-owned-secret" if change == "changed" else None
        )


@pytest.mark.parametrize("drift", ["source", "destination", "reference", "material"])
def test_scope_drift_invalidates_plan_without_overwriting(keyring_capture, drift):
    staging, path, store, backend = keyring_capture
    process_credentials(
        staging, inventory(path, "mcp.targets"), mode="rollback", encrypted=True
    )
    store.set_secret("shared-peer", "api_key", "different")
    plan = plan_credential_scopes(staging)
    if drift == "source":
        store.set_secret("shared-peer", "api_key", "changed-again")
    elif drift == "destination":
        purpose = json.loads(next(iter(plan.values())))["purpose"]
        store.set_secret("shared-peer", purpose, "someone-else")
    elif drift == "reference":
        path.write_text(
            json.dumps(
                {
                    "targets": [
                        {"server_id": "shared-peer", "auth_reference": "keyring:other"}
                    ]
                }
            )
        )
    else:
        material = staging / "credential-recovery.json"
        data = json.loads(material.read_text())
        data["records"][0]["value"] = "changed-after-plan"
        material.write_text(json.dumps(data))
    before = dict(backend.values)
    assert restore_credential_values(staging, plan)
    assert backend.values == before


def test_excluded_target_keeps_only_setup_marker(keyring_capture):
    staging, path, _store, backend = keyring_capture
    before = dict(backend.values)
    assert (
        process_credentials(
            staging, inventory(path, "mcp.targets"), mode="exclude", encrypted=False
        )
        == ()
    )
    assert backend.values == before
    assert (
        json.loads(path.read_text())["targets"][0]["auth_reference"]
        == "recovery:setup_required"
    )


def test_missing_keyring_reports_omission_without_secret_error(
    keyring_capture, monkeypatch
):
    staging, path, _, _ = keyring_capture

    def missing():
        raise RuntimeError("synthetic-provider-secret")

    monkeypatch.setattr(credentials, "_credential_store", missing)
    issues = process_credentials(
        staging, inventory(path, "mcp.targets"), mode="include", encrypted=True
    )
    assert issues and "synthetic-provider-secret" not in repr(issues)
    assert (
        "synthetic-provider-secret"
        not in (staging / "credential-recovery.json").read_text()
    )


def test_scope_destination_probe_redacts_backend_error(keyring_capture, monkeypatch):
    staging, path, store, backend = keyring_capture
    assert (
        process_credentials(
            staging, inventory(path, "mcp.targets"), mode="include", encrypted=True
        )
        == ()
    )
    store.set_secret("shared-peer", "api_key", "changed-peer-value")
    before = dict(backend.values)
    original = backend.get_password

    def read(service, username):
        if "recovery_" in username:
            raise RuntimeError("synthetic-destination-secret")
        return original(service, username)

    monkeypatch.setattr(backend, "get_password", read)
    with pytest.raises(ValueError, match="^credential_store_unavailable$") as error:
        plan_credential_scopes(staging)
    assert "synthetic-destination-secret" not in str(error.value)
    assert error.value.__suppress_context__
    assert backend.values == before


@pytest.mark.parametrize("unlock", [None, "incorrect", "correct"])
def test_encrypted_config_requires_its_own_unlock(tmp_path, monkeypatch, unlock):
    from tldw_chatbook import config
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    cipher = ConfigEncryption().encrypt_value("synthetic-unlocked-secret", "correct")
    staging, path = config_file(tmp_path, {"API": {"openai_api_key": cipher}})
    original = path.read_bytes()
    monkeypatch.setattr(config, "get_encryption_password", lambda: unlock)
    issues = process_credentials(
        staging, inventory(path), mode="include", encrypted=True
    )
    assert path.read_bytes() == original
    material = (staging / "credential-recovery.json").read_text()
    assert ("synthetic-unlocked-secret" in material) == (unlock == "correct")
    assert bool(issues) == (unlock != "correct")


def test_rollback_retains_unparseable_config_bytes(tmp_path):
    staging, path = config_file(tmp_path, {})
    path.write_bytes(b"broken-synthetic-secret")
    assert process_credentials(
        staging, inventory(path), mode="rollback", encrypted=True
    )
    assert path.read_bytes() == b"broken-synthetic-secret"


@pytest.mark.parametrize(
    "section,backend",
    [("image_generation", "openrouter"), ("video_generation", "minimax")],
)
def test_excluded_generation_config_never_reuses_shared_keyring(
    monkeypatch, section, backend
):
    import importlib

    module = importlib.import_module(
        "tldw_chatbook."
        + ("Image_Generation" if section == "image_generation" else "Video_Generation")
        + ".config"
    )
    reads = []
    for variable in module._SECRETS[backend][1]:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        module, "_keyring_get", lambda name: reads.append(name) or "peer-secret"
    )
    assert module._resolve_secret(backend, {})[1] == "peer-secret"
    reads.clear()
    sanitized = sanitize_config({section: {backend: {"api_key": "removed"}}})
    assert module._resolve_secret(backend, sanitized[section][backend])[1] is None
    assert reads == []


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_unprocessed_sidecars_refuse_credential_completeness(tmp_path, suffix):
    staging, path = config_file(tmp_path, {"api_key": "keep"})
    path.with_name(path.name + suffix).write_text("sidecar-secret")
    assert process_credentials(
        staging, inventory(path), mode="exclude", encrypted=False
    )
    assert "keep" in path.read_text()


def test_sqlite_encrypted_values_require_independent_unlock(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_sqlite_validation import research_candidate
    from tldw_chatbook import config
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    _, path = research_candidate(staging)
    path.chmod(0o600)
    value = ConfigEncryption().encrypt_value("sqlite-owned-secret", "correct")
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "UPDATE research_runs SET provider_overrides_json=?",
            (json.dumps({"api_key": value}),),
        )
        db.commit()
    before = path.read_bytes()
    monkeypatch.setattr(config, "get_encryption_password", lambda: None)
    issues = process_credentials(
        staging, inventory(path, "research.local"), mode="include", encrypted=True
    )
    assert any("unlock_required" in issue for issue in issues)
    assert path.read_bytes() == before
    assert (
        "sqlite-owned-secret" not in (staging / "credential-recovery.json").read_text()
    )


@pytest.mark.parametrize("value", ["revoked-token-fixture", "expired-token-fixture"])
def test_capture_does_not_claim_remote_token_validity(keyring_capture, value):
    staging, path, store, _ = keyring_capture
    store.set_secret("shared-peer", "api_key", value)
    assert (
        process_credentials(
            staging, inventory(path, "mcp.targets"), mode="include", encrypted=True
        )
        == ()
    )
    records = json.loads((staging / "credential-recovery.json").read_text())["records"]
    assert records[0]["value"] == value
    assert "valid" not in records[0]


def test_exclusion_marks_absent_generation_sections(tmp_path):
    staging, path = config_file(tmp_path, {})
    assert (
        process_credentials(staging, inventory(path), mode="exclude", encrypted=False)
        == ()
    )
    data = toml.loads(path.read_text())
    assert (
        data["image_generation"]["openrouter"]["auth_reference"]
        == "recovery:setup_required"
    )
    assert (
        data["video_generation"]["minimax"]["auth_reference"]
        == "recovery:setup_required"
    )


def test_unrecognized_mcp_launch_credentials_refuse_excluded_completeness(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    path = staging / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "command": "custom-server",
                        "args": ["--opaque", "synthetic-connection-secret"],
                    }
                ]
            }
        )
    )
    path.chmod(0o600)
    before = path.read_bytes()
    assert process_credentials(
        staging, inventory(path, "mcp.local"), mode="exclude", encrypted=False
    )
    assert path.read_bytes() == before


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch):
    """All credential tests use synthetic stores, including optional fixed slots."""
    backend = SimpleNamespace(get_password=lambda *args: None)
    monkeypatch.setattr(
        credentials,
        "_credential_store",
        lambda: SimpleNamespace(
            _keyring=backend,
            get_scoped_secret=lambda scope: None,
        ),
    )


def test_generation_keyring_only_without_config_subsections(tmp_path, monkeypatch):
    staging, path = config_file(tmp_path, {})
    reads = []

    def get(service, username):
        reads.append((service, username))
        return "synthetic-keyring-only" if username == "minimax" else None

    monkeypatch.setattr(
        credentials,
        "_credential_store",
        lambda: SimpleNamespace(
            _keyring=SimpleNamespace(get_password=get),
        ),
    )
    issues = process_credentials(
        staging, inventory(path), mode="include", encrypted=True
    )
    records = json.loads((staging / "credential-recovery.json").read_text())["records"]
    assert any(row.get("value") == "synthetic-keyring-only" for row in records)
    assert any("manual_recovery" in issue for issue in issues)
    assert ("tldw_chatbook_videogen", "minimax") in reads


@pytest.mark.parametrize("method", ["load_key", "provision_key"])
def test_excluded_citation_reference_never_reads_or_provisions_key(method):
    from tldw_chatbook.Chat.citation_trace_identity import (
        CitationFingerprintKeyUnavailable,
        KeyringCitationFingerprintKeyProvider,
    )

    provider = KeyringCitationFingerprintKeyProvider()
    provider._secure_backend = lambda: pytest.fail("excluded key reached backend")
    with pytest.raises(CitationFingerprintKeyUnavailable):
        getattr(provider, method)("recovery:setup_required")


@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_skill_trust_bytes_are_retained_with_explicit_key_cache_coverage(
    tmp_path, monkeypatch, mode
):
    from tldw_chatbook.Backup_Recovery.models import FileMetadata

    staging, path = config_file(tmp_path, {})
    path = staging / "snapshot.enc"
    path.write_bytes(b"synthetic-encrypted-trust")
    path.chmod(0o600)
    item = StorageItem(
        "skills",
        "profile:p:skills:trust",
        path,
        "included",
        (),
        metadata=FileMetadata(
            1, "skills", "trust/snapshots/old.enc", None, "file", 0o600, 0, "private"
        ),
    )
    monkeypatch.setattr(
        credentials, "_credential_store", lambda: pytest.fail("trust keyring read")
    )
    issues = process_credentials(
        staging,
        Inventory((item,), True, "scope", ()),
        mode=mode,
        encrypted=mode != "exclude",
    )
    assert issues == (
        ()
        if mode == "exclude"
        else ("credential_skill_trust_manual_unlock_required:" + item.logical_id,)
    )
    assert path.read_bytes() == b"synthetic-encrypted-trust"


@pytest.mark.parametrize("mode", ["exclude", "include", "invalid"])
def test_citation_credential_policy_uses_real_staged_identity(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        mode,
        "citation",
        script=r"""
import base64, json, sqlite3, sys
from pathlib import Path
from contextlib import closing
from types import SimpleNamespace
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
root=Path.home()/'stage'; root.mkdir(mode=0o700)
path=root/'notes.db'
db=CharactersRAGDB(path, client_id='test'); db.close_connection(); path.chmod(0o600)
with closing(sqlite3.connect(path)) as conn:
    conn.execute("UPDATE rag_identity_context SET fingerprint_key_id='synthetic-citation-key'")
    conn.commit()
reads=[]
mode=sys.argv[1]
def get(service, key):
    reads.append((service,key))
    if mode=='exclude': raise AssertionError('excluded key read')
    return 'invalid-base64' if mode=='invalid' else base64.b64encode(b'k'*32).decode()
credentials._credential_store=lambda: SimpleNamespace(_keyring=SimpleNamespace(get_password=get))
inv=Inventory((StorageItem('db.chachanotes.primary','profile:p:notes',path,'included',()),), True,'scope',())
issues=credentials.process_credentials(root,inv,mode='exclude' if mode=='exclude' else 'include',encrypted=mode!='exclude')
if mode=='exclude':
    assert not issues, issues
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute('SELECT fingerprint_key_id FROM rag_identity_context').fetchone()[0]=='recovery:setup_required'
    assert b'synthetic-citation-key' not in path.read_bytes()
    assert not reads
else:
    records=json.loads((root/'credential-recovery.json').read_text())['records']
    selected=[r for r in records if r['kind']=='citation']
    assert len(selected)==1, records
    assert reads==[('tldw_chatbook.citation-provenance.v1','synthetic-citation-key')]
    assert selected[0]['status']==('unreadable' if mode=='invalid' else 'captured')
    if mode=='include':
        before=list(reads)
        plan=credentials.plan_credential_scopes(root)
        assert credentials.restore_credential_values(root,plan)
        assert reads==before
print('retired and reopened')
""",
    )


@pytest.mark.parametrize("mode", ["exclude", "include"])
@pytest.mark.parametrize("shared_group", [None, "forged-semantic-tag"])
def test_known_rag_profile_credentials_use_typed_config_policy(
    tmp_path, mode, shared_group
):
    from tldw_chatbook.Backup_Recovery.models import FileMetadata

    staging, _unused = config_file(tmp_path, {})
    path = staging / "profile.json"
    path.write_text(
        json.dumps(
            {
                "name": "research",
                "rag_config": {"embedding": {"api_key": "synthetic-rag-secret"}},
            }
        )
    )
    path.chmod(0o600)
    item = StorageItem(
        "rag.definitions",
        "profile:p:rag",
        path,
        "included",
        (),
        shared_group=shared_group,
        metadata=FileMetadata(
            1, "rag", "profile.json", None, "file", 0o600, 0, "private"
        ),
    )
    assert not process_credentials(
        staging,
        Inventory((item,), True, "scope", ()),
        mode=mode,
        encrypted=mode == "include",
    )
    assert ("synthetic-rag-secret" in path.read_text()) == (mode == "include")


@pytest.mark.parametrize(
    "relative", ["", "experiments/run.json", "custom_profiles.json", "other.json"]
)
def test_unqualified_rag_definition_is_explicit_credential_coverage(tmp_path, relative):
    from tldw_chatbook.Backup_Recovery.models import FileMetadata

    staging, path = config_file(tmp_path, {})
    path.write_text('{"provider":{"api_key":"synthetic-unknown"}}')
    item = StorageItem(
        "rag.definitions",
        "profile:p:rag:unknown",
        path,
        "included",
        (),
        shared_group="rag-definitions:profiles",
        metadata=FileMetadata(1, "rag", relative, None, "file", 0o600, 0, "private"),
    )
    issues = process_credentials(
        staging, Inventory((item,), True, "scope", ()), mode="exclude", encrypted=False
    )
    assert issues == (
        "credential_rag_definition_format_unsupported:profile:p:rag:unknown",
    )
    assert "synthetic-unknown" in path.read_text()


@pytest.mark.parametrize("mode", ["exclude", "include"])
def test_shared_citation_payloads_apply_identical_credential_policy(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        mode,
        "aliases",
        script=r"""
import base64, json, shutil, sqlite3, sys
from pathlib import Path
from contextlib import closing
from types import SimpleNamespace
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
root=Path.home()/'stage'; root.mkdir(mode=0o700)
source=root/'notes.db'
db=CharactersRAGDB(source,client_id='test'); db.close_connection(); source.chmod(0o600)
with closing(sqlite3.connect(source)) as conn:
    conn.execute("UPDATE rag_identity_context SET fingerprint_key_id='synthetic-shared-citation'")
    conn.commit()
owners=['db.chachanotes.primary','chat.attachments','study.local','quiz.local','notes.sync_bindings']
paths=[source]
for index in range(1,len(owners)):
    path=root/f'alias{index}.db'; shutil.copyfile(source,path);path.chmod(0o600);paths.append(path)
items=tuple(StorageItem(owner,'profile:p:'+owner,path,'included',(),shared_group='shared:chachanotes:profile:p') for owner,path in zip(owners,paths))
credentials._credential_store=lambda: SimpleNamespace(_keyring=SimpleNamespace(get_password=lambda *args:base64.b64encode(b'x'*32).decode()))
mode=sys.argv[1]
issues=credentials.process_credentials(root,Inventory(items,True,'scope',()),mode=mode,encrypted=mode=='include')
assert not issues if mode=='exclude' else all('manual_recovery' in issue for issue in issues),issues
payloads=[path.read_bytes() for path in paths]
assert all(payload==payloads[0] for payload in payloads)
if mode=='exclude':
    assert all(b'synthetic-shared-citation' not in payload for payload in payloads)
else:
    records=json.loads((root/'credential-recovery.json').read_text())['records']
    assert {record['file'] for record in records if record['kind']=='citation'}=={path.name for path in paths}
print('retired and reopened')
""",
    )


def test_native_capture_snapshot_is_self_contained_without_changing_source_wal(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE retained(value TEXT)")
        connection.execute("INSERT INTO retained VALUES ('retained')")
        connection.commit()
    source.chmod(0o600)
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "candidate.sqlite"
    authority = application_authority(tmp_path, source, monkeypatch)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), stage),
    ):
        copy_private_sqlite("recovery.core.chachanotes", source, candidate)
        with closing(
            sqlite3.connect(candidate.as_uri() + "?mode=ro", uri=True)
        ) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
            assert connection.execute("SELECT value FROM retained").fetchall() == [
                ("retained",)
            ]
        assert not any(
            candidate.with_name(candidate.name + suffix).exists()
            for suffix in ("-wal", "-shm", "-journal")
        )
    assert source.read_bytes() == before
    assert before[18:20] == b"\x02\x02"
