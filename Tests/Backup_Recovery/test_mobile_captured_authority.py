"""Raw reference proof is issued only by an actual authenticated native capture."""

import json
import sqlite3
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


@pytest.mark.parametrize("distinct_selectors", (False, True))
def test_captured_shared_raw_owners_prove_exact_references_without_writes(
    tmp_path, monkeypatch, distinct_selectors
):
    from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

    with _capture_fixture(
        tmp_path, monkeypatch, distinct_selectors=distinct_selectors
    ) as values:
        session, journal, config, _, previous, _, _ = values
        data = user_data_dir(read_profile_config(config))
        persona_path = data / "tldw_chatbook_personas.json"
        permission_path = data / "mcp_permissions.json"
        persona_path.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "id": "persona-1",
                            "name": "Private fixture",
                            "is_active": True,
                            "policy_rules": [],
                        },
                        {
                            "id": "malformed-persona",
                            "version": "private-invalid-version",
                        },
                        {"id": "inactive-persona", "is_active": False},
                    ]
                }
            )
        )
        from Tests.MCP.test_permission_resolution import _valid_lifecycle

        permission_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kill_switch": False,
                    "profiles": {
                        "default": {"global_default": "ask", "servers": {}},
                        "profile-1": {"servers": {}},
                        "removed-profile": {
                            "profile_kind": "tool_pack_tombstone",
                            "tool_pack_lifecycle": _valid_lifecycle(origin="tombstone"),
                            "servers": {},
                        },
                    },
                }
            )
        )
        # Real Workspace references are captured with the raw owners they depend on.
        from tldw_chatbook.Backup_Recovery.profile_paths import legacy_mobile_dir

        with sqlite3.connect(
            legacy_mobile_dir(read_profile_config(config)) / "workspaces.db"
        ) as connection:
            connection.execute(
                "UPDATE workspace_records SET assistant_defaults=? WHERE workspace_id='conflicting-workspace'",
                (
                    json.dumps(
                        {
                            "assistant_kind": "persona",
                            "assistant_id": "persona-1",
                            "persona_memory_mode": "read_only",
                            "tool_policy_profile_id": "profile-1",
                        }
                    ),
                ),
            )
            connection.execute(
                "INSERT INTO workspace_agent_backfill VALUES ('assistant_defaults','2026-10-25')"
            )
        connection.close()
        persona_path.chmod(0o600)
        permission_path.chmod(0o600)
        # Seed only test-owned raw stores before this operation's actual review.
        with session._discovery_reads(private_sqlite=True):
            inventory = discover(
                tuple(dict.fromkeys((config, previous.target_selector)))
            )
            preview = _preview_selected_mobile(config, previous.target_selector)
        assert inventory.complete
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                workspace = reads.workspace_plan()
                assert (
                    not workspace.blockers
                )  # Referenced defaults are backfill-exempt.
                assert {
                    (proof.kind, proof.reference_id)
                    for proof in workspace.authority_references
                } == {
                    ("persona", "persona-1"),
                    ("permission_profile", "profile-1"),
                }
                record = next(
                    dict(row.values)
                    for row in workspace.rows
                    if row.table == "workspace_records"
                )
                from tldw_chatbook.Workspaces.registry_service import (
                    _workspace_from_row,
                )

                assert (
                    _workspace_from_row(record).assistant_defaults.assistant_id
                    == "persona-1"
                )
                persona = reads.authority_reference("persona", "persona-1")
                profile = reads.authority_reference("permission_profile", "profile-1")
                assert persona.reference_id == "persona-1"
                assert profile.reference_id == "profile-1"
                assert persona.owner_id == "personas"
                assert profile.owner_id == "mcp.permissions"
                assert len(profile.member_digest) == 64
                assert len(profile.provenance_digest) == 64
                with pytest.raises(ValueError, match="lifecycle_blocked"):
                    reads.authority_reference("permission_profile", "removed-profile")
                original_permissions = permission_path.read_bytes()
                try:
                    changed = json.loads(original_permissions)
                    changed["profiles"]["default"]["global_default"] = "allow"
                    permission_path.write_text(json.dumps(changed))
                    with pytest.raises(ValueError, match="source_coverage_unproven"):
                        reads.authority_reference("permission_profile", "profile-1")
                    changed = json.loads(original_permissions)
                    changed["kill_switch"] = True
                    permission_path.write_text(json.dumps(changed))
                    with pytest.raises(ValueError, match="source_coverage_unproven"):
                        reads.authority_reference("permission_profile", "profile-1")
                finally:
                    permission_path.write_bytes(original_permissions)

                assert "Private fixture" not in repr(persona)
                with pytest.raises(
                    ValueError, match="mobile_authority_policy_invalid"
                ) as invalid:
                    reads.authority_reference("persona", "malformed-persona")
                assert "private-invalid-version" not in str(invalid.value)
                with pytest.raises(ValueError, match="persona_unavailable"):
                    reads.authority_reference("persona", "inactive-persona")
                with pytest.raises(FrozenInstanceError):
                    persona.reference_id = "different"
                with pytest.raises(ValueError, match="reference_missing"):
                    reads.authority_reference("persona", "missing-persona")
                with pytest.raises(ValueError, match="reference_missing"):
                    reads.authority_reference("permission_profile", "missing-profile")
            assert _fingerprints(tmp_path / "profile") == before
            with pytest.raises(ValueError, match="inactive"):
                reads.authority_reference("persona", "persona-1")


@pytest.mark.parametrize(
    "payload,error_name",
    (
        (b"[general]\nflag=true\n[plugin]\nsetting=7\n", None),
        (b"[general\n", "TOMLDecodeError"),
        (b"general=7\n", "ValidationError"),
        (b"__reserved__={}\n", "ValueError"),
        (b"\xff", "UnicodeDecodeError"),
    ),
)
def test_owned_config_parser_preserves_native_selector_decoding(
    tmp_path, payload, error_name
):
    import tomllib

    from pydantic import ValidationError

    from tldw_chatbook.Backup_Recovery.inventory import (
        _parse_profile_config,
        read_profile_config,
    )
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY

    if payload == b"__reserved__={}\n":
        payload = (DISCOVERY_CONTEXT_KEY + "={}\n").encode()
    selected = tmp_path / "selected.toml"
    selected.write_bytes(payload)
    errors = {
        "TOMLDecodeError": tomllib.TOMLDecodeError,
        "ValidationError": ValidationError,
        "ValueError": ValueError,
        "UnicodeDecodeError": UnicodeDecodeError,
    }
    for read, value in (
        (read_profile_config, selected),
        (_parse_profile_config, payload),
    ):
        if error_name is not None:
            with pytest.raises(errors[error_name]):
                read(value)
        else:
            result = read(value)
            assert result["general"] == {"flag": True}
            assert result["plugin"] == {"setting": 7}
            assert result["paths"] == {}
