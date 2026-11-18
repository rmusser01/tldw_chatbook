"""Synthetic native-barrier test data; never use a real profile or keyring."""

import json
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "packages/tldw_profile_core"


def v2_fixture(name: str) -> dict:
    return json.loads((CORE_ROOT / "fixtures/v2" / f"{name}.json").read_text())


def sql_state(repository):
    from contextlib import closing

    from tldw_chatbook.DB.sql_validation import validate_identifier

    with closing(repository._connect()) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        assert all(validate_identifier(name, "table") for name in names)
        return tuple(
            (
                name,
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        f'SELECT * FROM "{name}" ORDER BY rowid'
                    )
                ),
            )
            for name in names
        )


def install_v2_manifest(repository):
    from tldw_profile_core.v2_contract import canonical_v2_bytes, validate_v2_object

    from tldw_chatbook.Personal_Context.crypto import EnvelopeCipher

    manifest = repository.get_manifest()
    body = v2_fixture("01-manifest")["data"]
    body.update(
        profile_id=manifest.profile_id,
        current_version_id="blocked-manifest-v2",
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        revision=manifest.revision,
        purge_generation=manifest.purge_generation,
    )
    valid = validate_v2_object(body)
    raw = canonical_v2_bytes(valid)
    keys = repository._require_keys()
    aad = repository._aad("manifest", valid.profile_id, valid.current_version_id)
    sealed = EnvelopeCipher(keys.encryption_key, key_version=keys.key_version).encrypt(
        raw, aad
    )
    with repository._transaction() as connection:
        connection.execute(
            "INSERT INTO encrypted_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "manifest",
                valid.profile_id,
                valid.current_version_id,
                None,
                0,
                sealed.algorithm,
                sealed.nonce,
                sealed.ciphertext,
                sealed.wrap_nonce + sealed.wrapped_dek,
                sealed.key_version,
                repository._integrity_tag(keys.integrity_key, aad, raw),
                valid.updated_at.isoformat(),
            ),
        )
        connection.execute(
            "UPDATE object_heads SET version_id=? WHERE object_type='manifest' AND object_id=?",
            (valid.current_version_id, valid.profile_id),
        )
        connection.execute(
            "UPDATE profile_meta SET current_manifest_version=? WHERE singleton=1",
            (valid.current_version_id,),
        )
    return valid


def blocked_repository(tmp_path, protector, record_factory):
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(tmp_path / "profile.db", key_protector=protector)
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id, value="native-body-canary")
    repo.commit_record_version(record, expected_version_id=None)
    assert repo.get_record(record.record_id) == record
    install_v2_manifest(repo)
    return repo, record


def replace_sealed_body(repository, *, object_type, object_id, version_id, raw):
    """Inject synthetic authenticated legacy damage without production ingress."""
    from tldw_chatbook.Personal_Context.crypto import EnvelopeCipher

    keys = repository._require_keys()
    aad = repository._aad(object_type, object_id, version_id)
    sealed = EnvelopeCipher(keys.encryption_key, key_version=keys.key_version).encrypt(
        raw, aad
    )
    with repository._transaction() as connection:
        changed = connection.execute(
            "UPDATE encrypted_objects SET algorithm=?,nonce=?,ciphertext=?,wrapped_dek=?,key_version=?,integrity_tag=? "
            "WHERE object_type=? AND object_id=? AND version_id=?",
            (
                sealed.algorithm,
                sealed.nonce,
                sealed.ciphertext,
                sealed.wrap_nonce + sealed.wrapped_dek,
                sealed.key_version,
                repository._integrity_tag(keys.integrity_key, aad, raw),
                object_type,
                object_id,
                version_id,
            ),
        )
        assert changed.rowcount == 1
