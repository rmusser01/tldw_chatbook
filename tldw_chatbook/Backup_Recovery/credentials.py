"""Managed credentials in disposable staged copies only (ADR-126).

The caller rebinds Inventory paths to captured files below staging. Source paths
are not resolved or inferred here. Arbitrary prose/logs/external files are not
claimed sanitized by these semantic policies.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import stat
import tomllib
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import toml

from tldw_chatbook.runtime_policy.server_credentials import RECOVERY_SETUP_REQUIRED
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

from .credential_policies import (
    AUTH_FIELDS,
    CITATION_OWNER,
    CITATION_SERVICE,
    CONFIG_OWNERS,
    GENERATION_KEYRINGS,
    HEADER_FIELDS,
    JSON_CONNECTION_OWNERS,
    REFERENCE_FIELDS,
    SQLITE_CREDENTIAL_COLUMNS,
    URL_FIELDS,
    URL_SECRET_PARAMETERS,
)
from .models import Inventory

_MAX_BYTES = 16 * 1024**2
_MATERIAL = "credential-recovery.json"


def _credential_store():
    from tldw_chatbook.runtime_policy.server_credentials import (
        build_default_server_credential_store,
    )

    return build_default_server_credential_store()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _read_scope(record, store):
    from tldw_chatbook.runtime_policy.server_credentials import ServerCredentialScope

    if record["kind"] == "server":
        return store.get_scoped_secret(
            ServerCredentialScope.legacy(record["server_id"], record["purpose"])
        )
    return store._keyring.get_password(record["service"], record["username"])


def _capture_record(record, material, issues, *, optional=False):
    if len(material) >= 10000:
        raise ValueError("credential_resource_limit")
    record["id"] = _fingerprint(record)
    try:
        value = _read_scope(record, _credential_store())
        if value is None:
            if optional:
                return
            record["status"] = "missing"
        elif not isinstance(value, str):
            record["status"] = "unreadable"
        else:
            if (
                record["kind"] == "citation"
                and len(base64.b64decode(value, validate=True)) != 32
            ):
                raise ValueError("invalid_citation_key")
            record.update(status="captured", value=value)
    except Exception:  # noqa: BLE001 - backend errors can contain secret values
        # Backends can include secret values in exception text. Only an opaque
        # coverage code leaves this boundary; no exception/log interpolation.
        record["status"] = "unreadable"
    if record["status"] != "captured":
        issues.append("credential_" + record["status"] + ":" + record["id"])
    elif not record["remappable"]:
        issues.append("credential_manual_recovery_required:" + record["id"])
    material.append(record)


def _config_server_id(data):
    # Match the installed legacy binding without importing/bootstraping config.
    from tldw_chatbook.MCP.unified_control_models import _normalize_server_identity

    api = data.get("tldw_api", {})
    if not isinstance(api, dict) or not api:
        raw = data.get("COMPREHENSIVE_CONFIG_RAW", {})
        api = raw.get("tldw_api", {}) if isinstance(raw, dict) else {}
    if (
        not isinstance(api, dict)
        or api.get("auth_reference") == RECOVERY_SETUP_REQUIRED
    ):
        return None
    value = str(
        api.get("base_url") or api.get("api_url") or api.get("url") or ""
    ).strip()
    return _normalize_server_identity(value)[0] if value else None


def _capture_owned(owner, path, data, staging, material, issues):
    relative = str(path.relative_to(staging))
    if owner == "mcp.targets":
        for target in _targets(data):
            reference = target.get("auth_reference", "") or ""
            if not isinstance(reference, str):
                raise TypeError("unsupported_credential_format")
            if reference == RECOVERY_SETUP_REQUIRED:
                continue
            purpose = (
                reference.removeprefix("keyring:")
                if reference.startswith("keyring:")
                else None
            )
            purposes = (
                (purpose,)
                if purpose
                else ("api_key", "bearer_token", "access_token", "refresh_token")
            )
            for selected in purposes:
                if selected not in {
                    "api_key",
                    "bearer_token",
                    "access_token",
                    "refresh_token",
                } and not re.fullmatch(
                    r"recovery_[0-9a-f]{32}_[A-Za-z0-9_.-]+", selected
                ):
                    issues.append("credential_scope_unsupported")
                    continue
                _capture_record(
                    {
                        "kind": "server",
                        "file": relative,
                        "server_id": target["server_id"],
                        "purpose": selected,
                        "remappable": purpose is not None,
                    },
                    material,
                    issues,
                )
    if owner in CONFIG_OWNERS:
        server_id = _config_server_id(data)
        if server_id:
            for purpose in ("api_key", "bearer_token", "access_token", "refresh_token"):
                _capture_record(
                    {
                        "kind": "server",
                        "file": relative,
                        "binding": "config",
                        "server_id": server_id,
                        "purpose": purpose,
                        "remappable": False,
                    },
                    material,
                    issues,
                )
        for section, service, names in GENERATION_KEYRINGS:
            values = data.get(section, {})
            if not isinstance(values, dict):
                raise TypeError("unsupported_credential_format")
            for name in names:
                _capture_record(
                    {
                        "kind": "generation",
                        "file": relative,
                        "service": service,
                        "username": name,
                        "remappable": False,
                    },
                    material,
                    issues,
                    optional=name not in values,
                )
        _capture_encrypted(data, relative, material, issues)


def _capture_encrypted(data, relative, material, issues):
    def walk(value, location=()):
        if len(location) > 64 or len(material) >= 10000:
            raise ValueError("credential_resource_limit")
        if isinstance(value, dict):
            for key, item in value.items():
                if key != "encryption":
                    walk(item, location + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, location + (index,))
        elif isinstance(value, str) and value.startswith("enc:"):
            from tldw_chatbook.Utils.config_encryption import unlock_recovery_value

            record = {
                "kind": "encrypted_config",
                "file": relative,
                "location": location,
                "remappable": False,
            }
            record["id"] = _fingerprint(record)
            try:
                record.update(status="captured", value=unlock_recovery_value(value))
            except (ValueError, TypeError):
                record["status"] = "locked"
                issues.append("credential_unlock_required:" + record["id"])
            material.append(record)

    walk(data)


def _material(staging):
    path = _staged_path(staging, Path(staging) / _MATERIAL)
    data = json.loads(_read(path))
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or data.get("mode") not in {"include", "rollback"}
        or not isinstance(data.get("records"), list)
    ):
        raise ValueError("unsupported_credential_material")
    records = data["records"]
    if len(records) > 10000:
        raise ValueError("credential_resource_limit")
    seen = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["id"])
            or record["id"] in seen
        ):
            raise ValueError("unsupported_credential_material")
        if (
            not isinstance(record.get("file"), str)
            or type(record.get("remappable")) is not bool
            or record.get("status")
            not in {"captured", "missing", "unreadable", "locked"}
        ):
            raise ValueError("unsupported_credential_material")
        if record.get("kind") not in {
            "server",
            "generation",
            "encrypted_config",
            "citation",
        }:
            raise ValueError("unsupported_credential_material")
        seen.add(record["id"])
        _staged_path(staging, Path(staging) / record["file"])
        if record["kind"] == "server":
            if (
                not isinstance(record.get("server_id"), str)
                or not record["server_id"]
                or not isinstance(record.get("purpose"), str)
                or not re.fullmatch(
                    r"(?:api_key|bearer_token|access_token|refresh_token|recovery_[0-9a-f]{32}_[A-Za-z0-9_.-]+)",
                    record["purpose"],
                )
            ):
                raise ValueError("unsupported_credential_material")
            if record.get("binding") == "config":
                if (
                    record["remappable"]
                    or _config_server_id(
                        tomllib.loads(_read(Path(staging) / record["file"]).decode())
                    )
                    != record["server_id"]
                ):
                    raise ValueError("credential_reference_changed")
            else:
                targets = _targets(json.loads(_read(Path(staging) / record["file"])))
                match = next(
                    (
                        row
                        for row in targets
                        if row.get("server_id") == record["server_id"]
                    ),
                    None,
                )
                if match is None or (
                    record["remappable"]
                    and match.get("auth_reference") != "keyring:" + record["purpose"]
                ):
                    raise ValueError("credential_reference_changed")
        elif record["kind"] == "generation":
            if not any(
                record["service"] == service and record["username"] in names
                for _, service, names in GENERATION_KEYRINGS
            ):
                raise ValueError("unsupported_credential_material")
        elif record["kind"] == "citation":
            if record.get("service") != CITATION_SERVICE or record["remappable"]:
                raise ValueError("unsupported_credential_material")
            from tldw_chatbook.DB.private_sqlite import open_recovery_validation

            from .sqlite_validation import _check, _installed_owner

            installed = _installed_owner(CITATION_OWNER)
            with open_recovery_validation(
                installed.owner_id, Path(staging) / record["file"], writable=False
            ) as source:
                if (
                    _check(source, installed, installed.schema_policy())[0]
                    or not source.execute(
                        "SELECT 1 FROM rag_identity_context WHERE fingerprint_key_id=?",
                        (record.get("username"),),
                    ).fetchone()
                ):
                    raise ValueError("credential_reference_changed")
            if record["status"] == "captured":
                try:
                    valid = (
                        len(base64.b64decode(record.get("value", ""), validate=True))
                        == 32
                    )
                except (ValueError, binascii.Error, TypeError):
                    valid = False
                if not valid:
                    raise ValueError("unsupported_credential_material")
        elif record["kind"] != "encrypted_config":
            raise ValueError("unsupported_credential_material")
        if record["status"] == "captured" and not isinstance(record.get("value"), str):
            raise ValueError("unsupported_credential_material")
    return records


def plan_credential_scopes(staging: Path) -> Mapping[str, str]:
    """Plan only: caller journals this mapping before applying any credential."""
    plans = {}
    for record in _material(staging):
        if record["status"] != "captured":
            continue
        if record["kind"] != "server" or not record["remappable"]:
            plans[record["id"]] = json.dumps(
                {"action": "retain", "material": _fingerprint(record)}
            )
            continue
        try:
            current = _read_scope(record, _credential_store())
        except Exception:  # noqa: BLE001 - backend errors can contain secret values
            raise ValueError("credential_store_unavailable") from None
        purpose = (
            record["purpose"]
            if current == record["value"]
            else "recovery_" + uuid4().hex + "_" + record["purpose"]
        )
        target = {**record, "purpose": purpose}
        try:
            occupied = (
                purpose != record["purpose"]
                and _read_scope(target, _credential_store()) is not None
            )
        except Exception:  # noqa: BLE001 - backend errors can contain secret values
            raise ValueError("credential_store_unavailable") from None
        if occupied:
            raise ValueError("credential_scope_collision")
        plans[record["id"]] = json.dumps(
            {
                "action": "reuse" if purpose == record["purpose"] else "create",
                "purpose": purpose,
                "expected": _fingerprint(current),
                "material": _fingerprint(record),
            },
            sort_keys=True,
        )
    return MappingProxyType(plans)


def restore_credential_values(
    staging: Path, scope_map: Mapping[str, str]
) -> tuple[str, ...]:
    """Recheck the journaled plan; isolate supported refs, retain other material."""
    from tldw_chatbook.runtime_policy.server_credentials import ServerCredentialScope

    try:
        records = _material(staging)
        captured = {
            record["id"]: record for record in records if record["status"] == "captured"
        }
        if set(scope_map) != set(captured):
            raise ValueError("credential_scope_changed")
        plans = {key: json.loads(value) for key, value in scope_map.items()}

        def check(record, plan):
            if not isinstance(plan, dict) or plan.get("material") != _fingerprint(
                record
            ):
                raise ValueError("credential_scope_changed")
            if plan.get("action") == "retain":
                return
            if record["kind"] != "server" or not record["remappable"]:
                raise ValueError("credential_scope_changed")
            if plan.get("action") == "reuse":
                if plan.get("purpose") != record["purpose"]:
                    raise ValueError("credential_scope_changed")
            elif plan.get("action") == "create":
                if not re.fullmatch(
                    "recovery_[0-9a-f]{32}_" + re.escape(record["purpose"]),
                    plan.get("purpose", ""),
                ):
                    raise ValueError("credential_scope_changed")
            else:
                raise ValueError("credential_scope_changed")
            current = _read_scope(record, _credential_store())
            if _fingerprint(current) != plan.get("expected") or (
                plan["action"] == "reuse" and current != record["value"]
            ):
                raise ValueError("credential_scope_changed")
            if (
                plan["action"] == "create"
                and _read_scope(
                    {**record, "purpose": plan["purpose"]}, _credential_store()
                )
                is not None
            ):
                raise ValueError("credential_scope_changed")

        for key, record in captured.items():
            check(record, plans[key])
        issues = [
            "credential_" + record["status"] + ":" + record["id"]
            for record in records
            if record["status"] != "captured"
        ]
        for key, record in captured.items():
            plan = plans[key]
            if plan["action"] == "retain":
                issues.append("credential_manual_recovery_required:" + key)
                continue
            check(record, plan)
            if plan["action"] == "create":
                _credential_store().set_recovery_secret_if_absent(
                    ServerCredentialScope.legacy(record["server_id"], plan["purpose"]),
                    record["value"],
                )
                path = Path(staging) / record["file"]
                data = json.loads(_read(path))
                for target in _targets(data):
                    if target["server_id"] == record["server_id"]:
                        target["auth_reference"] = "keyring:" + plan["purpose"]
                _write(path, json.dumps(data))
        return tuple(issues)
    except Exception:  # noqa: BLE001 - backend errors can contain secret values
        return ("credential_scope_apply_unavailable",)


def _staged_path(staging, path):
    staging, path = Path(staging), Path(path)
    if not staging.is_absolute() or staging.resolve() != staging:
        raise ValueError("credential_staging_required")
    root_info = staging.stat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or stat.S_IMODE(root_info.st_mode) & 0o077
    ):
        raise ValueError("credential_staging_required")
    path.relative_to(staging)
    if path.resolve() != path or path == staging:
        raise ValueError("credential_staging_required")
    for parent in (staging, *path.relative_to(staging).parents):
        selected = parent if parent.is_absolute() else staging / parent
        if selected.is_symlink():
            raise ValueError("credential_staging_required")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
    ):
        raise ValueError("credential_staging_required")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("credential_staging_required")
    return path


def _read(path):
    from tldw_chatbook.Utils.private_paths import open_private_binary

    from .storage_admission import _read_staged_credential_file

    captured = _read_staged_credential_file(path, max_bytes=_MAX_BYTES)
    if captured is not None:
        return captured
    with open_private_binary(path) as opened:
        if not opened.result.verified_private:
            raise ValueError("credential_staging_required")
        data = opened.stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("credential_resource_limit")
    return data


def _write(path, data):
    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    from .storage_admission import _write_staged_credential_file

    if _write_staged_credential_file(path, data):
        return
    result = atomic_private_write_text(path, data)
    if not result.verified_private:
        raise ValueError("credential_staging_required")


def _sanitize_url(value):
    if not isinstance(value, str):
        raise TypeError("unsupported_credential_format")
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.netloc.rsplit("@", 1)[-1]
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not is_sensitive_config_key(key) and key.lower() not in URL_SECRET_PARAMETERS
    ]
    return urlunsplit(
        (parsed.scheme, host, parsed.path, urlencode(query), parsed.fragment)
    )


def sanitize_config(config: Mapping[str, object]) -> dict[str, object]:
    """Copy known managed config, removing values/blobs/refs and keeping hints."""

    def walk(value, depth=0):
        if depth > 64:
            raise ValueError("credential_depth_limit")
        if isinstance(value, list):
            return [walk(item, depth + 1) for item in value]
        if not isinstance(value, Mapping):
            return "" if isinstance(value, str) and value.startswith("enc:") else value
        result = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("unsupported_credential_format")
            lower = key.lower()
            if lower == "encryption":
                result[key] = {"enabled": False}
            elif is_sensitive_config_key(key) or lower in REFERENCE_FIELDS:
                continue
            elif lower in HEADER_FIELDS or lower == "cookies":
                if not isinstance(child, Mapping):
                    raise ValueError("unsupported_credential_format")
                result[key] = {}
            elif lower in AUTH_FIELDS:
                if not isinstance(child, Mapping):
                    raise ValueError("unsupported_credential_format")
                result[key] = {
                    name: item
                    for name, item in child.items()
                    if name in {"type", "header", "api_key_env_var"}
                    and isinstance(item, str)
                }
            elif lower in URL_FIELDS:
                result[key] = _sanitize_url(child)
            else:
                result[key] = walk(child, depth + 1)
                if lower == "tldw_api" and isinstance(result[key], dict):
                    result[key]["auth_reference"] = RECOVERY_SETUP_REQUIRED
                for section, _service, backends in GENERATION_KEYRINGS:
                    if lower == section:
                        if not isinstance(result[key], dict):
                            raise TypeError("unsupported_credential_format")
                        for backend in backends:
                            configured = result[key].setdefault(backend, {})
                            if not isinstance(configured, dict):
                                raise TypeError("unsupported_credential_format")
                            configured["auth_reference"] = RECOVERY_SETUP_REQUIRED
        return result

    if not isinstance(config, Mapping):
        raise TypeError("unsupported_credential_format")
    return walk(config)


def _targets(data):
    targets = (
        data
        if isinstance(data, list)
        else data.get("targets")
        if isinstance(data, dict)
        else None
    )
    if not isinstance(targets, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("server_id"), str)
        for row in targets
    ):
        raise ValueError("unsupported_credential_format")
    return targets


def _sanitize_connections(owner, data):
    if owner == "mcp.local":
        profiles = data.get("profiles", []) if isinstance(data, dict) else None
        if not isinstance(profiles, list) or any(
            not isinstance(row, dict) or row.get("args") for row in profiles
        ):
            # Arbitrary external command argument grammars have no installed
            # credential policy. Explicit owner omission is required here.
            raise ValueError("unsupported_credential_format")
        result = sanitize_config(data)
        for row in result.get("profiles", []):
            for field in ("env", "env_literals", "legacy_env_literals"):
                if field in row:
                    row[field] = {}
        return result
    if owner == "mcp.targets":
        # Do not remove this marker: a missing reference invokes the resolver's
        # shared server-id fallback. This marker denotes setup, not a live scope.
        result = [sanitize_config(row) for row in _targets(data)]
        for row in result:
            row["auth_reference"] = RECOVERY_SETUP_REQUIRED
        if isinstance(data, list):
            return result
        return {**sanitize_config(data), "targets": result}
    return sanitize_config(data)


def _sanitize_column(value, kind):
    if kind == "citation":
        return RECOVERY_SETUP_REQUIRED
    if value is None or value == "":
        return value
    if kind == "url":
        return _sanitize_url(value)
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError("unsupported_credential_format")
    if kind == "headers":
        return "{}"
    if kind == "auth":
        decoded = sanitize_config({"auth": decoded})["auth"]
    else:
        decoded = sanitize_config(decoded)
    return json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))


def _rewrite_database(staging, path, owner_id, *, export=None):
    from tldw_chatbook.DB.private_sqlite import (
        connect_private_sqlite,
        open_recovery_validation,
    )

    from .sqlite_validation import _check, _installed_owner, validate_candidate

    installed = _installed_owner(owner_id)
    policy = installed.schema_policy()
    rules = {
        (rule.table, rule.column): rule.kind
        for rule in SQLITE_CREDENTIAL_COLUMNS
        if rule.owner == owner_id
    }
    destination = staging / (".credential-" + uuid4().hex + ".sqlite")
    try:
        with open_recovery_validation(
            installed.owner_id, path, writable=False
        ) as source:
            issues, version = _check(source, installed, policy)
            if issues:
                raise ValueError("unsupported_credential_schema")
            if export is not None:
                material, coverage = export
                for (table, column), kind in rules.items():
                    # Identifiers come only from the frozen installed policy.
                    for rowid, value in source.execute(
                        f'SELECT rowid,"{column}" FROM "{table}"'  # nosec B608 - installed policy identifiers only
                    ):
                        if kind == "citation":
                            if not isinstance(value, str) or not value:
                                raise TypeError("unsupported_credential_format")
                            if value != RECOVERY_SETUP_REQUIRED:
                                _capture_record(
                                    {
                                        "kind": "citation",
                                        "file": str(path.relative_to(staging)),
                                        "service": CITATION_SERVICE,
                                        "username": value,
                                        "remappable": False,
                                    },
                                    material,
                                    coverage,
                                )
                        elif value and kind != "url":
                            decoded = json.loads(value)
                            if not isinstance(decoded, dict):
                                raise TypeError("unsupported_credential_format")
                            _capture_encrypted(
                                {table: {str(rowid): {column: decoded}}},
                                str(path.relative_to(staging)),
                                material,
                                coverage,
                            )
                return
            actual_sql = tuple(
                row[0]
                for row in source.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
            schema = next(
                sql
                for known, sql in policy.schema_sql
                if known == version and sql == actual_sql
            )
            with closing(
                connect_private_sqlite("recovery.credentials", destination)
            ) as output:
                output.execute("PRAGMA trusted_schema=OFF")
                output.execute("BEGIN")
                tables = [
                    sql
                    for sql in schema
                    if sql.upper().startswith(("CREATE TABLE", "CREATE VIRTUAL TABLE"))
                ]
                for sql in tables:
                    if (
                        output.execute(
                            "SELECT 1 FROM sqlite_schema WHERE sql=?", (sql,)
                        ).fetchone()
                        is None
                    ):
                        output.execute(sql)
                layouts = {
                    row[1]: (row[2], row[4])
                    for row in source.execute("PRAGMA table_list")
                }
                names = [
                    row[0]
                    for row in source.execute(
                        "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name='sqlite_sequence',name"
                    )
                ]

                def quote(name):
                    return '"' + name.replace('"', '""') + '"'

                for name in names:
                    kind, without_rowid = layouts[name]
                    if kind == "virtual":
                        continue
                    columns = [
                        row[1]
                        for row in source.execute(f"PRAGMA table_xinfo({quote(name)})")
                        if row[6] == 0
                    ]
                    # Explicit rowid copy preserves external-content FTS and any
                    # semantic hidden row identities. Do not blindly VACUUM.
                    copied = columns if without_rowid else ["rowid", *columns]
                    selectors = ",".join(quote(column) for column in copied)
                    output.execute(f"DELETE FROM {quote(name)}")  # nosec B608 - exact validated installed catalog
                    for row in source.execute(f"SELECT {selectors} FROM {quote(name)}"):  # nosec B608 - exact validated installed catalog
                        values = tuple(
                            _sanitize_column(value, rules[(name, column)])
                            if (name, column) in rules
                            else value
                            for column, value in zip(copied, row)
                        )
                        output.execute(
                            f"INSERT INTO {quote(name)} ({selectors}) VALUES ({','.join('?' for _ in values)})",  # nosec B608 - validated identifiers, bound row values
                            values,
                        )
                for sql in schema:
                    if sql not in tables:
                        output.execute(sql)
                output.execute(
                    f"PRAGMA user_version={source.execute('PRAGMA user_version').fetchone()[0]}"
                )
                output.commit()
        if validate_candidate(installed, destination, Event(), migrate=False):
            raise ValueError("credential_reconstruction_failed")
        _staged_path(staging, path)
        os.replace(destination, path)
    finally:
        destination.unlink(missing_ok=True)


def process_credentials(
    staging: Path,
    inventory: Inventory,
    *,
    mode: Literal["exclude", "include", "rollback"],
    encrypted: bool,
) -> tuple[str, ...]:
    """Process a rebound staging inventory; report unsupported coverage honestly."""
    if mode in {"include", "rollback"} and not encrypted:
        raise ValueError("credentials_require_encryption")
    if mode not in {"include", "rollback", "exclude"}:
        raise ValueError("invalid_credential_mode")
    try:
        staging = Path(staging)
        if (staging / _MATERIAL).exists():
            raise ValueError("credential_material_already_present")
        candidates = []
        issues = []
        for item in inventory.items:
            if (
                mode != "exclude"
                and item.status == "included"
                and item.owner == "skills"
                and item.metadata
                and item.metadata.kind == "file"
                and Path(item.metadata.relative_path).parts[:1] == ("trust",)
            ):
                # These disk files contain historical trust metadata/skill content,
                # not the derived keys held only in the optional keyring cache.
                # Including credentials must disclose that cache's omission.
                issues.append(
                    "credential_skill_trust_manual_unlock_required:" + item.logical_id
                )

        sqlite_owners = {rule.owner for rule in SQLITE_CREDENTIAL_COLUMNS}
        for item in inventory.items:
            if (
                item.status != "included"
                or item.owner
                not in CONFIG_OWNERS
                | JSON_CONNECTION_OWNERS
                | sqlite_owners
                | {"rag.definitions"}
            ):
                continue
            if item.owner == "rag.definitions":
                relative = Path(item.metadata.relative_path) if item.metadata else None
                if (
                    relative is None
                    or len(relative.parts) != 1
                    or relative.suffix != ".json"
                    or relative.name
                    in {"custom_profiles.json", "custom_profiles.json.migrated"}
                ):
                    issues.append(
                        "credential_rag_definition_format_unsupported:"
                        + item.logical_id
                    )
                    continue
            path = _staged_path(staging, item.path)
            if any(
                Path(str(path) + suffix).exists()
                for suffix in ("-wal", "-shm", "-journal")
            ):
                raise ValueError("credential_sidecar_unprocessed")
            candidates.append((item, path))
        seen = set()
        material = []
        for item, path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if item.owner in sqlite_owners:
                if mode == "exclude":
                    _rewrite_database(staging, path, item.owner)
                else:
                    _rewrite_database(
                        staging, path, item.owner, export=(material, issues)
                    )
                continue
            try:
                data = (
                    tomllib.loads(_read(path).decode())
                    if item.owner in CONFIG_OWNERS
                    else json.loads(_read(path))
                )
            except (ValueError, UnicodeError):
                if mode != "rollback":
                    raise
                issues.append("credential_format_unreadable")
                continue
            if item.owner == "rag.definitions" and (
                not isinstance(data, dict)
                or not isinstance(data.get("rag_config"), dict)
            ):
                issues.append(
                    "credential_rag_definition_format_unsupported:" + item.logical_id
                )
                continue
            if mode == "exclude":
                if item.owner in CONFIG_OWNERS:
                    # These resolvers consult their fixed slots even when a
                    # backend subsection is absent from the source config.
                    if not isinstance(data, dict):
                        raise TypeError("unsupported_credential_format")
                    for section, _service, _backends in GENERATION_KEYRINGS:
                        data.setdefault(section, {})
                sanitized = (
                    sanitize_config(data)
                    if item.owner in CONFIG_OWNERS
                    else _sanitize_connections(item.owner, data)
                )
                _write(
                    path,
                    toml.dumps(sanitized)
                    if item.owner in CONFIG_OWNERS
                    else json.dumps(sanitized),
                )
            else:
                if item.owner == "mcp.local":
                    try:
                        _sanitize_connections(item.owner, data)
                    except (ValueError, TypeError):
                        issues.append("credential_connection_format_unsupported")
                if item.owner == "rag.definitions":
                    _capture_encrypted(
                        data, str(path.relative_to(staging)), material, issues
                    )
                else:
                    _capture_owned(item.owner, path, data, staging, material, issues)
        if mode != "exclude":
            encoded = json.dumps(
                {"version": 1, "mode": mode, "records": material}, ensure_ascii=False
            )
            if len(encoded.encode()) > _MAX_BYTES:
                raise ValueError("credential_resource_limit")
            _write(staging / _MATERIAL, encoded)
        return tuple(issues)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return ("credential_processing_unavailable",)
