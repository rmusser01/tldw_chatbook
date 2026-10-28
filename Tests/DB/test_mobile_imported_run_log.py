"""Enrolled runtime readers over committed historical fixture receipts.

These are reader fixtures. Actual import producer commits are independently
exercised by test_mobile_owning_transactions; these do not replace that proof.
"""

import hashlib
import json
from contextlib import contextmanager

import pytest


@contextmanager
def _history(
    tmp_path, monkeypatch, *, root_frame=True, enrolled=True, numeric_alias=False
):
    from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import (
        canonical_console_paths,
        user_data_dir,
    )
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.console_command_history import (
        _commit_command_history,
        _prepare_command_history,
    )
    from tldw_chatbook.DB.mobile_consolidation_receipts import _encode_owner_receipt

    control, base, config = (
        tmp_path / "bootstrap",
        tmp_path / "data",
        tmp_path / "config.toml",
    )
    base.mkdir(mode=0o700)
    config.write_text("[paths]\ndata_dir = " + json.dumps(str(base)) + "\n")
    authority = admission_authority(control)
    authority.register("profile", (config, base))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: control)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    if enrolled:
        bind_profile(control, config, ("profile",), control / "admission")
    values = read_profile_config(config)
    stores = CanonicalConsoleStores(*canonical_console_paths(values))
    stores.run_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner = ConsoleExecutionOwner.acquire(stores.run_path)
    db = None
    try:
        db = AgentRunsDB(stores.run_path, execution_owner=owner)
        root = db.create_run(conversation_id="retained-chat", agent_kind="primary")
        child = db.create_run(
            conversation_id="retained-chat", agent_kind="subagent", parent_run_id=root
        )
        historical = db.create_run(
            conversation_id="retained-chat", agent_kind="primary"
        )
        for identity in (root, child, historical):
            db.set_status(identity, "done", result="retained result")
        source, operation, root_id = "c" * 64, "committed-history", "original-root"
        namespace = (
            user_data_dir(values)
            / "agent_history"
            / "imports"
            / stores.identity
            / source
        )
        component = hashlib.sha256(root_id.encode()).hexdigest()
        imported_root = namespace / component
        log_dir = imported_root / "old-root"
        log_dir.mkdir(mode=0o700, parents=True)
        records = [
            RunLogRecord(
                1,
                "old-child",
                "subagent",
                "model",
                "native-clock",
                "Child private frame Ω",
            )
        ]
        if root_frame:
            records.insert(
                0,
                RunLogRecord(
                    0,
                    "old-root",
                    "primary",
                    "model",
                    "native-clock",
                    "Root private frame Ω",
                ),
            )
        (log_dir / "logs.0.txt").write_bytes(
            b"".join(encode_record(record) for record in records)
        )
        if numeric_alias:
            (log_dir / "logs.00.txt").write_bytes((log_dir / "logs.0.txt").read_bytes())
        (log_dir / "MANIFEST").write_text(
            json.dumps(
                {
                    "run_id": "old-root",
                    "superseded_run_id": "old-historical",
                    "segments": ["logs.0.txt"]
                    + (["logs.00.txt"] if numeric_alias else []),
                }
            )
        )
        for path in log_dir.iterdir():
            path.chmod(0o600)

        def entry(kind, data):
            return {"version": 1, "kind": kind, "data": data}

        entries = [
            entry("identity", {"domain": "run", "original": old, "mapped": new})
            for old, new in (
                ("old-root", root),
                ("old-child", child),
                ("old-historical", historical),
            )
        ]
        entries.append(
            entry(
                "provenance",
                {
                    "family": "snapshot_allocator",
                    "record": {
                        "source_sequence": None,
                        "canonical_sequence": None,
                        "required_frontier": 0,
                    },
                },
            )
        )
        entries.append(
            entry(
                "runlog_root",
                {
                    "root_id": root_id,
                    "location": component,
                    "artifact": observe_artifact(imported_root),
                },
            )
        )
        for relative in ("", "old-root", "old-root/MANIFEST", "old-root/logs.0.txt") + (
            ("old-root/logs.00.txt",) if numeric_alias else ()
        ):
            path = imported_root / relative
            raw = path.read_bytes() if path.is_file() else None
            framed = (
                [["old-child", child]] + ([["old-root", root]] if root_frame else [])
                if relative.endswith(".txt")
                else []
            )
            entries.append(
                entry(
                    "runlog",
                    {
                        "logical_id": root_id
                        if relative == ""
                        else "member:" + relative,
                        "root_id": root_id,
                        "relative_path": relative,
                        "location": component
                        if relative == ""
                        else component + "/" + relative,
                        "size": len(raw) if raw is not None else 0,
                        "sha256": hashlib.sha256(raw).hexdigest()
                        if raw is not None
                        else None,
                        "owner": ["old-root", root] if raw is not None else None,
                        "framed_runs": framed,
                        "historical_run_ids": ["old-historical"]
                        if relative.endswith("MANIFEST")
                        else [],
                        "segment_ids": (
                            ["member:old-root/logs.0.txt"]
                            + (["member:old-root/logs.00.txt"] if numeric_alias else [])
                        )
                        if relative.endswith("MANIFEST")
                        else [],
                        "artifact": observe_artifact(path),
                    },
                )
            )
        entries.append(
            entry(
                "history_publication",
                {
                    "destination_identity": stores.identity,
                    "source_fingerprint": source,
                    "location": str(namespace),
                    "artifact": observe_artifact(namespace.parent),
                },
            )
        )
        budget = _TraceReadBudget()
        encoded = _encode_owner_receipt(
            "runs",
            entries,
            source_manifest_sha256="a" * 64,
            canonical_manifest_sha256="b" * 64,
            source_runs_version=25,
            command_history_count=0,
            budget=budget,
        )
        passive = _prepare_command_history(operation, source, (), budget)
        with db.transaction() as conn:
            db.hold_imported_authority(
                conn,
                operation,
                [("run", identity) for identity in (root, child, historical)],
            )
            _commit_command_history(db, conn, operation, source, passive)
            for key, raw in encoded:
                conn.execute(
                    "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
                    (operation, source, key, raw, "2026-10-27T00:00:00Z"),
                )
        yield db, root, child, historical, operation, source, namespace, log_dir
    finally:
        try:
            if db is not None:
                db.close()
        finally:
            owner.close()


@pytest.mark.parametrize("root_frame", [True, False])
def test_enrolled_committed_reader_filters_original_ids_and_ignores_new_sibling(
    tmp_path, monkeypatch, root_frame
):
    with _history(tmp_path, monkeypatch, root_frame=root_frame) as (
        db,
        root,
        child,
        historical,
        operation,
        source,
        namespace,
        _log_dir,
    ):
        # A later import changes the publication ancestor's digest, not this root.
        sibling = namespace.parent / "later-independent-import"
        sibling.mkdir(mode=0o700)
        (sibling / "evidence").write_text("later history")
        page = db.read_imported_run_log_page(child, operation, source)
        assert [part.record.run_id for part in page.slices] == ["old-child"]
        assert (
            "".join(part.record.content for part in page.slices)
            == "Child private frame Ω"
        )
        own = db.read_imported_run_log_page(root, operation, source)
        assert [part.record.run_id for part in own.slices] == (
            ["old-root"] if root_frame else []
        )
        with pytest.raises(ValueError, match="membership"):
            db.read_imported_run_log_page(historical, operation, source)


def test_runtime_reader_requires_actual_enrollment(tmp_path, monkeypatch):
    with _history(tmp_path, monkeypatch, enrolled=False) as (  # noqa: SIM117 -- Keep the explicit native resource/fault lifetime boundary.
        db,
        _root,
        child,
        _historical,
        operation,
        source,
        _namespace,
        _log_dir,
    ):
        with pytest.raises((ValueError, RuntimeError), match="admi|enroll"):
            db.read_imported_run_log_page(child, operation, source)


def test_runtime_reader_rejects_byte_identical_replaced_member(tmp_path, monkeypatch):
    with _history(tmp_path, monkeypatch) as (
        db,
        _root,
        child,
        _historical,
        operation,
        source,
        _namespace,
        log_dir,
    ):
        path = log_dir / "logs.0.txt"
        replacement = log_dir / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(path)
        with pytest.raises(ValueError, match="witness|changed"):
            db.read_imported_run_log_page(child, operation, source)


def test_native_numeric_segment_alias_refuses_before_pager(tmp_path, monkeypatch):
    with _history(tmp_path, monkeypatch, numeric_alias=True) as (
        db,
        root,
        _child,
        _historical,
        operation,
        source,
        _namespace,
        _log_dir,
    ):
        from tldw_chatbook.Agents import run_log_paging

        def no_page(*args, **kwargs):
            pytest.fail("ambiguous native numeric segments reached the pager")

        monkeypatch.setattr(run_log_paging, "_load_page", no_page)
        with pytest.raises(ValueError, match="segment"):
            db.read_imported_run_log_page(root, operation, source)
