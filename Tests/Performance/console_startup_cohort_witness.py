"""Diagnostic-only pytest plugin for the original control process startup cohorts.

Observe original code only, global monitoring mask 0. No leases/guards replaced,
no private registry cleared, no deadline changes, no App launch by this module.
Load before collection; receipt uses hashed root spellings and metadata only.
"""

from __future__ import annotations

import hashlib
import platform
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time

from Tests.Performance.console_startup_cohort_code_local import StartupIOCensus


_SPECS = (
    (
        "tldw_chatbook.Backup_Recovery.storage_admission",
        "StorageLease.__init__",
        None,
        "lease_creation",
    ),
    (
        "tldw_chatbook.Backup_Recovery.storage_admission",
        "StorageLease.close",
        None,
        "lease_close",
    ),
    (
        "tldw_chatbook.Backup_Recovery.storage_admission",
        "_LocalPause.drain",
        None,
        "pause_drain",
    ),
    (
        "tldw_chatbook.Backup_Recovery.storage_admission",
        "_admit_startup",
        None,
        "startup_admission",
    ),
    (
        "tldw_chatbook.Backup_Recovery.qualification",
        "qualified_for",
        None,
        "qualification",
    ),
    (
        "tldw_chatbook.Backup_Recovery.qualification",
        "_qualified_identity",
        None,
        "qualified_identity",
    ),
)
_active = None


class Witness(StartupIOCensus):
    def __init__(self, repo):
        super().__init__(repo, secrets.token_bytes(32), specs=_SPECS)
        self.stock = {
            key: value
            for key, value in self.stock.items()
            if key in ("loader_exec", "loader_code")
        }
        self.phase = "collection"
        self.evidence, self.evidence_counts = [], {}
        self.max_evidence = 4096
        self.healthy_detail_aggregated = 0
        self.detail_cap_reached = False
        self.installed_sources_before = _installed_sources()
        self.startup_owners = {}  # Strong actual leases; no id ABA.
        self.actor_threads = {}  # Strong actual Thread objects.
        self.receipt_sources = {
            "tldw_chatbook/Backup_Recovery/storage_admission.py",
            "tldw_chatbook/Backup_Recovery/qualification.py",
        }
        self.source_before = {
            name: hashlib.sha256((self.repo / name).read_bytes()).hexdigest()
            for name in self.receipt_sources
        }

    def _record(self, row):
        key = row["kind"]
        if (
            key in ("qualification_return", "qualified_identity_return")
            and row.get("allowed") is True
            and self.evidence_counts.get(key, 0) >= 24
        ):
            self.evidence_counts[key] = self.evidence_counts.get(key, 0) + 1
            self.healthy_detail_aggregated += 1
            return  # Aggregate healthy repeated reads; reserve finite detail for ownership.
        self.evidence_counts[key] = self.evidence_counts.get(key, 0) + 1
        if len(self.evidence) < self.max_evidence:
            self.evidence.append(
                {
                    "phase": self.phase,
                    "time": time.perf_counter(),
                    "thread_ident": threading.get_ident(),
                    **row,
                }
            )
        else:
            self.detail_cap_reached = True

    def _census(self):
        module = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        if module is None:
            return {"unavailable": True}
        # These are ordinary defining-module containers, read under their exact
        # original lock. No owner callback, filesystem probe, or source gate runs.
        state = vars(module)
        with state["_lock"]:
            startups = tuple(state["_startups"].items())
            startup_leases = {lease for _, lease in startups}
            rows = []
            for (pid, root), lease in startups:
                values = vars(lease)
                key = values.get("_key")
                hold = state["_holds"].get(key)
                h = {} if hold is None else vars(hold)
                thread = h.get("thread") or values.get("resource_thread")
                rows.append(
                    {
                        "pid": pid,
                        "root_hmac_sha256": self._digest(root),
                        "lease_actor_id": id(lease),
                        "lease_type": type(lease).__name__,
                        "lease_defining_module": type(lease).__module__,
                        "native_key_is_none": key is None,
                        "actual_native_key_root_hmac": None
                        if key is None
                        else self._digest(key[1]),
                        "actual_native_key_pid": None if key is None else key[0],
                        "actual_hold_actor_id": None if hold is None else id(hold),
                        "actual_owner_thread_ident": None
                        if thread is None
                        else getattr(thread, "ident", None),
                        "actual_execution_selection_present": values.get(
                            "_execution_selection"
                        )
                        is not None,
                    }
                )
            return {
                "ordinary": len(state["_live_leases"] - startup_leases),
                "pending": len(state["_pending_acquisitions"]),
                "core": len(state["_operations"]),
                "raw": len(state["_raw_operations"]),
                "retiring": len(state["_retiring_holds"]),
                "startup_count": len(startups),
                "startup_keys_none": sum(
                    vars(lease).get("_key") is None for _, lease in startups
                ),
                "startups": rows,
            }

    def _caller_codes(self, frame):
        rows = []
        caller = frame.f_back
        while caller is not None and len(rows) < 6:
            rows.append(
                {
                    "module": caller.f_globals.get("__name__"),
                    "qualname": caller.f_code.co_qualname,
                    "first_line": caller.f_code.co_firstlineno,
                    "line": caller.f_lineno,
                }
            )
            caller = caller.f_back
        return rows  # No frames, values, source lines or exception bodies retained.

    def _start_event(self, code, offset):
        # The original callee frame is present directly under this callback.
        frame = sys._getframe(1)
        if not self.active or frame.f_code is not code:
            self.issues["ci_start_frame_mismatch"] += 1
            return
        try:
            label = self.labels[code]
            thread = threading.current_thread()
            self.actor_threads[id(thread)] = thread
            if label == "lease_close":
                lease = frame.f_locals.get("self")
                module = sys.modules.get(
                    "tldw_chatbook.Backup_Recovery.storage_admission"
                )
                with vars(module)["_lock"]:
                    registered = any(
                        owner is lease for owner in vars(module)["_startups"].values()
                    )
                    if registered or self.startup_owners.get(id(lease)) is lease:
                        self.startup_owners[id(lease)] = lease
                        self._record(
                            {
                                "kind": "startup_lease_close_start",
                                "actual_lease_actor_id": id(lease),
                                "registered_at_close_entry": registered,
                                "native_key_is_none_at_entry": vars(lease).get("_key")
                                is None,
                                "actual_thread_actor_id": id(thread),
                                "caller_code_metadata": self._caller_codes(frame),
                                "census": self._census(),
                            }
                        )
            if label == "pause_drain":
                owner = frame.f_locals.get("self")
                deadline = frame.f_locals.get("deadline")
                self._record(
                    {
                        "kind": "drain_start",
                        "pause_actor_id": id(owner),
                        "deadline_remaining_seconds": deadline - time.monotonic(),
                        "source_module": "tldw_chatbook.Backup_Recovery.storage_admission",
                        "source_first_line": code.co_firstlineno,
                        "census": self._census(),
                    }
                )
            if label not in ("loader_exec", "loader_code"):
                self._current_source(code)
            # This subclass callback cannot call the base callback, which would
            # observe a different immediate frame. Selected counts are explicit.
            self._count(label + "_entered")
        except Exception as error:
            self.issues["ci_start:" + type(error).__name__] += 1

    def _return_event(self, code, offset, value):
        frame = sys._getframe(1)
        if not self.active or frame.f_code is not code:
            self.issues["ci_return_frame_mismatch"] += 1
            return
        try:
            label = self.labels[code]
            values = frame.f_locals
            self._count(label + "_returned")
            if label == "loader_code":
                self._prepare_loaded_code(values.get("self"), value)
            elif label == "loader_exec":
                module = values.get("module")
                name = getattr(module, "__name__", None)
                if name is not None and sys.modules.get(name) is module:
                    self._bind(name)
            elif label == "pause_drain":
                owner = values.get("self")
                self._record(
                    {
                        "kind": "drain_return",
                        "pause_actor_id": id(owner),
                        "actual_bool_return": value if type(value) is bool else None,  # noqa: E721 - exact original metadata type.
                        "deadline_remaining_seconds": values.get("deadline")
                        - time.monotonic(),
                        "census": self._census(),
                    }
                )
            elif label in ("lease_creation", "lease_close"):
                lease = values.get("self")
                if label == "lease_creation" and vars(lease).get("_key") is None:
                    self.startup_owners[id(lease)] = lease
                    self._record(
                        {
                            "kind": "key_none_lease_creation_return",
                            "actual_lease_actor_id": id(lease),
                            "actual_thread_actor_id": id(threading.current_thread()),
                            "caller_code_metadata": self._caller_codes(frame),
                        }
                    )
                elif (
                    label == "lease_close"
                    and self.startup_owners.get(id(lease)) is lease
                ):
                    self._record(
                        {
                            "kind": "startup_lease_close_return",
                            "actual_lease_actor_id": id(lease),
                            "native_key_is_none_at_return": vars(lease).get("_key")
                            is None,
                            "actual_thread_actor_id": id(threading.current_thread()),
                            "caller_code_metadata": self._caller_codes(frame),
                            "census": self._census(),
                        }
                    )
            elif label == "startup_admission":
                key = values.get("key")
                issued = values.get("lease")
                if issued is not None:
                    self.startup_owners[id(issued)] = issued
                self._record(
                    {
                        "kind": "startup_return",
                        "census": self._census(),
                        "actual_key_root_hmac_sha256": self._digest(key[1])
                        if type(key) is tuple and len(key) == 2  # noqa: E721 - exact metadata types preserve original shape.
                        else None,
                        "actual_issued_lease_actor_id": None
                        if issued is None
                        else id(issued),
                        "source_module": "tldw_chatbook.Backup_Recovery.storage_admission",
                        "source_first_line": code.co_firstlineno,
                    }
                )
            elif label in ("qualification", "qualified_identity"):
                operation = values.get("operation")
                outcome = value if type(value) is tuple and len(value) == 2 else None  # noqa: E721 - exact metadata types preserve original shape.
                row = {
                    "kind": label + "_return",
                    "operation": operation if type(operation) is str else None,  # noqa: E721 - exact original metadata type.
                    "allowed": outcome[0]
                    if outcome is not None and type(outcome[0]) is bool  # noqa: E721 - exact original metadata type.
                    else None,
                    "reason": outcome[1]
                    if outcome is not None and type(outcome[1]) is str  # noqa: E721 - exact original metadata type.
                    else None,
                    "source_module": "tldw_chatbook.Backup_Recovery.qualification",
                    "source_first_line": code.co_firstlineno,
                }
                caller = frame.f_back
                if caller is not None:
                    row["actual_caller_source"] = {
                        "module": caller.f_globals.get("__name__"),
                        "qualname": caller.f_code.co_qualname,
                        "first_line": caller.f_code.co_firstlineno,
                    }
                if label == "qualification":
                    row["actual_root_hmac_sha256"] = self._digest(values.get("root"))
                else:
                    identity = values.get("identity")
                    if type(identity) is dict:  # noqa: E721 - exact original metadata type.
                        row["actual_native_identity"] = {
                            key: identity.get(key)
                            for key in (
                                "os",
                                "release",
                                "arch",
                                "python",
                                "filesystem",
                                "flags",
                            )
                        }
                self._record(row)
        except Exception as error:
            self.issues["ci_return:" + type(error).__name__] += 1


def _installed_sources():
    names = (
        "Tests.private_profile",
        "Tests.real_profile_guard",
        "Tests.network_guard",
        "Tests.windows_private_fixture_runner",
        "Tests.Performance.console_startup_cohort_code_local",
        "Tests.Performance.console_startup_cohort_witness",
        "tldw_chatbook",
        "tldw_profile_core",
        "tldw_chatbook.Backup_Recovery.storage_admission",
        "tldw_chatbook.Backup_Recovery.qualification",
    )
    records = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            records[name] = {"loaded": False}
            continue
        origin = vars(module).get("__file__")
        if not isinstance(origin, str):
            records[name] = {"loaded": True, "origin_missing": True}
            continue
        path = Path(origin).resolve()
        records[name] = {
            "loaded": True,
            "origin": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "modules": records,
    }


def _begin():
    global _active
    if _active is not None:
        return
    repo = Path(os.environ["TLDW_CI_WITNESS_REPO"]).absolute()
    receipt = Path(os.environ["TLDW_CI_WITNESS_RECEIPT"])
    assert not receipt.exists(), "Preserve existing receipts"
    _active = Witness(repo)
    _active.receipt = receipt
    _active.start()
    _active._record({"kind": "initial_actual_census", "census": _active._census()})


def pytest_load_initial_conftests(early_config, parser, args):
    _begin()  # Observe the genuine collection bootstrap when loaded early.


def pytest_sessionstart(session):
    _begin()  # Fallback records already-existing startup as provenance gap.


def pytest_runtest_logstart(nodeid, location):
    if _active is not None:
        _active.phase = nodeid  # Exact declared original node, metadata only.
        _active._record(
            {"kind": "original_node_start_census", "census": _active._census()}
        )


def pytest_sessionfinish(session, exitstatus):
    if _active is None:
        return
    _active._record({"kind": "final_actual_census", "census": _active._census()})
    observed = _active.stop()
    after = {
        name: hashlib.sha256((_active.repo / name).read_bytes()).hexdigest()
        for name in _active.receipt_sources
    }
    result = {
        "exitstatus": exitstatus,
        "diagnostic_only": True,
        "census_observations": _active.evidence,
        "census_counts": _active.evidence_counts,
        "detail_cap": _active.max_evidence,
        "healthy_qualification_detail_sampled_after": 24,
        "detail_cap_reached": _active.detail_cap_reached,
        "healthy_qualification_detail_aggregated": _active.healthy_detail_aggregated,
        "source_before": _active.source_before,
        "source_after": after,
        "source_unchanged": after == _active.source_before,
        "observer": observed,
        "actual_original_nodes_observed": [
            row["phase"]
            for row in _active.evidence
            if row["kind"] == "original_node_start_census"
        ],
        "actual_control_process_installed_sources_before": _active.installed_sources_before,
        "actual_control_process_installed_sources_after": _installed_sources(),
        "observes_shared_control_process_only": True,
        "limits": "Original tests/assertions/deadlines untouched. No global events or exceptional-completion inference. Initial existing startups without observed qualification return remain unattributed.",
    }
    _active.receipt.write_text(json.dumps(result, indent=2), encoding="utf8")
