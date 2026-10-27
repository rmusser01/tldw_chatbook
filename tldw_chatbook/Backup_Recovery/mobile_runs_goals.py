"""Fixed Goal history transformations; results are never native write authority."""

import hashlib
import json
from dataclasses import dataclass, field, replace

from .mobile_content_graph import (
    _ContentBlocker,
    _ContentRow,
    _mapped,
    _UnprovedReference,
)
from .mobile_identity_plan import _Domain, _reserve, _rows


@dataclass(frozen=True)
class _GoalProvenance:
    table: str
    original: tuple
    original_hash: str
    mapped_hash: str
    artifact_digest: str | None = None


@dataclass(frozen=True)
class _GoalRows:
    identities: object = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    provenance: tuple[_GoalProvenance, ...]

    def require_complete(self):
        raise ValueError("mobile_goal_owner_handlers_pending")


def _plan_goal_rows(reads, identities, *, _read_budget):
    from tldw_chatbook.Agents.goal_models import GoalRequest

    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_trace_inputs import _check_trace_members

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_goal_native_snapshot_required")
    reads.check()
    budget = _read_budget
    from .mobile_plan_manifest import _TABLES

    budget.container(len(_TABLES) * 4)
    tables = {
        s.table: s
        for s in _TABLES
        if s.table
        in {
            "goal_runs",
            "goal_iterations",
            "goal_reports",
            "goal_evidence",
            "goal_checkpoints",
            "goal_waits",
            "goal_payload_reservations",
        }
    }
    sources = {
        table: _rows(reads, "legacy", "runs", table, _read_budget=budget)
        for table in tables
    }
    binding_rows = _rows(
        reads, "legacy", "workspaces", "workspace_runtime_bindings", _read_budget=budget
    )
    budget.container(len(binding_rows) * 2)
    bindings = {row["binding_id"]: row for row in binding_rows}
    base = identities
    additions = []
    for domain in (
        _Domain("goal_launch", "runs", "goal_runs", "launch_id"),
        _Domain("goal_iteration_launch", "runs", "goal_iterations", "launch_id"),
    ):
        canonical = _rows(reads, "canonical", "runs", domain.table, _read_budget=budget)
        budget.container(len(sources[domain.table]) * 4)
        additions.extend(
            _reserve(
                domain,
                sources[domain.table],
                canonical,
                base.source_fingerprint,
                _read_budget=budget,
            )
        )
    budget.container(len(base.reservations) + len(additions))
    identities = replace(base, reservations=base.reservations + tuple(additions))
    source = sources["goal_runs"]
    budget.container(len(source) * 8)
    rows, originals, blockers, provenance = [], [], [], []
    for original in source:
        budget.value(original)
        budget.container(len(original) * 3)
        values = dict(original)
        identity = (original["id"],)
        originals.append(_ContentRow("goal_runs", identity, tuple(original.items())))
        dependencies = []

        def ref(domain, value, *, _dependencies=dependencies):
            mapped = _mapped(identities, domain, value)
            budget.container(2)
            _dependencies.append((domain, value))
            return mapped

        try:
            values["id"] = ref("goal", original["id"])
            values["launch_id"] = ref("goal_launch", original["launch_id"])
            values["conversation_id"] = ref("conversation", original["conversation_id"])
            values["chain_id"] = ref("work_chain", original["chain_id"])
            if original["status"] == "removed":
                if original["request_json"]:
                    raise ValueError("removed_goal_body")
            else:
                raw = original["request_json"]
                budget.json(raw)
                request = GoalRequest.model_validate_json(raw)
                budget.value(raw)
                if (
                    hashlib.sha256(raw.encode("utf-8")).hexdigest()
                    != original["payload_hash"]
                ):
                    raise ValueError("goal_request_hash")
                budget.json(raw)
                data = request.model_dump(mode="json")
                for binding in (data["binding"], *data["source_bindings"]):
                    actual = bindings.get(binding["binding_id"])
                    if (
                        actual is None
                        or actual["workspace_id"] != binding["workspace_id"]
                    ):
                        raise ValueError("goal_binding_owner_unproved")
                    binding["workspace_id"] = ref("workspace", binding["workspace_id"])
                    binding["binding_id"] = ref("binding", binding["binding_id"])
                budget.value(data, text_copies=3)
                # JSON validation preserves strict tuple fields and native version defaults.
                mapped = GoalRequest.model_validate_json(json.dumps(data))
                budget.value(data, text_copies=3)
                values["request_json"] = mapped.canonical_json()
                values["payload_hash"] = hashlib.sha256(
                    values["request_json"].encode("utf-8")
                ).hexdigest()
            budget.container(8)
            provenance.append(
                _GoalProvenance(
                    "goal_runs",
                    identity,
                    original["payload_hash"],
                    values["payload_hash"],
                )
            )
            rows.append(
                _ContentRow(
                    "goal_runs",
                    identity,
                    tuple(values.items()),
                    required_proofs=("A-runs:exact_goal_chain_attempt_holds",),
                    dependencies=tuple(dependencies),
                )
            )
        except _UnprovedReference as error:
            blockers.append(
                _ContentBlocker(
                    "goal_runs",
                    identity,
                    "dependency_unproven",
                    error.domain,
                    error.identity,
                )
            )
        except (TypeError, ValueError, KeyError):
            blockers.append(
                _ContentBlocker("goal_runs", identity, "native_goal_content_invalid")
            )
    budget.container(sum(len(v) for v in sources.values()) * 4)
    removed = {r["id"] for r in source if r["status"] == "removed"}
    evidence_owners = {r["id"]: r["goal_id"] for r in sources["goal_evidence"]}
    attempt_owners = {r["attempt_id"]: r["goal_id"] for r in sources["goal_iterations"]}
    for table, spec in tables.items():
        if table == "goal_runs":
            continue
        for original in sources[table]:
            budget.value(original)
            budget.container(len(original) * 4)
            identity = tuple(original[k] for k in spec.identity_columns)
            originals.append(_ContentRow(table, identity, tuple(original.items())))
            dependencies = []
            values = dict(original)

            def ref(domain, value, *, _dependencies=dependencies):
                mapped = _mapped(identities, domain, value)
                budget.container(2)
                _dependencies.append((domain, value))
                return mapped

            try:
                if original["goal_id"] in removed and (
                    table != "goal_iterations"
                    or original["check_results_json"] != "[]"
                    or original["evidence_refs_json"] != "[]"
                ):
                    raise ValueError("removed_goal_child_body")
                values["goal_id"] = ref("goal", original["goal_id"])
                for column, domain in {
                    "goal_iterations": (
                        ("id", "goal_iteration"),
                        ("attempt_id", "wake_attempt"),
                        ("launch_id", "goal_iteration_launch"),
                        ("run_id", "run"),
                    ),
                    "goal_checkpoints": (("attempt_id", "wake_attempt"),),
                    "goal_evidence": (
                        ("id", "goal_evidence"),
                        ("attempt_id", "wake_attempt"),
                    ),
                    "goal_reports": (
                        ("id", "goal_report"),
                        ("iteration_id", "goal_iteration"),
                    ),
                    "goal_payload_reservations": (("attempt_id", "wake_attempt"),),
                    "goal_waits": (),
                }[table]:
                    if original[column] is not None:
                        values[column] = ref(domain, original[column])

                def evidence_id(value, *, advisory=False, _goal_id=original["goal_id"]):
                    if value is None:
                        return None
                    owner = evidence_owners.get(value)
                    if owner is None and advisory:
                        return value
                    if owner != _goal_id:
                        raise ValueError("goal_evidence_owner")
                    return ref("goal_evidence", value)

                def mapped_payload(model, raw):
                    budget.json(raw)
                    parsed = model.model_validate_json(raw)
                    budget.json(raw)
                    return parsed.model_dump(mode="json")

                def dump(model, data):
                    budget.value(data, text_copies=4)
                    parsed = model.model_validate_json(json.dumps(data))
                    budget.value(data, text_copies=3)
                    return parsed.model_dump_json()

                def report_refs(data):
                    # Model-supplied missing evidence is retained advisory history.
                    data["evidence_ids"] = [
                        evidence_id(v, advisory=True) for v in data["evidence_ids"]
                    ]

                from tldw_chatbook.Agents.goal_models import (
                    GoalCheck,
                    GoalCheckpoint,
                    GoalEvidence,
                    GoalReport,
                )

                if table == "goal_evidence":
                    data = mapped_payload(GoalEvidence, original["payload_json"])
                    for column, domain in (
                        ("id", "goal_evidence"),
                        ("goal_id", "goal"),
                        ("attempt_id", "wake_attempt"),
                        ("run_id", "run"),
                    ):
                        data[column] = ref(domain, data[column])
                    values["payload_json"] = dump(GoalEvidence, data)
                elif table == "goal_reports":
                    data = mapped_payload(GoalReport, original["payload_json"])
                    report_refs(data)
                    values["payload_json"] = dump(GoalReport, data)
                elif table == "goal_checkpoints":
                    data = mapped_payload(GoalCheckpoint, original["payload_json"])
                    data["id"] = ref("wake_attempt", data["id"])
                    report_refs(data["report"])
                    decision = data["decision"]
                    if decision["checkpoint_id"] is not None:
                        if (
                            attempt_owners.get(decision["checkpoint_id"])
                            != original["goal_id"]
                        ):
                            raise ValueError("goal_checkpoint_owner_unproved")
                        decision["checkpoint_id"] = ref(
                            "wake_attempt", decision["checkpoint_id"]
                        )
                    decision["evidence_errors"] = [
                        evidence_id(v, advisory=True)
                        for v in decision["evidence_errors"]
                    ]
                    for check in decision["checks"]:
                        check["evidence_id"] = evidence_id(check["evidence_id"])
                    values["payload_json"] = dump(GoalCheckpoint, data)
                    budget.container(8)
                    provenance.append(
                        _GoalProvenance(
                            table,
                            identity,
                            original["payload_hash"],
                            original["payload_hash"],
                            data["artifact_digest"],
                        )
                    )
                elif table == "goal_iterations":
                    for column in ("check_results_json", "evidence_refs_json"):
                        budget.json(original[column])
                        data = json.loads(original[column])
                        if not isinstance(data, list):
                            raise TypeError("goal_list_required")
                        budget.value(data, text_copies=4)
                        if column == "evidence_refs_json":
                            data = [evidence_id(v) for v in data]
                        else:
                            checked = []
                            for item in data:
                                check = GoalCheck.model_validate_json(
                                    json.dumps(item)
                                ).model_dump(mode="json")
                                check["evidence_id"] = evidence_id(check["evidence_id"])
                                checked.append(check)
                            data = checked
                        budget.value(data, text_copies=3)
                        values[column] = json.dumps(data)
                budget.container(len(values) * 2 + len(dependencies) * 2)
                rows.append(
                    _ContentRow(
                        table,
                        identity,
                        tuple(values.items()),
                        required_proofs=("A-runs:exact_goal_chain_attempt_holds",),
                        dependencies=tuple(dependencies),
                    )
                )
            except _UnprovedReference as error:
                blockers.append(
                    _ContentBlocker(
                        table,
                        identity,
                        "dependency_unproven",
                        error.domain,
                        error.identity,
                    )
                )
            except (TypeError, ValueError, KeyError):
                blockers.append(
                    _ContentBlocker(table, identity, "native_goal_content_invalid")
                )
    _check_trace_members(reads)
    budget.container((len(rows) + len(originals) + len(blockers) + len(provenance)) * 2)
    return _GoalRows(
        identities, tuple(rows), tuple(originals), tuple(blockers), tuple(provenance)
    )
