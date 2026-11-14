"""Propagate later refusals through previously qualified native input ownership."""

from .mobile_content_graph import _ContentBlocker
from .mobile_trace_plan import _TraceInputIncidence


def _same_projection(qualified, current, identities, budget):
    budget.value(qualified.values)
    budget.value(current.values)
    before, after = dict(qualified.values), dict(current.values)
    if before == after:
        return True
    # Only this separately qualified metadata codec changes a C value in S.
    if qualified.table != "messages" or "T:trace_turn" not in qualified.required_proofs:
        return False
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
        parse_console_fork_message_metadata,
    )

    budget.json(before["metadata_json"])
    budget.value(before["metadata_json"], text_copies=2)
    source = parse_console_fork_message_metadata(before["metadata_json"])
    if source is None or source[2] is None:
        return False
    before["metadata_json"] = encode_console_fork_message_metadata(
        source[0], source[1], identities.mapped("message", source[2])
    )
    return before == after


def _close_qualified_inputs(trace, rows, blockers, *, _read_budget):
    """Use issued immutable proofs; never reconstruct a native graph again."""
    budget = _read_budget
    budget.value(rows)
    budget.value(trace.content.rows)
    budget.value(trace.rows)
    budget.value(trace.input_incidence)
    budget.container(
        (
            len(rows)
            + len(trace.content.rows)
            + len(trace.rows)
            + len(trace.input_incidence)
        )
        * 8
    )
    accepted = {(row.table, row.original): row for row in rows}
    qualified = {
        (row.table, row.original): row for row in (*trace.content.rows, *trace.rows)
    }
    proofs = {}
    for proof in trace.input_incidence:
        if type(proof) is _TraceInputIncidence:
            proofs.setdefault(proof.original_call_id, []).append(proof)
    refused = set()
    for key, row in accepted.items():
        if row.table != "console_trace_calls":
            continue
        budget.value(row.values)
        budget.container(8)
        candidates = proofs.get(row.original[0], ())
        reason = "integrated_native_incidence_unproved"
        if len(candidates) == 1:
            proof = candidates[0]
            budget.container(len(proof.inputs))
            if (
                proof.mapped_call_id == dict(row.values)["call_id"]
                and proof.source_manifest_digest
                == trace.identities.source_manifest_digest
                and proof.canonical_manifest_digest
                == trace.identities.canonical_manifest_digest
                and type(proof.inputs) is tuple
                and key in proof.inputs
                and len(set(proof.inputs)) == len(proof.inputs)
            ):
                for dependency in proof.inputs:
                    original = qualified.get(dependency)
                    current = accepted.get(dependency)
                    if original is None or current is None:
                        reason = "integrated_native_input_lost"
                        break
                    if not _same_projection(
                        original, current, trace.identities, budget
                    ):
                        reason = "integrated_native_input_changed"
                        break
                else:
                    continue
        blockers.append(_ContentBlocker(row.table, row.original, reason))
        refused.add(key)

    graph = trace.content.actors.workflows.settings.graph
    for row in trace.content.source_rows:
        if row.table in {"canvas_documents", "canvas_revisions"}:
            budget.value(row.values)
    budget.container(len(trace.content.source_rows) * 3)
    source = {
        (row.table, row.original): dict(row.values)
        for row in trace.content.source_rows
        if row.table in {"canvas_documents", "canvas_revisions"}
    }
    for key, row in accepted.items():
        if row.table != "canvas_documents":
            continue
        budget.value(row.values)
        budget.container(len(source) * 3 + len(graph.canvas_batches) + 6)
        owned = tuple(
            revision_key
            for revision_key, revision in source.items()
            if revision_key[0] == "canvas_revisions"
            and revision["canvas_id"] == row.original[0]
        )
        mapped_id = dict(row.values)["id"]
        batches = [
            batch
            for batch in graph.canvas_batches
            if any(document.canvas_id == mapped_id for document in batch.documents)
        ]
        proved = len(batches) == 1 and bool(owned)
        if proved:
            budget.container((len(batches[0].revisions) + len(owned)) * 4)
            batch_ids = {
                revision.revision_id
                for revision in batches[0].revisions
                if revision.canvas_id == mapped_id
            }
            proved = batch_ids == {
                trace.identities.mapped("canvas_revision", revision_key[1][0])
                for revision_key in owned
            }
        if proved:
            batch = batches[0]
            budget.container(len(batch.documents) + len(owned) * 2)
            budget.value(row.values)
            documents = [
                document
                for document in batch.documents
                if document.canvas_id == mapped_id
            ]
            values = dict(row.values)
            proved = (
                len(documents) == 1
                and batch.conversation_id == values["conversation_id"]
                and all(
                    getattr(documents[0], name) == values[name]
                    for name in ("conversation_id", "created_at", "deleted_at")
                )
            )
            for revision_key in owned:
                current = accepted.get(revision_key)
                if current is None:
                    proved = False
                    break
                budget.value(current.values)
                budget.container(len(batch.revisions))
                values = dict(current.values)
                revisions = [
                    revision
                    for revision in batch.revisions
                    if revision.revision_id == values["id"]
                ]
                if len(revisions) != 1 or any(
                    getattr(revisions[0], name) != values[column]
                    for name, column in (
                        ("canvas_id", "canvas_id"),
                        ("parent_revision_id", "parent_revision_id"),
                        ("sequence", "sequence"),
                        ("title", "title"),
                        ("runtime_profile", "runtime_profile"),
                        ("source", "html"),
                        ("content_sha256", "content_sha256"),
                        ("source_bytes", "html_bytes"),
                        ("actor_kind", "actor_kind"),
                        ("origin_message_id", "origin_message_id"),
                        ("origin_turn_id", "origin_turn_id"),
                        ("created_at", "created_at"),
                        ("deleted_at", "deleted_at"),
                    )
                ):
                    proved = False
                    break
        if proved:
            for dependency in (key, *owned):
                original, current = qualified.get(dependency), accepted.get(dependency)
                if (
                    original is None
                    or current is None
                    or original.values != current.values
                ):
                    proved = False
                    break
        if not proved:
            blockers.append(
                _ContentBlocker(
                    row.table, row.original, "integrated_canvas_history_incomplete"
                )
            )
            refused.add(key)
    budget.container(len(rows))
    return tuple(row for row in rows if (row.table, row.original) not in refused)
