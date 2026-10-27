"""Close fixed historical metadata references against accepted native plans."""

from dataclasses import replace

from .mobile_content_graph import _ContentBlocker, _UnprovedReference
from .mobile_content_settings import _METADATA_JSON, _typed


def _canvas(before, after, originals, accepted, identities):
    source = _typed(_METADATA_JSON, before["metadata_json"])
    mapped = _typed(_METADATA_JSON, after["metadata_json"])
    if len(source.canvas_cards) != len(mapped.canvas_cards):
        raise ValueError("canvas_card_count_changed")
    dependencies = []
    for old, new in zip(source.canvas_cards, mapped.canvas_cards, strict=True):
        canvas = accepted.get(("canvas_documents", (old.canvas_id,)))
        message = accepted.get(("messages", (old.origin.message_id,)))
        revisions = [
            (key, row)
            for key, row in originals.items()
            if key[0] == "canvas_revisions"
            and row["canvas_id"] == old.canvas_id
            and (old.revision_id is None or row["id"] == old.revision_id)
            and row["origin_message_id"] == old.origin.message_id
            and row["origin_turn_id"] == old.origin.run_id
        ]
        if canvas is None or message is None or not revisions:
            raise _UnprovedReference(
                "canvas_revision", old.revision_id or old.canvas_id
            )
        if new != replace(
            old,
            canvas_id=identities.mapped("canvas", old.canvas_id),
            revision_id=identities.mapped("canvas_revision", old.revision_id)
            if old.revision_id is not None
            else None,
            origin=replace(
                old.origin,
                message_id=identities.mapped("message", old.origin.message_id),
            ),
        ):
            raise ValueError("canvas_mapping_changed")
        for key, original in revisions:
            revision = accepted.get(key)
            if revision is None:
                raise _UnprovedReference("canvas_revision", original["id"])
            value = dict(revision.values)
            if (
                value["canvas_id"] != new.canvas_id
                or value["origin_message_id"] != new.origin.message_id
                or value["origin_turn_id"] != old.origin.run_id
                or dict(canvas.values)["conversation_id"]
                != dict(message.values)["conversation_id"]
            ):
                raise _UnprovedReference("canvas_revision", original["id"])
            dependencies.append(("canvas_revision", original["id"]))
    return tuple(dependencies)


def _fork(before, after, originals, accepted, identities):
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
        parse_console_fork_message_metadata,
    )

    source = parse_console_fork_message_metadata(before["metadata_json"])
    current = parse_console_fork_message_metadata(after["metadata_json"])
    if source is None or current != source or source[2] is None:
        raise ValueError("fork_metadata_changed")
    turn = source[2]
    mapped = identities.mapped("message", turn)
    calls = [
        (key, row)
        for key, row in originals.items()
        if key[0] == "console_trace_calls" and row["turn_id"] == turn
    ]
    if not calls or any(
        key not in accepted or dict(accepted[key].values)["turn_id"] != mapped
        for key, _row in calls
    ):
        raise _UnprovedReference("message", turn)
    return (
        encode_console_fork_message_metadata(source[0], source[1], mapped),
        tuple(("trace_call", row["call_id"]) for _key, row in calls),
    )


def _close_semantics(sync, workspace, rows, originals):
    """Proof completion changes no native owner/hold/quarantine authority."""
    accepted = {(r.table, r.original): r for r in rows}
    source = {(r.table, r.original): dict(r.values) for r in originals}
    frontiers = {(f.entity, (f.original_id,)) for f in sync.frontiers}
    successors = sync.trace.content.successors
    discharges, blockers = [], []
    for key, row in tuple(accepted.items()):
        pending = set(row.required_proofs)
        if not pending & {
            "I:binding",
            "T:canvas_turn",
            "T:trace_turn",
            "S:provider_continuation",
            "S:dispatch_owner",
        }:
            continue
        values = dict(row.values)
        dependencies = list(row.dependencies)
        closed = set()
        try:
            if "I:binding" in pending:
                from tldw_chatbook.Chat.console_project_instructions import (
                    decode_project_context_json,
                )

                old = decode_project_context_json(
                    source[key]["console_project_context_json"]
                )
                new = decode_project_context_json(
                    values["console_project_context_json"]
                )
                binding = accepted.get(
                    ("workspace_runtime_bindings", (old.working_folder_binding_id,))
                )
                if (
                    binding is None
                    or dict(binding.values)["binding_id"]
                    != new.working_folder_binding_id
                    or new.working_folder_binding_id not in workspace.binding_holds
                ):
                    raise _UnprovedReference("binding", old.working_folder_binding_id)
                dependencies.append(("binding", old.working_folder_binding_id))
                closed.add("I:binding")
            if "T:canvas_turn" in pending:
                dependencies.extend(
                    _canvas(source[key], values, source, accepted, sync.identities)
                )
                closed.add("T:canvas_turn")
            if "T:trace_turn" in pending:
                values["metadata_json"], refs = _fork(
                    source[key], values, source, accepted, sync.identities
                )
                dependencies.extend(refs)
                closed.add("T:trace_turn")
            for proof in ("S:provider_continuation", "S:dispatch_owner"):
                if proof in pending:
                    if key not in frontiers:
                        raise _UnprovedReference("message", row.original[0])
                    closed.add(proof)
        except _UnprovedReference as error:
            blockers.append(
                _ContentBlocker(
                    row.table,
                    row.original,
                    "integrated_semantic_dependency_unproven",
                    error.domain,
                    error.identity,
                )
            )
            continue
        except (KeyError, TypeError, ValueError):
            blockers.append(
                _ContentBlocker(row.table, row.original, "integrated_semantic_invalid")
            )
            continue
        if closed:
            accepted[key] = replace(
                row,
                values=tuple(values.items()),
                required_proofs=tuple(
                    p for p in row.required_proofs if p not in closed
                ),
                dependencies=tuple(dependencies),
            )
            for slot in successors:
                if (
                    slot.table,
                    slot.original,
                ) == key and slot.handler + ":" + slot.domain in closed:
                    discharges.append(
                        replace(
                            slot,
                            mapped_id=sync.identities.mapped(
                                "message", slot.original_id
                            ),
                        )
                        if slot.domain == "trace_turn"
                        else slot
                    )
    return tuple(accepted.values()), tuple(discharges), tuple(blockers)
