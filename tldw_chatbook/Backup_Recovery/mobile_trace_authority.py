"""Fixed native local-character ownership and eligibility evidence."""

from dataclasses import dataclass, field

from tldw_chatbook.DB.character_conversation_search import (
    CharacterConversationSearchRepository,
)


@dataclass(frozen=True)
class _LocalCharacterAssociation:
    conversation_id: str
    source_authority_id: str = field(repr=False)
    character_id: int | None
    joined: tuple = field(repr=False)
    source_owned: bool
    resolved: bool
    preview_eligible: bool


def _eligible(row, authority, resolved):
    return bool(
        resolved
        and not row["deleted"]
        and row["runtime_backend"] == "local"
        and row["assistant_kind"] == "character"
        and row["assistant_authority_id"] == authority
        and type(row["character_id"]) is int
        and row["character_id"] > 0
        and row["assistant_id"] == str(row["character_id"])
        and row["raw_card_id"] == row["character_id"]
        and row["card_deleted"] == 0
    )


def _local_characters(database, *, _read_budget=None):
    repository = CharacterConversationSearchRepository(database)
    authority = database.get_local_authority_id()
    result = []
    with database.transaction() as cursor:
        cursor.execute("""SELECT c.id, c.character_id, c.assistant_id,
                   c.assistant_authority_id, c.runtime_backend, c.assistant_kind,
                   c.deleted, card.id AS raw_card_id, card.deleted AS card_deleted,
                   CASE WHEN card.deleted=0 THEN card.id ELSE NULL END AS live_card_id
              FROM conversations AS c LEFT JOIN character_cards AS card ON card.id=c.character_id
             WHERE c.runtime_backend='local' AND c.assistant_kind='character'""")
        for row in cursor:
            if _read_budget is not None:
                _read_budget.value(row)
            resolved = repository._is_resolved_conversation(row)
            owned = bool(
                row["assistant_authority_id"] == authority
                and type(row["character_id"]) is int
                and row["character_id"] > 0
                and row["assistant_id"] == str(row["character_id"])
                and row["raw_card_id"] == row["character_id"]
            )
            result.append(
                _LocalCharacterAssociation(
                    row["id"],
                    authority,
                    row["character_id"],
                    tuple(zip(row.keys(), row, strict=True)),
                    owned,
                    resolved,
                    _eligible(row, authority, resolved),
                )
            )
    return tuple(result)


def _discharge_local_characters(reads, content, inputs, budget):
    """Fill only C's exact pending slot after captured native ownership proof."""
    from dataclasses import replace

    from .mobile_content_graph import _ContentBlocker
    from .mobile_trace_inputs import _RetainedTraceDatabase

    database = _RetainedTraceDatabase(reads._connections[("canonical", "chat")][0])
    canonical = CharacterConversationSearchRepository(database)
    authority = database.get_local_authority_id()
    if authority != inputs.canonical.citation.context.identity.local_authority_id:
        raise ValueError("mobile_trace_input_binding_changed")
    source_authority = inputs.source.citation.context.identity.local_authority_id
    source = {(r.table, r.original): dict(r.values) for r in content.source_rows}
    accepted = {(r.table, r.original): r for r in content.rows}
    associations = {p.conversation_id: p for p in inputs.source.local_characters}
    result, blockers = [], []
    for slot in content.authority_slots:
        key = ("conversations", (slot.conversation_id,))
        original = source.get(key)
        mapped_row = accepted.get(key)
        card_key = ("character_cards", (slot.original_character_id,))
        source_card = source.get(card_key)
        mapped_card_row = accepted.get(card_key)
        proof = associations.get(slot.conversation_id)
        reason = "native_local_authority_namespace_readability_unproved"
        if (
            original is not None
            and mapped_row is not None
            and source_card is not None
            and mapped_card_row is not None
            and proof is not None
            and proof.source_owned
        ):
            budget.value(mapped_row)
            mapped = dict(mapped_row.values)
            card = dict(mapped_card_row.values)
            joined = dict(proof.joined)
            source_fields = (
                "id",
                "character_id",
                "assistant_id",
                "assistant_authority_id",
                "runtime_backend",
                "assistant_kind",
                "deleted",
            )
            if (
                all(original[field] == joined[field] for field in source_fields)
                and original["assistant_authority_id"]
                == slot.original_authority_id
                == source_authority
                and proof.source_authority_id == source_authority
                and source_card["id"]
                == joined["raw_card_id"]
                == slot.original_character_id
                and source_card["deleted"] == joined["card_deleted"]
                and mapped["id"]
                == slot.mapped_conversation_id
                == content.identities.mapped("conversation", slot.conversation_id)
                and mapped["character_id"]
                == card["id"]
                == slot.mapped_character_id
                == content.identities.mapped("character", slot.original_character_id)
                and mapped["assistant_id"] == str(card["id"])
                and "assistant_authority_id" not in mapped
                and "assistant_authority_id" in mapped_row.pending_columns
                and all(
                    mapped[field] == original[field]
                    for field in ("runtime_backend", "assistant_kind", "deleted")
                )
                and card["deleted"] == source_card["deleted"]
            ):
                mapped["assistant_authority_id"] = authority
                mapped_joined = {field: mapped[field] for field in source_fields}
                mapped_joined.update(
                    raw_card_id=card["id"],
                    card_deleted=card["deleted"],
                    live_card_id=card["id"] if card["deleted"] == 0 else None,
                )
                resolved = canonical._is_resolved_conversation(mapped_joined)
                if (
                    resolved == proof.resolved
                    and _eligible(mapped_joined, authority, resolved)
                    == proof.preview_eligible
                ):
                    result.append(
                        replace(
                            mapped_row,
                            values=tuple(mapped.items()),
                            pending_columns=tuple(
                                c
                                for c in mapped_row.pending_columns
                                if c != "assistant_authority_id"
                            ),
                            required_proofs=tuple(
                                p
                                for p in mapped_row.required_proofs
                                if p != "T:character_authority"
                            ),
                        )
                    )
                    continue
                reason = "native_local_character_eligibility_changed"
        blockers.append(
            _ContentBlocker(
                "conversations",
                (slot.conversation_id,),
                reason,
                "local_authority",
                slot.original_authority_id,
            )
        )
    return tuple(result), tuple(blockers)
