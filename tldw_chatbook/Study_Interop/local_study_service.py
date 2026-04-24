"""Thin local study wrapper around ChaChaNotes_DB flashcard helpers."""

from __future__ import annotations

import csv
import io
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Optional


class LocalStudyService:
    """Thin sync wrapper around local study helpers."""

    def __init__(
        self,
        db: Any,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.db = db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local study backend is unavailable.")
        return self.db

    def _dispatch_local_notification(
        self,
        *,
        title: str,
        message: str,
        source_entity_id: str | None,
        source_entity_kind: str,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category="study",
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=source_entity_id,
                source_entity_kind=source_entity_kind,
                payload=payload or {},
            )
        except Exception:
            return None

    def list_decks(self, *, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_decks(limit=limit, offset=offset)

    def get_deck(self, deck_id: str) -> Any:
        return self._require_db().get_deck(deck_id)

    def create_deck(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scheduler_type: Optional[str] = None,
    ) -> Any:
        deck_id = self._require_db().create_deck(name, description)
        deck = self._require_db().get_deck(deck_id)
        self._dispatch_local_notification(
            title="Local study deck created",
            message=f"Local study deck created: {deck.get('name') or name}",
            source_entity_id=str(deck.get("id") or deck_id),
            source_entity_kind="study_deck",
            payload={"action": "deck_created", "deck_id": str(deck.get("id") or deck_id)},
        )
        return deck

    def update_deck(
        self,
        deck_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        review_prompt_side: Optional[str] = None,
        scheduler_type: Optional[str] = None,
        scheduler_settings: Optional[dict[str, Any]] = None,
        expected_version: Optional[int] = None,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {
                "review_prompt_side": review_prompt_side,
                "scheduler_type": scheduler_type,
                "scheduler_settings": scheduler_settings,
            }.items()
            if value is not None
        }
        self._require_db().update_deck(
            deck_id,
            name=name,
            description=description,
            metadata=metadata or None,
            expected_version=expected_version,
        )
        return self._require_db().get_deck(deck_id)

    def list_flashcards(
        self,
        *,
        deck_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_q = str(q or "").strip() or None
        return self._require_db().list_flashcards(deck_id=deck_id, q=normalized_q, limit=limit, offset=offset)

    def create_flashcard(
        self,
        *,
        deck_id: str,
        front: str,
        back: str,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {"notes": notes, "extra": extra}.items()
            if value not in {None, ""}
        }
        card_id = self._require_db().create_flashcard(
            {
                "deck_id": deck_id,
                "front": front,
                "back": back,
                "tags": " ".join(tags or []),
                "type": "basic",
                "metadata": metadata or None,
            }
        )
        return self._require_db().get_flashcard(card_id)

    def create_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        created_cards = [self.create_flashcard(**dict(card)) for card in cards]
        result = {"items": created_cards, "count": len(created_cards)}
        self._dispatch_local_notification(
            title="Local flashcards created",
            message=f"Created {len(created_cards)} local flashcard(s).",
            source_entity_id=None,
            source_entity_kind="flashcard_bulk_create",
            payload={"action": "flashcards_created", "count": len(created_cards)},
        )
        return result

    def get_flashcard(self, card_id: str) -> Any:
        return self._require_db().get_flashcard(card_id)

    def update_flashcard(
        self,
        card_id: str,
        *,
        deck_id: Optional[str] = None,
        front: Optional[str] = None,
        back: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        extra: Optional[str] = None,
        expected_version: Optional[int] = None,
        **extra_fields: Any,
    ) -> Any:
        metadata = {
            key: value
            for key, value in {"notes": notes, "extra": extra}.items()
            if value is not None
        }
        self._require_db().update_flashcard(
            card_id,
            deck_id=deck_id,
            front=front,
            back=back,
            tags=tags,
            metadata=metadata or None,
            expected_version=expected_version,
            **extra_fields,
        )
        return self._require_db().get_flashcard(card_id)

    def update_flashcards_bulk(self, cards: list[Mapping[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for card in cards:
            payload = dict(card)
            card_id = payload.pop("id", None) or payload.pop("uuid", None)
            if not card_id:
                results.append({"status": "error", "error": "Missing flashcard id", "flashcard": None})
                continue
            flashcard = self.update_flashcard(str(card_id), **payload)
            results.append({"status": "updated", "flashcard": flashcard})
        return {"results": results, "count": len(results)}

    def move_flashcard(
        self,
        card_id: str,
        *,
        target_deck_id: str,
        expected_version: Optional[int] = None,
    ) -> Any:
        moved = self._require_db().move_flashcard(
            card_id,
            target_deck_id,
            expected_version=expected_version,
        )
        if not moved:
            return None
        return self._require_db().get_flashcard(card_id)

    def reset_flashcard_scheduling(
        self,
        card_id: str,
        *,
        expected_version: Optional[int] = None,
    ) -> Any:
        self._require_db().reset_flashcard_scheduling(card_id, expected_version=expected_version)
        return self._require_db().get_flashcard(card_id)

    def set_flashcard_tags(self, card_id: str, *, tags: list[str]) -> Any:
        self._require_db().set_flashcard_tags(card_id, tags=tags)
        return self._require_db().get_flashcard(card_id)

    def get_flashcard_tags(self, card_id: str) -> dict[str, Any]:
        payload = self._require_db().get_flashcard_tags(card_id)
        if isinstance(payload, Mapping):
            items = list(payload.get("items") or [])
        else:
            items = list(payload or [])
        return {"items": items, "count": len(items)}

    def list_flashcard_tag_suggestions(
        self,
        *,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._require_db().list_flashcard_tag_suggestions(q=q, limit=limit)
        if isinstance(payload, Mapping):
            items = list(payload.get("items") or [])
        else:
            items = list(payload or [])
        return {"items": items, "count": len(items)}

    @staticmethod
    def _split_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).replace(",", " ")
        return [item.strip() for item in text.split() if item.strip()]

    @staticmethod
    def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            try:
                decoded = json.loads(metadata)
                return dict(decoded) if isinstance(decoded, Mapping) else {}
            except json.JSONDecodeError:
                return {}
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return {}

    def _resolve_or_create_deck_id(self, deck_ref: Any) -> str:
        deck_ref_text = str(deck_ref or "").strip()
        if not deck_ref_text:
            raise ValueError("Flashcard import row is missing a deck.")
        for deck in list(self.list_decks(limit=1000, offset=0) or []):
            if str(deck.get("id")) == deck_ref_text or str(deck.get("name") or "").strip() == deck_ref_text:
                return str(deck.get("id"))
        created = self.create_deck(name=deck_ref_text)
        return str(created.get("id"))

    @staticmethod
    def _header_map(header: list[str], row: list[str]) -> dict[str, str]:
        return {
            str(name).strip().lower().replace(" ", "_"): row[index].strip()
            for index, name in enumerate(header)
            if index < len(row)
        }

    def preview_structured_qa_import(
        self,
        content: str,
        *,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        lines = content.splitlines()
        drafts: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        skipped_blocks = 0

        if max_lines is not None and len(lines) > max_lines:
            errors.append({"line": max_lines + 1, "error": "Content exceeds max_lines."})
            lines = lines[:max_lines]

        block: list[tuple[int, str]] = []

        def flush_block() -> None:
            nonlocal skipped_blocks
            if not block:
                return
            front: str | None = None
            back: str | None = None
            for line_number, line_text in block:
                stripped = line_text.strip()
                lowered = stripped.lower()
                if lowered.startswith(("q:", "question:")):
                    front = stripped.split(":", 1)[1].strip()
                elif lowered.startswith(("a:", "answer:")):
                    back = stripped.split(":", 1)[1].strip()
                if max_line_length is not None and len(line_text) > max_line_length:
                    errors.append({"line": line_number, "error": "Line exceeds max_line_length."})
                    skipped_blocks += 1
                    return
            if not front or not back:
                errors.append({"line": block[0][0], "error": "Block must contain Q:/A: labels."})
                skipped_blocks += 1
                return
            if max_field_length is not None and (len(front) > max_field_length or len(back) > max_field_length):
                errors.append({"line": block[0][0], "error": "Field exceeds max_field_length."})
                skipped_blocks += 1
                return
            drafts.append(
                {
                    "front": front,
                    "back": back,
                    "line_start": block[0][0],
                    "line_end": block[-1][0],
                    "notes": None,
                    "extra": None,
                    "tags": [],
                }
            )

        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                block.append((line_number, line))
                continue
            flush_block()
            block = []
        flush_block()

        return {
            "drafts": drafts,
            "errors": errors,
            "detected_format": "qa_labels",
            "skipped_blocks": skipped_blocks,
        }

    def import_flashcards_tsv(
        self,
        content: str,
        *,
        delimiter: str = "\t",
        has_header: bool = False,
        max_lines: Optional[int] = None,
        max_line_length: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        rows = list(csv.reader(io.StringIO(content), delimiter=delimiter))
        header = [column.strip() for column in rows[0]] if has_header and rows else []
        data_rows = rows[1:] if has_header else rows
        items: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if max_lines is not None and len(data_rows) > max_lines:
            errors.append({"line": max_lines + (2 if has_header else 1), "error": "Content exceeds max_lines."})
            data_rows = data_rows[:max_lines]

        for index, row in enumerate(data_rows, start=2 if has_header else 1):
            if not row or not any(str(column).strip() for column in row):
                continue
            if max_line_length is not None and len(delimiter.join(row)) > max_line_length:
                errors.append({"line": index, "error": "Line exceeds max_line_length."})
                continue
            try:
                if has_header:
                    mapped = self._header_map(header, row)
                    deck_ref = mapped.get("deck_id") or mapped.get("deck") or mapped.get("deck_name")
                    front = mapped.get("front") or mapped.get("question")
                    back = mapped.get("back") or mapped.get("answer")
                    tags = self._split_tags(mapped.get("tags"))
                    notes = mapped.get("notes") or None
                    extra = mapped.get("extra") or None
                else:
                    if len(row) < 3:
                        raise ValueError("Row must contain deck, front, and back fields.")
                    deck_ref, front, back = row[0].strip(), row[1].strip(), row[2].strip()
                    tags = self._split_tags(row[3] if len(row) > 3 else None)
                    notes = row[4].strip() if len(row) > 4 and row[4].strip() else None
                    extra = row[5].strip() if len(row) > 5 and row[5].strip() else None
                if not front or not back:
                    raise ValueError("Row must contain front and back fields.")
                if max_field_length is not None and (len(front) > max_field_length or len(back) > max_field_length):
                    raise ValueError("Field exceeds max_field_length.")
                deck_id = self._resolve_or_create_deck_id(deck_ref)
                card = self.create_flashcard(deck_id=deck_id, front=front, back=back, tags=tags, notes=notes, extra=extra)
                card_id = str(card.get("uuid") or card.get("id"))
                items.append({"uuid": card_id, "deck_id": deck_id})
            except Exception as exc:
                errors.append({"line": index, "error": str(exc)})

        result = {"imported": len(items), "items": items, "errors": errors}
        self._dispatch_local_notification(
            title="Local flashcards imported",
            message=f"Imported {len(items)} local flashcard(s) from TSV.",
            source_entity_id=None,
            source_entity_kind="flashcard_import",
            severity="warning" if errors else "info",
            payload={"action": "flashcards_imported", "format": "tsv", "imported": len(items), "errors": len(errors)},
        )
        return result

    @staticmethod
    def _json_flashcard_records(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            records = payload["cards"] if "cards" in payload else payload.get("items")
        else:
            records = None
        if not isinstance(records, list):
            raise ValueError("JSON flashcard import must be a list or an object with cards/items.")
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("JSON flashcard import items must be objects.")
        return list(records)

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (int, float, bool)):
            text = str(value).strip()
            return text or None
        return json.dumps(value, ensure_ascii=False)

    def import_flashcards_json_file(
        self,
        file_path: Any,
        *,
        max_items: Optional[int] = None,
        max_field_length: Optional[int] = None,
    ) -> dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as handle:
            records = self._json_flashcard_records(json.load(handle))

        items: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if max_items is not None and len(records) > max_items:
            errors.append({"index": max_items + 1, "error": "Content exceeds max_items."})
            records = records[:max_items]

        for index, record in enumerate(records, start=1):
            try:
                deck_ref = record.get("deck_id") or record.get("deck") or record.get("deck_name")
                front = self._optional_text(record.get("front") or record.get("question"))
                back = self._optional_text(record.get("back") or record.get("answer"))
                tags = self._split_tags(record.get("tags"))
                notes = self._optional_text(record.get("notes"))
                extra = self._optional_text(record.get("extra"))
                if not front or not back:
                    raise ValueError("Item must contain front and back fields.")
                fields = [front, back]
                if notes:
                    fields.append(notes)
                if extra:
                    fields.append(extra)
                if max_field_length is not None and any(len(field) > max_field_length for field in fields):
                    raise ValueError("Field exceeds max_field_length.")
                deck_id = self._resolve_or_create_deck_id(deck_ref)
                card = self.create_flashcard(
                    deck_id=deck_id,
                    front=front,
                    back=back,
                    tags=tags,
                    notes=notes,
                    extra=extra,
                )
                card_id = str(card.get("uuid") or card.get("id"))
                items.append({"uuid": card_id, "deck_id": deck_id})
            except Exception as exc:
                errors.append({"index": index, "error": str(exc)})

        result = {"imported": len(items), "items": items, "errors": errors}
        self._dispatch_local_notification(
            title="Local flashcards imported",
            message=f"Imported {len(items)} local flashcard(s) from JSON.",
            source_entity_id=None,
            source_entity_kind="flashcard_import",
            severity="warning" if errors else "info",
            payload={"action": "flashcards_imported", "format": "json", "imported": len(items), "errors": len(errors)},
        )
        return result

    def export_flashcards(
        self,
        *,
        deck_id: str | int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
        tag: str | None = None,
        q: str | None = None,
        export_format: str = "csv",
        include_reverse: bool = False,
        delimiter: str = "\t",
        include_header: bool = False,
        extended_header: bool = False,
    ) -> bytes:
        if workspace_id or include_workspace_items:
            raise ValueError("Workspace Study is unavailable in local mode")
        records = list(self.list_flashcards(deck_id=str(deck_id) if deck_id is not None else None, q=q, limit=100000, offset=0) or [])
        if tag:
            tag_key = str(tag).strip()
            records = [record for record in records if tag_key in self._split_tags(record.get("tags"))]

        deck_names: dict[str, str] = {}
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        if include_header or extended_header:
            writer.writerow(["Deck", "Front", "Back", "Tags", "Notes", "Extra"])
        for record in records:
            record_deck_id = str(record.get("deck_id") or deck_id or "")
            if record_deck_id not in deck_names:
                try:
                    deck = self.get_deck(record_deck_id) if record_deck_id else {}
                    deck_names[record_deck_id] = str(deck.get("name") or record_deck_id)
                except Exception:
                    deck_names[record_deck_id] = record_deck_id
            metadata = self._metadata(record)
            tags_text = " ".join(self._split_tags(record.get("tags")))
            row = [
                deck_names[record_deck_id],
                str(record.get("front") or ""),
                str(record.get("back") or ""),
                tags_text,
                str(record.get("notes") or metadata.get("notes") or ""),
                str(record.get("extra") or metadata.get("extra") or ""),
            ]
            writer.writerow(row)
            if include_reverse:
                writer.writerow([row[0], row[2], row[1], row[3], row[4], row[5]])
        data = output.getvalue().encode("utf-8")
        self._dispatch_local_notification(
            title="Local flashcards exported",
            message=f"Exported {len(records)} local flashcard(s).",
            source_entity_id=str(deck_id) if deck_id is not None else None,
            source_entity_kind="flashcard_export",
            payload={
                "action": "flashcards_exported",
                "count": len(records),
                "format": export_format,
                "deck_id": str(deck_id) if deck_id is not None else None,
            },
        )
        return data

    def create_flashcard_template(
        self,
        *,
        name: str,
        model_type: str = "basic",
        front_template: str,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        template = self._require_db().create_flashcard_template(
            name=name,
            model_type=model_type,
            front_template=front_template,
            back_template=back_template,
            notes_template=notes_template,
            extra_template=extra_template,
            placeholder_definitions=placeholder_definitions,
        )
        self._dispatch_local_notification(
            title="Local flashcard template created",
            message=f"Created local flashcard template: {template.get('name') or name}",
            source_entity_id=str(template.get("id") or ""),
            source_entity_kind="flashcard_template",
            payload={"action": "flashcard_template_created", "template_id": str(template.get("id") or "")},
        )
        return template

    def list_flashcard_templates(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        return dict(self._require_db().list_flashcard_templates(limit=limit, offset=offset) or {})

    def get_flashcard_template(self, template_id: str) -> dict[str, Any] | None:
        return self._require_db().get_flashcard_template(str(template_id))

    def update_flashcard_template(
        self,
        template_id: str,
        *,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        front_template: Optional[str] = None,
        back_template: Optional[str] = None,
        notes_template: Optional[str] = None,
        extra_template: Optional[str] = None,
        placeholder_definitions: Optional[list[dict[str, Any]]] = None,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any] | None:
        return self._require_db().update_flashcard_template(
            str(template_id),
            name=name,
            model_type=model_type,
            front_template=front_template,
            back_template=back_template,
            notes_template=notes_template,
            extra_template=extra_template,
            placeholder_definitions=placeholder_definitions,
            expected_version=expected_version,
        )

    def delete_flashcard_template(self, template_id: str, *, expected_version: int) -> bool:
        deleted = bool(
            self._require_db().delete_flashcard_template(
                str(template_id),
                expected_version=expected_version,
            )
        )
        if deleted:
            self._dispatch_local_notification(
                title="Local flashcard template deleted",
                message="Deleted local flashcard template.",
                source_entity_id=str(template_id),
                source_entity_kind="flashcard_template",
                payload={"action": "flashcard_template_deleted", "template_id": str(template_id)},
            )
        return deleted

    def upload_flashcard_asset(self, file_path: Any) -> dict[str, Any]:
        path = Path(file_path)
        content = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        asset = self._require_db().create_flashcard_asset(
            original_filename=path.name,
            mime_type=mime_type,
            content=content,
        )
        self._dispatch_local_notification(
            title="Local flashcard asset stored",
            message=f"Stored local flashcard asset: {asset.get('original_filename') or path.name}",
            source_entity_id=str(asset.get("asset_uuid") or ""),
            source_entity_kind="flashcard_asset",
            payload={"action": "flashcard_asset_uploaded", "asset_uuid": str(asset.get("asset_uuid") or "")},
        )
        return asset

    def get_flashcard_asset_content(self, asset_uuid: str) -> bytes | None:
        return self._require_db().get_flashcard_asset_content(str(asset_uuid))

    def delete_flashcard(
        self,
        card_id: str,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_flashcard(
                card_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def delete_deck(
        self,
        deck_id: str,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_deck(
                deck_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def get_next_review_candidate(self, *, deck_id: Optional[str] = None) -> dict[str, Any]:
        cards = self._require_db().get_due_flashcards(deck_id=deck_id, limit=1)
        if not cards:
            return {"card": None, "selection_reason": "none"}
        return {"card": cards[0], "selection_reason": "due"}

    def submit_flashcard_review(self, card_id: str, *, rating: int) -> dict[str, Any]:
        self._require_db().update_flashcard_review(card_id, rating)
        return {"card": self._require_db().get_flashcard(card_id), "rating": rating}

    def end_review_session(self, review_session_id: int) -> None:
        return None
