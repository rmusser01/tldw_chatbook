"""Lossless structural saving, independent of workflow execution admission."""

import json
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Utils.input_validation import (
    WORKFLOW_MAX_PAGE_SIZE as MAX_PAGE_SIZE,
)
from tldw_chatbook.Utils.input_validation import (
    WorkflowSearchInput,
    validate_json_size,
)
from tldw_chatbook.Workflows.expressions import (
    ExpressionError,
    pointer_child,
    reference_paths,
)
from tldw_chatbook.Workflows.models import (
    Draft,
    DraftConflict,
    FieldEdit,
    InvalidDraft,
    Issue,
    Revision,
    RevisionConflict,
)

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_DOCUMENT_STEPS = 500
MAX_DOCUMENT_DEPTH = 64
MAX_DOCUMENT_NODES = 100000
MAX_GENERATION = 2**63 - 1
PAGE_SIZE = 20
# A validation state, never serialized provenance or user-controlled error text.
FRAGMENT_ERROR = "Incomplete field edit; repair the field or explicitly accept repaired Advanced JSON"
_EDITABLE_STEP_KEYS = {
    "id",
    "type",
    "name",
    "description",
    "config",
    "retry",
    "timeout_seconds",
    "on_success",
}


class _Step(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class _Document(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    steps: list[_Step]
    metadata: dict[str, Any] = Field(default_factory=dict)


class _Identity(BaseModel):
    """Admit only lowercase hyphenated UUIDs, without rewriting authored JSON."""

    model_config = ConfigDict(strict=True, extra="allow")

    format_version: int = Field(ge=1, le=1)
    workflow_id: str
    revision_id: str
    parent_revision_ids: list[str]

    @field_validator("workflow_id", "revision_id")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        if str(UUID(value)) != value:
            raise ValueError("UUIDs must use canonical lowercase hyphenated spelling")
        return value

    @field_validator("parent_revision_ids")
    @classmethod
    def valid_parents(cls, values: list[str]) -> list[str]:
        for value in values:
            cls.valid_uuid(value)
        if len(set(values)) != len(values):
            raise ValueError("Duplicate revision parents")
        return values


def _check_text(raw: str) -> None:
    if not isinstance(raw, str):
        raise InvalidDraft("Workflow text must be a string")
    # Check characters before allocating a UTF-8 copy, then enforce actual bytes.
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise InvalidDraft("Workflow text exceeds the 16 MiB storage limit")
    try:
        within_limit = validate_json_size(raw, MAX_DOCUMENT_BYTES)
    except UnicodeEncodeError:
        raise InvalidDraft("Workflow text must be valid UTF-8") from None
    if not within_limit:
        raise InvalidDraft("Workflow text exceeds the 16 MiB storage limit")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidDraft("Duplicate JSON object keys are not allowed")
        result[key] = value
    return result


def _check_complexity(value: Any) -> None:
    """Bound all JSON values, including opaque children, without recursion.

    The root container has depth one; object keys are not value nodes. Keeping
    iterators on the stack bounds traversal storage by depth rather than width.
    """
    pending = [iter((value,))]
    nodes = 0
    while pending:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        nodes += 1
        if nodes > MAX_DOCUMENT_NODES:
            raise InvalidDraft(
                f"Workflow JSON exceeds {MAX_DOCUMENT_NODES} values/containers"
            )
        if isinstance(value, (dict, list)):
            if len(pending) > MAX_DOCUMENT_DEPTH:
                raise InvalidDraft(
                    f"Workflow JSON exceeds {MAX_DOCUMENT_DEPTH} container levels"
                )
            pending.append(iter(value.values() if isinstance(value, dict) else value))


def _non_json_number(_value: str) -> None:
    raise InvalidDraft("Non-JSON numeric values are not allowed")


class _JSONNumber:
    """Retain a decoder-validated numeric token without host numeric limits."""

    def __init__(self, token: str) -> None:
        self.token = token


@dataclass(frozen=True)
class OpaqueNumber:
    """Exact numeric display token; never coerce this projection into a document."""

    text: str


def _integer(value: str) -> int | _JSONNumber:
    # Keep even integers beyond Python's decimal-to-int digit safety limit
    # lossless, without changing that process-wide limit.
    if value == "-0":
        return _JSONNumber(value)
    try:
        return int(value)
    except ValueError:
        return _JSONNumber(value)


def _encode(value: Any, *, display: bool = False) -> str:
    """Encode parsed JSON without rounding opaque decimal values to floats."""
    if isinstance(value, _JSONNumber):
        return value.token
    if display and isinstance(value, OpaqueNumber):
        return value.text
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(key) + ":" + _encode(item, display=display)
                for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_encode(item, display=display) for item in value) + "]"
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _serialize(document: dict) -> str:
    _check_complexity(document)
    try:
        raw = _encode(document)
    except RecursionError:
        raise InvalidDraft("Workflow JSON nesting is too deep") from None
    _check_text(raw)
    return raw


def _decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_non_json_number,
        parse_float=_JSONNumber,
        parse_int=_integer,
    )


def _decode(raw: str) -> Any:
    """One lossless decoder for both documents and explicitly edited fragments."""
    _check_text(raw)
    try:
        value = _decoder().decode(raw)
        _check_complexity(value)
        return value
    except json.JSONDecodeError as error:
        raise InvalidDraft(
            f"Invalid JSON at line {error.lineno}, column {error.colno}"
        ) from None
    except (RecursionError, OverflowError):
        raise InvalidDraft(
            "Workflow JSON nesting or numeric value is too large"
        ) from None


def _document(raw: str, base: Revision | None = None) -> tuple[dict, str]:
    try:
        document = _decode(raw)
        if (
            isinstance(document, dict)
            and isinstance(document.get("steps"), list)
            and len(document["steps"]) > MAX_DOCUMENT_STEPS
        ):
            raise InvalidDraft(f"Workflow exceeds {MAX_DOCUMENT_STEPS} steps")
        checked = _Document.model_validate(document)
        ids = [step.id for step in checked.steps]
        if len(set(ids)) != len(ids) or {"inputs", "last"}.intersection(ids):
            raise InvalidDraft("Step IDs must be unique and cannot be inputs or last")
        metadata = document.setdefault("metadata", {})
        if "tldw_workflow" not in metadata:
            if base is not None:
                raise InvalidDraft("Draft portable identity is required")
            metadata["tldw_workflow"] = {
                "format_version": 1,
                "workflow_id": str(uuid4()),
                "revision_id": str(uuid4()),
                "parent_revision_ids": [],
            }
            _check_complexity(document)
            raw = _serialize(document)
        identity = _Identity.model_validate(metadata["tldw_workflow"])
        if identity.revision_id in identity.parent_revision_ids:
            raise InvalidDraft("A revision cannot be its own parent")
        if base and (
            identity.workflow_id != base.workflow_id
            or identity.revision_id != base.revision_id
            or tuple(identity.parent_revision_ids) != base.parent_revision_ids
        ):
            raise InvalidDraft("Draft identity must match its saved base revision")
        return document, raw
    except json.JSONDecodeError as error:
        raise InvalidDraft(
            f"Invalid JSON at line {error.lineno}, column {error.colno}"
        ) from None
    except ValidationError:
        # Pydantic messages include input values and user-controlled field names.
        raise InvalidDraft("Invalid workflow structure or portable identity") from None
    except (RecursionError, OverflowError):
        raise InvalidDraft(
            "Workflow JSON nesting or numeric value is too large"
        ) from None


def _check_generation(generation: int) -> None:
    if type(generation) is not int or not 0 <= generation <= MAX_GENERATION:
        raise DraftConflict("Draft generation must be a nonnegative SQLite integer")


def _revision(row: sqlite3.Row) -> Revision:
    return Revision(
        row["workflow_id"],
        row["revision_id"],
        tuple(json.loads(row["parents_json"])),
        row["definition_json"],
    )


def _get_revision(
    cursor: sqlite3.Cursor, workflow_id: str, revision_id: str
) -> Revision:
    row = cursor.execute(
        "SELECT * FROM workflow_revisions WHERE workflow_id = ? AND revision_id = ?",
        (workflow_id, revision_id),
    ).fetchone()
    if row is None:
        raise RevisionConflict("The base revision does not belong to this workflow")
    return _revision(row)


def _get_draft(
    cursor: sqlite3.Cursor, workflow_id: str, base_revision_id: str
) -> Draft | None:
    row = cursor.execute(
        "SELECT * FROM workflow_drafts WHERE workflow_id = ? AND base_revision_id = ?",
        (workflow_id, base_revision_id),
    ).fetchone()
    return Draft(**dict(row)) if row else None


def _insert_revision(cursor: sqlite3.Cursor, revision: Revision) -> None:
    cursor.execute(
        "INSERT INTO workflow_revisions (workflow_id, revision_id, parents_json, definition_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            revision.workflow_id,
            revision.revision_id,
            json.dumps(revision.parent_revision_ids),
            revision.raw_json,
            datetime.now(UTC).isoformat(timespec="microseconds"),
        ),
    )


def _pointer_keys(pointer: str) -> list[str]:
    if pointer and not pointer.startswith("/"):
        raise InvalidDraft("Expected an absolute JSON pointer")
    return [key.replace("~1", "/").replace("~0", "~") for key in pointer.split("/")[1:]]


def _dependency_issues(
    document: dict, original_indices: dict | None = None, *, describe: bool = False
) -> tuple[Issue, ...]:
    positions = {step["id"]: index for index, step in enumerate(document["steps"])}
    issues = []

    def visit(value, pointer, index):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, pointer_child(pointer, key), index)
        elif isinstance(value, list):
            for key, item in enumerate(value):
                visit(item, pointer_child(pointer, key), index)
        elif isinstance(value, str):
            try:
                for path in reference_paths(value, pointer):
                    if path[0] == "inputs":
                        continue
                    if (
                        path[0] == "last"
                        or positions.get(path[0], len(positions)) >= index
                    ):
                        issues.append(
                            Issue(
                                pointer,
                                "dependency",
                                f"{'.'.join(path)} must name an earlier step output",
                            )
                        )
                    elif describe:
                        issues.append(
                            Issue(pointer, "reference", "Consumes " + ".".join(path))
                        )
            except ExpressionError:
                issues.append(
                    Issue(
                        pointer,
                        "opaque_expression",
                        "Expression semantics are unverified",
                    )
                )

    for index, step in enumerate(document["steps"]):
        authored_index = (original_indices or positions).get(step["id"], index)
        pointer = f"/steps/{authored_index}"
        visit(step.get("config", {}), pointer + "/config", index)
        if "on_success" in step:
            route = step["on_success"]
            if (
                not isinstance(route, str)
                or route not in positions
                or positions[route] <= index
            ):
                issues.append(
                    Issue(
                        pointer + "/on_success",
                        "success_route",
                        "Explicit success route must target a later existing step",
                    )
                )
            elif describe:
                issues.append(
                    Issue(
                        pointer + "/on_success",
                        "success_route",
                        "Success routes to " + route,
                    )
                )
    return tuple(issues)


class DocumentService:
    """Persist immutable revisions and independent generation-checked drafts.

    Revision/draft values are immutable. Public projections are detached display
    data, never serialization inputs. Field edits reuse the private lossless codec.
    """

    def __init__(self, db: WorkflowsDB) -> None:
        self._db = db

    @staticmethod
    def validate_draft(
        base: Revision, raw_text: str, generation: int, previous: Draft | None = None
    ) -> Draft:
        """Validate without I/O; retain the last valid projection and raw buffer."""
        _check_generation(generation)
        _check_text(raw_text)
        if previous and (previous.workflow_id, previous.base_revision_id) != (
            base.workflow_id,
            base.revision_id,
        ):
            raise DraftConflict("Previous draft belongs to another base")
        projection = previous.last_valid_json if previous else base.raw_json
        error = None
        if previous and previous.error == FRAGMENT_ERROR:
            return Draft(
                base.workflow_id,
                base.revision_id,
                generation,
                raw_text,
                projection,
                FRAGMENT_ERROR,
            )
        if previous and raw_text == previous.raw_text:
            return replace(previous, generation=generation)
        try:
            _, projection = _document(raw_text, base)
        except InvalidDraft as invalid:
            error = str(invalid)[:256]
        return Draft(
            base.workflow_id, base.revision_id, generation, raw_text, projection, error
        )

    @staticmethod
    def project(raw_json: str) -> dict:
        """Return detached inspection data; extreme numeric tokens stay opaque.

        This is a display projection, not a serialization input. Edit through
        edit_field/edit_steps to preserve every untouched numeric token.
        """
        document, _ = _document(raw_json)

        def display(value):
            if isinstance(value, _JSONNumber):
                return OpaqueNumber(value.token)
            if isinstance(value, dict):
                return {key: display(item) for key, item in value.items()}
            if isinstance(value, list):
                return [display(item) for item in value]
            return value

        return display(document)

    @staticmethod
    def field_text(raw_json: str, pointer: str, *, as_json: bool = True) -> str:
        """Read a JSON pointer without numeric conversion, suitable for an editor."""
        value, _ = _document(raw_json)
        return DocumentService.projected_field_text(value, pointer, as_json=as_json)

    @staticmethod
    def projected_field_text(
        document: dict, pointer: str, *, as_json: bool = True
    ) -> str:
        """Read display text from a prepared projection without reparsing.

        Opaque number tokens remain exact, including inside composite fields.
        The returned field text is display data, never a serialized revision.
        """
        value = document
        try:
            for key in _pointer_keys(pointer):
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, ValueError, TypeError):
            return ""
        return (
            value
            if not as_json and isinstance(value, str)
            else _encode(value, display=True)
        )

    @staticmethod
    def json_location(
        raw_json: str, pointer: str
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Locate a value by JSON pointer, including arrays and escaped/repeated keys.

        The existing lossless decoder validates and skips values. The traversal
        records positions only; it does not create a second document projection.
        """
        try:
            _decode(raw_json)
            keys = _pointer_keys(pointer)
        except InvalidDraft:
            return None
        decoder = _decoder()

        def space(index):
            while index < len(raw_json) and raw_json[index] in " \t\r\n":
                index += 1
            return index

        def locate(index, remaining):
            index = space(index)
            if not remaining:
                return index, decoder.raw_decode(raw_json, index)[1]
            container = raw_json[index]
            if container not in "[{":
                return None
            index = space(index + 1)
            item = 0
            while raw_json[index] not in "]}":
                if container == "{":
                    key, index = decoder.raw_decode(raw_json, index)
                    index = space(space(index) + 1)  # validated colon
                else:
                    key = str(item)
                if key == remaining[0]:
                    return locate(index, remaining[1:])
                index = space(decoder.raw_decode(raw_json, index)[1])
                if raw_json[index] == ",":
                    index = space(index + 1)
                item += 1
            return None

        span = locate(0, keys)
        if span is None:
            return None
        return tuple(
            (raw_json.count("\n", 0, index), index - raw_json.rfind("\n", 0, index) - 1)
            for index in span
        )

    @staticmethod
    def field_editable(raw_json: str, pointer: str, value_type: str) -> bool:
        """Whether a native field can represent this shape without coercion."""
        document, _ = _document(raw_json)
        return DocumentService.projected_field_editable(document, pointer, value_type)

    @staticmethod
    def projected_field_editable(document: dict, pointer: str, value_type: str) -> bool:
        """Check native field shape against an already prepared projection."""
        from tldw_chatbook.Workflows.catalog import STEP_CONTRACTS

        value = document
        try:
            keys = _pointer_keys(pointer)
            if len(keys) > 2 and keys[0] == "steps":
                step = value["steps"][int(keys[1])]
                contract = next(
                    (item for item in STEP_CONTRACTS if item.step_type == step["type"]),
                    None,
                )
                if (
                    contract is None
                    or set(step) - _EDITABLE_STEP_KEYS
                    or set(step.get("config", {})) - set(contract.fields)
                ):
                    return False
            for key in keys:
                if isinstance(value, dict) and key not in value:
                    return True
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        expected = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return type(value) is expected.get(value_type)

    @staticmethod
    def edit_field(
        raw_json: str,
        pointer: str,
        text: str,
        *,
        as_json: bool = False,
        allow_invalid_fragment: bool = False,
    ) -> str:
        """Replace one explicit field losslessly, rejecting identity changes.

        JSON fragment editing uses the same decoder as revision storage. Missing
        dictionary parents are created; arrays must already exist. Raw JSON is
        the escape hatch for unsupported shapes, never an implicit conversion.
        """
        document, _ = _document(raw_json)
        original_identity = deepcopy(document["metadata"]["tldw_workflow"])
        keys = _pointer_keys(pointer)
        if not keys:
            raise InvalidDraft("Use the raw document editor for the document root")
        # Compatibility keyword deliberately cannot bypass standalone validation.
        # Incomplete input requires edit_fragment's verifiable source intent.
        value = _decode(text) if as_json else text
        target = document
        try:
            for key in keys[:-1]:
                target = (
                    target[int(key)]
                    if isinstance(target, list)
                    else target.setdefault(key, {})
                )
            if isinstance(target, list):
                target[int(keys[-1])] = value
            else:
                target[keys[-1]] = value
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            raise InvalidDraft(
                "This field shape requires explicit Advanced JSON editing"
            ) from None
        identity = document.get("metadata", {}).get("tldw_workflow", {})
        if not isinstance(identity, dict) or any(
            identity.get(key) != original_identity[key]
            for key in (
                "workflow_id",
                "revision_id",
                "parent_revision_ids",
                "format_version",
            )
        ):
            raise InvalidDraft("Portable identity cannot be changed by a field edit")
        raw = _serialize(document)
        _document(raw)
        return raw

    @classmethod
    def _fragment_parts(cls, edit: FieldEdit) -> tuple[str, str]:
        marker = "field_fragment_" + uuid4().hex
        raw = cls.edit_field(edit.source.raw_text, edit.pointer, marker)
        return tuple(raw.split(json.dumps(marker), 1))

    @classmethod
    def edit_fragment(cls, base: Revision, edit: FieldEdit, generation: int) -> Draft:
        """Compute a field result; incomplete standalone JSON cannot gain siblings."""
        source = edit.source
        checked = cls.validate_draft(base, source.raw_text, source.generation)
        if checked != source or source.error or generation <= source.generation:
            raise DraftConflict(
                "Field source identity, generation or projection changed"
            )
        _check_generation(generation)
        prefix, suffix = cls._fragment_parts(edit)
        raw = prefix + edit.text + suffix
        _check_text(raw)
        try:
            _decode(edit.text)
            # Re-check identity/structure on the assembled result, too.
            _, projection = _document(raw, base)
        except InvalidDraft:
            return Draft(
                base.workflow_id,
                base.revision_id,
                generation,
                raw,
                source.last_valid_json,
                FRAGMENT_ERROR,
            )
        return Draft(
            base.workflow_id, base.revision_id, generation, raw, projection, None
        )

    @classmethod
    def repair_draft(cls, base: Revision, source: Draft, generation: int) -> Draft:
        """Explicit whole-document adoption; no passive load or write calls this."""
        if (source.workflow_id, source.base_revision_id) != (
            base.workflow_id,
            base.revision_id,
        ) or generation <= source.generation:
            raise DraftConflict("Repair source identity or generation changed")
        repaired = cls.validate_draft(base, source.raw_text, generation)
        if repaired.error:
            raise InvalidDraft(repaired.error)
        return repaired

    @classmethod
    def _check_field_source(
        cls, base: Revision, previous: Draft | None, edit: FieldEdit
    ) -> None:
        if previous is None:
            return
        source = edit.source
        if previous.generation == source.generation:
            if previous != source:
                raise DraftConflict("Stored field source changed")
        elif previous.generation > source.generation:
            # Subsequent fragment keystrokes retain the original valid source.
            # Prove the stored buffer differs from that source at this field only.
            prefix, suffix = cls._fragment_parts(edit)
            if not previous.raw_text.startswith(
                prefix
            ) or not previous.raw_text.endswith(suffix):
                raise DraftConflict("Stored fragment source changed")
            text = previous.raw_text[len(prefix) : len(previous.raw_text) - len(suffix)]
            if (
                cls.edit_fragment(base, replace(edit, text=text), previous.generation)
                != previous
            ):
                raise DraftConflict("Stored fragment provenance does not match")
        elif previous.error == FRAGMENT_ERROR:
            raise DraftConflict("Repair the protected stored fragment explicitly")

    @staticmethod
    def revision_content(source: Revision, base: Revision) -> str:
        """Copy historical content onto an explicit base identity without I/O."""
        if source.workflow_id != base.workflow_id:
            raise RevisionConflict("Historical content belongs to another workflow")
        document, _ = _document(source.raw_json, source)
        identity, _ = _document(base.raw_json, base)
        for key in (
            "workflow_id",
            "revision_id",
            "parent_revision_ids",
            "format_version",
        ):
            document["metadata"]["tldw_workflow"][key] = identity["metadata"][
                "tldw_workflow"
            ][key]
        return _serialize(document)

    @staticmethod
    def dependency_issues(raw_json: str) -> tuple[Issue, ...]:
        """Explain unresolved order/routes or opaque expression semantics."""
        document, _ = _document(raw_json)
        return _dependency_issues(document)

    @staticmethod
    def step_dependencies(raw_json: str) -> tuple[Issue, ...]:
        """Describe consuming fields and explicit routes, including valid ones."""
        document, _ = _document(raw_json)
        return _dependency_issues(document, describe=True)

    @staticmethod
    def edit_steps(
        raw_json: str,
        operation: str,
        step_id: str = "",
        *,
        offset: int = 1,
        step_type: str = "prompt",
        before: bool = False,
    ) -> str:
        """Apply a checked structural edit; never rewrite semantic dependencies."""
        from tldw_chatbook.Workflows.catalog import STEP_CONTRACTS, new_step

        document, _ = _document(raw_json)
        steps = document["steps"]
        original_indices = {step["id"]: i for i, step in enumerate(steps)}
        index = next((i for i, step in enumerate(steps) if step["id"] == step_id), None)
        if operation != "add" and index is None:
            raise InvalidDraft("Selected step no longer exists")
        # Branches, nested orchestration and unknown execution controls cannot be
        # rearranged by a sequential editor merely because their JSON parses.
        known = _EDITABLE_STEP_KEYS
        contracts = {item.step_type: item for item in STEP_CONTRACTS}
        if operation in ("move", "delete", "add", "duplicate"):
            for i, step in enumerate(steps):
                if (
                    step["type"] not in contracts
                    or set(step) - known
                    or set(step.get("config", {})) - set(contracts[step["type"]].fields)
                ):
                    raise InvalidDraft(
                        f"/steps/{i}: opaque step semantics; inspect Advanced JSON before changing order"
                    )
        if operation == "delete":
            steps.pop(index)
        elif operation == "move":
            destination = index + offset
            if not 0 <= destination < len(steps):
                raise InvalidDraft("Step is already at this end of the workflow")
            steps.insert(destination, steps.pop(index))
        elif operation in ("add", "duplicate"):
            if len(steps) >= 100:
                raise InvalidDraft("The local authoring limit is 100 steps")
            step = (
                deepcopy(steps[index])
                if operation == "duplicate"
                else new_step(step_type)
            )
            step["id"] = "step_" + uuid4().hex
            if operation == "duplicate":
                step["name"] = str(step.get("name", step["type"])) + " copy"
            steps.insert(len(steps) if index is None else index + (not before), step)
        else:
            raise InvalidDraft("Unknown step operation")
        issues = _dependency_issues(document, original_indices)
        if issues:
            raise InvalidDraft(
                "Repair references or Cancel: "
                + "; ".join(issue.pointer + ": " + issue.message for issue in issues)
            )
        raw = _serialize(document)
        _document(raw)
        return raw

    def create(self, raw_json: str) -> Revision:
        """Create a workflow, initializing absent identity without matching titles.

        Raises:
            InvalidDraft: The document is structurally invalid or exceeds storage bounds.
            RevisionConflict: The workflow or revision identity already exists.
        """
        document, raw_json = _document(raw_json)
        identity = document["metadata"]["tldw_workflow"]
        revision = Revision(
            identity["workflow_id"],
            identity["revision_id"],
            tuple(identity["parent_revision_ids"]),
            raw_json,
        )
        with self._db.transaction() as cursor:
            if cursor.execute(
                "SELECT 1 FROM workflow_revisions WHERE workflow_id = ? OR revision_id = ? LIMIT 1",
                (revision.workflow_id, revision.revision_id),
            ).fetchone():
                raise RevisionConflict("Workflow or revision identity already exists")
            _insert_revision(cursor, revision)
            cursor.execute(
                "INSERT INTO workflow_heads VALUES (?, ?)",
                (revision.workflow_id, revision.revision_id),
            )
        return revision

    def get_revision(self, workflow_id: str, revision_id: str) -> Revision:
        """Return a detached saved revision, or raise RevisionConflict."""
        with self._db.transaction(write=False) as cursor:
            return _get_revision(cursor, workflow_id, revision_id)

    @staticmethod
    def _check_page(page_size: int, offset: int) -> None:
        if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("Page size must be an integer from 1 to 100")
        if type(offset) is not int or not 0 <= offset <= MAX_GENERATION:
            raise ValueError("Offset must be a nonnegative SQLite integer")

    def get_head(self, workflow_id: str) -> Revision | None:
        """Read one exact saved head independently of list pages or filters."""
        with self._db.transaction(write=False) as cursor:
            row = cursor.execute(
                "SELECT r.* FROM workflow_heads h JOIN workflow_revisions r "
                "ON r.revision_id = h.revision_id WHERE h.workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            return _revision(row) if row else None

    def list_revisions(
        self, workflow_id: str, *, page_size: int = PAGE_SIZE, offset: int = 0
    ) -> tuple[Revision, ...]:
        """Return a bounded page of immutable history in creation order."""
        self._check_page(page_size, offset)
        with self._db.transaction(write=False) as cursor:
            return tuple(
                _revision(row)
                for row in cursor.execute(
                    "SELECT * FROM workflow_revisions WHERE workflow_id = ? "
                    "ORDER BY created_at, rowid LIMIT ? OFFSET ?",
                    (workflow_id, page_size, offset),
                ).fetchall()
            )

    def list_workflows(
        self, *, page_size: int = PAGE_SIZE, offset: int = 0, query: str = ""
    ) -> tuple[Revision, ...]:
        """Read a bounded head page, optionally searching names across the library.

        Search scans bounded name/identity batches without retaining unmatched
        full definitions. Unicode casefold matching agrees with the editor.

        Args:
            page_size: Maximum number of matching heads, from 1 to 100.
            offset: Nonnegative SQLite integer offset within matching heads.
            query: Display-name substring, at most 512 Python characters;
                empty returns an unfiltered page. No normalization is applied.

        Returns:
            Full saved revisions in workflow-ID order, never pending draft text.

        Raises:
            ValueError: Paging bounds are invalid, including boolean bounds,
                or the raw query exceeds 512 characters.
            TypeError: The search query is not text.
        """
        parameters = WorkflowSearchInput(
            query=query, page_size=page_size, offset=offset
        )
        with self._db.transaction(write=False) as cursor:
            if parameters.query:
                return tuple(
                    _get_revision(cursor, workflow_id, revision_id)
                    for _, workflow_id, revision_id in self._workflow_summaries(
                        cursor,
                        parameters.page_size,
                        parameters.offset,
                        parameters.query,
                    )
                )
            return tuple(
                _revision(row)
                for row in cursor.execute(
                    "SELECT r.* FROM workflow_heads h JOIN workflow_revisions r "
                    "ON r.revision_id = h.revision_id ORDER BY h.workflow_id LIMIT ? OFFSET ?",
                    (parameters.page_size, parameters.offset),
                ).fetchall()
            )

    def list_workflow_summaries(
        self, *, page_size: int = PAGE_SIZE, offset: int = 0, query: str = ""
    ) -> tuple[tuple[str, str, str], ...]:
        """Read only display names and identities for a bounded library page.

        Args:
            page_size: Maximum number of matching heads, from 1 to 100.
            offset: Nonnegative SQLite integer offset within matching heads.
            query: Unicode casefold substring of the displayed name, or empty;
                at most 512 Python characters, without normalization.

        Returns:
            Tuples of name, workflow ID and revision ID, ordered by workflow ID.
            Names are display data, not validation results. Legacy definitions
            beyond authoring limits remain listed; saved bytes stay untouched.

        Raises:
            ValueError: Paging bounds are invalid, including boolean bounds,
                or the raw query exceeds 512 characters.
            TypeError: The search query is not text.
        """
        parameters = WorkflowSearchInput(
            query=query, page_size=page_size, offset=offset
        )
        with self._db.transaction(write=False) as cursor:
            return self._workflow_summaries(
                cursor, parameters.page_size, parameters.offset, parameters.query
            )

    @staticmethod
    def _workflow_summaries(
        cursor: sqlite3.Cursor, page_size: int, offset: int, query: str
    ) -> tuple[tuple[str, str, str], ...]:
        sql = (
            "SELECT CASE WHEN json_type(r.definition_json, '$.name') IS NULL "
            "THEN 'Untitled workflow' "
            "WHEN json_type(r.definition_json, '$.name') = 'text' "
            "THEN CAST(json_extract(r.definition_json, '$.name') AS BLOB) "
            "ELSE json_extract(r.definition_json, '$.name') "
            "END AS name, h.workflow_id, h.revision_id "
            "FROM workflow_heads h JOIN workflow_revisions r "
            "ON r.revision_id = h.revision_id ORDER BY h.workflow_id"
        )

        def display_name(row: sqlite3.Row) -> str:
            # SQLite permits escaped lone surrogates in saved JSON. Preserve
            # those names without changing the connection's text factory.
            value = row["name"]
            return (
                value.decode("utf-8", errors="surrogatepass")
                if isinstance(value, bytes)
                else str(value)
            )

        if not query:
            return tuple(
                (display_name(row), row["workflow_id"], row["revision_id"])
                for row in cursor.execute(
                    sql + " LIMIT ? OFFSET ?", (page_size, offset)
                ).fetchall()
            )
        summaries = []
        matched = 0
        folded = query.casefold()
        cursor.execute(sql)
        while len(summaries) < page_size:
            rows = cursor.fetchmany(MAX_PAGE_SIZE)
            if not rows:
                break
            for row in rows:
                name = display_name(row)
                if folded not in name.casefold():
                    continue
                if matched >= offset:
                    summaries.append((name, row["workflow_id"], row["revision_id"]))
                matched += 1
                if len(summaries) == page_size:
                    break
        return tuple(summaries)

    def put_draft(
        self,
        workflow_id: str,
        base_revision_id: str,
        raw_text: str,
        generation: int,
        *,
        last_valid_json: str | None = None,
        field_edit: FieldEdit | None = None,
        repair_source: Draft | None = None,
    ) -> Draft:
        """Durably retain text and its last valid projection in one transaction.

        Equal generation/text replays are idempotent. Stale generations and
        equal generations with different text raise DraftConflict. Invalid
        bounded text is retained with a content-free error; oversized text
        raises InvalidDraft without replacing the previous recoverable buffer.
        """
        _check_generation(generation)
        _check_text(raw_text)
        with self._db.transaction() as cursor:
            base = _get_revision(cursor, workflow_id, base_revision_id)
            previous = _get_draft(cursor, workflow_id, base_revision_id)
            if field_edit is not None and repair_source is not None:
                raise DraftConflict("Choose field repair or whole-document repair")
            proven = None
            if field_edit is not None:
                proven = self.edit_fragment(base, field_edit, generation)
            elif repair_source is not None:
                proven = self.repair_draft(base, repair_source, generation)
            if proven and (
                proven.raw_text != raw_text
                or (
                    last_valid_json is not None
                    and proven.last_valid_json != last_valid_json
                )
            ):
                raise DraftConflict("Field or repair result does not match its source")
            if previous and generation <= previous.generation:
                if generation == previous.generation and raw_text == previous.raw_text:
                    if proven is not None and proven != previous:
                        raise DraftConflict(
                            "Persistence replay changes validation state"
                        )
                    return previous
                raise DraftConflict(
                    "A newer or different draft generation is already stored"
                )
            if field_edit is not None:
                self._check_field_source(base, previous, field_edit)
            if repair_source is not None and previous != repair_source:
                raise DraftConflict("Stored repair source changed; confirm again")
            draft = proven or self.validate_draft(base, raw_text, generation, previous)
            if (
                last_valid_json is not None
                and draft.error
                and proven is None
                and draft.error != FRAGMENT_ERROR
            ):
                if last_valid_json == base.raw_json:
                    # Retain the exact durable base even if newer admission
                    # bounds refuse its structure. The draft stays invalid.
                    projection = base.raw_json
                else:
                    _, projection = _document(last_valid_json, base)
                draft = replace(draft, last_valid_json=projection)
            cursor.execute(
                """INSERT INTO workflow_drafts
                   (workflow_id, base_revision_id, generation, raw_text, last_valid_json, error)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id, base_revision_id) DO UPDATE SET
                   generation = excluded.generation, raw_text = excluded.raw_text,
                   last_valid_json = excluded.last_valid_json, error = excluded.error""",
                (
                    workflow_id,
                    base_revision_id,
                    generation,
                    raw_text,
                    draft.last_valid_json,
                    draft.error,
                ),
            )
        return draft

    def get_draft(self, workflow_id: str, base_revision_id: str) -> Draft | None:
        """Return a detached durable buffer, or None when that draft is absent."""
        with self._db.transaction(write=False) as cursor:
            return _get_draft(cursor, workflow_id, base_revision_id)

    def list_drafts(
        self, workflow_id: str, *, page_size: int = PAGE_SIZE, offset: int = 0
    ) -> tuple[Draft, ...]:
        """Read a bounded page of buffers, including invalid/stale drafts."""
        self._check_page(page_size, offset)
        with self._db.transaction(write=False) as cursor:
            return tuple(
                Draft(**dict(row))
                for row in cursor.execute(
                    "SELECT * FROM workflow_drafts WHERE workflow_id = ? "
                    "ORDER BY base_revision_id LIMIT ? OFFSET ?",
                    (workflow_id, page_size, offset),
                ).fetchall()
            )

    def copy_draft_to_head(self, source: Draft, head: Revision) -> Draft:
        """Copy a confirmed valid source without changing its durable row.

        Source equality, current head and target cleanliness are checked under
        the same write transaction. Only portable base identity changes in the
        copy; no merge is performed. Invalid text must be repaired in place.

        Raises:
            DraftConflict: Source changed or the target already has edits.
            RevisionConflict: Target is not the exact current saved head.
            InvalidDraft: Source is invalid.
        """
        with self._db.transaction() as cursor:
            base = _get_revision(cursor, source.workflow_id, source.base_revision_id)
            if base.revision_id == head.revision_id:
                raise RevisionConflict("Recovery target head changed; confirm again")
            if (
                _get_draft(cursor, source.workflow_id, source.base_revision_id)
                != source
            ):
                raise DraftConflict("Recovery source changed; confirm again")
            if source.error:
                raise InvalidDraft("Repair raw JSON before copying the draft")
            return self._copy_clean_head(
                cursor,
                Revision(
                    base.workflow_id,
                    base.revision_id,
                    base.parent_revision_ids,
                    source.raw_text,
                ),
                head,
            )

    def copy_revision_to_head(self, source: Revision, head: Revision) -> Draft:
        """Transactionally copy immutable history; never stage it in another draft."""
        with self._db.transaction() as cursor:
            if _get_revision(cursor, source.workflow_id, source.revision_id) != source:
                raise RevisionConflict("Historical source changed")
            return self._copy_clean_head(cursor, source, head)

    def _copy_clean_head(
        self, cursor: sqlite3.Cursor, source: Revision, head: Revision
    ) -> Draft:
        target = _get_revision(cursor, source.workflow_id, head.revision_id)
        current_head = cursor.execute(
            "SELECT revision_id FROM workflow_heads WHERE workflow_id = ?",
            (source.workflow_id,),
        ).fetchone()
        if (
            target != head
            or current_head is None
            or current_head[0] != head.revision_id
        ):
            raise RevisionConflict("Recovery target head changed; confirm again")
        previous = _get_draft(cursor, head.workflow_id, head.revision_id)
        if previous and (
            previous.error
            or previous.raw_text != head.raw_json
            or previous.last_valid_json != head.raw_json
        ):
            raise DraftConflict(
                "Saved head has a local draft. Open it without overwriting."
            )
        raw = self.revision_content(source, head)
        generation = previous.generation + 1 if previous else 0
        draft = self.validate_draft(head, raw, generation)
        if draft.error is not None:
            raise InvalidDraft(draft.error)
        cursor.execute(
            """INSERT INTO workflow_drafts
                   (workflow_id, base_revision_id, generation, raw_text, last_valid_json, error)
                   VALUES (?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(workflow_id, base_revision_id) DO UPDATE SET
                   generation = excluded.generation, raw_text = excluded.raw_text,
                   last_valid_json = excluded.last_valid_json, error = NULL""",
            (
                draft.workflow_id,
                draft.base_revision_id,
                draft.generation,
                draft.raw_text,
                draft.last_valid_json,
            ),
        )
        return draft

    def save_revision(
        self, workflow_id: str, base_revision_id: str, expected_generation: int
    ) -> Revision:
        """Save one valid generation only if its base is still the current head.

        The new UUID/parent snapshot and compare-and-swap head update commit
        together. Keep the old draft's generation watermark so delayed writes
        cannot resurrect an earlier buffer after a save.

        Raises:
            RevisionConflict: The base is missing, foreign, or no longer the head.
            DraftConflict: The stored generation differs from the expected one.
            InvalidDraft: The draft is absent or invalid.
        """
        _check_generation(expected_generation)
        with self._db.transaction() as cursor:
            base = _get_revision(cursor, workflow_id, base_revision_id)
            head = cursor.execute(
                "SELECT revision_id FROM workflow_heads WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            if head is None or head[0] != base_revision_id:
                raise RevisionConflict("The workflow head has changed")
            draft = _get_draft(cursor, workflow_id, base_revision_id)
            if draft is None:
                raise InvalidDraft("No draft exists for this base revision")
            if draft.generation != expected_generation:
                raise DraftConflict("The stored draft generation has changed")
            if draft.error is not None:
                raise InvalidDraft(draft.error)
            document, _ = _document(draft.last_valid_json, base)
            revision_id = str(uuid4())
            document["metadata"]["tldw_workflow"].update(
                revision_id=revision_id, parent_revision_ids=[base_revision_id]
            )
            revision = Revision(
                workflow_id, revision_id, (base_revision_id,), _serialize(document)
            )
            _insert_revision(cursor, revision)
            cursor.execute(
                "UPDATE workflow_heads SET revision_id = ? WHERE workflow_id = ? AND revision_id = ?",
                (revision_id, workflow_id, base_revision_id),
            )
            if cursor.rowcount != 1:
                raise RevisionConflict("The workflow head has changed")
        return revision
