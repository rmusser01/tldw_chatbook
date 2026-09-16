"""One app-owned, in-memory sequential workflow; no execution persistence."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_chatbook.Chat.sampling_params import validate_sampling_params
from tldw_chatbook.LLM_Calls.llamacpp_bounded import (
    BoundedLlamaError,
    BoundedLlamaRequest,
    complete_llama_bounded,
    estimate_llama_reservation,
    resolve_llama_loopback_url,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.expressions import (
    IDENTIFIER,
    ExpressionError,
    json_copy,
    reference_paths,
    resolve_value,
)
from tldw_chatbook.Workflows.local_steps import (
    LocalNoteCleanupError,
    LocalNoteDestination,
    capture_local_note_destination,
    create_local_note,
    read_local_note,
    read_local_text,
)
from tldw_chatbook.Workflows.models import InvalidDraft, Revision
from tldw_chatbook.Workflows.session_permissions import (
    EffectRequest,
    WorkflowPermissions,
)

_DEFINITION_BYTES = 2 * 1024 * 1024
_INPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_BYTES = 1024 * 1024
_AGGREGATE_BYTES = 100 * 1024 * 1024
_ACTIVE_SECONDS = 60 * 60
_TOKEN_UNITS = 100_000
_SAMPLING = {"temperature", "top_p", "top_k", "min_p", "seed"}
_CONFIG = {
    "media_ingest": ({"sources", "extraction"}, set()),
    "prompt": ({"template"}, set()),
    "llm": (
        {"provider", "model", "prompt", "max_tokens"},
        {"request_timeout_seconds"} | _SAMPLING,
    ),
    "wait_for_human": (
        {"instructions", "assigned_to_user_id"},
        {"timeout_seconds"},
    ),
    "notes": ({"action", "title", "content"}, set()),
}


class SessionError(ValueError):
    """Payload-free session failure identified by a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _keys(value: Any, required: set, optional: set, code: str) -> None:
    if (
        type(value) is not dict
        or not required <= value.keys()
        or value.keys() - (required | optional)
    ):
        raise SessionError(code)


def _number(value: Any, code: str, *, positive: bool = True) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise SessionError(code)


def _text(value: Any, code: str, *, nonblank: bool = False) -> None:
    if type(value) is not str or (nonblank and not value.strip()):
        raise SessionError(code)


def _schema(schema: Any) -> None:
    _keys(
        schema,
        {"type"},
        {"properties", "required", "minLength", "additionalProperties"},
        "input_schema",
    )
    if schema["type"] == "string":
        if schema.keys() - {"type", "minLength"}:
            raise SessionError("input_schema")
        minimum = schema.get("minLength", 0)
        if type(minimum) is not int or minimum < 0:
            raise SessionError("input_schema")
    elif schema["type"] == "object":
        if "minLength" in schema:
            raise SessionError("input_schema")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (
            type(properties) is not dict
            or type(required) is not list
            or any(type(key) is not str or key not in properties for key in required)
            or len(set(required)) != len(required)
            or type(schema.get("additionalProperties", True)) is not bool
        ):
            raise SessionError("input_schema")
        for child in properties.values():
            _schema(child)
    else:
        raise SessionError("input_schema")


def _validate_inputs(value: Any, schema: dict) -> None:
    if schema["type"] == "string":
        _text(value, "input_type")
        if len(value) < schema.get("minLength", 0):
            raise SessionError("input_min_length")
        return
    if type(value) is not dict:
        raise SessionError("input_type")
    properties = schema.get("properties", {})
    if not set(schema.get("required", [])) <= value.keys():
        raise SessionError("input_required")
    if (
        not schema.get("additionalProperties", True)
        and value.keys() - properties.keys()
    ):
        raise SessionError("input_additional")
    for key in value.keys() & properties.keys():
        _validate_inputs(value[key], properties[key])


def _requirements(value: Any) -> set[str]:
    if type(value) is not dict:
        raise SessionError("requirements")
    owned = set()
    kinds = set()
    for requirement in value.values():
        _keys(requirement, {"kind", "required", "input_keys"}, set(), "requirements")
        kind = requirement["kind"]
        keys = requirement["input_keys"]
        if (
            type(kind) is not str
            or kind not in {"file", "model", "actor", "notes"}
            or kind in kinds
            or requirement["required"] is not True
            or type(keys) is not list
            or len(keys) != {"file": 1, "model": 2, "actor": 1, "notes": 0}[kind]
            or any(
                type(key) is not str or not IDENTIFIER.fullmatch(key) for key in keys
            )
            or len(set(keys)) != len(keys)
            or owned.intersection(keys)
        ):
            raise SessionError("requirements")
        owned.update(keys)
        kinds.add(kind)
    return owned


def _config(step: dict) -> None:
    kind, config = step["type"], step["config"]
    _keys(config, *_CONFIG[kind], "config_fields")
    if kind == "media_ingest":
        sources = config["sources"]
        if type(sources) is not list or len(sources) != 1:
            raise SessionError("source_config")
        _keys(sources[0], {"uri"}, set(), "source_config")
        _text(sources[0]["uri"], "source_config", nonblank=True)
        if (
            config["extraction"] != {"extract_text": True}
            or config["extraction"]["extract_text"] is not True
        ):
            raise SessionError("source_config")
    elif kind == "prompt":
        _text(config["template"], "prompt_config")
    elif kind == "llm":
        for key in ("provider", "model", "prompt"):
            _text(config[key], "model_config", nonblank=key != "prompt")
        if type(config["max_tokens"]) is not int or config["max_tokens"] <= 0:
            raise SessionError("max_tokens")
        if "request_timeout_seconds" in config:
            timeout = config["request_timeout_seconds"]
            _number(timeout, "request_timeout")
            if timeout > step["timeout_seconds"]:
                raise SessionError("request_timeout")
        sampling = {key: config[key] for key in config.keys() & _SAMPLING}
        for value in sampling.values():
            _number(value, "sampling", positive=False)
        if validate_sampling_params(sampling):
            raise SessionError("sampling")
    elif kind == "wait_for_human":
        _text(config["instructions"], "review_config")
        _text(config["assigned_to_user_id"], "review_actor", nonblank=True)
        _number(config.get("timeout_seconds", 0), "review_timeout", positive=False)
    elif kind == "notes":
        if config["action"] != "create":
            raise SessionError("note_action")
        _text(config["title"], "note_title", nonblank=True)
        _text(config["content"], "note_content")


def _references(value: Any, available: dict) -> None:
    if type(value) is dict:
        for child in value.values():
            _references(child, available)
    elif type(value) is list:
        for child in value:
            _references(child, available)
    elif type(value) is str:
        for path in reference_paths(value):
            current = available
            for key in path:
                if type(current) is not dict or key not in current:
                    raise SessionError("reference")
                current = current[key]


def _schema_shape(schema: dict) -> Any:
    if schema["type"] == "string":
        return None
    return {
        key: _schema_shape(child) for key, child in schema.get("properties", {}).items()
    }


def admit_definition(revision: Revision) -> dict[str, Any]:
    """Detach a saved revision and refuse unsupported execution before effects.

    Raises:
        SessionError: Invalid, oversized or unsupported definition (code only).
    """
    try:
        raw = revision.raw_json
        if (
            type(raw) is not str
            or len(raw) > _DEFINITION_BYTES
            or len(raw.encode("utf-8")) > _DEFINITION_BYTES
        ):
            raise SessionError("definition_limit")
        document = json_copy(DocumentService.project(raw), byte_limit=_DEFINITION_BYTES)
        _keys(
            document,
            {"name", "version", "steps", "metadata"},
            {"inputs", "description"},
            "definition_fields",
        )
        _text(document["name"], "definition_name", nonblank=True)
        if type(document["version"]) is not int or document["version"] < 1:
            raise SessionError("definition_version")
        _keys(document["metadata"], {"tldw_workflow"}, set(), "metadata_fields")
        identity = document["metadata"]["tldw_workflow"]
        _keys(
            identity,
            {"format_version", "workflow_id", "revision_id", "parent_revision_ids"},
            {"requirements", "input_schema"},
            "metadata_fields",
        )
        if (
            identity["workflow_id"],
            identity["revision_id"],
            tuple(identity["parent_revision_ids"]),
        ) != (revision.workflow_id, revision.revision_id, revision.parent_revision_ids):
            raise SessionError("revision_identity")
        owned = _requirements(identity.get("requirements", {}))
        schema = identity.get("input_schema", {"type": "object"})
        _schema(schema)
        if schema["type"] != "object" or type(document.get("inputs", {})) is not dict:
            raise SessionError("input_schema")
        inputs = {
            **document.get("inputs", {}),
            **_schema_shape(schema),
            **dict.fromkeys(owned),
        }
        available = {"inputs": inputs}
        steps = document["steps"]
        if not 1 <= len(steps) <= 100:
            raise SessionError("step_limit")
        previous = None
        for step in steps:
            _keys(
                step,
                {"id", "type", "config"},
                {"retry", "timeout_seconds", "name", "description"},
                "step_fields",
            )
            if (
                not IDENTIFIER.fullmatch(step["id"])
                or step["id"] in available
                or step["id"] == "last"
            ):
                raise SessionError("step_id")
            if step["type"] not in _CONFIG:
                raise SessionError("step_type")
            if type(step.get("retry")) is not int or step["retry"] != 0:
                raise SessionError("retry_unsupported")
            if (
                type(step.get("timeout_seconds")) is not int
                or step["timeout_seconds"] <= 0
            ):
                raise SessionError("attempt_timeout")
            _config(step)
            _references(step["config"], available)
            if step["type"] == "wait_for_human" and previous not in {
                "media_ingest",
                "prompt",
                "llm",
                "wait_for_human",
            }:
                raise SessionError("review_source")
            available[step["id"]] = (
                {"note_id": None} if step["type"] == "notes" else {"text": None}
            )
            previous = step["type"]
        return document
    except (
        InvalidDraft,
        ExpressionError,
        TypeError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ):
        raise SessionError("definition_invalid") from None


@dataclass(frozen=True)
class ModelSelection:
    """Captured non-secret model request settings, independent of config reloads."""

    provider_id: str
    selected_url: str = field(repr=False)
    model: str
    request_timeout_seconds: float = 120
    sampling: tuple[tuple[str, int | float], ...] = ()


@dataclass(frozen=True)
class RunSetup:
    """App-supplied selections awaiting worker capture and final confirmation."""

    source: Path = field(repr=False)
    model: ModelSelection
    review_actor: str = field(repr=False)
    protected_paths: tuple[Path, ...] = field(repr=False)


@dataclass(frozen=True)
class RunBindings:
    """Exact displayed launch destinations; imported JSON carries no authority."""

    source: Path = field(repr=False)
    model: ModelSelection
    notes: LocalNoteDestination = field(repr=False)
    dispatch_url: str = field(repr=False)
    review_actor: str = field(repr=False)
    protected_paths: tuple[Path, ...] = field(repr=False)


@dataclass(frozen=True)
class RunView:
    """Detached current projection; private payloads never enter its repr."""

    run_id: str | None
    workflow_id: str
    revision_id: str
    step_id: str | None
    state: str
    message_code: str | None = None
    review_text: str | None = field(default=None, repr=False)
    review_instructions: str | None = field(default=None, repr=False)
    note_id: str | None = None
    generation: int = 0
    pending_effect: EffectRequest | None = field(default=None, repr=False)


class _Stopped(Exception):
    """Internal control transfer after cancellation; never an effect receipt."""


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class WorkflowSession:
    """Own one sequential run on the app loop and retain all physical work.

    Admission, controls and subscribers belong to the constructing loop. Blocking
    effects use the existing domain owners, with workers retained until settled.
    """

    def __init__(
        self,
        permissions: WorkflowPermissions,
        *,
        notes_scope: Callable[[], NotesScopeService],
        notes_user: Callable[[], str],
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._permissions = permissions
        self._notes_scope = notes_scope
        self._notes_user = notes_user
        self._view: RunView | None = None
        self._callbacks: set[Callable[[], None]] = set()
        self._generation = 0
        self._serial = 0
        self._prepared = None
        self._consumed: tuple[int, str] | None = None
        self._captures: set[asyncio.Task] = set()
        self._run_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._model_task: asyncio.Task | None = None
        self._model_cancelled = False
        self._fenced = False
        self._stopped = False
        self._stop_code = "cancelled"
        self._drain_error: str | None = None
        self._wake = asyncio.Event()
        self._review_answer: bool | None = None
        self._review_valid = False
        self._review_deadline: float | None = None
        self._effect_answer: bool | None = None
        self._pending_effect: EffectRequest | None = None
        self._idle_seconds = 0.0
        self._bindings: RunBindings | None = None

    def view(self) -> RunView | None:
        """Return an immutable projection, never mutable execution context."""
        return self._view

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to changes; detaching the view does not affect execution."""
        self._callbacks.add(callback)
        return lambda: self._callbacks.discard(callback)

    def _publish(self, **changes: Any) -> None:
        self._generation += 1
        if self._view is not None:
            self._view = replace(self._view, generation=self._generation, **changes)
        for callback in tuple(self._callbacks):
            try:
                callback()
            except Exception:  # noqa: BLE001 - subscribers cannot release physical ownership
                # A detached/broken view cannot abandon the session's worker.
                self._callbacks.discard(callback)

    def _admission_open(self) -> None:
        if self._fenced or self._close_task is not None or self._drain_error:
            raise SessionError(self._drain_error or "session_closing")
        if self._run_task is not None and not self._run_task.done():
            raise SessionError("session_busy")

    def discard_setup(self) -> None:
        """Invalidate unused launch authority without abandoning capture work."""
        self._serial += 1
        self._prepared = None

    async def prepare(
        self, revision: Revision, inputs: dict[str, Any], setup: RunSetup
    ) -> int:
        """Validate and capture a detached launch, retaining cancelled workers.

        Raises:
            SessionError: Admission, capture, or superseded setup failure.
        """
        self._admission_open()
        self.discard_setup()
        serial = self._serial
        document = admit_definition(revision)
        try:
            if not isinstance(setup, RunSetup) or not isinstance(
                setup.model, ModelSelection
            ):
                raise SessionError("setup_invalid")
            model = setup.model
            for value in (
                model.provider_id,
                model.model,
                model.selected_url,
                setup.review_actor,
            ):
                _text(value, "setup_invalid", nonblank=True)
                value.encode("utf-8")
            _number(model.request_timeout_seconds, "request_timeout")
            if type(model.sampling) is not tuple:
                raise SessionError("sampling")
            sampling = {}
            for pair in model.sampling:
                if (
                    type(pair) is not tuple
                    or len(pair) != 2
                    or pair[0] not in _SAMPLING
                    or pair[0] in sampling
                ):
                    raise SessionError("sampling")
                _number(pair[1], "sampling", positive=False)
                sampling[pair[0]] = pair[1]
            if validate_sampling_params(sampling):
                raise SessionError("sampling")
            if (
                not isinstance(setup.source, Path)
                or not setup.source.is_absolute()
                or setup.source.suffix.lower() != ".txt"
            ):
                raise SessionError("source_binding")
            protected = tuple(setup.protected_paths)
            if any(
                not isinstance(path, Path) or not path.is_absolute()
                for path in protected
            ):
                raise SessionError("protected_paths")
            setup = replace(setup, protected_paths=protected)
            values = json_copy(inputs, byte_limit=_INPUT_BYTES)
            if type(values) is not dict:
                raise SessionError("input_type")
            values = {**document.get("inputs", {}), **values}
            identity = document["metadata"]["tldw_workflow"]
            bindings = {
                "file": (str(setup.source),),
                "model": (model.provider_id, model.model),
                "actor": (setup.review_actor,),
                "notes": (),
            }
            for requirement in identity.get("requirements", {}).values():
                values.update(
                    zip(
                        requirement["input_keys"],
                        bindings[requirement["kind"]],
                        strict=True,
                    )
                )
            values = json_copy(values, byte_limit=_INPUT_BYTES)
            _validate_inputs(values, identity.get("input_schema", {"type": "object"}))
            for step in document["steps"]:
                config = step["config"]
                context = {"inputs": values}
                if step["type"] == "media_ingest":
                    if resolve_value(config["sources"][0]["uri"], context) != str(
                        setup.source
                    ):
                        raise SessionError("source_binding")
                elif step["type"] == "llm":
                    if (
                        resolve_value(config["provider"], context) != model.provider_id
                        or resolve_value(config["model"], context) != model.model
                    ):
                        raise SessionError("model_binding")
                    if (
                        model.request_timeout_seconds > step["timeout_seconds"]
                        or config.get(
                            "request_timeout_seconds", model.request_timeout_seconds
                        )
                        != model.request_timeout_seconds
                    ):
                        raise SessionError("request_timeout")
                    if any(
                        config[key] != sampling.get(key)
                        for key in config.keys() & _SAMPLING
                    ):
                        raise SessionError("sampling_binding")
                elif (
                    step["type"] == "wait_for_human"
                    and resolve_value(config["assigned_to_user_id"], context)
                    != setup.review_actor
                ):
                    raise SessionError("review_actor")
            scope, user = self._notes_scope(), self._notes_user()
            if user != setup.review_actor:
                raise SessionError("review_actor")
        except Exception as error:
            if isinstance(error, SessionError):
                raise
            raise SessionError("setup_invalid") from None

        def capture() -> RunBindings:
            url = resolve_llama_loopback_url(model.selected_url)
            destination = capture_local_note_destination(scope, user_id=user)
            return RunBindings(
                setup.source,
                model,
                destination,
                url,
                setup.review_actor,
                setup.protected_paths,
            )

        task = asyncio.create_task(asyncio.to_thread(capture))
        self._captures.add(task)

        def captured(completed: asyncio.Task) -> None:
            self._captures.discard(completed)
            if not completed.cancelled() and isinstance(
                completed.exception(), LocalNoteCleanupError
            ):
                self._drain_error = "note_cleanup_failed"
                self._fenced = True
                self._publish(state="failed", message_code=self._drain_error)

        task.add_done_callback(captured)
        try:
            binding = await asyncio.shield(task)
        except asyncio.CancelledError:
            if serial == self._serial:
                self.discard_setup()
            raise
        except Exception:  # noqa: BLE001 - capture diagnostics must contain no private payload
            raise SessionError(self._drain_error or "capture_failed") from None
        self._admission_open()
        if serial != self._serial:
            raise SessionError("setup_stale")
        self._prepared = (serial, revision, document, values, binding)
        return serial

    def bindings(self, ticket: int) -> RunBindings:
        """Expose only the latest unused capture for displayed confirmation."""
        if (
            type(ticket) is not int
            or self._prepared is None
            or ticket != self._serial
            or ticket != self._prepared[0]
        ):
            raise SessionError("setup_stale")
        return self._prepared[4]

    def start(self, ticket: int) -> str:
        """Consume current confirmation once; duplicate delivery returns its ID."""
        if (
            type(ticket) is int
            and self._consumed
            and self._consumed[0] == ticket == self._serial
        ):
            return self._consumed[1]
        self._admission_open()
        binding = self.bindings(ticket)
        _, revision, document, values, _ = self._prepared
        self._prepared = None
        run_id = str(uuid4())
        self._consumed = (ticket, run_id)
        self._bindings = binding
        self._stopped = False
        self._stop_code = "cancelled"
        self._idle_seconds = 0.0
        self._pending_effect = None
        self._view = RunView(
            run_id, revision.workflow_id, revision.revision_id, None, "ready"
        )
        self._run_task = asyncio.create_task(self._run(document, values))
        self._publish()
        return run_id

    def run_bindings(self, run_id: str) -> RunBindings:
        """Return the current run's captured destination, including after completion.

        Raises:
            SessionError: No run, or the requested run is not the current singleton.
        """
        if self._view is None or self._bindings is None or self._view.run_id != run_id:
            raise SessionError("run_stale")
        return self._bindings

    def _control(self, run_id: str, step_id: str, state: str) -> bool:
        return bool(
            not self._fenced
            and not self._stopped
            and self._view
            and self._view.run_id == run_id
            and self._view.step_id == step_id
            and self._view.state == state
        )

    def update_review(self, run_id: str, step_id: str, text: str) -> bool:
        """Retain exact edits; invalid new input disables accepting any old value."""
        if (
            not self._control(run_id, step_id, "review")
            or self._review_answer is not None
        ):
            return False
        self._review_valid = False
        try:
            _text(text, "review_invalid")
            json_copy({"text": text}, byte_limit=_OUTPUT_BYTES)
        except (SessionError, ExpressionError):
            self._publish(review_text=None, message_code="review_invalid")
            return False
        self._review_valid = True
        self._publish(review_text=text, message_code=None)
        return True

    def answer_review(self, run_id: str, step_id: str, *, accept: bool) -> bool:
        """Atomically consume a timely response from the captured current actor."""
        if (
            type(accept) is not bool
            or not self._control(run_id, step_id, "review")
            or self._review_answer is not None
        ):
            return False
        if (
            self._review_deadline is not None
            and self._loop.time() >= self._review_deadline
        ):
            self._wake.set()
            return False
        try:
            if self._notes_user() != self._bindings.review_actor:
                return False
        except Exception:  # noqa: BLE001 - unavailable actor fails closed
            return False
        if accept and not self._review_valid:
            return False
        self._review_answer = accept
        self._wake.set()
        self._publish()
        return True

    def answer_effect(
        self, run_id: str, step_id: str, payload_json: str, *, approve: bool
    ) -> bool:
        """Consume only the exact pending resolved effect, once."""
        effect = self._pending_effect
        if (
            type(approve) is not bool
            or not self._control(run_id, step_id, "approval")
            or effect is None
            or self._effect_answer is not None
            or (effect.run_id, effect.step_id, effect.payload_json)
            != (run_id, step_id, payload_json)
        ):
            return False
        self._effect_answer = approve
        self._pending_effect = None
        self._wake.set()
        self._publish(pending_effect=None)
        return True

    def cancel(self, run_id: str) -> None:
        """Stop advancement; retain the slot until all physical work settles."""
        if (
            not self._view
            or self._view.run_id != run_id
            or not self._run_task
            or self._run_task.done()
        ):
            return
        self._stopped = True
        self._wake.set()
        if (
            self._model_task
            and not self._model_task.done()
            and not self._model_cancelled
        ):
            self._model_cancelled = True
            self._model_task.cancel()
        self._publish(state="stopping", message_code="cancelling", pending_effect=None)

    def begin_close(self) -> None:
        """Fence controls/effects while the controller flushes authoring drafts."""
        self._fenced = True
        self._wake.set()
        self._publish()

    def abort_close(self) -> None:
        """Reopen after failed draft flush; accepted cancellation is irreversible."""
        if self._close_task is None and not self._drain_error:
            self._fenced = False
            self._wake.set()
            self._publish()

    async def close(self) -> None:
        """Idempotently cancel and drain, independently of any cancelled waiter.

        Raises:
            SessionError: Physical cleanup failed; dependent owners must stay open.
        """
        if self._close_task is None:
            self.begin_close()
            self.discard_setup()
            if self._view and self._view.run_id:
                self.cancel(self._view.run_id)
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        pending = list(self._captures)
        if self._run_task is not None:
            pending.append(self._run_task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._drain_error:
            raise SessionError(self._drain_error)

    async def _wait(self, deadline: float | None = None, *, live: bool = False) -> None:
        start = self._loop.time()
        try:
            async with asyncio.timeout_at(deadline):
                await self._wake.wait()
                self._wake.clear()
        finally:
            if not live:
                self._idle_seconds += self._loop.time() - start

    async def _ready(self, *, live: bool = False, note: bool = False) -> None:
        while self._fenced and not self._stopped and not self._drain_error:
            self._publish(message_code="close_fenced")
            await self._wait(live=live)
        if self._drain_error:
            raise SessionError(self._drain_error)
        if self._stopped:
            raise _Stopped
        if note:
            await self._identity()

    async def _identity(self) -> None:
        # App-composed getters remain on the app loop, including readback after Stop.
        binding = self._bindings
        if (
            self._notes_scope() is not binding.notes.scope
            or self._notes_user() != binding.notes.user_id
            or binding.notes.scope.local_notes_service is not binding.notes.owner
        ):
            raise SessionError("note_destination_changed")

    def _check(self, effect: EffectRequest, approved: bool) -> Any:
        try:
            return self._permissions.check(effect, approved_once=approved)
        except Exception:  # noqa: BLE001 - authority errors are payload-free refusals
            raise SessionError("permission_unavailable") from None

    async def _authorize(self, effect: EffectRequest) -> bool:
        await self._ready()
        decision = await self._physical(
            asyncio.create_task(asyncio.to_thread(self._check, effect, False))
        )
        await self._ready()
        approved = False
        if decision.refusal_code == "approval_required":
            self._pending_effect = effect
            self._effect_answer = None
            self._publish(state="approval", pending_effect=effect)
            while self._effect_answer is None:
                await self._ready()
                if self._effect_answer is None:
                    await self._wait()
            approved = self._effect_answer
            if not approved:
                raise SessionError("effect_rejected")
        elif decision.refusal is not None:
            raise SessionError("permission_denied")
        await self._ready()
        self._publish(state="running", pending_effect=None, message_code=None)
        return approved

    def _before_effect(
        self, effect: EffectRequest, approved: bool, deadline: float
    ) -> None:
        # This worker stays owned even if queued before Stop or blocked by Quit.
        while True:
            asyncio.run_coroutine_threadsafe(
                self._ready(live=True, note=effect.kind == "note"), self._loop
            ).result()
            if self._loop.time() >= deadline:
                raise SessionError("attempt_timeout")
            decision = self._check(effect, approved)
            if decision.refusal is not None:
                raise SessionError("permission_denied")
            if not self._fenced:
                if self._stopped:
                    raise _Stopped
                return

    async def _physical(self, task: asyncio.Task, deadline: float | None = None) -> Any:
        if deadline is not None:
            done, _ = await asyncio.wait(
                {task}, timeout=max(0, deadline - self._loop.time())
            )
            if not done:
                self.cancel(self._view.run_id)
                self._stop_code = "attempt_timeout"
                self._publish(message_code="attempt_timeout")
        # No wait_for on a worker: cancellation of its wrapper is not retirement.
        return await asyncio.shield(task)

    async def _review(self, config: dict, previous: dict) -> dict:
        if config["assigned_to_user_id"] != self._bindings.review_actor:
            raise SessionError("review_actor")
        text = previous["text"]
        json_copy({"text": text}, byte_limit=_OUTPUT_BYTES)
        self._review_answer = None
        self._review_valid = True
        seconds = config.get("timeout_seconds", 0)
        self._review_deadline = self._loop.time() + seconds if seconds > 0 else None
        self._publish(
            state="review",
            review_text=text,
            review_instructions=config["instructions"],
            message_code=None,
        )
        while self._review_answer is None:
            if self._stopped:
                raise _Stopped
            try:
                await self._wait(self._review_deadline)
            except TimeoutError:
                raise SessionError("review_expired") from None
        if not self._review_answer:
            raise SessionError("review_rejected")
        return {"text": self._view.review_text}

    async def _note(
        self,
        effect: EffectRequest,
        config: dict,
        approved: bool,
        deadline: float,
        note_id: str,
    ) -> dict:
        binding = self._bindings
        cleanup_failed = False

        def write() -> str:
            self._before_effect(effect, approved, deadline)
            return create_local_note(
                binding.notes,
                create_note_id=note_id,
                title=config["title"],
                content=config["content"],
                before_write=lambda: self._before_effect(effect, approved, deadline),
            )

        try:
            await self._physical(
                asyncio.create_task(asyncio.to_thread(write)), deadline
            )
            confirmed = True
        except (_Stopped, SessionError):
            raise
        except Exception as error:  # noqa: BLE001 - any write response may follow a commit
            cleanup_failed = isinstance(error, LocalNoteCleanupError)

            # The same attempted ID and captured owner are the only receipt route.
            def reconcile() -> bool:
                asyncio.run_coroutine_threadsafe(self._identity(), self._loop).result()
                if self._check(effect, approved).refusal is not None:
                    return False
                row = read_local_note(binding.notes, note_id=note_id)
                return bool(
                    row
                    and row["title"] == config["title"].strip()
                    and row["content"] == config["content"]
                )

            try:
                confirmed = await self._physical(
                    asyncio.create_task(asyncio.to_thread(reconcile))
                )
            except Exception as read_error:  # noqa: BLE001 - failed readback is explicitly uncertain
                cleanup_failed |= isinstance(read_error, LocalNoteCleanupError)
                confirmed = False
        if confirmed:
            self._publish(note_id=note_id)
        if cleanup_failed:
            self._drain_error = "note_cleanup_failed"
            raise SessionError(self._drain_error)
        if not confirmed:
            raise SessionError("note_uncertain")
        return {"note_id": note_id}

    def _remaining_active(self) -> float:
        return _ACTIVE_SECONDS - (
            self._loop.time() - self._active_started - self._idle_seconds
        )

    async def _dispatch(self, step: dict, config: dict, previous: dict) -> dict:
        kind = step["type"]
        binding = self._bindings
        if kind == "prompt":
            return {"text": config["template"]}
        if kind == "wait_for_human":
            return await self._review(config, previous)
        note_id = str(uuid4()) if kind == "notes" else None
        payload = {"config": config}
        if kind == "media_ingest":
            if config["sources"][0]["uri"] != str(binding.source):
                raise SessionError("source_binding")
            payload["source"] = str(binding.source)
            effect_kind = "file"
        elif kind == "llm":
            if (
                config["provider"] != binding.model.provider_id
                or config["model"] != binding.model.model
            ):
                raise SessionError("model_binding")
            payload.update(
                dispatch_url=binding.dispatch_url,
                sampling=binding.model.sampling,
                request_timeout_seconds=binding.model.request_timeout_seconds,
            )
            effect_kind = "model"
        elif kind == "notes":
            payload.update(
                note_id=note_id,
                user=binding.notes.user_id,
                db_path=binding.notes.db_path,
            )
            effect_kind = "note"
        else:
            raise SessionError("step_type")
        effect = EffectRequest(
            self._view.run_id, step["id"], effect_kind, _encoded(payload)
        )
        approved = await self._authorize(effect)
        remaining = self._remaining_active()
        if remaining <= 0:
            raise SessionError("active_budget")
        deadline = self._loop.time() + min(step["timeout_seconds"], remaining)
        if kind == "media_ingest":

            def read() -> str:
                self._before_effect(effect, approved, deadline)
                return read_local_text(
                    binding.source,
                    protected_paths=binding.protected_paths,
                    before_read=lambda: self._before_effect(effect, approved, deadline),
                )

            return {
                "text": await self._physical(
                    asyncio.create_task(asyncio.to_thread(read)), deadline
                )
            }
        if kind == "llm":
            reservation = estimate_llama_reservation(
                config["prompt"], max_tokens=config["max_tokens"]
            )
            if reservation > self._tokens_remaining:
                raise SessionError("token_budget")
            self._tokens_remaining -= reservation
            while True:
                await self._ready()
                decision = await self._physical(
                    asyncio.create_task(
                        asyncio.to_thread(self._check, effect, approved)
                    ),
                    deadline,
                )
                if not self._fenced:
                    break
            if self._stopped:
                raise _Stopped
            if decision.refusal is not None:
                raise SessionError("permission_denied")
            request = BoundedLlamaRequest(
                binding.model.provider_id,
                binding.model.selected_url,
                binding.dispatch_url,
                binding.model.model,
                config["prompt"],
                config["max_tokens"],
                binding.model.request_timeout_seconds,
                binding.model.sampling,
            )
            self._model_cancelled = False
            self._model_task = asyncio.create_task(
                complete_llama_bounded(request, deadline_at=deadline)
            )
            try:
                return await self._physical(self._model_task, deadline)
            except asyncio.CancelledError:
                raise _Stopped from None
            except Exception as error:
                if not isinstance(error, BoundedLlamaError) or error.code == "cleanup":
                    self._drain_error = "model_cleanup_failed"
                    raise SessionError(self._drain_error) from None
                raise
            finally:
                self._model_task = None
        return await self._note(effect, config, approved, deadline, note_id)

    async def _run(self, document: dict, inputs: dict) -> None:
        context = {"inputs": inputs}
        previous = {}
        output_bytes = 0
        self._tokens_remaining = _TOKEN_UNITS
        self._active_started = self._loop.time()
        try:
            for step in document["steps"]:
                await self._ready()
                if self._remaining_active() <= 0:
                    raise SessionError("active_budget")
                self._publish(
                    step_id=step["id"],
                    state="running",
                    message_code=None,
                    review_text=None,
                    review_instructions=None,
                )
                resolved = resolve_value(
                    step["config"], context, byte_limit=_OUTPUT_BYTES
                )
                _config({**step, "config": resolved})
                result = await self._dispatch(step, resolved, previous)
                # A quit fence preserves this result in the retained coroutine.
                await self._ready()
                if self._remaining_active() <= 0:
                    raise SessionError("active_budget")
                result = json_copy(result, byte_limit=_OUTPUT_BYTES)
                output_bytes += len(_encoded(result).encode("utf-8"))
                if output_bytes > _AGGREGATE_BYTES:
                    raise SessionError("output_budget")
                context[step["id"]] = result
                previous = result
            self._publish(state="completed", message_code=None)
        except _Stopped:
            state = "failed" if self._stop_code == "attempt_timeout" else "cancelled"
            code = self._stop_code
            if self._view.note_id:
                code = (
                    "saved_after_timeout" if state == "failed" else "saved_after_cancel"
                )
            self._publish(state=state, message_code=code)
        except Exception as error:  # noqa: BLE001 - session failures expose codes, never payloads
            if isinstance(error, SessionError):
                code = error.code
            elif isinstance(error, BoundedLlamaError):
                code = "model_failed"
            elif isinstance(error, ExpressionError):
                code = "resolved_limit_or_type"
            else:
                code = "operation_failed"
            state = (
                "uncertain"
                if code == "note_uncertain"
                else "rejected"
                if code in {"review_rejected", "effect_rejected"}
                else "failed"
            )
            if self._drain_error:
                self._fenced = True
            self._publish(state=state, message_code=code, pending_effect=None)
