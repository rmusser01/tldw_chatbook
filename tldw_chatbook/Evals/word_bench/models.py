"""Typed models for the word bench engine.

A word bench measures the model's next-token distribution after each of a set
of snippets, under each of a set of targets. These types are the contract
between capture, analysis, storage, and execution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import InitVar, dataclass
from typing import Literal, Optional

PromptMode = Literal["raw", "chat"]
CanaryVerdict = Literal["pass", "degenerate", "unchecked"]

#: Preflight states mapped onto the design contract's readable status labels.
#:
#: ``"mode_unsupported"`` (the design spec's "raw mode unsupported by
#: endpoint" case) is produced by
#: ``capture_client._preflight_state_for_error`` specifically for a 404 on
#: the capture request -- ``capture_client._build_request`` always posts to
#: a fixed, mode-selected path, so a 404 reliably means that path does not
#: exist on this server. Every other 4xx stays the more generic
#: ``"no_logprobs"`` (both labels read "Blocked" below either way).
_STATUS_LABELS: dict[str, str] = {
    "ok": "Ready",
    "unreachable": "Unavailable",
    "no_logprobs": "Blocked",
    "mode_unsupported": "Blocked",
    "no_content_token": "Blocked",
}


@dataclass(frozen=True)
class Snippet:
    """One text fragment whose continuation is measured.

    ``id`` is assigned at authoring time and must be stable: positional ids
    would silently remap historical results when a dataset is reordered.
    ``text_hash`` lets a grid flag "this snippet was edited after the run".
    """

    id: str
    text: str
    group: Optional[str] = None
    note: Optional[str] = None

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Target:
    """One (provider, model, steering) column of the grid.

    Steering is mode-specific and deliberately not one field: chat mode
    delivers a ``system_prompt`` as a system message, raw mode prepends a
    literal ``prefix``. Raw completions have no system-message slot, so
    silently concatenating one would change what is measured while claiming
    not to.
    """

    id: str
    name: str
    provider: str
    model_id: str
    prefix: Optional[str] = None
    system_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        if self.prefix is not None and self.system_prompt is not None:
            raise ValueError(
                f"Target {self.name!r} sets both prefix and system_prompt; "
                "a target belongs to exactly one prompt mode."
            )

    def is_valid_for_mode(self, mode: PromptMode) -> bool:
        if mode == "raw":
            return self.system_prompt is None
        return self.prefix is None


@dataclass(frozen=True)
class BenchConfig:
    """A word bench definition.

    ``strict`` (default ``True``) gates only the ``target_ids``-uniqueness
    check below, and is an ``InitVar`` -- accepted by ``__init__`` but not
    stored as a field, so it never appears in equality, ``repr``, or
    ``dataclasses.asdict``.

    Every WRITE call site (user-facing bench creation/edit, ``save_bench``,
    ``create_run_group``, ``WordBenchRunner.run``) must go through a
    ``strict=True`` construction (the default -- nothing needs to opt in)
    so a duplicate can never be created or run: every per-target map
    downstream (``WordBenchRunner``'s ``clients``, its preflight/canary
    dicts, ``storage.create_run_group``'s ``run_ids``) is keyed by target
    id, and a duplicate would otherwise silently collapse two targets into
    one with no error.

    ``storage.load_bench`` is the ONE exception, constructing with
    ``strict=False``: it rebuilds a ``BenchConfig`` from whatever is
    already sitting in ``eval_tasks.config_data``, including a bench saved
    before this validation existed (task-1132). Rejecting on READ would
    make that legacy bench permanently unopenable instead of merely
    unrunnable -- worse than the bug this validation fixes, since the user
    could no longer even see the duplicate to remove it. A lenient load
    preserves both ids exactly as stored so the bench editor and readiness
    inspector render every row (see their own duplicate-safe, index-keyed
    widget ids), and the run control stays blocked until the write path
    (which still validates unconditionally) accepts an edited config.

    ``target_ids`` element-type validation (a list/tuple of non-empty
    ``str``) is a SEPARATE check from the uniqueness one above and always
    runs, regardless of ``strict`` -- task-1132 only ever gated the
    uniqueness check; ``target_ids`` element shape was never validated
    anywhere, on read or write, before or after that fix. It is worth
    closing now because ``load_bench``'s read-leniency means this class
    deliberately accepts more from stored data than it used to: a
    corrupted ``config_data.target_ids`` entry (e.g. an int, or a nested
    list) previously loaded without complaint and only failed later, deep
    inside ``db.get_model(target_id)``, as an opaque sqlite
    parameter-binding error far from the actual cause (``eval_models.id``
    is ``TEXT``). Validating the shape here, unconditionally, turns that
    into a clear failure at the boundary instead.
    """

    name: str
    prompt_mode: PromptMode
    top_k: int
    dataset_id: str
    target_ids: tuple[str, ...]
    probes: tuple[str, ...] = ()
    description: str = ""
    concurrency: int = 1
    #: task-1710: opt into ALSO capturing a short continuation of every
    #: measured SNIPPET (not just the fixed canary prompt -- see
    #: ``PreflightResult.continuation``, task-1691's per-target sibling of
    #: this field), one extra request per cell in raw mode (free in chat
    #: mode -- see ``capture_client.WordBenchCaptureClient.
    #: capture_with_continuation``). Additive and defaulted to ``False`` so
    #: every existing bench, and every stored ``config_data`` that predates
    #: this field, keeps loading and keeps costing exactly what it costs
    #: today -- see ``storage.save_bench``/``load_bench``, which persist it
    #: the same way they already do ``concurrency``. A snippets x targets
    #: bench pays snippets x targets extra requests when this is on (raw
    #: mode), a real, user-visible cost that must be chosen, not inherited
    #: -- hence opt-in rather than always-on.
    capture_continuations: bool = False
    strict: InitVar[bool] = True

    def __post_init__(self, strict: bool) -> None:
        if self.prompt_mode not in ("raw", "chat"):
            raise ValueError(
                f"prompt_mode must be 'raw' or 'chat', got {self.prompt_mode!r}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        # Element-type validation, unlike the uniqueness check below, is NOT
        # gated by `strict`: it must catch genuinely malformed data on every
        # path, including the lenient `load_bench` read. It also has to run
        # BEFORE the `set(self.target_ids)` call below -- an unhashable
        # element (e.g. a nested list) would otherwise blow up that line
        # with an opaque TypeError instead of the diagnosable ValueError
        # this check produces.
        if not isinstance(self.target_ids, (list, tuple)):
            raise ValueError(
                f"target_ids must be a list or tuple, got {self.target_ids!r} "
                f"(type: {type(self.target_ids).__name__})"
            )
        for target_id in self.target_ids:
            if not isinstance(target_id, str) or not target_id:
                raise ValueError(
                    f"target_ids elements must be non-empty strings, got "
                    f"{target_id!r} (type: {type(target_id).__name__})"
                )
        if strict and len(set(self.target_ids)) != len(self.target_ids):
            # Every per-target map downstream (WordBenchRunner's `clients`,
            # its preflight/canary dicts, storage.create_run_group's
            # `run_ids`) is keyed by target id. A duplicate silently
            # collapses two targets into one -- caught here rather than
            # letting it surface as "the grid only has N-1 columns".
            duplicates = sorted(
                {tid for tid in self.target_ids if self.target_ids.count(tid) > 1}
            )
            raise ValueError(
                f"target_ids must be unique, got duplicates: {duplicates!r}"
            )


@dataclass(frozen=True)
class TokenProb:
    """One token and its log probability.

    ``bytes_`` is the raw UTF-8 of the token's surface form and is the
    identity key: two tokens with identical bytes emit identical text, in any
    tokenizer, with no escaping conventions to disagree about.

    ``token_id`` is recorded for debugging and provenance but is deliberately
    NOT used for matching. A token id is only meaningful within one model --
    id 1623 in one tokenizer is an unrelated token in another -- and the whole
    point of the grid is comparing distributions ACROSS models. Matching on it
    would silently equate unrelated tokens and report divergence 0 for
    completely disjoint distributions.
    """

    token: str
    logprob: float
    bytes_: tuple[int, ...] = ()
    token_id: Optional[int] = None

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    def identity(self) -> tuple:
        """A key comparable ACROSS models. See the class docstring.

        Always keyed in the SAME namespace, ``"bytes"``, whether or not the
        provider sent a ``bytes`` field: a bytes-less token (e.g. an OpenAI
        legacy completions response) falls back to the UTF-8 encoding of its
        surface text. Two disjoint namespaces (bytes vs. token) would make a
        bytes-carrying provider compared against a bytes-less one report
        maximal divergence for identical distributions -- the same defect
        matching on provider-local token ids would cause.
        """
        return ("bytes", self.bytes_ or tuple(self.token.encode("utf-8")))


@dataclass(frozen=True)
class CellCapture:
    """One measured (snippet, target) cell.

    ``continuation`` (task-1710) is a short, best-effort generated
    continuation of THIS cell's own snippet -- the per-cell sibling of
    ``PreflightResult.continuation`` (task-1691's per-target continuation of
    the fixed canary prompt) -- captured only when
    ``BenchConfig.capture_continuations`` is ``True``; see
    ``WordBenchCaptureClient.capture_with_continuation`` for how it is
    produced without ever perturbing this cell's own ``top_k``/
    ``k_returned``/``content_offset``. It is additive and defaults to
    ``""`` so every pre-existing construction, and every historical cell
    stored before this field existed (``storage.save_cell``/
    ``_cell_from_payload``), keeps working unchanged.
    """

    prompt_mode: PromptMode
    k_requested: int
    k_returned: int
    content_offset: int
    top_k: tuple[TokenProb, ...]
    canary: CanaryVerdict
    captured_at: str
    schema: str = "word_bench/1"
    continuation: str = ""

    @property
    def top1_mass(self) -> float:
        return self.top_k[0].prob if self.top_k else 0.0

    @property
    def truncated_mass(self) -> float:
        """Probability mass not observed. Clamped to [0, 1] against float drift."""
        observed = sum(t.prob for t in self.top_k)
        return max(0.0, min(1.0, 1.0 - observed))


@dataclass(frozen=True)
class CellError:
    """A cell that failed. Written as a row so 'failed' and 'not yet run' differ."""

    reason: str
    detail: str = ""


@dataclass(frozen=True)
class PreflightResult:
    """One target's readiness, resolved before a run.

    ``continuation`` (task-1691) is a short, best-effort generated
    continuation of ``capture_client.CANARY_PROMPT``, captured through the
    same steering the run itself uses -- see
    ``WordBenchCaptureClient.preflight``'s own docstring for how it is
    produced and why a failure to capture it degrades to ``""`` rather than
    blocking preflight. It is additive and defaults to ``""`` so every
    pre-existing construction, and every historical run snapshot recorded
    before this field existed, keeps working unchanged.
    """

    state: str
    k_returned: Optional[int]
    canary: CanaryVerdict
    detail: str = ""
    checked_at: str = ""
    continuation: str = ""

    @property
    def status_label(self) -> str:
        return _STATUS_LABELS.get(self.state, "Blocked")

    @property
    def is_warned(self) -> bool:
        """Ready but with a caveat the grid must carry."""
        return self.state == "ok" and self.canary == "degenerate"
