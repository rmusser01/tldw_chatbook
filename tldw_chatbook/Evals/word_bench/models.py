"""Typed models for the word bench engine.

A word bench measures the model's next-token distribution after each of a set
of snippets, under each of a set of targets. These types are the contract
between capture, analysis, storage, and execution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Optional

PromptMode = Literal["raw", "chat"]
CanaryVerdict = Literal["pass", "degenerate", "unchecked"]

#: Preflight states mapped onto the design contract's readable status labels.
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
    """A word bench definition."""

    name: str
    prompt_mode: PromptMode
    top_k: int
    dataset_id: str
    target_ids: tuple[str, ...]
    probes: tuple[str, ...] = ()
    description: str = ""
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.prompt_mode not in ("raw", "chat"):
            raise ValueError(
                f"prompt_mode must be 'raw' or 'chat', got {self.prompt_mode!r}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


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
    """One measured (snippet, target) cell."""

    prompt_mode: PromptMode
    k_requested: int
    k_returned: int
    content_offset: int
    top_k: tuple[TokenProb, ...]
    canary: CanaryVerdict
    captured_at: str
    schema: str = "word_bench/1"

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
    """One target's readiness, resolved before a run."""

    state: str
    k_returned: Optional[int]
    canary: CanaryVerdict
    detail: str = ""
    checked_at: str = ""

    @property
    def status_label(self) -> str:
        return _STATUS_LABELS.get(self.state, "Blocked")

    @property
    def is_warned(self) -> bool:
        """Ready but with a caveat the grid must carry."""
        return self.state == "ok" and self.canary == "degenerate"
