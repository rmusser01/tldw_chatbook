# Evals Rebuild PR 2 — Word Bench Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the word-bench engine — capture, analysis, storage, execution — as a self-contained package with no UI, verifiable on its own.

**Architecture:** A new `tldw_chatbook/Evals/word_bench/` package, deliberately not routed through `EvalRunner`. Five modules with one responsibility each: typed models, an HTTP capture seam, distribution analysis, `Evals_DB` mapping, and a grid runner. The engine's deliverable is a correct grid emitted as JSON; PR 3 renders it.

**Tech Stack:** Python 3.11+, httpx (already a dependency), pytest, SQLite. No new dependencies.

## Global Constraints

- Base branch: `feat/evals-pr1-retire-unreachable-ui` (PR #922, open). This PR is **stacked** on it; rebase onto `dev` once PR 1 merges.
- **A git worktree has no `.venv`.** Use the primary checkout's interpreter with cwd set to the worktree:
  ```bash
  cd <worktree> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...
  ```
  Verify `python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves **inside** the worktree before the first test run. If it resolves to the primary checkout, stop — tests would verify the wrong tree.
- **Never run `pytest Tests/UI`** (5,200+ tests, ~51 min, exceeds the platform's hard 10-minute per-call cap). This PR's tests live in `Tests/Evals/`, which is small and fast. Run `python -m pytest Tests/Evals -q` freely.
- **Do not modify anything under `tldw_chatbook/UI/`.** This PR has no UI.
- **Do not modify `Tests/UI/test_evals_deletion_guard.py`** or its 19-entry tuples — PR 1 owns them.
- `Evals_DB.SCHEMA_VERSION` goes 3 → 4. **Re-verify at merge** that no concurrent branch also took 4.
- The `timeout` command is not available.
- Do not push and do not create a pull request without explicit authorization.

## Verified facts this plan is built on

Probed against llama.cpp `b8795` on `http://127.0.0.1:9099` before planning. Fixtures are committed at `Tests/Evals/fixtures/word_bench/`.

**Both endpoints return the same shape**, and it carries a token `id`:

```json
"logprobs": {"content": [{
  "id": 4775, "token": " a", "bytes": [32, 97], "logprob": -0.698,
  "top_logprobs": [{"id": 4775, "token": " a", "bytes": [32,97], "logprob": -0.698}, …]
}]}
```

**Chat mode's first token is a control token.** Observed `<|channel>` at `logprob 0.0` (probability 1.0), alternatives at −21. Measuring position 0 in chat mode yields zero-entropy cells across the whole grid.

**Raw mode can be out-of-distribution.** The same model continues `"The protestors were met with"` sanely (`" a"` −0.70) but `"The capital of France is"` with `"thought"` (−0.44), not `" Paris"`.

Spec sections amended to match, in commit `2cf86c2df`.

## File Structure

All new. No existing file is modified except `Evals_DB.py` (Task 5).

| File | Responsibility |
|---|---|
| `tldw_chatbook/Evals/word_bench/__init__.py` | Package exports |
| `tldw_chatbook/Evals/word_bench/models.py` | Frozen dataclasses: `Snippet`, `Target`, `BenchConfig`, `TokenProb`, `CellCapture`, `CellError`, `PreflightResult` |
| `tldw_chatbook/Evals/word_bench/normalizer.py` | Provider response → `list[TokenProb]`; control-token detection; content-position selection |
| `tldw_chatbook/Evals/word_bench/capture_client.py` | The HTTP seam: both endpoints, neutral sampler, preflight + canary |
| `tldw_chatbook/Evals/word_bench/analysis.py` | Entropy, JSD over K+1 support, min-K truncation, probe resolution, spread, group means |
| `tldw_chatbook/Evals/word_bench/storage.py` | `Evals_DB` mapping: bench ↔ `eval_tasks`, run group ↔ `eval_runs`, cell ↔ `eval_results` |
| `tldw_chatbook/Evals/word_bench/runner.py` | Grid execution: row-major, sequential, cancel, progress |
| `tldw_chatbook/DB/Evals_DB.py` | **Modified**: `SCHEMA_VERSION` 3→4, `run_group_id` column + index |

Tests mirror the structure under `Tests/Evals/word_bench/`.

---

### Task 1: Typed models

**Files:**
- Create: `tldw_chatbook/Evals/word_bench/__init__.py`, `tldw_chatbook/Evals/word_bench/models.py`
- Test: `Tests/Evals/word_bench/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: every later task imports from `models.py`. Exact definitions below are the contract.

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/word_bench/test_models.py`:

```python
"""Word bench dataclass contracts."""

from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)


def test_snippet_is_frozen_and_carries_stable_id():
    s = Snippet(id="a1", text="The protestors were", group="loaded")
    assert s.text_hash, "snippet must expose a content hash for post-run edit detection"
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "changed"


def test_snippet_hash_tracks_text_not_id():
    a = Snippet(id="a1", text="same text")
    b = Snippet(id="b2", text="same text")
    c = Snippet(id="a1", text="other text")
    assert a.text_hash == b.text_hash
    assert a.text_hash != c.text_hash


def test_target_steering_field_is_mode_specific():
    raw = Target(id="t1", name="base", provider="llama_cpp", model_id="m", prefix="Note: ")
    chat = Target(id="t2", name="safe", provider="llama_cpp", model_id="m", system_prompt="Be safe.")
    assert raw.is_valid_for_mode("raw") is True
    assert raw.is_valid_for_mode("chat") is False
    assert chat.is_valid_for_mode("chat") is True
    assert chat.is_valid_for_mode("raw") is False


def test_target_without_steering_is_valid_in_both_modes():
    plain = Target(id="t3", name="plain", provider="llama_cpp", model_id="m")
    assert plain.is_valid_for_mode("raw") is True
    assert plain.is_valid_for_mode("chat") is True


def test_target_rejects_both_steering_fields_at_once():
    with pytest.raises(ValueError, match="prefix.*system_prompt|system_prompt.*prefix"):
        Target(id="t4", name="bad", provider="p", model_id="m", prefix="a", system_prompt="b")


def test_bench_config_rejects_unknown_prompt_mode():
    with pytest.raises(ValueError, match="prompt_mode"):
        BenchConfig(name="b", prompt_mode="telepathy", top_k=20, dataset_id="d", target_ids=("t1",))


def test_cell_capture_computes_truncated_mass():
    cap = CellCapture(
        prompt_mode="raw",
        k_requested=3,
        k_returned=3,
        content_offset=0,
        top_k=(
            TokenProb(token=" a", logprob=-0.5, bytes_=(32, 97), token_id=1),
            TokenProb(token=" b", logprob=-1.5, bytes_=(32, 98), token_id=2),
        ),
        canary="pass",
        captured_at="2026-07-26T00:00:00Z",
    )
    # exp(-0.5) + exp(-1.5) = 0.6065 + 0.2231 = 0.8296
    assert cap.truncated_mass == pytest.approx(1 - 0.8296, abs=1e-3)
    assert cap.top1_mass == pytest.approx(0.6065, abs=1e-3)


def test_cell_error_is_distinguishable_from_capture():
    err = CellError(reason="unreachable", detail="connection refused")
    assert err.reason == "unreachable"


def test_preflight_result_maps_to_contract_status_label():
    ok = PreflightResult(state="ok", k_returned=20, canary="pass")
    unreachable = PreflightResult(state="unreachable", k_returned=None, canary="unchecked")
    degenerate = PreflightResult(state="ok", k_returned=20, canary="degenerate")
    assert ok.status_label == "Ready"
    assert unreachable.status_label == "Unavailable"
    assert degenerate.status_label == "Ready"
    assert degenerate.is_warned is True
    assert ok.is_warned is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /private/tmp/tldw-evals-pr2
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_models.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'tldw_chatbook.Evals.word_bench'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Evals/word_bench/__init__.py`:

```python
"""Word bench: next-token distribution comparison across snippets and targets."""
```

Create `tldw_chatbook/Evals/word_bench/models.py`:

```python
"""Typed models for the word bench engine.

A word bench measures the model's next-token distribution after each of a set
of snippets, under each of a set of targets. These types are the contract
between capture, analysis, storage, and execution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
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

    ``token_id`` is the provider's token id where available (llama.cpp
    supplies it, OpenAI does not) and is the identity key when present --
    exact within a model, where string comparison across differing escaping
    conventions is not. ``bytes_`` is the fallback.
    """

    token: str
    logprob: float
    bytes_: tuple[int, ...] = ()
    token_id: Optional[int] = None

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    def identity(self) -> tuple:
        """A key that is stable within one model."""
        if self.token_id is not None:
            return ("id", self.token_id)
        if self.bytes_:
            return ("bytes", self.bytes_)
        return ("token", self.token)


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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_models.py -q
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/ Tests/Evals/word_bench/
git commit -m "feat(evals): word bench typed models

Frozen dataclasses forming the contract between capture, analysis,
storage, and execution. Steering is deliberately two fields -- prefix for
raw mode, system_prompt for chat -- because raw completions have no
system-message slot."
```

---

### Task 2: Response normalizer

**Files:**
- Create: `tldw_chatbook/Evals/word_bench/normalizer.py`
- Test: `Tests/Evals/word_bench/test_normalizer.py`
- Read (do not modify): `Tests/Evals/fixtures/word_bench/llamacpp_raw_completions.json`, `llamacpp_chat_completions.json`

**Interfaces:**
- Consumes: `TokenProb` from `models.py`.
- Produces:
  - `normalize_logprobs(payload: dict, *, want_content_token: bool) -> tuple[list[TokenProb], int]` — returns `(top_k, content_offset)`. Raises `NormalizerError` when the shape is unrecognized or no content token exists.
  - `is_control_token(token: str, logprob: float) -> bool`
  - `class NormalizerError(Exception)`

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/word_bench/test_normalizer.py`:

```python
"""Normalizer pinned to payloads captured from a live llama.cpp server.

The spec's predicted shapes were WRONG -- it expected a legacy
token->logprob dict on /v1/completions. Both endpoints actually return the
modern content[] form. These fixtures are the reason that was caught, so
they are the test, not documentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Evals.word_bench.normalizer import (
    NormalizerError,
    is_control_token,
    normalize_logprobs,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "word_bench"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_raw_completions_fixture_normalizes():
    top_k, offset = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    assert offset == 0
    assert len(top_k) == 5
    assert top_k[0].token == " a"
    assert top_k[0].logprob == pytest.approx(-0.698, abs=1e-2)
    assert top_k[0].token_id is not None, "llama.cpp supplies token ids; keep them"
    assert top_k[0].bytes_ == (32, 97)


def test_top_k_is_returned_in_descending_logprob_order():
    top_k, _ = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    logprobs = [t.logprob for t in top_k]
    assert logprobs == sorted(logprobs, reverse=True)


def test_chat_fixture_normalizes_with_the_same_shape():
    """Both endpoints share one shape -- this is the corrected assumption."""
    top_k, _ = normalize_logprobs(
        _load("llamacpp_chat_completions.json"), want_content_token=False
    )
    assert len(top_k) == 5
    assert all(t.token_id is not None for t in top_k)


def test_identity_prefers_token_id_when_present():
    top_k, _ = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    assert top_k[0].identity()[0] == "id"


def test_unrecognized_shape_raises_rather_than_guessing():
    with pytest.raises(NormalizerError, match="shape"):
        normalize_logprobs({"choices": [{"logprobs": {"top_logprobs": [{"a": -1.0}]}}]},
                           want_content_token=False)


def test_missing_logprobs_raises():
    with pytest.raises(NormalizerError, match="logprobs"):
        normalize_logprobs({"choices": [{"message": {"content": "hi"}}]},
                           want_content_token=False)


def test_control_tokens_are_detected_structurally():
    assert is_control_token("<|channel>", 0.0) is True
    assert is_control_token("<|im_start|>", 0.0) is True
    assert is_control_token("<start_of_turn>", -0.001) is True
    assert is_control_token(" a", -0.698) is False
    assert is_control_token("Paris", -0.2) is False


def test_a_bracketed_token_with_real_uncertainty_is_not_a_control_token():
    """Deterministic-ness is part of the signal; a genuinely uncertain
    bracket-shaped token is content (e.g. code, markup)."""
    assert is_control_token("<div>", -3.4) is False


def test_want_content_token_skips_leading_control_positions():
    """The reason chat mode needs this: position 0 was <|channel> at p=1.0."""
    payload = {
        "choices": [{"logprobs": {"content": [
            {"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0,
             "top_logprobs": [{"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0}]},
            {"id": 7, "token": " I", "bytes": [32, 73], "logprob": -0.9,
             "top_logprobs": [
                 {"id": 7, "token": " I", "bytes": [32, 73], "logprob": -0.9},
                 {"id": 8, "token": " Sure", "bytes": [32, 83], "logprob": -1.4},
             ]},
        ]}}]
    }
    top_k, offset = normalize_logprobs(payload, want_content_token=True)
    assert offset == 1, "must measure the first non-control position"
    assert top_k[0].token == " I"


def test_no_content_token_in_window_raises():
    payload = {
        "choices": [{"logprobs": {"content": [
            {"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0,
             "top_logprobs": [{"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0}]},
        ]}}]
    }
    with pytest.raises(NormalizerError, match="no_content_token"):
        normalize_logprobs(payload, want_content_token=True)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_normalizer.py -q
```

Expected: collection error — no module named `normalizer`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Evals/word_bench/normalizer.py`:

```python
"""Provider response -> a normalized top-K distribution.

Pinned to payloads captured from a live llama.cpp server, not to
documentation. The spec originally predicted two different shapes -- a
modern content[] form for chat and a legacy token->logprob dict for raw
completions. Observation showed both endpoints return the modern form and
carry a token id the spec had not anticipated.

A provider whose shape is not pinned by a fixture is not supported. Shapes
are never inferred.
"""

from __future__ import annotations

import re
from typing import Any

from .models import TokenProb

#: Tokens shaped like <|foo|>, <|foo>, or <foo_bar> -- chat-template markers.
_BRACKETED = re.compile(r"^<\|?[A-Za-z0-9_\-]+\|?>$")

#: A control token is near-deterministic. A bracket-shaped token the model is
#: genuinely uncertain about is content (markup, code), not template.
_CONTROL_LOGPROB_CEILING = -0.05

#: How many positions to search for a content token before giving up.
CONTENT_TOKEN_WINDOW = 8


class NormalizerError(Exception):
    """The response shape was unrecognized, or held no usable distribution."""


def is_control_token(token: str, logprob: float) -> bool:
    """Structural control-token test.

    Identified by shape plus near-certainty rather than a hardcoded list,
    because every chat template uses different markers.
    """
    return bool(_BRACKETED.match(token)) and logprob >= _CONTROL_LOGPROB_CEILING


def _content_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        choices = payload["choices"]
        logprobs = choices[0]["logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise NormalizerError(
            f"response carries no logprobs; got keys {list(payload)!r}"
        ) from exc
    if not isinstance(logprobs, dict) or "content" not in logprobs:
        raise NormalizerError(
            "unrecognized logprobs shape: expected a 'content' array "
            f"(got {list(logprobs) if isinstance(logprobs, dict) else type(logprobs)!r}). "
            "Capture a fixture for this provider before claiming support."
        )
    content = logprobs["content"]
    if not content:
        raise NormalizerError("logprobs.content was empty")
    return content


def _to_token_probs(entry: dict[str, Any]) -> list[TokenProb]:
    raw = entry.get("top_logprobs") or []
    out = [
        TokenProb(
            token=item["token"],
            logprob=float(item["logprob"]),
            bytes_=tuple(item.get("bytes") or ()),
            token_id=item.get("id"),
        )
        for item in raw
    ]
    out.sort(key=lambda t: t.logprob, reverse=True)
    return out


def normalize_logprobs(
    payload: dict[str, Any], *, want_content_token: bool
) -> tuple[list[TokenProb], int]:
    """Return ``(top_k, content_offset)``.

    Args:
        payload: The provider's decoded JSON response.
        want_content_token: When True (chat mode), skip leading control
            tokens and measure the first content position. When False (raw
            mode), measure position 0.

    Raises:
        NormalizerError: shape unrecognized, or no content token in window.
    """
    content = _content_positions(payload)

    if not want_content_token:
        return _to_token_probs(content[0]), 0

    for offset, entry in enumerate(content[:CONTENT_TOKEN_WINDOW]):
        if not is_control_token(entry.get("token", ""), float(entry.get("logprob", 0.0))):
            return _to_token_probs(entry), offset

    raise NormalizerError(
        "no_content_token: every position within the first "
        f"{CONTENT_TOKEN_WINDOW} was a control token. This target's template "
        "emits only control tokens in the measured window."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_normalizer.py -q
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/normalizer.py Tests/Evals/word_bench/test_normalizer.py
git commit -m "feat(evals): word bench response normalizer

Pinned to fixtures captured from a live llama.cpp server. Both endpoints
share the modern content[] shape and carry a token id, contradicting the
spec's original prediction.

Chat mode skips leading control tokens: the first token of an assistant
turn was observed as <|channel> at probability 1.0, which would have
produced a grid of identical zero-entropy cells."
```

---

### Task 3: Analysis

**Files:**
- Create: `tldw_chatbook/Evals/word_bench/analysis.py`
- Test: `Tests/Evals/word_bench/test_analysis.py`

**Interfaces:**
- Consumes: `CellCapture`, `TokenProb` from `models.py`.
- Produces:
  - `entropy(cap: CellCapture) -> float`
  - `divergence(a: CellCapture, b: CellCapture) -> tuple[float, bool]` — returns `(jsd, is_bounded)`; `is_bounded` is True when combined truncated mass exceeds `TRUNCATION_WARN_THRESHOLD`
  - `resolve_probe(cap: CellCapture, probe: str, *, ever_observed: bool) -> ProbeReading`
  - `class ProbeReading` with `state: Literal["observed", "bounded", "never_observed"]`, `logprob: float | None`
  - `spread(caps: Sequence[CellCapture]) -> float`
  - `group_means(rows) -> dict[str, float]`
  - `TRUNCATION_WARN_THRESHOLD: float = 0.25`

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/word_bench/test_analysis.py`:

```python
"""Distribution analysis. This module carries the methodology, so it carries
the bulk of the coverage."""

from __future__ import annotations

import math

import pytest

from tldw_chatbook.Evals.word_bench.analysis import (
    TRUNCATION_WARN_THRESHOLD,
    divergence,
    entropy,
    group_means,
    resolve_probe,
    spread,
)
from tldw_chatbook.Evals.word_bench.models import CellCapture, TokenProb


def _cap(pairs, k_returned=None, canary="pass"):
    top = tuple(
        TokenProb(token=t, logprob=math.log(p), token_id=i)
        for i, (t, p) in enumerate(pairs)
    )
    return CellCapture(
        prompt_mode="raw",
        k_requested=len(top),
        k_returned=k_returned if k_returned is not None else len(top),
        content_offset=0,
        top_k=top,
        canary=canary,
        captured_at="2026-07-26T00:00:00Z",
    )


def test_entropy_of_a_certain_distribution_is_zero():
    assert entropy(_cap([("a", 1.0)])) == pytest.approx(0.0, abs=1e-9)


def test_entropy_of_a_uniform_pair_is_ln_two():
    assert entropy(_cap([("a", 0.5), ("b", 0.5)])) == pytest.approx(math.log(2), abs=1e-9)


def test_entropy_accounts_for_unobserved_mass_as_one_bucket():
    """Half the mass unobserved must not be silently ignored."""
    e = entropy(_cap([("a", 0.5)]))
    assert e == pytest.approx(math.log(2), abs=1e-9)


def test_divergence_of_identical_distributions_is_zero():
    a = _cap([("x", 0.6), ("y", 0.4)])
    jsd, bounded = divergence(a, a)
    assert jsd == pytest.approx(0.0, abs=1e-9)
    assert bounded is False


def test_divergence_of_disjoint_distributions_is_maximal():
    a = _cap([("x", 1.0)])
    b = _cap([("y", 1.0)])
    jsd, _ = divergence(a, b)
    assert jsd == pytest.approx(math.log(2), abs=1e-6)


def test_divergence_is_symmetric():
    a = _cap([("x", 0.7), ("y", 0.3)])
    b = _cap([("x", 0.2), ("y", 0.8)])
    assert divergence(a, b)[0] == pytest.approx(divergence(b, a)[0], abs=1e-12)


def test_divergence_flags_bounded_when_truncated_mass_is_material():
    a = _cap([("x", 0.4)])   # 0.6 unobserved
    b = _cap([("x", 0.5)])   # 0.5 unobserved
    _, bounded = divergence(a, b)
    assert bounded is True, f"combined truncation exceeds {TRUNCATION_WARN_THRESHOLD}"


def test_divergence_truncates_both_cells_to_min_k():
    """A K=100 cell vs a K=20 cell must not have its divergence driven by K.

    Both are cut to min(k_returned) before comparison, so the rich cell's
    extra tail cannot inflate the number.
    """
    rich = _cap([("a", 0.5), ("b", 0.3), ("c", 0.1)], k_returned=3)
    poor = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)
    jsd_mixed, _ = divergence(rich, poor)
    rich_cut = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)
    jsd_even, _ = divergence(rich_cut, poor)
    assert jsd_mixed == pytest.approx(jsd_even, abs=1e-9)


def test_probe_observed_when_present_in_top_k():
    cap = _cap([(" Sure", 0.6), (" I", 0.4)])
    r = resolve_probe(cap, " Sure", ever_observed=True)
    assert r.state == "observed"
    assert r.logprob == pytest.approx(math.log(0.6), abs=1e-9)


def test_probe_bounded_when_absent_but_seen_elsewhere_in_the_run():
    cap = _cap([(" I", 0.9)])
    r = resolve_probe(cap, " Sure", ever_observed=True)
    assert r.state == "bounded"
    assert r.logprob == pytest.approx(math.log(0.9), abs=1e-9), "bound is the K-th logprob"


def test_probe_never_observed_is_distinct_from_bounded():
    """The tokenizer-difference case: a probe that never appears anywhere for
    this target is most likely not a token in its vocabulary at all, and must
    not be rendered as a comparable bound."""
    cap = _cap([(" I", 0.9)])
    r = resolve_probe(cap, " Sure", ever_observed=False)
    assert r.state == "never_observed"
    assert r.logprob is None


def test_spread_is_max_pairwise_divergence():
    a = _cap([("x", 1.0)])
    b = _cap([("x", 1.0)])
    c = _cap([("y", 1.0)])
    assert spread([a, b]) == pytest.approx(0.0, abs=1e-9)
    assert spread([a, b, c]) == pytest.approx(math.log(2), abs=1e-6)


def test_spread_of_a_single_cell_is_zero():
    assert spread([_cap([("x", 1.0)])]) == 0.0


def test_group_means_exclude_ungrouped_rows():
    rows = [
        ("loaded", 0.4),
        ("loaded", 0.2),
        ("neutral", 0.1),
        (None, 0.9),
    ]
    means = group_means(rows)
    assert means == {"loaded": pytest.approx(0.3), "neutral": pytest.approx(0.1)}
    assert None not in means
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_analysis.py -q
```

Expected: collection error — no module named `analysis`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Evals/word_bench/analysis.py`:

```python
"""Distribution analysis for the word bench.

Two properties are stated rather than hidden:

1. Divergence is a LOWER BOUND. Unobserved mass is lumped into one shared
   "other" symbol, which assumes both tails overlap perfectly when they may
   be disjoint. The error has a known direction.
2. Mixed K biases comparison. A K=100 cell and a K=20 cell have
   systematically different truncated mass, so both are cut to min(K) before
   comparison and the difference reflects behaviour rather than settings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

from .models import CellCapture

#: Combined truncated mass above which a divergence is annotated as a bound.
TRUNCATION_WARN_THRESHOLD = 0.25

ProbeState = Literal["observed", "bounded", "never_observed"]


@dataclass(frozen=True)
class ProbeReading:
    """One probe's value in one cell.

    ``bounded`` means the probe fell outside this cell's top-K, so its
    logprob is an upper bound, never a measurement. ``never_observed`` means
    it did not appear in top-K in ANY cell for this target across the whole
    run -- most likely it is not a single token in that model's vocabulary,
    and rendering it as a bound would invite a cross-model comparison that
    means nothing.
    """

    probe: str
    state: ProbeState
    logprob: Optional[float]


def _distribution(cap: CellCapture, k: Optional[int] = None) -> list[float]:
    """Probabilities over top-K plus one lumped 'other' bucket.

    The 'other' bucket is what makes this a distribution: without it the
    masses do not sum to 1 and divergence is undefined.
    """
    top = cap.top_k[:k] if k is not None else cap.top_k
    probs = [t.prob for t in top]
    observed = sum(probs)
    probs.append(max(0.0, 1.0 - observed))
    return probs


def entropy(cap: CellCapture) -> float:
    """Shannon entropy in nats over top-K plus the unobserved bucket."""
    return -sum(p * math.log(p) for p in _distribution(cap) if p > 0.0)


def _aligned(a: CellCapture, b: CellCapture, k: int) -> tuple[list[float], list[float]]:
    """Both cells as distributions over the union of their token identities."""
    a_top, b_top = a.top_k[:k], b.top_k[:k]
    a_map = {t.identity(): t.prob for t in a_top}
    b_map = {t.identity(): t.prob for t in b_top}
    keys = list(dict.fromkeys([*a_map, *b_map]))
    pa = [a_map.get(key, 0.0) for key in keys]
    pb = [b_map.get(key, 0.0) for key in keys]
    pa.append(max(0.0, 1.0 - sum(a_map.values())))
    pb.append(max(0.0, 1.0 - sum(b_map.values())))
    return pa, pb


def divergence(a: CellCapture, b: CellCapture) -> tuple[float, bool]:
    """Jensen-Shannon divergence in nats, and whether it is a material bound.

    Returns:
        ``(jsd, is_bounded)``. ``is_bounded`` is True when the two cells'
        combined unobserved mass exceeds ``TRUNCATION_WARN_THRESHOLD``, in
        which case the caller must render the value as ">= jsd".
    """
    k = min(a.k_returned, b.k_returned, len(a.top_k), len(b.top_k))
    pa, pb = _aligned(a, b, k)

    jsd = 0.0
    for p, q in zip(pa, pb):
        m = 0.5 * (p + q)
        if m <= 0.0:
            continue
        if p > 0.0:
            jsd += 0.5 * p * math.log(p / m)
        if q > 0.0:
            jsd += 0.5 * q * math.log(q / m)

    combined_truncation = pa[-1] + pb[-1]
    return max(0.0, jsd), combined_truncation > TRUNCATION_WARN_THRESHOLD


def resolve_probe(
    cap: CellCapture, probe: str, *, ever_observed: bool
) -> ProbeReading:
    """Read one probe out of a cell's top-K.

    Args:
        ever_observed: whether this probe appeared in top-K in ANY cell for
            this target across the run. Distinguishes "unlikely here" from
            "not a token in this vocabulary".
    """
    for tok in cap.top_k:
        if tok.token == probe:
            return ProbeReading(probe=probe, state="observed", logprob=tok.logprob)
    if not ever_observed:
        return ProbeReading(probe=probe, state="never_observed", logprob=None)
    bound = cap.top_k[-1].logprob if cap.top_k else None
    return ProbeReading(probe=probe, state="bounded", logprob=bound)


def spread(caps: Sequence[CellCapture]) -> float:
    """Max pairwise divergence across a row -- where targets disagree most."""
    if len(caps) < 2:
        return 0.0
    return max(
        divergence(caps[i], caps[j])[0]
        for i in range(len(caps))
        for j in range(i + 1, len(caps))
    )


def group_means(rows: Iterable[tuple[Optional[str], float]]) -> dict[str, float]:
    """Mean divergence per snippet group. Ungrouped rows are excluded."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for group, value in rows:
        if group is not None:
            buckets[group].append(value)
    return {g: sum(v) / len(v) for g, v in buckets.items()}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_analysis.py -q
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/analysis.py Tests/Evals/word_bench/test_analysis.py
git commit -m "feat(evals): word bench distribution analysis

JSD over top-K plus one lumped 'other' bucket, so the support sums to 1
and the divergence is well-defined. Two properties are surfaced rather
than hidden: it is a lower bound (tails are assumed to overlap), and
mixed-K cells are cut to min(K) so the number reflects behaviour rather
than settings.

Probes carry three states -- observed, bounded, never_observed -- because
a probe absent from every cell for a target is most likely not a token in
that vocabulary, not merely unlikely."
```

---

### Task 4: Capture client and preflight

**Files:**
- Create: `tldw_chatbook/Evals/word_bench/capture_client.py`
- Test: `Tests/Evals/word_bench/test_capture_client.py`

**Interfaces:**
- Consumes: `models.py`, `normalizer.py`.
- Produces:
  - `class WordBenchCaptureClient(base_url: str, api_key: str | None = None, timeout: float = 120.0)`
  - `async capture(snippet_text, target, mode, top_k) -> CellCapture | CellError`
  - `async preflight(target, mode, top_k) -> PreflightResult`
  - `NEUTRAL_SAMPLER: dict` — the pinned parameters
  - `CANARY_PROMPT: str`, `CANARY_EXPECT: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/word_bench/test_capture_client.py`. Uses a fake transport — no network:

```python
"""Capture client. Network is faked; the real-server contract is pinned by
the normalizer's fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Evals.word_bench.capture_client import (
    CANARY_EXPECT,
    CANARY_PROMPT,
    NEUTRAL_SAMPLER,
    WordBenchCaptureClient,
)
from tldw_chatbook.Evals.word_bench.models import CellCapture, CellError, Target

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "word_bench"
RAW = json.loads((FIXTURES / "llamacpp_raw_completions.json").read_text())


def _client(handler) -> WordBenchCaptureClient:
    transport = httpx.MockTransport(handler)
    return WordBenchCaptureClient(base_url="http://127.0.0.1:9099", transport=transport)


def test_neutral_sampler_does_not_collapse_the_distribution():
    """temperature must be 1.0, not 0 -- zero collapses what we measure."""
    assert NEUTRAL_SAMPLER["temperature"] == 1.0
    assert NEUTRAL_SAMPLER["top_p"] == 1.0
    assert NEUTRAL_SAMPLER["top_k"] == 0


@pytest.mark.asyncio
async def test_raw_mode_posts_to_completions_with_neutral_sampler():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("The protestors were met with", target, "raw", 5)

    assert seen["url"].endswith("/v1/completions")
    assert seen["body"]["max_tokens"] == 1
    assert seen["body"]["temperature"] == 1.0
    assert seen["body"]["top_p"] == 1.0
    assert seen["body"]["top_k"] == 0
    assert isinstance(result, CellCapture)
    assert result.content_offset == 0


@pytest.mark.asyncio
async def test_raw_mode_prepends_target_prefix_to_the_snippet():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m", prefix="Note: ")
    await _client(handler).capture("the snippet", target, "raw", 5)
    assert seen["body"]["prompt"] == "Note: the snippet"


@pytest.mark.asyncio
async def test_chat_mode_sends_system_prompt_as_a_message():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m",
                    system_prompt="Be careful.")
    await _client(handler).capture("the snippet", target, "chat", 5)

    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be careful."}
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "the snippet"}


@pytest.mark.asyncio
async def test_chat_mode_requests_a_window_not_a_single_token():
    """It must be able to skip leading control tokens."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    await _client(handler).capture("s", target, "chat", 5)
    assert seen["body"]["max_tokens"] > 1


@pytest.mark.asyncio
async def test_transport_failure_becomes_a_cell_error_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "unreachable"


@pytest.mark.asyncio
async def test_http_error_status_becomes_a_cell_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "http_error"


@pytest.mark.asyncio
async def test_preflight_reports_ok_and_actual_k():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 20)
    assert result.state == "ok"
    assert result.k_returned == 5, "must report K actually returned, not requested"
    assert result.status_label == "Ready"


@pytest.mark.asyncio
async def test_preflight_marks_a_degenerate_canary_without_blocking():
    """A model that continues the canary with nonsense is still runnable --
    it may be exactly what the user wants to study -- but the whole column
    must carry the warning."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)  # " a", not " Paris"

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.canary == "degenerate"
    assert result.state == "ok"
    assert result.is_warned is True


@pytest.mark.asyncio
async def test_preflight_passes_canary_when_expected_token_is_present():
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " Paris", "bytes": [], "logprob": -0.2,
            "top_logprobs": [{"id": 1, "token": " Paris", "bytes": [], "logprob": -0.2}],
        }]}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.canary == "pass"
    assert result.is_warned is False


@pytest.mark.asyncio
async def test_preflight_reports_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.state == "unreachable"
    assert result.status_label == "Unavailable"


@pytest.mark.asyncio
async def test_preflight_reports_no_logprobs_as_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.state == "no_logprobs"
    assert result.status_label == "Blocked"


def test_canary_expectation_is_a_widely_agreed_continuation():
    assert "capital of France" in CANARY_PROMPT
    assert any("Paris" in tok for tok in CANARY_EXPECT)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_capture_client.py -q
```

Expected: collection error — no module named `capture_client`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Evals/word_bench/capture_client.py`:

```python
"""The one HTTP seam for word bench capture.

Word bench calls are provider calls and follow the LLM_Calls precedent --
direct to the user's configured endpoint, no egress policy. That is only
safe because the endpoint comes from configuration and never from bench
content: a bench that could name its own endpoint would be an SSRF vector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from loguru import logger

from .models import CellCapture, CellError, PreflightResult, PromptMode, Target
from .normalizer import NormalizerError, normalize_logprobs

#: Pinned neutral sampling. Servers -- llama.cpp especially -- apply samplers
#: BEFORE reporting logprobs, so a server configured with top_k=40 would make
#: every number an artifact of that setting. temperature is 1.0, NOT 0:
#: temperature zero collapses the distribution being observed.
NEUTRAL_SAMPLER: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

#: Chat mode must look past leading control tokens, so it asks for a window.
CHAT_TOKEN_WINDOW = 8

#: Distribution sanity canary. Confirming a target RETURNS logprobs is not the
#: same as confirming they mean anything: a heavily chat-tuned model was
#: observed continuing this prompt with "thought" rather than " Paris".
CANARY_PROMPT = "The capital of France is"
CANARY_EXPECT = (" Paris", "Paris")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class WordBenchCaptureClient:
    """Captures one next-token distribution per call."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_request(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"model": target.model_id, **NEUTRAL_SAMPLER}
        if mode == "raw":
            prompt = f"{target.prefix}{snippet}" if target.prefix else snippet
            payload.update({"prompt": prompt, "max_tokens": 1, "logprobs": top_k})
            return f"{self._base_url}/v1/completions", payload

        messages: list[dict[str, str]] = []
        if target.system_prompt:
            messages.append({"role": "system", "content": target.system_prompt})
        messages.append({"role": "user", "content": snippet})
        payload.update(
            {
                "messages": messages,
                "max_tokens": CHAT_TOKEN_WINDOW,
                "logprobs": True,
                "top_logprobs": top_k,
            }
        )
        return f"{self._base_url}/v1/chat/completions", payload

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def capture(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> CellCapture | CellError:
        """Measure one cell. Never raises -- failures become CellError."""
        url, payload = self._build_request(snippet, target, mode, top_k)
        try:
            data = await self._post(url, payload)
        except httpx.HTTPStatusError as exc:
            return CellError(reason="http_error", detail=f"{exc.response.status_code}")
        except httpx.HTTPError as exc:
            return CellError(reason="unreachable", detail=str(exc))

        try:
            tokens, offset = normalize_logprobs(data, want_content_token=(mode == "chat"))
        except NormalizerError as exc:
            reason = "no_content_token" if "no_content_token" in str(exc) else "no_logprobs"
            return CellError(reason=reason, detail=str(exc))

        return CellCapture(
            prompt_mode=mode,
            k_requested=top_k,
            k_returned=len(tokens),
            content_offset=offset,
            top_k=tuple(tokens),
            canary="unchecked",
            captured_at=_utcnow(),
        )

    async def preflight(
        self, target: Target, mode: PromptMode, top_k: int
    ) -> PreflightResult:
        """Resolve a target's readiness, including distribution sanity.

        A degenerate canary does NOT block the run -- a target whose raw
        continuation is out-of-distribution may be exactly what a user wants
        to study. It downgrades to a warned state that every cell carries.
        """
        result = await self.capture(CANARY_PROMPT, target, mode, top_k)
        checked_at = _utcnow()

        if isinstance(result, CellError):
            state = result.reason if result.reason != "http_error" else "unreachable"
            if state not in ("unreachable", "no_logprobs", "no_content_token"):
                state = "no_logprobs"
            return PreflightResult(
                state=state, k_returned=None, canary="unchecked",
                detail=result.detail, checked_at=checked_at,
            )

        observed = {tok.token for tok in result.top_k}
        canary = "pass" if observed & set(CANARY_EXPECT) else "degenerate"
        if canary == "degenerate":
            logger.warning(
                "Word bench canary degenerate for target %s: %r continued with %r",
                target.name, CANARY_PROMPT,
                [t.token for t in result.top_k[:3]],
            )
        return PreflightResult(
            state="ok", k_returned=result.k_returned, canary=canary,
            checked_at=checked_at,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_capture_client.py -q
```

Expected: 13 passed.

- [ ] **Step 5: Verify against the live server**

A fixture proves the parser; only a live call proves the request. If `http://127.0.0.1:9099` is reachable, run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import asyncio, json
from tldw_chatbook.Evals.word_bench.capture_client import WordBenchCaptureClient
from tldw_chatbook.Evals.word_bench.models import Target

async def main():
    c = WordBenchCaptureClient(base_url="http://127.0.0.1:9099", timeout=180.0)
    t = Target(id="t", name="live", provider="llama_cpp",
               model_id="gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf")
    pf = await c.preflight(t, "raw", 5)
    print("preflight:", pf.state, "K=", pf.k_returned, "canary=", pf.canary, "warned=", pf.is_warned)
    cap = await c.capture("The protestors were met with", t, "raw", 5)
    print("cell:", [(x.token, round(x.logprob, 3)) for x in cap.top_k])
    print("offset:", cap.content_offset, "truncated:", round(cap.truncated_mass, 3))
asyncio.run(main())
PY
```

Expected: `preflight: ok K= 5 canary= degenerate warned= True` (this model fails the canary — that is the correct, informative outcome), and a sane cell distribution led by `" a"` or `" much"`. If the server is unreachable, record that in the report and move on; it is not a blocker.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/capture_client.py Tests/Evals/word_bench/test_capture_client.py
git commit -m "feat(evals): word bench capture client and preflight

Pins neutral sampling on every request -- temperature 1.0, not 0, since
zero collapses the distribution being measured -- because servers apply
samplers before reporting logprobs.

Preflight resolves readiness AND distribution sanity: a canary prompt
with a near-universally agreed continuation catches targets whose raw
continuation is out-of-distribution. It warns rather than blocks, since
a degenerate target may be what the user wants to study."
```

---

### Task 5: Schema v4 and storage

**Files:**
- Modify: `tldw_chatbook/DB/Evals_DB.py`
- Create: `tldw_chatbook/Evals/word_bench/storage.py`
- Test: `Tests/Evals/word_bench/test_storage.py`

**Interfaces:**
- Consumes: `models.py`.
- Produces:
  - `save_bench(db, config) -> str` (task id)
  - `load_bench(db, task_id) -> BenchConfig`
  - `create_run_group(db, task_id, config, targets, snippets) -> tuple[str, dict[str, str]]` — returns `(run_group_id, {target_id: run_id})`, snapshotting the resolved config into each run's `config_overrides`
  - `save_cell(db, run_id, snippet, capture_or_error) -> None`
  - `load_grid(db, run_group_id) -> dict` — the pivot: snapshot plus `{(snippet_id, target_id): CellCapture | CellError}`

- [ ] **Step 1: Add the shared fixtures**

`Evals_DB.create_run` validates that its `model_id` exists (`raise InputError` if not), and
`create_model` mints its own UUID. So a `Target.id` must be a **real `eval_models` row id** —
tests cannot invent one. Create `Tests/Evals/word_bench/conftest.py`:

```python
"""Shared fixtures. A Target's id must be a real eval_models row id:
Evals_DB.create_run rejects an unknown model_id, and create_model mints
its own UUID rather than accepting one."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import Snippet, Target


@pytest.fixture
def db(tmp_path):
    return EvalsDB(db_path=str(tmp_path / "evals.db"), client_id="test")


@pytest.fixture
def snippets():
    return [
        Snippet(id="s1", text="The protestors were", group="neutral"),
        Snippet(id="s2", text="The rioters were", group="loaded"),
    ]


@pytest.fixture
def targets(db):
    """Two real eval_models rows, returned as Targets carrying their ids."""
    base_id = db.create_model(name="base", provider="llama_cpp", model_id="m")
    steered_id = db.create_model(name="steered", provider="llama_cpp", model_id="m")
    return [
        Target(id=base_id, name="base", provider="llama_cpp", model_id="m"),
        Target(id=steered_id, name="steered", provider="llama_cpp", model_id="m",
               prefix="Be careful. "),
    ]


@pytest.fixture
def config(targets):
    from tldw_chatbook.Evals.word_bench.models import BenchConfig
    return BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id="d1", target_ids=tuple(t.id for t in targets),
        probes=(" Sure", " I"),
    )
```

Because target ids are now database-assigned UUIDs, **tests must key on `target.name` or on the
fixture objects, never on a literal `"t1"`.**

- [ ] **Step 2: Write the failing test**

Create `Tests/Evals/word_bench/test_storage.py`, using the fixtures above rather than
module-level constants:

```python
"""Storage round-trip on a real in-memory SQLite, per project convention."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import SCHEMA_VERSION, EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    Snippet,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import (
    create_run_group,
    load_bench,
    load_grid,
    save_bench,
    save_cell,
)


# db, snippets, targets, config come from conftest.py -- target ids are real
# eval_models row ids, so nothing here may reference a literal "t1".


def _capture(token=" a"):
    return CellCapture(
        prompt_mode="raw", k_requested=20, k_returned=2, content_offset=0,
        top_k=(TokenProb(token=token, logprob=-0.5, token_id=1),
               TokenProb(token=" the", logprob=-1.5, token_id=2)),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )


def test_schema_version_is_four():
    assert SCHEMA_VERSION == 4


def test_run_group_id_column_exists(db):
    cols = {r[1] for r in db.get_connection().execute("PRAGMA table_info(eval_runs)")}
    assert "run_group_id" in cols


def test_bench_round_trips_through_eval_tasks(db, config):
    task_id = save_bench(db, config)
    loaded = load_bench(db, task_id)
    assert loaded.name == "loaded-nouns v1"
    assert loaded.prompt_mode == "raw"
    assert loaded.top_k == 20
    assert loaded.probes == (" Sure", " I")
    assert len(loaded.target_ids) == 2


def test_bench_is_stored_as_a_logprob_task_with_a_bench_type_discriminator(db, config):
    """task_type's CHECK constraint permits only 4 values, so word bench
    rides on 'logprob' and is distinguished by config_data.bench_type."""
    task_id = save_bench(db, config)
    row = db.get_task(task_id)
    assert row["task_type"] == "logprob"
    assert row["config_data"]["bench_type"] == "word_bench"


def test_run_group_creates_one_run_per_target(db, config, targets, snippets):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    assert len(run_ids) == 2
    assert set(run_ids) == {t.id for t in targets}
    for run_id in run_ids.values():
        assert db.get_run(run_id)["run_group_id"] == group_id


def test_run_snapshot_carries_snippet_text_not_only_ids(db, config, targets, snippets):
    """A grid must still render after its dataset is edited or deleted."""
    task_id = save_bench(db, config)
    _, run_ids = create_run_group(db, task_id, config, targets, snippets)
    overrides = db.get_run(next(iter(run_ids.values())))["config_overrides"]
    snap_snippets = overrides["snapshot"]["snippets"]
    assert snap_snippets[0]["text"] == "The protestors were"
    assert snap_snippets[0]["text_hash"]


def test_run_snapshot_records_the_sampler_as_sent(db, config, targets, snippets):
    task_id = save_bench(db, config)
    _, run_ids = create_run_group(db, task_id, config, targets, snippets)
    overrides = db.get_run(next(iter(run_ids.values())))["config_overrides"]
    assert overrides["snapshot"]["sampler"]["temperature"] == 1.0


def test_grid_pivots_cells_by_snippet_and_target(db, config, targets, snippets):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    for target_id, run_id in run_ids.items():
        for snippet in snippets:
            save_cell(db, run_id, snippet, _capture())

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4
    cell = grid["cells"][("s1", targets[0].id)]
    assert isinstance(cell, CellCapture)
    assert cell.top_k[0].token == " a"


def test_failed_cells_are_stored_and_distinguishable_from_not_yet_run(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    first = targets[0].id
    save_cell(db, run_ids[first], snippets[0], CellError(reason="unreachable", detail="x"))

    grid = load_grid(db, group_id)
    assert isinstance(grid["cells"][("s1", first)], CellError)
    assert ("s2", first) not in grid["cells"], "absent means not yet run"


def test_grid_renders_from_the_snapshot_after_the_bench_is_edited(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    save_cell(db, run_ids[targets[0].id], snippets[0], _capture())

    edited = BenchConfig(
        name="loaded-nouns v2", prompt_mode="chat", top_k=5,
        dataset_id="d1", target_ids=(targets[0].id,), probes=(),
    )
    save_bench(db, edited, task_id=task_id)

    grid = load_grid(db, group_id)
    assert grid["snapshot"]["prompt_mode"] == "raw", "historical run keeps its own config"
    assert grid["snapshot"]["top_k"] == 20
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_storage.py -q
```

Expected: `test_schema_version_is_four` fails (`assert 3 == 4`), and the rest error on the missing `storage` module.

- [ ] **Step 3: Add the schema v4 migration**

In `tldw_chatbook/DB/Evals_DB.py`, change line 39:

```python
SCHEMA_VERSION = 4
```

In `_create_schema`, inside the `eval_runs` CREATE TABLE, add the column after `config_overrides TEXT`:

```sql
                run_group_id TEXT,
```

And after the existing `idx_eval_runs_model` index line, add:

```python
        conn.execute("CREATE INDEX idx_eval_runs_group ON eval_runs (run_group_id)")
```

In `_migrate_schema`, immediately before the closing `conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")`, add a branch matching the style of the v3 branch above it:

```python
        if current_version < 4 and SCHEMA_VERSION >= 4:
            logger.info("Migrating to version 4: Adding eval_runs.run_group_id")

            existing = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
            if "run_group_id" not in existing:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN run_group_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_runs_group "
                "ON eval_runs (run_group_id)"
            )
```

Existing rows keep `NULL`, which reads as a single-run group.

- [ ] **Step 4: Write the storage module**

Create `tldw_chatbook/Evals/word_bench/storage.py`:

```python
"""Map the word bench onto the existing Evals_DB tables.

The grid is a pivot of eval_results over a run group, not a new structure:

    bench      -> eval_tasks   (task_type='logprob', config_data.bench_type)
    run group  -> N eval_runs  sharing run_group_id, one per target
    cell       -> eval_results (run_id = target, sample_id = snippet)

Results render from a snapshot taken at launch, never from the live task:
eval_tasks is mutable, so editing a bench would otherwise silently
reinterpret every historical grid.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional, Sequence

from ...DB.Evals_DB import EvalsDB
from .capture_client import NEUTRAL_SAMPLER
from .models import (
    BenchConfig,
    CellCapture,
    CellError,
    Snippet,
    Target,
    TokenProb,
)

BENCH_TYPE = "word_bench"


def save_bench(db: EvalsDB, config: BenchConfig, task_id: Optional[str] = None) -> str:
    """Persist a bench as an eval_tasks row.

    task_type is 'logprob' because its CHECK constraint permits only four
    values; config_data.bench_type is the real discriminator.
    """
    config_data = {
        "bench_type": BENCH_TYPE,
        "prompt_mode": config.prompt_mode,
        "top_k": config.top_k,
        "probes": list(config.probes),
        "target_ids": list(config.target_ids),
        "concurrency": config.concurrency,
    }
    if task_id is not None:
        db.update_task(task_id, {"name": config.name, "config_data": config_data})
        return task_id
    return db.create_task(
        name=config.name,
        description=config.description,
        task_type="logprob",
        config_format="custom",
        config_data=config_data,
        dataset_id=config.dataset_id,
    )


def load_bench(db: EvalsDB, task_id: str) -> BenchConfig:
    row = db.get_task(task_id)
    data = row["config_data"]
    return BenchConfig(
        name=row["name"],
        description=row.get("description") or "",
        prompt_mode=data["prompt_mode"],
        top_k=int(data["top_k"]),
        dataset_id=row.get("dataset_id") or "",
        target_ids=tuple(data.get("target_ids", ())),
        probes=tuple(data.get("probes", ())),
        concurrency=int(data.get("concurrency", 1)),
    )


def _snapshot(
    config: BenchConfig, targets: Sequence[Target], snippets: Sequence[Snippet]
) -> dict[str, Any]:
    """The fully-resolved configuration a grid renders from.

    Snippet TEXT is stored, not only ids and hashes, so a grid still renders
    after its dataset is edited or deleted. The hash then serves its real
    purpose: flagging "this snippet was edited after the run".
    """
    return {
        "bench_name": config.name,
        "prompt_mode": config.prompt_mode,
        "top_k": config.top_k,
        "probes": list(config.probes),
        "sampler": dict(NEUTRAL_SAMPLER),
        "targets": [
            {
                "id": t.id, "name": t.name, "provider": t.provider,
                "model_id": t.model_id, "prefix": t.prefix,
                "system_prompt": t.system_prompt,
            }
            for t in targets
        ],
        "snippets": [
            {"id": s.id, "text": s.text, "text_hash": s.text_hash, "group": s.group}
            for s in snippets
        ],
    }


def create_run_group(
    db: EvalsDB,
    task_id: str,
    config: BenchConfig,
    targets: Sequence[Target],
    snippets: Sequence[Snippet],
) -> tuple[str, dict[str, str]]:
    """Create one eval_runs row per target, sharing a run_group_id."""
    group_id = uuid.uuid4().hex
    snapshot = _snapshot(config, targets, snippets)
    run_ids: dict[str, str] = {}

    for target in targets:
        run_id = db.create_run(
            name=f"{config.name} · {target.name}",
            task_id=task_id,
            model_id=target.id,
            config_overrides={"snapshot": snapshot, "target_id": target.id},
        )
        db.update_run(
            run_id, {"run_group_id": group_id, "total_samples": len(snippets)}
        )
        run_ids[target.id] = run_id

    return group_id, run_ids


def save_cell(
    db: EvalsDB, run_id: str, snippet: Snippet, result: CellCapture | CellError
) -> None:
    """Persist one cell. Failures are written as rows so that 'failed' and
    'not yet run' remain distinguishable in a partial grid."""
    if isinstance(result, CellError):
        payload = {"schema": "word_bench/1", "error": {
            "reason": result.reason, "detail": result.detail}}
    else:
        payload = {
            "schema": result.schema,
            "prompt_mode": result.prompt_mode,
            "k_requested": result.k_requested,
            "k_returned": result.k_returned,
            "content_offset": result.content_offset,
            "top_k": [
                {"id": t.token_id, "token": t.token,
                 "logprob": t.logprob, "bytes": list(t.bytes_)}
                for t in result.top_k
            ],
            "canary": result.canary,
            "captured_at": result.captured_at,
        }

    db.store_result(
        run_id=run_id,
        sample_id=snippet.id,
        input_data={"text": snippet.text, "group": snippet.group},
        actual_output=None,
        logprobs=payload,
        metrics={},
    )


def _cell_from_payload(payload: dict[str, Any]) -> CellCapture | CellError:
    if "error" in payload:
        return CellError(
            reason=payload["error"]["reason"], detail=payload["error"].get("detail", "")
        )
    return CellCapture(
        prompt_mode=payload["prompt_mode"],
        k_requested=payload["k_requested"],
        k_returned=payload["k_returned"],
        content_offset=payload.get("content_offset", 0),
        top_k=tuple(
            TokenProb(
                token=t["token"], logprob=t["logprob"],
                bytes_=tuple(t.get("bytes") or ()), token_id=t.get("id"),
            )
            for t in payload["top_k"]
        ),
        canary=payload.get("canary", "unchecked"),
        captured_at=payload.get("captured_at", ""),
    )


def load_grid(db: EvalsDB, run_group_id: str) -> dict[str, Any]:
    """Pivot a run group into a grid.

    Returns ``{"snapshot": …, "cells": {(snippet_id, target_id): cell}}``.
    The snapshot is the run's own, never the live task's.
    """
    runs = [
        run for run in db.list_runs(limit=10_000)
        if run.get("run_group_id") == run_group_id
    ]
    if not runs:
        raise ValueError(f"no runs found for run group {run_group_id!r}")

    overrides = runs[0].get("config_overrides") or {}
    snapshot = overrides.get("snapshot", {})

    cells: dict[tuple[str, str], CellCapture | CellError] = {}
    for run in runs:
        target_id = (run.get("config_overrides") or {}).get("target_id")
        for result in db.get_run_results(run["id"]):
            payload = result.get("logprobs")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not payload:
                continue
            cells[(result["sample_id"], target_id)] = _cell_from_payload(payload)

    return {"snapshot": snapshot, "cells": cells}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_storage.py -q
```

Expected: 11 passed. If `db.update_run` or `db.store_result` signatures differ from those used here, adapt the call sites to the real signatures in `Evals_DB.py` — do not change `Evals_DB`'s API.

- [ ] **Step 6: Verify the migration path from a v3 database**

The tests above create a fresh v4 database. The upgrade path needs its own check:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import sqlite3, tempfile, os
from pathlib import Path
tmp = Path(tempfile.mkdtemp()) / "old.db"

import tldw_chatbook.DB.Evals_DB as m
real = m.SCHEMA_VERSION
m.SCHEMA_VERSION = 3
db3 = m.EvalsDB(db_path=str(tmp), client_id="t"); db3.close()
print("created at user_version:", sqlite3.connect(tmp).execute("PRAGMA user_version").fetchone()[0])

m.SCHEMA_VERSION = real
db4 = m.EvalsDB(db_path=str(tmp), client_id="t")
cols = {r[1] for r in db4.get_connection().execute("PRAGMA table_info(eval_runs)")}
ver = db4.get_connection().execute("PRAGMA user_version").fetchone()[0]
print("after migrate: user_version =", ver, "| run_group_id present:", "run_group_id" in cols)
assert ver == 4 and "run_group_id" in cols
print("MIGRATION OK")
PY
```

Expected: `MIGRATION OK`.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/DB/Evals_DB.py tldw_chatbook/Evals/word_bench/storage.py Tests/Evals/word_bench/test_storage.py
git commit -m "feat(evals): word bench storage and schema v4

Adds eval_runs.run_group_id so a grid is N runs (one per target) sharing
a group. Everything else maps onto existing tables: bench -> eval_tasks
with a bench_type discriminator (task_type's CHECK permits only four
values), cell -> eval_results.logprobs.

Runs snapshot their fully-resolved config at launch, including snippet
TEXT rather than only ids, so a historical grid still renders after its
dataset is edited or deleted."
```

---

### Task 6: Grid runner

**Files:**
- Create: `tldw_chatbook/Evals/word_bench/runner.py`
- Test: `Tests/Evals/word_bench/test_runner.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces:
  - `class WordBenchRunner(db, client_factory)`
  - `async run(config, targets, snippets, task_id, progress=None, cancel_token=None) -> str` (run group id)
  - `class CancelToken` with `.cancel()` and `.is_cancelled`

- [ ] **Step 1: Write the failing test**

Create `Tests/Evals/word_bench/test_runner.py`:

```python
"""Grid execution: order, progress, cancel, and preflight propagation."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig, CellCapture, CellError, PreflightResult, Snippet, Target, TokenProb,
)
from tldw_chatbook.Evals.word_bench.runner import CancelToken, WordBenchRunner
from tldw_chatbook.Evals.word_bench.storage import load_grid, save_bench


# db, snippets, targets, config come from conftest.py. Target ids are real
# eval_models row ids, so tests key on target.name, never on a literal id.


def _cap(canary="pass"):
    return CellCapture(
        prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
        canary=canary, captured_at="2026-07-26T00:00:00Z",
    )


class FakeClient:
    """Records (snippet_text, target_name) so assertions never need an id."""

    def __init__(self, order, *, canary="pass", fail_target=None):
        self._order = order
        self._canary = canary
        self._fail_target = fail_target

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary=self._canary)

    async def capture(self, snippet, target, mode, top_k):
        self._order.append((snippet, target.name))
        if target.name == self._fail_target:
            return CellError(reason="unreachable", detail="x")
        return _cap(self._canary)


@pytest.mark.asyncio
async def test_runner_fills_the_grid_row_major(db, config, targets, snippets):
    """Complete comparable rows appear while the run is still going."""
    order = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient(order))
    await runner.run(config, targets, snippets, task_id)

    assert order == [
        ("The protestors were", "base"), ("The protestors were", "steered"),
        ("The rioters were", "base"), ("The rioters were", "steered"),
    ]


@pytest.mark.asyncio
async def test_every_cell_is_persisted(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4


@pytest.mark.asyncio
async def test_failed_cells_are_persisted_too(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], fail_target="steered"))
    group_id = await runner.run(config, targets, snippets, task_id)

    base, steered = targets[0].id, targets[1].id
    grid = load_grid(db, group_id)
    assert isinstance(grid["cells"][("s1", steered)], CellError)
    assert isinstance(grid["cells"][("s1", base)], CellCapture)


@pytest.mark.asyncio
async def test_progress_reports_group_level_totals(db, config, targets, snippets):
    seen = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    await runner.run(config, targets, snippets, task_id,
                     progress=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (4, 4), "progress is over the whole grid, not per run"


@pytest.mark.asyncio
async def test_cancel_stops_the_run_and_keeps_completed_cells(db, config, targets, snippets):
    token = CancelToken()
    order = []

    class CancellingClient(FakeClient):
        async def capture(self, snippet, target, mode, top_k):
            result = await super().capture(snippet, target, mode, top_k)
            if len(order) == 2:
                token.cancel()
            return result

    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: CancellingClient(order))
    group_id = await runner.run(config, targets, snippets, task_id, cancel_token=token)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 2, "a cancelled run is a real, partial measurement"


@pytest.mark.asyncio
async def test_degenerate_canary_propagates_onto_every_cell(db, config, targets, snippets):
    """The preflight warning must not be lost between preflight and grid."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="degenerate"))
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert all(c.canary == "degenerate" for c in grid["cells"].values())


@pytest.mark.asyncio
async def test_targets_invalid_for_the_mode_are_rejected_before_any_call(
    db, config, snippets
):
    bad = Target(id="unused", name="c", provider="p", model_id="m", system_prompt="x")
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))

    with pytest.raises(ValueError, match="raw"):
        await runner.run(config, [bad], snippets, task_id)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_runner.py -q
```

Expected: collection error — no module named `runner`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Evals/word_bench/runner.py`:

```python
"""Grid execution.

Row-major: complete, comparable rows appear while the run is still going,
which is the point of the grid doubling as the progress view. Fail-fast on a
dead target is preflight's job, not the fill order's.

Sequential within and across targets by default -- local servers are
frequently single-slot, and concurrent requests either queue or 503.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from loguru import logger

from ...DB.Evals_DB import EvalsDB
from .models import (
    BenchConfig, CellCapture, CellError, PreflightResult, PromptMode, Snippet, Target,
)
from .storage import create_run_group, save_cell

ProgressFn = Callable[[int, int], None]


class CaptureClientLike(Protocol):
    async def preflight(
        self, target: Target, mode: PromptMode, top_k: int
    ) -> PreflightResult: ...

    async def capture(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> CellCapture | CellError: ...


class CancelToken:
    """Cancels a whole run group, not a single run."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class WordBenchRunner:
    """Executes a bench over a snippet set and a target set."""

    def __init__(
        self, db: EvalsDB, client_factory: Callable[[Target], CaptureClientLike]
    ) -> None:
        self._db = db
        self._client_factory = client_factory

    async def run(
        self,
        config: BenchConfig,
        targets: Sequence[Target],
        snippets: Sequence[Snippet],
        task_id: str,
        progress: Optional[ProgressFn] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> str:
        """Execute the grid and return its run group id."""
        for target in targets:
            if not target.is_valid_for_mode(config.prompt_mode):
                raise ValueError(
                    f"Target {target.name!r} is not valid for {config.prompt_mode!r} mode: "
                    "raw mode takes a prefix, chat mode takes a system_prompt."
                )

        clients = {t.id: self._client_factory(t) for t in targets}

        # Preflight before any measurement, so a dead or degenerate target is
        # known up front rather than discovered N cells in.
        canaries: dict[str, str] = {}
        for target in targets:
            result = await clients[target.id].preflight(
                target, config.prompt_mode, config.top_k
            )
            canaries[target.id] = result.canary
            if result.is_warned:
                logger.warning(
                    "Word bench target %s preflighted degenerate; its column "
                    "carries a warning.", target.name,
                )

        group_id, run_ids = create_run_group(
            self._db, task_id, config, targets, snippets
        )

        total = len(snippets) * len(targets)
        done = 0

        for snippet in snippets:  # row-major
            for target in targets:
                if cancel_token is not None and cancel_token.is_cancelled:
                    logger.info(
                        "Word bench run group %s cancelled after %d/%d cells",
                        group_id, done, total,
                    )
                    return group_id

                result = await clients[target.id].capture(
                    snippet.text, target, config.prompt_mode, config.top_k
                )
                result = self._stamp_canary(result, canaries[target.id])
                save_cell(self._db, run_ids[target.id], snippet, result)

                done += 1
                if progress is not None:
                    progress(done, total)

        return group_id

    @staticmethod
    def _stamp_canary(
        result: CellCapture | CellError, canary: str
    ) -> CellCapture | CellError:
        """Carry the target's preflight verdict onto the cell.

        Without this the warning is lost between preflight and the grid, and a
        divergence produced by out-of-distribution behaviour reads as a
        finding about the model's content.
        """
        if isinstance(result, CellError):
            return result
        return CellCapture(
            prompt_mode=result.prompt_mode,
            k_requested=result.k_requested,
            k_returned=result.k_returned,
            content_offset=result.content_offset,
            top_k=result.top_k,
            canary=canary,
            captured_at=result.captured_at,
            schema=result.schema,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_runner.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Run the whole engine suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals -q
python -c "import tldw_chatbook.app; print('app imports OK')"
```

Expected: all pass, no collection errors, output pristine.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/runner.py Tests/Evals/word_bench/test_runner.py
git commit -m "feat(evals): word bench grid runner

Row-major fill so complete comparable rows appear during the run;
fail-fast on a dead target is preflight's job. Sequential by default --
local servers are frequently single-slot.

Cancel operates on the run group and keeps completed cells: a cancelled
run is a real, if partial, measurement. The preflight canary verdict is
stamped onto every cell so a degenerate target's warning cannot be lost
before the grid renders it."
```

---

### Task 7: End-to-end engine check

Proves the modules compose. No new production code.

**Files:**
- Test: `Tests/Evals/word_bench/test_engine_end_to_end.py`

**Interfaces:**
- Consumes: everything.
- Produces: nothing.

- [ ] **Step 1: Write the test**

```python
"""The engine's deliverable: a correct grid, end to end, with no UI."""

from __future__ import annotations

import math

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.analysis import divergence, group_means, spread
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig, CellCapture, PreflightResult, Snippet, Target, TokenProb,
)
from tldw_chatbook.Evals.word_bench.runner import WordBenchRunner
from tldw_chatbook.Evals.word_bench.storage import load_grid, save_bench


# db, snippets, targets, config come from conftest.py.

#: "steered" diverges from "base" only on the loaded snippet -- the shape a
#: real finding has. Keyed on target NAME: ids are database-assigned.
SCRIPT = {
    ("The protestors were", "base"):    [(" a", 0.7), (" the", 0.3)],
    ("The protestors were", "steered"): [(" a", 0.7), (" the", 0.3)],
    ("The rioters were", "base"):       [(" a", 0.7), (" the", 0.3)],
    ("The rioters were", "steered"):    [(" not", 0.8), (" a", 0.2)],
}


class ScriptedClient:
    def __init__(self, target):
        self._target = target

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=top_k, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        pairs = SCRIPT[(snippet, target.name)]
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=len(pairs),
            content_offset=0,
            top_k=tuple(
                TokenProb(token=t, logprob=math.log(p), token_id=i)
                for i, (t, p) in enumerate(pairs)
            ),
            canary="pass", captured_at="2026-07-26T00:00:00Z",
        )


@pytest.mark.asyncio
async def test_engine_produces_a_grid_whose_divergence_finds_the_steered_cell(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4

    base, steered = targets[0].id, targets[1].id
    # Column baseline: base. Only the loaded snippet moved.
    neutral, _ = divergence(grid["cells"][("s1", base)], grid["cells"][("s1", steered)])
    loaded, _ = divergence(grid["cells"][("s2", base)], grid["cells"][("s2", steered)])

    assert neutral == pytest.approx(0.0, abs=1e-9)
    assert loaded > 0.3
    assert loaded > neutral

    # Group means are the headline number for a control/treatment set.
    by_group = {s.id: s.group for s in snippets}
    means = group_means([
        (by_group["s1"], neutral),
        (by_group["s2"], loaded),
    ])
    assert means["loaded"] > means["neutral"]


@pytest.mark.asyncio
async def test_spread_identifies_the_row_where_targets_disagree(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)
    grid = load_grid(db, group_id)

    s1 = spread([grid["cells"][("s1", t.id)] for t in targets])
    s2 = spread([grid["cells"][("s2", t.id)] for t in targets])
    assert s2 > s1


@pytest.mark.asyncio
async def test_grid_survives_the_bench_being_edited_afterwards(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)

    save_bench(db, BenchConfig(name="renamed", prompt_mode="chat", top_k=99,
                               dataset_id="d", target_ids=(targets[0].id,)),
               task_id=task_id)

    grid = load_grid(db, group_id)
    assert grid["snapshot"]["prompt_mode"] == "raw"
    assert grid["snapshot"]["top_k"] == 20
    assert len(grid["cells"]) == 4
```

- [ ] **Step 2: Run it**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench/test_engine_end_to_end.py -q
```

Expected: 3 passed. If any fail, the defect is in composition, not in a single module — fix it before proceeding.

- [ ] **Step 3: Final verification**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_evals_deletion_guard.py -q
python -c "import tldw_chatbook.app; print('app imports OK')"
```

All must pass. The deletion guard is included because this PR adds files under `Evals/`, and the guard asserts nothing retired in PR 1 has returned.

- [ ] **Step 4: Commit**

```bash
git add Tests/Evals/word_bench/test_engine_end_to_end.py
git commit -m "test(evals): word bench engine end-to-end

Proves the modules compose: a scripted two-by-two grid where only the
steered target moves on the loaded snippet, and divergence, spread, and
group means each locate it. Also pins that a grid still renders from its
snapshot after the bench is edited."
```

---

## Notes for the reviewer

- **The normalizer's fixtures are the spec's own correction.** The spec predicted two response shapes and was wrong; both llama.cpp endpoints return the same modern form. Any change to `normalizer.py` must keep the fixture tests passing, and a new provider needs a captured fixture before it is claimed as supported.
- **`analysis.py` carries the methodology** and therefore most of the coverage. The three properties to guard are: the `other` bucket makes the support sum to 1, mixed-K cells are cut to `min(K)`, and probes have three states rather than two.
- **`temperature` is 1.0 and must stay 1.0.** Zero collapses the distribution being measured. This is the single easiest thing to "fix" wrongly.
- **No UI in this PR.** Anything under `tldw_chatbook/UI/` is out of scope.
- Schema goes 3 → 4. **Re-verify at merge** that no concurrent branch also took 4 — this repo has hit migration-number collisions repeatedly.
