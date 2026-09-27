# Personal Context exact text-span digests Implementation Plan

> **For agentic workers:** Use executing-plans for native inline execution in the existing isolated worktree. The user's native execution preference persists. Steps use checkboxes for tracking. The user endorsed this slice subject to review on 2026-09-26; native execution follows the reviewed corrections below.

**Goal:** Provide one shared calculation of exact representation and span digests for the accepted evidence convention.

**Architecture:** A pure function in shared core receives already-owned text and codepoint offsets and returns two named immutable digests. It uses strict UTF-8 and SHA-256 without text normalization, source lookup, persistence or admission decisions. The function is a primitive for the accepted future contract; no existing Chatbook consumer switches to V2.

**Tech Stack:** Existing Python >=3.11 shared package; native Python 3.12 interpreter; stdlib dataclasses/hashlib; existing pytest and Ruff. No dependency or package-version change.

**Spec:** [Accepted evidence design](../specs/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-design.md), specifically its Evidence bindings and Exact Unicode span vector sections.

**Task:** [TASK-25907.12](../../../backlog/tasks/task-25907.12%20-%20Add-exact-text-span-digests-to-shared-profile-core.md).

**ADR required:** no new ADR; implement the exact text/span convention already accepted in ADR-185.

**ADR path:** [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md), retaining [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md) ownership.

**Reason:** This slice introduces a data-only calculation in the accepted shared-core boundary; it makes no storage, schema, resolver, grant, retirement or temporal policy decision.

## Global Constraints

- The source convention is “zero-based Unicode codepoint offsets `[start, end)` and SHA-256 of the exact UTF-8 span.” No Unicode normalization is allowed.
- A matching digest proves byte identity, not semantic support, approval, truth or source authority.
- V1 schemas, canonical bytes, fixtures, exports and SERIALIZED_SCHEMA_VERSION remain unchanged. No V2 model, vocabulary or capability is advertised.
- Shared core owns data calculations; native/server owners retain source access, encryption, policy, current-version verification and transactions.
- Do not reuse the helper to upgrade V1 source_references/source_hashes or change the existing profile-tool substring checks.
- A returned pair is not a complete EvidenceBinding: source authority/scope/version, roles, binding/claim canonical digests and current authorization remain separate future work.
- Native execution stays in `/Users/macbook-dev/.codex/worktrees/personal-context-memory-baseline/tldw_chatbook`; import shared core from this worktree, not a stale editable installation.
- Synthetic targeted checks only. No full-suite run, real profile, app, provider, network, background job, push, PR or merge.
- Preserve independent TASK-25907.10 and the roadmap section starting `## Follow-up: generated-answer effectiveness` byte-for-byte. Stage only this task's explicit paths and the owned roadmap prefix.

## File map and contract

Create at execution:

- `packages/tldw_profile_core/src/tldw_profile_core/evidence.py`: the pure calculation and frozen result, with public Args/Returns/Raises documentation.
- `packages/tldw_profile_core/tests/test_evidence.py`: fixed exact-text vectors and rejected input cases through that API.
- `backlog/docs/personal-context-exact-span-digests.md`: explicit import, convention, result/error semantics, and remaining owner-level gates.
- `Docs/superpowers/reviews/2026-09-26-personal-context-exact-span-digests-review.md`: actual checks and bounded review after implementation.

Update this plan, TASK-25907.12 and the owned roadmap prefix at execution milestones. Leave `__init__.py`, `pyproject.toml`, canonical.py, schema exports, all V1 fixtures and native consumers untouched.

**Produces:** `ExactTextSpanDigests(representation_sha256: str, span_sha256: str)` and `exact_text_span_digests(representation: str, start: int, end: int) -> ExactTextSpanDigests`, explicitly imported from `tldw_profile_core.evidence`.

**Consumes:** a built-in Python str selected by its caller and built-in int offsets. Subclasses, booleans and every float, including integral floats, are rejected before encoding or range checks. This native function's scalar checks do not change the existing canonical JSON integer semantics. Valid bounds satisfy `0 <= start <= end <= len(representation)`. Empty spans are valid calculations, with the standard empty UTF-8 hash; this does not make them supporting evidence. No arbitrary source-size policy is invented: callers must enforce their own source access and resource limits before supplying text. Invalid bounds raise ValueError; wrong scalar types raise TypeError; an unencodable representation raises UnicodeEncodeError, a ValueError subclass, even if the invalid codepoint lies outside the selected span.

## Unit 1: Fixed conformance and rejected input controls

- [x] **Step 1: Create the new test module with these actual API controls.** Keep expected digests literal; do not calculate the expected values using the production helper or a duplicate algorithm.

```python
from dataclasses import FrozenInstanceError

import pytest
from tldw_profile_core.evidence import exact_text_span_digests


class CustomText(str):
    def encode(self, *_args, **_kwargs):
        raise AssertionError("custom encoding must not be called")


class CustomOffset(int):
    pass


SOURCE = "Hi 👋 — café"
SOURCE_SHA = "bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef"
SPAN_SHA = "850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_accepted_unicode_vector_and_immutable_result():
    result = exact_text_span_digests(SOURCE, 7, 11)
    assert result.representation_sha256 == SOURCE_SHA
    assert result.span_sha256 == SPAN_SHA
    assert exact_text_span_digests(SOURCE, 7, 11) == result
    with pytest.raises(FrozenInstanceError):
        result.span_sha256 = EMPTY_SHA


@pytest.mark.parametrize(
    "text,start,end,representation_sha,span_sha",
    [
        (
            "Hi 👋 — cafe\u0301",
            7,
            12,
            "b433fe833253863d292ee2dbf15f9ea928a33bc28eb61bca9aeaba23bb2d030c",
            "81ef060bcd98adc7824eb5c1ada83c32491b16018e11e79f00ab9d09e04b015a",
        ),
        (
            "A\r\nB",
            1,
            3,
            "255e24970eef1cf6a0503f246be4b2ecd25d69bbbeb4073f4abb26d62886f64b",
            "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
        ),
        (
            "A\nB",
            1,
            2,
            "23519a43c66b4c342f25b32e09797ec5f3fc0be388cd8243fb3449afbdce4013",
            "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        ),
        (
            "👋",
            0,
            1,
            "1d0452e3d194cc7950909b578c611d5ad4cd15105c6aeefc38ce213240ffc457",
            "1d0452e3d194cc7950909b578c611d5ad4cd15105c6aeefc38ce213240ffc457",
        ),
    ],
)
def test_exact_text_vectors(text, start, end, representation_sha, span_sha):
    result = exact_text_span_digests(text, start, end)
    assert result.representation_sha256 == representation_sha
    assert result.span_sha256 == span_sha


def test_outside_span_edit_changes_source_identity_only():
    original = exact_text_span_digests(SOURCE, 7, 11)
    edited = exact_text_span_digests(SOURCE + "!", 7, 11)
    assert edited.span_sha256 == original.span_sha256 == SPAN_SHA
    assert edited.representation_sha256 == (
        "acb2a8f3ad9c66ce276c5db2c069c559e25c3d55c779f3607f0c92b0af7905c9"
    )
    assert edited.representation_sha256 != original.representation_sha256


def test_valid_but_shifted_offset_does_not_match_accepted_span():
    shifted = exact_text_span_digests(SOURCE, 8, 11)
    assert shifted.representation_sha256 == SOURCE_SHA
    assert shifted.span_sha256 == (
        "6a43309ff7bbc6b3fd79d434521fcc2d1a77e71cfbdc894a11de20ede7ee929b"
    )
    assert shifted.span_sha256 != SPAN_SHA


@pytest.mark.parametrize("text,index", [("", 0), (SOURCE, 0), (SOURCE, 11)])
def test_empty_span_at_valid_boundary(text, index):
    result = exact_text_span_digests(text, index, index)
    assert result.span_sha256 == EMPTY_SHA
    assert result.representation_sha256 == (EMPTY_SHA if not text else SOURCE_SHA)


@pytest.mark.parametrize(
    "text,start,end",
    [
        (None, 0, 0),
        (b"abc", 0, 1),
        (42, 0, 0),
        ("abc", True, 2),
        ("abc", 0, False),
        ("abc", 1.0, 2),
        ("abc", 0, 2.0),
        ("abc", 0, 1.5),
        ("abc", "0", 2),
        ("abc", 0, None),
        pytest.param(CustomText("abc"), 0, 1, id="string-subclass"),
        pytest.param("abc", CustomOffset(0), 1, id="start-subclass"),
        pytest.param("abc", 0, CustomOffset(1), id="end-subclass"),
    ],
)
def test_wrong_scalar_types_are_not_coerced(text, start, end):
    with pytest.raises(TypeError):
        exact_text_span_digests(text, start, end)


@pytest.mark.parametrize("start,end", [(-1, 2), (0, -1), (2, 1), (0, 4), (4, 4)])
def test_invalid_bounds_are_not_silently_sliced(start, end):
    with pytest.raises(ValueError):
        exact_text_span_digests("abc", start, end)


@pytest.mark.parametrize("text,start,end", [("\ud800", 0, 1), ("a\udfff", 0, 1)])
def test_unencodable_whole_representation_is_rejected(text, start, end):
    with pytest.raises(UnicodeEncodeError):
        exact_text_span_digests(text, start, end)
```

A shifted valid span returns its own identity; this utility has no expected binding to compare and cannot detect a caller's intent. Offsets past the representation must reject instead of Python slicing's silent clipping. The fixed astral vector catches accidental UTF-16 indexing or grapheme indexing. Testing a surrogate outside the span establishes whole-representation validation.

- [x] **Step 2: Run the new test module RED from the worktree.**

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py -q
```

Start Step 1 with only `test_accepted_unicode_vector_and_immutable_result` and its constants; put the explicit function import inside that test body. The first run must report one failed test for the missing module, rather than a collection error. Confirm the import targets this worktree and record the failure. Create the minimum named result and exact UTF-8 hash calculation from Unit 2 before adding its validation, then add the remaining full test module above and run it again. Record actual invalid-input failures before adding strict type/bounds validation. This two-stage RED process distinguishes missing implementation from controls that reject an overly permissive hash calculation; neither failure qualifies a production source resolver.

## Unit 2: Minimal pure implementation and affected core checks

- [x] **Step 1: Create evidence.py with the following final implementation.** For Unit 1's initial canonical test, begin with only the frozen result, strict UTF-8 encoding and two hashes; add the shown type/range checks only after the full negative controls fail. Document the conventions and public arguments/errors from the contract above in the module/function docstrings.

```python
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class ExactTextSpanDigests:
    """Exact UTF-8 identities; neither field establishes evidence support."""

    representation_sha256: str
    span_sha256: str


def exact_text_span_digests(
    representation: str, start: int, end: int
) -> ExactTextSpanDigests:
    """Hash an exact representation and its half-open codepoint span.

    Args:
        representation: Already-owned built-in string, without normalization.
        start: Inclusive codepoint offset as a built-in integer.
        end: Exclusive codepoint offset as a built-in integer.

    Returns:
        Named lowercase SHA-256 hex digests, without source text.

    Raises:
        TypeError: Text or offsets have unsupported scalar types.
        ValueError: Bounds do not satisfy the half-open range convention.
        UnicodeEncodeError: Any source codepoint cannot be encoded as UTF-8.
    """
    if type(representation) is not str:
        raise TypeError("representation must be a built-in string")
    if type(start) is not int or type(end) is not int:
        raise TypeError("span offsets must be built-in integers")
    if not 0 <= start <= end <= len(representation):
        raise ValueError("span must satisfy 0 <= start <= end <= text length")
    representation_utf8 = representation.encode("utf-8", errors="strict")
    span_utf8 = representation[start:end].encode("utf-8", errors="strict")
    return ExactTextSpanDigests(
        representation_sha256=sha256(representation_utf8).hexdigest(),
        span_sha256=sha256(span_utf8).hexdigest(),
    )
```

Use the exact string UTF-8 bytes, not canonical JSON bytes, quoted JSON, NFC/NFD normalization, normalized line endings, joined fields, UTF-16 code units or encoded byte offsets. No verifier, resolver registry, cache or framework is needed in this slice.

- [x] **Step 2: Run the new module and the four existing affected shared-core modules GREEN.**

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q
```

This is a targeted shared-core run, not the application suite. Keep the existing V1 canonical/HMAC and source/export schema parity checks unchanged. Do not add broad runtime tests for native routes that remain untouched, or claim server conformance from Python-only checks.

- [x] **Step 3: Add the API documentation.** Show `from tldw_profile_core.evidence import exact_text_span_digests` with the accepted source and both fixed hashes. State the half-open codepoint convention, all error/empty-span semantics, no normalization, caller-owned input resource bounds and no retained source in the returned result. State explicitly that V2 bindings, owner authorization/current-source verification, semantic assessment, migration, negotiation, forgetting, disclosure and server qualification are unimplemented by this helper. Distinguish these exact-text SHA values from future RFC 8785 binding/claim digests and existing keyed V1 integrity tags. Link ADR-185, this plan and the task; create no V2 wire fixtures or capability declarations.

## Unit 3: Review and verified closeout

- [x] **Step 1: Run whole-file lint/format and whitespace checks for the two new modules.**

```bash
.venv/bin/python -m ruff check --no-cache packages/tldw_profile_core/src/tldw_profile_core/evidence.py packages/tldw_profile_core/tests/test_evidence.py
.venv/bin/python -m ruff format --no-cache --check packages/tldw_profile_core/src/tldw_profile_core/evidence.py packages/tldw_profile_core/tests/test_evidence.py
git diff --check
```

Format the new files if needed, then rerun the affected checks only if changes can alter behavior. Confirm existing V1 package files and native consumers have no diff relative to `260e58dab8`.

- [x] **Step 2: Self-review, then use one bounded read-only review under requesting-code-review.** Check the codepoint/strict UTF-8 contract, invalid-input controls, literal vectors, immutable result, unchanged V1 surface, and absence of source I/O or runtime enablement. Resolve verified blocking findings; do not dispatch implementation agents or repeat unchanged audits. The already-selected native execution mode persists.

- [x] **Step 3: Record actual evidence in the execution review.** Include starting revision, commands/exit codes and actual distinct targeted test count; distinguish the expected RED import failure, passing API checks, whole-file static analysis and untouched V1 files. List unresolved owner-level controls plainly. No claim of citation resolution, model privacy, semantic support or V2 rollout follows from these checks.

- [x] **Step 4: Complete task/tracker hygiene only after all six criteria pass.** Use Backlog CLI to check the criteria, add concise Implementation Notes/ADR links and mark Done. Update this plan and roadmap prefix. Verify original 59 criteria and .11's six criteria, local links, unique IDs/backward dependencies, independent .10/suffix bytes, original fixtures/reports and the exact owned-file set. Stage only explicit owned paths and the roadmap prefix, commit locally and preserve the existing branch/worktree.

## Plan self-review and planning checkpoint

This directly scopes the exact text/span calculation in the accepted architectural design. Complete V2 record/proposal/manifest models and canonical binding/claim projections, source capture/resolution, temporal admission, retirement, disclosure and client/server qualification remain separate future releases; they are not promised by this utility.

The short pure function plus fixed conformance tests is the selected approach. Wiring it into V1 metadata would imply guarantees V1 cannot express; a resolver or complete V2 rollout would exceed this independently testable primitive. Empty-span and error rules are native API semantics, not a new canonical schema policy. Named immutable fields avoid swapping the full-source and span hashes.

The examples parse and the fixed digest values are checked independently with stdlib on synthetic text at planning time. This is arithmetic/syntax inspection, not execution of the proposed API or passing runtime tests. Planning changes task/plan/roadmap documentation only. Implementation and an execution review remain unchecked deliverables.

## Native planning validation

The planning guard passed for five scoped documents: 62 local Markdown links,
66 task documentation/reference paths, 13 unique family IDs, all 59 unchanged
original criteria and .11's six unchanged completed criteria. TASK-25907.12
has six unchecked criteria and nine unchecked execution steps. Both Python
example blocks parse, and the synthetic source/span values match independent
stdlib calculations. These checks do not execute or qualify the proposed API.

The repository Backlog guard passed against the exact 13-file task family,
checking filenames/frontmatter IDs and Windows-compatible paths. Allocation
was checked across 566 available refs and 80 worktrees, including completed
and archived task buckets; no fetch or open-PR search was performed. Recheck
allocation at integration. Unrelated malformed tasks reported during CLI
hydration were left untouched; they did not prevent creating this task.

Independent TASK-25907.10 and its roadmap suffix retain their original hashes.
The exact change set is this plan, task .12 and the owned roadmap prefix.
Production, test, shared-core, V1 schema/fixture and historical report files
remain unchanged. Whitespace passes. No runtime tests were run or claimed.

## Follow-up plan review

The user requested review before continuing on 2026-09-26. Inline review found
that isinstance accepts str/int subclasses whose Python methods can override
encoding or range checks. The native API now requires exact built-in types;
three subclass controls cover that contract, including a custom encoding method
that must never run. This does not change canonical JSON integer semantics.

The RED sequence was tightened to start with a single test-local import and
then exercise the remaining negative controls against the minimal calculation
before validation. A missing-module collection error alone would not show that
bool, negative-index and oversize-span rejection tests distinguish correct
behavior. No new architectural policy or task criterion is introduced.

## Execution closeout

The user endorsed this slice subject to review on 2026-09-26. Inline and
independent review resolved the strict-type and meaningful-RED improvements;
no further implementation finding remains. All nine steps are complete.
The meaningful input-validation RED produced 14 failures/16 passes against
the minimal calculation before validation. The final five-module run passed
181 distinct cases (30 new, 151 existing); full-file Ruff/format, native import
ownership and the executed documentation example passed.

The custom string fixture needed explicit parameter IDs because pytest called
its encoding method during automatic naming. This collection failure was
resolved separately and was not called runtime RED evidence. Import-order
lint was fixed in the new test only. The
[execution review](../reviews/2026-09-26-personal-context-exact-span-digests-review.md)
records exact commands, review attribution and remaining owner-level controls.
The API is documented in
[exact-span digests](../../../backlog/docs/personal-context-exact-span-digests.md).
Native consumers, V1 canonical files, schema version, package exports,
fixtures, permissions and historical evidence remain unchanged; the broader
V2 contract and known device-only disclosure gap remain unimplemented.
