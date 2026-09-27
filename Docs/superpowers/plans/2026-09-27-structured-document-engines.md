# Local Document Engines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce qualified local check/format engines with bounded, terminable execution and common conformance evidence.
**Architecture:** Pure per-language adapters behind the programme's shared request/result model, with separate process/worker owners. No engine imports a Notes store or editor widget.
**Tech Stack:** Python standard library plus a qualified YAML candidate; TypeScript source-aware JSON/YAML candidates; multiprocessing/browser workers, pytest/Vitest.
**Spec:** [Approved design](../specs/2026-09-27-structured-document-editing-design.md)

## Global Constraints

- All [programme interfaces and constraints](2026-09-27-structured-document-editing.md) apply.
- "Initial common limits are 1 MiB of UTF-8 source, nesting depth 100, 100 displayed diagnostics, and a 2-second processing deadline per request."
- "Do not implement formatting as a generic object parse followed by serialization."
- "Formatting is deterministic and idempotent within each engine; identical whitespace between the two engines is not required."
- ADR required: yes. ADR path: `C/backlog/decisions/194-structured-note-language-and-local-editor-validation.md`. Dependency adoption must include its qualification evidence and repository-local ADR assessment before integration.
- Status: Not Started. Qualification may proceed before foundations land; it does not authorize production UI wiring.

## E1: Shared fixtures and pure contracts

**Files C:** create `Tests/fixtures/document_editing/v1/cases.json`, `Tests/Document_Editing/conftest.py`, `test_contract.py`, and `tldw_chatbook/Document_Editing/__init__.py`, `models.py`.
**Files U:** create `src/services/document-editing/types.ts`, `__tests__/fixtures/v1/cases.json`, `__tests__/contract.test.ts`.
**Interfaces:** the programme's `DocumentRequest`, `CheckResult`, `FormatResult`, `RequestKey`, and `Diagnostic` definitions. Fixture fields are `id`, `language`, `text`, `valid`, `codes`, `format`, `must_keep`; `format` is `allowed` or `blocked`.

- [ ] Add the first shared cases (JSON escaping in the fixture file must preserve the intended source):

```json
[
  {"id":"json-large-int","language":"json","text":"{\"n\":9007199254740993}","valid":true,"codes":[],"format":"allowed","must_keep":["9007199254740993"]},
  {"id":"json-nan","language":"json","text":"{\"n\":NaN}","valid":false,"codes":["syntax"],"format":"blocked","must_keep":[]},
  {"id":"json-duplicate","language":"json","text":"{\"a\":1,\"a\":2}","valid":true,"codes":["duplicate_key"],"format":"blocked","must_keep":[]},
  {"id":"jsonl-blank","language":"jsonl","text":"{}\n\n{}\n","valid":false,"codes":["blank_record"],"format":"blocked","must_keep":[]},
  {"id":"yaml-stream","language":"yaml","text":"---\na: 1\n...\n---\nb: 2\n","valid":true,"codes":[],"format":"allowed","must_keep":["...","b:"]}
]
```

- [ ] Expand the same fixture set with escaped duplicate names, sibling-object keys, comments/trailing commas, top-level scalars, zero-record JSONL, final newline, `1e400`, `-0`, decimal precision, escaped strings, YAML comments/anchors/aliases/merges, multi-document streams, 1.1/1.2 directives, complex keys, tags, and literal/folded/chomping scalars. Separate deliberately unsupported YAML constructs from malformed syntax.
- [ ] Add contract tests rejecting negative/out-of-order ranges, replacement on blocked formats, `format_allowed` for incomplete checks, and success for excerpts. Define a pytest `request_for` factory and equivalent TS `requestFor` test factory using the programme key with authority `test`, document `n1`, generations/revisions `1`, and a fresh request ID; expose optional field overrides. Export the TS factory from `__tests__/request-fixture.ts`. Python fixture implementation:

```python
import itertools
import pytest
from tldw_chatbook.Document_Editing.models import DocumentRequest, RequestKey

@pytest.fixture
def request_for():
    ids = itertools.count(1)
    def make(text, language="json", **overrides):
        key = dict(authority_id="test", document_id="n1", session_generation=1,
                   draft_revision=1, language_revision=1, request_id=next(ids))
        for field in tuple(key):
            if field in overrides:
                key[field] = overrides.pop(field)
        return DocumentRequest(key=RequestKey(**key), text=text,
                               language=language, **overrides)
    return make
```
- [ ] Implement only the models and fixture validation; run `python -m pytest Tests/Document_Editing/test_contract.py -q` from C and `bun run test src/services/document-editing/__tests__/contract.test.ts` from U.
- [ ] Keep the copies byte-identical with SHA-256 recorded in each test report. Updating fixture contract v1 requires both consumers' evidence; no new shared runtime package or network fixture fetch. Commit each repository's model/fixture files.

## E2: Python JSON/JSONL and YAML qualification

**Dependencies:** E1.
**Files C:** create `tldw_chatbook/Document_Editing/engine.py`, `json_document.py`, `yaml_document.py`, `source_positions.py`; create `Tests/Document_Editing/test_json_document.py`, `test_yaml_document.py`, `test_source_positions.py`; qualification report `Docs/superpowers/reviews/2026-09-27-document-engine-python.md`. Modify `pyproject.toml` and `tldw_chatbook/Utils/optional_deps.py` only if a YAML dependency is accepted.
**Interfaces:** programme `check_document`/`format_document`. `source_positions.py` converts parser positions into zero-based Unicode scalar offsets; JSON adapter preserves original token slices. YAML adapter emits unsupported results for unqualified constructs.

- [ ] Add failing real-engine tests:

```python
def test_format_keeps_large_number_and_is_idempotent(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document
    first = format_document(request_for('{"n":9007199254740993,"x":-0}'))
    assert first.state == "formatted"
    assert "9007199254740993" in first.replacement
    assert '"x": -0' in first.replacement
    second = format_document(request_for(first.replacement))
    assert second.state == "unchanged"
    assert second.replacement is None
```

- [ ] Run the JSON file red. Use strict stdlib `json` validation with a rejecting `parse_constant`, token-preserving numeric callbacks, and duplicate-aware pairs. A bounded lexical pass records original JSON strings, numbers, punctuation, and property locations; it guards nesting before object construction and changes whitespace only after grammar validation. Treat numerical implementation limits as resource-limited, not invented syntax errors.
- [ ] Track decoded object names per object scope, so `"a"` and `"\u0061"` collide while nested/sibling keys do not. Preserve token order; compare the full nontrivia token stream before and after formatting. JSONL applies this per physical record and adjusts offsets into the original document, preserving final-newline presence.
- [ ] Qualify `ruamel.yaml` source/round-trip facilities against E1 before adopting it. Parse events/tokens without application constructors or alias expansion, enforce depth during the event walk, and retain comments/anchors/directives/scalar tokens. Candidate output must preserve the spec's numeric/string tokens and semantic graph; plain object equality is not sufficient. Mixed YAML-version streams and unknown tags are explicit controls.
- [ ] Require successful nontrivial YAML formatting with comments, a multi-document stream, and anchors/block scalars, plus refusal for unqualified constructs. A candidate that merely declines every YAML format has failed. If no candidate meets these controls, report the exact failing fixture and revisit the formatter design before E4/I2; do not ship a generic dump fallback.
- [ ] Add Hypothesis-generated strict JSON cases comparing token sequences and idempotence, plus Unicode/CRLF/EOF position cases. Ensure source-bearing parser exceptions are normalized before logging or worker return.
- [ ] Run the four explicit contract/JSON/YAML/position files. Record candidate version, license, fixture hash, supported constructs, rejected constructs and startup/memory observations in the qualification report. Pin accepted dependencies through existing packaging conventions and commit the qualified engine; no parser work occurs at application startup import.

## E3: Browser JSON/JSONL and YAML qualification

**Dependencies:** E1; compare results with E2 before release, not necessarily before coding.
**Files U:** create `src/services/document-editing/engine.ts`, `json-document.ts`, `yaml-document.ts`, `source-positions.ts`; create `__tests__/json-document.test.ts`, `yaml-document.test.ts`, `source-positions.test.ts` in that directory. Modify `U/package.json` and `S/apps/bun.lock` only after candidate qualification. Write `S/Docs/superpowers/reviews/2026-09-27-document-engine-browser.md`.
**Interfaces:** programme `checkDocument`/`formatDocument`; same fixture categories and scalar-offset output as Python. JavaScript numeric object values are never the formatting source.

- [ ] Add a failing precision test using the E1 TS request factory:

```ts
it("formats source without rounding a JSON number", () => {
  const result = formatDocument(requestFor('{"n":9007199254740993,"z":-0}'));
  expect(result.state).toBe("formatted");
  expect(result.replacement).toContain("9007199254740993");
  expect(result.replacement).toContain('"z": -0');
  expect(formatDocument(requestFor(result.replacement!)).state).toBe("unchanged");
});
```

- [ ] Run the JSON selection red. Qualify Microsoft's `jsonc-parser` scanner/visitor/format edits with strict settings: `disallowComments: true`, `allowTrailingComma: false`, `allowEmptyContent: false`. Check every error list; tolerant parse output never establishes validity. Formatting applies text edits to source, with token-stream equality checked afterward.
- [ ] Track duplicate decoded keys and early nesting limits during visitation, never from `JSON.parse` object keys. Format each JSONL record compactly without changing its token sequence or line boundary. Handle CRLF and the optional final terminator without JavaScript `split` producing a phantom empty record.
- [ ] Qualify the `yaml` package's source-aware document/token APIs, using YAML 1.2 defaults and supported explicit directives. Keep big integer/source tokens and unexecuted custom tags. Apply the same YAML preservation/refusal gate as E2 and record actual supported constructs; the two engines must agree on validity/diagnostic category for supported fixtures.
- [ ] Translate UTF-16 parser offsets to Unicode scalar offsets by counting code points before each range. Use raw-source slices when checking tokens; do not round-trip through resolved JS values. Test emoji before an error, combining characters, EOF, tabs, and CRLF.
- [ ] Run the explicit contract/JSON/YAML/position tests from U; run scoped lint/typecheck and a shared module loading check in the web and extension bundlers. Record/pin only qualified runtime dependencies; repository dev-only Prettier is not an implicit browser formatter dependency. Commit after the qualification report passes.

Primary candidate references: [jsonc-parser](https://github.com/microsoft/node-jsonc-parser), [yaml](https://eemeli.org/yaml/), [ruamel.yaml](https://yaml.dev/doc/ruamel.yaml/detail/). These describe available APIs, not evidence that our invariants pass.

## E4: Worker ownership, bounds, and stale-result handling

**Dependencies:** E2 for C; E3 for U.
**Files C:** create `tldw_chatbook/Document_Editing/worker.py`, `worker_process.py`, `scheduler.py`; create `Tests/Document_Editing/test_worker_lifecycle.py`, `test_scheduler.py`.
**Files U:** create `src/services/document-editing/worker.ts`, `worker-client.ts`, `scheduler.ts`; create `__tests__/worker-client.test.ts`, `scheduler.test.ts`.
**Interfaces:** Python `DocumentWorker.check(request)` / `.format(request)` are async and return programme results; `.close()` terminates/joins resources. TS `DocumentWorkerClient.check(request)` / `.format(request)` return Promises; `.close()` terminates the browser worker. `DocumentCheckScheduler.submit(request)` replaces pending checks and starts after 400 ms idle; `.close()` clears timers and references.

- [ ] Add lifecycle regressions with a deliberately stalled child and a successful control. Launch the real child, first prove a small valid document checks successfully, then stall a test-injected engine and assert deadline result plus child termination. Avoid tests that only mock `terminate()` or treat any exception as timeout success.
- [ ] Add a scheduler test using a controllable worker:

```python
@pytest.mark.asyncio
async def test_only_latest_pending_revision_survives(scheduler, worker, request_for):
    worker.hold_first_call()
    scheduler.submit(request_for("{}", draft_revision=1))
    await worker.wait_for_first_call()
    for revision in range(2, 50):
        scheduler.submit(request_for("{}", draft_revision=revision))
    worker.release_first_call()
    await worker.wait_for_calls(2)
    assert [r.key.draft_revision for r in worker.requests] == [1, 49]
```

The test-local worker fixture owns an asyncio gate and records real requests;
`wait_for_calls` waits on an event with a bounded assertion timeout. Use a fake
timer only for debounce scheduling, not to simulate operating-system child death.

- [ ] Use a spawn-context Python child with a structured bounded pipe protocol and explicit parent/child shutdown; no shell command and no `pickle` from an external source. The parent sends only trusted model objects to its own child. On a two-second deadline, terminate, join, close pipes, then allow a replacement. Browser client terminates the Web Worker and settles affected promises similarly.
- [ ] One service per app manages active-editor leases; inactive mounted editors release their lease and pending references. Limit each active owner to one in-flight snapshot plus the latest pending snapshot; formatting shares that owner rather than launching an unbounded second lane.
- [ ] Check UTF-8 size and excerpts before dispatch; reject lone surrogate/unsupported source representation explicitly without corrupting original text. Enforce depth during parsing. Cap diagnostics independently of parse completion, so a 100-item cap never masquerades as a complete valid result.
- [ ] Qualify child startup, shutdown mid-initialization, timeout, malformed reply, worker crash, account switch, canceled awaiter, alias-heavy source, and rapid navigation. Assert no source appears in logs and no worker retains credentials or filesystem authority. Measure UI-side copying/encoding on the 1 MiB boundary.
- [ ] Run the four worker/scheduler selections plus engine controls. Commit each runtime's owner separately; report process/browser support limits explicitly.
