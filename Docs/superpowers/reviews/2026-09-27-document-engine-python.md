# Python document-engine qualification — E2 / TASK-34654

Date: 2026-10-07. Immutable task base: `b7141cb0d381fbce4cb4a9f09b723beba5e5837d`.
Status: qualified for the pure-engine scope, with the limits below. Worker/editor integration is not part of this evidence.
Governing decision: [ADR-194](../../../backlog/decisions/194-structured-note-language-and-local-editor-validation.md).

## Dependency decision (recorded before packaging adoption)

Accept `ruamel.yaml==0.19.1` as a base dependency, pure Python `YAML(typ="rt", pure=True).scan/parse` instance APIs only. Default installation has no required transitive packages; native extensions are optional extras and are not installed for this feature. No load, compose, application constructors, serializer, or dump call occurs in production. Existing PyYAML remains used by legacy note import; its constructors and YAML 1.1 defaults do not satisfy this editor contract.

Candidate distribution metadata: version 0.19.1; MIT license, copyright 2014–2026 Anthon van der Neut / Ruamel bvba. Installed only into `/private/tmp/tldw-structured-chatbook-venv` for qualification; shared dependencies and editable bindings were not changed. Wheel SHA-256 (`ruamel_yaml-0.19.1-py3-none-any.whl`): `27592957fedf6e0b62f281e96effd28043345e0e66001f97683aa9a40c667c93`. Installed LICENSE SHA-256: `3e7632bb0dc1150f15f8161f5eec48dec8a99aaa1dfc9e8f63c02931e0fba7f8`.

Upstream references retrieved 2026-10-07: [ruamel API/dependency notes](https://yaml.dev/doc/ruamel.yaml/), [round-trip details](https://yaml.dev/doc/ruamel.yaml/detail/), [YAML 1.2.2 Core specification](https://yaml.org/spec/1.2.2/). Acceptance is based on the actual candidate tests, not the upstream round-trip claim. A real candidate `load`/`dump` experiment changed `-0` to `0`, `1e400` to `.inf`, and `0.12345678901234567890123456789` to `0.12345678901234567736988623210`. That formatting path is rejected.

## Actual owners and format admission

E1 owns `Document_Editing/models.py` and the byte-identical fixture below. `CheckResult.for_request` validates result/source ranges; `FormatResult` is the existing frozen constructor with the matching request key and authoritative check. The already-existing `Widgets/document_editing_bridge.py` owns revision/editability/history admission; this engine does not call it. `Notes/note_import_parsers.py` owns legacy/portable file imports and keeps its current PyYAML path. No engine call is added to startup, a UI, a save owner, or a filesystem/network path.

JSON uses a depth-bounded lexical pass plus strict stdlib `json.loads`, rejecting `parse_constant`, `parse_int=str`, `parse_float=str`, and duplicate-aware pairs. Stdlib `scanstring` finds string boundaries without a high-allocation per-character regex. Decoded keys are compared within each object scope, with exact original source spans. Formatting only substitutes whitespace around the original tokens; it compares the complete nontrivia token stream. JSONL does this per LF/CRLF physical record, preserving each existing terminator and final-newline presence. Empty JSONL has zero records. Visible or metadata-only JSONL BOMs block formatting.

YAML scans tokens, then walks parser events with a depth guard. It records symbolic collection/scalar/anchor/alias/tag/version order without following aliases, including cyclic/large alias graphs. YAML 1.2 uses Core scalar resolution plus the approved merge compatibility extension; 1.1 uses the candidate's versioned scalar resolver. Scalar-key identities are tag-aware and exact (integer/Decimal normalization, no binary floating point); different types do not collide merely because Python would compare their constructed objects equal. Merge overrides do not duplicate explicit keys. Undefined aliases are errors, and unqualified merge targets decline.

YAML formatting edits only spaces/tabs outside tokens/comments and uses two-space block indentation. Before return, it reparses the candidate and compares its event graph, every raw scalar/string/number/tag/anchor/directive/marker token, and exact comment/significant-trivia sequence. Multiline scalar token bodies and chomping are never rewritten. Failed output verification returns a source-free actionable failure with no replacement. Both engines produce deterministic idempotent proposals, and unchanged results carry no edit.

## Supported and explicitly refused YAML

Supported: empty streams; comments; single/multiple documents and explicit markers; consistent 1.1/1.2 directives; Core string/null/bool/integer/float scalars; qualified standard scalar/collection tags; quoted escapes and numeric lexemes; indented block mappings/sequences and flow collections; anchors/aliases (including symbolic cycles); qualified mapping merges and explicit overrides; literal/folded/chomped scalars whose token bytes require no reindent. A nontrivial successful formatting test changes indentation and separator spacing in a commented, anchored, merged, multi-document stream with exact block content, large numbers, high-precision decimals, and quoted escapes. Nested block-sequence formatting has its own exact-output/idempotence test.

Explicit unsupported results (not invented syntax errors): mixed effective YAML-version streams; directives other than 1.1/1.2; custom/executable tags; non-Core/unqualified tags (including implicit YAML 1.1 timestamps); tag/node-kind or tag/value combinations not qualified; complex/alias mapping keys; unqualified merge values; indentless block sequences; reindenting a multiline scalar's header when that would require changing its original scalar token. These decline with a reason rather than a generic dump fallback. Implicit YAML 1.2 dates remain Core strings. Ordinary duplicate mapping keys are errors. YAML 1.1 octal/boolean and exact numeric-equivalence controls are exercised.

## Limits and observations

Pure engine checks 1 MiB UTF-8 source/replacement size, nesting depth 100 before JSON construction and during YAML token/event walks, 100 displayed diagnostics (truncation never claims completeness), and cooperative two-second monotonic deadlines at parser-token/event/format checkpoints. Numeric key conversion implementation limits are resource-limited. Unknown runtime exceptions are failures, never fabricated resource successes. Parser messages/snippets are neither logged nor returned.

The engine does **not** enforce hard termination or a heap/RSS ceiling. A parser call, encoding, allocation, or cleanup can run between checkpoints. Later E4 must own an isolated terminable process, deadline enforcement, retirement, admission/copy accounting, and process lifecycle. Checks here do not establish typing responsiveness, worker IPC, editor save limits, app-startup budgets, or crash isolation. They grant no file/write authority and do not override the existing 200,000-character File Notes limit.

Fresh macOS arm64 / Python 3.12 child observations below are single runs, not guarantees. RSS is whole-process high-water bytes, including interpreter and source/token/event allocations. Each child builds only its own fixture. Encoding is measured separately. A separate minimal `python -I -c` import measured 0.0712 seconds / 38,109,184 bytes RSS with neither YAML nor JSON adapter loaded. Engine/package binding resolved to the isolated checkout. These are import observations, not a mounted application startup test.

```text
IMPORT {"engine_seconds": 0.047874542069621384, "rss_bytes": 38404096, "yaml_loaded": false, "engine_path": "/Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/tldw_chatbook/Document_Editing/engine.py"}
RESOURCE {"probe": "json-positive", "bytes": 50001, "encode_seconds": 1.291045919060707e-06, "check_seconds": 0.020715750055387616, "state": "checked", "complete": true, "codes": [], "rss_bytes": 44679168} child_wall_seconds 0.10097483405843377 exit 0
RESOURCE {"probe": "json-large-string", "bytes": 1048576, "encode_seconds": 1.40840420499444e-05, "check_seconds": 0.0015290420269593596, "state": "checked", "complete": true, "codes": [], "rss_bytes": 40730624} child_wall_seconds 0.08190262503921986 exit 0
RESOURCE {"probe": "json-dense", "bytes": 1048575, "encode_seconds": 1.5249941498041153e-05, "check_seconds": 0.378625167068094, "state": "checked", "complete": true, "codes": [], "rss_bytes": 189677568} child_wall_seconds 0.46343058394268155 exit 0
RESOURCE {"probe": "yaml-positive", "bytes": 90, "encode_seconds": 5.409820005297661e-07, "check_seconds": 0.005449375021271408, "state": "checked", "complete": true, "codes": [], "rss_bytes": 38977536} child_wall_seconds 0.09952137502841651 exit 0
RESOURCE {"probe": "yaml-large-scalar", "bytes": 1048576, "encode_seconds": 1.591700129210949e-05, "check_seconds": 0.29082308302167803, "state": "checked", "complete": true, "codes": [], "rss_bytes": 45531136} child_wall_seconds 0.3838438750244677 exit 0
RESOURCE {"probe": "yaml-wide", "bytes": 348890, "encode_seconds": 2.8084032237529755e-05, "check_seconds": 1.2290312080876902, "state": "checked", "complete": true, "codes": [], "rss_bytes": 105267200} child_wall_seconds 1.3264175419462845 exit 0
RESOURCE {"probe": "yaml-deadline", "bytes": 948890, "encode_seconds": 7.504189852625132e-05, "check_seconds": 2.0224202499957755, "state": "resource_limited", "complete": false, "codes": ["resource_limited"], "rss_bytes": 186580992} child_wall_seconds 2.122623416944407 exit 0
RESOURCE {"probe": "json-depth", "bytes": 203, "encode_seconds": 3.7497375160455704e-07, "check_seconds": 0.0001989169977605343, "state": "resource_limited", "complete": false, "codes": ["resource_limited"], "rss_bytes": 37502976} child_wall_seconds 0.08177916600834578 exit 0
RESOURCE {"probe": "yaml-depth", "bytes": 203, "encode_seconds": 4.169996827840805e-07, "check_seconds": 0.006117291050031781, "state": "resource_limited", "complete": false, "codes": ["resource_limited"], "rss_bytes": 39321600} child_wall_seconds 0.08598916698247194 exit 0
RESOURCE {"probe": "utf8-size", "bytes": 1048578, "encode_seconds": 0.0001882920041680336, "check_seconds": 0.0002697919262573123, "state": "resource_limited", "complete": false, "codes": ["resource_limited"], "rss_bytes": 40583168} child_wall_seconds 0.08072545798495412 exit 0
```

The dense JSON and deadline YAML cases reached about 190 MB and 187 MB RSS respectively. No low-memory guarantee is claimed. The observed YAML deadline result at about 2.02 seconds illustrates cooperative cleanup overhead. Replacing the first string-scanning regex with stdlib `scanstring` reduced the 1 MiB string probe from an observed 172 MB high water to about 41 MB; earlier probes built extra fixtures and are not treated as exact before/after benchmarks.

## Reproduction and scoped verification

C: `/Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook`.
Interpreter: `/private/tmp/tldw-structured-chatbook-venv/bin/python`.
No full suite, user settings/data, scheduler, or UI was exercised.

```sh
HYPOTHESIS_STORAGE_DIRECTORY=/private/tmp/tldw-task8-hypothesis /private/tmp/tldw-structured-chatbook-venv/bin/python -m pytest Tests/Document_Editing/test_contract.py Tests/Document_Editing/test_json_document.py Tests/Document_Editing/test_yaml_document.py Tests/Document_Editing/test_source_positions.py -q --confcutdir=Tests/Document_Editing -o cache_dir=/private/tmp/tldw-task8-pytest-cache
```

Pre-adoption result: **120 passed in 0.28s**, exit 0, including 120 generated strict-JSON examples, frozen fixtures, real parser spans, actual positive formatting, deliberate refusal controls, and injected preservation failures. `--confcutdir` excludes unrelated app fixtures for this pure-engine selection. Scoped Ruff passes, Black (`--target-version py312 --check`) leaves all seven new files unchanged, and Bandit reports no production-source findings, each exit 0. Existing Python files were not modified, so no inherited Python debt is hidden by these new-file checks. The only packaging edit is the qualified exact dependency pin; TOML/base-diff validation is recorded in the full task report.

Local full raw logs, resource-probe script, and wheel are in `/private/tmp/tldw-task8-evidence/`. The full execution report with exact commands/results is `.superpowers/sdd/2026-09-27-structured-document-editing/task-8-report.md` in C. Task stays In Progress for independent controller review.

## Frozen shared fixture outcomes

Fixture v1 SHA-256: `fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7`.
No fixture edits. Spans are zero-based Unicode-scalar `[start,end)` into the exact fixture source. `checked` rows are complete; `unsupported` rows are incomplete. Duplicate JSON keys are warnings; YAML duplicates and syntax/blank-record diagnostics are errors.

| Fixture | Check | Codes | Scalar spans | Format allowed | Format result |
| --- | --- | --- | --- | --- | --- |
| json-large-int | checked | — | [] | true | formatted |
| json-nan | checked | syntax | [[5, 6]] | false | blocked |
| json-duplicate | checked | duplicate_key | [[7, 10]] | false | blocked |
| jsonl-blank | checked | blank_record | [[3, 3]] | false | blocked |
| yaml-stream | checked | — | [] | true | unchanged |
| json-escaped-duplicate | checked | duplicate_key | [[7, 15]] | false | blocked |
| json-sibling-keys | checked | — | [] | true | formatted |
| json-comment | checked | syntax | [[7, 8]] | false | blocked |
| json-trailing-comma | checked | syntax | [[7, 8]] | false | blocked |
| json-infinity | checked | syntax | [[0, 1]] | false | blocked |
| json-empty | checked | syntax | [[0, 0]] | false | blocked |
| json-scalar | checked | — | [] | true | unchanged |
| json-overflow | checked | — | [] | true | unchanged |
| json-negative-zero | checked | — | [] | true | unchanged |
| json-precision | checked | — | [] | true | unchanged |
| json-escaped-string | checked | — | [] | true | unchanged |
| jsonl-empty | checked | — | [] | true | unchanged |
| jsonl-final-newline | checked | — | [] | true | unchanged |
| jsonl-no-final-newline | checked | — | [] | true | unchanged |
| jsonl-crlf | checked | — | [] | true | unchanged |
| jsonl-scalars | checked | — | [] | true | unchanged |
| jsonl-whitespace-record | checked | blank_record | [[3, 5]] | false | blocked |
| jsonl-multiline-record | checked | syntax, syntax, syntax | [[1, 1], [5, 6], [8, 9]] | false | blocked |
| yaml-empty | checked | — | [] | true | unchanged |
| yaml-comments | checked | — | [] | true | unchanged |
| yaml-anchors-aliases-merges | checked | — | [] | true | unchanged |
| yaml-directive-1.1 | checked | — | [] | true | unchanged |
| yaml-directive-1.2 | checked | — | [] | true | unchanged |
| yaml-mixed-versions | unsupported | unsupported | [[35, 38]] | false | blocked |
| yaml-complex-key | unsupported | unsupported | [[2, 3]] | false | blocked |
| yaml-custom-tag | unsupported | unsupported | [[7, 21]] | false | blocked |
| yaml-standard-tag | checked | — | [] | true | unchanged |
| yaml-literal | checked | — | [] | true | unchanged |
| yaml-folded | checked | — | [] | true | unchanged |
| yaml-chomping | checked | — | [] | true | unchanged |
| yaml-duplicate | checked | duplicate_key | [[5, 6]] | false | blocked |
| yaml-malformed | checked | syntax | [[9, 9]] | false | blocked |
