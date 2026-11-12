# Exact text-span digests: review and execution evidence

Date: 2026-09-26
Task: [TASK-25907.12](../../../backlog/tasks/task-25907.12%20-%20Add-exact-text-span-digests-to-shared-profile-core.md)
Starting revision: `0720fa68e18c00580d8eb8645b97cf5ec01a79bd`
Branch: `codex/personal-context-memory-baseline`
Execution: native Python 3.12.11 in the existing isolated worktree.

[Plan](../plans/2026-09-26-personal-context-exact-span-digests.md)
· [API and limits](../../../backlog/docs/personal-context-exact-span-digests.md)
· [ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md)

## Result

Added the pure exact_text_span_digests function and frozen named digest pair.
It hashes strict UTF-8 for the entire exact representation and its half-open
codepoint slice. Exact built-in str/int checks reject subclasses, booleans and
floats before source methods or range checks run; invalid bounds reject instead
of Python's default clipping/negative slicing. Unencodable codepoints anywhere
in the representation reject. Empty-span calculation remains explicit and
does not imply evidence support.

## Review findings and corrections

The user endorsed the concrete slice subject to review. Inline plan review
found the original isinstance checks allowed subclasses to override encoding
or comparisons; the plan and implementation now use exact built-in types,
with three subclass controls. Inline review also improved the RED sequence:
a missing-module collection error alone cannot qualify rejection behavior.
The first test imports in its own body, then the remaining negative controls
run against the minimal calculation before validation is added.

One bounded independent reviewer inspected the plan and later the actual
new modules. The plan review found no additional material issue beyond those
two resolved corrections. The implementation review found no Critical,
Important or Minor findings. It confirmed the exact convention, immutable
result, 30 actual test cases and unchanged existing tracked core/native files.
The reviewer ran no tests or provider calls; test/static evidence below belongs
to the native executor. API documentation was written after the independent
code review and received inline review here; it was not independently reviewed.

The custom str test initially failed collection because pytest called its
encode method with unicode_escape to generate a parameter ID. This reproduced
with collect-only, and installed pytest's ascii_escaped implementation confirmed
the cause. Explicit pytest.param IDs fixed the fixture without changing the
production calculation. Those collection failures are not counted as runtime
RED evidence. The incident is recorded in
[testing lessons](../../../backlog/docs/lessons-testing-evidence.md).

## Executed checks

Each pytest command used the worktree's `.venv/bin/python` with
`PYTHONPATH=.:packages/tldw_profile_core/src`. Fresh explicit basetemp roots
avoided unrelated pytest temporary-directory cleanup debt.

| Stage | Command selection | Result |
| --- | --- | --- |
| Canonical RED | `-m pytest packages/tldw_profile_core/tests/test_evidence.py -q --basetemp=/private/tmp/exact-span-import-red-20260926` | Exit 1; one failed test, ModuleNotFoundError for the absent API; no collection error |
| Canonical minimal GREEN | same test module, `--basetemp=/private/tmp/exact-span-canonical-green-20260926` | Exit 0; 1 passed in 0.10s |
| Fixture issue | full new module, then `--collect-only --tb=long` | Exit 2 twice; custom encode invoked by pytest parameter naming, corrected before meaningful RED |
| Input-validation RED | full new module, `--basetemp=/private/tmp/exact-span-input-red-fixed-20260926 --tb=short` | Exit 1; 14 failed / 16 passed in 0.16s against the calculation without type/range checks |
| Affected GREEN | the five modules below, `--basetemp=/private/tmp/exact-span-core-green-20260926 --junitxml=/private/tmp/exact-span-core-green-20260926.xml` | Exit 0; 181 passed in 0.48s |
| Final after import sorting | five modules below, `--basetemp=/private/tmp/exact-span-core-final-20260926 --junitxml=/private/tmp/exact-span-core-final-20260926.xml` | Exit 0; 181 passed in 1.05s |

The exact affected command is:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q --basetemp=/private/tmp/exact-span-core-final-20260926 --junitxml=/private/tmp/exact-span-core-final-20260926.xml
```

The JUnit receipt contains 181 unique cases, including 30 new helper cases and
151 existing V1/shared-core cases, with zero failures/errors. Repeats are not
added to the distinct count. Fixed expectations cover precomposed/decomposed
accents, CRLF/LF, astral text, outside-span source edits and shifted ranges.
The native documentation example executed successfully; import paths point to
this worktree and SERIALIZED_SCHEMA_VERSION remains 1.

Whole-file static checks passed for both new modules:

```bash
.venv/bin/python -m ruff check --no-cache packages/tldw_profile_core/src/tldw_profile_core/evidence.py packages/tldw_profile_core/tests/test_evidence.py
.venv/bin/python -m ruff format --no-cache --check packages/tldw_profile_core/src/tldw_profile_core/evidence.py packages/tldw_profile_core/tests/test_evidence.py
git diff --check
```

The first Ruff run found one I001 in the new test import group. It was fixed
with Ruff's import sorter and both full-file checks reran successfully. No
unrelated lint debt was hidden or changed. No dependency, license, persistent
state or native route was added; the pure calculation's input-dependent work
and temporary memory are documented rather than given an invented source cap.

## Delivered scope and limits

The new primitive has no native consumer activation, source I/O, logging,
repository write or provider behavior. Existing V1 exports, schemas, canonical
bytes, 45 fixture files in both source/package locations, integrity checks,
native profile-tool substring handling and historical evaluation reports remain
unchanged. The result is not a canonical binding or an authorization receipt.
SHA-256 identities remain governed source metadata; an encoding exception can
carry its failing text, so future callers must not disclose the exception object.

No source resolver, support assessment, temporal admission, migration,
retirement/forgetting, destination/purpose disclosure or server conformance is
qualified by these checks. The known device-only disclosure gap remains. This
slice does not activate V2, quoted evidence or consolidation. No real profile,
app launch, full application test sweep, network/provider execution, push, PR
or merge occurred. The independent TASK-25907.10 addition is preserved.

ADR required: no new ADR. ADR-201 supplies this exact data convention and
ADR-182 retains ownership. The local branch/worktree stay in place.

## Closeout validation

All six task criteria are checked, Implementation Notes are recorded and the
Backlog CLI set TASK-25907.12 Done at its correct path. All nine native plan
steps are complete. Scoped validation passed across seven documents: 74 local
Markdown links, 68 task reference/documentation paths, 13 unique family IDs,
three parsed Python examples, all 59 unchanged original criteria and .11's
six unchanged criteria. The new task retains all six original criterion texts.

The repository Backlog guard passed for the exact 13-file task family,
including both filename/frontmatter IDs and Windows path rules. Closeout
allocation checks covered 566 available refs and 79 worktrees across active,
draft, completed and archived buckets: only the existing canonical .12 path
was found. No fetch or open-PR query was made; integration must recheck.

The exact owned change set is eight paths. Existing V1/native/fixture/history
files and unrelated lesson content remain unchanged. Independent .10 and its
roadmap suffix match their original SHA-256 values. Whitespace passes; only
the owned roadmap prefix is staged. The branch/worktree are preserved locally.
