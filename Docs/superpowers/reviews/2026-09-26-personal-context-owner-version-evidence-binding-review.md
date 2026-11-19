# Owner-version evidence binding execution review

Task: TASK-25907.14; base: bf0d556467512dae2a133004f683262eeb6a3cdc.
[Plan](../plans/2026-09-26-personal-context-owner-version-evidence-binding.md),
[specification](../specs/2026-09-26-personal-context-owner-version-evidence-binding-design.md),
[API](../../../backlog/docs/personal-context-owner-version-evidence-binding.md).
ADR required: no new ADR; direct data-only convention from
[ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md).

## Requested design review

The review found the new fixture directory would not ship under the existing
setuptools package-data rules. The corrected plan adds only its resource glob
and data-files entry and qualifies actual wheel loading. It also makes instance
revalidation and structured-error/raw-JSON limitations explicit. No source
access or profile schema is introduced.

## Native verification

All commands used the native .venv Python 3.12.11 and
PYTHONPATH=.:packages/tldw_profile_core/src in the existing isolated worktree.
Only the affected shared-core modules ran; no application-wide suite or provider.

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q --basetemp=/private/tmp/memory-binding-baseline-20260926 --junitxml=/private/tmp/memory-binding-baseline-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py -q -k 'not offline_wheel' --basetemp=/private/tmp/memory-binding-red-20260926 --junitxml=/private/tmp/memory-binding-red-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py::test_offline_wheel_contains_its_own_binding_and_fixture -q --basetemp=/private/tmp/memory-binding-wheel-red-20260926 --junitxml=/private/tmp/memory-binding-wheel-red-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q --basetemp=/private/tmp/memory-binding-green-20260926 --junitxml=/private/tmp/memory-binding-green-20260926.xml
.venv/bin/python -m ruff check packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py packages/tldw_profile_core/tests/test_evidence_binding.py
.venv/bin/python -m ruff format --check packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py packages/tldw_profile_core/tests/test_evidence_binding.py
```

- Baseline: 181 distinct existing cases passed.
- Component RED: 156 expected missing-API assertions, zero errors, one fixture-parity control passed.
- Wheel RED: one expected assertion that the new fixture was absent from a genuinely built offline wheel.
- GREEN: 339 distinct cases passed (158 new, 181 existing), including code/resource loading directly from the wheel. Repeated runs are not additional cases.
- Whole-file Ruff and format passed. Initial lint findings were import order, explicit subprocess check=False and a deliberate naive-datetime rejection control; no negative assertion was weakened.
- Source ownership checks imported this worktree's core/module; all pre-existing tracked shared-core files other than the narrowly amended pyproject matched task base byte-for-byte. Parsed package metadata/dependencies/version, V1 data-file entries and package-data prefixes are unchanged.

Wheel tests use a temporary package copy, --no-deps --no-build-isolation --no-index
--no-cache-dir and a Python -I child whose package __file__ must come from that
wheel. They do not install it, alter the native environment or use a package index.
These checks do not qualify native source authorization, retained historical
text, semantic support, runtime admission or cross-server Profile V2 behavior.

## Final review checkpoint

Reviewer: /root/owner_binding_final_review, one bounded fresh read-only pass
under requesting-code-review/executing-plans. Reviewed bf0d556467..cfa89943c1
against the specification and plan. Critical: none; Important: none; Minor: none.
It independently ran a native probe for all 18 fields, source/span/binding
digests and rejection of unknown-key/surrogate unsafe copies. It inspected test
code/receipts but did not independently run the 339 cases or wheel build.

Assessment: ready within the data-only scope. Its declined areas are future
runtime authorization, historical-text retention, server interoperability and
Profile V2 admission. Those are explicitly unimplemented; the task makes no
claims about them. They are scope limits, not unaddressed implementation findings.

Root's final task run passed the same 339 distinct cases in 1.47s; receipt:
/private/tmp/memory-binding-task-done-20260926.xml. No code changed after that
run. Only review attribution, tracker and documentation closeout followed.
No repeated run is counted as extra coverage; no full application sweep ran.

Closeout checks passed all eight criteria and six plan steps, 69 local links,
15 unique task-family IDs, unchanged previous task bytes and exactly 11 owned
changed paths across planning/implementation/closeout. Allocation checks covered
available Git objects and 81 worktrees, with no competing task title; no fetch
or open-PR check was performed. Native V1/core bytes and original package
metadata are unchanged; no code changed after the final 339-case run.

Final scope ruling on the reviewer's declined areas: future source authorization,
historical-text retention, server interoperability and V2 admission remain
unimplemented. A user gets data identity only; future runtime adoption requires
separate implementation/qualification. No current authority is inferred from
unreviewed runtime behavior. No deferred Minor or material code finding remains.

Keep the branch/worktree locally under the existing approved scope. No push,
PR, merge, full application sweep or real-profile/provider access occurred.
Independent TASK-25907.10 and its roadmap suffix remain unmodified and unstaged.
