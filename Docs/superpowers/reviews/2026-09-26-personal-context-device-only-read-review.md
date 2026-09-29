# Device-only canonical agent read correction

Date: 2026-09-26
Task: [TASK-25907.16](../../../backlog/tasks/task-25907.16%20-%20Exclude-device-only-records-from-current-agent-profile-reads.md)
Status: Approved bounded native implementation, independently reviewed and verified.
Authority: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md)
and [ADR-203](../../../backlog/decisions/203-personal-context-provider-disclosure-authority.md).
ADR required: no new ADR; this enforces the existing device-only ceiling without
changing schemas, storage, grants, provider contracts or portable policy.

## Delivered behavior

The canonical authorized agent view omits device-only records. Context and tool
consumers also require syncable records before relevance ranking, workspace
overrides, whole-record budgets, explanations and serialization. A hidden
workspace override cannot suppress a permitted global record. Hidden additions
do not add omission rows/count hints, candidate IDs or expiry deadlines; opaque
freshness fences may advance when canonical state changes.

The existing shared agent eligibility guard rejects device-only direct updates,
update/archive proposals and promotions. CREATE key collisions with device-only
records use the existing content-free private-duplicate review result before
quota reservation. Denied target heads/controls remain unchanged and no target
proposal is created; routine unrelated proposal expiry housekeeping remains.
Successful syncable operations retain their existing checks and behavior.

Owner get/list/export and manual archive/restore remain available. V1 canonical
storage, shared-core schemas/conformance fixtures, Sync controls, permissions
and grants are unchanged. No UI feature or lifecycle forwarding was added.
Mechanical Ruff cleanup in affected older files preserves behavior; existing
fail-closed catches now explain their content-free boundary.

## Native verification

Native Python 3.12.11; temporary synthetic encrypted SQLite profiles,
InMemoryProfileKeyProtector, offline baseline estimator and test config/keyring
sandbox. No real profile, provider, server, keyring, network or full suite.

- Privacy RED: 12 cases failed at intended assertions before production edits
  (`/private/tmp/memory-device-only-red-20260926.xml`, 23.23 s).
- Initial focused GREEN: 13 passed, including exact-message successful syncable
  update (`/private/tmp/memory-device-only-green-20260926.xml`, 37.52 s).
- Context/service/proposal/tool/Console regression: 161 passed; the sole failure
  was an older successful proposal fixture using device-only. Its fixture now
  uses syncable, retaining inheritance assertions and separate device-only
  negatives. Initial receipt: `/private/tmp/memory-device-only-regression-20260926.xml`
  (254.30 s). Final proposal file: 31 passed
  (`/private/tmp/memory-device-only-proposal-final-20260926.xml`, 54.63 s).
- Baseline positive-control RED: one isolated d10 regression failed because the
  shortest historical candidate was the denied device record. Filter only the
  positive probe candidates by static syncability; measured outputs, relevance
  scoring and all frozen labels remain untouched. Focused GREEN: one passed
  (`/private/tmp/memory-device-only-baseline-control-green-20260926.xml`, 8.02 s).
- Final baseline/export/outbox/canonical/schema run: 116 passed
  (`/private/tmp/memory-device-only-preservation-final-20260926.xml`, 486.15 s),
  including all 24 measured cases and independent-root reproduction.
- These receipts establish 278 distinct targeted test cases; repeated
  runs are not added to that count. Nine affected Python files passed Ruff
  check and format check with `--no-cache`; `git diff --check` passed.

The first baseline preservation run stopped after its verified
`positive_search_failed` control error (2 failed, 60 passed); it is not a success
receipt. A test-only variable cleanup hit an earlier used variable; Ruff caught
it, the partial proposal run confirmed NameError, and the exact-function fix
passed the final 31-case run. Those interrupted runs do not supply completion
claims. A default Ruff cache write was sandbox-denied; `--no-cache` checks passed.

Exact targeted commands (run in the managed worktree; every pytest command also
used its matching `/private/tmp` cache, basetemp and JUnit destination):

```sh
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest Tests/Personal_Context/test_context_service.py Tests/Personal_Context/test_service.py Tests/Personal_Context/test_proposal_service.py Tests/Agents/test_profile_tool_provider.py Tests/Chat/test_console_personal_context_snapshot.py -q -o cache_dir=/private/tmp/memory-device-only-regression-cache --basetemp=/private/tmp/memory-device-only-regression-20260926 --junitxml=/private/tmp/memory-device-only-regression-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest Tests/Personal_Context/test_proposal_service.py -q -o cache_dir=/private/tmp/memory-device-only-proposal-final-cache --basetemp=/private/tmp/memory-device-only-proposal-final-20260926 --junitxml=/private/tmp/memory-device-only-proposal-final-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src TLDW_MEMORY_BASELINE_REPORT=/private/tmp/memory-device-only-after-20260926.json .venv/bin/python -m pytest Tests/Personal_Context/test_memory_baseline.py Tests/Personal_Context/test_export_service.py Tests/Personal_Context/test_profile_sync_outbox.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_schema_fixtures.py -q -o cache_dir=/private/tmp/memory-device-only-preservation-final-cache --basetemp=/private/tmp/memory-device-only-preservation-final-20260926 --junitxml=/private/tmp/memory-device-only-preservation-final-20260926.xml
```

## Frozen baseline comparison

[Before native report](evidence/personal-context-memory/device-only-preflight-v1.json)
and [after native report](evidence/personal-context-memory/device-only-v1.json)
use the unchanged 24-case fixture SHA-256
`64b076162e0e6c46d641c970cbc067f2caa5f37322f5cc9451949e109287243c`. Only d10's case observation changes.

| d10 measurement | Before | After |
| --- | --- | --- |
| Returned / selected labels | control, device | control |
| Device-only policy gap | present | absent |
| Historical selection mismatch | absent | present |
| Historical recall@3 | 1.0 | 0.5 |
| Precision@3 | 2/3 | 1/3 |
| Context bytes | 662 | 432 |

The frozen expected/relevant labels still include device. Its intentional denial
therefore lowers the historical raw retrieval score and leaves
`context_checks_passed=false`. The report's `disclosure_checks_passed=true`
means only these synthetic harness checks passed; it does not qualify all
application disclosure routes. No labels, relevance denominators, report scoring
or policy failures were rewritten to hide the change. Answer quality, semantic
support and full V2 disclosure remain unmeasured.

## Review, preservation and limits

One fresh read-only reviewer inspected the scoped runtime/test diff and then
reviewed the positive-control harness correction. Both passes found no
Critical, Important or Minor issue; the reviewer ran no tests and mutated no
files. Root checked the diff, actual receipts, successful syncable/manual
controls and frozen-case deltas. No production edits followed that review.

The scoped closeout guard checks acceptance text, 17 unique family task IDs,
local links, unchanged prior task/shared-core/repository/fixture bytes and
independent TASK-25907.10 hashes. The foreign task and roadmap follow-up suffix
remain byte-for-byte preserved and unstaged. The branch/worktree stay local;
no PR, push, merge or fetch was requested.

This corrects current canonical agent reads and target mutations only.
Previously disclosed history/caches, already captured or queued payloads,
derivatives and downstream copies are not erased or requalified. Qualified
on-device enrollment, V2 admission, source inspection, final adapter publication
fences and full retirement/disclosure still require separately qualified work.
