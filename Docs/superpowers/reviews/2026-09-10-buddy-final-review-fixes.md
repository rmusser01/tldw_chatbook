# Buddy/Petdex final review fix report

Date: 2026-09-10
Fix base: `0e4622a1055953fe9af60501141c2d7d84cf0cba`
Task: TASK-32238

## Scope and outcome

This wave closes the authoritative `final-review.md` list: 0 Critical, 4 Important
and 2 Minor findings.

1. Local Petdex folder, `pet.json` and ZIP reads now probe the complete POSIX
   descriptor capability set and otherwise use a bounded `lstat`/open/read/recheck
   path. The fallback rejects linked and special entries, snapshots every ancestor
   and visited directory, retains package/member/depth/byte limits, and makes changed
   sources stale.
2. Accepted ADR-139 is restored byte-for-byte to current `origin/dev`. New Accepted
   ADR-146 records the independent Buddy publication and character-creation entry
   points, explicitly supersedes only ADR-145's saved-Persona-only restriction, and
   retains ADR-145's saved Persona route plus its source and HTTPS trust decisions.
   ADR-145 carries the permitted partial-supersession metadata link. Proposed ADR-074
   remains unchanged by this fix wave. The decision index, task plan, integration
   plan and programme spec now point to ADR-146.
3. The reviewed `left_rail.py` fail-soft debug statement is pinned in the regenerated
   production diagnostic inventory.
4. The staged hint, guide, plan, spec and ADR now qualify cancellation as side-effect
   free before Apply/Save. A publication-then-settings failure reports that the Buddy
   is installed, previous settings remain selected, and Retry Apply or reopen/verify
   is the recovery. No destructive publication rollback was added.
5. One shared path-free publication-error translator serves staged Petdex and native
   Buddy pack publication. Changed sources direct the user to a fresh review; other
   publication failures direct the user to profile storage permissions/free space.
6. Only the reviewer-named Ruff formatting hunks were changed in
   `personas_screen.py` and `personas_library_pane.py`. Unrelated baseline drift in
   `personas_screen.py` was left untouched.

The directly affected coordinator retry test used a plain `Path` where production
requires an immutable review snapshot. Its fixture now uses `BuddySnapshot`, so the
test exercises publication caching, rollback metadata and retry rather than failing
at `dataclasses.replace` before publication.

## Test evidence

Every `python` below denotes this exact command prefix:

```text
PYTHONPATH=/private/tmp/chatbook-buddy-qualification:/private/tmp/chatbook-buddy-qualification/packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

TDD discrimination before the implementation changes:

- The 12 new Petdex platform cases produced 6 expected failures for missing POSIX
  flags and unsupported descriptor-relative operations; the existing changed-source
  and linked-source paths remained green before they were forced through the fallback.
- The staged/native publication and partial-Apply assertions produced 7 expected
  failures before the shared translator and recovery copy were added.
- The mounted settings-failure journey produced 1 expected failure on the old generic
  save error before the partial-Apply message was added.

Fresh final targeted runs:

```text
python -m pytest -q Tests/Petdex/test_sources.py Tests/Petdex/test_conversion.py Tests/Petdex/test_publication.py
39 passed, 1 dependency warning in 3.90s

python -m pytest -q Tests/UI/test_buddy_management_petdex.py Tests/Persona_Buddy/test_buddy_management_import.py Tests/UI/test_buddy_management_modal.py
57 passed, 1 dependency warning in 72.73s

python -m pytest -q Tests/Persona_Buddy/test_buddy_management_coordinator.py
17 passed, 1 dependency warning in 1.95s
```

The environment also emitted non-test cleanup warnings for an unrelated concurrent
pytest garbage directory; each pytest process exited 0. No full pytest sweep was run.
No Windows host was available, so Windows support is established by capability-forced
tests that remove POSIX flags/reject `dir_fd` use, plus retained stale/link rejection.

## Static, generated-asset and governance checks

```text
python -m ruff check tldw_chatbook/Petdex/sources.py tldw_chatbook/UI/Navigation/buddy_management.py tldw_chatbook/Widgets/Persona_Widgets/buddy_management_modal.py Tests/Petdex/test_sources.py Tests/Persona_Buddy/test_buddy_management_coordinator.py Tests/Persona_Buddy/test_buddy_management_import.py Tests/UI/test_buddy_management_modal.py Tests/UI/test_buddy_management_petdex.py
All checks passed!

python -m ruff format --check tldw_chatbook/Petdex/sources.py tldw_chatbook/UI/Navigation/buddy_management.py tldw_chatbook/Widgets/Persona_Widgets/buddy_management_modal.py Tests/Petdex/test_sources.py Tests/Persona_Buddy/test_buddy_management_coordinator.py Tests/Persona_Buddy/test_buddy_management_import.py Tests/UI/test_buddy_management_modal.py Tests/UI/test_buddy_management_petdex.py
8 files already formatted

python -m ruff format --check --diff tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py
personas_library_pane.py already formatted; the only reported personas_screen.py
diffs are the six pre-existing baseline hunks near 1924, 1976, 2038, 2058, 6848 and
15547. None of the reviewer-named Buddy hunks remains in Ruff output.

python -m tldw_chatbook.css.build_css
CSS build complete; 53 modules; 413,413 characters; widget/screen defaults built

python tldw_chatbook/css/check_bundle_sync.py
CSS bundle, both widget-default bundles, both screen bundles and all six split
screen/feature bundles reproduce from source

python scripts/check_persistent_diagnostic_inventory.py --statements tldw_chatbook/UI/Console_Modules/left_rail.py --since 617a0843f459668e4d3e960ff4a8f0309e61d26b
2 -> 3 calls; 0 moved, 0 removed, 1 added: debug 0424c30e7466a134 at line 1439

python scripts/check_persistent_diagnostic_inventory.py --diff
No drift; 594 owners, 1,351 TASK-492 calls, 55 TASK-31551 calls,
7,652 TASK-494 calls and 12 sink files

git diff --exit-code origin/dev -- backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
exit 0 (Accepted ADR-139 restored exactly)

git diff --check
exit 0
```

Immediately before ADR creation, the required all-ref and all-worktree decision-file
scan found ADR-145 as the highest claimed ID and no ADR-146 claim. ADR-146 was then
created and indexed. The added diagnostic is a constant debug-level message with
exception metadata only; it contains no user content, secret, path or URL and keeps
the disposable geometry remount fail-soft.

The earlier exact-app downloaded-source qualification receipt remains at
`Docs/superpowers/reviews/2026-09-10-independent-buddy-journey-verification.md`.
Per the final-fix brief, root owns the final downloaded-source app rerun and complete
preflight after integrating the current `origin/dev` head.
