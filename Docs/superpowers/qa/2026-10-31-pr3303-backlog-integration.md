# PR3303 backlog integration evidence

Source PR head: `bf79400bf0c5e8979334d359a85ca169b0b3b740`.
Rebased onto dev: `0c9ec2825d3ca16848628ca340247496d96df963`.

The independent source review found that a short first prompt could be copied in full into the close/quit dialog through its automatic tab title. The repair uses `Tab N` in actual tab-strip order for every title kind, including explicit renames. Loss-free tabs still occupy their ordinal positions. The tab strip itself is unchanged. ADR-098 records this content-free disclosure rule; ADR-033's ownership and non-persistence rules remain intact.

Current dev's recovered-image admission/drain, view-worker retirement, config shutdown, lazy no-controller guard, fleet revision fence and runtime disposal owners were preserved. Existing notice counting, duplicate logical parked-input counting and decrease-only reconfirmation remain tracked in TASK-33628.16/.18/.19. Positional labels eliminate title-only reconfirmation without changing the revision fence.

## Local receipts

Candidate receipts are under `%USERPROFILE%/.codex/worktrees/pr3298-resync/`; baseline receipts are under sibling `pr3301-resync/`. Each contains the exact command, raw source hashes before/after, pytest log, JUnit XML and result JSON. All runs below report unchanged source hashes. Runs were sequential with fresh private profiles and no real provider requests. Counts are per receipt and overlap; they must not be added as unique coverage.

| Receipt | Result | Meaning |
| --- | --- | --- |
| `r-8c0d18fa` | 2 failed | Close reproduced the full short-prompt leak; the other case failed before parking its Temporary turn and is not causal RED evidence. |
| `r-2c19bcf2` | 6 failed | Five title-kind controls plus a real Temporary send / quit from Home causally demonstrated the privacy defect. |
| `r-9cf3c1ad` | 43 passed, 2 failed | Both Home privacy cases and close privacy passed. Failures were an earlier parking prerequisite and a stale `Chat 1` expectation. The fixture now waits for actual composer ownership, focus and enabled Send before pressing Enter; this is not proof of the cause of every prior timeout. |
| `r-9a44d185` | 28 passed | Final production code: all 14 close-impact cases, all nine real quit-loss cases, the corrected app-quit label assertion and four exact changed-module size checks. Includes actual Rename modal, Home navigation and Stay preservation. |
| `r-3ec45566` | 5 passed, 3 failed | Existing close scheduling/refusal/landed-close/retry/compact-layout journeys passed. Two narrow-wrapper cases could not find their ownership fixture. Pending approval close reached the disclosure and removed the tab, then failed the combined cancellation/round-settlement assertion. |
| baseline `r-add7c496` | 1 failed | Exact dev reproduces that same post-removal pending-approval predicate. Neither conjunct was isolated; later round kinds were not reached. No introduced regression was established. |
| `r-4482a256` | 21 passed, 6 failed, 7 setup errors | Browser quit fence (10), recovered images (2), view-worker retirement (4), quit under modal (2), vanished prompt (2) and close under quit (1) passed. Two wrapper cases imported ownership but did not expose the helper's factory namespace. Four workflow cases failed at ingest before quit; seven failed at the obsolete RunHooksEngine constructor already tracked in TASK-35577. |
| `r-a0d198f8` | 2 passed, 1 failed | Corrected wrapper exposes the original factory through the existing ownership-fixture discovery seam. Internal-close error redaction/retry and undrained-close state preservation pass. Pending approval completes; the next `chat_create` iteration fails while arming, before close. Remaining kinds are not qualified. |
| baseline `r-c575bcbb` | 5 failed, 7 setup errors | All 11 workflow outcomes match the candidate by test identity and failure condition: four ingest failures before quit and seven obsolete-constructor setup errors. The corrected pending-close wrapper also passes approval and fails arming `chat_create`, matching the candidate. The tracked baseline tree was verified identical to dev before and after this run. |

The temporary narrow wrapper executes the existing helper functions and their original scoped cleanup. Its ownership correction captures the actual app/runtime/databases and preserves disposal waits and loud teardown failures. It substitutes no production owner. It is retained with the external receipt evidence, not shipped as a new test entry.

## Static and inventory checks

- Ruff passes for the new loss inventory and changed test files. The two formatted test files pass formatting checks. Existing large-file lint diagnostics are unchanged: session 10 on base and candidate, app lifecycle zero on both, no introduced diagnostics.
- Module measurements are session 6,982 lines and app lifecycle 2,147 lines. Their exact pins reflect the user's size waiver; the incoming duplicate composer budget was removed, preserving dev's single 7,154-line row.
- The UI census retains all 1,249 dev entries as an exact prefix and adds four entries, for 1,253. Every path exists. The two new Home privacy parameter cases took 52.265 seconds together in `r-9cf3c1ad`, below their shared 60-second census allowance.
- `git diff --check` passes. No full suite, hosted CI or new live-provider qualification is claimed. Hosted CI and size limits were explicitly waived for this backlog integration; independent requesting-code-review replaces Qodo.

## Remaining qualification

The pending-round baseline failure has unresolved cause and remains visible here, linked to the owning TASK-33621.16. TASK-35587 describes a different shutdown CancelledError and is not evidence for this failure. Creating a separate follow-up task was rejected by automatic approval review as beyond validation scope; no task was created. Baseline parity is not a claim that the workflow or later pending-round behaviors are correct. Their cause and missing coverage remain qualified.

The independent reviewer inspected production preservation, ordinal order, regression assertions and receipts. Four stale title expectations and an unused title-length constant/comment identified during integration review were corrected. Final immutable-head approval and merge/protection-restoration evidence are retained outside the checkout in the PR handoff, so this document does not claim a merge before it occurs.
