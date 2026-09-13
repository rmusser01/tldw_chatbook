# PR2665 rebase and review corrections

PR: https://github.com/rmusser01/tldw_chatbook/pull/2665

## Scope and provenance

The user requested current-dev integration, resolution of Qodo feedback and merge.
All work uses the existing isolated `agent-orchestration-pr` checkout. The original
published head `08037e50c6a62f776061b9636ea9b582bb3181ae` is preserved by backup
branch `codex/agent-orchestration-remaining-pre-rebase-20260913`. All 84 feature
commits were replayed from original base `d66908a69ef03066fed77a92edf77a326f44bd89`
onto dev `3e28c9e7279c9d59b37e7d57662bc8e2fd20295f`, producing `73fba740c6` before
these corrections. No feature commit was skipped. The later dev update
`a51edff97f` contains only approval-wave Backlog documentation; all 85 commits replayed cleanly onto that update.

Existing ADR148 (run hooks),153,154,155 and158 govern the changes. ADR153 is
amended to state physical resolver ownership and its limits. No new backend,
automatic checkout deletion, durable webhook guarantee or fleet feature is added.
TASK13154 is reopened for the final integration criteria. Earlier completion
records remain evidence for their historical heads.

## Qodo dispositions

The thirteen original inline comments were checked against the implementation;
review suggestions were not treated as specifications.

| Finding | Disposition |
| --- | --- |
|1 Missing Git merge identity | Preserve ADR155's configured destination identity. Preflight effective author and committer before consent and again before capture; missing identity returns an actionable no-effect refusal. Real Git tests cover removal before/during consent and successful configured attribution. |
|2 DNS can stall all webhook deliveries | Full-coroutine deadline covers lookup and POST. Normalize captured timeouts. Limit native executor admission to two jobs even after waiter cancellation; retain a retiring generation until physical settlement. Tests gate actual native DNS, saturation and thread-start failures. OS resolution itself remains uncancellable. |
|3 Environment denial-limit override | Nonblank `TLDW_AGENTS_DENIAL_CIRCUIT_BREAKER_LIMIT` wins over TOML; explicit zero disables the breaker, invalid values use the shared default, blank values use TOML. |
|4 Run-log API documentation | Document cursor/page fields and actual bounded paging behavior. |
|5 Usage event documentation | Document counters, units, attribution and best-effort observation. |
|6 Recovery API documentation | Document lifecycle, admission, ownership, arguments and outcomes. |
|7 Recovery property return type | Add `ConsoleWorktreeRecovery` return annotation via TYPE_CHECKING, preserving lazy runtime import. |
|8 Import ordering | Restore contiguous first-party imports in the MCP provider. |
|9 Duplicated denial default | Use `DEFAULT_DENIAL_CIRCUIT_BREAKER_LIMIT` for both configuration and exception fallback. |
|10 Invalid run IDs lack tests | Add thirteen malformed-input cases before any Git/path work and four service-boundary propagation cases. Existing validation was already correct. |
|11 Checkout retained after admission failure | Retention is the accepted conservative policy; deleting after authority/persistence failure would be unsafe. Add a path-free refusal explaining retention/manual review, without claiming a durable row or picker entry exists. Existing actual creation/persistence failure cases verify absent routing and retained work. |
|12 Central path validation | Use the existing canonical-directory validator on raw stored spellings and digest roots. Keep identity, binding, Git metadata, writer and post-consent checks. Canonicalize only new application-owned allocation before recording it. |
|13 Shared input validation | Add small strict action/confirmation Pydantic models. Allow remains a literal boolean; host metadata never grants authority. Invalid actions and truthy values refuse without mutation. |

## Integration corrections found during verification

- Current dev's session-close path read `session_id` before assigning it. Grant
  cleanup now follows successful ticket/generation validation and assignment;
  rejected closure preserves grants. Five reproduced failures became a 42-test
  passing shutdown/chat-create/hook selection.
- Project-instruction preview referenced live-only chat-tool closures. Both
  disposable previews now take explicit fork/new-chat availability flags derived
  from the same pair of hooks used by live dispatch. Forty-eight real controller
  combinations compare both previews with the actual provider request, including
  absent/partial/full hooks, worktree surfaces, progress inboxes and fleet sizes.
- Restored worktree approvals now emit `ApprovalRequested` exactly once at
  admission. The envelope identifies the requesting primary run (or no run for
  manual recovery), while bounded arguments retain the recovered child's ID and
  the actual apply/merge/discard action. Remounts do not emit duplicate hooks.
- Splitting Settings CSS reversed two equal-specificity rules and clipped the
  enabled Switch. Scope the existing token-backed height to its Agents form;
  unchanged paint/focus/Space-toggle tests now pass at both widths. Rebuild the
  generated bundle from sources.
- A real macOS-style symlinked temporary base exposed a producer/consumer path
  mismatch: new worktrees were recorded with aliases that strict recovery
  correctly rejects. Resolve the newly allocated parent before invoking Git;
  never normalize loaded records or selected authority.
- Add the missing query-plan census entry for the new worktree scope index only
  after checking the actual production paged query without sqlite_stat1.

## Verification evidence

Commands, raw output and exit codes are retained under
`.superpowers/sdd/2026-09-13-agent-orchestration-pr-2665/`. Each pytest command uses
the existing isolated Python3.12 environment, repository test isolation and a
fresh owned basetemp. These are targeted selections; overlapping counts are not
summed. No full suite, Windows or live-provider certification is claimed.

| Selection | Result |
| --- | --- |
| Recovery/Git/fleet/budget/denial neighbors | 320 passed |
| Real preview parity | 48 passed |
| Remaining recovery UI/lifetimes/Settings/tokens/CSS build | 98 passed |
| Context/chat-create/project-instruction/log neighbors | 121 passed |
| Final worktree approval-hook selection | 58 passed |
| Final webhook including shutdown exceptions | 40 passed |
| Canonical allocation plus recovery modules | 106 passed |
| Worktree persistence and real query-plan pin | 14 passed |

Initial failures are retained honestly: the broad root RED selection contained
thirteen intended behavior failures and thirteen fixture mistakes (an autouse
fixture already populated tmp_path); the latter were corrected without claiming
product defects. Malformed-ID production behavior already passed. The initial
UI selection had twelve preview failures and two real CSS failures, all covered
by subsequent passing selections above.

The architecture selection passed 73 cases, skipped one historical-object check,
and failed seven checks before inventory regeneration. The actual inventory
drift was corrected after inspecting every changed statement. Remaining failures
are inherited stale historical diagnostic-label expectations, Chat/Library size
ratchets, and a Console Environment closure test's final queued-job count.
The Environment test, controller and scanner are byte-identical to pinned dev;
it passes its import-deferral assertions and fails only after the real scan and
second poll. The size budgets were not increased: dev Chat is 25117 lines/760
methods and the feature adds 21 thin-view lines/3 methods; Library is unchanged
at 34981/1318. Historical diagnostic labels are absent in pinned dev as well.
These failures are disclosed, not relabeled as passing checks.

Startup UI-ready census passes at 973/973 with zero headroom and its existing
snapshot-drift warning. The normal Requests dependency warning remains. Static
comparison of the root correction files adds no Ruff diagnostics; the two
undefined preview names and undefined close-session variable are removed.
Pre-existing whole-file lint/format debt is not silently reformatted.

Generated diagnostics were reviewed with `--statements --since 3e28c9e7` before
regeneration. Nine changed owners add only fixed text/exception class names;
old exception-repr and traceback/run-ID diagnostics are removed in the affected
webhook/log UI paths. No persistent sink topology changed. CSS reproducibility,
profile-owned path inventory, regenerated diagnostic inventory, unique task IDs
and schema table allowlist checks pass. The index census now covers all 300 indexes, including the worktree scope
index, with its production-query regression passing. Independent webhook
rereview confirms the shutdown exception finding is addressed. Final edited-range
formatting checks pass and no new Ruff diagnostics are introduced in the
correction files; unrelated whole-file debt remains.

## Final integration record

The reviewed correction commit `cd6a725463` was replayed with the84 preceding
feature commits onto dev `a51edff97f`, producing `a51812e5c6`. The rebase was
conflict-free and its tree differs from the verified candidate only by the three
upstream Backlog documentation files; production, tests and generated sources
are identical. The branch contains all current dev commits.

All confirmed local review findings are addressed. Publication, review-thread
replies and remote checks are pending. This record is not a merge claim.


### Windows checkout correction

The first published-head Windows GGUF jobs failed before Python setup in runs
34764839164 and34764839191: two inherited Backlog filenames exceeded the default
checkout path limit. Shortened only the filenames for TASK32540 and TASK32555;
task IDs, titles, statuses and bodies are byte-identical. Their actual Library
work remains To Do. No workflow gate or long-path setting was bypassed.
