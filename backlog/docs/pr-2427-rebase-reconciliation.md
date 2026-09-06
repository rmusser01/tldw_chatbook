# PR 2427 rebase reconciliation — 2026-09-06

Tracked by TASK-31932. This is an in-progress integration record, not a merge-readiness claim.

## Git state

The 176 review commits were replayed onto dev `c4d45c0926580a8756cfa13c5463b1d0fc808c1a`.
Rebased checkpoint: `7bd5b9f4a38bd988db900b6dc5faa885bb56e5e2`.
The original pushed head `0135bc20190bba55fc2b48d3d5863ffe1ea449dc`
is retained on `codex/dev-test-review-backup-0135bc2019`.
The original dirty user checkout and every worktree are preserved.

Conflict resolutions retain upstream Prompts state/controller ownership, Media
path redaction/recovery callouts and speaker-rename cache, named handoff timing,
Canvas terminal transaction contributions, locked promotion publication, and
Buddy listening cleanup. The review's ownership and lifecycle repairs remain.

After committing the reconciliation as `aab86d1b87`, the 177-commit series
rebased cleanly onto the newer dev `c47e0da6002475416240252b2954fcf9761d4aac`,
producing `438f6e9122188c5a92d445715d90ce36cff1b140`. The tree delta from the
preceding checkpoint is exactly dev's six files for Persona Inspector avatar
clearing and boot-worker warning handling. Backlog and diagnostic inventory
checks pass again on this revision. The bounded PR Fast Lane plus those two
newly landed test files is being verified in
`/private/tmp/pr2427-latest-dev-fast-lane.xml` and its matching log.

## Review-only task renumbering

The user-approved policy preserves upstream task identities even where a review
task has the earlier creation date. The final machine census found **31**
collisions (earlier prose counts 33/34 were counting errors). Immediately before
allocation, a NUL-delimited prefix scan covered 1,029 refs and 319 worktrees:
maximum 31900, with 31901–31932 unused. The CLI subsequently created TASK-31932.

Only the following review-created records moved. Creation dates, completed work,
literal XML/log paths and historical collision reports are retained. Mixed
documents were edited by classified reference line, not by global ID replacement.
Earlier checkpoint statements that 31714/31737/31758 remained duplicated are
historical and are superseded by this mapping.

| Former review ID | Current ID | Review task |
| --- | --- | --- |
| 31714 | 31901 | Preserve Loguru capture sinks across app mount |
| 31732 | 31902 | Defer Chunking Lab action imports beyond screen preimport |
| 31737 | 31903 | Close agent swap fixture owned runtime and database resources |
| 31741 | 31904 | Give provider grammar adapter fixture its required assistant owner |
| 31742 | 31905 | Align skill acceptance hook regression with published turn ownership |
| 31743 | 31906 | Separate historical migration assertions from current schema upgrades |
| 31744 | 31907 | Reconcile atomic promotion context policy revision ownership |
| 31745 | 31908 | Forward Console Environment worker scheduling arguments explicitly |
| 31746 | 31909 | Remove inert legacy Notes auto sync timer residue |
| 31747 | 31910 | Restore readable File Notes error text across shipped themes |
| 31748 | 31911 | Restore Skills shadow name coverage for current runtime and Console commands |
| 31749 | 31912 | Move pure Console rewind and settings draft policy to their existing owners |
| 31756 | 31913 | Align unified MCP fixtures with current tool and dispatcher contracts |
| 31758 | 31914 | Watchlists failure policy test bypasses the live check coordinator |
| 31796 | 31915 | Verify current Watchlists source creation off loop |
| 31797 | 31916 | Avoid rewriting committed project context after promotion |
| 31798 | 31917 | Fence combined Console settings live publication |
| 31799 | 31918 | Reconcile reviewed fork transition route inventories |
| 31800 | 31919 | Retain fork ownership through display name persistence |
| 31801 | 31920 | Fence Console conversation binding publication |
| 31808 | 31921 | Reconcile detached and delegated fork census routes |
| 31809 | 31922 | Restore terminal exchange flush and temporary chat completion |
| 31812 | 31923 | Close fixture owned rewind database and controller resources |
| 31813 | 31924 | Separate connection setup from intentional Qwen retry read timeouts |
| 31815 | 31925 | Give real MCP child reap verification a bounded scheduling allowance |
| 31816 | 31926 | Close newly attributed Console controller and hydration fixture handles |
| 31821 | 31927 | Close remaining inventory UI fixture owned database resources |
| 31822 | 31928 | Repair Console Stop clipping after Redirect action was added |
| 31823 | 31929 | Consume character Chat handoffs on cached Console resume |
| 31824 | 31930 | Ignore late screen rebuild notifications after app stack teardown |
| 31825 | 31931 | Classify synthesized leading system rows as rendered system trace provenance |

Upstream TASK-31861's renumbering provenance refers historically to review
TASK-31825, which now resolves to TASK-31931; its own former Canvas ID remains
unchanged. The upstream document is preserved as historical evidence.

## Fresh verification

Initial six-file post-rebase selection: **356 passed, 6 failed**, 2 dependency
warnings, 104.36s. Evidence: `/private/tmp/pr2427-rebase-initial.xml` and matching
log. Failures: Console 17312/16811 line ceiling; Library slack; Media browse
478/371 line ceiling; Conversations controller slack; stale Library assembly
order; Ingest modal presenter inventory. Media behavioral cases passed.

The corrected complete six-file architecture/modal/Media selection has **360
passed, 2 failed**, 2 warnings, 98.69s. Only genuine Console 17312/16811 and Media
browse 478/371 size failures remain. Library and Conversations budgets were
tightened to measured reductions; neither failing ceiling was raised. The
assembly assertion now covers dev's Prompts construction order, and the Ingest
modal edge points to its actual controller. Evidence:
`/private/tmp/pr2427-rebase-inventory-qualified.xml` and matching log.

The complete Canvas, dispatch recovery, roleplay, dictation, and character
navigation selection has **138 passed**, 6 warnings, 135.88s. The first run's
40 missing-html5lib failures were isolated-environment setup: the declared
dependency is now installed in that temporary environment, without dependency
or original-checkout changes. The new dictation regression calls the existing
dictation owner instead of retired private screen delegates. Native attribution
still found seven character-navigation cases retaining fixture-owned SQLite
handles, so passing behavior alone did not qualify resource cleanup. Evidence:
`/private/tmp/pr2427-rebase-behavior-qualified.xml` and matching log.

The complete agent-loop, Prompts state/seam/wiring/characterization, and durable
turn acceptance files have **247 passed**, 3 warnings, 29.28s, with no
`FD_RETAINED` entries. Evidence: `/private/tmp/pr2427-rebase-owner-contracts.xml`
and matching log.

CSS bundle reproduction, profile-owned path census, 113-table allowlist and
281-index decision census pass. Backlog Guard passes across 3,538 records after
the 31 renumbers and new reconciliation task. The diagnostic manifest was
regenerated only after reviewing 49 Console and four Library statements moving
to their existing controller owners: 48 Console and all four Library statements
match exactly; the remaining watchdog warning retains its copy and level with
the owner-injected timeout argument. No diagnostic statements were added to
either screen and sink topology is unchanged. The refreshed inventory verifies
596 owners and 12 sink files.

Duplicate constructor imports have been removed; undefined/redefined-name
checks across affected runtime owners and scoped full lint checks pass. Existing
canonical Library screen re-export imports remain intact. No full-repository
sweep was requested or run.

The new character-navigation file now opts into the same exact-owner real-app
fixture adapter as adjacent reuse tests. Its three constructors use the existing
builder; no behavior assertions or shared fixture internals changed. The complete
file and shared cleanup fault controls have **20 passed**, 3 warnings, 51.53s,
with zero `FD_RETAINED` entries.
Evidence: `/private/tmp/pr2427-rebase-character-resources.xml` and matching log.

## Integration gates

Do not merge until required checks pass and Qodo has reviewed the final revision.
Absence of Qodo comments and the draft-skipped CodeRabbit check are not review
approval. A thread heartbeat watches PR 2427; pause it after confirmed normal merge.

The Console Canvas/citation ownership cleanup still awaits the user's design
approval. Media browse remains 107 lines over its unchanged ceiling; its recovery
and independent page/facet fences must be preserved in any separately reviewed
ownership reduction. Publication is a progress/review checkpoint, not permission
to merge with these failures.
