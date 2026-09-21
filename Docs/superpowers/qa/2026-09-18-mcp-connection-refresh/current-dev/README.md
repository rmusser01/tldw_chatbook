# PR2714 current-dev catalog refresh qualification

## Approved merge closeout — 2026-09-20

PR2714 merged into `dev` as `e4096e2059` after the owner-approved visual review.
The final conflict-free rebase retained all six patches; 41 affected local cases,
all eight current artifact checks, and all thirteen native source/runner hashes
pass. GitHub run35542566467 passes 1,152 Fast Lane cases and the required artifact
gate on the exact merged tree. Qodo reports zero open bugs/rules; all nine review
threads are resolved or dismissed. TASK-32830 is Done.

PR2716 tool-result errors now resumes on a fresh branch from merged dev. Compact
notification placement, catalog reachability and the other pending screen reviews
remain separate scope. [Merge receipt](merge-closeout.json).

Current evidence is native run 004 after the second Qodo pass; see that section
below. Earlier qualification paragraphs retain their historical context.

TASK-32830 resumes from merged PR2713 dev `5e0f9f82c3`. The saved service
patch applies without conflicts and changes only `local_control_service.py`.
No screen, widget, CSS, token, transport API, schema or permission boundary changes.
Existing ADR-111 and ADR-161 apply; no new ADR.

Connected refresh now reconnects and discovers current tools, resources and
prompts. Disconnected refresh restores its disconnected state. Observe and launch
checks precede replacement; failures clean only the established session owned by
the call. Busy or denied refresh leaves another pending connection untouched.

## Verification

- [Current-dev red](red-current-dev.txt): four expected failures reproduce stale
  catalog, missing launch check, missed failed refresh, and temporary-session leak.
  Five preservation/compatibility cases already pass. An unrelated pytest shared
  temp cleanup warning is retained; green runs use dedicated temporary roots.
- [15 service cases](targeted-service.txt) pass, including nine real stdio cases
  and six neighbors. [36 QA/service cases](targeted-qa.txt) pass (nine repeated).
  [Final 17 runner cases](targeted-runner-final.txt) pass (ten repeated, seven
  additional adjacent inspector-runner inputs). Total: **49 distinct local cases**.
- [Seven artifact guards](preflight.txt) pass. [Static comparison](static.json):
  no introduced Ruff diagnostics; five existing service diagnostics remain.
  Modified/new test and runner files pass lint; four edited files pass formatting.
- [Independent review](independent-review.json) cleared the service. Its runner
  finding was fixed by placing fixture files in the exclusive evidence directory.
  Existing root-level symlinks and their unrelated sentinel targets remain intact.
- [Integration receipt](integration.json) records the saved head and current base.
  No duplicate TASK-32830 path was found across all Git refs; artifact guard checks
  4,184 current task files. No full repository test sweep was run.

## Native and visual qualification

[Supported runner](native_check.py) uses validated CLI arguments before app imports,
a fresh private profile, real TldwCli/LinuxDriver/TTY, real governance, store,
client and local stdio subprocesses. The fixture changes catalog versions and
rejects initialization for the failure phase. No client/service substitution,
external network or tool execution is used; network attempts are blocked and zero.

[Four passed cells](native-result.json): dark/light at 80×24 and 170×48. Each
connects, discovers changed catalog, fails refresh while retaining saved discovery,
recovers through Refresh tools, then reconnects and disconnects. [Wire trace](fixture-trace.jsonl)
contains twenty initializations and sixteen complete three-section discoveries.
[Lifecycle receipt](native-lifecycle.json) confirms all twenty fixture PIDs and
app absent, exit 0, App.run returned, released lock, ten healthy private databases,
zero conversations/messages, unchanged default config/UI/policy files, preserved
fixture-file sentinels, no errors/faulthandler output, and matching source/runner hashes.

All sixteen settled and sixteen feedback SVGs were rendered and inspected.
Wide views show original → updated → recovered catalog names. Settled compact
views show Refresh tools after failure and Connect after disconnected recovery.
**Immediate compact notifications temporarily cover inspector actions**; feedback
captures preserve this limitation explicitly. Current workbench notification and
layout code is unchanged. Compact toolbar clipping (PR2712), this notification
placement, and below-fold catalog scrolling remain follow-up UI review items.
Cairo glyph/font fallbacks are not interpreted as terminal defects; original SVGs
and paired terminal text remain authoritative. [Inspection receipt](visual-inspection.json).

The [superseded first run](superseded-native-run.json) was interrupted for the
runner ownership fix; its app and all owned children exited. It does not qualify
the final runner. [Export hashes](export-manifest.json) preserve raw provenance
through trailing-whitespace normalization. Historical evidence remains one level up;
the obsolete original runner is linked at its immutable saved commit.

Owner approved the original 32-capture gallery at `f8d1731abc` (see
[approval receipt](owner-approval.json)); it remains preserved unchanged.
Current-head CI and accumulated review remain merge gates.


## Qodo follow-up on current dev

Rebased without conflicts onto dev `7bfd330046`. No intervening MCP, workbench or
CSS changes. Qodo raised one ownership-race claim and three rule findings:

- Fixture state/trace now pass the central path validator beneath an explicit
  trusted root before any file I/O. Six malicious-path cases fail on the original
  fixture ([red evidence](qodo-path-red.txt)); all eight path cases pass after the
  repair, including valid relative/absolute paths.
- Both public service methods now document permission checks, discovery, saved
  catalog behavior, temporary cleanup and failures. Production executable code
  is unchanged by these follow-ups.
- Ten isolated service cases complement ten real stdio cases. The latter include
  a real connection queued during snapshot save: temporary cleanup reaps its own
  process and leaves the replacement alive. The production client does not yield
  between publishing its session and teardown capturing that identity; the
  reported interleaving would require a different, yielding describe method.
  Independent review confirmed no alternate client or off-loop mutation path.
- [139 focused cases](qodo-targeted.txt) pass, including the affected native-runner
  argument/ownership tests. [Seven preflight guards](qodo-preflight.txt) pass;
  [no introduced Ruff diagnostics](qodo-static.json). No full test sweep.

Fresh native run 003 passed all four cells and reaped twenty fixture children and
the app, with unchanged defaults and sentinel targets, healthy private databases,
released lock, no logged errors and zero network attempts. Run 003 exports were retained at commit `5e3749a998`; approved run 002 remains in commit `f8d1731abc` and the
unchanged owner gallery. All 32 new captures were rendered and inspected. The
fixture's explicit root adds command text (and one row in wide catalog placement);
notification carryover may vary. Product controls, state and styles were unchanged in that first pass,
including the disclosed compact toast overlap. Four captures render identically;
[comparison hashes](qodo-render-comparison.json) retain the full comparison.

During this run independent review corrected only the connect docstring's error
description. The [source receipt](native-doc-only-change.json) verifies identical
executable AST; the lifecycle receipt distinguishes this from byte-identical
source. That run is historical; the second-pass qualification below uses exact source hashes.


## Second Qodo pass: canonical ownership and incomplete cleanup

Qodo dismissed the original race and unit-coverage claims after reviewing the
actual client and isolated tests. Its next pass found two valid boundary bugs:
accepted whitespace-padded IDs looked up the wrong session key, and failed
temporary cleanup could return a successful refresh while the owner stayed live.

The service now uses the stored canonical ID for connection-state and ownership
checks. After temporary cleanup it raises only if that same session remains
registered; a replacement is preserved and cancellation still propagates. The
fresh snapshot stays saved, while the real control plane records the cleanup
outcome as `ok=False`. A later real disconnect reaps the retained process.
This keeps existing public APIs, permissions, UI controls and styles.

The fixture now checks input size through the shared validator and uses private
strict Pydantic request/initialize models. Malformed JSON, request shape or
initialize parameters produce bounded JSON-RPC errors, and the next valid request
still succeeds. Fixture-only schemas remain in test code rather than creating a
new production protocol API. The shared policy helper has Args/Raises docs.

[Four boundary regressions](round2-red.txt) and [six further cases](round2-red-extra.txt)
fail before repair. [41 focused cases](round2-green.txt) pass after repair, with
the updated real control-plane outcome assertion separately [verified](round2-cleanup-record.txt).
[Six unchanged-service/transport neighbors](round2-neighbors.txt) pass: **47
distinct affected cases**. Earlier native-runner argument/ownership coverage
remains recorded in the 139-case first-pass run; those runner paths are unchanged.
[Seven guards](round2-preflight.txt), [unchanged lint baseline](round2-static.json)
and independent review pass. No full test sweep.

Final native run 004 qualifies the repaired executable sources with exact hashes
(no documentation-only exception). It repeats all four dark/light compact/wide
journeys, 32 captures, twenty child-process exits, normal app shutdown, lock release,
healthy private databases, unchanged default files/symlink sentinels and zero
network attempts. The native journey covers normal refresh/failure/retry;
the failed-cleanup boundary is qualified by the controlled real-client test, not
by claiming that a real OS kill failure was induced during the visual walk.
The owner-approved product controls/layout remain unchanged; the existing compact
toast overlap and separate toolbar/catalog-scroll work remain disclosed.

Current-head CI, accumulated review and the final dev/conflict check still gate merge.
