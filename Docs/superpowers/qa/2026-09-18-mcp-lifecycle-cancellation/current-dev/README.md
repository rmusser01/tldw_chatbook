# PR2713 current-dev integration — TASK-32880

Resumes saved TASK-32829 on dev `ab57681864`. The old number conflicts with an
unrelated landed task, which remains byte-identical to dev. This task is now
TASK-32880 and stays In Progress until review/CI and owner visual approval.

Cancellation retains the original worker through cleanup and final readiness
collection; repeated or retired Cancel cannot stop a retry. Service work starts
lazily, and eager native completion releases admission. Existing ADR-161 applies;
no new ADR, transport, permission, persistence or CSS contract is introduced.

The [one conflict](integration.json) keeps both current recovery-token checks
around inspector readiness and the saved captured operation/cancelling arguments.
Current selected-tool refresh is retained. [Visual gallery](GALLERY.md).

## Current verification

96 distinct targeted cases pass: 72 component cases plus 24 runner/input checks.
The component inventory is 11 lifecycle regressions, 55 adjacent lifecycle,
inspector refresh, restored-root and design checks, and six navigation-interruption
cases covering the conflict seam. The held-render case passes again after its
harness cleanup fix; repeated cases are not counted twice. No full suite ran.

The initial current-dev baseline produced nine assertion failures, one pass, and
one deliberately interrupted held-render case. Independent review identified the
harness deadlock; after bounded cleanup, that same case fails cleanly on unchanged
dev and passes on the integrated implementation. See the separate red/green logs.

The post-Qodo run has 35 passes: eleven repeated lifecycle cases plus three
real-store fixture cases, seven runner CLI cases and fourteen shared input cases.
The collision regression first failed on the unguarded save and then passed with
the original store bytes and runtime state preserved. [Review follow-up](qodo-followup.md).

All seven preflight guards pass. A sandboxed download failed for the pinned
Mermaid input; the normal guard succeeded with authorized network access. New test/runner lint and format pass; both large
production modules retain the same 78 Ruff diagnostics, with none introduced.
[Independent review](independent-review.json) has no remaining findings.

## Native evidence and limits

Four real TldwCli/LinuxDriver/TTY cells at 80x24 and 170x48 in dark/light pass.
Twelve captures were inspected: Cancel, disabled Cancelling and Refresh tools
remain readable and reachable for this bounded journey. Only the client's
connection is replaced by controlled held cleanup/failure; actual control-plane,
profile store, governance, cancellation persistence and UI remain in use.
No network attempts occurred. External successful transport and execution remain
unqualified; the compact detail toolbar has the separate PR2712 clipping boundary.

[Lifecycle](native-lifecycle.json) verifies the post-Qodo native run with normal App.run return/exit 0, absent
PID, released/reacquired lock, ten healthy private databases, zero conversations
and messages, unchanged user defaults, no app errors, and exact source/runner
hashes. Original bytes remain in the private native root; the export manifest
records whitespace-only normalization. A first runner attempt inspected the
lazy workbench module before navigation and exited before App.run; its failure
is preserved separately and contributes no visual qualification.

Current-head CI, accumulated remote review, current dev/conflict review and owner
approval of this fresh gallery remain before merge. Older evidence one directory
up is historical, at the saved PR head rather than this integrated source.

The repeated native journey matches the final runner hash and all unchanged
production hashes. A preparation attempt omitted a required private database
directory and was rejected by the shared parser before app imports; its receipt
is preserved. The twelve fresh captures were inspected again.
