# Agent orchestration final integration repairs

ADR required: no new ADR
ADR path: backlog/decisions/154-agent-denial-streak-boundary.md; backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/156-live-per-run-stream-usage-attribution.md
Reason: restore established continuation and optional-observation behavior, align tests with the accepted interim refusal, and update the required diagnostic artifact. No new authority, storage, provider or runtime boundary is introduced.

Review basis: final whole-branch review at 564e5f27db, covering d66908a69 through 564e5f27db. Exact findings and 50-ruling triage are retained in .superpowers/sdd/2026-09-12-agent-orchestration-remaining/scratch/final-review-1.md. TASK18929 AC9 and TASK18923 AC6 are In Progress before source repair. TASK31210/31211 functional criteria and parent TASK13154 remain open.

## One combined correction wave

- [x] Restore the ordinary non-denial coherent continuation boundary by advancing the extra history capture only when the denial breaker actually terminates. Preserve fully settled denied batches and cancellation precedence. Keep the unchanged loop-top regression and qualify adjacent cancellation/native/fence history cases.
- [x] Replace the two obsolete worktree-positive continuation tests with active refusal tests using actual coordinator-retained isolated history. Prove no provider/Git/shared-tree fallback, retained original history, exact failed row/handle behavior, and owned harness settlement. Keep ordinary continuation/tool-containment coverage; no skips or unsafe admission.
- [x] Contain per-chunk optional usage snapshot/count extraction with a bounded content-free diagnostic and disable repeated failed observation for that call. Exercise attributed actual adapter text streams for snapshot/extractor faults, full successful text and finished/cleared state; genuine provider errors and final accounting remain meaningful.
- [x] Review every resulting diagnostic statement against the last inventory revision with the supported interpreter. Regenerate only the required diagnostic inventory and confirm its gate passes; no privacy or ratchet bypass.
- [x] Run the affected continuation module after the repairs and focused neighboring runtime/adapter tests appropriate to the amended paths. Reuse unchanged successful gates. Review scoped static deltas and edited-line formatting; no full suite or dependency cleanup.
- [x] Root commits the single correction wave and requests one independent scoped re-review against the previous reviewed HEAD. Record all residual decisions and actual task state; do not claim functional worktree recovery or parent closure.

## Ownership and evidence

Work exclusively in the preserved isolated orchestration worktree, branch codex/agent-orchestration-remaining. One fresh implementer owns source/tests and the generated diagnostic artifact. Root owns Git, Backlog, plans and status notes. No worker subagents, commits or staging.

Every pytest invocation uses python3 .superpowers/sdd/2026-09-12-agent-orchestration-remaining/run_pytest.py UNIQUE-LABEL Tests/path.py::node (or the explicitly affected continuation module). This supplies the existing isolated interpreter and a unique owned basetemp. Temporary product probes must be under Tests and exact-node selected. Preserve command/stdout/stderr/exit and all historical failures. Do not import product modules standalone or read user config, install dependencies, call providers/network, manipulate foreign resources, raise guards, or delete workspaces.

Diagnostic inspection/regeneration is an AST-only script and must use the existing supported interpreter with -I; host python3 is 3.9 and cannot parse current source. The initial failed 3.9 trace and the supported inventory drift are distinct retained evidence. Root already reviewed two fixed-string additions, with no interpolated data or new sink; final warning changes require the same statement review before --write.

Completed in 7852cf47ba after targeted verification and the one independent scoped re-review. All four findings were addressed with no new Critical/Important breakage. Current functional worktree and parent limitations remain as stated above. Final record: Docs/superpowers/reviews/2026-09-12-agent-orchestration-remaining.md.
