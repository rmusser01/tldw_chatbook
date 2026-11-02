# PR3337 documentation integration plan

> **For agentic workers:** Use superpowers:subagent-driven-development for this coupled documentation integration, then independent superpowers:requesting-code-review. Existing backlog execution authorization remains in force.

**Goal:** Integrate the historical run-start measurements with accurate ranking, explicit ADR disposition and evidence limits.

**Architecture:** Documentation only. Preserve the original measured results and open follow-ups; distinguish historical reported evidence from verification performed during integration.

**Tech stack:** Markdown, Backlog CLI and Git. No application execution or dependency changes.

**Spec:** TASK33620.15.21 and its existing follow-ups .38/.47/.48; .superpowers/preparations/pr3337/repair-plan.md and pr3337-integration-preflight.md; source review at C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3337-initial-review.md.

**ADR required:** no. **ADR path:** N/A. **Reason:** Measurement documentation corrections change no runtime, storage, ownership or service boundary.

## Global constraints

- Selected live source6c9ac6b83bf8dd22820e374c39eb3376d0417903 on perf/task-33620.15.21-run-start-high-load, original current base42eab3c3c7770d2003688dfe2fa576b328ba2d90. Actual integration base15f786ba7a09f92c13d079840e1729b540ea32c8. Existing isolated branch codex/pr3337-resync starts at selected source; rebase first and retain every contributor change and both sides of accumulated lessons conflicts.
- Root owns remote mutations. Implementer alone owns source/index and scoped document verification; no subagents or new worktrees, stash, dependency changes, application imports, pytest, generators, benchmarks or paid-provider runs. Main checkout and interpreter remain read-only. No drafts,3267/3297, releases3245/3246 or main/release work.
- User authorizes hosted-CI and numeric-size waivers plus independent requesting-code-review instead of Qodo. No correctness/security/source/custody guard or numeric cap change is authorized here. No new production code belongs to this PR.
- The measured 4.55ms prompt-history first-send median is based on5 high-load first sends; its wider subset is5+15 high and2+6 low. Other headline medians cover the wider sample. Preserve the below10ms conclusion, original arithmetic and qualification of pooled later sends.
- Historical72-send raw/probe archive tools/evidence/bd5/M21/ was absent in preparation. Check actual selected contents; if still unavailable, explicitly retain reported/unverified status and do not claim independent recalculation or fresh preflight results. Preserve original historical evidence rather than erase or invent it.
- .38/.47/.48 remain open and their acceptance criteria unchanged. Record explicit ADR no/N/A/reason in the completed measurement task's plan and notes using installed Backlog CLI for task edits.

## Review focus

1. Headline ranking includes the smaller prompt-history subset without implying identical sample sizes.
2. Missing raw archive is disclosed; reported historical experiments do not become current performance certification.
3. Rebase preserves the accumulated live-verification lessons and latest concurrent source changes.
4. Follow-up task IDs, links, open statuses and outcome criteria remain intact.
5. Final base-to-head changes are documentation only and whitespace-clean; no runtime or guard alteration is hidden.

### Task 1: Rebase and qualify the measurement documentation

**Files:** Existing five-file source delta: backlog/docs/lessons-live-verification.md and task-33620.15.21/.38/.47/.48 Markdown files. Commit this canonical plan as well.

**Interfaces:** No executable interface changes. Requirements and exact wording/data come from the selected task, source review and global constraints above.

- [ ] Read current AGENTS and relevant task/ADR/testing/live/backlog lessons, source review and preparation. Rebase onto15f786ba7a09f92c13d079840e1729b540ea32c8 without dropping contributor work; record conflicts and replay count.
- [ ] Verify both prior minor findings still apply. Correct the headline ranking with4.55ms/5high-first sample qualifier while preserving other reported numbers and the below10ms conclusion. Add explicit ADR disposition to task plan and notes; do not implement any follow-up feature.
- [ ] Check actual archive availability. If absent, add a concise integration note making historical reported results and inability to independently recalculate explicit. Retain original measurement scope/status and open .38/.47/.48.
- [ ] Verify affected task ID uniqueness, referenced task/link existence, documented sample arithmetic and documentation-only scope. Run full git diff --check BASE..FINAL. No tests or application imports are warranted for prose-only corrections.
- [ ] Self-review and commit a clean head. Provide task-1-report.md/final-qa.md with exact refs, commands/outcomes, scope and limits plus whole-branch patch and hashes explicitly distinguishing Git blobs from raw checkout bytes. Release ownership for independent whole-branch review; root performs fresh eligibility, exact-lease publication and merge.
