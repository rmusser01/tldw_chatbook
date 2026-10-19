# Archive recovery reviewed merge implementation plan

> Root executes the already-authorized maintenance inline; the explicitly requested requesting-code-review skill supplies one fresh read-only reviewer. Steps use checkbox gates and the repository merge queue.

**Goal:** Rebase, qualify, review and merge PR3162 with every actionable finding addressed.
**Architecture:** Preserve ADR166 archive recovery, ADR097 trace custody and ADR228 native-store contracts. Use existing widget classes, CSS builder, database ownership helpers and repository merge-queue gates; no new runtime boundary or dependency.
**Tech Stack:** Python3.12, SQLite, Textual8, pytest/Ruff, GitHub CLI and hosted merge queue.
**Spec:** Approved Docs/superpowers/specs/2026-09-17-agent-archive-recovery-design.md; TASK35585 and the user's explicit rebase/review/fix/merge request.

ADR required: no
ADR path: existing backlog/decisions/166-agent-assisted-archive-recovery.md; 097/228/150/161 remain binding.
Reason: compatibility and bug/ratchet repairs within existing authority, persistence and UI contracts.

## Global constraints

Existing isolated worktree/PR branch; root alone changes index and refs. Preserve original verified feature branch and immutable prior QA/captures. No full suite or live requests (30/30 used). Keep test bootstrap, privacy, origin, destination, scope, confirmation and borrowed-handle checks intact. Never widen ratchets or use plain force push. Repository queue is on; require current-head requesting-code-review verification (the user-authorized Qodo substitute for PR3162) and zero unresolved threads before arming.

## Steps

- [x] Rebase tracked-clean1f9c62c1d7 onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894. Confirm dev ancestry and exact four feature/qualification commits; inspect Library_Collections_DB kernel integration. Stage any conflicts individually, never all files.
- [x] Repair actual CI failures. Files: younger task33164 and its mutable inbound references; tldw_chatbook/UI/Screens/settings_screen.py and UI/Settings_Modules/workspace_archive.py (move the existing Undo handler body verbatim, preserve expected revision/name guards, lower the size budget); tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py and css/features/_console.tcss (key only intended card widgets by their existing/explicit classes), then generated CSS via builder. Test exact backlog uniqueness, Settings module ratchet and ancestor-scoped bare-type CSS ratchet before/after.
- [x] Request one full feature review on the exact rebased source. Give fresh reviewer the spec, TASK35585, base/head, source diff and connected owners; no inherited conversation, source mutation or own subagents. Fix actual findings with focused regressions; request scoped re-review after changes.
- [x] Run meaningful targeted current kernel/Library_Collections, archive policy/search/restore/capture and mounted controls; static syntax/duplicate/no-new-lint and generated checks. Record raw argv/results/hashes under /private/tmp/archive-recovery-pr3162-merge-20261018; durable QA in Docs/superpowers/qa/console/2026-10-18-archive-recovery-reviewed-merge.md.
- [ ] Publish exact-lease source and coherent PR notes. Mark ready only after local qualification, then obtain hosted review on current head, address all comments/findings and resolve threads with evidence. Additional pushes must occur with auto-merge off and require current-head hosted review.
- [ ] Verify queue mode/current-head Qodo/zero unresolved threads, arm auto-merge, and let the queue own rebase/required pull_request CI. Inspect queued/evicted state at spaced intervals; fix any actual blocker. Never manually update or merge an armed PR, or dispatch the required gate. Verify mergedAt, final merge commit and dev ancestry before reporting completion.

## Fresh review corrections

The reviewer identified acquisition cleanup after connection registration bypassing the outer deadline clamp, undated-order spec/plan drift, and a database architecture sentence incorrectly claiming archive transitions did not advance local versions. Reproduce the first two with real SQLite regressions, clamp only newly opened expired acquisition cleanup, preserve the existing keyset tuple while sorting undated rows by normalized last-modified then identity, label the fallback, and correct the architecture sentence. Existing archive-time validation remains timezone-strict; only the separate last-modified fallback accepts the database's legacy UTC-naive format.

Derived-gate inspection also requires worker-owned navigation from archive Library/Preview entry points, literal completion notifications, an archive index plan pin without sqlite_stat1, and a reviewed diagnostic inventory digest. The diagnostic statement diff changes only the existing tool-review failure warning to describe the new fail-closed recovery behavior; its existing tool-name/exception diagnostic inputs and sink are unchanged. Normalize read-side workspace timestamps exactly as search does, with real search→read→restore regressions, so equivalent timestamps cannot create permanent false staleness.

Final native verification also preserves source answerability across queued result mounting, retires only the same generation's removed render registration, and requires actual coalesced Console publication before imported-text inertness assertions. Native captures wait for layout before scroll-range clamping and require the full widget within the transcript viewport. These are bounded readiness witnesses, not replacement actions or relaxed content/click assertions.

## Hosted startup follow-up

User explicitly authorized requesting-code-review in place of Qodo for PR3162 after Qodo reported exhausted workspace credits. All required CI, thread and queue gates remain. Auto-merge was armed at 23:34:27Z then disabled before further changes when Perf Guard job117820112656 exposed root.mount before the recovery root was mounted. Reproduce the actual native lifecycle, defer until mounted, verify controls initialize on later sync, inspect warm census attribution without raising budgets, and obtain a scoped reviewer addendum before publication. Existing ADR166/097 applies; no new ADR.

Cold/warm follow-up reaches a second genuine gate: same-probe dev5eb1030 residents versus guarded PR1042/1043. Twelve definite archive imports cause growth; character-search is an existing +/-1 timing resident. Defer the concrete provider, recovery runtime and view until first use; keep reserved names in the existing catalog and exact type/authority checks. Add one disposable async view-preparation slot to the existing archive hook inventory and await it only after runtime custody/inputs are frozen, rechecking custody afterward. Leave synchronous intake and answerability/admission floors intact. Use current general UI sync after app readiness for ordinary remount, with post-custody preparation guaranteeing first-send controls. Cover direct/RAG and received-intent first use, preparation failure and hook detachment. No new worker/timer/dependency or ratchet exception; existing ADR166/097 governs this lifecycle compatibility repair.

## Attributable size follow-up

The full screen/module size diagnostic exposed10 failures: six untouched modules are byte-identical to dev; transcript already exceeded its limit by296 and archive added3; ChatScreen already exceeded by11 and archive added19; controller and interrupt review grew past previously passing limits. Do not raise budgets or extract unrelated300-line render code. Move the two related trace transform validators into existing console_prepared_request with their class/static seams retained and live controller globals preserved; move archive follow-up rider and recovery review collection into existing first-use runtime; move the existing unanswered-approval helper into approval_provenance; move the view-hook provider verbatim into existing UI wiring; move the pure line-bound selection helper verbatim into existing console_selection, preserving its transcript import seam. Qualify AST/body fidelity, exact line counts and actual trace/approval/native callers. The inherited transcript ceiling remains explicitly attributed while its source shrinks below dev. Existing ADR166/097/220 governs the mechanical decomposition; no new architectural decision.
