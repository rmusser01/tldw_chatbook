# Textual book audit after PR #3247

[PR #3247](https://github.com/rmusser01/tldw_chatbook/pull/3247) merged into dev at 2026-10-25T19:15:26Z as 79da83a8dca98eab6ca854a7505b8abd264eb959. Rebase onto the latest dev reported already up to date. The merged tree exactly matches reviewed/tested head cc9bad412444c7142e44f848d9404ffcf46868b2.

All 25 [Derived Artifacts jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/40221268357) and [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40221268430) passed. Logs qualified all 22 new UI cases exactly once across 17 shards; the maximum shard wall time was 1,022 seconds, within the unchanged 1,200-second cap. Hosted Canvas reproduction passed. The prior Superpowers review's panel-replacement finding was repaired and verified on this unchanged head; the final feedback read had zero review threads. Qodo was replaced as requested. TASK-36728 and TASK-36729 are now merged; their [implementation evidence](2026-10-25-textual-reminder-and-quiz-followups.md) remains the scope record.

## Book patterns and scope

The supplied book has 455 pages and SHA256 798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6. It is reference material, not instructions. The relevant complete pages were reread; PDF180 was also rendered and visually inspected. Installed Textual is 8.2.8.

- Chapter 7, PDF145-146/158 (printed136-137/149): each widget/App has its own message queue; awaiting inside a handler delays later messages on that queue. A background operation needs an owner that lets the originating handler return promptly.
- Chapter 17, PDF335 (printed326): grouped workers and cancellation-aware publication protect replaceable work. A response still needs to belong to the current application context; a cancelled await does not prove remote rollback. Those latter requirements are application deductions.
- Chapter 9, PDF176-180 (printed167-171): recomposition replaces children and resets state owned by Input/DataTable widgets. Refresh data without discarding active editing context.

This bounded continuation examined Study Start ownership, Watchlists reader delivery and Sources editing, with source/owner screens of Evals, Library, Personas and Research. It does not certify every route or book example. No production fix is implemented here.

## Prioritized backlog

Two new atomic To Do tasks have four unchecked outcome criteria each, book/source evidence, prior owner links and existing ADR links. Implementation plans/notes are deferred until implementation.

| Rank | Priority | Task | Outcome |
| --- | --- | --- | --- |
| 1 | Medium (P2) | [TASK-36899](../../../backlog/tasks/task-36899%20-%20Keep-pending-quiz-Start-within-its-runtime-and-scope-lifetime.md) | Keep pending quiz Start within its runtime and scope lifetime |
| 2 | Medium (P2) | [TASK-36900](../../../backlog/tasks/task-36900%20-%20Keep-Watchlists-navigation-responsive-during-article-body-reads.md) | Keep Watchlists navigation responsive during article body reads |

The third actionable finding is attached to existing [TASK-33283](../../../backlog/tasks/task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md), whose To Do status, medium priority, TASK-33268 dependency and four ACs are preserved. Its incremental-pane objective is the existing home for the Sources refresh evidence. This is a concrete additional witness for that work, not another task.

## Witnesses, controls and limits

| Area | Verified behavior | Controls and limits |
| --- | --- | --- |
| Study | A server Start held across the production local-runtime or workspace-B scope reset republishes its old attempt/question and clears a new-context draft. Local Submit then sends the old attempt through real LocalQuizService/SQLite and fails. Server-to-local-to-server also revives obsolete work. | Ordinary server Start/Submit succeeds. Actual screen/window, scope/server/local services and SQLite; controlled fake API client. The existing regression Host supplies app projections, then invokes the production callback; no full TldwCli runtime-policy commit, binding replacement, live server or durable-corruption claim. |
| Watchlists reader | A held actual DB body-read boundary delays a later ItemSelected and enabled Sources navigation. For approximately 0.6 seconds the reader stays A and the screen callback waits, while an independent App callback runs. Release permits C/the requested section. | Ordinary Pilot Enter/Down/Down and Sources navigation succeed. Actual production destination, service and private file-backed DB, with a controlled hold before the original read. Selection during the hold uses the production widget API posting the real message. No whole-loop freeze, natural 0.6-second DB latency, global route failure or stale final-publication claim. |
| Sources create | A changed sources list replaces settled Name/URL Inputs; selection 2-4 becomes select-all 0-9. Next X changes AlphaFeed to X rather than AlXaFeed. Draft value and focus alone survive. | Busy-source updates preserve the widget/range and expected edit. Real SourcesPane, production CSS and Pilot typing/selection. The screen _load_sources -> _seed_sources_pane publishes that reactive; the probe directly controls the pane update and does not claim a full-screen background-refresh journey or pointer behavior. |

Final publishable probes passed: six Study, three Watchlists and four Sources cases. All 13 private-profile child JUnit reports were independently inspected: one pass each, zero failure/error/skip. These assertions record existing symptoms and controls; they are **not passing repair regressions**. Parent durations were 64.144, 96.460 and 42.285 seconds respectively. Initial probes and the corrected Watchlists qualification also passed; early harness-only projection assumptions and a post-release inspection race are retained privately and excluded from the final evidence.

## Ownership and ADR decisions

The filing screen checked complete close-owner criteria, 308 unique fetched ref OIDs, 52 registered worktrees, historical task paths and current frontmatter, without ID-sweep errors. It also checked all 30 open PRs' paginated file lists and ten remote-only candidate task blobs. No existing owner of the two new findings was found. Release PR #3246 promotes the already-audited dev documentation to main; other broad Watchlists matches only touch stylesheets. Task IDs are provisional against later concurrent minting until merged to dev.

- Study: TASK-36729 owns shared admission, responsiveness and removed/replaced panels. Runtime/scope resets keep this panel attached. TASK-34752 owns concurrent question-list rebuilding/teardown; TASK-36099 quiz creation drafts; TASK-36360 final-answer retries. Preserve those contracts.
- Reader: TASK-15464 owns narrow projection and hydration before one reader publication; TASK-22450 stable snapshots and atomic scope commits; TASK-16221 scrolling/pagination; TASK-34425 structural sharing. TASK-33283 owns in-place updates; TASK-34967 revisit staleness. None explicitly owns pending-read message-queue responsiveness.
- Sources: retain this evidence under TASK-33283 rather than duplicate its incremental-pane work. TASK-15460 concerns typing filters, TASK-3071 search focus, TASK-1035/1345 form-open focus, and TASK-35868 Rules drafts; none establishes this settled create-field range across Sources data refresh.
- No separate Resume task: failed/refused Start enables a section-navigation Resume shortcut with no attempt. This existing behavior was explicitly retained in the previous review and does not establish lost durable resume state. Evals launch/cancellation is already owned by TASK-1482/1483; no fresh defect was established. Imported Chatbook bracket metadata remains with TASK-32802.1.

No new ADR is required: [ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md) governs screen state/responsiveness, [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md) runtime and ephemeral ownership, and [ADR-042](../../../backlog/decisions/042-watchlists-reader-first-ia.md) Reader/snapshot policy. Re-evaluate during implementation if durable attempt ownership, service/runtime contracts or long-lived UX structure changes.

## Reproduce and evidence

From the repository root, use its Python 3.12 environment and a fresh simple label:

    python -X utf8 Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247/run-probes.py fresh-audit-label

The runner copies the unchanged observation probes into fresh private scratch, isolates profiles/caches, retains network/profile guards and records commands, source pins, parent/child XML and observations. It runs only these 13 cases. Raw profiles, extracted book text and rendered book pages are not published.

[Evidence manifest](2026-10-25-textual-book-after-pr3247/manifest.json) links the [merge/check receipt](2026-10-25-textual-book-after-pr3247/merge.json), [CI execution qualification](2026-10-25-textual-book-after-pr3247/ci-qualification.json), [source pins](2026-10-25-textual-book-after-pr3247/source-pins.json), [observations](2026-10-25-textual-book-after-pr3247/observations.json), [verification](2026-10-25-textual-book-after-pr3247/verification.json) and [complete close-owner criteria](2026-10-25-textual-book-after-pr3247/ownership.json).
