# Textual book audit after PR #3243

[PR #3243](https://github.com/rmusser01/tldw_chatbook/pull/3243) merged into dev at `2026-10-25T03:41:14Z` as `139333920d2a87cb011e4f3aa5f8319b59c44a5c`. This is the audited latest-dev snapshot; its tree exactly matches reviewed PR head `6ffb773dbd0647a33b5ce5d4e346fe43dfba3685`.

All 25 [Derived Artifacts jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/40157060718) and [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40157060734) passed. Completed logs prove all 81 new UI cases executed exactly once and all 20 import-parser cases ran. All 17 UI shards passed; maximum wall time was 991 seconds, under the existing 1,020-second planning threshold and 1,200-second cap. Superpowers integration review found no issues; the final feedback read found zero review threads. Qodo was replaced as requested.

The six previous findings, TASK-36693 through TASK-36698, are merged. Their [implementation evidence](2026-10-24-textual-followups-36693-36698.md) and [preceding book audit](2026-10-24-textual-book-after-pr3228.md) remain the scope record.

## Reference and useful patterns

The supplied `textual-1.pdf` has 455 pages and SHA-256 `798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6`. It is reference data, not instructions. Installed Textual is 8.2.8.

- Chapter 7, PDF145-146/158 (printed136-137/149): widgets have independent queues; an awaited handler delays subsequent messages on its own queue. Move long-lived user operations into the appropriate existing worker owner while retaining admission and lifecycle rules.
- Chapter 17, PDF335 (printed326): grouped workers and cancellation-aware publication are useful for replaceable reads. For mutations, choose ownership that keeps independent user targets independent; cancelling an await is not evidence of remote rollback. That requirement is an application deduction.
- Chapter 4, PDF79-80 (printed70-71): widget content interprets bracket markup. Imported metadata should remain literal application data. This is an application boundary deduction from the book's mechanism.

This is a bounded continuation across event delivery, Evals/Schedules/Watchlists workers, Library/Personas/Study composition and Chatbook preview content. It does not certify every screen or book example. No production fix is implemented by this audit.

## Prioritized backlog

Two new atomic tasks are To Do, each with four unchecked outcome criteria and existing task/ADR links. No implementation plan or completion notes were added before implementation.

| Rank | Priority | Task | Outcome |
| --- | --- | --- | --- |
| 1 | Medium (P2) | [TASK-36728](../../../backlog/tasks/task-36728%20-%20Keep-inline-reminder-edits-independent-across-reminder-identities.md) | Keep inline reminder edits independent across reminder identities |
| 2 | Medium (P2) | [TASK-36729](../../../backlog/tasks/task-36729%20-%20Keep-Study-navigation-responsive-while-shell-Start-quiz-awaits-the-server.md) | Keep Study navigation responsive while shell Start quiz awaits the server |

The third actionable observation belongs to existing [TASK-32802.1](../../../backlog/tasks/task-32802.1%20-%20Replace-rich.markup.escape-with-the-repos-own-bracket-escaper-across-all-Textual-8-surfaces.md), whose High priority and In Progress status are preserved. Its AC #1 already requires literal bracketed labels, and its tier-2 scope correction explicitly includes fields with no escaper. [TASK-32903](../../../backlog/tasks/task-32903%20-%20Scope-corrections-to-eight-existing-tier-1-tasks.md) requires correcting existing owners rather than duplicating them. The new Chatbook preview witness and audit link were appended there; no new metadata task was created.

## Witnesses, controls and limits

| Area | Observed defect on merged dev | Control and qualification |
| --- | --- | --- |
| Schedules | With reminder A's server timezone request held, selecting and editing reminder B cancels A's worker; private mirror values settle at UTC / Europe/London with no warning. | Serial edits save both; same-reminder later intent wins. Real production screen/service and private SQLite, controlled fake server client. Only timezone measured. Enter opens the row; assigning Select.value delivers the real Changed event. No claim of physical dropdown browsing, natural frequency or remote cancellation/rollback. |
| Study | Shell Start quiz holds the screen queue: a later enabled Dashboard event waits for the server response. | The inner Start attempt worker permits Dashboard during the same hold. An independent App message is processed in both arms, proving screen-queue starvation rather than an event-loop freeze. Real screen/window and scope/server services; held local fake API client; one start per arm. Full TldwCli, physical pointer, Escape and actual-unmount behavior not measured. |
| Chatbook preview | A real valid private ZIP retains `[TODO] plan` in name/description/author, but the mounted production preview drops `[TODO]` from all three displays. Validation passes. | Ordinary metadata renders exactly. Actual importer and step on_show worker, lightweight host. No storage corruption, malformed-closing-tag crash, physical file-picker or whole-wizard claim. Attached to existing owner. |

The final publishable scripts ran on merged dev: 2 Study, 3 Schedules and 2 metadata observations/controls passed, with no failure/error or skipped case. These tests assert existing symptoms to document the audit; they are not passing repair regressions. Before formatting, the same seven also passed on merged dev. Earlier candidate-head probes, an excluded missing-host-store setup attempt, and a missing-bootstrap-root attempt remain private diagnostic provenance.

## Duplicate-work and ADR checks

The filing screen covered all current tasks, archive/drafts locations, complete criteria for close owners, 312 unique fetched head/ref OIDs, 52 registered worktrees, historical task paths and audited-checkout frontmatter. No ID sweep error occurred. The final open-PR screen covered 29 PRs and their complete paginated file lists; no candidate owner file overlapped. The changed Console portrait PR was refreshed. IDs remain provisional against concurrent future minting until their branch lands on dev.

- Schedules: TASK-19559 covers named groups/default-group guards; TASK-19870 covers refresh serialization; TASK-36036 covers marked-reminder bulk admission; TASK-35866 Results actions; TASK-31711/31712 timezone/inspector UI. None specifically owns cross-reminder inline save cancellation. Existing Accepted ADR-018 applies; Proposed ADR-116 is contextual and is not ratified here.
- Study: TASK-1320/2834 cover mount/synchronous blocking I/O; TASK-33284/33289 cover their enumerated performance findings, with no all-Study responsiveness promise. TASK-34752/34755 and TASK-36099/36132 retain their list, teardown, status and draft scopes. Existing ADR-011 and ADR-033 apply.
- Metadata: existing TASK-32802.1/32903 own unescaped markup sinks. TASK-34000.67 continues to own other wizard entry/key/run/Close issues. Existing ADR-011 and ADR-161 apply.
- Evals launch flags are already claimed by Done TASK-1482/1483; no new task was filed without a fresh witness and owner reconciliation. No additional Watchlists defect or Library/Personas recompose/focus failure was established in this bounded pass.

No new ADR is needed for these localized corrections. Implementation must re-evaluate that decision if it changes durable attempt/mutation ownership, service/runtime boundaries, sync/conflict policy or long-lived UX structure.

## Reproduce and evidence

From the repository root with its Python 3.12 environment:

```powershell
python -X utf8 Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243/run-probes.py audit-new-label
```

Use a fresh simple label. The runner copies only the three probes and conftest into new private scratch directories, supplies isolated profiles and caches, retains network/real-profile guards, and records source blobs, child/parent JUnit, logs and observations. It runs only these seven bounded cases.

[Evidence package](2026-10-24-textual-book-after-pr3243/manifest.json): [merge receipt](2026-10-24-textual-book-after-pr3243/merge.json), [current-head CI execution](2026-10-24-textual-book-after-pr3243/ci-qualification.json), [Superpowers review](2026-10-24-textual-book-after-pr3243/integration-review.md), [merged-dev source pins](2026-10-24-textual-book-after-pr3243/source-pins.json), [verification](2026-10-24-textual-book-after-pr3243/verification.json), [observations](2026-10-24-textual-book-after-pr3243/observations.json), and [task/ADR ownership screen](2026-10-24-textual-book-after-pr3243/ownership.json). Raw private profiles/logs and extracted book-page text are not published.
