# Textual book audit after PR #3248

[PR #3248](https://github.com/rmusser01/tldw_chatbook/pull/3248) merged into dev at 2026-10-25T23:15:01Z as `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05`. All five commits rebased cleanly onto live dev `b166a47c41d458d0789df5468f7b86c28eb1ce9a`; range-diff preserved every patch. The actual merge tree matches reviewed/tested head `42243f8efcfd40996051c3359bfd3599fbc4d1f1`. The audit and its nine observation probes ran against fetched dev `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05`.

All 25 [Derived Artifacts jobs](https://github.com/rmusser01/tldw_chatbook/actions/runs/40241711952) and [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40241711939) passed. The logs qualify all 26 new UI cases exactly once across 17 shards; maximum shard wall time was 1081 seconds, within the unchanged 1200-second cap. Hosted Canvas reproduction passed. The existing Superpowers whole-branch review's Important navigation finding was already fixed; no unresolved review threads remained. Qodo was replaced as requested. The queue owned the final merge after auto-merge was armed. Earlier CI evidence remains qualified to its own source, rather than being relabelled as the rebased run.

## Book patterns and audit scope

The supplied 455-page book is reference material, not repository instructions. SHA256: `798aafc3ac58704c5053249ed228f5a45629355c3627e9611a3e78da815e9ab6`. Complete relevant pages were reread; PDF335 was rendered and visually inspected. Installed Textual is 8.2.8.

- Chapter 7, PDF145-146/158 (printed136-137/149): asynchronous work can leave the UI loop free while an awaited handler retains its own screen/window message queue. Admit an owned operation and let that handler return promptly.
- Chapter 9, PDF179-180/189 (printed170-171/180): recomposition replaces children and resets Input/DataTable editing and focus state. Update the changing region while retaining interactive widget identity.
- Chapter 17, PDF335 (printed326): blocking synchronous work belongs off the loop, and background publication must respect its UI lifetime. Runtime/scope/attempt generations and truthful prior-context feedback are application deductions, not a new remote rollback rule from the book.

This continuation examined Study Submit, Watchlists inbox selection and storage, Personas Lore Settings Save, and Research Runs follow-up admission. Settings/Personas sibling draft/save owners, Sources/Runs/Briefings publication and same-source Study binding were also screened. This is a bounded correctness/responsiveness audit, not certification of every route or a visual redesign. No product fix is implemented here.

## Prioritized backlog

Three new atomic To Do tasks have four unchecked outcome criteria each, source/book evidence and direct links to previous owners and canonical ADRs. Plans and completion notes are deferred until implementation.

| Rank | Priority | Task | Outcome |
| --- | --- | --- | --- |
| 1 | Medium (P2) | [TASK-36930](../../../backlog/tasks/task-36930%20-%20Retire-pending-quiz-Submit-publication-when-its-Study-context-changes.md) | Retire pending quiz Submit publication when its Study context changes |
| 2 | Medium (P2) | [TASK-36931](../../../backlog/tasks/task-36931%20-%20Keep-Watchlists-inbox-SQLite-work-off-the-UI-event-loop.md) | Keep Watchlists inbox SQLite work off the UI event loop |
| 3 | Medium (P2) | [TASK-36932](../../../backlog/tasks/task-36932%20-%20Keep-Personas-controls-responsive-during-Lore-settings-saves.md) | Keep Personas controls responsive during Lore settings saves |
| 4 | Medium (P2), existing | [TASK-33283](../../../backlog/tasks/task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md) | Preserve Notifications table focus during selection; retain the already attached Sources create-field editing-context witness. |
| 5 | Medium (P2), existing | [TASK-33279](../../../backlog/tasks/task-33279%20-%20PERF-20-HTTP-client-reuse---pooled-sessions-cached-SSLContext-nothing-built-on-the-loop.md) | Ensure Research Ask Follow-up releases its window queue as part of its existing off-loop objective. |

The two existing records receive new bounded evidence without changing their status, priority or acceptance criteria. TASK-33283 retains its TASK-33268 dependency. The original PERF-24 appendix explicitly lists Notifications recomposition, while its Mark-all-read database item names ScheduledTasksDB automation results. The inbox's synchronous ClientNotificationsDB calls therefore receive the separate narrowly scoped task above. Three new tasks and two existing owners account for all five fresh actionable findings; no finding is silently dropped.

## Witnesses and practical limits

| Area | Verified behavior | Control and scope |
| --- | --- | --- |
| Study final Submit | An old server completion after the actual runtime-reset observer and a new local Start clears the newer attempt ID and exact unsent draft, disables Submit and paints the old score. Real SQLite still has the newer attempt incomplete with no answers. | Ordinary same-context Submit succeeds once. Actual mounted Study screen/controller, schema/router and server/local services; controlled API client and Host runtime projection. No full TldwCli policy commit, live server or remote rollback claim. |
| Notifications storage | Original file-backed Mark read runs on the UI/main thread. A real competing writer held 0.7s produces 0.719 seconds write time and an independent heartbeat gap of 1.313 seconds; the durable read flag is correct after release. | Matched ordinary file-backed success; its measured write/gap is 3.531/4.265 seconds. The ordinary control is not a speed guarantee. Source also evaluates list/dismiss SQLite before _maybe_await. No fake slow DB result, natural contention frequency or native-frame claim. File/memory connection semantics must be preserved. |
| Lore Settings Save | A pending existing threaded service call retains the Personas screen pump: independent App callback runs, enabled rail-collapse and screen callback wait. Release processes them and real SQLite contains the typed name. | Matched ordinary save responds. A deliberate thread-boundary hold simulates about 0.6s storage latency. Real writer contention was tested and refuted as this trigger: it fails promptly on snapshot upgrade. No whole-loop freeze or data-loss claim. |
| Notifications selection | First actual Down selects row 1 and replaces the DataTable, losing focus; second Down does nothing. | Refocus then Down selects row 2. Production destination/CSS, private file-backed notification rows and actual Pilot keys; existing TASK-33283 owns this. |
| Research follow-up | Held async answerer lets the independent App tick while enabled Clear events remains queued; release renders the answer and performs Clear. | Ordinary control clears immediately. Actual ResearchWindow/engine/scope/local service and SQLite, controlled answerer boundary, real Button messages. No default-network timing claim; existing TASK-33279 owns this. |

The final source-pinned packet qualifies nine observation cases across separate domain invocations and every actual private child XML: one ordinary Study pass plus one intended retention assertion failure, three clean Watchlists cases, two clean Lore cases and two clean Research cases. Every child has zero errors/skips. These observations demonstrate current defects; they are **not passing repair acceptance tests**. Durations, exact observations and report hashes are in the linked evidence.

## Ownership and decisions

The duplicate screen covered 310 unique fetched branch/ref OIDs, 52 registered worktrees, historical task paths, audited-checkout frontmatter and all 30 open PRs' paginated file lists. 77 remote-only candidate task records and complete close-owner ACs are retained. No open implementation PR owns these three new corrections. Release PR #3246 promotes existing dev work to main; relevant Personas PRs #3148/#3050/#3049 do not change Lore Save, and PR #3166 changes Settings. IDs use a 30-ID lead above all observed claims; recheck before a future implementation PR if concurrent minting advances.

- Study: completed TASK-36899/36729 own Start; TASK-36360 definite failure/retry; TASK-34752 question-list overlap/teardown; TASK-36099 creation drafts. A successful pending final Submit after reset is a separate operation and owner.
- Inbox: TASK-15466 connection lifecycle, TASK-282 Home offload, TASK-21238 corrupt-store isolation and TASK-19559/19870 worker-group/refresh ordering remain intact. TASK-36900 Reader body reads are a different boundary.
- Lore: TASK-15469 already moved backend work off the loop; TASK-2834's blocking-call guard stops at to_thread. TASK-33910.3 covers list windows and TASK-33910.8/.14 guard/draft/transition behavior. None releases this ordinary Save handler's screen queue. Existing Dictionary/Character/Persona/Theme pending-save tasks govern their own forms.
- Sources create selection loss remains with TASK-33283; no duplicate. Same-source active-server binding replacement remains uncertified: Study's real legacy-config client provider differs from a fake retargetable provider, so no cross-server defect is claimed. Runs' generation/backend checks and Briefings' loader ownership yielded no additional demonstrated issue.

ADR required: no new ADR. Canonical [ADR-007](../../../backlog/decisions/007-personas-workbench-route-consolidation.md), [ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md), [ADR-042](../../../backlog/decisions/042-watchlists-reader-first-ia.md) and [ADR-068](../../../backlog/decisions/068-local-research-execution-engine.md) already establish the relevant UI/session/service boundaries. Historical task number references do not establish canonical ADR authority; unrelated ADR046/115 claims were rejected. Re-evaluate if implementation changes durable ownership, service contracts or long-lived UX structure.

## Reproduction and evidence

From the repository root, with its Python 3.12 environment and a fresh simple label:

    python -X utf8 Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248/run-probes.py fresh-audit-label

The runner checks the audited product/harness blobs, creates private profiles/TMP/bootstrap roots, retains repository network/profile guards and independently inspects parent and child XML. By default it executes all nine cases; optional trailing domain names select a bounded subset. Success requires the exact selected case counts and their qualified outcomes, including the named intended Study failure when Study is selected. The evidence records the actual separate domain invocations. Future repaired sources fail the historical-source check rather than being misreported as regressions. No full suite was run.

Early fixture assumptions, the in-memory inbox contention setup, the initial Research Pilot idle-wait timeout, the Windows report-path collection error and failed PDF runtime imports are excluded from final qualification. The merged repeat also refuted an unsupported ordinary-write heartbeat cutoff; an uncontended UI-thread write took 1.31s with a 1.91s loop gap. The diagnostic now keeps ordinary correctness/thread/timing evidence without assuming fast storage, while preserving the contended-write and heartbeat assertions. The first Lore ordinary control also confused service-thread completion with handler completion while its post-save reloads were still pending; its corrected precondition waits for the actual screen callback before the ordinary rail action, retaining the original responsiveness assertions. One interrupted diagnostic command and a parent timeout are excluded. Their private receipts remain retained. Study was qualified before that correction; the other three domains were qualified afterward. All executed probe sources and actual commands are retained, and all nine use the exact merged product/harness blobs.

[Manifest](2026-10-25-textual-book-after-pr3248/manifest.json) links the [merge receipt](2026-10-25-textual-book-after-pr3248/merge.json), [CI execution qualification](2026-10-25-textual-book-after-pr3248/ci-qualification.json), [source pins](2026-10-25-textual-book-after-pr3248/source-pins.json), [observations](2026-10-25-textual-book-after-pr3248/observations.json), [verification](2026-10-25-textual-book-after-pr3248/verification.json), [ownership screen](2026-10-25-textual-book-after-pr3248/ownership.json) and [commands](2026-10-25-textual-book-after-pr3248/commands.json).
