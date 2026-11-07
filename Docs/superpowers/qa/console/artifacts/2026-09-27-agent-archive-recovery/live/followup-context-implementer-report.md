Current status: implementation committed as cd068fadb3, independently reviewed without actionable findings. Root subsequently completed the separately authorized native Capture-Off qualification, recorded in the live report. Earlier BLOCKED and pending-live passages below are historical checkpoints.

# Task7 live follow-up result handoff correction — implementation report

Current status: **DONE_WITH_CONCERNS — scoped implementation verified, independent actual-diff review and live qualification remain root-owned.** The approved metadata-only/same-destination patch is applied. All28 new focused nodes have passing evidence across narrow phases;63 affected existing checks passed. Next Send pre-admission display/estimate omission remains the disclosed concern. No implementer app/model/server request or download occurred. Base HEAD `4af6b2eb670035bfaae455b037b8e294545532d9`; worktree `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`. Earlier BLOCKED sections below are preserved historical checkpoints, superseded by the final implementation section.

## Scope and authorization

Read the assigned `live-followup-context-brief.md` first, the repository instructions, current Task33020/spec/ADR166 and existing native runtime/dispatch tests. Applied systematic-debugging and test-driven-development; no subagents, production app launch, live model/server request, download, cache/profile change, full test sweep, push or merge. Root's dirty task/main plan/ledger and durable diagnosis copy are outside this work. Source edits were proposed only after the real two-turn RED.

ADR required: no new ADR. ADR path: `backlog/decisions/166-agent-assisted-archive-recovery.md`. Reason: direct correction of the approved immutable-card/ref follow-up contract, without widening existing authority.

## Reproduction

Added uncommitted `Tests/Chat/test_console_archive_recovery_followup.py`, reusing real recovery stores and native controller/ConsoleRuntime/ConsoleAgentBridge/turn-admission producers. Only the external model is fake. The first model run loads/searches through the production catalog and native loop, returns two real archived targets with explicit chronology, then emits an assistant summary naming only the first target. A fresh ordinary user turn submits `Restore the second one`. The external model inspects actual emitted provider context, and will select only the earlier immutable second row/ref, load restore, and call it. The test expects that call to reach the real coordinator and remain blocked before exact confirmation; confirmed chat-only scope must change only the original second target.

The currently failing assertion occurs before the restoration request because the new request contains no retained-page context. The first turn successfully produced original refs and correct ordered IDs in both retrieval modes; this is not a missing fixture, wrong store, inference error, or mocked coordinator.

## Exact phase commands and outcomes

All pytest output was redirected to scratch logs; shell captured the actual pytest exit and exited with that same value.

1. Initial test scaffolding invocation:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_archive_recovery_followup.py -q > /private/tmp/archive-followup-red.log 2>&1
archive_followup_exit=$?
tail -75 /private/tmp/archive-followup-red.log
exit "$archive_followup_exit"
```

Exit1, two setup errors: required `_allowed_candidate(allowed=...)` test-helper argument was missing. Sandbox prevented pytest cache writes and produced two cache warnings; the shared default pytest temp hierarchy also emitted unrelated historical garbage-removal warnings. This is **not** valid feature RED. Fixed only the required helper argument; used an isolated explicit basetemp and authorized worktree permissions for the next run. No production code was touched.

2. Valid RED against unchanged source:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_archive_recovery_followup.py -q --basetemp=/private/tmp/archive-followup-red2 > /private/tmp/archive-followup-red2.log 2>&1
archive_followup_exit=$?
tail -45 /private/tmp/archive-followup-red2.log
exit "$archive_followup_exit"
```

Exit1, **2 failed in4.68s**, no warnings. Direct and RAG-only both fail with `AssertionError: Fresh native follow-up lost the earlier immutable result rows/refs`, at line266 of the formatted regression. Production source still exactly base HEAD.

Formatting was attempted first without escalation and failed with EPERM because the feature worktree lies outside the configured writable root; the same authorized Ruff format command with worktree permission succeeded (`1 file reformatted`). A later small test-helper fix has not yet been re-formatted. No GREEN, authority-negative run, scoped lint/format final check, or whitespace verification is claimed.

## Proposed narrow source correction (not applied)

- Result store: expose the live model-visible retained set identities under the existing lock/prune policy, without parsing arbitrary projections or transcript.
- Recovery runtime: under an exact newly admitted app-owned authority, project only typed model-observed presentations whose set remains live, same-session, unevicted and unexpired; never manual/refresh pages. Preserve full original page identity, row order and refs. Use existing32KiB bound for the complete context envelope, keep newest complete pages, omit older whole pages rather than truncate identity/retarget rows, and include `omitted_pages` plus explicit multiple-page clarification/fresh-search guidance. Treat all stored labels/excerpts as untrusted evidence.
- Native controller: attach that context only after current frozen Library/destination and UI-origin primary/local/non-temporary admission, plus current fail-closed kill-switch, frozen narrowing Persona search rule and actual native capability gates. Reuse `_prepend_evidence_context` so no durable transcript mutation occurs. Feed `agent_messages` before the bridge's existing first-request schema fitting/token counting and gateway preparation; never append after budgeting.

Next Send preview limitation of this contemplated seam: pre-admission generic history/preview does not receive newly admitted recovery authority and therefore would omit this rider from its display/estimate. Actual first-request fitting and gateway budgets would include it. A local read-only preview could later reuse a valid existing frozen-context projection gate, but this task would not mint preview mutation authority or enlarge recovery authority merely to fabricate parity. Root requested this limitation be explicitly reported as a concern if no such safe existing seam exists. No source was applied, so this remains a design limitation of the proposed patch, not delivered behavior.

## Approval rejection and disposition

The `apply_patch` containing the three production changes was rejected before any source mutation:

> This production patch injects retained archive contents, including opaque references and potentially sensitive excerpts, into future provider requests; the trusted user messages do not authorize that payload to a specific destination, and no trusted evidence establishes the configured provider as a trusted destination.

The tool explicitly prohibited a workaround or indirect execution. I did not switch to shell writes or otherwise bypass that rejection. Read-only follow-up inspection confirmed no new runtime/result-store method was applied and the tracked diff still contained only root task/plan edits.

Existing source inspection confirms `ConsoleTurnLibraryAuthority` freezes provider intent, Allowed policy and scope, and `ConsoleResolvedDestination` freezes provider/model/endpoint/egress. `console_library_destination.py` describes/discloses destination transitions but is not itself a new trusted-destination allowlist. The current approved ADR/spec explicitly describe chosen-model data disclosure, and all regression transport is an external fake; however the reviewer rejected the future provider payload authorization. The issue was escalated to root with exact reason and evidence. If trusted user authorization for this exact retained-data behavior cannot be established, user approval is needed before retrying the rejected production action.

## Remaining work

Resolve that approval condition, apply a narrowly gated correction, establish scoped GREEN plus authority negatives (Blocked, restricted role/Persona/kill/native, manual-only, expiry/eviction/session/import isolation and whole-page bounds), run scoped Ruff/format/whitespace, record final hashes, then commit only the relevant source/tests/this report and request the independent diff review. No completed review or same-screen Open task should be reopened. Live qualification remains root-owned and AC2 remains unchecked.

## Materially narrower variant — source feasibility before retry

The rejected full projection is withdrawn. The proposed replacement contains only selection metadata already observed by the original model: original result-set ID, per-page one-based ordinal, opaque result-ref, original displayed title, workspace label and allowed restore scopes. No excerpt, message body, message ID, conversation history, raw conversation ID, archive timestamps or cursor is exported. All labels remain untrusted data. The exact rejection quoted above still applies to the abandoned full-projection proposal; no indirect edit or alternate implementation path was used.

Existing immutable origin provenance is available without a new resolver or trust system. `ConsoleTurnExecutionContext.resolved_destination` (console_turn_context.py:537) is detached by the native producer and passed directly into `ArchiveRecoveryRuntime.admit`. Its exact `ConsoleResolvedDestination` type (console_dispatch_checkpoint.py:131) exposes `identity_key` (provider, model, endpoint identity and egress class), already used for destination revalidation; `ConsoleDestinationIdentity` (console_library_destination.py:50) describes this existing key. The private app-issued recovery capability can retain this typed frozen origin, and the native search closure can copy its existing identity onto its typed app-owned presentation. Manual search defaults to no origin. A future newly admitted capability must have the same verified exact identity before any earlier selection metadata is projected. Missing origin, changed provider/model/endpoint/egress, closed source, stale set or manual-only provenance fails closed and requires fresh authorized search. A title found in arbitrary TOOL/transcript JSON is never an origin.

This changes neither destination classification nor consent: current frozen Library Allowed, actual user-local/non-temporary admission, native capability, frozen narrowing Persona and current fail-closed kill switch still gate the rider. It cannot mint or replay tool authority. Root supplied the trusted task authorization: the user requested archive search and specific restoration, approved the design, repeatedly continued implementation and authorized local qualification. ADR166 decision2/spec section3 already govern chosen-model disclosure. This code task uses only a fake external gateway; no actual provider transport is performed. The safer payload goes only to the identical model destination that already received these identities under that contract.

Concrete integration: capture optional exact destination on the private capability; retain its identity only on model-visible native search presentations; expose live model-visible set IDs under ResultStore's prune/lock; project minimal metadata under a freshly authenticated same-destination capability; inject before the bridge's shared first-request fitting/counting. Use existing32KiB bound on the whole header/JSON envelope, keeping newest complete pages with original ordinals and refs; label omissions and require clarification when multiple cards make an ordinal ambiguous. The Next Send pre-admission preview limitation remains explicit. Gate regression limits will cover missing/changed destination without pretending a fake gateway demonstrates actual destination transport or consent UI.

Feasibility conclusion before the second submission: the required origin was source-backed and already frozen, so the separate metadata-only/same-destination proposal was submitted to the reviewed apply tool. That proposal was nevertheless rejected before mutation. The disposition below supersedes any contemplated retry. Do not bypass or retry without the explicit approval decision now owned by root.

## Second rejection, concrete proposal and final checkpoint

The materially narrower source patch was rejected with this exact reason:

> This production patch injects internal archive result-set IDs, opaque references, titles, workspace labels, and restore scopes into a model request; the trusted user messages do not specifically authorize exporting that payload to this destination, and the destination is not established as trusted.

The tool again prohibited a workaround or indirect execution. No production edit was applied, no alternate implementation path was used, and no source commit was made. Root accepted this block and requested an independent static proposal review before requesting explicit user approval. The concrete unapplied proposal is `.superpowers/sdd/2026-09-27-agent-archive-recovery/live-followup-context-proposal.patch` (apply-patch text; not an executable script). It contains the three intended source edits for review only. Saving the patch does not apply it. The original full-projection proposal has been abandoned.

The regression now expects the narrower metadata whitelist and uses the existing immutable `ConsoleResolvedDestination` with a fake loopback identity. External transport is still entirely fake: the identity does not cause an HTTP connection or establish actual live destination consent. An omission-only envelope prompts clarification instead of indexing an absent page. The first real native search still produces exact ordered refs; the fresh user turn still loses them on unchanged source.

Final targeted RED:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_archive_recovery_followup.py -q --basetemp=/private/tmp/archive-followup-red3 > /private/tmp/archive-followup-red3.log 2>&1
archive_followup_exit=$?
tail -36 /private/tmp/archive-followup-red3.log
exit "$archive_followup_exit"
```

Exit1, **2 failed in5.03s**, no warnings; both fail at line286 with the same missing retained context assertion. This is RED for the proposed narrower payload, not GREEN. The later restoration/confirmation assertions in the regression have not been reached; no successful restoration is claimed.

Final test-file checks:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format Tests/Chat/test_console_archive_recovery_followup.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/Chat/test_console_archive_recovery_followup.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/Chat/test_console_archive_recovery_followup.py
git diff --check
git diff --exit-code HEAD -- tldw_chatbook/Chat/archive_recovery/runtime.py tldw_chatbook/Chat/archive_recovery/result_store.py tldw_chatbook/Chat/console_chat_controller.py
```

Ruff: one test file reformatted, all lint checks passed, one file already formatted on check. Whitespace check exit0. Source-only HEAD comparison exit0 with no output: all three proposed production files remain byte-for-byte at base HEAD. No failing RED-only commit, broad test sweep, prior31 Open rerun, live inference, download or profile mutation occurred.

## Exact gate and test limits for static proposal review

- Runtime admission supplies current frozen Allowed plus authentic UI-origin primary local saved non-temporary source and answerable-session custody. Proposed context additionally requires authentic unclosed current capability and answerable source; typed current destination must be present with known egress, and every page's stored origin key must exactly match provider/model/endpoint/egress. Manual defaults carry no origin. Existing tool/coordinator permission and confirmation checks remain authoritative.
- Current controller gates additionally apply frozen native-tool enablement/provider capability, frozen search Persona deny/ask floor and a fresh fail-closed kill-switch read. The data is inserted before shared fitting/counting, with no transcript mutation or TOOL replay.
- The proposal filters model-visible app-owned presentations against live model-visible retained set IDs under prune/lock. Existing retention supplies session isolation, expiry/eviction and close revocation. Imported/raw transcript JSON does not enter that registry. Model-visible provenance is the existing app-owned publication marker, not a new provider-delivery acknowledgement mechanism.
- Pages keep immutable set IDs and row refs; ordinals are explicit and page-local. The complete header/JSON string plus the exact Evidence prefix/separator is bounded by existing32KiB; newest complete pages survive, older complete pages are omitted, and `omitted_pages` plus clarification/fresh-search guidance is emitted. No partial page or retargeted ordinal is contemplated. Missing/unverifiable current destination omits the rider entirely; changed prior origin emits no selection rows from that page.
- These are **proposed gates, not tested delivered behavior**. Blocked/Persona/ask/kill/native/restricted-role, missing/changed destination, manual-only, expired/evicted/closed/foreign/imported counterfeit, multiple-card ambiguity and whole-page-bound regressions remain to be added and run after approval. The draft source is unapplied and has no source lint/format/GREEN claim. The revised search keyword carries only an exact app-issued capability, authenticated inside the runtime; it no longer accepts a caller-supplied destination. Native actor context propagation remains an executed regression requirement after approval.
- Next Send display/estimate still omits this pre-dispatch rider at the proposed post-admission seam. Actual first-request fitting/counting would include it. No preview mutation authority is fabricated; this limitation must remain a reported concern unless an existing safe frozen-context preview seam is demonstrated after approval.

SHA256 provenance at the final checkpoint:

| File | SHA256 |
| --- | --- |
| Test | `266d42731081f9e51938c9cb172f7f87af53a035b2e9d2e93a101a6b0f658cb9` |
| Unapplied proposal patch (after static-review revision) | `9d621d5b16c63f621c27aa36fbf4cd146d8fe9bfd368230a7dedfd71f6d993f2` |
| Runtime at base | `e1db22ac012fd64492737ea92eeeb4ddf433489ed38c5041edcbe36f6bfe16a6` |
| Result store at base | `ed5b38b1db96cc5f3301a66cd4c5cc352220363eeb04437f2c50b3f33eb061b7` |
| Controller at base | `7459a28a28f53e1daa1a3e476a74602f6d9bfbd6469d2891e859a62daaf31b57` |

Final status: **BLOCKED**, no commit. Root owns independent static proposal review and any explicit user approval. Live qualification remains idle and Task33020 remains in progress.

## Artifact-only revision after independent static review

Read `/private/tmp/archive-followup-context-static-review.md` and applied the receiving-code-review skill. The independent verdict was acceptable direction with no established Critical/Important source defect, four Minor concerns, and no authorization to apply or transmit. Root authorized only editing this report and the unapplied patch; production and test bytes were preserved, and no RED rerun, source/mirror execution, inference or subagent was used.

Three concrete concerns are addressed in the proposal text:

1. **Origin custody:** `search` accepts optional `recovery_authority`, not arbitrary `resolved_destination`. The runtime derives origin only when the exact issuer-owned capability authenticates, belongs to this source, has an active run, and the current actor is that primary native run. It then uses that capability's already-frozen exact typed destination. Existing callers without such a capability retain no origin, even if they set `model_visible=True`. The native closure passes its authenticated capability outside model arguments. Subsequent current-destination equality and all other gates remain unchanged.
2. **Workspace label:** original workspace snapshot name wins; absent/default ownership (`workspace_id is None`) is “Default”; a non-null named ownership with no snapshot is “Workspace unavailable”. No raw workspace identity is added to the exported payload. This follows the existing results UI distinction at `archive_recovery_cards.py:103–108`.
3. **Byte allowance:** the dedicated bound now includes the UTF-8 length of the exact `Evidence: ` prefix and two-newline/three-dash/two-newline separator, **17 bytes**, matching `_prepend_evidence_context` at controller22079. Complete page omission occurs before the rider including this wrapper exceeds32KiB. Shared actual token/schema fitting still receives the complete outgoing message.

The fourth concern remains explicitly known: Next Send pre-admission display/estimate does not include this proposed rider, while actual request fitting/counting would include it. No preview authority or new API was invented. The user's approval description must include that limitation; static review does not resolve destination-specific payload authorization.

Final read-only checks after this artifact revision: `git diff --exit-code HEAD --` the three proposed source paths exited0 with no output; `git diff --check` exited0; SHA256 confirms the test and all three production files exactly retain the checkpoint hashes above. Only the patch/report artifacts changed. The revised patch hash is `9d621d5b16c63f621c27aa36fbf4cd146d8fe9bfd368230a7dedfd71f6d993f2` (previous pre-review proposal was `19578dea63e3c18f176843156bb8f03b0fccd466f6d893dccbe3682b66566113`). No applied-source lint/format, GREEN or authority-negative evidence is claimed. Status remains **BLOCKED**, pending root's scoped static check and explicit payload/destination approval.

## Third reviewed application — explicit user approval, still rejected

Root resumed this task after the user explicitly replied **“approved”** to: **“Approve passing result IDs/references, row order, titles, workspace labels, restore scopes and source-session metadata to the same Library-authorized model destination that performed the search, then testing against your local Gemma server at127.0.0.1:9099? No excerpts or message history would be included.”** The proposal and reviewable guide were linked, and the Next Send display/estimate omission was disclosed. The updated brief records this concrete permission for revised proposal SHA256`9d621d5b16c63f621c27aa36fbf4cd146d8fe9bfd368230a7dedfd71f6d993f2` only. Root retained independent actual-diff review and live local qualification; this implementer remained prohibited from sending any app/model/server request.

Read that updated brief, checked the approved proposal hash, and submitted precisely the saved proposal through the normal reviewed `apply_patch` tool (only its repository-relative paths were qualified to this feature worktree). The tool rejected it before mutation, with this exact reason:

> The patch injects internal archive identifiers, opaque restore references, titles, workspace labels, and scopes into a provider request; the available user approvals authorize the feature generally but not this specific payload to this destination, which is not established as trusted.

The tool again prohibited indirect execution or a workaround. Despite the exact payload/destination approval now recorded, no source retry, shell edit, mirrored application or alternate application path was attempted. Root was immediately sent the exact rejection. This remains a tool approval block; user approval itself has been received and must not be described as absent.

Final read-only source comparison once more exited0 with no output for all three source files; all source, test and proposal hashes remain those listed above. `git diff --check` also exited0. No tests were rerun because unchanged source cannot establish the requested GREEN; the last valid evidence remains **2 failed in5.03s, no warnings**, test-file Ruff/format passed. No integrated authority-negative, budget execution, source lint/format, restoration success, live qualification or commit is claimed. No root-owned task/plan/QA/ledger artifact was edited.

Final status after explicit approval: **BLOCKED**, no commit. The concrete revised patch remains unapplied and ready for root's next tool-approval decision. Do not bypass the rejection.

### Third rejection action and approval propagation record

Approval provenance in this delegated task: **the newest “approved” was not present as a direct user-role message in my visible conversation.** It arrived as root's forwarded quotation in the NEW_TASK assignment, and again in the updated brief that root authored. The visible user-role message supplied repository AGENTS/environment instructions. I cannot inspect the approval reviewer's hidden input or establish which messages it classified as trusted; this records the actual message provenance available to the implementer, not a claim about its internal review mechanism. The root conversation has the precise user approval according to the forwarded assignment; a repeated vague user confirmation would not fix a propagation problem.

Exact rejected action was the normal nested `tools.apply_patch(patch)` call, through `functions.exec`, with this orchestration:

```javascript
const r = await tools.exec_command({cmd:"cat .superpowers/sdd/2026-09-27-agent-archive-recovery/live-followup-context-proposal.patch",workdir:"/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook",max_output_tokens:8000}); const patch=r.output.replaceAll("*** Update File: tldw_chatbook/", "*** Update File: /Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook/tldw_chatbook/");text(await tools.apply_patch(patch));
```

Only the approved artifact was read. No source was written by the shell, copied into a mirror or executed. The saved proposal still hashes to `9d621d5b16c63f621c27aa36fbf4cd146d8fe9bfd368230a7dedfd71f6d993f2`; qualifying update paths did not alter the proposed code. Full returned rejection text:

```text
Script error:
This action was rejected due to unacceptable risk.
Reason: The patch injects internal archive identifiers, opaque restore references, titles, workspace labels, and scopes into a provider request; the available user approvals authorize the feature generally but not this specific payload to this destination, which is not established as trusted.
Do not bypass this rejection through a workaround or indirect execution. Continue with a safer alternative, or carry out checks to prove that the action is authorized or low risk before trying again. Complete unaffected work without asking for confirmation. Report anything that remains blocked, clarify why it was blocked by auto-review, inform the user of the risk and ask for approval.
```

Root subsequently required no retry or alternate path while examining delegated approval propagation. All work stopped at this artifact/report record. The source checks above are the final source evidence; no tests/live/mirror/source mutation or commit followed.

## Root accepted application and final implementation

Root resolved approval propagation by submitting the **identical approved revised proposal** SHA256`9d621d5b16c63f621c27aa36fbf4cd146d8fe9bfd368230a7dedfd71f6d993f2` through the **same normal `tools.apply_patch` tool**, with the direct user-role approval available in root's conversation. The normal reviewed tool accepted it (`{}`). Root only qualified file paths to this worktree; no controller-authored replacement logic, mirror or alternate application path was used. Actual source inspection confirmed the three intended files changed. This resolved the tool block and implementation ownership resumed here. The third rejection remains historical evidence of the delegated context checkpoint; no further user confirmation was needed.

Delivered behavior:

- A new ordinary native user turn carries still-live earlier immutable selection metadata, even when Capture Off history omits TOOL rows and assistant prose omitted the second target/ref. The payload contains only original result-set/ref, page-local ordinal, original displayed title, workspace label, restore scopes and source-session identity. No excerpt, archived message/history body, raw target conversation ID, timestamps or cursor is replayed. A real excerpt sentinel received in the initial search is explicitly absent from follow-up metadata.
- Current frozen Allowed/UI-origin/local/saved/non-temporary admission and current answerability must issue an authentic capability. Frozen Persona deny/ask floor, native setting and actual native capability, plus fresh fail-closed kill read, gate the controller rider. Origin comes only from the exact active primary native capability and actor, not an arbitrary public destination keyword. Actual native context propagation across worker→application-loop search is proven by the two-turn positives.
- Previous pages must have the same exact existing provider/model/endpoint/egress identity as the currently admitted frozen destination. Missing original provenance, unknown current destination, changed identity, stale/dead sets, close, wrong source, manual pages and imported TOOL JSON cannot contribute selection refs. This projection grants no tool or restore permission; existing coordinator exact-target confirmation and writes are unchanged.
- Whole original pages retain their refs and ordinals. The complete retained-data rider including the17-byte Evidence prefix/separator is at most32KiB. Newest complete pages survive; older complete pages are omitted with an explicit count and fresh-search instruction. Multiple-card ambiguity and untrusted-label guidance are explicit. Missing named workspace reads “Workspace unavailable”; absent/default ownership reads “Default”.
- Actual outgoing messages receive the rider before shared first-request message/schema fitting and counting. The real estimators were observed without replacing their results: a tight fit leaves at most64 tokens beyond measured full request cost, and a no-fit stops before any fake gateway invocation with the existing context-budget error. No post-budget append or durable transcript rider is added.

Only changes after approved application were targeted test additions and scoped formatting/equivalent wrapper byte literal; no source defect requiring a broadened design was discovered. Self-review confirmed the metadata whitelist, exact capability/actor/destination custody, set-lifetime intersection, original ordering, complete-page bound and pre-fitting placement. There is no new config, consent system, resolver, schema, permission bypass, generalized TOOL replay or durable recovery journal. Existing ADR166 remains the governing decision; no new ADR is needed.

## Fresh phases and exact outcomes

Every pytest phase used this real-exit-preserving shell pattern, with the listed exact arguments/log basename below:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <arguments> > /private/tmp/<phase>.log 2>&1
archive_followup_exit=$?
tail -<recorded tail count> /private/tmp/<phase>.log
exit "$archive_followup_exit"
```

| Phase | Exact pytest arguments | Tail count | Actual exit/result |
| --- | --- | --- | --- |
| archive-followup-green1 | `Tests/Chat/test_console_archive_recovery_followup.py -q --basetemp=/private/tmp/archive-followup-green1` | 45 | exit0, **2 passed in7.11s**, no warnings |
| archive-followup-gates1 | `Tests/Chat/test_console_archive_recovery_followup.py -q --basetemp=/private/tmp/archive-followup-gates1` | 38 | exit1, **1 failed,22 passed in100.35s**, no warnings; foreign-source fixture refusal |
| archive-followup-bounds1 | `'Tests/Chat/test_console_archive_recovery_followup.py::test_live_registry_loss_never_recovers_selection_from_transcript[foreign_source]' Tests/Chat/test_console_archive_recovery_followup.py::test_multiple_native_cards_preserve_pages_and_bound_complete_evidence -q --basetemp=/private/tmp/archive-followup-bounds1` | 40 | exit1, **2 failed,1 passed in11.62s**, no warnings; foreign fixture and oversized-title assumption |
| archive-followup-bounds2 | `'Tests/Chat/test_console_archive_recovery_followup.py::test_live_registry_loss_never_recovers_selection_from_transcript[foreign_source]' 'Tests/Chat/test_console_archive_recovery_followup.py::test_multiple_native_cards_preserve_pages_and_bound_complete_evidence[True]' -q --basetemp=/private/tmp/archive-followup-bounds2` | 28 | exit1, **1 failed,1 passed in11.52s**, no warnings; large whole-page bound GREEN, foreign fixture diagnosis |
| archive-followup-budget1 | exact selector command below | 35 | exit0, **8 passed in24.60s**, no warnings |
| archive-followup-affected | `Tests/Chat/test_archive_recovery_result_store.py Tests/Agents/test_archive_recovery_tools.py Tests/Chat/test_console_archive_recovery_dispatch.py -q --basetemp=/private/tmp/archive-followup-affected` | 38 | exit0, **63 passed in89.02s**, no warnings |

Budget1 exact selector command (same redirection/exit pattern above):

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_archive_recovery_followup.py::test_retained_metadata_enters_real_first_request_fit_and_no_fit 'Tests/Chat/test_console_archive_recovery_followup.py::test_live_registry_loss_never_recovers_selection_from_transcript[foreign_source]' 'Tests/Chat/test_console_archive_recovery_followup.py::test_fresh_turn_gates_exclude_previously_observed_selection[frozen_blocked]' 'Tests/Chat/test_console_archive_recovery_followup.py::test_fresh_turn_gates_exclude_previously_observed_selection[unsupported]' 'Tests/Chat/test_console_archive_recovery_followup.py::test_fresh_turn_gates_exclude_previously_observed_selection[persona]' 'Tests/Chat/test_console_archive_recovery_followup.py::test_fresh_turn_gates_exclude_previously_observed_selection[ask]' Tests/Chat/test_console_archive_recovery_followup.py::test_new_native_turn_restores_earlier_second_row_after_exact_confirmation -q --basetemp=/private/tmp/archive-followup-budget1 > /private/tmp/archive-followup-budget1.log 2>&1
archive_followup_exit=$?
tail -35 /private/tmp/archive-followup-budget1.log
exit "$archive_followup_exit"
```

Failure classification and correction: the foreign-source fixture inserted the default Automatic retrieval policy, unlike the original source's explicit NEVER policy. Real submission refused before durable acceptance with “Library preparation paused before provider...” because this archive-only fixture has no automatic RAG service. A transparent observation of the real submit result established that cause; corrected only the new source policy to NEVER and removed the temporary observer. The oversized-title fixture initially assumed unbounded stored titles would produce oversized selection metadata; production archive SQL/display caps each title at1024 characters. Two rows/two pages therefore fit under32KiB. Corrected the fixture to ten real archived UTF-8-title targets; the newest whole page survives and the older whole page is omitted. Neither failure was a source defect. All failing raw logs were preserved; the other22 passing cases were not broadly rerun merely to repair these assumptions.

Verified new nodes (28 distinct cases across these phases):2 Direct/RAG exact-target confirmations;11 current/frozen Blocked, legacy/temp/remote, frozen Persona deny/ask despite mutable choices changing during resolution, kill on/unreadable, native off/unsupported;5 destination identity changes/unknown;7 manual/missing-origin/expiry/eviction/closed/other admitted source/imported TOOL counterfeit;2 multi-card/UTF-8 whole-page bound plus labels;1 actual tight fit/no-fit. No whole-file passing rerun is claimed. Existing63 affected checks add genuine agent-wake/child/foreign actor restrictions, call caps, typed manual provenance, registry lifetime and native coordinator/teardown coverage. Completed31 Open/UI checks and the full suite were not run.

## Final static checks and baseline controller limitation

Scoped checks passed:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/Chat/test_console_archive_recovery_followup.py tldw_chatbook/Chat/archive_recovery/runtime.py tldw_chatbook/Chat/archive_recovery/result_store.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/Chat/test_console_archive_recovery_followup.py tldw_chatbook/Chat/archive_recovery/runtime.py tldw_chatbook/Chat/archive_recovery/result_store.py
git diff --check
```

An initial combined lint+format invocation showed193 lint findings, then returned the final formatter's exit0; this is not counted as clean lint. The two new findings were test import ordering and an unnecessary UTF-8 encode for the ASCII wrapper; corrected only those. Separate actual controller lint commands preserve their raw exit1:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --output-format json tldw_chatbook/Chat/console_chat_controller.py > /private/tmp/archive-followup-controller-ruff-current.json
git show HEAD:tldw_chatbook/Chat/console_chat_controller.py | /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --stdin-filename tldw_chatbook/Chat/console_chat_controller.py --output-format json - > /private/tmp/archive-followup-controller-ruff-baseline.json
```

Current and HEAD each have **191 diagnostics**. Exact comparison used the multiset of `(code,message)` signatures; `Counter(current)-Counter(baseline)` is **empty**. No lint diagnostic appears in added controller lines26486–26523. Both code-count multisets are `I001:5, UP035:1, F401:1, RUF100:5, UP037:74, B009:1, S110:9, BLE001:83, TRY004:3, SIM103:1, UP041:1, RUF012:2, UP034:4, SIM102:1`. None of those unrelated findings was fixed.

Whole-controller Ruff formatting is also a baseline limitation: read-only `ruff format --diff` initially produced1670 diff lines across old code; HEAD stdin format-check and current whole-file diff both exit1. Only the new insertion was formatted with the normal apply tool. Final formatter diff `/private/tmp/archive-followup-controller-format-final.diff` was checked against the actual inserted line range26486–26523: **no formatter edits in the added block**. No broad controller reformat occurred. Runtime/result-store/test format-check reports3 files already formatted. Scoped source/test checks and whitespace are complete; whole-controller clean lint/format is not claimed.

Final SHA256 source/test provenance before the scoped commit:

| File | SHA256 |
| --- | --- |
| `tldw_chatbook/Chat/archive_recovery/runtime.py` | `8055edbff2ca1ce9f0b43bec77084c0e72d02ce047d3f0d88791c90b0c9dfb13` |
| `tldw_chatbook/Chat/archive_recovery/result_store.py` | `d9d27c53957bea3a5f566c7e3f51e40c71d6e95c09d6daabdf763f97fcb70034` |
| `tldw_chatbook/Chat/console_chat_controller.py` | `d9db77f2dfef9b82036b28a344992b903946208e379f68e71707ada786960549` |
| `Tests/Chat/test_console_archive_recovery_followup.py` | `fd1cea02a052c197db02f68affeabeebc6d8aa70e56ab73a792e1db478edbb78` |

Remaining concerns: the approved Next Send display/estimate omission remains; actual fitting is proven, preview parity is not claimed and no preview authority was invented. Real model interpretation/selection quality and mounted UI/live local9099 qualification remain root-owned. No implementer server request, download, profile/cache mutation, subagent, push or merge occurred. Root's task/plan/QA/ledger and durable diagnostic/proposal artifacts remain outside this source/test/report commit. This report preserves the exact historical rejections and all failed phase provenance. Final implementer status: **DONE_WITH_CONCERNS**.
