# Runtime-owned initial Console hook review implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the user-authorized parallel lanes, with root as sole integration owner. Steps use checkbox tracking; all native runs remain sequential.

**Goal:** Give initial Send hook review an app-owned decision lifetime before enabling immediate received-input feedback.

**Architecture:** Add one named asynchronous hook-review lifecycle to the resident InterruptRoundHost. Reuse its lock, registries, payload ordering and pending-round accounting. Runtime methods expose the request and checked actions; disposable views present the existing modal. No executor thread or native lease waits for human input.

**Tech stack:** Existing Python 3.12 asyncio, Textual 8 and HookPermissions; no new dependency, scheduler, storage format or cache.

**Spec:** [Console Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md), sections 1, 3 and 4.
**Task:** TASK-34563.10.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: direct implementation of the accepted runtime initial-review prerequisite; ADR-197 consent/launch and ADR-220 resident decision ownership remain authoritative.

## Global constraints

- Preserve saved-turn failure: stop Send and keep the draft; temporary chats, WAL/NORMAL, required traces/consent history and awaited history remain unchanged.
- Early receipt stays disabled. Current pre-custody UI dispatch may still retain its existing caller and perform its generation/draft checks. The new runtime operation itself must retain no Screen, widget or dispatch continuation.
- Migrate stock waiting-for-Send review only. The explicit Hooks button and Settings review retain their existing route; background submissions still refuse unreviewed hooks without opening a modal.
- Preserve the injected three-positional review callback and original live permission-method lookup. A completed review is never launch clearance; actual config/consent and launch guards remain.
- No new visual controls, CSS, keybindings, module/size caps or timing budgets. Reuse the current modal and token-backed presentation.
- No full suite. Preserve node deadlines and the existing private-profile runner. Every native batch has explicit coordinated OPEN/CLOSED and physical retirement evidence.

## Ownership and files

- Shared lane: `Chat/console_interrupt_rounds.py` and new lightweight `Chat/console_hook_review.py` (review result/projection types only). No broad controller receiver or second registry.
- Integration lane (root until handoff): `Chat/console_runtime.py`, `Chat/console_chat_controller.py`, `Chat/console_chat_models.py`, `UI/Console_Modules/hooks.py`, `UI/Console_Modules/wiring.py`, `Widgets/Console/console_hooks_review_modal.py`. Add an explicitly named small Console view adapter only if modal projection cannot remain clear in the existing module. Avoid adding ChatScreen methods or TaskResumeState fields.
- Verification lane: new `Tests/Chat/test_console_initial_hook_review.py` and `Tests/UI/test_console_initial_hook_review.py`; existing tests remain unchanged unless a separately evidenced fixture repair is necessary.
- Root: task, plan, review resolution, final integrated runs and local commit. Implementers never run native checks concurrently.

## Agreed boundary

There is no existing hook-review request kind or InterruptKind enum. Add `CONSOLE_PENDING_HOOK_REVIEW_KIND = "hook_review"` to the existing string-kind model. Relocate `HookReviewResult` unchanged to the lightweight Chat module and reexport it from its existing UI location. Define a frozen `ConsoleHookReviewProjection` containing review ID, owning session, request generation, current HookReviewSnapshot, waiting-for-Send flag, busy state and an opaque presentation token; hide snapshot/token in repr. It contains no executable callback or permission authority.

The resident host provides these named methods:

```python
begin_hook_review(session_id: str, review_id: str, generation: int,
                  snapshot: HookReviewSnapshot, *, waiting_for_send: bool,
                  owner: HookPermissions,
                  loop: asyncio.AbstractEventLoop) -> asyncio.Future[HookReviewResult]
resolve_hook_review(review_id: str, generation: int,
                    result: HookReviewResult, *, presentation_token: object) -> bool
retire_hook_review(review_id: str, generation: int) -> bool
claim_hook_review_presentation(review_id: str, generation: int,
                               attachment_generation: int) -> ConsoleHookReviewProjection | None
release_hook_review_presentation(review_id: str, generation: int,
                                 presentation_token: object) -> bool
release_hook_review_attachment(attachment_generation: int) -> None
begin_hook_review_operation(review_id: str, generation: int,
                            presentation_token: object, purpose: str) -> HookReviewOperation | None
finish_hook_review_operation(operation: HookReviewOperation,
                             snapshot: HookReviewSnapshot | None, *,
                             error: BaseException | None = None) -> bool
hook_review_retirements(session_id: str | None = None) -> tuple[asyncio.Future[None], ...]
cancel_hook_reviews(session_id: str | None = None) -> None
```

`begin` retains the existing session identity/weak reference, conversation-binding revision and ephemeral state, and checks current store membership at admission/action/Ready. It checks source owner and Close/dispose fencing before atomically publishing one state into the existing `registries["hook_review"]` and payload FIFO. Close writes its generation under this same host lock. Store the private answer Future/loop, owner, snapshot, generation and issued-operation handles on that one record; do not mirror them in a runtime request registry. Duplicate IDs cannot replace an existing record. Settle futures and call projection/attention callbacks only after releasing the non-reentrant lock. `retire` removes only that exact terminal record after all issued permission work has physically retired; late cleanup cannot remove a successor.

The operation purpose is a finite literal: approve/revoke/disable/recover/reset/verify. One in-flight operation is allowed on a record, including the final ready verification. The opaque operation retains its exact owner, record identity, original native Future/outcome and an always-settled retirement completion. Successful reservation is the finite operation issuance boundary: an explicitly requested consent change may finish after navigation/Cancel; its result cannot resume a stale Send. Reserve it before executor submission; submission failure must finish that reservation too. The operation retains its driver Task, which holds the original native Future until physical outcome; no separate runtime registry is added. Do not expose host registries to runtime integration. Finish updates only the matching record, clears busy only after physical completion and autonomously retires a terminal record; retirement never depends on a requester still awaiting. A remount projects busy and cannot issue another action or Ready while work is outstanding. Release of a presentation invalidates that exact token without cancelling the review. Native outcomes can update consent state after Cancel, but cannot turn a terminal request into ready.

Extend only the relevant mixed-decision payload/state scans, exact-state lookup, hidden fixed-copy notice, cancellation and accounting to the new kind. Do not route hook review through MCP approval/question authority. Do not use blocking `run_round` for an asyncio human wait or introduce a generic asynchronous-round engine.

Runtime facade:

```python
async def request_initial_hook_review(
    session_id: str, request_id: str, generation: int,
    snapshot: HookReviewSnapshot, *, waiting_for_send: bool = True,
) -> HookReviewResult

async def apply_hook_review_action(
    review_id: str, generation: int,
    action: Literal["approve", "revoke", "disable", "recover", "reset"],
    expected: HookReviewSnapshot, keys: tuple[str, ...] = (),
    *, presentation_token: object,
) -> HookReviewSnapshot

def resolve_initial_hook_review(
    review_id: str, generation: int, result: HookReviewResult,
    *, presentation_token: object,
) -> bool
```

The finite action dispatcher uses explicit branches to the current captured owner's `approve`, `revoke`, `disable`, `recover` or `reset_invalid_state`; never reflected method lookup. Revoke/disable require exactly one key, recover/reset no keys. Existing expected-snapshot checks remain in HookPermissions. Keep callback lookup live at action time and pass the original expected snapshot/keys unchanged. Source-owner displacement refuses the exact request; same-source revision conflicts use the existing fresh-review/error presentation, without automatically producing ready.

Before issuing any action, validate its exact current presentation token, request/generation, selected session and FIFO head. The UI cannot supply another request's authority by copying a snapshot. Register the operation with that resident record before scheduling it; an eager task factory must not run untracked work. A private executor Future retains the finite native callback and actual outcome through repeated cancellation. Modal cancellation cannot cancel that Future. No native lease remains after an action returns or across the human wait. A UI ready result is only intent: reserve a verify operation, run an original fresh owner snapshot, then recheck exact owner/session/presentation before settling the authoritative answer ready. A revoked or changed grant refuses continuation; answer settlement never precedes that verification. Runtime disposal/Close seals reviews, drains these exact issued operations before releasing their physical resource creators, and then retires the records. HookPermissions.close remains its launch seal; do not weaken it or delay sealing launches merely to wait for workers.

The request facade awaits `asyncio.shield(answer)`. Cancellation of a UI worker/request waiter propagates to that waiter without cancelling the authoritative answer or review. An abandoned waiter can never dispatch; the review remains available until explicit Cancel/Settings/Stop/Close/dispose or a completed ready verification. There is no waiter reattachment or second continuation. Completing consent after waiter loss retires the review; the user sends again. A surviving old caller still passes the existing generation/session/stash fences. Stop and destructive session Close cancel only that session's reviews; disposal cancels all. Session Close never closes the shared HookPermissions owner used by another session.

## Disposable presentation

The runtime's existing pending-decision projector handles `hook_review` through a named modal projector before the ordinary TaskResumeState card route. Opening/reprojecting a modal is synchronous and does not await it on an app or screen message pump. The modal uses its own answer Future, not a push_screen callback whose requester may be waiting.

A modal presentation claims the existing `set_decision_view` API with an opaque token, never the modal/Screen as the retained owner key. Include the exact review ID and request generation. The Console is suspended while its modal is topmost, so presentation validity uses the current modal token and view-attachment generation, not `has_answerable_view()` alone. On navigation/unmount, release the token and projection without settling the runtime review. On explicit Cancel, Escape or Settings, settle only the matching review; Settings retains its existing navigation effect. Superseded modal actions and answers are rejected. A newly reconciled view remounts the same current pending snapshot; it cannot resume an old dispatch closure.

Keep `request_console_hooks_review(screen, snapshot, waiting, cancel)` and injected `request_review(snapshot, waiting, cancel)` shapes. The coordinator stores an exact `(session_id, generation)` Send identity at dispatch entry and clears only that identity at settlement. Expose it with a named read-only `pending_send_identity` property; the stock three-argument adapter reads that original identity before awaiting, never a later foreground session. The stock waiting branch captures runtime/request identity and awaits the runtime facade; manual waiting=False retains `request_hook_review`. Existing UI caller generation/stash checks remain until received-intent migration. Do not claim the whole current Send caller is screen-free merely because the runtime bridge is.

## Review focus and tests

1. **Detached/replaced view:** original view is collectible during a direct runtime review; a fresh view sees the same review. Old tokens cannot approve, mutate, cancel or clear it. Cancel the original runtime waiter separately from the action worker, remount, and prove the resident answer remains uncancelled while the old Send never dispatches.
2. **Ready followed by revocation:** use real HookPermissions, revoke after the displayed answer and before the required fresh snapshot; no continuation or native launch occurs.
3. **Native consent held:** hold the original native read/write while its actual operations/leases are live, repeatedly cancel/Close/dispose, and prove ownership persists until the actual result retires. Preserve a completed durable grant; do not invent rollback.
4. **Eager scheduling and pump input:** registration precedes scheduling, actual mounted review accepts input, and no executor worker is parked for the human wait. Repeated Send/request IDs do not create two continuations; duplicate begin rejects rather than joining the answer. Remount during an issued action retains busy and rejects overlapping action/Ready. Executor submission failure retires its reserved operation.
5. **Mixed FIFO and exact cleanup:** an earlier pending decision remains head, hidden hook review stays retained, cancellation affects only its session, and late cleanup cannot remove a newer generation.

Use the existing real `hook_file` fixture with `pytest.mark.bootstrap_profile`; do not redirect config after import. `_OriginalVisit` in `Tests/UI/test_console_hook_refresh_lifetime.py` can hold original HookPermissions snapshot/config work and prove native leases. For consent writes, observe the original operation inside its owned scope rather than replacing approve/to_thread with a synthetic sleep. Preserve original config/permission and callback assertions.

## Implementation and verification

- [x] Baseline unchanged callback handoff/cancellation, real stale consent and held native snapshot controls plus an actual mounted review flow on c00e010424. Confirm mounted fixture readiness before interpreting any result.
- [x] Independent plan review resolves identity, FIFO, owner replacement, modal unmount and issued-operation lifetime before product changes.
- [x] Verification wrote focused runtime-host and mounted controls. Meaningful RED on unchanged c00e reached the actual stock modal, then failed the existing pending_decision_projection(None) assertion. Thirteen original controls passed; two unchanged worker-route controls hit their existing five-second wait-for-modal bound before modal assertions. No deadline was changed.
- [x] Shared lane adds the named host lifecycle; integration wires runtime and the existing modal. Only the reviewed interface crosses ownership lanes. Preserve legacy/custom routes and remove duplicate migrated ownership.
- [x] Root runs new controls and affected original review/host/consent/lifecycle controls after both implementations are ready, then scoped lint/format/diff and source-current retirement audit. No deadline widening or weakened cleanup.
- [x] Independent final review, task/report updates and local commit. Record remaining received-admission/promotion and measured responsiveness work; no whole-goal completion claim.

## Self-review

This is one independently testable lifecycle prerequisite under the accepted architecture. It changes initial-review ownership without moving durable dispatch or history and without adding another orchestration framework. Consent state stays in HookPermissions, decision state stays in the resident host, and presentation stays disposable. Actual immediate receipt and under-one-second Send remain separate unqualified goals.

## Plan-review rulings

- Shield the resident answer and retire independently of the waiter. Cancelling a disposable worker does not cancel the request; completion after waiter loss grants no old-draft dispatch. This closes the authoritative-Future cancellation gap without a second continuation owner.
- Use explicit atomic presentation and operation transitions on the same resident record. Ready verification is an operation, and projected busy survives remount. This closes the untracked overlap and early-record-retirement gap; no second runtime ledger or generic async-round framework is introduced.


## Final integration refinements

Integrated tests exposed three native issuance edges: all helper Tasks cancelled while an original consent write remains live, an executor submission that enqueues before raising, and a driver cancelled before its first step. Retain a private executor Future with copied context and a submission-success seal; a driver done callback retires an unstarted reservation. Capture the issuing host before reservation and pass it through every native and completion path, so controller replacement cannot target a successor host. These implement the existing finite-operation contract; they do not add an owner.

Keep the new 17 hook-review host methods and operation record in `Chat/console_hook_review_host.py` as a stateless `InitialHookReviewMixin`. `InterruptRoundHost` remains the single instance, lock, registry, FIFO and accounting owner. Preserve all relocated method ASTs and the agreed host API. This packaging refinement avoids placing approximately 500 new lines into the already ratcheted general host; it creates no service or second registry and does not claim to resolve the pre-existing module-size baseline. Recheck the focused import/integration boundary after relocation.


## Started-then-raising task-factory review correction

A post-commit review reproduced a configured factory constructing an eager stdlib Task, issuing the original consent write, and then raising before returning its handle. Closing that entered coroutine incorrectly signalled retirement while the native write remained live. The held-original-body regression fails on ed3477fc16. Construct this private finite driver directly with the lazy stdlib Task constructor so a configurable factory cannot partially start it and lose its handle. Keep resident registration before scheduling, pre-entry cancellation cleanup, the private native Future and exact issuing-host completion unchanged. Run the held-body regression and affected runtime/mounted controls sequentially, then source review and a follow-up commit before integration. No early receipt or timing claim.


## Remount token review correction

The follow-up integrated run exposed an actual mounted Cancel failure. Textual posts ScreenResume before the old modal finishes Unmount; the new modal could reuse its still-registered token, then lose authority when old Unmount released it. A deterministic original-projector test reproduced identical tokens before correction. Add keyword-only replace_existing=False to the host presentation claim; only new-modal creation requests replacement. Same-modal refresh and covered-modal projection preserve tokens. Verify late old-modal cleanup cannot release or answer the replacement, and rerun the original mounted assertions without deadline changes.
