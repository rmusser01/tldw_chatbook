# Independent prompt mix-in PR review

Review date: 2026-11-01.
Base: `2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5` (current dev).
Product candidate: `5dac2bb0207285eca67d571565a3ee88113e4a84`.
Evidence follow-up inspected: `963fe12e87ae4d2fb415bff3814943ca448cb0e0` (product/tests unchanged).
PR: https://github.com/rmusser01/tldw_chatbook/pull/3346

This is a fresh feature-and-port review under requesting-code-review/SKILL.md and code-reviewer.md, not a replay of the old review verdict. No helper agents, Qodo, PR comments, checkout/index/HEAD edits, or broad suite runs were used. Review proceeded in bounded passes through the spec/plan/ADR, source diff and connected owners; historical raw ANSI/SVG/log dumps were not exhaustively read.

## Strengths

- Dedicated schema-v3 literal definitions and portable record-v2 discriminate mix-ins from ordinary Prompt/Recipe flows. Normal insertion/improvement paths retain explicit type checks; current server capture is refused without fallback to Local.
- Current source identity and revision fences survive the important preparation awaits. Queued and Retry inputs retain immutable definitions. Composition uses the exact current native user anchor, strips private bookkeeping at serialization, and keeps canonical transcript/history clean.
- Operational receipts are owned separately from optional traces. Fresh writes require real user/assistant anchors; fork-owned history can retain detached predecessors without acquiring new execution proof. Same-row marker transitions and replacement-result proof check exact owners rather than display counts.
- Migration allocation is correctly based on current dev: ChaChaNotes 85→86 and Prompts 5→6. Real historical Canvas/portrait/region/archive catalogs remain distinct. Recovery allowlists and packaging resources cover the new migration rather than renaming the old experimental feature schema as a released historical database.
- Anthropic clean-history markers bind to exact converted input and use only an available breakpoint under existing opt-in/kill/TTL policy. OpenAI behavior remains automatic; no speculative cache API fields or savings claims were introduced.
- The evidence follow-up distinguishes current port checks from old live captures, reports missing historical objects as UNAVAILABLE, and records a fresh shallow clone check without object borrowing. Baseline failures, the broad-run FD warning, and absent cloud/cache/OS-font qualification are disclosed rather than counted green.

## Issues

### Critical

None found.

### Important — I1: Refused checkbox toggles leave a false enabled state on screen

**Locations:** `tldw_chatbook/Widgets/Console/console_prompts_modal.py:451–455` and `:503–509`; originating control at `tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py:197–208`.

A Textual checkbox changes its value before it emits the selection intent. If the Library editor is dirty, `_mixins_intent` displays the dirty guard and returns without applying the intent or restoring the checkbox. Choosing **Keep editing** only hides that guard. The checkbox therefore remains unchecked while the chat preference still says enabled. The exception branch in `_apply_mixin_intent` has the same missing reconciliation when an action fails: it only changes status text.

**Reproduction:** Open a Workbench with Style enabled; Inspect/Edit it; modify Instructions without saving; focus its enabled checkbox and press Space; choose Keep editing. A mounted production-Workbench diagnostic using the existing `runtime`, `WorkbenchHarness`, `seed`, `add`, and `ready` helpers reproduced `checkbox=False actual_enabled=True`. The diagnostic's equality assertion failed in 3.04 seconds. No product method was replaced; the extra diagnostic callable was injected in memory into one selected pytest node solely to use the existing fixture and profile isolation.

**Impact:** The user sees the mix-in as off although a later send will still inject it. This breaks the feature's central reliable-off promise. The enabled count and checkbox can also contradict each other.

**Smallest complete fix:** Restore or reproject the affected checkbox from the owning preferences on every refusal/error path without discarding the editor draft. Alternatively disable selection controls while dirty, while still reconciling failed async actions. Avoid emitting a second mutation while restoring the checkbox. Cover Keep editing and save/discard transitions plus a failed source/stale-intent toggle with a mounted regression; assert both rendered control value and actual preference.

### Minor — M1: Pre-admission capture records a revision as used even when Send is refused

**Locations:** `tldw_chatbook/Chat/console_prompt_mixin_service.py:453–464`; `tldw_chatbook/Chat/console_prompt_mixins.py:384–389`; call ordering at `tldw_chatbook/Chat/console_chat_controller.py:9396–9409` (capture precedes hook admission).

`capture()` calls `reconcile_mixin_placements`, which sets `last_used_version` to the captured revision, then immediately publishes/schedules that preference value. This happens before hook admission, provider validation, and turn acceptance. A subsequent refusal leaves the changed preferences behind even though no attempt was accepted.

**Reproduction:** With the existing native `trace_harness`, select a fresh mix-in, then have `hook_admission_reason` return a refusal. `submit_draft` returned `accepted=False`, `provider_started=False`, and no user/assistant message IDs; the adapter-entry list remained empty. Nevertheless `last_used_version` changed from `None` to `1`. A narrow in-memory diagnostic assertion comparing preferences before and after the refused send failed in 1.12 seconds. The test used the existing source/service setup and native controller; only the hook refusal was supplied through the established monkeypatch fixture.

**Impact:** The UI's “updated since last send” indication can disappear after an entirely refused send. A revision is represented as used when it was never accepted. The same premature publication can change observed placement/group ordering during a refused preparation, contrary to the explicit preservation requirement. This does not change the captured body of an accepted attempt, so severity is Minor rather than Important.

**Smallest complete fix:** Keep capture reconciliation provisional through admission, and publish the used-revision facts only when the appropriate manual/queue acceptance succeeds. Preserve read-only Preview and exact queued/Retry snapshots. If observed name/placement refresh is intentionally kept separate, it must not update `last_used_version` before acceptance. Add a native pre-admission refusal regression and verify the eventual successful acceptance still updates the intended revision.

## Verification and scope limits

I inspected the changed artifact/source/portable codecs, native selection and admission owners, queue/replay paths, prepared-request/provider adapters, receipt storage/fork/promotion/trace proof, current migrations/recovery policies, and Console/Library editor and handoff wiring. I also read the current integration report, verifier, fresh-clone qualification, current static-scope record, and relevant test coverage. Reported green runs were not rerun merely to reproduce their counts.

Two new diagnostics above resolve specific unanswered code doubts; they are failures of additional review assertions, not claims that the unchanged original selected test bodies failed. They ran through pytest with the existing bootstrap-profile fixtures, no cache provider, `PYTHONDONTWRITEBYTECODE=1`, and separate private basetemp directories. The first checkbox probe's initial tool yield did not retain a completion result; its repeated completed invocation is the result cited above. No source/test file was written by the reviewer, and the checkout stayed clean.

Existing M4 browser-retirement logging and the unchanged module launcher limitation remain outside this feature review's repair scope. The integration report's demonstrated target-dev failures and unqualified broad-run FD growth remain limitations, not passing results or new findings without causal evidence. The current live check does not qualify a current-port Send or external cache effectiveness; the report states that accurately.

## Recommendations

Fix I1 and M1 at their existing owners, add the narrow regressions described above, and re-review those changes. No new feature, redesign, dependency, global repository cleanup, or additional approval flow is recommended.

## Assessment

**Ready to merge? With fixes.**

The implementation broadly follows the accepted design and current-dev boundaries, with strong storage and source fencing. The false-off checkbox is a user-visible correctness defect, and the premature used-revision publication is a smaller state-accounting defect; both should be corrected before approval.

Skipped full-suite, external model/cache, and whole live-journey reruns; no new claims are made for those boundaries. Until I1 is fixed, an unchecked Workbench control can still correspond to an enabled mix-in.
