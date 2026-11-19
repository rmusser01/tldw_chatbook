### Spec Compliance

- **Compliant** for the scoped Task 7 correction/extraction and prior two model-config test corrections. Reviewed immutable `223c4db6d15f0ca9e9e3da844cc81cd1f2a68aa1 → 19f2904496c05682ef20f5f21bd75fc172ed2bae`; package SHA256 `624fefa85c6f6fa1a2e92b7fa7cca8751757d41d44bafa14cdd6378ef63e9892` verified. The complete diff was read once in four consecutive sections. No mutable source, Git command, test replay, or sibling SDD was used.
- **Conditional external requirement:** final combined startup/import/public-navigation qualification and latest-dev ancestry remain Task 8 work. `task-7-final-handoff-proof.json:1` explicitly distinguishes integrated `df2ba424…` from observed `a1571e86…`. This review does not establish publication or merge readiness.

### Strengths

- `tldw_chatbook/UI/Console_Modules/session.py:3569`, `:3580`, `:3596`: the fixed routes now supply the exact intent type, provider, owner and currentness predicate. The shared transaction retains attachment/session fences, exact claim acknowledgment, physical compensation, release recovery and cancellation propagation. No default writer or new retry policy was introduced.
- `tldw_chatbook/UI/Console_Modules/session.py:3596`: preparation and controller synchronization precede the compensation snapshot. The clean FULL draft uses the existing canonical rebaser; configured target URL remains in settings and the verified URL remains in the ephemeral policy. Prompt, character, prefill and context behavior agree with the unchanged dependency contracts and new controls.
- `tldw_chatbook/UI/Console_Modules/wiring.py:1528`, `tldw_chatbook/UI/Console_Modules/session.py:1086`: five explicit late-bound dependencies and live attachment replace screen reach-through. The three moved methods reverse to the separately frozen functional AST. Mount/resume consumers at `tldw_chatbook/UI/Screens/chat_screen.py:16629` and `:23586` register both providers in the existing order; no method aliases hide responsibilities.
- `tldw_chatbook/UI/Console_Modules/wiring.py:140`, `tldw_chatbook/UI/Screens/chat_screen.py:4804`: the hooks modal adapter retains its live screen/permission owner; the worker resolves the current hooks controller at execution time and preserves its exclusive group. HooksController remains unchanged and DOM-free. `Tests/UI/test_console_hooks_review.py:399` directly checks deferred lookup.
- `Tests/UI/test_llamacpp_consumers.py:339`, `:453`, `:478`, `:527`, `:595`: controls witness real adoption, rollback/cancellation, refusal, resume, dropped-sampler rebasing, same-pair/default behavior and preparation order. `Tests/UI/test_console_provider_apply_defaults_flow.py:2015` arms failures only after actual adoption/rollback, retains `calls == 2`, original projections/replay assertions, and adds target/restored-phase witnesses.

### Focused dependency and evidence checks

- **Risk: hidden preparation sync invalidates compensation.** Inspected only the immutable `_ensure_console_chat_controller`, summary builder and context-estimate methods at `chat_screen.py:9747`, `:8205`, `:7967`, supplied in `task-7-immutable-dependency-methods-19f2904496c0.md`. Its verified SHA256 is `654adfa913219250f4c687e25939aad8ca7f975f5b8f2cfa98b2dc414919ee5b`; the receipt maps each snippet to frozen source hashes. These methods confirm why preparation must precede snapshotting. Retained `task-7-safe-evidence/child-0312-pytest.log:1` shows the original failing sync originated in summary preparation; no adoption-entry witness occurred. Its manifest hash matched.
- **Risk: the new rebase carries dirty fields or loses session context.** Inspected immutable `chat_screen.py:2810` and `console_chat_controller.py:14602`. The initial draft has no dirty fields, remembered targets or endpoint edit; the rebaser selects established unseen-target defaults, retains same-pair supported snapshot values, carries prompt/character/prefill and leaves context overrides intact. The adoption layer deliberately restores configured target base URL and marks source user, as required.
- **Risk: extraction changes consumers or dependency patchability.** `task-7-adoption-move-invariants.json:1` records exact reversed ASTs, unchanged nonmoved methods and full test-retarget reversal; `task-7-adoption-move-proof.json:1` and `task-7-final-freeze-map.json:1` record dependency/caller mappings. The retained DI/resume run reports 3 passed. Its carry uses exact Session/test bytes and Session-constructor AST plus the separately qualified hooks move; whole-file equality is not claimed.
- **Risk: test repairs weaken the original contract.** `task-7-final-test-source-proof.json:1` preserves all original assertions in 42 provider-flow, 9 llama-consumer and 8 setup functions. Only the two declared entered-adoption instrumentation functions differ after prologue/consumer reversal. `task-7-docstring-proof.json:1` reverses the sole prose change to the qualified full-module AST. Fifteen original admission prologues remain narrowly scoped.
- **Risk: diagnostic refresh hides a changed sink.** The diff changes only three inventory rows. `task-7-diagnostic-union-proof.json:1` and `task-7-diagnostic-pin-proof.json:1` preserve other diagnostic ASTs, rows and sink topology; four moved warnings interpolate only the two fixed provider labels and retain their revision/channel/exception-category arguments.
- **Risk: prior model-config repairs obscure integration changes.** The appended two-file diff changes formatting and only the existing private-profile import/decorator/request for `Tests/UI/test_console_native_chat_flow.py:1391`. Complete helper reversal and strict formatter AST proofs are retained. The immutable upstream log records pre-mount `raw_source_selection_changed`; the repaired router node reports 1 passed. `task-6-modelconfig-final-preservation.json:1` maps 283 exact incoming paths, five explained overlaps, four function unions with no function changed by both sides, and the two repair paths. Picker/modal/CSS owners remain exact. The 24 historical formatter debts are not waived or presented as newly passing.
- **Qualification checked, not replayed:** retained logs report 40 adoption cases, 20 hooks cases, 3 DI/resume cases and 3 cap/census cases passing. Receipt before/after source maps and HEADs are stable; final-source differences are explicitly covered by the docstring or hooks carry mappings. Fatal/formatter/whitespace, source-global, worker/UI-census and diagnostic logs report success. ChatScreen is 25,188 lines/759 method definitions against unchanged 25,218/759 caps. Selected receipt/log hashes match the frozen map/safe manifest.
- **Failure preservation checked:** `task-7-authority-corrected-red.log:1211` remains 8 failed/11 passed; `task-7-functional-adoption-green.log:211` remains 1 failed/39 passed; `task-7-controller-order-green.log:390` remains 2 failed. Their names were not treated as success. `task-7-final-preservation.json:1` contains 11,572 historical QA rows, all with equal prior/final blobs. Earlier feature/schema/scanner reviews remain carried, not repeated.

### Issues

#### Critical

- None found in the scoped diff.

#### Important

- None found in the scoped diff.

#### Minor

- **M1 — Teardown descriptor growth remains unexplained.** `task-7-hooks-affected.log:4` records `Tests/conftest.py:609` warning that open descriptors grew by 223 (14 → 237; limit 200), despite all 20 hooks cases passing. This prevents describing the qualification output as pristine. The reviewed source does not establish a new load-bearing leak; inherited attribution remains unproven. Preserve this warning and investigate fixture/app resource cleanup separately before claiming a leak fix; do not suppress it or increase its limit.

### Assessment

**Task quality: Approved.** The correction follows the existing provider/session authority and canonical rebase contracts; the extraction retains explicit ownership and late-bound dependencies, with meaningful behavior evidence and exact preservation mappings. Approval is limited to frozen Task 7 and the narrow prior test corrections; Task 8 and normal current-head publication gates remain required.
