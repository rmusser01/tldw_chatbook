# PR3346 UI fixes — frozen implementation handoff

Implemented the confirmed UI findings under TASK-38544 AC18/20/21/22/24. All owned product and test files are frozen; their exact identities are in [frozen-source-proof.json](/private/tmp/mixins-pr3346-ui-fixes/frozen-source-proof.json). No Git state, GitHub state, task files, documentation, QA manifests, size pins, or census files were edited by this agent.

- Library retained-history compatibility and restore confirmation now use the existing supported-update gate for literal Mix-ins and block artifacts. Current artifact type comes from the canonical editor state; confirmations disclose Mix-in/Prompt/Recipe changes. Dirty edits refresh the history gate, and conditional-update authority is checked again when confirmation returns. Existing scope/version/mutation/restore-currentness seams remain in place.
- Browse → Mix-ins now admits its mode transition through the existing screen-owned worker group. Close and Back can process while a definition read is held, and retired completions cannot replace the current mode.
- Explicit original-mix-in inspection explains that the conversation retains exact Retry instructions with Capture Off and that trace purge does not remove them. Bodies retain the original explicit-disclosure route.
- Valid literal Mix-ins project their metadata presentation independently of the saved Basic/Info preference. Author and Keywords remain reachable and editable, including after More → History. This changes presentation only; the ordinary Prompt/Recipe mode preference is retained.
- The Library save coroutine now shares the existing `prompt_save_ack.py` owner with its acknowledgement. The original facade keeps its signature and original public documentation. Its defining globals dictionary supplies all 13 live dependencies to the helper; executable save logic and existing acknowledgement are unchanged. Controller size is **5,065**, below the original **5,100** ceiling; the receiving helper is **548** lines. No pin was raised or source compressed to meet a limit.

The owned product files are `UI/Library_Modules/library_prompts_controller.py`, `UI/Library_Modules/prompt_save_ack.py`, `Widgets/Console/console_prompts_modal.py`, `Widgets/Console/console_conversation_inspector.py`, and `Widgets/Library/library_prompts_canvas.py`. Targeted regressions are in the existing three UI test files named below.

Verification on the frozen bytes:

| Check | Result | Evidence |
|---|---:|---|
| Focused actual restore/type round trips, read-only/dirty/stale gates, defining-module save callbacks, held navigation, Capture-Off inspection, Info metadata | 10 passed / 38.18s | `green-focused-final.log` |
| Full affected `test_library_prompt_mixins.py`, `test_console_prompt_mixins.py`, `test_console_conversation_inspector.py` | 119 passed / 157.52s | `affected-ui.log` |
| Original app save/cancellation/history controls under their required private-profile ownership | 28 passed / 107.57s | `original-app-controls-profile-owned.log` and exact node inventory |
| Original widget mode controls in ordinary invocation | 4 passed before unrelated app-profile setup refusal | `original-controls.log` |
| Library source wiring, diagnostic-path privacy, exact original controller ceiling and anti-slack guards | 113 passed / 4.69s | `architecture-final.log` |
| Earlier original ceiling + anti-slack pair, pins unchanged | 2 passed / 0.39s | `controller-guards.log` |
| Save executable-body AST, public signature/documentation, original acknowledgement AST | Equal to b110 after reversing explicit globals qualification | `save-extraction-proof.json` |
| Ruff comparison across all eight owned files | No new diagnostics; helper and added/modified test code clean | `baseline-lint.json`, `current-lint.json` |
| Whole helper/test formatting, changed source ranges, owned diff whitespace | Passed | Fresh command output; no unrelated formatting changes |

The original app control file does not opt into the collection-bound profile. Its ordinary invocation failed **before app construction/save logic** with `RecoveryRequired('raw_source_selection_changed')`. Under root's explicit authorization, a private in-memory collection plugin added the repository's existing `bootstrap_profile` marker **only to the exact named app cases**. It changed no test callable, assertion, product admission, source file, or timeout. The exact 28 parametrized IDs and mechanism are in [original-app-controls-profile-owned-nodes.json](/private/tmp/mixins-pr3346-ui-fixes/original-app-controls-profile-owned-nodes.json). These results are qualified native body evidence, not an ordinary-CI-green claim for that legacy file.

RED evidence is preserved. The first cross-kind fixture attempted a public Mix-in→Recipe update, which the existing service correctly refuses; that fixture was replaced with permitted Prompt/Recipe→Mix-in histories followed by explicit retained restoration in both directions. The new driver also stopped requesting a redundant second list navigation after Discard and now presses its settled row, then waits for the actual restored body/placement projection. No final content/version assertion or timeout was loosened. The final round trips verify exact body whitespace, Before/After placement, type, metadata, v3/v4, read-only previews, initial/late read-only refusal, dirty refusal and stale scope refusal.

The historical review and initial real-DB/held-route reproduction inventory remain at [review-reproduction-inventory.json](/private/tmp/mixins-pr3346-ui-fixes/review-reproduction-inventory.json) and [the original independent review](/private/tmp/mixins-pr3346-premerge-ui-review.md). The newly confirmed Info-mode probe and failing regression are `info-mode-probe.log` / `metadata-red.log`; its editing/saving regression is now green. Six production-CSS 80×24, 120×40 and 170×48 SVG captures are catalogued with hashes in `visual-capture-inventory.json`.

A broader static guarded-modal census encountered five unrelated undecided modal guards. Read-only analysis of **all** production Python blobs at base88af and headb110 proved that exact five-item result identical; see `modal-guard-baseline.json`. No exemption or source change was made for those pre-PR findings. The stale 43-versus-45 Library field census and all broader screen/governance repairs are root-owned; the final source-wiring run includes root's current census correction.

Ready for independent scoped re-review. All native test processes have ended and the app slot is free. All eight owned product/test hashes still match the frozen identity file after root released final-five isolation; the exact check is in `source-proof-validation.json`. The complete, bounded artifact list is [evidence-inventory.json](/private/tmp/mixins-pr3346-ui-fixes/evidence-inventory.json). `cohort-source-proof.json` is the preserved earlier seven-file/pre-Info identity, not the final proof. Initial diagnostic probe code/output was transient; its observations are preserved in the original review report and reproduction inventory, with no standalone probe script/log claimed.

Not run: full suite, real model/provider inference, or a native PTY tour. Remaining integration, governance, publication, current CI, and merge approval belong to root; this handoff does not claim them complete.
