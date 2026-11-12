# PR3346 size repair evidence

The authorized ownership moves fit all three existing owner ceilings without changing the pins. Production source is frozen. The only post-freeze change is the root-authorized `bootstrap_profile` marker on the new pure Browse regression; its exact ordinary pytest node now passes without a private plugin or proof injection. Task38544 AC23 and the implementation plan were recorded by root before these edits. The independent integration reviewer preapproved the boundary. No new ADR was needed: these are mechanical moves preserving the existing Library and Console ownership contracts.

| Owner | Before repair | Frozen current | Existing ceiling |
| --- | --- | --- | --- |
| library_prompt_browse_controller.py | 286 lines | 281 lines | 281 lines |
| LibraryScreen | 38266 lines / 1379 methods | 38177 / 1373 | 38242 / 1374 |
| ChatScreen | 24675 lines / 745 methods | 24590 / 741 | 24605 / 741 |

ChatScreen also fits below the already-over-budget base (24652 / 744). The repair does not merely restore that baseline. Root owns ratchet metadata enrollment and any lowering warranted by the measured result; this change did not edit any guard or pin.

The Browse dispatch at line161 now reads the screen's current `run_worker` directly. Its removed private property was a redundant forwarding layer with no external callers. The new regression replaces the worker after controller construction and observes the replacement, unchanged exclusive worker group and selected Mix-in filter.

Five Study handoff derivations moved to the existing screen_helpers.py (starting line305); the sixth removed method only forwarded to `_has_local_sources`. The screen's canvas and action entry points remain. Four Console command history/popup derivations moved to the existing command_handoff.py (starting line220). Screen-owned registry, history, ranking cache, workers and task lifetime remain on the actual screen. Calls pass the original screen module's live globals dict; the nested command-recording coroutine reads its current logger after await. The narrow tests check minimal command token capture without argument logging, live post-await logger replacement, revision-based ranking cache, current registry, Study source IDs/order/titles and current local/server readiness copy.

`ast-reverse-proof.json` proves ten original body/predicate comparisons after reversing only the approved routing/global-lookup changes and docstring indentation. Original bodies are revision-qualified to b1108874cab28d23f660aa22c11c863466a84a8e in original-bodies.json. Retired private-name census found no production/test callers outside the removed definitions and rewritten owner calls. Existing exported support names and the unmoved ingest-options native globals are verified by the support-layer surface tests, including a subprocess import-cycle check.

No new module was created and no aliases or inheritance layers were introduced. Runtime top-level import module multisets for the two screens and two existing helper owners show no additions. Both screen-to-helper imports stay lazy. The first nonempty slash popup can now load command_handoff before the first accepted command; there is no claim here that its latency is unchanged. Root and the UI producer own the serialized mounted-app and final startup/performance qualification. This agent ran no mounted-app or performance cohort.

The recorded targeted results are distinct selections, not one aggregate sweep:

- RED: the exact three ceiling nodes failed before repair (3 failed in 0.66s).
- GREEN: those three ceilings, three associated no-slack nodes and the existing Library support-layer surface file passed (14 passed in 7.10s).
- GREEN: ordinary pytest for the new bootstrap-profile helper file passed (3 passed in 1.42s).
- GREEN: ordinary pytest for the exact new case-only-bootstrap Browse node passed (1 passed in 1.10s). The other 21 old cases and their assertions remain unmodified and unmarked; root can enroll the exact new node without enrolling the legacy file.
- Private paired Browse controls passed 22 cases on the original controller and 22 on the current controller (3 assert-rewrite warnings each). These explicitly add the bootstrap profile in a private collection plugin and are contextual comparison evidence, not ordinary CI qualification for the unmarked legacy file.

Early harness errors are retained and not attributed to the product. Initial unmarked runs of the old Browse file stop in setup at `raw_source_selection_changed`; the combined initial run similarly stopped the new helpers before their profile marker was supplied. An initial baseline probe imported the installed main checkout before putting this worktree first and bootstrapping Tests.conftest, yielding 17 false baseline failures involving an older PromptBrowseScope. Those baseline-browse/baseline-profile-browse logs are invalid product comparisons. The corrected probe bootstraps before product imports, loads the exact git revision and produces the paired22/22 controls. No old assertion was weakened to clear these errors. A reusable lesson for root is that private probes must establish the intended worktree and test bootstrap before importing product modules; this incident produced 17 misattributed failures.

Frozen scoped Ruff has 388 diagnostics, matching baseline 388 with no code/message multiplicity changes across the five changed production files. Moved broad-except diagnostics follow their unchanged bodies. Both test files are lint-clean and formatter-clean. Existing production formatter debt is unchanged in exact diff extent: LibraryScreen 31 input/70 output changed-format lines, ChatScreen 64/168, screen_helpers one preexisting blank-line insertion; Browse and command_handoff are already formatted. No broad reformat was applied. Scoped diff whitespace is clean.

The owned final patch contains only five production files and the two narrow test files. The earlier public legacy export fix is frozen separately under /private/tmp/mixins-pr3346-export-fix and was not changed by this repair. Root's task/docs/CI metadata and the UI producer's files are outside this patch. No Git index, HEAD, branches or hosted review state was changed by this agent.

All exact selected commands and outputs are indexed in commands.json. frozen-source-proof.json is the authoritative current seven-file hash/measurement list; source-proof.json is superseded intermediate evidence. report-files.json inventories every top-level report/probe artifact and explicitly excludes generated pytest directories and redundant before snapshots. The patch includes the previously untracked new helper test. Tests are stopped and production freeze remains in place. Independent source review and root's final native/performance cohort remain the merge decision boundary.
