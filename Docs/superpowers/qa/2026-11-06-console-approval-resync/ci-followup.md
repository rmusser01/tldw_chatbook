# Hosted CI follow-up

Fresh CI on `7cfdb2bd690c526f7b4bf59f4d7a66e7234f685e` found three branch regressions: a project-chat UAT assertion expected the old technical label, eager application imports exceeded the boot budget, and six approval helpers loaded before `_ui_ready`.

The repair checks the visible List directory action and original captured identity, defers approval-only data and Details to first use, and keeps the content-free feedback Message in the existing approval widget with an identical compatibility alias. Grant writers retain their original return/exception and policy behavior. A locked store initializer preserves one instance and bridge association.

Deferring Details exposed an actual restoration race. A page delivered before TextArea layout could retain virtual height 1 and clamp saved scroll 3 to 0. A deterministic regression failed first. The product fix reloads the same bounded page through public `load_text` only when the original load had zero width and the existing fully fenced restoration callback has usable width. No worker release, guard or timing threshold is changed.

## Verification

- Wrong-family UAT: 1 passed, preserving actual project, request, execution and evidence assertions.
- Feedback / UI feedback / observations / real relay / scope: 32 / 16 / 34 / 6 / 14 passed.
- Final lifecycle and currentness / Details / privacy: 25 / 24 / 10 passed.
- Actual mounted MessagePump dispatch passed, with an independent missing-handler negative.
- Exact-dev and final Windows `_ui_ready` both contain 1,038 modules at the unchanged 1,038 limit. All six new approval helper modules are explicitly absent.
- Exact-dev and final Windows import counts are both 688. Both retain the inherited Windows failure of strict `<688`; fresh hosted Linux validation is still required. This record does not turn that platform difference into a pass.
- CI census now preserves all 1,754 preceding members in order and adds the actual MessagePump case, floor 1,755. The first-use store regression is an exact fast-lane target. CI target pins pass 118 cases, the census checker 84, and executable target resolution/nonoverlap 2. Existing limits remain unchanged.
- All 80 changed Python files parse. Ruff comparison with exact dev introduces zero diagnostics (181 current, 186 base). The diagnostic inventory remains unchanged and verifies. Independent delta review found no material finding. No full suite was run.

[Source binding](ci-followup-source-manifest.json) contains the exact tested checkout hashes. Failed and successful original receipts remain under the private `ci-fix-20261107T0232-a702db` and `ci-qualification-enrollment-67e5d0a3` runs until final qualification packaging. The earlier source manifest/receipt archive describe the preceding resync phase.

The owner authorized Playwright/CDP and native capture. Both capture methods have synthetic capability evidence. Product timing and the physical matrix remain open; auto-merge remains off. Product measurements will bind the subsequent local source commit before any further source edits.
