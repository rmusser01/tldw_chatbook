- **I1 — Unhandled validation from Save as new / Duplicate can close the application and lose unsaved work** — **ADDRESSED**. Both strict serializer calls now catch `ValueError` and return before constructing or detaching a copy (`tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:4233`, `:5319`). The shared feedback helper changes only status/conflict feedback and focuses Instructions (`:3993`); raw draft, metadata, dirty/conflict state, selected identity, version and source stay owned by the existing editor. Strict `LiteralMixinDefinition` construction remains unchanged (`tldw_chatbook/Widgets/Prompts/prompt_mixin_editor.py:58`); no placeholder or validation weakening was introduced.
- **Valid recovery/copy custody** — checked. Duplicate constructs exact literal artifact fields before adopting the detached detail, then clears its obsolete raw draft (`tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:4234`, `:4264–4266`), allowing the generated `(copy)` name to survive remount. Save-as-new retains the existing same-name copy flow. The parametrized regression rejects both invalid actions without a message-pump exception and verifies raw name/body/placement, conflict/source/version and unchanged DB row (`Tests/UI/test_library_prompt_mixins.py:822`, `:923–932`); corrected recovery checks canonical schema-v3 kind/body/placement, copy name, author/description/keywords and local source (`:967–999`).

### New Breakage in the Fix Diff

- **None** — no new Critical/Important breakage found in the supplied fix diff.

### Out-of-Scope Observations

- **None** — original whole-slice review and previously qualified Console/provider/OS limitations were not reopened.

### Checks

- Read supplied fix package `review-398e79d50b..f0e1d640c4.diff` for BASE `398e79d50b6a3232d63ae37a8c925ca0e765e36f` → HEAD `f0e1d640c4ae20c6975b5fefe66587dec3694b84`; inspected only named validation/copy boundaries and their immediate state helpers.
- Confirmed report names exact covering tests and RED/GREEN commands; inspected existing `/private/tmp/task7-fix1-red.log` (**2 failed, 20 deselected**, both escaping strict validation) and `/private/tmp/task7-fix1-green.log` (**4 passed, 18 deselected**, invalid copy recovery plus save/conflict/create coverage). No test rerun was needed.
- Compared existing whole-file Ruff JSON diagnostic multisets for both touched Python files: **17 BASE / 17 current; zero introduced**. No git regeneration, product/index/HEAD mutation, subagents or new native capture; feedback uses existing Static surfaces and no layout/style changed.

### Verdict

- **Fix round:** **All findings addressed, no new Critical/Important breakage.**
- **Skipped:** full suites, external providers and OS-attached verification. **Risk:** broader qualification remains bounded by the original Task 7 review; no new fix-specific blocker found.
