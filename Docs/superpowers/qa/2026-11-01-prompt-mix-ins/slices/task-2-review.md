### Spec Compliance

- ❌ Issues found: the Markdown no-fallback/no-body-logging guarantee is incomplete. A UTF-8-BOM-prefixed Mix-in envelope misses the new discriminator at `tldw_chatbook/Prompt_Management/Prompts_Interop.py:806` and reaches the legacy raw-snippet warning at `:875` (Important I1).
- ✅ The remaining reviewed slice follows its brief: explicit v2-for-Mix-in/v1-for-Prompt-or-Recipe dispatch (`prompt_chatbook_record.py:78`), strict versioned-versus-legacy routing (`:222`), retained exact definition with size/projection checks (`:117-145`), distinct single-line JSON Markdown (`prompt_markdown_export.py:109-113`), known server schema identity (`tldw_api/prompt_chatbook_schemas.py:11`), and source refusal without a guessed endpoint (`prompt_scope_service.py:1920-1943`).
- ⚠️ Cannot verify from the diff: live server behavior, subsequent runtime source-identity checks around capture, future UI availability messaging, or whole-branch ordinary-execution exclusion. These remain the owning slices'/controller's checks. Current server capture refusal makes no claim of supporting revision snapshots.

### Strengths

- `prompt_chatbook_record.py:117-145` validates a detached copy but returns the original accepted definition string, preserving exact whitespace instead of normalizing its storage. It checks raw definition bytes, both request representations, and matching compatibility lanes. Real archive coverage verifies exact text after destination storage (`Tests/Chatbooks/test_chatbook_prompt_round_trip.py:814`).
- `prompt_source_capabilities.py:52,85,198` truthfully separates snapshot reads (local True, server False) and conditional updates (server False). The two public adapters and scope entry point refuse supplied write preconditions before conversion can discard them (`prompt_scope_service.py:292,1988`; `server_prompt_service.py:153`). No new server route appears in the diff.
- `Prompts_DB.py:2828` adds name to the existing UUID/version/definition SELECT, maintaining one transactional snapshot. The rename regression checks stable UUID, current name/version and definition (`Tests/Prompt_Management/test_prompt_scope_service.py:2780`).
- Shared keyword creation/reuse/failure log messages now avoid keyword and exception text (`DB/keyword_store.py:166,198,203,232,541`), and the real Mix-in archive regression exercises a literal body reused as accepted keyword metadata (`Tests/Chatbooks/test_chatbook_prompt_round_trip.py:757`).
- The existing private-profile seam is used rather than weakening product recovery guards (`Tests/Chatbooks/test_chatbook_prompt_round_trip.py:33`; existing `Tests/conftest.py:1137,1300,1684`).

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- **I1 — BOM-prefixed Mix-in Markdown falls through to raw snippet logging.** `tldw_chatbook/Prompt_Management/Prompts_Interop.py:806` uses `content.lstrip().startswith(MIXIN_MARKDOWN_HEADER)`. `str.lstrip()` does not remove U+FEFF. The file import path reads using `encoding="utf-8"` (`:1096`), which retains a UTF-8 BOM. Therefore a Mix-in envelope saved with that common encoding bypasses the strict codec; when no classic TITLE exists, the old parser logs `content[:200]` at `:875`, potentially including literal instruction JSON, then returns an empty result rather than a bounded refusal. This violates the explicit malformed-envelope privacy/no-fallback requirement. Treat an optional BOM consistently before envelope detection and framing validation (either support it or reject it with the bounded record error); add a BOM-prefixed malformed/valid-envelope regression proving no raw snippet and no legacy interpretation. A focused AST-isolated execution of the actual parser body confirmed the strict decoder was never reached, the result was `[]`, and the warning contained the supplied body sentinel. The harness removed only the function-local imports and supplied a capturing logger plus a decoder that fails if called; no package/suite rerun or product edit was required.

#### Minor (Nice to Have)

- `task-2-report.md`, Final GREEN / Exact-BASE attribution: the scoped suite is not entirely green without five deselections, and static checks retain inherited debt. The report explicitly identifies all five unchanged digit-sentinel failures, 515 inherited Ruff diagnostics, and two inherited formatter-debt owners. These are transparent pre-existing concerns, not introduced product defects or a reason to expand this slice. Keep the qualifications in the controller's final evidence; do not summarize the result as an unqualified clean suite/lint pass.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The portability/source implementation is small and consistent with existing owners, and its tests cover meaningful real storage and refusal boundaries. The newly introduced Markdown discriminator still allows an ordinary UTF-8 BOM to bypass those boundaries and expose content through the legacy logger; fix that privacy edge before accepting the task.

### Checks and review boundaries

- Read the supplied 1,747-line diff once in four bounded chunks; read the brief, binding context and implementer's complete report. No git command, subagent, product/index/HEAD change, push, merge, or full suite was used.
- Diff contexts ended mid-function for portable decode, capability validation/normalization, scope save, and Markdown parse/import. Focused continuations were inspected to judge strict legacy dispatch, effective capabilities, write fencing and downstream diagnostics. These were not broad changed-file rereads.
- Named unchanged-code check: inspected `server_prompt_adapter.py` for the claimed already-supported Mix-in conversion path; `normalize_artifact_type` admits Mix-ins and both conversion directions retain artifact identity.
- Named unchanged-code check: inspected existing client `get_prompt` / `update_prompt` (`tldw_api/client.py:12149-12166`) and `list_prompt_versions` (`:13704-13711`) to assess truthful revision/conditional-write guarantees. They expose an unversioned identifier GET, a PUT without expected-version input, and a versions list; refusing capture/fenced writes is justified.
- Named unchanged-code check: inspected the private-profile marker references in `Tests/conftest.py` to assess whether the new suite marker weakens a fence. It is an existing supported private collection-profile seam; no guard is altered. An attempted read of a root `conftest.py` reported that it does not exist.
- Read the relevant source-capability paragraph of the approved design (`Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md:100-106`) to assess selection/write/source-replacement claims; this does not make later runtime identity protection verified.
- Ran only the focused AST-isolated parser check described in I1. It answers a specific uncovered privacy doubt; it does not replace a real import regression.
- Read existing `/private/tmp/mixin-task2-ruff-{current,base}.json` artifacts and compared per-file diagnostic-code/message multiplicities: 515 diagnostics each, equal. No Ruff rerun.
- Cannot independently certify the reported 423-pass run, exact-BASE five-failure reproduction, formatter-diff equality, or git diff --check execution from the diff/report alone; their commands and attribution are documented, not rerun. The code diff leaves those five sentinel assertions untouched. No additional output warning is visible beyond the report's inherited test/static-check debt and auto-gc warning.

Skipped full-suite/live-server/UI testing and independent regeneration of reported test/format evidence. Remaining material risk: I1 can log literal Mix-in content on malformed/BOM Markdown import.
