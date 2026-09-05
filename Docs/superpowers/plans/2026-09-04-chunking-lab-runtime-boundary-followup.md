# Chunking Lab Runtime Boundary Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Resolve TASK-31428 AC15's two reproducible architecture-guard failures without changing Lab execution or resource-admission semantics.

**Architecture:** Keep vendored processor and sanitation access inside the existing `template_runtime` seam. The runner continues to own pre-operation admission checks, cumulative payload accounting, process supervision and limits; it calls narrow runtime operations instead of importing vendor types or the private reporting chunker.

**Tech Stack:** Existing Python 3.12 environment, pytest, vendored Python chunker; no dependencies added.

**Spec:** `Docs/superpowers/specs/2026-09-04-chunking-lab-design.md`, runtime ownership under ADR-078 and ADR-118, and TASK-31428 AC15. This is the user-requested continuation after the prior final-pass handoff, not a restart of completed tasks or another broad feature audit.

ADR required: yes (existing decisions apply).
ADR path: `backlog/decisions/078-chunking-template-convergence.md` and `backlog/decisions/118-chunking-lab-local-execution-and-recovery.md`.
Reason: implement the existing runtime boundary; no new ownership or policy.

## Global Constraints

- Do not edit vendored engine files, widen the consumer allowlist, disable guards, or change global parity validation.
- Preserve all current resource ceilings, prescan ordering, metadata accounting, cancellation/reaping and full-pipeline execution semantics.
- No second template mapper, store, process owner, general plugin abstraction or dependency.
- Targeted tests only; no full repository sweep without user opt-in.
- All app-importing tests run beneath `Tests/` with the repository isolation harness; no bare app imports or normal-profile probes.
- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/chunking-lab`; no merge, push or unrelated checkout changes.
- Retain the earlier non-green gate and platform/resource/privacy qualifications as historical evidence.

### Task 1: Put admission-time vendor operations behind the runtime seam

**Files:**
- Modify: `tldw_chatbook/Chunking/template_runtime.py`
- Modify: `tldw_chatbook/Chunking/lab_runner.py`
- Test: `Tests/Chunking/test_template_runtime.py`
- Test: `Tests/Chunking/test_lab_runner.py`
- Controller final bookkeeping: TASK-31428, existing design/plan/ADR status, `Docs/Chunking_Lab_Verification.md`.

**Interfaces:**
- Existing `_child_admission(request: RunRequest, limits: PreviewLimits) -> int` keeps its signature and accounting.
- Runtime provides `run_template_preprocessing_operation(text: str, operation: str, config: dict[str, Any]) -> str | dict[str, Any]` and `sanitize_template_input(text: str) -> str`. These narrow internal-consumer APIs accept already-preflighted operations; they do not add new executable capability or expose processor instances.
- Preserve `registered_template_operations` and existing execution/report APIs.

- [ ] Reproduce the two existing RED guards before changing production:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Chunking/test_template_runtime.py::TestEnumerationGuards::test_exactly_one_flat_mapper_in_production Tests/Chunking/test_template_runtime.py::TestEnumerationGuards::test_the_mapper_guard_can_see_what_it_guards -q --tb=short
```

- [ ] Add RED behavior tests for the two runtime APIs before implementing them. The first must preserve actual string output and operation metadata; the second must match real sanitation without producing a security log. Include a test that would fail if the runner skipped its prescan or stopped using runtime-owned behavior. Use existing admission/resource cases where they already discriminate; avoid fake processor-return tests.

```python
def test_admission_runtime_operation_preserves_text():
    assert tr.run_template_preprocessing_operation(
        "alpha   beta", "normalize_whitespace", {}
    ) == "alpha beta"

def test_admission_runtime_sanitizes_control_characters():
    assert tr.sanitize_template_input("alpha\x00 beta") == "alpha beta"
```

- [ ] Implement the narrow adapters beside `registered_template_operations`, with type hints/docstrings and no vendor re-export:

```python
def run_template_preprocessing_operation(text, operation, config):
    return TemplateProcessor()._operations[operation](text, config)

def sanitize_template_input(text):
    return _ReportingChunker()._sanitize_input(text, suppress_security_log=True)
```

The current processor initializes a stateless operation registry and no chunker until requested. A fresh short-lived processor for each of at most 16 operations avoids introducing a factory/cache/lifetime abstraction. If source inspection shows that changes actual operation behavior, stop and report the evidence rather than assuming equivalence.

- [ ] In `_child_admission`, replace the two private/vendor imports and calls with lazy imports of the runtime adapters. Delete its local processor construction. Leave the accounting loop and every pre-operation check in place:

```python
result = run_template_preprocessing_operation(text, name, config)
sanitized = sanitize_template_input(text)
```

- [ ] Run the amended runtime/runner/preflight/execution/coordinator selection and static/format/whitespace checks. Existing legacy runtime lint exclusions may be retained only with the documented baseline comparison. Record exact command/output and warnings.
- [ ] Commit the scoped code/test change and write an implementation report with cause, RED/GREEN evidence, changed interfaces and limitations. Release the index. Controller gets a fresh independent spec/quality review, then reruns the prior 468-case targeted gate plus new tests on stable code.
- [ ] After independent review and a green targeted gate, controller checks AC15, reconciles documentation status and marks TASK-31428 Done via CLI. No blanket full-suite/cross-platform claim or merge/push.
