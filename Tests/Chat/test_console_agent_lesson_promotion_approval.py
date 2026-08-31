"""Exact two-review flow for repository Agent Lesson promotion."""

from __future__ import annotations

import json
from pathlib import Path

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.local_tool_provider import (
    PROMOTION_APPROVAL_REQUIRED,
    PROMOTION_FOREGROUND_REQUIRED,
    PROMOTION_STALE_REFUSAL,
    LocalApprovalEffect,
    LocalToolExposure,
    LocalToolProvider,
    LocalToolSpec,
    RunAdmittedWorkspaceRoot,
)
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionPromotionSnapshot,
    ProjectInstructionResolver,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    PromotionSnapshotRevalidation,
)
from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    use_run_actor,
    use_tool_call_id,
)
from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.local_tool_impls import write_file

RUN = "promotion-run"


def _evidence() -> dict:
    return {
        "lesson_note_ids": ["note-public-1"],
        "summary": "Atomic writes need a current-state check.",
        "provenance": "Observed during a repository instruction edit.",
        "verification": "A focused race test passed.",
        "principle": "Preserve unrelated edits.",
        "rationale": "This narrow rule prevents stale full-file replacement.",
        "procedural": True,
        "reusable": True,
        "independently_verified": True,
        "verification_command": "pytest -q Tests/Tools/test_local_tool_impls.py",
        "verification_text": "The deterministic compare-and-swap test passed.",
    }


def _prepare_args(content: str = "# Updated\n") -> dict:
    return {
        "path": "AGENTS.md",
        "content": content,
        "dry_run": True,
        "promotion": _evidence(),
    }


class _LiveInstructionContext:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.binding_id = "binding-1"
        self.fingerprint = "fingerprint-1"
        self.resolver = ProjectInstructionResolver()

    def snapshot(self, relative_path: str) -> InstructionPromotionSnapshot:
        return self.resolver.snapshot_promotion_target(
            binding_id=self.binding_id,
            binding_root=self.root,
            locator_fingerprint=self.fingerprint,
            target_path=self.root / relative_path,
            activation_revision=0,
        )

    def revalidate(
        self, prepared: InstructionPromotionSnapshot
    ) -> PromotionSnapshotRevalidation:
        if prepared.binding_id != self.binding_id:
            return PromotionSnapshotRevalidation(False, "binding_changed")
        if prepared.locator_fingerprint != self.fingerprint:
            return PromotionSnapshotRevalidation(False, "binding_changed")
        current = self.snapshot(prepared.target_relative_path)
        if (
            current.expected_sha256 != prepared.expected_sha256
            or current.expected_absent != prepared.expected_absent
        ):
            return PromotionSnapshotRevalidation(False, "target_state_changed")
        if current.effective_chain_digest != prepared.effective_chain_digest:
            return PromotionSnapshotRevalidation(False, "effective_chain_changed")
        return PromotionSnapshotRevalidation(True, "eligible")


def _provider(root: Path, context: _LiveInstructionContext) -> LocalToolProvider:
    spec = LocalToolSpec(
        name="fs_write",
        description="Write a complete file.",
        parameters={"type": "object"},
        handler=lambda args: write_file(
            args["path"],
            args["content"],
            workspace_root=root,
            dry_run=args.get("dry_run", False),
            expected_sha256=args.get("expected_sha256"),
            expected_absent=args.get("expected_absent", False),
        ),
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(LocalApprovalEffect.MUTATES_LOCAL,),
        tags=("mutates",),
    )
    return LocalToolProvider(
        workspace_root=root,
        specs=[spec],
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        promotion_snapshotter=context.snapshot,
        promotion_revalidator=context.revalidate,
    )


class _InlineWorkspaceExecutor:
    """Exercise admitted-root routing without a subprocess test harness."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        assert operation == "fs_write"
        return write_file(
            arguments["path"],
            arguments["content"],
            workspace_root=self.root,
            dry_run=arguments.get("dry_run", False),
            expected_sha256=arguments.get("expected_sha256"),
            expected_absent=arguments.get("expected_absent", False),
        )


def _admitted_provider(
    root: Path, context: _LiveInstructionContext
) -> LocalToolProvider:
    authority = RunAdmittedWorkspaceRoot(
        workspace_id="workspace-1",
        binding_id=context.binding_id,
        alias=context.binding_id,
        root=root,
        locator_fingerprint=context.fingerprint,
        root_identity=((str(root), 1, 2, 0o40755),),
        allow_write=True,
        guard=lambda _write: True,
        workspace_executor=_InlineWorkspaceExecutor(root),
    )
    return LocalToolProvider(
        workspace_root=root,
        admitted_roots=(authority,),
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        promotion_snapshotter=context.snapshot,
        promotion_revalidator=context.revalidate,
    )


def _review_and_invoke(
    provider: LocalToolProvider,
    args: dict,
    call_id: str,
    reviewer,
):
    hook = build_local_review_hook(provider, reviewer)
    actor = CurrentRunActor("primary", RUN, None)
    with use_run_actor(actor):
        verdicts = hook([ToolCall("fs_write", args, call_id)], RUN)
        with use_tool_call_id(call_id):
            result = provider.invoke("fs_write", args)
    return verdicts, result


def test_broad_allow_still_requires_exact_prepare_and_apply_reviews(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Old\n")
    context = _LiveInstructionContext(tmp_path)
    provider = _provider(tmp_path, context)
    cards = []

    def approve(rows):
        cards.append(rows)
        assert len(rows) == 1
        assert rows[0].options == ("approve_once", "deny")
        return {rows[0].call_id: "approve_once"}

    _, prepared = _review_and_invoke(
        provider, _prepare_args(), "prepare-call", approve
    )

    assert prepared.ok
    proposal = json.loads(prepared.content)
    assert proposal["binding_id"] == "binding-1"
    assert proposal["target_path"] == "AGENTS.md"
    assert proposal["replacement_content"] == "# Updated\n"
    assert proposal["bounded_diff"]
    assert proposal["evidence"]["lesson_note_ids"] == ["note-public-1"]
    assert target.read_text() == "# Old\n"

    apply_args = {
        "path": proposal["target_path"],
        "content": proposal["replacement_content"],
        "expected_sha256": proposal["expected_sha256"],
        "proposal_digest": proposal["proposal_digest"],
    }
    _, applied = _review_and_invoke(provider, apply_args, "apply-call", approve)

    assert applied.ok
    assert target.read_text() == "# Updated\n"
    assert len(cards) == 2
    assert cards[1][0].arguments["proposal"] == proposal
    with use_run_actor(CurrentRunActor("primary", RUN, None)):
        with use_tool_call_id("apply-call"):
            reused = provider.invoke("fs_write", apply_args)
    assert reused.error == PROMOTION_APPROVAL_REQUIRED


def test_promotion_accepts_and_preserves_latest_dev_root_alias(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Old\n")
    context = _LiveInstructionContext(tmp_path)
    provider = _admitted_provider(tmp_path, context)

    def approve(rows):
        return {rows[0].call_id: "approve_once"}

    prepare_args = {**_prepare_args(), "root_alias": context.binding_id}
    _, prepared = _review_and_invoke(
        provider, prepare_args, "prepare-call", approve
    )

    assert prepared.ok
    proposal = json.loads(prepared.content)
    apply_args = {
        "root_alias": context.binding_id,
        "path": proposal["target_path"],
        "content": proposal["replacement_content"],
        "expected_sha256": proposal["expected_sha256"],
        "proposal_digest": proposal["proposal_digest"],
    }
    _, applied = _review_and_invoke(provider, apply_args, "apply-call", approve)

    assert applied.ok
    assert target.read_text() == "# Updated\n"


def test_target_change_after_preview_refuses_without_overwriting(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Old\n")
    context = _LiveInstructionContext(tmp_path)
    provider = _provider(tmp_path, context)
    def approve(rows):
        return {rows[0].call_id: "approve_once"}
    _, prepared = _review_and_invoke(
        provider, _prepare_args(), "prepare-call", approve
    )
    proposal = json.loads(prepared.content)
    target.write_text("# User edit\n")
    apply_args = {
        "path": "AGENTS.md",
        "content": proposal["replacement_content"],
        "expected_sha256": proposal["expected_sha256"],
        "proposal_digest": proposal["proposal_digest"],
    }

    _, applied = _review_and_invoke(provider, apply_args, "apply-call", approve)

    assert not applied.ok
    assert PROMOTION_STALE_REFUSAL in applied.error
    assert target.read_text() == "# User edit\n"


def test_denied_preparation_retains_no_proposal(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Old\n")
    provider = _provider(tmp_path, _LiveInstructionContext(tmp_path))
    _, result = _review_and_invoke(
        provider,
        _prepare_args(),
        "prepare-call",
        lambda rows: {rows[0].call_id: "deny"},
    )

    assert result.error == PROMOTION_APPROVAL_REQUIRED
    assert provider._promotion_proposals == {}


def test_subagent_cannot_present_or_invoke_promotion(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Old\n")
    provider = _provider(tmp_path, _LiveInstructionContext(tmp_path))
    seen = []
    hook = build_local_review_hook(provider, lambda rows: seen.extend(rows) or {})
    args = _prepare_args()

    with use_run_actor(CurrentRunActor("subagent", "child-1", RUN)):
        assert hook([ToolCall("fs_write", args, "child-call")], "child-1") == {}
        with use_tool_call_id("child-call"):
            result = provider.invoke("fs_write", args)

    assert seen == []
    assert result.error == PROMOTION_FOREGROUND_REQUIRED


def test_unencodable_preparation_does_not_arm_a_review(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Old\n")
    provider = _provider(tmp_path, _LiveInstructionContext(tmp_path))
    seen = []
    hook = build_local_review_hook(provider, lambda rows: seen.extend(rows) or {})
    args = _prepare_args("bad \ud800")

    with use_run_actor(CurrentRunActor("primary", RUN, None)):
        assert hook([ToolCall("fs_write", args, "bad-call")], RUN) == {}

    assert seen == []
