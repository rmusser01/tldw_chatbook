"""Document/view coordination with no DOM and no workflow execution ownership."""

import asyncio

from tldw_chatbook.Workflows.catalog import FIELDS, discover
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import (
    Draft,
    InvalidDraft,
    Issue,
    Revision,
)


def visible_pane_ids(content_width: int) -> tuple[str, ...]:
    """Eligible panes measured against usable content width, after app chrome."""
    if content_width >= 132:
        return ("workflows-library", "workflows-navigator", "workflows-editor")
    if content_width >= 96:
        return ("workflows-navigator", "workflows-editor")
    return ("workflows-editor",)


class WorkflowsController:
    """Coordinate explicitly injected owners; callbacks/DOM belong to regions."""

    def __init__(self, documents: DocumentService, drafts: DraftSession):
        if drafts.documents is not documents:
            raise ValueError("Workflow document and draft owners must match")
        self.documents = documents
        self.drafts = drafts
        self.workflows: tuple[Revision, ...] = ()
        self.versions: tuple[Revision, ...] = ()
        self.inspection: Revision | None = None
        self.section = "overview"
        self.issues: tuple[Issue, ...] = ()

    @property
    def draft(self) -> Draft | None:
        if self.inspection:
            revision = self.inspection
            return Draft(
                revision.workflow_id,
                revision.revision_id,
                0,
                revision.raw_json,
                revision.raw_json,
                None,
            )
        return self.drafts.current

    async def load(self, selection: tuple[str, str] | None = None) -> None:
        # Settle an accepted recovery before reading current or choosing a base.
        # A removed/cancelled screen is not the lifetime of that transaction.
        if self.drafts.editing_locked:
            prior = self.drafts.current
            await self.drafts.flush()
            if prior and selection == (prior.workflow_id, prior.base_revision_id):
                selection = None
        self.workflows = await asyncio.to_thread(self.documents.list_workflows)
        if selection is None and self.drafts.current:
            selection = (
                self.drafts.current.workflow_id,
                self.drafts.current.base_revision_id,
            )
        if selection is None and self.workflows:
            selection = (self.workflows[0].workflow_id, self.workflows[0].revision_id)
        if selection:
            await self.select_workflow(*selection)

    @property
    def head(self) -> Revision | None:
        """Last explicitly read head for the exact selected workflow."""
        draft = self.drafts.current
        return next(
            (
                item
                for item in self.workflows
                if draft and item.workflow_id == draft.workflow_id
            ),
            None,
        )

    async def select_workflow(self, workflow_id: str, base_revision_id: str) -> None:
        await self.drafts.select(workflow_id, base_revision_id)
        self.inspection = None
        self.section = "overview"
        self.versions = await asyncio.to_thread(
            self.documents.list_revisions, workflow_id
        )
        self.validate()

    async def select_section(self, section: str) -> None:
        if self.drafts.current:
            await self.drafts.flush()
        self.section = section

    async def inspect_revision(self, workflow_id: str, revision_id: str) -> None:
        await self.drafts.flush()
        self.inspection = await asyncio.to_thread(
            self.documents.get_revision, workflow_id, revision_id
        )
        self.section = "overview"

    async def edit_inspected_revision(self) -> None:
        """Historical edits are deliberate and cannot replace a dirty head draft."""
        source = self.inspection
        if source is None:
            return
        await self.drafts.flush()
        heads = await asyncio.to_thread(self.documents.list_workflows)
        head = next(item for item in heads if item.workflow_id == source.workflow_id)
        await self.drafts.copy_revision_to_head(source, head)
        self.inspection = None

    def edit_field(self, pointer: str, text: str, *, as_json: bool = False) -> Draft:
        draft = self.drafts.current
        if self.inspection or draft is None or (draft.error and not as_json):
            raise InvalidDraft(
                "Forms show the last valid structure. Repair raw JSON before editing fields."
            )
        result = (
            self.drafts.update_field(pointer, text)
            if as_json
            else self.drafts.update(
                self.documents.edit_field(draft.raw_text, pointer, text)
            )
        )
        self.validate()
        return result

    def change_steps(self, operation: str, step_id: str = "", **options) -> Draft:
        draft = self.drafts.current
        if self.inspection or draft is None or draft.error:
            raise InvalidDraft("Return to a valid editable draft before changing steps")
        result = self.drafts.update(
            self.documents.edit_steps(draft.raw_text, operation, step_id, **options)
        )
        self.validate()
        return result

    async def create(self, name: str) -> None:
        if not name.strip():
            raise InvalidDraft("Name the workflow before creating it")
        if self.drafts.current:
            await self.drafts.flush()
        # Initial JSON is built by the document owner through the same field seam.
        raw = self.documents.edit_field(
            '{"steps":[],"inputs":{}}', "/name", name.strip()
        )
        revision = await asyncio.to_thread(self.documents.create, raw)
        self.workflows = await asyncio.to_thread(self.documents.list_workflows)
        await self.select_workflow(revision.workflow_id, revision.revision_id)

    def validate(self) -> tuple[Issue, ...]:
        draft = self.draft
        if draft is None:
            self.issues = ()
            return ()
        if draft.error:
            self.issues = (Issue("", "invalid_json", draft.error),)
            return self.issues
        issues = list(self.documents.dependency_issues(draft.last_valid_json))
        document = self.documents.project(draft.last_valid_json)
        if not document["steps"]:
            issues.append(Issue("/steps", "empty", "Add the first step before running"))
        for index, step in enumerate(document["steps"]):
            if step["type"] not in FIELDS:
                issues.append(
                    Issue(
                        f"/steps/{index}",
                        "unavailable",
                        "Local operation unavailable; preserved for inspection",
                    )
                )
            for field in FIELDS.get(step["type"], ()):
                pointer = f"/steps/{index}/" + field.path
                if field.required and self.documents.field_text(
                    draft.last_valid_json, pointer
                ) in ("", '""', "null"):
                    issues.append(
                        Issue(pointer, "required", field.label + " is required")
                    )
        self.issues = tuple(issues)
        return self.issues


def step_label(step: dict) -> str:
    """Readable label independent of the immutable step ID."""
    entry = next((item for item in discover() if item.step_type == step["type"]), None)
    return str(step.get("name") or (entry.label if entry else step["type"]))
