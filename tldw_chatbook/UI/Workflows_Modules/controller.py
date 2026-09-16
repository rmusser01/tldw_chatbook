"""Document/view coordination with no DOM and no workflow execution ownership."""

import asyncio
from typing import TypedDict, Unpack

from tldw_chatbook.Workflows.catalog import FIELDS, discover
from tldw_chatbook.Workflows.document_service import PAGE_SIZE, DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import (
    Draft,
    InvalidDraft,
    Issue,
    Revision,
)


class StepEditOptions(TypedDict, total=False):
    """Optional keyword arguments accepted by document structural edits."""

    offset: int
    step_type: str
    before: bool


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
        self.library_rows: tuple[tuple[str, str, str], ...] = ()
        self.library_offset = 0
        self.library_query = ""
        self.library_has_next = False
        self._library_request = (0, "", object())
        self._head: Revision | None = None
        self.inspection: Revision | None = None
        self.section = "overview"
        self.issues: tuple[Issue, ...] = ()

    @property
    def draft(self) -> Draft | None:
        """Return the inspected revision as a draft view, or the current draft.

        Returns:
            A detached historical view, the live draft, or None without a selection.
        """
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
        """Load the first library page and restore an exact draft selection.

        Args:
            selection: Workflow/base-revision IDs to select. When omitted, retain
                the current draft or select the first library entry, if any.

        Raises:
            DraftWriteFailed: Pending draft work cannot settle or selection is locked.
            RevisionConflict: The requested base does not belong to the workflow.
        """
        # Settle an accepted recovery before reading current or choosing a base.
        # A removed/cancelled screen is not the lifetime of that transaction.
        if self.drafts.editing_locked:
            prior = self.drafts.current
            await self.drafts.flush()
            if prior and selection == (prior.workflow_id, prior.base_revision_id):
                selection = None
        await self.load_library()
        if selection is None and self.drafts.current:
            selection = (
                self.drafts.current.workflow_id,
                self.drafts.current.base_revision_id,
            )
        if selection is None and self.library_rows:
            selection = self.library_rows[0][1:]
        if selection:
            await self.select_workflow(*selection)

    @property
    def head(self) -> Revision | None:
        """Last explicitly read head for the exact selected workflow."""
        draft = self.drafts.current
        return (
            self._head
            if draft and self._head and self._head.workflow_id == draft.workflow_id
            else None
        )

    async def load_library(self, offset: int = 0, query: str = "") -> None:
        """Load a bounded library page, publishing only the latest request.

        Args:
            offset: Nonnegative SQLite integer offset within matching workflow heads.
            query: Case-insensitive name search; empty selects all heads.

        Raises:
            ValueError: The offset is not a nonnegative SQLite integer.
            TypeError: The query is not text.
        """
        request = self._library_request = (offset, query, object())
        rows = await asyncio.to_thread(
            self.documents.list_workflow_summaries,
            page_size=PAGE_SIZE + 1,
            offset=offset,
            query=query,
        )
        if request is not self._library_request:
            return
        self.library_rows = rows[:PAGE_SIZE]
        self.library_has_next = len(rows) > PAGE_SIZE
        self.library_offset, self.library_query = offset, query

    async def refresh_head(self) -> None:
        """Read the selected draft's saved head, clearing it without a draft."""
        draft = self.drafts.current
        self._head = (
            await asyncio.to_thread(self.documents.get_head, draft.workflow_id)
            if draft
            else None
        )

    async def refresh_library(self) -> None:
        """Refresh the requested library page and the selected draft's saved head.

        Raises:
            ValueError: The requested page offset is invalid.
            TypeError: The requested search query is not text.
        """
        await self.load_library(*self._library_request[:2])
        await self.refresh_head()

    async def select_workflow(self, workflow_id: str, base_revision_id: str) -> None:
        """Flush pending edits and open an exact draft in the Overview section.

        Args:
            workflow_id: Portable identity of the workflow to open.
            base_revision_id: Saved revision whose local draft should be recovered.

        Raises:
            DraftWriteFailed: The draft owner is locked, closed, or cannot flush.
            RevisionConflict: The base is missing or belongs to another workflow.
        """
        await self.drafts.select(workflow_id, base_revision_id)
        self.inspection = None
        self.section = "overview"
        await self.refresh_head()
        self.validate()

    async def select_section(self, section: str) -> None:
        """Flush the current draft before changing the visible editor section.

        Args:
            section: Overview or step section identifier supplied by the navigator.

        Raises:
            DraftWriteFailed: Pending changes cannot be persisted.
        """
        if self.drafts.current:
            await self.drafts.flush()
        self.section = section

    async def inspect_revision(self, workflow_id: str, revision_id: str) -> None:
        """Flush the live draft and inspect an immutable revision in Overview.

        Args:
            workflow_id: Portable identity of the workflow to inspect.
            revision_id: Exact saved revision to show without replacing the draft.

        Raises:
            DraftWriteFailed: The current draft cannot be flushed.
            RevisionConflict: The revision is missing or belongs to another workflow.
        """
        await self.drafts.flush()
        self.inspection = await asyncio.to_thread(
            self.documents.get_revision, workflow_id, revision_id
        )
        self.section = "overview"

    async def edit_inspected_revision(self) -> None:
        """Copy inspected content to a clean head draft; do nothing without inspection.

        Raises:
            InvalidDraft: The head is unavailable or the source cannot be edited.
            DraftConflict: The head has a local draft that would be overwritten.
            RevisionConflict: The source or target revision changed during the copy.
            DraftWriteFailed: Draft work is locked or persistence fails.
        """
        source = self.inspection
        if source is None:
            return
        await self.drafts.flush()
        await self.refresh_head()
        head = self.head
        if head is None:
            raise InvalidDraft("The workflow's saved head is unavailable")
        await self.drafts.copy_revision_to_head(source, head)
        self.inspection = None

    def edit_field(self, pointer: str, text: str, *, as_json: bool = False) -> Draft:
        """Update a draft field and refresh issues without waiting for persistence.

        Args:
            pointer: JSON pointer of the editable field.
            text: Literal string value or JSON fragment, according to as_json.
            as_json: Parse text as a JSON field edit, retaining incomplete fragments
                for repair with the last valid form projection.

        Returns:
            The updated draft, including any retained field-validation error.

        Raises:
            InvalidDraft: The view, field, or edit is unavailable or protected.
            DraftConflict: The edit's draft provenance or generation is invalid.
            DraftWriteFailed: The draft owner is locked or closed.
        """
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

    def change_steps(
        self, operation: str, step_id: str = "", **options: Unpack[StepEditOptions]
    ) -> Draft:
        """Apply a structural edit to a valid draft and refresh validation issues.

        Args:
            operation: One of add, duplicate, move, or delete.
            step_id: Selected step ID; an absent selection appends an added step.
            **options: Optional offset (int, default 1) for move, step_type (str,
                default prompt) for add, and before (bool, default False) for
                insertion relative to the selected step.

        Returns:
            The updated draft scheduled for local persistence.

        Raises:
            InvalidDraft: The view is not editable or the change breaks structure,
                dependencies, opaque semantics, or authoring limits.
            ValueError: The requested new step type is unavailable for authoring.
            DraftConflict: The next draft generation is invalid.
            DraftWriteFailed: The draft owner is locked or closed.
        """
        draft = self.drafts.current
        if self.inspection or draft is None or draft.error:
            raise InvalidDraft("Return to a valid editable draft before changing steps")
        result = self.drafts.update(
            self.documents.edit_steps(draft.raw_text, operation, step_id, **options)
        )
        self.validate()
        return result

    async def create(self, name: str) -> None:
        """Flush pending edits, create a named empty workflow, and select it.

        Args:
            name: Nonblank workflow name; surrounding whitespace is removed.

        Raises:
            InvalidDraft: The name is blank or the definition exceeds storage bounds.
            DraftWriteFailed: Pending edits cannot be flushed or selection is locked.
            RevisionConflict: The generated portable identity already exists.
        """
        if not name.strip():
            raise InvalidDraft("Name the workflow before creating it")
        if self.drafts.current:
            await self.drafts.flush()
        # Initial JSON is built by the document owner through the same field seam.
        raw = self.documents.edit_field(
            '{"steps":[],"inputs":{}}', "/name", name.strip()
        )
        revision = await asyncio.to_thread(self.documents.create, raw)
        await self.refresh_library()
        await self.select_workflow(revision.workflow_id, revision.revision_id)

    def validate(self) -> tuple[Issue, ...]:
        """Refresh structural and authoring issues for the visible draft or revision.

        Returns:
            The issues also stored in self.issues, or an empty tuple without a
            selection. Invalid JSON is reported as an issue. These checks do not
            establish runtime availability or authorize workflow execution.
        """
        draft = self.draft
        if draft is None:
            self.issues = ()
            return ()
        if draft.error:
            self.issues = (Issue("", "invalid_json", draft.error),)
            return self.issues
        try:
            issues = list(self.documents.dependency_issues(draft.last_valid_json))
            document = self.documents.project(draft.last_valid_json)
        except InvalidDraft as error:
            # Older saved definitions remain readable/exportable as raw text.
            self.issues = (Issue("", "invalid_json", str(error)),)
            return self.issues
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
                if field.required and self.documents.projected_field_text(
                    document, pointer
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
