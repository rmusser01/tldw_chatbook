"""Continuous Workflows form; canonical documents stay with the draft service.

THESIS: select an ordered step and author it without losing unfinished work.
OWN-WORLD: Operate/Textual, semantic panel/field/focus tokens and native controls.
STORY: library → navigator → continuous editable form, with durable status pinned.
FIRST VIEWPORT: selected workflow/step above Inputs, Action, Outputs, Execution,
Advanced; Action open and missing Inputs revealed. Neighbor rows select steps.
FORM: approved spatial contract; no concept roll or replacement identity.
FINISH: actual 160x48/110x36/60x20 captures and independent controller review.
"""

from rich.text import Text
from textual import events, on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Static, TextArea
from textual.widgets.collapsible import CollapsibleTitle

from tldw_chatbook.Widgets.form_components import create_form_field
from tldw_chatbook.Workflows.catalog import FIELDS, FieldSpec, output_types
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.models import Draft, FieldEdit, Issue

from .controller import step_label
from .library import compact_button


class WorkflowEditor(Vertical):
    """A region renders one Draft and emits edits; never writes a database."""

    class FieldEdited(Message):
        def __init__(
            self,
            pointer: str,
            text: str,
            as_json: bool,
            identity: tuple[str, str],
            step_id: str | None = None,
        ):
            super().__init__()
            self.pointer, self.text, self.as_json, self.identity = (
                pointer,
                text,
                as_json,
                identity,
            )
            self.step_id = step_id

    class RawEdited(Message):
        def __init__(self, text: str, identity: tuple[str, str]):
            super().__init__()
            self.text = text
            self.identity = identity

    class Selected(Message):
        def __init__(self, section: str):
            super().__init__()
            self.section = section

    class ReferenceRequested(Message):
        def __init__(self, pointer: str, value_type: str, opener: Widget):
            super().__init__()
            self.pointer, self.value_type, self.opener = pointer, value_type, opener

    def __init__(self, documents: DocumentService, **kwargs):
        super().__init__(**kwargs)
        self.documents = documents
        self.draft: Draft | None = None
        self.document = {"steps": []}
        self.section = "overview"
        self.read_only = False
        self.field_edit: FieldEdit | None = None
        self.field_bindings: dict[str, tuple[str, FieldSpec]] = {}
        self.views: dict[str, dict] = {}
        self._building = False
        self._expanded: TextArea | None = None
        self._raw_only_reason: str | None = None

    def compose(self):
        heading = Static(
            "Select a workflow", id="workflow-editor-heading", markup=False
        )
        heading.can_focus = True
        yield heading
        with VerticalScroll(id="workflow-form"):
            yield Vertical(id="workflow-general")
            yield Vertical(id="workflow-step-form")
            with Collapsible(
                title="Advanced · canonical JSON",
                collapsed=True,
                id="workflow-section-advanced",
            ):
                yield Static(
                    "Preserves unknown configuration and metadata. Invalid text is recoverable; forms show the last valid structure.",
                    markup=False,
                    classes="workflow-help",
                )
                yield TextArea(
                    id="workflow-raw-json", soft_wrap=True, show_line_numbers=True
                )
                yield compact_button("Expand JSON editor", "workflow-expand-json")
                yield Static(
                    "",
                    id="workflow-raw-repair-help",
                    markup=False,
                    classes="workflow-help",
                )
                yield compact_button(
                    "Accept repaired whole document…", "workflow-repair-raw"
                )
        yield Static("Scroll form for more", id="workflow-scroll-hint")

    def capture_view(self):
        if not self.is_mounted:
            return
        focused = self.app.focused
        anchor = (
            {"id": focused.id} if focused and focused in self.walk_children() else {}
        )
        if isinstance(focused, TextArea):
            anchor["selection"] = (focused.selection.start, focused.selection.end)
        elif isinstance(focused, Input):
            anchor["cursor"] = focused.cursor_position
        self.views[self.section] = {
            "scroll": self.query_one("#workflow-form").scroll_y,
            "sections": {item.id: item.collapsed for item in self.query(Collapsible)},
            "focus": anchor,
        }

    def restore_view(self, *, focus=True):
        view = self.views.get(self.section, {})
        focused_when_queued = self.app.focused
        for identifier, collapsed in view.get("sections", {}).items():
            matches = self.query("#" + identifier)
            if matches:
                matches.first(Collapsible).collapsed = collapsed

        def after_layout():
            current_focus = self.app.focused
            if (
                current_focus is not focused_when_queued
                and current_focus is not None
                and current_focus.is_attached
                and self in current_focus.ancestors
            ):
                # A newer field selection owns both focus and its scroll.
                return
            self.query_one("#workflow-form").scroll_to(
                y=view.get("scroll", 0), animate=False, force=True
            )
            if focus:
                self.focus_field(view.get("focus", {}))
            elif self.app.focused in self.walk_children():
                # A changed form can move the persistent raw editor below the
                # saved scroll offset. Keep its existing focus/cursor visible.
                self._scroll_field(self.app.focused)

        self.call_after_refresh(after_layout)

    def focus_field(self, anchor=None):
        anchor = anchor or {}
        matches = self.query("#" + anchor["id"]) if anchor.get("id") else []
        eligible = [
            field
            for field in self.query("Input,TextArea,Button")
            if field.display
            and not field.disabled
            and all(parent.display for parent in field.ancestors)
        ]
        required = [
            field
            for field in eligible
            if field.id in self.field_bindings
            and self.field_bindings[field.id][1].required
        ]
        target = matches.first() if matches else next(iter(required or eligible), None)
        if target:
            target.focus(scroll_visible=False)
            self._scroll_field(target)
            if isinstance(target, TextArea) and "selection" in anchor:
                from textual.document._document import Selection

                target.selection = Selection(*anchor["selection"])
            elif isinstance(target, Input) and "cursor" in anchor:
                target.cursor_position = anchor["cursor"]

    def _scroll_field(self, target: Widget) -> None:
        """Keep the label with a focused short-viewport form control."""
        if not target.is_attached:
            return
        if (
            self.screen.has_class("workflow-short")
            and target.id in self.field_bindings
            and not target.has_class("workflow-expanded")
        ):
            target.parent.scroll_visible(animate=False, top=True)
        else:
            target.scroll_visible(animate=False)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if (
            self.screen.has_class("workflow-short")
            and event.widget.id in self.field_bindings
        ):
            self.call_after_refresh(self._scroll_field, event.widget)

    async def render_draft(
        self,
        draft: Draft,
        *,
        section: str | None = None,
        read_only=False,
        rebuild=True,
        focus=False,
        field_edit: FieldEdit | None = None,
        raw_repair_required: bool = False,
        raw_only_reason: str | None = None,
    ):
        previous_document = self.document
        if raw_only_reason is not None:
            self.document = {"steps": []}
            read_only = True
        elif (
            self.draft is None
            or self._raw_only_reason is not None
            or self.draft.last_valid_json != draft.last_valid_json
        ):
            self.document = self.documents.project(draft.last_valid_json)
        if (
            rebuild
            and self.draft is not None
            and not self._building
            and not focus
            and raw_only_reason is None
            and self._raw_only_reason is None
            and (draft.workflow_id, draft.base_revision_id)
            == (self.draft.workflow_id, self.draft.base_revision_id)
            and section in (None, self.section)
            and read_only == self.read_only
            and [(step["id"], step["type"]) for step in previous_document["steps"]]
            == [(step["id"], step["type"]) for step in self.document["steps"]]
            and all(
                self.documents.projected_field_editable(
                    previous_document, pointer, spec.value_type
                )
                == self.documents.projected_field_editable(
                    self.document, pointer, spec.value_type
                )
                for pointer, spec in self.field_bindings.values()
            )
        ):
            rebuild = False
        if self.draft and rebuild:
            self.capture_view()
        self._raw_only_reason = raw_only_reason
        self.draft, self.read_only = draft, read_only
        self.field_edit = field_edit
        if section is not None:
            self.section = section
        raw = self.query_one("#workflow-raw-json", TextArea)
        raw.read_only = read_only
        self.query_one("#workflow-repair-raw").display = (
            raw_repair_required and not read_only
        )
        help_text = self.query_one("#workflow-raw-repair-help", Static)
        help_text.display = raw_repair_required
        help_text.update(
            "Repair the active field in place, or edit this whole document and explicitly accept it. Typing here leaves all forms protected until acceptance."
            if field_edit
            else "Field origin is unavailable. Inspect and repair the whole raw document, then explicitly accept it. Loading, typing and saving do not adopt these bytes automatically."
        )
        if raw.text != draft.raw_text:
            selection = raw.selection
            with raw.prevent(TextArea.Changed):
                raw.load_text(draft.raw_text)
            raw.selection = selection
        if raw_only_reason is not None:
            self._building = True
            try:
                general = self.query_one("#workflow-general", Vertical)
                await general.remove_children()
                await self.query_one("#workflow-step-form", Vertical).remove_children()
                self.field_bindings = {}
                self.query_one("#workflow-editor-heading", Static).update(
                    "Advanced JSON · raw inspection only"
                )
                await general.mount(
                    Static(raw_only_reason, markup=False, classes="workflow-help")
                )
                self.query_one(
                    "#workflow-section-advanced", Collapsible
                ).collapsed = False
                if focus:
                    raw.focus()
            finally:
                self._building = False
            return
        if not rebuild:
            for identifier, (pointer, spec) in self.field_bindings.items():
                field = self.query_one("#" + identifier)
                field.disabled = (
                    read_only
                    or (draft.error is not None and not self._active_fragment(pointer))
                    or not self.documents.projected_field_editable(
                        self.document, pointer, spec.value_type
                    )
                )
                value = self._field_value(pointer, spec)
                if isinstance(field, Input) and field.value != value:
                    with field.prevent(Input.Changed):
                        field.value = value
                elif isinstance(field, TextArea) and field.text != value:
                    selection = field.selection
                    with field.prevent(TextArea.Changed):
                        field.load_text(value)
                    field.selection = selection
            for button in self.query(
                ".workflow-step-actions Button, .workflow-field Button"
            ):
                if (button.id or "").endswith("-reference"):
                    button.disabled = self.query_one("#" + button.id[:-10]).disabled
                elif not (button.id or "").endswith("-expand"):
                    button.disabled = read_only or bool(draft.error)
            if self.section.startswith("step:"):
                index = next(
                    (
                        i
                        for i, step in enumerate(self.document["steps"])
                        if step["id"] == self.section[5:]
                    ),
                    None,
                )
                if index is None:
                    self.section = "overview"
                    return
                self._update_step_heading(index)
                for direction, neighbor_index in (
                    ("Previous", index - 1),
                    ("Next", index + 1),
                ):
                    if 0 <= neighbor_index < len(self.document["steps"]):
                        button = self.query_one(
                            "#workflow-" + direction.lower(), Button
                        )
                        label = f"{direction}: {step_label(self.document['steps'][neighbor_index])}"
                        if button.label.plain != label:
                            button.label = Text(label)
                for section_name in ("inputs", "action", "outputs", "execution"):
                    self.query_one(
                        "#workflow-section-" + section_name, Collapsible
                    ).title = section_name.capitalize() + self._step_summary(
                        index, section_name
                    )
            elif self.section == "overview":
                for index, (button, step) in enumerate(
                    zip(self.query(".workflow-linear-step"), self.document["steps"])
                ):
                    label = Text(
                        f"{index + 1:02}  {step_label(step)} ({step['type']})\n{step['id']} · select to edit"
                    )
                    if button.label != label:
                        button.label = label
            return
        self._building = True
        self.app.capture_mouse(None)
        general = self.query_one("#workflow-general", Vertical)
        step_form = self.query_one("#workflow-step-form", Vertical)
        await general.remove_children()
        await step_form.remove_children()
        self.field_bindings = {}
        if self.section.startswith("step:"):
            step_id = self.section[5:]
            index = next(
                (
                    i
                    for i, step in enumerate(self.document["steps"])
                    if step["id"] == step_id
                ),
                None,
            )
            if index is not None:
                await self._mount_step(step_form, index)
            else:
                self.section = "overview"
        if not self.section.startswith("step:"):
            await self._mount_general(general)
        self._building = False
        self.query_one("#workflow-section-advanced", Collapsible).collapsed = not bool(
            draft.error
        )
        # Compact authoring sections must override the app's roomy collapse chrome.
        for section_widget in self.query(Collapsible):
            section_widget.add_class("workflow-collapse")
            title = section_widget.query_one(CollapsibleTitle)
            title.add_class("workflow-collapse-title")
            section_widget.collapsed_symbol = ">"
            section_widget.expanded_symbol = "v"
        self.restore_view(focus=focus)

    def _field(self, pointer: str, spec: FieldSpec) -> Vertical:
        identifier = f"workflow-field-{len(self.field_bindings)}"
        self.field_bindings[identifier] = (pointer, spec)
        value = self._field_value(pointer, spec)
        editable = self.documents.projected_field_editable(
            self.document, pointer, spec.value_type
        )
        disabled = (
            self.read_only
            or (bool(self.draft.error) and not self._active_fragment(pointer))
            or not editable
        )
        fields = list(
            create_form_field(
                spec.label,
                identifier,
                "textarea" if spec.multiline else "input",
                default_value=value,
                required=spec.required,
                disabled=disabled,
            )
        )
        if not editable:
            fields.append(
                Static(
                    "Unrepresented field shape · inspect Advanced JSON; no automatic conversion.",
                    markup=False,
                    classes="workflow-help",
                )
            )
        # Shared builder supplies Textual-native labels/controls. IDs are local
        # indices; the canonical pointer is data, never generated CSS from IDs.
        if (
            spec.path.startswith("config/")
            and spec.section != "execution"
            and spec.value_type == "string"
        ):
            fields.append(
                compact_button(
                    "Choose value source…", identifier + "-reference", disabled=disabled
                )
            )
        if spec.multiline:
            fields.append(compact_button("Expand text", identifier + "-expand"))
        return Vertical(*fields, classes="workflow-field")

    def _active_fragment(self, pointer):
        return self.field_edit is not None and self.field_edit.pointer == pointer

    def _field_value(self, pointer, spec):
        if self._active_fragment(pointer):
            return self.field_edit.text
        return self.documents.projected_field_text(
            self.document, pointer, as_json=spec.value_type != "string"
        )

    def _update_step_heading(self, index):
        step = self.document["steps"][index]
        self.query_one("#workflow-editor-heading", Static).update(
            f"{index + 1:02} · {step_label(step)} · {step['id']} ({step['type']})"
        )

    def _step_summary(self, index, section):
        if section == "execution":
            retry = (
                self.documents.projected_field_text(
                    self.document, f"/steps/{index}/retry"
                )
                or "unset"
            )
            timeout = (
                self.documents.projected_field_text(
                    self.document, f"/steps/{index}/timeout_seconds"
                )
                or "unset"
            )
            return f" · retry {retry} / timeout {timeout}s"
        missing = any(
            self.documents.projected_field_text(
                self.document, f"/steps/{index}/" + field.path
            )
            in ('""', "", "null")
            for field in FIELDS.get(self.document["steps"][index]["type"], ())
            if field.section == section and field.required
        )
        return " · missing required value" if missing else ""

    async def _mount_step(self, target, index):
        steps = self.document["steps"]
        step = steps[index]
        self._update_step_heading(index)
        neighbors = []
        for direction, neighbor_index in (("Previous", index - 1), ("Next", index + 1)):
            if 0 <= neighbor_index < len(steps):
                button = compact_button(
                    f"{direction}: {step_label(steps[neighbor_index])}",
                    "workflow-" + direction.lower(),
                )
                neighbors.append(button)
        await target.mount(Horizontal(*neighbors, classes="workflow-neighbors"))
        await target.mount(
            self._field(
                f"/steps/{index}/name", FieldSpec("name", "Step name", required=False)
            )
        )
        await target.mount(
            Horizontal(
                *(
                    compact_button(
                        label,
                        identifier,
                        disabled=self.read_only or bool(self.draft.error),
                    )
                    for label, identifier in (
                        ("Insert…", "workflow-insert"),
                        ("Move…", "workflow-move"),
                        ("Delete…", "workflow-delete"),
                    )
                ),
                classes="workflow-step-actions",
            )
        )
        specs = FIELDS.get(step["type"], ())
        for section, title in (
            ("inputs", "Inputs"),
            ("action", "Action"),
            ("outputs", "Outputs"),
            ("execution", "Execution"),
        ):
            content = []
            fields = [field for field in specs if field.section == section]
            if section == "execution":
                fields = [
                    FieldSpec(
                        "retry", "Additional attempts (0–3)", "integer", "execution"
                    ),
                    FieldSpec(
                        "timeout_seconds",
                        "Step timeout (seconds)",
                        "integer",
                        "execution",
                    ),
                ]
            for spec in fields:
                content.append(self._field(f"/steps/{index}/" + spec.path, spec))
            if section == "inputs" and not fields:
                content.append(
                    Static(
                        "Use Choose value source on an Action field for fixed values, workflow inputs, or earlier outputs.",
                        markup=False,
                        classes="workflow-help",
                    )
                )
            if section == "outputs":
                content.append(
                    Static(
                        "Documented result shape · no run selected",
                        markup=False,
                        classes="workflow-help",
                    )
                )
                for name, kind in output_types(step["type"]):
                    content.append(
                        Static(
                            f"{step['id']}.{name} · {kind}"
                            + (
                                " · runtime validation required"
                                if kind == "unverified"
                                else ""
                            ),
                            markup=False,
                        )
                    )
                if not output_types(step["type"]):
                    content.append(
                        Static(
                            "Schema unverified; inspect Advanced JSON.", markup=False
                        )
                    )
            if section == "action" and not specs:
                content.append(
                    Static(
                        "This local operation is unavailable. Configuration remains inspectable in Advanced JSON.",
                        markup=False,
                    )
                )
            summary = self._step_summary(index, section)
            await target.mount(
                Collapsible(
                    *content,
                    title=title + summary,
                    collapsed=section != "action"
                    and not (section == "inputs" and summary),
                    id="workflow-section-" + section,
                )
            )

    async def _mount_general(self, target):
        titles = {
            "overview": "Overview · ordered steps",
            "inputs": "Run inputs · portable defaults",
            "requirements": "Requirements · installation bindings",
            "versions": "Versions · saved definitions",
        }
        self.query_one("#workflow-editor-heading", Static).update(
            titles.get(self.section, "Overview")
        )
        if self.section == "overview":
            await target.mount(self._field("/name", FieldSpec("name", "Workflow name")))
            if not self.document["steps"]:
                await target.mount(
                    Static("No steps yet. Add the first step to begin.", markup=False),
                    compact_button("Add first step", "workflow-add-first"),
                )
            buttons = []
            for index, step in enumerate(self.document["steps"]):
                button = Button(
                    Text(
                        f"{index + 1:02}  {step_label(step)} ({step['type']})\n{step['id']} · select to edit"
                    ),
                    id=f"workflow-overview-{index}",
                    classes="workflow-linear-step",
                    tooltip="Select this step to edit",
                )
                button.add_class("workflow-compact")
                buttons.append(button)
            if buttons:
                await target.mount(*buttons)
        elif self.section == "inputs":
            await target.mount(
                Static(
                    "Defaults belong to the definition. Requirement-owned values are bound explicitly before running; never store credentials here.",
                    markup=False,
                    classes="workflow-help",
                )
            )
            await target.mount(
                self._field(
                    "/inputs",
                    FieldSpec(
                        "inputs", "Run input defaults (JSON)", "object", multiline=True
                    ),
                )
            )
            await target.mount(
                self._field(
                    "/metadata/tldw_workflow/input_schema",
                    FieldSpec(
                        "input_schema",
                        "Declared input schema (JSON)",
                        "object",
                        required=False,
                        multiline=True,
                    ),
                )
            )
        elif self.section == "requirements":
            await target.mount(
                Static(
                    "Local setup is checked by the application before execution. These logical requirements carry no grants or credentials.",
                    markup=False,
                    classes="workflow-help",
                )
            )
            await target.mount(
                self._field(
                    "/metadata/tldw_workflow/requirements",
                    FieldSpec(
                        "requirements",
                        "Portable requirements (JSON)",
                        "object",
                        required=False,
                        multiline=True,
                    ),
                )
            )
        else:
            await target.mount(
                compact_button("Inspect saved revision…", "workflow-versions"),
                compact_button("Return to draft", "workflow-return-draft"),
            )

    def show_step(self, step_id: str) -> None:
        """Request checked navigation; the screen/controller flush before applying."""
        self.post_message(self.Selected("step:" + step_id))

    def show_issue(self, issue: Issue) -> None:
        """Explicit activation only; passive validation never calls this method."""
        if not issue.pointer:
            self.query_one("#workflow-section-advanced", Collapsible).collapsed = False
            target = self.query_one("#workflow-raw-json", TextArea)
        else:
            entry = max(
                (
                    (identifier, pointer)
                    for identifier, (pointer, spec) in self.field_bindings.items()
                    if pointer == issue.pointer
                    or (
                        spec.value_type != "string"
                        and issue.pointer.startswith(pointer + "/")
                    )
                ),
                key=lambda entry: len(entry[1]),
                default=None,
            )
            if not entry:
                self.query_one(
                    "#workflow-section-advanced", Collapsible
                ).collapsed = False
                target = self.query_one("#workflow-raw-json", TextArea)
            else:
                target = self.query_one("#" + entry[0])
                for parent in target.ancestors:
                    if isinstance(parent, Collapsible):
                        parent.collapsed = False
        self.call_after_refresh(
            lambda: (
                target.focus(scroll_visible=False),
                self._scroll_field(target),
            )
        )
        if issue.pointer:
            pointer = issue.pointer
            if target.id != "workflow-raw-json":
                pointer = pointer[len(entry[1]) :]
            raw_text = target.text if isinstance(target, TextArea) else target.value
            location = self.documents.json_location(raw_text, pointer)
            if location and isinstance(target, TextArea):
                from textual.document._document import Selection

                target.selection = Selection(*location)
            elif location and isinstance(target, Input):
                target.cursor_position = location[0][1]

    @on(TextArea.Changed)
    def text_changed(self, event):
        event.stop()
        if event.text_area not in self.query(TextArea):
            return
        if self._building or not self.draft or self.read_only:
            return
        if event.text_area.id == "workflow-raw-json":
            if event.text_area.text != self.draft.raw_text:
                self.post_message(
                    self.RawEdited(
                        event.text_area.text,
                        (self.draft.workflow_id, self.draft.base_revision_id),
                    )
                )
        else:
            self._field_changed(event.text_area.id, event.text_area.text)

    @on(Input.Changed)
    def input_changed(self, event):
        event.stop()
        if event.input not in self.query(Input):
            return
        if not self._building:
            self._field_changed(event.input.id, event.value)

    def _field_changed(self, identifier, text):
        if (
            identifier not in self.field_bindings
            or self.draft is None
            or self.read_only
        ):
            return
        pointer, spec = self.field_bindings[identifier]
        if self.draft.error and not self._active_fragment(pointer):
            return
        if not self.documents.projected_field_editable(
            self.document, pointer, spec.value_type
        ):
            return
        as_json = spec.value_type != "string"
        if text != self._field_value(pointer, spec):
            self.post_message(
                self.FieldEdited(
                    pointer,
                    text,
                    as_json,
                    (self.draft.workflow_id, self.draft.base_revision_id),
                    self.section[5:] if self.section.startswith("step:") else None,
                )
            )

    @on(Button.Pressed)
    def button_pressed(self, event):
        identifier = event.button.id or ""
        if identifier.startswith("workflow-overview-"):
            event.stop()
            self.show_step(
                self.document["steps"][int(identifier.rsplit("-", 1)[1])]["id"]
            )
        elif identifier in ("workflow-previous", "workflow-next"):
            event.stop()
            index = next(
                i
                for i, step in enumerate(self.document["steps"])
                if "step:" + step["id"] == self.section
            )
            self.show_step(
                self.document["steps"][
                    index + (-1 if identifier.endswith("previous") else 1)
                ]["id"]
            )
        elif (
            identifier.endswith("-reference")
            and identifier[:-10] in self.field_bindings
        ):
            event.stop()
            pointer, spec = self.field_bindings[identifier[:-10]]
            self.post_message(
                self.ReferenceRequested(
                    pointer, spec.value_type, self.query_one("#" + identifier[:-10])
                )
            )
        elif identifier == "workflow-expand-json" or identifier.endswith("-expand"):
            field_id = (
                "workflow-raw-json"
                if identifier == "workflow-expand-json"
                else identifier[:-7]
            )
            if self.query("#" + field_id):
                event.stop()
                field = self.query_one("#" + field_id, TextArea)
                field.toggle_class("workflow-expanded")
                self._expanded = field if field.has_class("workflow-expanded") else None
                field.focus(scroll_visible=False)
                self._scroll_field(field)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and self._expanded and self._expanded.is_attached:
            event.stop()
            field, self._expanded = self._expanded, None
            field.remove_class("workflow-expanded")
            field.focus(scroll_visible=False)
            self._scroll_field(field)
