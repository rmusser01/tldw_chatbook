"""Lossless B authoring controls with explicit JSON/pending authority."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Select, Static, TextArea

CONTROLS = {
    "lab-max-size": "chunking.config.max_size",
    "lab-overlap": "chunking.config.overlap",
    "lab-language": "chunking.config.language",
    "lab-min-size": "chunking.config.min_chunk_size",
    "lab-preserve": "chunking.config.preserve_sentences",
}


class EditorRegion(VerticalScroll):
    """Emit immutable field deltas; no persistence or execution in widgets."""

    BUNDLED_CSS = """
    EditorRegion { height: 1fr; min-height: 0; padding: 0 1; }
    EditorRegion Static { height: auto; }
    EditorRegion Input, EditorRegion Select { width: 100%; }
    EditorRegion TextArea { height: 12; min-height: 6; }
    EditorRegion Button { min-width: 12; width: auto; }
    """

    class Edited(Message):
        def __init__(self, kind: str, field: str, value: str):
            self.kind, self.field, self.value = kind, field, value
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("Configure B · editable recipe", markup=False)
        yield Static("Name", markup=False)
        yield Input(id="lab-name", placeholder="Name for a reusable template")
        yield Static("Description", markup=False)
        yield Input(id="lab-description")
        yield Static("Tags · comma separated", markup=False)
        yield Input(id="lab-tags")
        yield Static("Method", markup=False)
        yield Select(
            [("Words", "words"), ("Fixed size", "fixed_size")],
            value="words",
            allow_blank=False,
            id="lab-method",
        )
        yield Static("Maximum size · words", id="lab-size-label", markup=False)
        yield Input(id="lab-max-size", placeholder="Effective default: 400")
        yield Static("Overlap · words", id="lab-overlap-label", markup=False)
        yield Input(id="lab-overlap", placeholder="Effective default: 50")
        yield Static("Language", id="lab-language-label", markup=False)
        yield Input(id="lab-language", placeholder="Effective default: en")
        yield Static(
            "Preserve sentences · true / false", id="lab-preserve-label", markup=False
        )
        yield Input(id="lab-preserve", placeholder="Effective default: false")
        yield Static(
            "Minimum chunk size · words", id="lab-min-size-label", markup=False
        )
        yield Input(id="lab-min-size", placeholder="Effective default: 0")
        yield Static(
            "Full JSON · metadata and classifier preserved; classifier is not run",
            markup=False,
        )
        yield TextArea(id="lab-json", soft_wrap=True)
        yield Static("", id="lab-validation", markup=False)
        yield Button("Discard invalid edit", id="lab-discard")
        yield Button("Undo last edit", id="lab-undo")

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        event.stop()
        field = event.input.id or ""
        if field in CONTROLS:
            self.post_message(self.Edited("control", CONTROLS[field], event.value))
        elif field in {"lab-name", "lab-description", "lab-tags"}:
            self.post_message(
                self.Edited("record", field.removeprefix("lab-"), event.value)
            )

    @on(Select.Changed, "#lab-method")
    def method_changed(self, event: Select.Changed) -> None:
        event.stop()
        if isinstance(event.value, str):
            self.post_message(self.Edited("control", "chunking.method", event.value))

    @on(TextArea.Changed, "#lab-json")
    def json_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        self.post_message(self.Edited("json", "", event.text_area.text))

    def present(self, draft: dict, validation: str, document: dict) -> None:
        """Render a prepared draft while suppressing programmatic edit messages."""
        with self.prevent(Input.Changed, Select.Changed, TextArea.Changed):
            fields = draft["record_fields"]
            for field in ("name", "description", "tags"):
                value = fields.get(field, [] if field == "tags" else "")
                if field == "tags":
                    value = fields.get("tags_text", ", ".join(value))
                widget = self.query_one(f"#lab-{field}", Input)
                if widget.value != value:
                    widget.value = value
            chunking = (
                document.get("chunking", {}) if isinstance(document, dict) else {}
            )
            if not isinstance(chunking, dict):
                chunking = {}
            # Invalid-but-parsed JSON remains authored data, not a render crash.
            method = str(chunking.get("method", "words"))
            selector = self.query_one("#lab-method", Select)
            options = [("Words", "words"), ("Fixed size", "fixed_size")]
            if method not in {"words", "fixed_size"}:
                options.append((f"Preserved: {method}", str(method)))
            selector.set_options(options)
            selector.value = str(method)
            config = chunking.get("config", {})
            if not isinstance(config, dict):
                config = {}
            for widget_id, path in CONTROLS.items():
                value = draft["pending_controls"].get(path)
                if value is None:
                    value = config.get(path.rsplit(".", 1)[-1], "")
                    if isinstance(value, bool):
                        value = str(value).lower()
                    value = str(value)
                widget = self.query_one(f"#{widget_id}", Input)
                if widget.value != value:
                    widget.value = value
                widget.disabled = draft["parse_error"] is not None
            units = "characters" if method == "fixed_size" else "words"
            self.query_one("#lab-size-label", Static).update(f"Maximum size · {units}")
            self.query_one("#lab-overlap-label", Static).update(f"Overlap · {units}")
            for suffix in ("language", "preserve", "min-size"):
                self.query_one(f"#lab-{suffix}").display = method == "words"
                self.query_one(f"#lab-{suffix}-label").display = method == "words"
            selector.disabled = draft["parse_error"] is not None
            editor = self.query_one("#lab-json", TextArea)
            editor.read_only = bool(draft["pending_controls"])
            if editor.text != draft["raw_json"]:
                editor.load_text(draft["raw_json"])
        self.query_one("#lab-validation", Static).update(validation)
        self.query_one("#lab-discard", Button).disabled = not (
            draft["parse_error"] or draft["pending_controls"]
        )
