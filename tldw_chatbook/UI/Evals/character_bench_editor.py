"""Detail-pane content for a selected character-probe bench.

Mirrors ``bench_editor.py``'s ``BenchEditor`` (name/description/sampler
editing, display-only-until-Save, a failed Save renders in-place without
recomposing) but is a genuinely SEPARATE widget class: word benches and
character-probe benches never share a detail surface, per this whole
program's own design decision (a probe/conversation eval reads generated
TEXT, never per-token logprobs -- the two forms have no field, no
vocabulary, and no validation in common). In particular, NOTHING in this
module ever mentions top-K, logprobs, a normalizer, or a canary check --
that vocabulary belongs to ``bench_editor.py``'s word-bench world and would
be a straight-up lie about what this eval type measures.

Task 5 (task-1691 phase 2) wires ``EvalsScreen``'s selection routing to
mount this widget for a character-probe selection (a new ``"character_
bench"`` ``SelectionKind``) and adds the "create a bench" flow from the
rail. Until then this widget is fully functional on its own -- constructed
directly with an already-loaded ``EvalsViewModel`` and an already-fetched
``cards`` sequence (see ``__init__``) -- and this module's own test suite
(``Tests/UI/test_evals_character_bench_editor.py``) exercises it that way,
mounted into a real ``EvalsScreen``'s ``#evals-detail-pane`` rather than
through ``EvalsScreen.select()``'s kind-routing, which does not know about
this bench type yet.

**Editable in this task:** name, description, character selection (via
``card_picker.CardPicker``), and the four sampler fields (samples per cell,
seed, temperature, max tokens). **Read-only in this task**, mirroring
``BenchEditor``'s own permanently-read-only dataset field: the probe set
(``probe_set_id``, resolved and displayed, never reassignable here -- it is
a create-time-only choice, same as a word bench's dataset) and the target
list (``target_ids``, carried through verbatim on Save with no Add/Remove
control of its own -- unlike ``BenchEditor``'s Task 6 target editing, there
is no equivalent task in this program's plan; a character bench's targets
are set at creation time, Task 5's job). ``concurrency`` and ``extra_tags``
have no UI control either and round-trip verbatim from ``self._loaded_
config``, the same passthrough ``BenchEditor._on_save_pressed`` already
uses for its own ``concurrency``.

Probe turns render through ``snippet_editor.render_snippet_cell`` for the
␣-marker whitespace convention -- a probe's turn text is a verbatim prompt,
and leading/trailing/interior-run whitespace changes what is actually sent
to the model, the identical stakes ``snippet_editor.py``'s own module
docstring documents for a word bench's snippets. Turns render through
``snippet_editor.guard_single_line`` first so a probe (possibly several
turns, any of which may itself carry an embedded newline) still renders as
exactly one row in the read-only ``#evals-cb-probes`` listing -- the same
shared newline guard ``bench_editor.py``'s own ``_steering_preview_text``
now delegates to (see that function's updated docstring).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static

from ...DB.Evals_DB import ConflictError, EvalsDB
from ...Evals.character_probe.models import CharacterProbeConfig, Probe
from ...Evals.character_probe.storage import (
    load_character_bench,
    load_probe_set,
    save_character_bench,
)
from .card_picker import CardPicker
from .evals_state import EvalsViewModel
from .snippet_editor import guard_single_line, render_snippet_cell

#: Verbatim. The pinned error for an unparseable or sub-1 samples-per-cell
#: value -- asserted exactly by
#: ``test_an_invalid_samples_value_renders_the_error_and_keeps_typed_state``.
SAMPLES_ERROR_TEXT = "Samples per cell must be a whole number of 1 or more."

#: Verbatim. Unlike every other numeric field here, a NEGATIVE seed is a
#: real, intentional value (llama.cpp's "pick a random seed" sentinel, see
#: ``CharacterProbeConfig``'s own docstring) -- only "not a whole number at
#: all" is rejected.
SEED_ERROR_TEXT = "Seed must be a whole number (negative allowed) or left blank."

#: Verbatim. Mirrors ``storage._stored_temperature``'s own floor.
TEMPERATURE_ERROR_TEXT = "Temperature must be zero or a positive number."

#: Verbatim. Mirrors ``storage._stored_int_field``'s own floor -- zero is a
#: real, explicit value (not silently replaced), only negative is rejected.
MAX_TOKENS_ERROR_TEXT = "Max tokens must be zero or a whole number."


def _probe_preview_text(probe: Probe) -> str:
    """A single-line preview source string for one probe, BEFORE it goes
    through ``render_snippet_cell``'s own ␣-marker convention.

    A probe with more than one turn has its turns joined with the same
    newline guard ``guard_single_line`` uses for an embedded newline within
    a single turn -- both collapse to the same visible "⏎"; this listing
    draws no distinction between "the next turn starts here" and "this
    turn itself contains a line break", since either way the point of this
    preview is that ONE probe still reads as ONE row.

    Args:
        probe: The probe to preview.

    Returns:
        str: ``probe``'s turns joined and guarded to a single line.
    """
    return guard_single_line("\n".join(probe.turns))


class CharacterBenchEditor(Vertical):
    """Character-probe bench editor: name, description, character
    selection, and the sampler are editable (Save/Revert); the probe set
    and target list stay read-only in this task -- see the module
    docstring."""

    class Saved(Message, namespace="character_bench_editor"):
        """Posted after ``save_character_bench`` succeeds. Carries the
        bench's own ``eval_tasks`` id (unchanged across a save -- this is
        always an edit, never a create) so a future routing handler
        (Task 5) can re-select it the same way ``BenchEditor.Saved`` is
        handled today, without this widget reaching into ``self.screen``
        itself.
        """

        def __init__(self, bench_id: str) -> None:
            super().__init__()
            self.bench_id = bench_id

    def __init__(
        self,
        view_model: EvalsViewModel,
        bench_id: str,
        cards: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Args:
            view_model: The read side for this workbench.
            bench_id: The ``eval_tasks`` id of the character-probe bench to
                edit.
            cards: Already-fetched character-card rows (``EvalsViewModel.
                character_cards()``'s own shape) for the ``CardPicker`` --
                this widget never opens ``ChaChaNotes_DB`` itself, mirroring
                ``CardPicker``'s own "receives already-fetched rows" design
                (see that module's docstring for why: cards live in a
                different database from the one ``view_model`` wraps).
        """
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id
        self._cards = list(cards)
        #: The config `compose()` most recently loaded -- read back by the
        #: Save handler for the fields this widget carries through
        #: verbatim (`probe_set_id`, `target_ids`, `concurrency`,
        #: `extra_tags`) rather than exposing an edit control for them in
        #: this task. `None` only when `compose()` bailed out before
        #: reaching the form (no db, or an unreadable row).
        self._loaded_config: Optional[CharacterProbeConfig] = None

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            yield Static(
                "The evaluation service is unavailable.",
                id="evals-cb-editor-unavailable",
            )
            return
        try:
            config = load_character_bench(db, self._bench_id)
        except Exception:
            # Fail loudly at the point of first use, never silently: a
            # corrupt/missing bench row must not degrade to an empty or
            # half-filled form that LOOKS like a fresh, blank bench --
            # mirrors BenchEditor.compose()'s identical broad catch for the
            # identical reason (load_character_bench itself already names
            # the bench in every ValueError it raises; this is the
            # boundary where that error becomes a rendered fact instead of
            # crashing the whole screen).
            yield Static(
                "This bench's configuration could not be read.",
                id="evals-cb-editor-error",
            )
            return
        self._loaded_config = config

        yield Static("Name", classes="evals-cb-field-label")
        yield Input(value=config.name, id="evals-cb-name")

        yield Static("Description", classes="evals-cb-field-label")
        yield Input(value=config.description, id="evals-cb-description")

        yield from self._build_probe_set_section(db, config)

        yield Static("Characters", classes="destination-section evals-pane-title")
        yield CardPicker(self._cards, config.character_ids, id="evals-cb-cards")

        yield from self._build_targets_section(db, config)

        yield Static("Sampler", classes="destination-section evals-pane-title")
        yield Static("Samples per cell", classes="evals-cb-field-label")
        yield Input(value=str(config.samples_per_cell), id="evals-cb-samples")
        yield Static(
            "Seed (optional — blank is unseeded; -1 is llama.cpp's own "
            "random-seed sentinel)",
            classes="evals-cb-field-label",
        )
        yield Input(
            value="" if config.seed is None else str(config.seed), id="evals-cb-seed"
        )
        yield Static("Temperature", classes="evals-cb-field-label")
        yield Input(value=str(config.temperature), id="evals-cb-temperature")
        yield Static("Max tokens", classes="evals-cb-field-label")
        yield Input(value=str(config.max_tokens), id="evals-cb-max-tokens")

        # `.ds-recovery-callout` is deliberately withheld until an actual
        # failure -- see `_show_form_error`'s own comment and
        # `BenchEditor.compose()`'s identical, longer-documented rationale
        # for the SAME class on `#evals-bench-form-error`: an always-
        # classed-but-hidden Static still matches a screen-wide `not
        # screen.query(".ds-recovery-callout")` assertion elsewhere in this
        # workbench, and an always-visible empty one is a permanent blank
        # bordered box.
        error_widget = Static("", id="evals-cb-form-error", markup=False)
        error_widget.display = False
        yield error_widget

        with Horizontal(id="evals-cb-form-actions", classes="ds-toolbar"):
            yield Button(
                "Save",
                id="evals-cb-save",
                classes="console-action-primary",
                tooltip="Save name, description, characters, and sampler settings.",
            )
            yield Button(
                "Revert",
                id="evals-cb-revert",
                classes="console-action-secondary",
                tooltip="Discard unsaved changes and reload this bench.",
            )

    def _build_probe_set_section(
        self, db: EvalsDB, config: CharacterProbeConfig
    ) -> list[Any]:
        """The read-only probe-set line plus the ␣/⏎-marker probe listing
        (``#evals-cb-probes``).

        A referenced probe set that is missing or corrupt (``load_probe_
        set`` raising) degrades to an inline "(probe set unavailable)"
        state for THIS section only, rather than failing the whole widget
        the way an unreadable BENCH row does in ``compose()`` -- mirrors
        ``BenchEditor``'s own treatment of a deleted TARGET (``_build_
        target_row``: "(deleted target ...) — unresolvable" inline, not a
        whole-widget failure): a dangling reference to a sibling record is
        a real, recoverable state a user should still be able to see and
        fix the REST of this form around (rename the bench, fix the
        sampler), not a hard stop.

        Returns:
            list[Any]: Widgets to yield, in order.
        """
        probe_set_row = self._view_model.dataset_by_id(config.probe_set_id)
        probe_set_name = (
            str(probe_set_row.get("name") or config.probe_set_id)
            if probe_set_row is not None
            else "(probe set not found)"
        )
        try:
            probe_set = load_probe_set(db, config.probe_set_id)
        except ValueError:
            probe_set = None

        probe_set_text = f"Probe set: {probe_set_name}"
        if probe_set is not None:
            probe_set_text += f" ({len(probe_set.probes)} probes)"
        # markup=False: probe_set_name is user-authored free text -- see
        # BenchEditor.compose()'s identical `dataset_static` comment for
        # why a bare `[/]` in it would otherwise raise `MarkupError`.
        probe_set_static = Static(probe_set_text, id="evals-cb-probe-set", markup=False)
        probe_set_static.tooltip = (
            "The probe set is chosen when a bench is created and cannot be "
            "changed here."
        )
        widgets: list[Any] = [probe_set_static]

        widgets.append(
            Static(
                "Probes (read-only) — leading/trailing/interior-run "
                "whitespace shown as ␣, line breaks as ⏎",
                classes="evals-cb-field-label",
            )
        )
        if probe_set is None:
            widgets.append(
                Static("(probe set unavailable)", id="evals-cb-probes", markup=False)
            )
        elif not probe_set.probes:
            widgets.append(Static("(no probes yet)", id="evals-cb-probes", markup=False))
        else:
            lines = [
                render_snippet_cell(_probe_preview_text(probe))
                for probe in probe_set.probes
            ]
            combined = Text("\n").join(lines)
            widgets.append(Static(combined, id="evals-cb-probes", markup=False))
        return widgets

    @staticmethod
    def _build_targets_section(db: EvalsDB, config: CharacterProbeConfig) -> list[Any]:
        """The read-only target listing -- no Add/Remove control in this
        task (see the module docstring: targets are set at bench-creation
        time, Task 5's job, and carried through verbatim by ``_on_save_
        pressed`` here). Deliberately renders NO readiness/preflight state
        -- that vocabulary (``PreflightResult``, "Ready"/"Blocked", a
        canary check) belongs entirely to ``bench_editor.py``'s word-bench
        world; a character probe reads generated text, not per-token
        logprobs, and has no equivalent concept.

        A dangling target reference (a deleted ``eval_models`` row)
        degrades to an inline "(deleted target ...) — unresolvable" label,
        mirroring ``BenchEditor._build_target_row``'s identical treatment
        of the same case for a word bench.

        Returns:
            list[Any]: Widgets to yield, in order.
        """
        widgets: list[Any] = [
            Static("Targets", classes="destination-section evals-pane-title")
        ]
        if not config.target_ids:
            widgets.append(Static("No targets configured yet.", id="evals-cb-targets-empty"))
            return widgets
        rows = []
        for index, target_id in enumerate(config.target_ids):
            model = db.get_model(target_id)
            if model is None:
                label: Any = f"(deleted target {target_id}) — unresolvable"
            else:
                label = f"{model['name']} ({model['provider']})"
            rows.append(
                Static(
                    label,
                    id=f"evals-cb-target-{index}",
                    classes="evals-cb-target-row",
                    markup=False,
                )
            )
        widgets.append(Vertical(*rows, id="evals-cb-targets"))
        return widgets

    def _show_form_error(self, message: str) -> None:
        """Renders ``message`` in ``#evals-cb-form-error`` IN PLACE --
        never via ``self.refresh(recompose=True)``. Mirrors ``BenchEditor.
        _show_form_error``'s identical contract: a recompose here would
        rebuild every field from the last-saved config, discarding whatever
        the user had just typed."""
        error_widget = self.query_one("#evals-cb-form-error", Static)
        error_widget.update(message)
        error_widget.add_class("ds-recovery-callout")
        error_widget.display = True

    @on(Button.Pressed, "#evals-cb-save")
    def _on_save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        db = self._view_model.db
        loaded = self._loaded_config
        if db is None or loaded is None:
            # Defensive only: this button is never composed unless both a
            # db and a successfully loaded config exist (see compose()'s
            # own early returns above).
            return

        name = self.query_one("#evals-cb-name", Input).value
        description = self.query_one("#evals-cb-description", Input).value
        samples_raw = self.query_one("#evals-cb-samples", Input).value
        seed_raw = self.query_one("#evals-cb-seed", Input).value
        temperature_raw = self.query_one("#evals-cb-temperature", Input).value
        max_tokens_raw = self.query_one("#evals-cb-max-tokens", Input).value
        character_ids = self.query_one(CardPicker).selected_ids()

        try:
            samples_per_cell = int(samples_raw.strip())
            if samples_per_cell < 1:
                raise ValueError("samples_per_cell below 1")
        except ValueError:
            self._show_form_error(SAMPLES_ERROR_TEXT)
            return

        seed_text = seed_raw.strip()
        seed: Optional[int]
        if seed_text == "":
            seed = None
        else:
            try:
                seed = int(seed_text)
            except ValueError:
                self._show_form_error(SEED_ERROR_TEXT)
                return

        try:
            temperature = float(temperature_raw.strip())
            if temperature < 0:
                raise ValueError("temperature below 0")
        except ValueError:
            self._show_form_error(TEMPERATURE_ERROR_TEXT)
            return

        try:
            max_tokens = int(max_tokens_raw.strip())
            if max_tokens < 0:
                raise ValueError("max_tokens below 0")
        except ValueError:
            self._show_form_error(MAX_TOKENS_ERROR_TEXT)
            return

        try:
            config = CharacterProbeConfig(
                name=name,
                description=description,
                probe_set_id=loaded.probe_set_id,
                character_ids=character_ids,
                # Carried through verbatim -- no UI control for either in
                # this task, see the module docstring.
                target_ids=loaded.target_ids,
                concurrency=loaded.concurrency,
                samples_per_cell=samples_per_cell,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_tags=loaded.extra_tags,
            )
        except ValueError as exc:
            # CharacterProbeConfig.__post_init__ -- most reachably, an
            # empty character_ids (the CardPicker's selection cleared to
            # nothing) or empty target_ids (only possible if the loaded
            # bench itself already had none, since this task offers no way
            # to add one).
            self._show_form_error(str(exc))
            return

        try:
            save_character_bench(db, config, self._bench_id)
        except ConflictError:
            # Mirrors BenchEditor._on_save_pressed's identical ConflictError
            # copy verbatim -- eval_tasks.name's UNIQUE index has no
            # deleted_at exemption, so the same "a deleted bench may still
            # be holding it" trap applies here too.
            self._show_form_error(
                f'A bench named "{config.name}" already exists -- choose a '
                "different name. (Deleting a bench does not free its name: "
                "a deleted bench may still be holding it.)"
            )
            return
        except ValueError as exc:
            # save_character_bench's update path: the bench was deleted (by
            # this process or another) between this form loading it and
            # this Save -- see that function's own docstring.
            self._show_form_error(str(exc))
            return

        self.post_message(self.Saved(self._bench_id))

    @on(Button.Pressed, "#evals-cb-revert")
    def _on_revert_pressed(self, event: Button.Pressed) -> None:
        """Discards unsaved edits by re-selecting this same bench --
        mirrors ``BenchEditor._on_revert_pressed``'s identical contract.
        ``kind="character_bench"`` is Task 5's own selection kind; calling
        it here now is forward-compatible plumbing for that task to finish
        wiring, not a claim that routing already works end to end."""
        event.stop()
        screen_select = getattr(self.screen, "select", None)
        if callable(screen_select):
            screen_select(kind="character_bench", id=self._bench_id)
