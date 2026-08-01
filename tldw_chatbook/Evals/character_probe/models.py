"""Pure data for character probe evals. No I/O, no Textual, no provider calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Probe:
    """One scripted exchange: an ordered list of user turns.

    A "one-off" question is simply a probe with a single turn -- there is no
    separate type for it. Turn text is verbatim, including interior newlines,
    because prompt formatting changes model behaviour.

    Raises:
        ValueError: If the probe has no turns, or if any turn is empty or
            contains only whitespace.
    """

    turns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("A probe needs at least one turn.")
        for turn in self.turns:
            if not turn.strip():
                raise ValueError("A turn cannot be empty or whitespace-only.")


@dataclass(frozen=True)
class ProbeSet:
    """An ordered collection of probes, the unit a bench runs."""

    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class CharacterProbeConfig:
    """A character probe bench definition.

    ``character_ids`` are ``character_cards.id`` INTEGERs, unlike every eval
    id in this slice, which is TEXT. They are deliberately not normalised to
    strings: the cross-database lookup against ``ChaChaNotes_DB`` binds them
    as integers. This is enforced here, not merely documented: a caller
    constructing one directly with string ids (e.g. ``("3", "7")``, easy to
    do by accident from a form field) is rejected at construction rather
    than being accepted here only to come back out as ``int`` after a
    ``storage.save_character_bench``/``load_character_bench`` round trip --
    an asymmetry a future UI caller would otherwise have to discover the
    expensive way.

    Defaults are conservative on purpose: ``samples_per_cell=1`` (no
    surprise fan-out cost), ``seed=None`` (no false sense of determinism
    unless explicitly requested), and ``concurrency=1`` (no surprise
    parallel-request load against a target).

    Raises:
        ValueError: If ``samples_per_cell`` or ``concurrency`` is less than
            1, or if ``character_ids``/``target_ids`` is empty -- a bench
            with no characters or no targets can never produce a cell.
        ValueError: If any element of ``character_ids`` is not an ``int``
            (``bool`` included, since it is an ``int`` subclass but never a
            real card id) -- see the type-fidelity note above.
    """

    name: str
    probe_set_id: str
    character_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    description: str = ""
    concurrency: int = 1
    samples_per_cell: int = 1
    seed: Optional[int] = None
    temperature: float = 0.8
    max_tokens: int = 512
    extra_tags: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.samples_per_cell < 1:
            raise ValueError("samples_per_cell must be 1 or more.")
        if self.concurrency < 1:
            raise ValueError("concurrency must be 1 or more.")
        if not self.character_ids:
            raise ValueError("A character probe bench needs at least one character.")
        if not self.target_ids:
            raise ValueError("A character probe bench needs at least one target.")
        for cid in self.character_ids:
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ValueError(
                    f"character_ids must be int (character_cards.id), got "
                    f"{cid!r} of type {type(cid).__name__}."
                )


@dataclass(frozen=True)
class CardSnapshot:
    """A character card's text, copied at run time.

    Cards live in ``ChaChaNotes_DB`` while runs live in ``Evals_DB``, with no
    foreign keys between them. Copying the text into the run means editing or
    deleting a card later never rewrites what a past run shows -- the same
    provenance rule word_bench applies to snippets.
    """

    id: int
    name: str
    system_prompt: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    post_history_instructions: str = ""
    message_example: str = ""
