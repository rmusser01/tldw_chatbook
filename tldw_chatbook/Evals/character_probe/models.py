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
    as integers.

    Defaults are conservative on purpose: ``samples_per_cell=1`` (no
    surprise fan-out cost), ``seed=None`` (no false sense of determinism
    unless explicitly requested), and ``concurrency=1`` (no surprise
    parallel-request load against a target).

    Raises:
        ValueError: If ``samples_per_cell`` or ``concurrency`` is less than
            1, or if ``character_ids``/``target_ids`` is empty -- a bench
            with no characters or no targets can never produce a cell.
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
