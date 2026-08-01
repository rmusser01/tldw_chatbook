"""Pure data for character probe evals. No I/O, no Textual, no provider calls."""

from __future__ import annotations

from dataclasses import dataclass


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
