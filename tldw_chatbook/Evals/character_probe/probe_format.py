"""The plain-text probe format: `---` between turns, `===` between probes.

Turns are delimited explicitly rather than by line breaks so a single turn can
be a multi-paragraph prompt -- complex prompts are exactly what this eval
exists to study, and a newline-delimited format could not express one.
"""

from __future__ import annotations

from .models import Probe, ProbeSet

#: A line containing only this separates turns within a probe.
TURN_DELIMITER = "---"
#: A line containing only this separates probes within a set.
PROBE_DELIMITER = "==="


def _split_on_delimiter(text: str, delimiter: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == delimiter:
            chunks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    chunks.append("\n".join(current))
    return chunks


def _clean_turn(raw: str) -> str:
    """Strip only leading/trailing blank lines; interior whitespace is data."""
    return raw.strip("\n").strip() if not raw.strip() else raw.strip("\n")


def parse_probe_text(text: str) -> ProbeSet:
    """Parse the plain-text probe format into a ``ProbeSet``.

    Args:
        text: The file's contents.

    Returns:
        ProbeSet: The parsed probes, in file order.

    Raises:
        ValueError: If the text contains no probes, or if any probe has no
            turns (naming the 1-based probe number so the author can find it).
    """
    probe_chunks = _split_on_delimiter(text, PROBE_DELIMITER)
    probes: list[Probe] = []
    for index, chunk in enumerate(probe_chunks, start=1):
        if not chunk.strip():
            if len(probe_chunks) == 1 or index in (1, len(probe_chunks)):
                # A wholly empty document, or trailing/leading delimiter noise.
                continue
            raise ValueError(f"probe {index} has no turns")
        turns = [
            _clean_turn(raw)
            for raw in _split_on_delimiter(chunk, TURN_DELIMITER)
            if raw.strip()
        ]
        if not turns:
            raise ValueError(f"probe {index} has no turns")
        probes.append(Probe(turns=tuple(turns)))
    if not probes:
        raise ValueError("The probe file contains no probes.")
    return ProbeSet(probes=tuple(probes))


def format_probe_text(probe_set: ProbeSet) -> str:
    """Render a ``ProbeSet`` back to the plain-text format.

    Args:
        probe_set: The set to render.

    Returns:
        str: Text that ``parse_probe_text`` round-trips to an equal ProbeSet.
    """
    return f"\n{PROBE_DELIMITER}\n".join(
        f"\n{TURN_DELIMITER}\n".join(probe.turns) for probe in probe_set.probes
    )
