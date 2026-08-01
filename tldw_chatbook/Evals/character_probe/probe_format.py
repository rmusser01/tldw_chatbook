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
    """Strip leading/trailing blank lines (including whitespace-only lines).

    Interior whitespace is preserved exactly.
    """
    lines = raw.split("\n")
    # Strip leading blank lines (lines with only whitespace)
    while lines and not lines[0].strip():
        lines.pop(0)
    # Strip trailing blank lines (lines with only whitespace)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


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
        raw_turns = _split_on_delimiter(chunk, TURN_DELIMITER)
        turns: list[str] = []
        for i, raw in enumerate(raw_turns):
            if not raw.strip():
                # Check if this is a stray delimiter (not at edges)
                if i > 0 and i < len(raw_turns) - 1:
                    raise ValueError(
                        f"probe {index} has a stray or duplicated turn delimiter"
                    )
                # Skip leading/trailing empty turns from delimiter noise
            else:
                turns.append(_clean_turn(raw))
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
        str: Text in the plain-text format. Note: if any turn contains a line
            that strips to ``---`` or ``===``, it will not round-trip through
            parse_probe_text, as those are treated as delimiters. Escaping is
            not supported in this format version.
    """
    return f"\n{PROBE_DELIMITER}\n".join(
        f"\n{TURN_DELIMITER}\n".join(probe.turns) for probe in probe_set.probes
    )
