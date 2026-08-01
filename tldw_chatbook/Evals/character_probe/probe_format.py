"""The plain-text probe format: `---` between turns, `===` between probes.

Turns are delimited explicitly rather than by line breaks so a single turn can
be a multi-paragraph prompt -- complex prompts are exactly what this eval
exists to study, and a newline-delimited format could not express one.

**Delimiter matching is lenient about surrounding whitespace**: a line
delimits when ``line.strip()`` equals the delimiter, so ``"  ---  "`` is a
delimiter just as ``"---"`` is. The spec's wording ("a line of ``---``")
reads stricter than that, and the leniency is a deliberate ruling rather
than an oversight. The two failure modes are not symmetric: with a strict
match, an INVISIBLE trailing space makes a delimiter silently fail to
delimit, merging two turns into one prompt that then runs and produces
plausible-looking results -- likely to happen (editors and copy-paste add
trailing whitespace constantly) and very hard to see. With the lenient
match, the cost is that an indented literal ``---`` inside a turn is eaten
as a delimiter -- less likely, and it fails visibly as a probe split in the
wrong place. The lenient reading is chosen for that reason; the cost is
documented rather than hidden, and pinned by test.
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

    A line delimits when its STRIPPED content equals the delimiter, so
    ``"  ---  "`` and ``"\t==="`` delimit exactly as a bare ``---``/``===``
    does. That leniency is deliberate (see the module note below), and it
    widens the v1 escaping limitation: no line whose stripped content is
    ``---`` or ``===`` can appear inside a turn, indented or padded or
    otherwise.

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
            that STRIPS to ``---`` or ``===`` -- including an indented or
            trailing-space form such as ``"  ---  "``, since delimiter
            matching compares stripped content (see the module docstring for
            why that leniency is deliberate) -- it will not round-trip
            through parse_probe_text, as such a line is treated as a
            delimiter. Escaping is not supported in this format version.
    """
    return f"\n{PROBE_DELIMITER}\n".join(
        f"\n{TURN_DELIMITER}\n".join(probe.turns) for probe in probe_set.probes
    )
