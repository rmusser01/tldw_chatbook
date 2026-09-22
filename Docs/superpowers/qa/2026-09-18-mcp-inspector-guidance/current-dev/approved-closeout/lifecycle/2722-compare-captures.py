import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

old = Path("Docs/superpowers/qa/2026-09-18-mcp-inspector-guidance/current-dev/native")
new = Path("<tmp>/tldw-32836-approved-001/evidence")


def normalize_svg(path):
    """Normalize generated identifiers and timestamps without changing geometry.

    Args:
        path: Path to a UTF-8 Rich terminal SVG capture.

    Returns:
        Serialized SVG bytes with stable styles and timestamp digits.
    """
    text = re.sub(r"terminal-\d+", "terminal", path.read_text())
    styles = dict(re.findall(r"\.(terminal-r\d+)\s*\{([^}]*)\}", text))
    for name, body in sorted(styles.items(), key=lambda item: -len(item[0])):
        text = re.sub(
            r"\b" + re.escape(name) + r"\b",
            "style-" + hashlib.sha256(body.strip().encode()).hexdigest()[:12],
            text,
        )
    root = ET.fromstring(text)
    for node in root.iter():
        if node.tail and not node.tail.strip():
            node.tail = None
        if node.text and not node.text.strip():
            node.text = None
        if node.tag.endswith("style"):
            css = node.text or ""
            rules = re.findall(r"\.style-[a-f0-9]+\s*\{[^}]*\}", css)
            css = re.sub(r"\.style-[a-f0-9]+\s*\{[^}]*\}", "", css)
            node.text = css + "".join(sorted(set(rules)))
        if (
            node.tag.endswith("text")
            and node.text
            and re.fullmatch(r'[\d\sT:,.+\-"/]+', node.text)
            and any(c in node.text for c in ["-", ":", "T"])
        ):
            node.text = re.sub(r"\d", "#", node.text)
    return ET.tostring(root)


def text_rows(path):
    """Read terminal rows with timestamp digits and trailing whitespace normalized.

    Args:
        path: Path to the UTF-8 plain terminal capture.

    Returns:
        Normalized terminal rows in their original order.
    """
    rows = []
    for s in path.read_text().splitlines():
        # Timestamps wrap in the compact inspector; normalize only timestamp runs.
        s = re.sub(
            r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:?\d{0,2}:?\d{0,2}(?:\.\d+)?(?:\+\d{2}:\d{2})?)?",
            lambda m: re.sub(r"\d", "#", m[0]),
            s,
        )
        s = re.sub(
            r":\d{2}:\d{2}\.\d+(?:\+\d{0,2})?", lambda m: re.sub(r"\d", "#", m[0]), s
        )
        s = re.sub(r'\b\d{2}:\d{2}"', lambda m: re.sub(r"\d", "#", m[0]), s)
        rows.append(s.rstrip())
    return rows


def inspector_elements(path):
    """Select normalized inspector geometry and text for the recorded viewport.

    Args:
        path: SVG capture path whose name identifies compact 80x24 or wide layout.

    Returns:
        Ordered text/rectangle tags, attributes and content in the inspector region.
    """
    root = ET.fromstring(normalize_svg(path))
    minimum = 732 if "80x24" in path.name else 1464
    return [
        (n.tag, sorted(n.attrib.items()), n.text)
        for n in root.iter()
        if n.tag.rsplit("}", 1)[-1] in ["rect", "text"]
        and float(n.attrib.get("x", -1)) >= minimum
    ]


result = []
for p in sorted(new.glob("*.svg")):
    name = p.name
    prev = old / name
    a = text_rows(prev.with_suffix(".txt"))
    b = text_rows(p.with_suffix(".txt"))
    row = {
        "capture": name,
        "semantic_svg_equal": normalize_svg(prev) == normalize_svg(p),
        "terminal_equal": a == b,
        "differing_terminal_rows": [
            i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y
        ],
    }
    row["inspector_equal"] = inspector_elements(prev) == inspector_elements(p)
    result.append(row)
Path("<tmp>/2722-capture-comparison.json").write_text(
    json.dumps(
        {
            "normalization": "Generated Rich IDs/styles, insignificant XML whitespace and numeric timestamps only; no layout or color normalization. Inspector comparison retains all text/rect geometry and styles at inspector x coordinates.",
            "captures": result,
        },
        indent=2,
    )
    + "\n"
)
print(
    "SVG equal",
    sum(r["semantic_svg_equal"] for r in result),
    "of",
    len(result),
    "terminal equal",
    sum(r["terminal_equal"] for r in result),
)

assert all(r["inspector_equal"] for r in result)
