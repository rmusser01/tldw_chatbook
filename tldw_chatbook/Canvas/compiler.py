"""Compile untrusted Canvas HTML into the closed Canvas V1 render plan."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, NoReturn

import html5lib
import tinycss2
from html5lib._tokenizer import HTMLTokenizer
from html5lib.constants import tokenTypes

from .limits import (
    CanvasLimitError,
    CanvasLimits,
    DecodedDataUrl,
    decode_data_url,
    validate_asset_payloads,
    validate_count,
    validate_utf8_text,
    validate_utf8_text_parts,
)
from .models import (
    CanvasCompatibilityIssue,
    CanvasRenderPlan,
    CanvasSourceIdentity,
    RenderAsset,
    RenderNode,
)


_HTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/"
_MAX_FATAL_ISSUES = 16

_HTML_ELEMENTS = frozenset(
    {
        "a",
        "abbr",
        "address",
        "article",
        "aside",
        "b",
        "bdi",
        "bdo",
        "blockquote",
        "body",
        "br",
        "button",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "div",
        "dl",
        "dt",
        "em",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "i",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "main",
        "mark",
        "menu",
        "meta",
        "meter",
        "nav",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "pre",
        "progress",
        "q",
        "s",
        "samp",
        "script",
        "section",
        "select",
        "small",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "u",
        "ul",
        "var",
        "wbr",
    }
)
_SVG_ELEMENTS = frozenset(
    {
        "svg",
        "g",
        "circle",
        "ellipse",
        "line",
        "path",
        "polygon",
        "polyline",
        "rect",
        "text",
        "tspan",
    }
)
_EXTRACTED_ELEMENTS = frozenset({"script", "style"})

_GLOBAL_ATTRIBUTES = frozenset(
    {"class", "dir", "hidden", "id", "lang", "role", "style", "tabindex", "title"}
)
_HTML_ATTRIBUTES: dict[str, frozenset[str]] = {
    "a": frozenset({"href"}),
    "button": frozenset({"disabled", "name", "type", "value"}),
    "col": frozenset({"span"}),
    "colgroup": frozenset({"span"}),
    "data": frozenset({"value"}),
    "del": frozenset({"datetime"}),
    "fieldset": frozenset({"disabled", "name"}),
    "form": frozenset({"name", "novalidate"}),
    "img": frozenset({"alt", "decoding", "height", "loading", "src", "width"}),
    "input": frozenset(
        {
            "checked",
            "disabled",
            "form",
            "list",
            "max",
            "maxlength",
            "min",
            "minlength",
            "multiple",
            "name",
            "pattern",
            "placeholder",
            "readonly",
            "required",
            "size",
            "step",
            "type",
            "value",
        }
    ),
    "ins": frozenset({"datetime"}),
    "label": frozenset({"for"}),
    "li": frozenset({"value"}),
    "meta": frozenset({"charset"}),
    "meter": frozenset({"high", "low", "max", "min", "optimum", "value"}),
    "ol": frozenset({"reversed", "start", "type"}),
    "optgroup": frozenset({"disabled", "label"}),
    "option": frozenset({"disabled", "label", "selected", "value"}),
    "output": frozenset({"for", "form", "name"}),
    "progress": frozenset({"max", "value"}),
    "select": frozenset({"disabled", "form", "multiple", "name", "required", "size"}),
    "td": frozenset({"colspan", "headers", "rowspan"}),
    "textarea": frozenset(
        {
            "cols",
            "disabled",
            "form",
            "maxlength",
            "minlength",
            "name",
            "placeholder",
            "readonly",
            "required",
            "rows",
            "wrap",
        }
    ),
    "th": frozenset({"abbr", "colspan", "headers", "rowspan", "scope"}),
    "time": frozenset({"datetime"}),
}
_SVG_ATTRIBUTES = frozenset(
    {
        "cx",
        "cy",
        "d",
        "fill",
        "fill-opacity",
        "height",
        "points",
        "preserveAspectRatio",
        "r",
        "rx",
        "ry",
        "stroke",
        "stroke-dasharray",
        "stroke-dashoffset",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
        "text-anchor",
        "transform",
        "vector-effect",
        "viewBox",
        "width",
        "x",
        "x1",
        "x2",
        "y",
        "y1",
        "y2",
    }
)
_SVG_PAINT_ATTRIBUTES = frozenset({"fill", "stroke"})
_URL_OR_NAVIGATION_ATTRIBUTES = frozenset(
    {
        "action",
        "archive",
        "background",
        "cite",
        "classid",
        "codebase",
        "data",
        "dynsrc",
        "formaction",
        "href",
        "icon",
        "longdesc",
        "lowsrc",
        "manifest",
        "ping",
        "poster",
        "profile",
        "src",
        "srcdoc",
        "srcset",
        "target",
        "usemap",
        "xlink:href",
    }
)
_INPUT_TYPES = frozenset(
    {
        "button",
        "checkbox",
        "color",
        "date",
        "datetime-local",
        "email",
        "hidden",
        "month",
        "number",
        "password",
        "radio",
        "range",
        "reset",
        "search",
        "submit",
        "tel",
        "text",
        "time",
        "url",
        "week",
    }
)
_BUTTON_TYPES = frozenset({"button", "reset", "submit"})
_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})

_CSS_PROPERTIES = frozenset(
    {
        "align-content",
        "align-items",
        "align-self",
        "animation",
        "animation-delay",
        "animation-direction",
        "animation-duration",
        "animation-fill-mode",
        "animation-iteration-count",
        "animation-name",
        "animation-play-state",
        "animation-timing-function",
        "appearance",
        "aspect-ratio",
        "backdrop-filter",
        "backface-visibility",
        "background",
        "background-attachment",
        "background-blend-mode",
        "background-clip",
        "background-color",
        "background-image",
        "background-origin",
        "background-position",
        "background-repeat",
        "background-size",
        "block-size",
        "border",
        "border-block",
        "border-block-color",
        "border-block-end",
        "border-block-start",
        "border-block-style",
        "border-block-width",
        "border-bottom",
        "border-bottom-color",
        "border-bottom-left-radius",
        "border-bottom-right-radius",
        "border-bottom-style",
        "border-bottom-width",
        "border-collapse",
        "border-color",
        "border-inline",
        "border-inline-color",
        "border-inline-end",
        "border-inline-start",
        "border-inline-style",
        "border-inline-width",
        "border-left",
        "border-left-color",
        "border-left-style",
        "border-left-width",
        "border-radius",
        "border-right",
        "border-right-color",
        "border-right-style",
        "border-right-width",
        "border-spacing",
        "border-style",
        "border-top",
        "border-top-color",
        "border-top-left-radius",
        "border-top-right-radius",
        "border-top-style",
        "border-top-width",
        "border-width",
        "bottom",
        "box-shadow",
        "box-sizing",
        "break-after",
        "break-before",
        "break-inside",
        "caption-side",
        "caret-color",
        "clear",
        "color",
        "column-gap",
        "column-width",
        "columns",
        "display",
        "empty-cells",
        "filter",
        "flex",
        "flex-basis",
        "flex-direction",
        "flex-flow",
        "flex-grow",
        "flex-shrink",
        "flex-wrap",
        "float",
        "font",
        "font-family",
        "font-feature-settings",
        "font-kerning",
        "font-size",
        "font-stretch",
        "font-style",
        "font-variant",
        "font-weight",
        "gap",
        "grid",
        "grid-area",
        "grid-auto-columns",
        "grid-auto-flow",
        "grid-auto-rows",
        "grid-column",
        "grid-column-end",
        "grid-column-gap",
        "grid-column-start",
        "grid-gap",
        "grid-row",
        "grid-row-end",
        "grid-row-gap",
        "grid-row-start",
        "grid-template",
        "grid-template-areas",
        "grid-template-columns",
        "grid-template-rows",
        "height",
        "hyphens",
        "inline-size",
        "inset",
        "inset-block",
        "inset-block-end",
        "inset-block-start",
        "inset-inline",
        "inset-inline-end",
        "inset-inline-start",
        "isolation",
        "justify-content",
        "justify-items",
        "justify-self",
        "left",
        "letter-spacing",
        "line-height",
        "list-style",
        "list-style-position",
        "list-style-type",
        "margin",
        "margin-block",
        "margin-block-end",
        "margin-block-start",
        "margin-bottom",
        "margin-inline",
        "margin-inline-end",
        "margin-inline-start",
        "margin-left",
        "margin-right",
        "margin-top",
        "max-block-size",
        "max-height",
        "max-inline-size",
        "max-width",
        "min-block-size",
        "min-height",
        "min-inline-size",
        "min-width",
        "mix-blend-mode",
        "object-fit",
        "object-position",
        "opacity",
        "order",
        "outline",
        "outline-color",
        "outline-offset",
        "outline-style",
        "outline-width",
        "overflow",
        "overflow-wrap",
        "overflow-x",
        "overflow-y",
        "padding",
        "padding-block",
        "padding-block-end",
        "padding-block-start",
        "padding-bottom",
        "padding-inline",
        "padding-inline-end",
        "padding-inline-start",
        "padding-left",
        "padding-right",
        "padding-top",
        "perspective",
        "perspective-origin",
        "place-content",
        "place-items",
        "place-self",
        "pointer-events",
        "position",
        "resize",
        "right",
        "rotate",
        "row-gap",
        "scale",
        "scroll-behavior",
        "shape-rendering",
        "stroke",
        "stroke-dasharray",
        "stroke-dashoffset",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
        "table-layout",
        "text-align",
        "text-decoration",
        "text-decoration-color",
        "text-decoration-line",
        "text-decoration-style",
        "text-indent",
        "text-overflow",
        "text-shadow",
        "text-transform",
        "top",
        "transform",
        "transform-origin",
        "transform-style",
        "transition",
        "transition-delay",
        "transition-duration",
        "transition-property",
        "transition-timing-function",
        "translate",
        "unicode-bidi",
        "user-select",
        "vertical-align",
        "visibility",
        "white-space",
        "width",
        "word-break",
        "word-spacing",
        "writing-mode",
        "z-index",
        "fill",
        "fill-opacity",
    }
)
_CSS_FUNCTIONS = frozenset(
    {
        "calc",
        "clamp",
        "color",
        "color-mix",
        "conic-gradient",
        "cubic-bezier",
        "hsl",
        "hsla",
        "hwb",
        "lab",
        "lch",
        "linear-gradient",
        "matrix",
        "matrix3d",
        "max",
        "min",
        "oklab",
        "oklch",
        "perspective",
        "radial-gradient",
        "repeating-conic-gradient",
        "repeating-linear-gradient",
        "repeating-radial-gradient",
        "rgb",
        "rgba",
        "rotate",
        "rotatex",
        "rotatey",
        "rotatez",
        "scale",
        "scale3d",
        "scalex",
        "scaley",
        "scalez",
        "skew",
        "skewx",
        "skewy",
        "steps",
        "translate",
        "translate3d",
        "translatex",
        "translatey",
        "translatez",
    }
)
_PSEUDO_CLASSES = frozenset(
    {
        "active",
        "checked",
        "disabled",
        "empty",
        "enabled",
        "first-child",
        "first-of-type",
        "focus",
        "focus-visible",
        "focus-within",
        "hover",
        "invalid",
        "last-child",
        "last-of-type",
        "only-child",
        "only-of-type",
        "optional",
        "read-only",
        "read-write",
        "required",
        "root",
        "target",
        "valid",
    }
)
_PSEUDO_FUNCTIONS = frozenset(
    {
        "dir",
        "is",
        "lang",
        "not",
        "nth-child",
        "nth-last-child",
        "nth-last-of-type",
        "nth-of-type",
        "where",
    }
)


class CanvasCompileError(ValueError):
    """A content-free failure to compile the whole Canvas document."""

    def __init__(self, issues: tuple[CanvasCompatibilityIssue, ...]) -> None:
        if not issues:
            raise ValueError("Canvas compile error requires at least one issue")
        self.issues = issues[:_MAX_FATAL_ISSUES]
        super().__init__("Canvas document is incompatible with the canvas-v1 runtime")


@dataclass(slots=True)
class _PendingNode:
    tag: str
    attributes: tuple[tuple[str, str], ...] = ()
    text: str | None = None
    children: list["_PendingNode"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _CompiledCssRule:
    text: str
    rule_count: int


class _SourceLocator:
    """Best-effort source positions used only for content-free diagnostics."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._offsets: dict[str, int] = {}

    def element(self, tag: str) -> str:
        pattern = re.compile(r"<\s*" + re.escape(tag) + r"(?=[\s/>])", re.IGNORECASE)
        match = pattern.search(self._source, self._offsets.get(tag, 0))
        if match is None:
            return "line 1, column 1"
        self._offsets[tag] = match.end()
        before = self._source[: match.start()]
        line = before.count("\n") + 1
        last_newline = before.rfind("\n")
        column = match.start() + 1 if last_newline < 0 else match.start() - last_newline
        return f"line {line}, column {column}"


def compile_canvas_document(
    source: str, *, limits: CanvasLimits | None = None
) -> CanvasRenderPlan:
    """Compile exact HTML source into a validated, immutable Canvas V1 plan.

    Args:
        source: The exact complete HTML source (or a deterministically wrapped fragment).
        limits: Optional lower test/qualification ceilings for the compiler boundary.

    Returns:
        A closed render plan containing no browser markup or resolvable asset URL.

    Raises:
        CanvasCompileError: If any part of the document is malformed or unsupported.
    """
    active_limits = limits or CanvasLimits()
    try:
        validate_utf8_text(
            source, limit=active_limits.html_bytes, field_name="HTML source"
        )
    except (CanvasLimitError, TypeError):
        _fail("html-limit", "HTML source exceeds the Canvas V1 boundary.", "document")

    fragment = not _has_explicit_html_root(source)
    parser = html5lib.HTMLParser(
        tree=html5lib.getTreeBuilder("etree"), namespaceHTMLElements=True
    )
    document = None
    try:
        document = parser.parse(source)
    except (
        Exception
    ):  # html5lib must not leak parser/source details across this boundary
        pass
    if document is None:
        raise CanvasCompileError(
            (
                _issue(
                    "html-parse-error", "HTML could not be parsed safely.", "document"
                ),
            )
        ) from None

    parse_issues = []
    for (line, column), code, _details in parser.errors:
        if fragment and code.startswith("expected-doctype"):
            continue
        parse_issues.append(
            _issue(
                "html-parse-error",
                "HTML is malformed or requires unsupported parser recovery.",
                f"line {line}, column {column}",
            )
        )
        if len(parse_issues) == _MAX_FATAL_ISSUES:
            break
    if parse_issues:
        raise CanvasCompileError(tuple(parse_issues))

    locator = _SourceLocator(source)
    element_locations: dict[int, str] = {}
    element_info: dict[int, tuple[str, str]] = {}
    assets: list[RenderAsset] = []
    css_rules: list[str] = []
    scripts: list[str] = []
    source_ids: set[str] = set()
    css_rule_count = 0

    elements = list(document.iter())
    for element in elements:
        if not isinstance(element.tag, str):
            continue
        namespace, tag = _expanded_name(element.tag)
        location = locator.element(tag)
        element_locations[id(element)] = location
        element_info[id(element)] = (namespace, tag)
        _validate_element(namespace, tag, location)

        if namespace == _HTML_NAMESPACE and tag == "script":
            scripts.append(_extract_script(element, location))
            continue
        if namespace == _HTML_NAMESPACE and tag == "style":
            _validate_extracted_element_attributes(element, tag, location)
            compiled = _compile_stylesheet(element.text or "", location)
            css_rules.extend(rule.text for rule in compiled)
            css_rule_count += sum(rule.rule_count for rule in compiled)
            if css_rule_count > active_limits.css_rules:
                _fail(
                    "css-rule-limit", "CSS exceeds the Canvas V1 rule limit.", location
                )
            continue

        for raw_name, raw_value in element.attrib.items():
            if _is_safe_namespace_declaration(raw_name, str(raw_value), namespace, tag):
                continue
            name = _attribute_name(raw_name, location)
            value = str(raw_value)
            if name.startswith("on"):
                _fail(
                    "event-handler",
                    "Native event-handler attributes are not supported.",
                    location,
                )
            if name == "id":
                if value in source_ids:
                    _fail(
                        "duplicate-source-id",
                        "Source element IDs must be unique.",
                        location,
                    )
                source_ids.add(value)
            if name == "style":
                _compile_declarations(value, location)
            if namespace == _SVG_NAMESPACE and name in _SVG_PAINT_ATTRIBUTES:
                _validate_css_tokens(
                    tinycss2.parse_component_value_list(value), location
                )

    try:
        validate_utf8_text_parts(
            scripts, limit=active_limits.script_bytes, field_name="script"
        )
    except CanvasLimitError:
        _fail("script-limit", "Scripts exceed the Canvas V1 byte limit.", "document")

    pending_by_element: dict[int, _PendingNode | None] = {}
    for element in reversed(elements):
        if not isinstance(element.tag, str):
            pending_by_element[id(element)] = None
            continue
        namespace, tag = element_info[id(element)]
        location = element_locations[id(element)]
        if namespace == _HTML_NAMESPACE and tag in _EXTRACTED_ELEMENTS:
            pending_by_element[id(element)] = None
            continue

        attributes = _compile_attributes(
            element,
            namespace=namespace,
            tag=tag,
            location=location,
            assets=assets,
            limits=active_limits,
        )
        pending = _PendingNode(tag=tag, attributes=attributes)
        if element.text:
            pending.children.append(_PendingNode(tag="#text", text=element.text))
        for child in list(element):
            child_pending = pending_by_element.get(id(child))
            if child_pending is not None:
                pending.children.append(child_pending)
            if child.tail:
                pending.children.append(_PendingNode(tag="#text", text=child.tail))
        pending_by_element[id(element)] = pending

    pending_root = pending_by_element.get(id(document))
    if pending_root is None or pending_root.tag != "html":
        _fail(
            "html-root",
            "HTML did not normalize to the required document root.",
            "document",
        )

    root = _freeze_tree(pending_root, active_limits)
    compatibility_issues = (
        (
            _issue(
                "fragment-wrapped",
                "HTML fragment was wrapped as html, head, and body.",
                "line 1, column 1",
            ),
        )
        if fragment
        else ()
    )
    try:
        identity = CanvasSourceIdentity.from_source(source)
        return CanvasRenderPlan(
            runtime_profile="canvas-v1",
            source_identity=identity,
            root=root,
            assets=tuple(assets),
            css_rules=tuple(css_rules),
            scripts=tuple(scripts),
            compatibility_issues=compatibility_issues,
        )
    except CanvasLimitError as exc:
        raise CanvasCompileError(
            (
                _issue(
                    "render-plan-limit",
                    "Compiled document exceeds the Canvas V1 boundary.",
                    "document",
                ),
            )
        ) from exc


def _has_explicit_html_root(source: str) -> bool:
    try:
        for token in HTMLTokenizer(source):
            if token["type"] == tokenTypes["StartTag"]:
                return token["name"] == "html"
    except Exception:
        return False
    return False


def _expanded_name(name: str) -> tuple[str, str]:
    if name.startswith("{"):
        namespace, local_name = name[1:].split("}", maxsplit=1)
        return namespace, local_name
    return _HTML_NAMESPACE, name


def _attribute_name(raw_name: str, location: str) -> str:
    if raw_name == "xmlns" or raw_name.startswith("{") or ":" in raw_name:
        _fail(
            "unsupported-namespace",
            "Namespaced attributes are not supported.",
            location,
        )
    return raw_name


def _is_safe_namespace_declaration(
    raw_name: str, value: str, element_namespace: str, tag: str
) -> bool:
    if raw_name == "xmlns":
        if (
            element_namespace == _HTML_NAMESPACE
            and tag == "html"
            and value == _HTML_NAMESPACE
        ):
            return True
        return False
    if raw_name.startswith("{"):
        attribute_namespace, local_name = _expanded_name(raw_name)
        return (
            attribute_namespace == _XMLNS_NAMESPACE
            and local_name == "xmlns"
            and element_namespace == _SVG_NAMESPACE
            and tag == "svg"
            and value == _SVG_NAMESPACE
        )
    return False


def _validate_element(namespace: str, tag: str, location: str) -> None:
    if namespace == _HTML_NAMESPACE:
        if tag not in _HTML_ELEMENTS or "-" in tag:
            _fail(
                "unsupported-element",
                "An HTML element is not supported by Canvas V1.",
                location,
            )
        return
    if namespace == _SVG_NAMESPACE:
        if tag not in _SVG_ELEMENTS:
            _fail(
                "unsupported-element",
                "An SVG element is not supported by Canvas V1.",
                location,
            )
        return
    _fail(
        "unsupported-namespace",
        "A document namespace is not supported by Canvas V1.",
        location,
    )


def _validate_extracted_element_attributes(
    element: Any, tag: str, location: str
) -> None:
    if tag == "style" and element.attrib == {"type": "text/css"}:
        return
    if element.attrib:
        code = "script-attribute" if tag == "script" else "style-attribute"
        _fail(code, "This extracted element has unsupported attributes.", location)


def _extract_script(element: Any, location: str) -> str:
    allowed_types = {"", "application/javascript", "text/javascript"}
    script_type = ""
    for raw_name, raw_value in element.attrib.items():
        name = _attribute_name(raw_name, location)
        if name == "src":
            _fail(
                "script-source", "External script sources are not supported.", location
            )
        if name == "type":
            script_type = str(raw_value).strip().lower()
            continue
        _fail(
            "script-attribute",
            "Inline scripts have an unsupported attribute.",
            location,
        )
    if script_type == "module":
        _fail("script-module", "Module scripts are not supported.", location)
    if script_type not in allowed_types:
        _fail(
            "script-type",
            "Only classic JavaScript script types are supported.",
            location,
        )
    script = element.text or ""
    if "\x00" in script:
        _fail(
            "script-syntax", "Script source contains unsupported code points.", location
        )
    has_module_semantics, lexically_complete = _analyze_script_source(script)
    if not lexically_complete:
        _fail("script-syntax", "Script source is lexically incomplete.", location)
    if has_module_semantics:
        _fail(
            "script-module",
            "JavaScript import and export semantics are not supported.",
            location,
        )
    return script


def _analyze_script_source(script: str) -> tuple[bool, bool]:
    """Recognize module keywords and incomplete strings/comments/templates."""

    def skip_quoted(index: int, quote: str) -> tuple[int, bool]:
        while index < len(script):
            if script[index] == "\\":
                index += 2
                continue
            if script[index] == quote:
                return index + 1, True
            if quote != "`" and script[index] in "\r\n":
                return index, False
            index += 1
        return index, False

    def scan_template(index: int) -> tuple[int, bool, bool]:
        found_module = False
        while index < len(script):
            if script[index] == "\\":
                index += 2
                continue
            if script[index] == "`":
                return index + 1, found_module, True
            if script[index : index + 2] == "${":
                index, nested_module, complete = scan_code(
                    index + 2, stop_on_brace=True
                )
                found_module = found_module or nested_module
                if not complete:
                    return index, found_module, False
                continue
            index += 1
        return index, found_module, False

    def scan_code(index: int, *, stop_on_brace: bool) -> tuple[int, bool, bool]:
        found_module = False
        brace_depth = 0
        while index < len(script):
            char = script[index]
            following = script[index + 1] if index + 1 < len(script) else ""
            if char == "/" and following == "/":
                newline = min(
                    (
                        position
                        for position in (
                            script.find("\n", index + 2),
                            script.find("\r", index + 2),
                        )
                        if position >= 0
                    ),
                    default=len(script),
                )
                index = newline
                continue
            if char == "/" and following == "*":
                closing = script.find("*/", index + 2)
                if closing < 0:
                    return len(script), found_module, False
                index = closing + 2
                continue
            if char in {'"', "'"}:
                index, complete = skip_quoted(index + 1, char)
                if not complete:
                    return index, found_module, False
                continue
            if char == "`":
                index, nested_module, complete = scan_template(index + 1)
                found_module = found_module or nested_module
                if not complete:
                    return index, found_module, False
                continue
            if char == "{":
                brace_depth += 1
                index += 1
                continue
            if char == "}" and stop_on_brace:
                if brace_depth == 0:
                    return index + 1, found_module, True
                brace_depth -= 1
                index += 1
                continue
            if char == "_" or char == "$" or char.isalpha():
                end = index + 1
                while end < len(script) and (
                    script[end] == "_" or script[end] == "$" or script[end].isalnum()
                ):
                    end += 1
                found_module = found_module or script[index:end] in {"export", "import"}
                index = end
                continue
            index += 1
        return index, found_module, not stop_on_brace

    _, has_module_semantics, complete = scan_code(0, stop_on_brace=False)
    return has_module_semantics, complete


def _compile_attributes(
    element: Any,
    *,
    namespace: str,
    tag: str,
    location: str,
    assets: list[RenderAsset],
    limits: CanvasLimits,
) -> tuple[tuple[str, str], ...]:
    compiled: list[tuple[str, str]] = []
    for raw_name, raw_value in element.attrib.items():
        if _is_safe_namespace_declaration(raw_name, str(raw_value), namespace, tag):
            continue
        name = _attribute_name(raw_name, location)
        value = str(raw_value)
        if name.startswith("on"):
            _fail(
                "event-handler",
                "Native event-handler attributes are not supported.",
                location,
            )
        if name == "data-canvas-asset":
            _fail(
                "reserved-attribute",
                "A compiler-reserved attribute was supplied.",
                location,
            )

        if tag == "img" and name == "src":
            asset = _compile_asset(value, len(assets) + 1, location)
            candidate_assets = [*assets, asset]
            try:
                validate_asset_payloads(
                    [
                        DecodedDataUrl(mime_type=item.mime_type, data=item.data)
                        for item in candidate_assets
                    ],
                    per_asset_limit=limits.asset_bytes,
                    aggregate_limit=limits.aggregate_asset_bytes,
                )
            except CanvasLimitError:
                _fail(
                    "asset-limit",
                    "Image assets exceed the Canvas V1 byte limit.",
                    location,
                )
            assets.append(asset)
            compiled.append(("data-canvas-asset", asset.asset_id))
            continue

        if name in _URL_OR_NAVIGATION_ATTRIBUTES:
            if tag == "a" and name == "href" and _is_local_fragment(value):
                compiled.append((name, value))
                continue
            _fail(
                "external-url",
                "A URL, navigation, or submission attribute is not supported.",
                location,
            )

        if namespace == _HTML_NAMESPACE:
            if not _html_attribute_allowed(tag, name):
                _fail(
                    "unsupported-attribute",
                    "An HTML attribute is not supported by Canvas V1.",
                    location,
                )
            _validate_html_attribute_value(tag, name, value, location)
        elif namespace == _SVG_NAMESPACE:
            if not _global_attribute_allowed(name) and name not in _SVG_ATTRIBUTES:
                _fail(
                    "unsupported-attribute",
                    "An SVG attribute is not supported by Canvas V1.",
                    location,
                )
        if name == "style":
            value = _compile_declarations(value, location)
        compiled.append((name, value))
    return tuple(sorted(compiled))


def _global_attribute_allowed(name: str) -> bool:
    return (
        name in _GLOBAL_ATTRIBUTES
        or name.startswith("aria-")
        or name.startswith("data-")
    )


def _html_attribute_allowed(tag: str, name: str) -> bool:
    return _global_attribute_allowed(name) or name in _HTML_ATTRIBUTES.get(tag, ())


def _validate_html_attribute_value(
    tag: str, name: str, value: str, location: str
) -> None:
    if tag == "input" and name == "type" and value.strip().lower() not in _INPUT_TYPES:
        _fail(
            "unsupported-attribute-value",
            "An input type is not supported by Canvas V1.",
            location,
        )
    if (
        tag == "button"
        and name == "type"
        and value.strip().lower() not in _BUTTON_TYPES
    ):
        _fail(
            "unsupported-attribute-value",
            "A button type is not supported by Canvas V1.",
            location,
        )
    if tag == "meta" and name == "charset" and value.strip().lower() != "utf-8":
        _fail(
            "unsupported-attribute-value",
            "Only UTF-8 document metadata is supported.",
            location,
        )


def _is_local_fragment(value: str) -> bool:
    return value.startswith("#") and not any(character.isspace() for character in value)


def _compile_asset(value: str, number: int, location: str) -> RenderAsset:
    try:
        decoded = decode_data_url(value, field_name="image asset")
    except CanvasLimitError:
        _fail(
            "asset-data", "Image source must be a strict base64 data asset.", location
        )
    if decoded.mime_type not in _IMAGE_MIME_TYPES:
        _fail(
            "asset-mime",
            "Image asset MIME type is not supported by Canvas V1.",
            location,
        )
    if not _image_signature_matches(decoded.mime_type, decoded.data):
        _fail(
            "asset-signature",
            "Image asset bytes do not match the declared MIME type.",
            location,
        )
    return RenderAsset(
        asset_id=f"asset-{number}", mime_type=decoded.mime_type, data=decoded.data
    )


def _image_signature_matches(mime_type: str, data: bytes) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if mime_type == "image/webp":
        return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return False


def _compile_stylesheet(css: str, location: str) -> tuple[_CompiledCssRule, ...]:
    _validate_css_lexical_closure(css, location)
    parsed = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
    compiled: list[_CompiledCssRule] = []
    for rule in parsed:
        if rule.type == "error":
            _fail(
                "css-parse-error", "CSS contains an unsupported parse error.", location
            )
        if rule.type == "qualified-rule":
            compiled.append(_compile_qualified_rule(rule, location))
            continue
        if (
            rule.type != "at-rule"
            or rule.lower_at_keyword != "media"
            or rule.content is None
        ):
            _fail(
                "css-at-rule", "A CSS at-rule is not supported by Canvas V1.", location
            )
        _validate_css_tokens(rule.prelude, location)
        media_query = tinycss2.serialize(rule.prelude).strip()
        if not media_query:
            _fail("css-parse-error", "CSS media query is empty.", location)
        inner_rules = tinycss2.parse_rule_list(
            rule.content, skip_comments=True, skip_whitespace=True
        )
        if not inner_rules:
            _fail("css-parse-error", "CSS media rule is empty.", location)
        inner_compiled: list[_CompiledCssRule] = []
        for inner_rule in inner_rules:
            if inner_rule.type != "qualified-rule":
                _fail(
                    "css-at-rule",
                    "Nested CSS at-rules are not supported by Canvas V1.",
                    location,
                )
            inner_compiled.append(_compile_qualified_rule(inner_rule, location))
        inner_text = "".join(item.text for item in inner_compiled)
        compiled.append(
            _CompiledCssRule(
                text=f"@media {media_query}{{{inner_text}}}",
                rule_count=1 + sum(item.rule_count for item in inner_compiled),
            )
        )
    return tuple(compiled)


def _compile_qualified_rule(rule: Any, location: str) -> _CompiledCssRule:
    _validate_selector(rule.prelude, location)
    selector = tinycss2.serialize(rule.prelude).strip()
    if not selector:
        _fail("css-parse-error", "CSS selector is empty.", location)
    declarations = _compile_declaration_tokens(rule.content, location)
    return _CompiledCssRule(text=f"{selector}{{{declarations}}}", rule_count=1)


def _compile_declarations(css: str, location: str) -> str:
    _validate_css_lexical_closure(css, location)
    tokens = tinycss2.parse_declaration_list(
        css, skip_comments=True, skip_whitespace=True
    )
    return _compile_parsed_declarations(tokens, location)


def _validate_css_lexical_closure(css: str, location: str) -> None:
    """Reject EOF recovery while matching CSS Syntax token boundaries."""
    normalized = (
        css.replace("\0", "\N{REPLACEMENT CHARACTER}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\f", "\n")
    )
    expected_closers: list[str] = []
    index = 0
    while index < len(normalized):
        if normalized.startswith("/*", index):
            comment_end = normalized.find("*/", index + 2)
            if comment_end < 0:
                _fail(
                    "css-parse-error",
                    "CSS contains an unclosed lexical construct.",
                    location,
                )
            index = comment_end + 2
            continue

        character = normalized[index]
        if character in {'"', "'"}:
            index = _consume_css_string(normalized, index, location)
            continue
        if character == "\\":
            index = _consume_css_escape(normalized, index)
            continue
        if character in "{[(":
            expected_closers.append({"{": "}", "[": "]", "(": ")"}[character])
            index += 1
            continue
        if character in "}])":
            if not expected_closers or expected_closers.pop() != character:
                _fail(
                    "css-parse-error",
                    "CSS contains an unmatched lexical delimiter.",
                    location,
                )
        index += 1

    if expected_closers:
        _fail(
            "css-parse-error",
            "CSS contains an unclosed lexical construct.",
            location,
        )


def _consume_css_string(css: str, opening: int, location: str) -> int:
    quote = css[opening]
    index = opening + 1
    while index < len(css):
        if css[index] == quote:
            return index + 1
        if css[index] == "\\":
            if index + 1 < len(css) and css[index + 1] == "\n":
                index += 2
            else:
                index = _consume_css_escape(css, index)
            continue
        if css[index] == "\n":
            _fail(
                "css-parse-error",
                "CSS contains an unclosed lexical construct.",
                location,
            )
        index += 1
    _fail(
        "css-parse-error",
        "CSS contains an unclosed lexical construct.",
        location,
    )


def _consume_css_escape(css: str, backslash: int) -> int:
    index = backslash + 1
    if index >= len(css) or css[index] == "\n":
        return index
    if css[index] not in "0123456789abcdefABCDEF":
        return index + 1
    hex_end = index
    while hex_end < len(css) and hex_end - index < 6:
        if css[hex_end] not in "0123456789abcdefABCDEF":
            break
        hex_end += 1
    if hex_end < len(css) and css[hex_end] in " \n\t":
        hex_end += 1
    return hex_end


def _compile_declaration_tokens(tokens: list[Any], location: str) -> str:
    declarations = tinycss2.parse_declaration_list(
        tokens, skip_comments=True, skip_whitespace=True
    )
    return _compile_parsed_declarations(declarations, location)


def _compile_parsed_declarations(declarations: list[Any], location: str) -> str:
    compiled: list[str] = []
    for declaration in declarations:
        if declaration.type == "error":
            _fail(
                "css-parse-error",
                "CSS contains an unsupported declaration parse error.",
                location,
            )
        if declaration.type != "declaration":
            _fail(
                "css-at-rule",
                "CSS declaration lists cannot contain at-rules.",
                location,
            )
        if declaration.name.startswith("--"):
            _fail(
                "css-custom-property",
                "CSS custom properties are not supported by Canvas V1.",
                location,
            )
        if declaration.lower_name not in _CSS_PROPERTIES:
            _fail(
                "css-property",
                "A CSS property is not supported by Canvas V1.",
                location,
            )
        _validate_css_tokens(declaration.value, location)
        value = tinycss2.serialize(declaration.value).strip()
        if not value:
            _fail("css-parse-error", "CSS declaration value is empty.", location)
        important = "!important" if declaration.important else ""
        compiled.append(f"{declaration.lower_name}:{value}{important}")
    return ";".join(compiled)


def _validate_selector(tokens: list[Any], location: str) -> None:
    colon_count = 0
    for token in tokens:
        if token.type == "error":
            _fail("css-parse-error", "CSS selector contains a parse error.", location)
        if token.type == "literal" and token.value == "|":
            _fail(
                "css-namespace",
                "CSS namespaces are not supported by Canvas V1.",
                location,
            )
        if token.type == "literal" and token.value == ":":
            colon_count += 1
            continue
        if token.type in {"whitespace", "comment"}:
            continue
        if colon_count:
            pseudo_name = (
                token.value.lower()
                if token.type == "ident"
                else getattr(token, "lower_name", "")
            )
            if pseudo_name == "visited":
                _fail(
                    "css-visited",
                    "The visited-link selector is not supported.",
                    location,
                )
            if colon_count > 1:
                _fail(
                    "css-selector",
                    "CSS pseudo-elements are not supported by Canvas V1.",
                    location,
                )
            allowed = _PSEUDO_CLASSES if token.type == "ident" else _PSEUDO_FUNCTIONS
            if pseudo_name not in allowed:
                _fail(
                    "css-selector",
                    "A CSS pseudo-class is not supported by Canvas V1.",
                    location,
                )
            colon_count = 0
        _validate_one_css_token(token, location, selector=True)
    if colon_count:
        _fail(
            "css-parse-error",
            "CSS selector ends with an incomplete pseudo-class.",
            location,
        )


def _validate_css_tokens(tokens: list[Any], location: str) -> None:
    for token in tokens:
        _validate_one_css_token(token, location, selector=False)


def _validate_one_css_token(token: Any, location: str, *, selector: bool) -> None:
    if token.type == "error":
        _fail("css-parse-error", "CSS contains a parse error.", location)
    if token.type == "url":
        _fail(
            "css-resource",
            "CSS resource URLs are not supported by Canvas V1.",
            location,
        )
    if token.type == "function":
        function_name = token.lower_name
        if function_name in {
            "url",
            "src",
            "image",
            "image-set",
            "cross-fade",
            "element",
            "paint",
            "var",
            "env",
            "attr",
        }:
            _fail(
                "css-resource",
                "CSS resource or computed tokens are not supported.",
                location,
            )
        if selector:
            if function_name not in _PSEUDO_FUNCTIONS:
                _fail(
                    "css-selector",
                    "A CSS selector function is not supported by Canvas V1.",
                    location,
                )
        elif function_name not in _CSS_FUNCTIONS:
            _fail(
                "css-function",
                "A CSS function is not supported by Canvas V1.",
                location,
            )
        if selector:
            _validate_selector(token.arguments, location)
        else:
            _validate_css_tokens(token.arguments, location)
        return
    content = getattr(token, "content", None)
    if content is not None:
        if selector:
            _validate_selector(content, location)
        else:
            _validate_css_tokens(content, location)


def _freeze_tree(pending_root: _PendingNode, limits: CanvasLimits) -> RenderNode:
    preorder: list[_PendingNode] = []
    stack = [pending_root]
    while stack:
        pending = stack.pop()
        preorder.append(pending)
        if len(preorder) > limits.dom_nodes:
            _fail(
                "dom-limit",
                "Document exceeds the Canvas V1 DOM node limit.",
                "document",
            )
        stack.extend(reversed(pending.children))
    validate_count(len(preorder), limit=limits.dom_nodes, field_name="DOM nodes")

    node_ids = {
        id(pending): f"node-{number}"
        for number, pending in enumerate(preorder, start=1)
    }
    frozen: dict[int, RenderNode] = {}
    for pending in reversed(preorder):
        frozen[id(pending)] = RenderNode(
            node_id=node_ids[id(pending)],
            tag=pending.tag,
            attributes=pending.attributes,
            text=pending.text,
            children=tuple(frozen[id(child)] for child in pending.children),
        )
    return frozen[id(pending_root)]


def _issue(code: str, message: str, location: str) -> CanvasCompatibilityIssue:
    return CanvasCompatibilityIssue(code=code, message=message, location=location)


def _fail(code: str, message: str, location: str) -> NoReturn:
    raise CanvasCompileError((_issue(code, message, location),)) from None


__all__ = ["CanvasCompileError", "compile_canvas_document"]
