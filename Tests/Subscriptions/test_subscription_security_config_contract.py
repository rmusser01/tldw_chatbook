"""Source-level ownership contracts for subscription URL security policy."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

from tldw_chatbook.config import CONFIG_TOML_CONTENT
from tldw_chatbook.Subscriptions.security import SecurityValidator


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
CANONICAL_METADATA_ENDPOINTS = frozenset(
    {
        "169.254.169.254",
        "100.100.100.200",
        "fd00:ec2::254",
        "metadata.google.internal",
        "metadata.azure.com",
    }
)
EGRESS_POLICY_PATH = Path("Utils/egress.py")
DISALLOWED_SCHEMES = frozenset({"file", "ftp", "gopher", "javascript", "data"})


def _production_python_files() -> list[Path]:
    """Return Python sources belonging to the shipped application package."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _source_tree(source_path: Path) -> ast.Module:
    """Parse one production source file without reading comments as policy."""
    return ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def _literal_collection_values(node: ast.AST) -> set[str] | None:
    """Return literal string elements for a supported source collection."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if len(values) == len(node.elts):
            return values
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _literal_collection_values(node.args[0])
    return None


def test_shipped_subscription_config_has_no_security_child_table() -> None:
    """Subscription security policy is not duplicated in generated config."""
    config_defaults = tomllib.loads(CONFIG_TOML_CONTENT)

    assert "subscriptions" in config_defaults
    assert "security" not in config_defaults["subscriptions"]


def test_canonical_metadata_endpoints_are_owned_only_by_egress_policy() -> None:
    """Metadata endpoint literals have one canonical production policy owner."""
    endpoint_paths = {endpoint: set() for endpoint in CANONICAL_METADATA_ENDPOINTS}

    for source_path in _production_python_files():
        relative_path = source_path.relative_to(PACKAGE_ROOT)
        for node in ast.walk(_source_tree(source_path)):
            if isinstance(node, ast.Constant) and node.value in endpoint_paths:
                endpoint_paths[node.value].add(relative_path)

    for endpoint, paths in endpoint_paths.items():
        assert paths == {EGRESS_POLICY_PATH}, (
            f"{endpoint} must be declared only in {EGRESS_POLICY_PATH}; found {paths}"
        )


def test_disallowed_url_scheme_collections_are_not_duplicated() -> None:
    """The egress allowlist, rather than blocked-scheme tables, owns scheme policy."""
    violations: list[tuple[Path, int, set[str]]] = []

    for source_path in _production_python_files():
        relative_path = source_path.relative_to(PACKAGE_ROOT)
        for node in ast.walk(_source_tree(source_path)):
            values = _literal_collection_values(node)
            if values is not None and len(values & DISALLOWED_SCHEMES) >= 3:
                violations.append((relative_path, node.lineno, values & DISALLOWED_SCHEMES))

    assert not violations, f"duplicate disallowed URL-scheme policy: {violations}"


def test_subscription_validator_retains_only_its_http_scheme_boundary() -> None:
    """Subscription validation keeps its boundary but delegates shared policy."""
    assert "BLOCKED_SCHEMES" not in SecurityValidator.__dict__
    assert "METADATA_ENDPOINTS" not in SecurityValidator.__dict__
    assert SecurityValidator.ALLOWED_SCHEMES == {"http", "https"}
