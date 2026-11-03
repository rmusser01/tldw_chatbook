"""Reverse only the approved routing/global lookup changes and compare bodies."""

import ast
import inspect
import json
from pathlib import Path

root = Path(__file__).parent
original = json.loads((root / "original-bodies.json").read_text())
recipients = {
    "library": "tldw_chatbook/UI/Library_Modules/screen_helpers.py",
    "chat": "tldw_chatbook/UI/Console_Modules/command_handoff.py",
}


class NormalizeDocs(ast.NodeTransformer):
    def visit_Module(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body[0].value.value = inspect.cleandoc(node.body[0].value.value)
        return node


proof = []
for key, path in recipients.items():
    rows = [row for row in original["records"] if row["key"] == key]
    names = {row["method"].lstrip("_"): row["method"] for row in rows}
    moved = {
        node.name: node
        for node in ast.parse(Path(path).read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    class Reverse(ast.NodeTransformer):
        def visit_Subscript(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "source"
                and isinstance(node.slice, ast.Constant)
            ):
                return ast.Name(node.slice.value, ast.Load())
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in names:
                node.func = ast.Attribute(
                    ast.Name("self", ast.Load()), names[node.func.id], ast.Load()
                )
                node.args = node.args[1:]
                node.keywords = [item for item in node.keywords if item.arg != "source"]
            elif (
                key == "library"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_has_local_sources"
            ):
                node.func.attr = "_has_source_study_context"
            return node

    for row in rows:
        prior = ast.parse(row["source"]).body[0]
        if row["method"] == "_has_source_study_context":
            assert ast.unparse(prior.body[0]) == "return self._has_local_sources()"
            proof.append({"method": row["method"], "redundant_predicate": True})
            continue
        current = Reverse().visit(moved[row["method"].lstrip("_")])
        before = NormalizeDocs().visit(ast.Module(body=prior.body, type_ignores=[]))
        after = NormalizeDocs().visit(ast.Module(body=current.body, type_ignores=[]))
        equal = ast.dump(before, include_attributes=False) == ast.dump(
            after, include_attributes=False
        )
        assert equal, row["method"]
        proof.append({"method": row["method"], "equal": equal})

(root / "ast-reverse-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(f"{len(proof)} exact body/predicate checks passed")
