"""Build the tldw_chatbook import graph from source (AST, no imports executed)."""
import ast, json, os, sys
from collections import defaultdict

ROOT = sys.argv[1]
PKG = "tldw_chatbook"
base = os.path.join(ROOT, PKG)

mods = {}  # modname -> path
for dp, dn, fn in os.walk(base):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, ROOT)[:-3].replace(os.sep, ".")
            if rel.endswith(".__init__"):
                rel = rel[: -len(".__init__")]
            mods[rel] = p


def resolve(name):
    """Longest prefix of name that is a known module."""
    parts = name.split(".")
    while parts:
        n = ".".join(parts)
        if n in mods:
            return n
        parts.pop()
    return None


def is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


edges = []  # (src, dst, kind, line)  kind: top | lazy | typing
ext = defaultdict(set)  # module -> external top-level libs (top-level imports)
loc = {}
for m, p in mods.items():
    src = open(p, encoding="utf-8", errors="replace").read()
    loc[m] = src.count("\n") + 1
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    is_pkg = p.endswith("__init__.py")

    def visit(node, kind):
        for child in ast.iter_child_nodes(node):
            k = kind
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                k = "lazy" if kind == "top" else kind
            if isinstance(child, ast.If) and is_type_checking(child.test):
                for b in child.body:
                    visit_stmt(b, "typing")
                for b in child.orelse:
                    visit_stmt(b, kind)
                continue
            visit_stmt(child, k)

    def visit_stmt(child, k):
        if isinstance(child, ast.Import):
            for a in child.names:
                t = resolve(a.name) if a.name.startswith(PKG) else None
                if t:
                    edges.append((m, t, k, child.lineno))
                elif k == "top":
                    ext[m].add(a.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            if child.level:
                pkgparts = m.split(".") if is_pkg else m.split(".")[:-1]
                up = pkgparts[: len(pkgparts) - (child.level - 1)] if child.level > 1 else pkgparts
                basename = ".".join(up + ([child.module] if child.module else []))
            else:
                basename = child.module or ""
            if basename.startswith(PKG):
                for a in child.names:
                    t = resolve(basename + "." + a.name) or resolve(basename)
                    if t:
                        edges.append((m, t, k, child.lineno))
            elif k == "top" and basename:
                ext[m].add(basename.split(".")[0])
        if not isinstance(child, (ast.Import, ast.ImportFrom)):
            visit(child, k)

    visit(tree, "top")

json.dump({"mods": mods, "edges": edges, "ext": {k: sorted(v) for k, v in ext.items()}, "loc": loc},
          open(sys.argv[2], "w"))
print(len(mods), "modules", len(edges), "edges")
