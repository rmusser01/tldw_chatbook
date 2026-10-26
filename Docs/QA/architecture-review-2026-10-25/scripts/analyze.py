import json, sys
from collections import defaultdict, Counter

g = json.load(open(sys.argv[1]))
mods, edges, loc, ext = g["mods"], g["edges"], g["loc"], g["ext"]


def pkg(m):
    p = m.split(".")
    if len(p) == 1:
        return "<root>"
    seg = p[1]
    if len(p) == 2 and m + ".__init__" not in mods and not any(k.startswith(m + ".") for k in mods):
        # root-level module file
        if seg == "app" or seg.startswith("app_"):
            return "app(shell)"
        return seg + ".py"
    return seg


def scc(nodes, adj):
    idx, low, on, st, out, c = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(100000)

    def sc(v):
        idx[v] = low[v] = c[0]; c[0] += 1; st.append(v); on.add(v)
        for w in adj.get(v, ()):
            if w not in idx:
                sc(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = st.pop(); on.discard(w); comp.append(w)
                if w == v:
                    break
            out.append(comp)
    for v in nodes:
        if v not in idx:
            sc(v)
    return [c for c in out if len(c) > 1]


section = sys.argv[2] if len(sys.argv) > 2 else "all"

# ---- module-level cycles (top-level imports only => import-time cycles)
top = defaultdict(set)
for s, d, k, _ in edges:
    if k == "top" and s != d:
        # importing a submodule also runs the parent package __init__, ignore that implicit edge
        top[s].add(d)
cyc = scc(list(mods), top)
cyc.sort(key=len, reverse=True)
if section in ("all", "cycles"):
    print("== import-time module SCCs (top-level imports):", len(cyc), "sizes", [len(c) for c in cyc[:25]])
    for c in cyc[:6]:
        pk = Counter(pkg(m) for m in c)
        print("  size", len(c), dict(pk.most_common(12)))

alle = defaultdict(set)
for s, d, k, _ in edges:
    if s != d:
        alle[s].add(d)
cyc2 = scc(list(mods), alle)
cyc2.sort(key=len, reverse=True)
if section in ("all", "cycles"):
    print("== module SCCs incl lazy+typing imports:", len(cyc2), "sizes", [len(c) for c in cyc2[:15]])
    for c in cyc2[:2]:
        pk = Counter(pkg(m) for m in c)
        print("  size", len(c), "packages", len(pk), dict(pk.most_common(25)))

# ---- package graph
pe = defaultdict(Counter)  # (src pkg) -> Counter(dst pkg) all kinds
pet = defaultdict(Counter)
for s, d, k, _ in edges:
    a, b = pkg(s), pkg(d)
    if a != b:
        pe[a][b] += 1
        if k == "top":
            pet[a][b] += 1
pkgs = sorted(set(pkg(m) for m in mods))
ploc = Counter()
pfiles = Counter()
for m in mods:
    ploc[pkg(m)] += loc[m]; pfiles[pkg(m)] += 1
ca = Counter()
for a in pe:
    for b in pe[a]:
        ca[b] += 1
if section in ("all", "pkg"):
    print("== package metrics (all import kinds; Ce/Ca = distinct packages)")
    rows = []
    for p in pkgs:
        ce = len(pe[p]); a = ca[p]
        inst = ce / (ce + a) if ce + a else 0
        rows.append((ploc[p], p, pfiles[p], ce, a, inst))
    rows.sort(reverse=True)
    print(f"{'pkg':28}{'loc':>8}{'files':>6}{'Ce':>5}{'Ca':>5}{'I':>6}")
    for l, p, f, ce, a, i in rows[:45]:
        print(f"{p:28}{l:8}{f:6}{ce:5}{a:5}{i:6.2f}")
    pc = scc(pkgs, {k: set(v) for k, v in pe.items()})
    print("== package SCCs (all kinds):", [len(c) for c in pc])
    for c in pc:
        print("  ", sorted(c)[:200])
    pct = scc(pkgs, {k: set(v) for k, v in pet.items()})
    print("== package SCCs (top-level only):", [len(c) for c in pct])
    for c in pct:
        print("  ", len(c), sorted(c))

if section in ("all", "cycles", "pkg"):
    exe = defaultdict(set)
    for s, d, k, _ in edges:
        if k in ("top", "lazy") and s != d:
            exe[s].add(d)
    ce = sorted(scc(list(mods), exe), key=len, reverse=True)
    print("== executable (top+lazy) largest SCC:", len(ce[0]), "modules /", len({pkg(m) for m in ce[0]}), "packages")
    mut = lambda gr: sum(1 for a in gr for b in gr[a] if a < b and a in gr.get(b, ()))
    print("== mutually-importing package pairs: top-level", mut(pet), "/ all kinds", mut(pe))

json.dump({k: dict(v) for k, v in pe.items()}, open(sys.argv[1] + ".pkg.json", "w"))
