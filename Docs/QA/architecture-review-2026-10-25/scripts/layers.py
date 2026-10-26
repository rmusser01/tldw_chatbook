import json, sys
from collections import defaultdict, Counter
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
g = json.load(open(sys.argv[1]))
mods, edges, loc, ext = g["mods"], g["edges"], g["loc"], g["ext"]

PRES = {"UI", "Widgets", "Event_Handlers", "css", "app(shell)", "cli.py", "Web_Server"}
KERNEL = {"Utils", "Logging_Config.py", "Constants.py", "config.py", "Third_Party", "Metrics", "state",
          "runtime_policy", "<root>"}
PERSIST = {"DB"}


def pkg(m):
    p = m.split(".")
    if len(p) == 1:
        return "<root>"
    seg = p[1]
    if len(p) == 2 and not any(k.startswith(m + ".") for k in mods):
        if seg == "app" or seg.startswith("app_"):
            return "app(shell)"
        return seg + ".py"
    return seg


def rank(p):
    if p in PRES: return 3
    if p in PERSIST: return 1
    if p in KERNEL: return 0
    return 2


up = Counter(); upmods = defaultdict(Counter); upsamples = defaultdict(list)
for s, d, k, line in edges:
    a, b = pkg(s), pkg(d)
    if a == b: continue
    ra, rb = rank(a), rank(b)
    if rb > ra:
        key = (["kernel", "DB", "domain", "pres"][ra], ["kernel", "DB", "domain", "pres"][rb], k)
        up[key] += 1
        upmods[key][s] += 1
        if len(upsamples[key]) < 400:
            upsamples[key].append((s, d, line))
print("== UPWARD edges (lower layer imports higher layer), by kind")
for k, v in sorted(up.items()):
    print(" ", k, v, "edges from", len(upmods[k]), "modules")

# domain -> presentation: which source packages / target modules
dp = Counter(); dpt = Counter(); dps = Counter()
for s, d, k, line in edges:
    if rank(pkg(s)) == 2 and rank(pkg(d)) == 3:
        dp[(pkg(s), k)] += 1; dpt[d] += 1; dps[s] += 1
print("\n== domain->presentation by source pkg/kind (top 30)")
for (p, k), n in dp.most_common(30): print(f"   {p:24}{k:7}{n}")
print("== most-imported presentation targets from domain")
for d, n in dpt.most_common(20): print("  ", n, d)

# kernel -> higher
print("\n== kernel(Utils/config/...) -> higher layers")
kc = Counter()
for s, d, k, line in edges:
    if rank(pkg(s)) == 0 and rank(pkg(d)) > 0:
        kc[(s, pkg(d), k)] += 1
for (s, p, k), n in kc.most_common(40): print(f"   {n:3} {s} -> {p} [{k}]")

# DB -> domain/pres
print("\n== DB -> domain/presentation")
dc = Counter()
for s, d, k, line in edges:
    if pkg(s) == "DB" and rank(pkg(d)) >= 2:
        dc[(s.split('.')[-1], d, k)] += 1
for (s, d, k), n in dc.most_common(60): print(f"   {s} -> {d} [{k}]")

# presentation -> DB direct
pd = Counter(); pdm = set()
for s, d, k, line in edges:
    if rank(pkg(s)) == 3 and pkg(d) == "DB":
        pd[(pkg(s), k)] += 1; pdm.add(s)
print("\n== presentation -> DB edges", dict(pd), "distinct pres modules", len(pdm))
pres_mods = [m for m in mods if rank(pkg(m)) == 3]
print("   presentation modules total", len(pres_mods))

# textual imported outside presentation
tx = Counter(); txm = []
for m, libs in ext.items():
    if "textual" in libs and rank(pkg(m)) < 3:
        tx[pkg(m)] += 1; txm.append(m)
print("\n== non-presentation modules importing textual at top level:", len(txm))
for p, n in tx.most_common(40): print(f"   {p:24}{n}")
json.dump(txm, open(sys.argv[1] + ".txm.json", "w"))
