"""Assemble report.md Findings + appendix sections from the per-slice phase2 reports."""
import pathlib, re, sys
SP = pathlib.Path("/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/5cdb48ca-db0f-47a4-930a-9ef5b33bceed/scratchpad")
COMPLETE = ["ENTRY-app","ENTRY-config","CHAT-controller","CHAT-store","CHAT-bridge","CHAT-rest-1","AGENTS","TOOLS-MCP","DB-chacha","DB-media-base","DB-rest","LLM","UTILS","RAG","EVENTS","UI-chat","UI-library","UI-settings","UI-personas","UIM-console","UIM-library"]
def sections(text):
    """split a report into (heading, body) blocks at '## '"""
    out = {}
    cur = None; buf = []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur: out[cur] = "\n".join(buf)
            cur = line[3:].strip(); buf = []
        elif cur: buf.append(line)
    if cur: out[cur] = "\n".join(buf)
    return out
findings = []   # (sev, dim, slice, text)
other = {"Candidate dispositions": [], "Verified-fine": [], "Retired": [], "Left UNVERIFIED": [], "Coverage": [], "Dead or under-adopted shared helpers": []}
for name in COMPLETE:
    p = SP/"phase2"/f"{name}.md"
    if not p.exists(): print("MISSING", name); continue
    text = p.read_text(encoding="utf-8")
    secs = sections(text)
    for k in other:
        for kk, vv in secs.items():
            if kk.lower().startswith(k.lower()):
                other[k].append((name, vv.strip()))
    body = next((v for k, v in secs.items() if k.lower().startswith("findings")), "")
    blocks = re.split(r"\n(?=### )", body)
    for b in blocks:
        m = re.match(r"### (P\d)\s*\[?(D\d)?", b.strip())
        if not m: continue
        findings.append((m.group(1), m.group(2) or "D?", name, b.strip()))
order = {"P0":0,"P1":1,"P2":2,"P3":3}
findings.sort(key=lambda f: (order.get(f[0],9), f[1], f[2]))
out = SP/"assembly"
out.mkdir(exist_ok=True)
with open(out/"findings_all.md","w") as f:
    for sev, dim, name, text in findings:
        first = text.splitlines()[0]
        rest = "\n".join(text.splitlines()[1:])
        f.write(f"{first}  ·  _slice: {name}_\n{rest}\n\n")
for k, v in other.items():
    with open(out/(k.lower().replace(" ","_")+".md"),"w") as f:
        for name, body in v:
            f.write(f"\n### {name}\n{body}\n")
print("findings:", {s: sum(1 for x in findings if x[0]==s) for s in ("P0","P1","P2","P3")})
print("by dimension:", {d: sum(1 for x in findings if x[1]==d) for d in sorted({x[1] for x in findings})})
for sev in ("P0","P1"):
    print(f"\n--- {sev} ---")
    for s,d,n,t in findings:
        if s==sev: print(f"  [{d}] {n}: {t.splitlines()[0][len('### '):][:150]}")
