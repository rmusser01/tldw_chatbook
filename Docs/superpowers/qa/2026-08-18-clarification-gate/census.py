"""TASK-16072 Step 0, REDONE (Qodo PR-1791 findings 1+2).

Finding 1 was right: counting RELEVANCE LABELS asks what the fixture intends,
not what the corpus CONTAINS. A gate fires on the corpus. So condition 2 is
now measured as: how many corpus documents are plausible readings of this
query -- i.e. contain the query's content words -- versus how many are
labelled relevant. Matching > relevant means the corpus holds an alternative
reading the query does not distinguish.

Emits the PER-QUERY answer AC#1 requires, for all 60.
"""
import re
from Tests.RAG_Eval.harness.goldenset import load_fixtures

STOP = {"a","an","the","of","for","to","in","on","and","or","is","are","was",
        "were","be","been","with","at","by","from","how","what","when","which",
        "do","does","did","can","should","would","my","our","that","this","it"}

corpus, golden = load_fixtures()
docs = {d.slug: f"{getattr(d,'title','')}\n{getattr(d,'content','')}".lower()
        for d in corpus}

def content_words(q):
    return [w for w in re.findall(r"[a-z0-9\-]+", q.lower()) if w not in STOP and len(w) > 2]

rows, qualifying = [], []
for g in golden:
    cw = content_words(g.query)
    rel = set(g.relevant_slugs or ())
    # a "plausible reading": a document containing EVERY content word
    matching = {s for s, txt in docs.items() if cw and all(w in txt for w in cw)}
    alt = matching - rel
    qual = bool(rel) and len(alt) >= 1
    if qual:
        qualifying.append(g)
    rows.append((g.query, g.category, len(rel), len(matching), len(alt), qual))

print("PROBE PROOF: docs with non-empty text:", sum(1 for v in docs.values() if len(v)>50), "of", len(docs))
print(f"{'query':46s} {'category':20s} rel match alt  gate?")
for q, c, r, m, a, ql in sorted(rows, key=lambda x: (not x[5], x[1])):
    print(f"{q[:45]:46s} {c:20s} {r:3d} {m:5d} {a:4d}  {'YES' if ql else '-'}")
print(f"\nQUALIFYING (corpus holds an unlabelled alternative reading): {len(qualifying)} of {len(golden)}")
for g in qualifying:
    print(f"   {g.query!r} [{g.category}]")
