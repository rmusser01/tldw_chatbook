"""TASK-17855 AC#1: characterise the residual zero-row queries.

The question is whether a keyword construction could EVER reach the target.
A construction can only rearrange the terms the user typed, so the decisive
test is lexical overlap: does the relevant document contain ANY content word
from the query? If not, no AND/OR/prefix/stopword arrangement of those terms
can retrieve it, and the residual is the semantic leg's job by construction.
"""
import re, tempfile, pathlib, collections
from Tests.RAG_Eval.harness.goldenset import load_fixtures
from Tests.RAG_Eval.harness.ingest import build_eval_runtime
from Tests.RAG_Eval.harness.runner import run_eval

STOP = {"a","an","the","of","for","to","in","on","and","or","is","are","was","were",
        "be","been","with","at","by","from","how","what","when","which","do","does",
        "did","can","should","would","my","our","that","this","it","not","no","if",
        "you","i","me","us","they","them","there","here","about","into","than","then"}

corpus, golden = load_fixtures()
docs = {d.slug: (d.title + "\n" + d.content).lower() for d in corpus}
print("PROBE PROOF: docs with text:", sum(1 for v in docs.values() if len(v) > 50), "of", len(docs))

tmp = pathlib.Path(tempfile.mkdtemp(prefix="zr17855-"))
runtime = build_eval_runtime(corpus, tmp)
try:
    rep = run_eval(runtime, golden, k=10, modes=("plain",))
    qs = rep.modes["plain"].queries
finally:
    runtime.close()

def cw(q):
    return [w for w in re.findall(r"[a-z0-9\-]+", q.lower()) if w not in STOP and len(w) > 2]

zero_scored = [q for q in qs if q.rows_returned == 0 and (q.relevant_slugs or ())]
print(f"residual zero-row queries WITH ground truth: {len(zero_scored)}")

buckets = collections.Counter()
detail = []
for q in zero_scored:
    words = cw(q.query)
    targets = [docs.get(s, "") for s in (q.relevant_slugs or ())]
    # how many query content words appear in ANY relevant document?
    present = [w for w in words if any(w in t for t in targets)]
    if not present:
        bucket = "UNREACHABLE (no shared content word)"
    elif len(present) == len(words):
        bucket = "all words present (construction could reach it)"
    else:
        bucket = f"partial ({len(present)}/{len(words)} words present)"
    buckets[bucket] += 1
    detail.append((q.category, q.query[:44], len(words), len(present), bucket))

print()
for b, n in buckets.most_common():
    print(f"  {n:3d}  {b}")
print()
for cat, qtext, nw, npres, b in sorted(detail):
    if not b.startswith("UNREACHABLE"):
        print(f"   REACHABLE? [{cat:20s}] {qtext!r}  {npres}/{nw} words present")
