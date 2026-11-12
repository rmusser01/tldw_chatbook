#!/usr/bin/env python3
"""Mechanical layout metrics for rp-review plain-text captures.

Usage: metrics.py <captures-dir> <size> [<size> ...]   -> markdown table on stdout

Corpus: every user-data string in the golden profile (character cards, personas, dictionary
entries, world-book entries, notes, prompts, media, conversation titles + messages), read
from a COPY of the golden DBs (never the live files): $CORPUS_DIR, default
$HARNESS_STATE/corpus_tmp/, copied from $HARNESS_STATE/golden/data/rp_review/ on first use
(delete it after re-seeding golden).

Cell classes (every cell of the W x H screen is exactly one):
  border : box/frame glyphs  ─│┌┐└┘╭╮╰╯├┤┬┴┼▊▎▔▁┃━┏┓┗┛▕▏ and scrollbar blocks ▂▃▄▅▆▇█
  data   : glyphs of a text segment that is (part of) seeded user data
           (segment = run of text between border glyphs / 2+ spaces; after an optional
            'Label: ' prefix; trailing '…' allowed; must be a substring of the corpus,
            len>=4, case-sensitive)
  label  : every other non-space glyph (nav tabs, headings, buttons, hints, field labels,
           counts, metadata like '2 messages · 14m', vertical grip letters)
  blank  : spaces
"content %" = data cells / all cells.  "ink %" = data / (data+label+border).
"""
import re, sys, json, sqlite3, pathlib, os, shutil

sys.dont_write_bytecode = True  # keep the harness directory free of __pycache__
import harness_guard  # noqa: E402 - same directory as this script

STATE = pathlib.Path(harness_guard.check_state_dir(os.environ.get("HARNESS_STATE") or harness_guard.default_state()))
CT = pathlib.Path(os.environ.get("CORPUS_DIR") or STATE / "corpus_tmp")
CORPUS_FILES = ["tldw_chatbook_ChaChaNotes.db", "tldw_chatbook_prompts.db", "tldw_chatbook_media_v2.db",
                "tldw_chatbook_personas.json"]


def ensure_corpus():
    """Copy the golden DBs (+ WAL sidecars) once; the copies are what we open."""
    if (CT / CORPUS_FILES[0]).exists():
        return
    src = STATE / "golden" / "data" / "rp_review"
    if not (src / CORPUS_FILES[0]).exists():
        sys.exit(f"no corpus: {src} has no golden DBs (run make_profiles.sh) and {CT} is empty")
    CT.mkdir(parents=True, exist_ok=True)
    for name in CORPUS_FILES:
        for suffix in ("", "-wal", "-shm"):
            f = src / (name + suffix)
            if f.exists():
                shutil.copy2(f, CT / f.name)
BORDER = set("─│┌┐└┘╭╮╰╯├┤┬┴┼▊▎▔▁┃━┏┓┗┛▕▏▂▃▄▅▆▇█╴╶╷╵")


def corpus():
    ensure_corpus()
    texts, names = [], {"character": [], "persona": [], "dictionary": [], "lore": [],
                        "note": [], "prompt": [], "media": [], "conversation": []}
    c = sqlite3.connect(CT / "tldw_chatbook_ChaChaNotes.db")
    for row in c.execute("select name, description, personality, scenario, system_prompt, first_message,"
                         " message_example, creator_notes, alternate_greetings, tags, creator, character_version"
                         " from character_cards where deleted=0"):
        names["character"].append(row[0]); texts += [str(x) for x in row if x]
    for row in c.execute("select name, description from chat_dictionaries where deleted=0"):
        names["dictionary"].append(row[0]); texts += [str(x) for x in row if x]
    for row in c.execute("select content from chat_dictionaries where deleted=0"):
        texts.append(str(row[0] or ""))
    for row in c.execute("select name, description from world_books where deleted=0"):
        names["lore"].append(row[0]); texts += [str(x) for x in row if x]
    for row in c.execute("select keys, content from world_book_entries"):
        texts += [str(x) for x in row if x]
        try:
            texts += [", ".join(json.loads(row[0]))]
        except Exception:
            pass
    for row in c.execute("select title, content from notes where deleted=0"):
        names["note"].append(row[0]); texts += [str(x) for x in row if x]
    for row in c.execute("select title from conversations where deleted=0"):
        if row[0]:
            names["conversation"].append(row[0]); texts.append(row[0])
    for row in c.execute("select content from messages where deleted=0"):
        texts.append(str(row[0] or ""))
    p = sqlite3.connect(CT / "tldw_chatbook_prompts.db")
    for row in p.execute("select name, details, system_prompt, user_prompt from Prompts where deleted=0"):
        names["prompt"].append(row[0]); texts += [str(x) for x in row if x]
    m = sqlite3.connect(CT / "tldw_chatbook_media_v2.db")
    for row in m.execute("select title, content from Media where deleted=0"):
        names["media"].append(row[0]); texts += [str(x) for x in row if x]
    pj = json.loads((CT / "tldw_chatbook_personas.json").read_text())
    for rec in (pj.get("profiles") or pj.get("persona_profiles") or []):
        if not rec.get("deleted"):
            names["persona"].append(rec.get("name", ""))
            texts += [str(rec.get(k) or "") for k in ("name", "description", "system_prompt", "personality_traits")]
    big = "\n".join(texts)
    norm = re.sub(r"\s+", " ", big)
    # markdown emphasis is rendered without '*' in some panes: index a de-starred copy too
    return norm + "\n" + norm.replace("*", ""), names


SINGLES = set()
CORPUS, NAMES = corpus()
SEG = re.compile(r"[^\s" + re.escape("".join(BORDER)) + r"](?:[^" + re.escape("".join(BORDER)) + r"\s]| (?! ))*")


def is_data(seg: str) -> bool:
    s = seg.strip().rstrip("…").rstrip(".").strip()
    if ": " in s and s not in CORPUS:
        s = s.split(": ", 1)[1].strip()
    if len(s) < 3:
        return False
    if " " not in s:                      # single word: only exact item names / entry keys
        return s in SINGLES or any(n.startswith(s) for n in SINGLES if len(s) >= 6)
    return s in CORPUS


def classify(lines, W, H, first_row=0):
    grid = [list(l.ljust(W)[:W]) for l in lines[:H]] + [[" "] * W for _ in range(max(0, H - len(lines)))]
    cls = [["blank"] * W for _ in range(H)]
    data_rows = set()
    for r, row in enumerate(grid):
        line = "".join(row)
        for ch_i, ch in enumerate(row):
            if ch in BORDER:
                cls[r][ch_i] = "border"
            elif ch != " ":
                cls[r][ch_i] = "label"
        if r < first_row:
            continue
        parts = []
        for mt in SEG.finditer(line):
            off = mt.start()
            for piece in mt.group(0).split(" · "):     # "Title · 2m · Default" -> evaluate parts separately
                parts.append((off, off + len(piece), piece)); off += len(piece) + 3
        for start, end, seg in parts:
            if is_data(seg):
                lab_off = 0
                if seg.strip() not in CORPUS and ": " in seg:
                    lab_off = seg.index(": ") + 2
                for k in range(start + lab_off, end):
                    if grid[r][k] not in (" ",) and grid[r][k] not in BORDER:
                        cls[r][k] = "data"
                data_rows.add(r)
    return grid, cls, data_rows


def boxes_on(row):
    """(start,end) col pairs (0-based, inclusive) of bordered boxes opened on this row."""
    out, st = [], None
    for i, ch in enumerate(row):
        if ch in "┌╭┏" and st is None:
            st = i
        elif ch in "┐╮┓" and st is not None:
            out.append((st, i)); st = None
    return out


def analyze(path: pathlib.Path):
    lines = path.read_text().rstrip("\n").split("\n")
    m = re.search(r"-(\d+)x(\d+)$", path.stem)
    W, H = int(m.group(1)), int(m.group(2))
    grid0 = [list(l.ljust(W)[:W]) for l in lines[:H]]
    pt = None
    for r in range(3, min(H, len(grid0))):
        bx = boxes_on(grid0[r])
        if bx and (len(bx) >= 2 or (bx[0][1] - bx[0][0]) < W - 4):
            pt = r; break
    if pt is None:
        for r in range(3, min(H, len(grid0))):
            if boxes_on(grid0[r]):
                pt = r; break
    grid, cls, data_rows = classify(lines, W, H, first_row=(pt + 1) if pt is not None else 3)
    tot = W * H
    cnt = {k: sum(row.count(k) for row in cls) for k in ("data", "label", "border", "blank")}
    # first row of pane interiors: the first row (>3) whose box layout has >=2 boxes or
    # a box narrower than the screen, +1
    pane_top = None
    for r in range(3, H):
        bx = boxes_on(grid[r])
        if bx and (len(bx) >= 2 or (bx[0][1] - bx[0][0]) < W - 4):
            pane_top = r; break
    if pane_top is None:                       # borderless panes (Library grips + list + reader): outer frame
        for r in range(3, H):
            if boxes_on(grid[r]):
                pane_top = r; break
    top_chrome = (pane_top + 1) if pane_top is not None else None   # rows above first pane-interior row
    first_data = (min(data_rows) + 1) if data_rows else None        # 1-based row of first data text
    # bottom chrome: trailing rows with no data and only border/blank/hint
    bottom = None
    if pane_top is not None:
        starts = [a for a, _ in boxes_on(grid[pane_top])]
        for r in range(H - 1, pane_top, -1):
            if any(grid[r][a] in "╰└┗" for a in starts):
                bottom = H - r; break
    # column budget on the pane-top row
    budget = ""
    regions = None
    if pane_top is not None:
        bx = boxes_on(grid[pane_top])
        widths = [b - a + 1 for a, b in bx]
        # gaps: grips / borderless panes / frame margins
        gaps, prev = [], 0
        for a, b in bx:
            gaps.append(a - prev); prev = b + 1
        gaps.append(W - prev)
        budget = "boxes " + "+".join(map(str, widths)) + f" / gaps {'+'.join(map(str, gaps))}"
        regions, prev = [], 0
        for a_, b_ in bx:
            if a_ - prev > 0:
                regions.append((prev, a_ - 1))
            regions.append((a_, b_)); prev = b_ + 1
        regions.append((prev, W - 1))
    # list items visible: item names (any type) whose occurrence starts in the leftmost box that has hits
    return dict(W=W, H=H, tot=tot, **cnt, top=top_chrome, first_data=first_data, bottom=bottom,
                budget=budget, grid=grid, regions=regions)


def count_items(grid, kinds, first_row=0, regions=None):
    """Distinct item names visible inside the ONE pane region holding the most of them
    (so a name repeated in a detail/inspector pane is not double-counted)."""
    names = [n for k in kinds for n in NAMES[k] if n]
    hits = {}
    for r, row in enumerate(grid):
        if r < first_row:
            continue
        line = "".join(row)
        for n in names:
            for probe in (n, n[: max(6, len(n) // 2)].rstrip()):
                if len(probe) >= 4:
                    c = line.find(probe)
                    if c >= 0:
                        reg = 0
                        for i, (a, b) in enumerate(regions or [(0, len(line))]):
                            if a <= c <= b:
                                reg = i; break
                        hits.setdefault(reg, {}).setdefault(n, r + 1)
                        break
    if not hits:
        return 0, None
    best = max(hits.values(), key=len)
    return len(best), min(best.values())


EDITOR_LABELS = ["Name", "First message", "Description", "Personality", "System prompt", "Voice & Speech",
                 "Scenario", "Post-history instructions", "Creator notes", "Alternate greetings", "Creator",
                 "Version", "Tags (comma-separated)", "Avatar:", "Title", "Keywords", "Body", "Instructions",
                 "Message template"]


def fields_visible(grid):
    txt = ["".join(r) for r in grid]
    seen = 0
    for lab in EDITOR_LABELS:
        pat = re.compile(r"(^|[│┃ ])" + re.escape(lab) + r"(  |$| Generate|\s*$)")
        if any(pat.search(t) for t in txt):
            seen += 1
    return seen


def main():
    cdir = pathlib.Path(sys.argv[1]); sizes = sys.argv[2:]
    rows = []
    for size in sizes:
        for p in sorted(cdir.glob(f"*-{size}.txt")):
            if not (p.name.startswith("roleplay-") or p.name.startswith("library-")):
                continue
            a = analyze(p)
            g = a["grid"]
            if "roleplay-dictionar" in p.name:
                kinds = ["dictionary"]
            elif "roleplay-lore" in p.name:
                kinds = ["lore"]
            elif "roleplay-persona" in p.name:
                kinds = ["persona"]
            elif p.name.startswith("roleplay-"):
                kinds = ["character"]
            elif "notes" in p.name:
                kinds = ["note"]
            elif "prompts" in p.name:
                kinds = ["prompt"]
            elif "media" in p.name:
                kinds = ["media"]
            elif "conversations" in p.name:
                kinds = ["conversation"]
            else:
                kinds = ["note", "media", "conversation", "prompt"]
            items, first_item = count_items(g, kinds, first_row=a["top"] or 3, regions=a["regions"])
            flds = fields_visible(g) if ("editor" in p.name or "item-open" in p.name and "prompts" in p.name) else ""
            pct = lambda k: f"{100 * a[k] / a['tot']:.1f}"
            ink = a["data"] + a["label"] + a["border"]
            rows.append([p.stem, a["top"], first_item, a["first_data"], a["bottom"], a["budget"], items, flds,
                         pct("data"), pct("label"), pct("border"), pct("blank"),
                         f"{100 * a['data'] / ink:.0f}" if ink else "-"])
    hdr = ["capture", "rows above pane interior", "1st list-item row", "1st data row", "bottom chrome rows", "column budget (pane-top row)",
           "items visible", "editor fields visible", "data %", "label %", "border %", "blank %", "data share of ink %"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join("" if v is None else str(v) for v in r) + " |")


if __name__ == "__main__":
    main()
