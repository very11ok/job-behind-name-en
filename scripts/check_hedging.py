"""check_hedging.py — hedging detection for the repository's documentation.

Lexicon source:
  aproko/keyword_hedge_detector's chinese_hedges.csv (github.com/aproko/keyword_hedge_detector),
  filtered to technical-documentation hedges and epistemic-modality words after removing colloquial
  fillers (um/er/this/that/someone etc.), supplemented with English hedge words
  (Hyland 1998 categories: content-oriented relational hedges).
Usage: python check_hedging.py [directory]   by default scans all .md under gh_publish
Principle (GMU Writing Center / EAP Foundation guidance): hedge only where evidence is genuinely
uncertain; do not hedge conclusions backed by numbers; no strong claim with weak support; delete
vague qualifiers outright.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Level A: conclusion-weakening / vague qualifiers — rewrite by default when numbers or clear evidence exist
A_LEVEL = [
    # HREL epistemic modality (selection from the aproko lexicon)
    "可能", "或许", "也许", "似乎", "看起来", "恐怕", "有可能", "未必",
    "不好说", "说不清", "据估计", "据说", "据传",
    # HPROP vague qualifiers (selection from the aproko lexicon)
    "某种程度上", "一定程度上", "某种意义上", "基本上是", "差不多", "大体上",
    "大概", "大致上", "多多少少", "或多或少", "部分地", "有所", "有点",
    # Self-added: common weakening phrases in technical documents
    "待验证", "待观察", "待补", "需要进一步", "有望", "相对较", "相对而言",
    "总的来说", "一般而言", "应该说", "可以说", "算是", "算得上", "倾向于",
]
# Level B: neutral frequency words — legitimate when describing real fluctuation/habitual patterns; judge case by case
B_LEVEL = ["通常", "往往", "一般来说", "大概", "大致", "经常", "部分", "初步", "可能存在"]
# English hedges
EN_LEVEL = [
    r"\bmay\b", r"\bmight\b", r"\bpossibly\b", r"\bpresumably\b", r"\bseems?\b",
    r"\bappears? to\b", r"\bsomewhat\b", r"\brelatively\b", r"\broughly\b",
    r"\bapproximately\b", r"\bpotentially\b", r"\blikely to\b", r"\bsuggests? that\b",
    r"\bfuture work\b", r"\bneeds? further\b", r"\bremains? to be\b",
]

PATTERNS = ([(w, "A", w) for w in A_LEVEL]
            + [(w, "B", w) for w in B_LEVEL]
            + [(p, "EN", p) for p in EN_LEVEL])


def scan(path: Path):
    hits = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return hits
    for i, ln in enumerate(lines, 1):
        for pat, lvl, disp in PATTERNS:
            for mo in re.finditer(pat, ln, flags=re.IGNORECASE if lvl == "EN" else 0):
                s = max(0, mo.start() - 25)
                frag = ln[s:mo.end() + 25].strip()
                hits.append((path.name, i, lvl, disp, frag))
    return hits


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "gh_publish"
    all_hits = []
    for f in sorted(root.rglob("*.md")):
        all_hits.extend(scan(f))
    from collections import Counter
    cnt = Counter((lvl, w) for _, _, lvl, w, _ in all_hits)
    print(f"=== {len(all_hits)} hits, word frequency ===")
    for (lvl, w), c in cnt.most_common():
        print(f"  [{lvl}] {w}: {c}")
    print("\n=== Per-hit details ===")
    for fn, ln, lvl, w, frag in all_hits:
        print(f"  {fn}:{ln} [{lvl}] \"{w}\"  ...{frag}...")
