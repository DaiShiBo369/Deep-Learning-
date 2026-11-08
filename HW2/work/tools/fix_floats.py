# -*- coding: utf-8 -*-
"""Use loose float placement for Q1-Q3 screenshots, keep [H] for Q4 screenshots."""
import re
from pathlib import Path

p = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\work\report\HW2_report.tex")
text = p.read_text(encoding="utf-8")


def fix(match):
    block = match.group(0)
    if "fig:q4" in block:
        return block.replace("[H]", "[H]", 1)
    return block.replace("[H]", "[htbp]", 1)


new = re.sub(r"\\begin\{figure\}\[[^\]]*\]", lambda m: m.group(0), text)
new = re.sub(r"\\begin\{figure\}\[H\](?:(?!\\end\{figure\}).)*?\\end\{figure\}",
             fix, text, flags=re.S)
p.write_text(new, encoding="utf-8")
print("htbp figures:", new.count("[htbp]"), "H figures:", new.count("[H]"))
