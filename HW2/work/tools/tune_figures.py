# -*- coding: utf-8 -*-
"""Restore [H] placement for Q1-Q3 screenshots and tune their widths."""
import re
from pathlib import Path

p = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\work\report\HW2_report.tex")
text = p.read_text(encoding="utf-8")

widths = {
    "q1_3.png": "0.85",
    "q2_1a.png": "0.68",
    "q2_1b.png": "0.68",
    "q2_2a.png": "0.85",
    "q2_2b.png": "0.85",
    "q3_a.png": "0.75",
    "q3_b.png": "0.80",
    "q3_c.png": "0.80",
}


def fix_block(match):
    block = match.group(0)
    m = re.search(r"\{(q[0-9a-z_]+\.png)\}", block)
    if not m:
        return block
    name = m.group(1)
    if name not in widths:
        return block
    block = block.replace("[htbp]", "[H]", 1)
    idx = block.find(r"\includegraphics[")
    if idx != -1:
        end = block.find("]", idx)
        block = (block[:idx] + r"\includegraphics[width=" + widths[name]
                 + r"\linewidth]" + block[end + 1:])
    return block


new = re.sub(r"\\begin\{figure\}\[[^\]]*\](?:(?!\\end\{figure\}).)*?\\end\{figure\}",
             fix_block, text, flags=re.S)
p.write_text(new, encoding="utf-8")
print("htbp left:", new.count("[htbp]"))
print("figure [H]:", new.count(r"\begin{figure}[H]"))
