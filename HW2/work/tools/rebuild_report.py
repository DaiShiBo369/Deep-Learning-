# -*- coding: utf-8 -*-
"""Widen the first column of the environment table and rebuild."""
import shutil
import subprocess
from pathlib import Path

root = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2")
rep = root / "work" / "report"
build = root / "work" / "build"
src = rep / "HW2_report.tex"

t = src.read_text(encoding="utf-8")
old = r"\begin{tabular}{p{5.2cm}p{10.0cm}}"
new = r"\begin{tabular}{p{6.0cm}p{9.2cm}}"
if old in t:
    t = t.replace(old, new)
    src.write_text(t, encoding="utf-8")
    print("table width updated")
else:
    print("table width already updated or pattern missing")

build.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, build / "HW2_report.tex")
shutil.rmtree(build / "img", ignore_errors=True)
shutil.copytree(rep / "img", build / "img")

xelatex = r"C:\CTEX\MiKTeX\miktex\bin\x64\xelatex.exe"
for i in range(2):
    r = subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error",
                        "HW2_report.tex"], cwd=str(build),
                       capture_output=True, text=True, errors="replace")
    tail = [l for l in r.stdout.splitlines() if "Output written" in l or l.startswith("!")]
    print("run", i + 1, tail[-3:])
print("pdf size:", (build / "HW2_report.pdf").stat().st_size)
