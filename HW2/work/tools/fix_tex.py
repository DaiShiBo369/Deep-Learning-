# -*- coding: utf-8 -*-
"""Apply the remaining text fixes to the report source (file may be shared by an editor)."""
from pathlib import Path

p = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\work\report\HW2_report.tex")
t = p.read_text(encoding="utf-8")

print("A1 restored:", "合法标签范围是 $[0,C-1]$" in t)
print("A2 restored:", "控制“要不要记录梯度”" in t)

old4 = "预测错得越离谱梯度越大(3) softmax+交叉熵"
new4 = "预测错得越离谱梯度越大；(3) softmax+交叉熵"
print("A3 needs fix:", old4 in t)
t = t.replace(old4, new4)

old5 = "只反序列化张量等安全类型\n"
new5 = "只反序列化张量等安全类型，避免加载任意 Python 对象，安全性更好。\n"
print("A4 needs fix:", old5 in t)
t = t.replace(old5, new5)

p.write_text(t, encoding="utf-8")
print("written, size", p.stat().st_size)
