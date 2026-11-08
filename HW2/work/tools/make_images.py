# -*- coding: utf-8 -*-
"""Render the captured console output and notebook cell output as screenshot-style PNGs."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2")
WORK = ROOT / "work"
IMG = WORK / "report" / "img"
IMG.mkdir(parents=True, exist_ok=True)

CONSOLA = r"C:\Windows\Fonts\consola.ttf"
CONSOLAB = r"C:\Windows\Fonts\consolab.ttf"

TERM_BG = (30, 30, 30)
TERM_FG = (212, 212, 212)
JUP_BG = (255, 255, 255)
JUP_BORDER = (208, 215, 222)
JUP_CODE = (36, 41, 46)
JUP_PROMPT = (9, 105, 218)
JUP_SEP = (234, 236, 239)


def terminal_image(lines, out_path, font_size=26, line_h=34, pad_x=26, pad_y=22):
    font = ImageFont.truetype(CONSOLA, font_size)
    char_w = font.getlength("M")
    width = int(pad_x * 2 + max(1, max((len(l) for l in lines), default=1)) * char_w)
    height = int(pad_y * 2 + max(1, len(lines)) * line_h)
    img = Image.new("RGB", (width, height), TERM_BG)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad_x, pad_y + i * line_h), line, font=font, fill=TERM_FG)
    img.save(out_path)
    print(f"wrote {out_path.name} {img.size}")


def jupyter_image(cell_source, cell_output, exec_count, out_path,
                  font_size=24, line_h=32, pad=20):
    code_font = ImageFont.truetype(CONSOLA, font_size)
    prompt_font = ImageFont.truetype(CONSOLAB, font_size - 2)
    code = cell_source.rstrip("\n").split("\n")
    out_lines = cell_output.rstrip("\n").split("\n") if cell_output.strip() else []

    prompt = f"In [{exec_count}]:"
    prompt_w = prompt_font.getlength(prompt)
    char_w = code_font.getlength("M")
    max_len = max([len(l) for l in code] + [len(l) for l in out_lines] + [1])
    code_x = pad + 16 + prompt_w + 18
    width = int(code_x + max_len * char_w + pad + 8)

    code_h = len(code) * line_h
    out_h = (16 + len(out_lines) * line_h) if out_lines else 0
    height = int(pad * 2 + code_h + out_h)

    img = Image.new("RGB", (width, height), JUP_BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([1, 1, width - 2, height - 2], radius=10,
                           outline=JUP_BORDER, width=2)

    # input prompt slightly offset down to match first code line baseline
    draw.text((pad + 16, pad + 1), prompt, font=prompt_font, fill=JUP_PROMPT)
    for i, line in enumerate(code):
        draw.text((code_x, pad + i * line_h), line, font=code_font, fill=JUP_CODE)

    if out_lines:
        sep_y = pad + code_h + 8
        draw.line([(code_x, sep_y), (width - pad - 8, sep_y)], fill=JUP_SEP, width=2)
        for i, line in enumerate(out_lines):
            draw.text((code_x, sep_y + 14 + i * line_h), line, font=code_font,
                      fill=JUP_CODE)
    img.save(out_path)
    print(f"wrote {out_path.name} {img.size}")


def read_lines(name):
    text = (WORK / "q13" / name).read_text(encoding="utf-8")
    return text.rstrip("\n").split("\n")


def find(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return i
    raise SystemExit(f"marker not found: {needle}")


def render_q13():
    q1 = read_lines("q1_3.txt")
    terminal_image(q1, IMG / "q1_3.png")

    q2 = read_lines("q2.txt")

    def seg(lines, a, b):
        """1-based inclusive line range."""
        return lines[a - 1:b]

    terminal_image(seg(q2, 1, 29), IMG / "q2_1a.png")
    terminal_image(seg(q2, 31, 57), IMG / "q2_1b.png")
    terminal_image(seg(q2, 59, 78), IMG / "q2_2a.png")
    terminal_image(seg(q2, 80, 93), IMG / "q2_2b.png")

    q3 = read_lines("q3.txt")
    terminal_image(seg(q3, 1, 28), IMG / "q3_a.png")
    terminal_image(seg(q3, 30, 43), IMG / "q3_b.png")
    terminal_image(seg(q3, 45, 62), IMG / "q3_c.png")


def cell_output_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            t = o.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            parts.append(t.rstrip("\n"))
        elif o.get("output_type") in ("execute_result", "display_data"):
            t = o.get("data", {}).get("text/plain", "")
            if isinstance(t, list):
                t = "".join(t)
            parts.append(t.rstrip("\n"))
        elif o.get("output_type") == "error":
            parts.append("ERROR: " + o.get("ename", ""))
    return "\n".join(parts)


def render_q4():
    nb = json.loads((ROOT / "HW2" / "HW2_PyTorch_Basics.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: c for c in nb["cells"] if c.get("cell_type") == "code"}
    plan = [("data-check", "q4_data.png"), ("model-check", "q4_model.png"),
            ("run", "q4_run.png"), ("reload-check", "q4_reload.png")]
    for cid, name in plan:
        cell = cells[cid]
        src = "".join(cell.get("source", []))
        out = cell_output_text(cell)
        jupyter_image(src, out, cell.get("execution_count"), IMG / name)


if __name__ == "__main__":
    render_q13()
    render_q4()
