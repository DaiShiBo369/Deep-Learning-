"""Extract the plain-text output of key cells from the executed notebook."""
import json
import re
import sys
from pathlib import Path

HB = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2")
NB = HB / "HW2" / "HW2_PyTorch_Basics.ipynb"
OUT = HB / "work" / "q4"
IDS = ["setup", "data-check", "model-check", "run", "reload-check"]


def text_of(value):
    if isinstance(value, list):
        return "".join(value)
    if isinstance(value, str):
        return value
    return ""


def cell_text(cell):
    parts = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            parts.append(text_of(output.get("text")))
        elif kind in ("execute_result", "display_data"):
            parts.append(text_of(output.get("data", {}).get("text/plain", "")))
        elif kind == "error":
            parts.append("ERROR: " + output.get("ename", "") + ": " + output.get("evalue", ""))
    return "".join(parts)


def parse_run(text, reload_text):
    history = [
        {"epoch": int(e), "loss": float(loss), "accuracy": float(acc) / 100.0}
        for e, loss, acc in re.findall(
            r"Epoch (\d+)/5: train loss=([\d.]+), accuracy=([\d.]+)%", text)
    ]
    initial = re.search(
        r"Full training set: initial loss=([\d.]+), final loss=([\d.]+)", text)
    final_acc = re.search(r"Full training set: final accuracy=([\d.]+)%", text)
    test = re.search(r"Final test set: loss=([\d.]+), accuracy=([\d.]+)%", text)
    digits = re.search(r"Predicted digits: (\[[^\]]*\])", reload_text)
    return {
        "history": history,
        "initial_train_loss": float(initial.group(1)),
        "final_train_loss": float(initial.group(2)),
        "final_train_accuracy": float(final_acc.group(1)) / 100.0,
        "test_loss": float(test.group(1)),
        "test_accuracy": float(test.group(2)) / 100.0,
        "predicted_digits": json.loads(digits.group(1)) if digits else None,
    }


def main(tag):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = {c["id"]: c for c in nb["cells"] if c.get("cell_type") == "code"}
    texts = {}
    for cid in IDS:
        text = cell_text(code_cells[cid])
        if "\x1b" in text:
            raise SystemExit(f"ANSI escape found in cell {cid}")
        (OUT / f"{cid}.txt").write_text(text, encoding="utf-8")
        texts[cid] = text
        print(f"=== {cid} ===")
        print(text)
    runs = [cell_text(code_cells[i]) for i in ("data-check", "run", "reload-check")]
    if any("Traceback" in t for t in runs):
        raise SystemExit("traceback found in output")
    metrics = parse_run(texts["run"], texts["reload-check"])
    if len(metrics["history"]) != 5:
        raise SystemExit(f"expected 5 epochs, got {len(metrics['history'])}")
    if metrics["predicted_digits"] is None:
        raise SystemExit("predicted digits not found")
    (OUT / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run1")
