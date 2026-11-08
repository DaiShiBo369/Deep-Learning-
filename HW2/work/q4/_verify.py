"""Independent verification: replay the notebook definitions in a plain Python
process, get exact metrics, verify the checkpoint, and write results.json."""
import json
import os
import sys
from pathlib import Path

HB = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2")
NB = HB / "HW2" / "HW2_PyTorch_Basics.ipynb"
HW2 = HB / "HW2"
OUT = HB / "work" / "q4"

os.chdir(HW2)
sys.path.insert(0, str(HW2))

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = {c["id"]: c for c in nb["cells"] if c.get("cell_type") == "code"}

# Definitions only; the runner and the reload check are replayed manually below
# so that the notebook-produced checkpoint is left untouched.
DEFINITIONS = ["setup", "loaders", "data-check", "model", "model-check",
               "components", "train", "evaluate", "reload"]
namespace = {"__name__": "__main__"}
for cid in DEFINITIONS:
    exec(compile("".join(cells[cid]["source"]), cid, "exec"), namespace)

import torch  # noqa: E402  (already imported by the setup cell)

# ---- replay the provided runner cell, capturing exact (unrounded) metrics ----
torch.manual_seed(2026)
model = namespace["MLP"]()
train_loader = namespace["train_loader"]
train_loader.generator.manual_seed(2026)
loss_fn, optimizer = namespace["make_training_components"](model)
train_eval_loader = namespace["DataLoader"](train_loader.dataset, batch_size=64,
                                            shuffle=False, num_workers=0)
initial_train_loss, initial_train_accuracy = namespace["evaluate"](
    model, train_eval_loader, loss_fn)
history = []
for epoch in range(5):
    train_loss, train_accuracy = namespace["train_one_epoch"](
        model, train_loader, loss_fn, optimizer)
    history.append({"epoch": epoch + 1, "loss": train_loss, "accuracy": train_accuracy})
    print(f"Epoch {epoch + 1}/5: train loss={train_loss:.6f}, accuracy={train_accuracy:.2%}")
final_train_loss, final_train_accuracy = namespace["evaluate"](
    model, train_eval_loader, loss_fn)
test_loss, test_accuracy = namespace["evaluate"](
    model, namespace["test_loader"], loss_fn)

# ---- checkpoint check: reload the notebook-written file, compare logits ----
images = namespace["images"]
labels = namespace["labels"]
ckpt = HW2 / "mnist_mlp.pt"
assert ckpt.is_file(), "checkpoint missing"
ckpt_size = ckpt.stat().st_size
state = torch.load(ckpt, map_location="cpu", weights_only=True)
fresh = namespace["MLP"]()
fresh.load_state_dict(state)
fresh.eval()
probe = images[:16].clone()
model.eval()
with torch.no_grad():
    expected = model(probe)
    checkpoint_logits = fresh(probe)
torch.testing.assert_close(checkpoint_logits, expected)
max_abs_diff = (checkpoint_logits - expected).abs().max().item()
checkpoint_digits = checkpoint_logits.argmax(dim=1).tolist()

# ---- check save_and_reload() itself without touching mnist_mlp.pt ----
tmp_ckpt = OUT / "_tmp_mlp.pt"
restored = namespace["save_and_reload"](model, path=str(tmp_ckpt))
assert restored is not model and not restored.training
with torch.no_grad():
    restored_logits = restored(probe)
torch.testing.assert_close(restored_logits, expected)
save_reload_ok = True
tmp_ckpt.unlink()
assert not tmp_ckpt.exists()

# ---- cross-check exact values against the notebook's printed output ----
printed = json.loads((OUT / "metrics_run2.json").read_text(encoding="utf-8"))
assert len(printed["history"]) == 5
for exact, shown in zip(history, printed["history"]):
    assert exact["epoch"] == shown["epoch"]
    assert abs(exact["loss"] - shown["loss"]) < 5.1e-7, (exact, shown)
    assert abs(exact["accuracy"] - shown["accuracy"]) < 5.1e-5, (exact, shown)
assert abs(initial_train_loss - printed["initial_train_loss"]) < 5.1e-7
assert abs(final_train_loss - printed["final_train_loss"]) < 5.1e-7
assert abs(final_train_accuracy - printed["final_train_accuracy"]) < 5.1e-5
assert abs(test_loss - printed["test_loss"]) < 5.1e-7
assert abs(test_accuracy - printed["test_accuracy"]) < 5.1e-5
assert checkpoint_digits == printed["predicted_digits"], (
    checkpoint_digits, printed["predicted_digits"])

# ---- notebook JSON sanity checks ----
code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
assert all(c.get("execution_count") is not None for c in code_cells), "unexecuted code cell"
assert len(history) == 5
for cell in code_cells:
    for output in cell.get("outputs", []):
        assert output.get("output_type") != "error", f"error output in {cell['id']}"
        if output.get("output_type") == "stream" and output.get("name") == "stderr":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            assert "Traceback" not in text, f"traceback in {cell['id']}"

results = {
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "torchvision_version": namespace["torchvision"].__version__,
    "device": "cpu",
    "per_epoch": [{"epoch": h["epoch"], "loss": h["loss"], "accuracy": h["accuracy"]}
                  for h in history],
    "initial_train_loss": initial_train_loss,
    "initial_train_accuracy": initial_train_accuracy,
    "final_train_loss": final_train_loss,
    "final_train_accuracy": final_train_accuracy,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "predicted_digits": checkpoint_digits,
    "batch": {
        "images_shape": list(images.shape),
        "images_dtype": str(images.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "image_min": images.min().item(),
        "image_max": images.max().item(),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
    },
    "dataset_sizes": {
        "train": len(train_loader.dataset),
        "test": len(namespace["test_loader"].dataset),
    },
    "checkpoint": {
        "path": "HW2/mnist_mlp.pt",
        "size_bytes": ckpt_size,
        "reload_ok": True,
        "save_and_reload_ok": save_reload_ok,
        "logits_match_trained_model_on_images16": True,
        "max_abs_logit_diff": max_abs_diff,
    },
    "asserts_passed": True,
}
(OUT / "results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
