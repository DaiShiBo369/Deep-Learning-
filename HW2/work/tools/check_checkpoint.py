# -*- coding: utf-8 -*-
"""Independent QA: reload the checkpoint and verify test metrics + predictions."""
import json
import os
import sys
from pathlib import Path

HW2 = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\HW2")
os.chdir(HW2)
sys.path.insert(0, str(HW2))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

torch.manual_seed(2026)
torch.set_num_threads(2)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])
train_set = datasets.MNIST(root="data", train=True, transform=transform, download=False)
test_set = datasets.MNIST(root="data", train=False, transform=transform, download=False)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(2026))
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
images, labels = next(iter(train_loader))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x))))


state = torch.load(HW2 / "mnist_mlp.pt", map_location="cpu", weights_only=True)
model = MLP()
model.load_state_dict(state)
model.eval()

probe = images[:16].clone()
with torch.no_grad():
    logits = model(probe)
digits = logits.argmax(dim=1).tolist()

loss_fn = nn.CrossEntropyLoss(reduction="sum")
total_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)
        total_loss += loss_fn(out, yb).item()
        correct += (out.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

result = {
    "checkpoint": str(HW2 / "mnist_mlp.pt"),
    "size_bytes": (HW2 / "mnist_mlp.pt").stat().st_size,
    "test_loss": total_loss / total,
    "test_accuracy": correct / total,
    "test_examples": total,
    "predicted_digits": digits,
    "expected_digits": [6, 2, 9, 1, 4, 2, 1, 0, 9, 7, 7, 8, 3, 9, 2, 2],
    "digits_match": digits == [6, 2, 9, 1, 4, 2, 1, 0, 9, 7, 7, 8, 3, 9, 2, 2],
    "torch_version": torch.__version__,
}
out_path = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\work\q4\checkpoint_check.json")
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
