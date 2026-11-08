"""Fill the TODO cells of HW2_PyTorch_Basics.ipynb with the required implementations."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\17857\Desktop\大二\深度学习导论\HW2\HW2")
NB = ROOT / "HW2_PyTorch_Basics.ipynb"

CELLS = {}

CELLS["loaders"] = '''def make_mnist_loaders(root="data"):
    available = prepare_mnist(root=root)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform,
                                   download=not available)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform,
                                  download=not available)
    generator = torch.Generator().manual_seed(2026)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=0, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=0)
    return train_loader, test_loader
'''

CELLS["model"] = '''class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
'''

CELLS["components"] = '''def make_training_components(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return loss_fn, optimizer
'''

CELLS["train"] = '''def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images_batch, labels_batch in loader:
        optimizer.zero_grad()
        logits = model(images_batch)
        loss = loss_fn(logits, labels_batch)
        loss.backward()
        optimizer.step()
        batch_size = labels_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
        total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples
'''

CELLS["evaluate"] = '''def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images_batch, labels_batch in loader:
            logits = model(images_batch)
            loss = loss_fn(logits, labels_batch)
            batch_size = labels_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
            total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples
'''

CELLS["reload"] = '''def save_and_reload(model, path="mnist_mlp.pt"):
    torch.save(model.state_dict(), path)
    restored_model = MLP()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    restored_model.load_state_dict(state_dict)
    restored_model.eval()
    return restored_model
'''

nb = json.loads(NB.read_text(encoding="utf-8"))
updated = []
for cell in nb["cells"]:
    cell_id = cell.get("id")
    if cell_id in CELLS:
        cell["source"] = CELLS[cell_id].splitlines(keepends=True)
        updated.append(cell_id)

missing = sorted(set(CELLS) - set(updated))
if missing:
    raise SystemExit(f"cells not found: {missing}")
if updated != ["loaders", "model", "components", "train", "evaluate", "reload"]:
    raise SystemExit(f"unexpected update order: {updated}")

# keep nbformat style: one-space indentation, unicode preserved, trailing newline
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("updated cells:", updated)
print("todo markers left:", sum("TODO" in line for cell in nb["cells"] if cell.get("cell_type") == "code" for line in cell["source"]))
