"""Prepare the local MNIST files for torchvision.datasets.MNIST."""
from pathlib import Path
import argparse
import gzip
import shutil

FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

def prepare_mnist(source=None, root="data"):
    base = Path(__file__).resolve().parent
    candidates = [base / "mnist_data", base.parent / "HWX2" / "mnist_data",
                  base.parent / "HW1" / "mnist_data"]
    if source is not None:
        candidates = [Path(source)]
    folder = next((p for p in candidates if all((p / (f + ".gz")).is_file()
                   for f in FILES)), None)
    raw = Path(root) / "MNIST" / "raw"
    if all((raw / f).is_file() for f in FILES):
        return True
    if folder is None:
        return False
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        with gzip.open(folder / (name + ".gz"), "rb") as src:
            with (raw / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    ok = prepare_mnist(args.source, args.root)
    print("Local MNIST ready." if ok else "No local source found; use download=True.")
