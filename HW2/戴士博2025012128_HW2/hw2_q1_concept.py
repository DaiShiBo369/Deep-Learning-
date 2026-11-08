"""Homework 2 - Question 1.3: is y.grad populated by default, and how to retain it.

The assignment snippet is:
    x = torch.tensor(2.0, requires_grad=True)
    y = 3 * x
    loss = y ** 2
    loss.backward()

Analytical derivatives:
    loss = y ** 2 = (3x) ** 2 = 9 x ** 2
    d loss / d x = 18 x  ->  18 * 2 = 36.0
    d loss / d y = 2 y   ->  2 * 6 = 12.0

Autograd populates .grad only for leaf tensors by default, therefore y.grad is
None although y participates in the graph.  y.retain_grad() must be called before
backward() to keep the gradient of the non-leaf tensor y.
"""

import warnings

import torch

# Reading .grad from a non-leaf tensor emits a UserWarning in recent PyTorch
# releases.  That is exactly the behaviour this question demonstrates, so the
# warning is expected and filtered out to keep the printed result readable.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*The \.grad attribute of a Tensor that is not a leaf Tensor.*",
)


def section(title):
    """Print a readable section header for the report screenshot."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    # ------------------------------------------------------------------
    # Part 1: run the assignment snippet exactly as given.
    # ------------------------------------------------------------------
    section("Part 1: the assignment snippet (only the leaf x keeps .grad)")

    x = torch.tensor(2.0, requires_grad=True)
    y = 3 * x
    loss = y ** 2
    loss.backward()

    print(f"x                = {x.item()}")
    print(f"y = 3 * x        = {y.item()}")
    print(f"loss = y ** 2    = {loss.item()}")
    print(f"x.grad           = {x.grad.item()}    "
          f"(expected 36.0 = d loss / d x = 18 x)")
    # y.grad is None because y is an intermediate (non-leaf) tensor.
    print(f"y.grad           = {y.grad}    "
          f"(expected None: autograd only keeps .grad for leaves)")
    print(f"y.is_leaf        = {y.is_leaf}")

    # ------------------------------------------------------------------
    # Part 2: build a fresh graph and retain the gradient of y.
    # ------------------------------------------------------------------
    section("Part 2: retaining the gradient of y (fresh graph, y2.retain_grad())")

    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = 3 * x2
    y2.retain_grad()          # must run BEFORE backward(), while y2 is still in the graph
    loss2 = y2 ** 2
    loss2.backward()

    print(f"y2 = 3 * x2      = {y2.item()}")
    print(f"loss2 = y2 ** 2  = {loss2.item()}")
    print(f"y2.grad          = {y2.grad.item()}    "
          f"(expected 12.0 = d loss / d y = 2 y)")
    print(f"x2.grad          = {x2.grad.item()}    "
          f"(expected 36.0 = d loss / d x = 18 x)")

    # The first graph's y was never retained, so it is still None.
    print(f"y.grad (first graph, unchanged) = {y.grad}")

    # ------------------------------------------------------------------
    # Loud checks.
    # ------------------------------------------------------------------
    assert x.grad is not None and float(x.grad) == 36.0, \
        f"x.grad should be 36.0, got {x.grad}"
    assert y.grad is None, f"y.grad should be None by default, got {y.grad}"
    assert y.is_leaf is False, f"y should not be a leaf tensor, got {y.is_leaf}"
    assert y2.grad is not None and float(y2.grad) == 12.0, \
        f"y2.grad should be 12.0, got {y2.grad}"
    assert x2.grad is not None and float(x2.grad) == 36.0, \
        f"x2.grad should be 36.0, got {x2.grad}"
    assert y.grad is None, "the first graph's y.grad must stay None"

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
