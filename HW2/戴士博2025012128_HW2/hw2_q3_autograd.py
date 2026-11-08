"""Homework 2 - Question 3: autograd, gradient accumulation and reset.

Let X be a float32 random tensor of shape (3, 3) with requires_grad=True and
    M = [[1, 0, 2],
         [1, 1, 0],
         [0, 3, 1]]
    L(X) = sum_ij X_ij * M_ij      (the multiplication is elementwise).

Analytical derivative: d L / d X = M, because each X_ij appears linearly with
coefficient M_ij.  Therefore:
    (a) after one backward pass, X.grad   == M
    (b) after a second backward pass on a freshly built graph without clearing,
        the gradients accumulate: X.grad     == 2 * M
    (c) after X.grad = None and one more backward pass, X.grad == M again
"""

import torch


def section(title):
    """Print a readable section header for the report screenshot."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    torch.manual_seed(2026)
    torch.set_printoptions(precision=4, sci_mode=False)

    # ------------------------------------------------------------------
    # Setup.
    # ------------------------------------------------------------------
    section("Question 3: setup (seed 2026)")

    X = torch.rand(3, 3, dtype=torch.float32, requires_grad=True)
    M = torch.tensor([[1.0, 0.0, 2.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 3.0, 1.0]], dtype=torch.float32)

    print("X (leaf tensor, requires_grad=True) =")
    print(X)
    print("\nM (constant coefficient matrix) =")
    print(M)
    print("\nL(X) = sum_ij X_ij * M_ij, so the analytical gradient dL/dX is M.")
    print(f"X.requires_grad = {X.requires_grad}, X.is_leaf = {X.is_leaf}")

    # ------------------------------------------------------------------
    # (a) One backward pass.
    # ------------------------------------------------------------------
    section("(a) First backward pass: L = (X * M).sum()")

    L = (X * M).sum()
    print(f"L = sum(X * M) = {L.item():.6f}")
    L.backward()
    grad_a = X.grad.detach().clone()
    print("\nX.grad after one backward (expected M) =")
    print(X.grad)
    print(f"\nX.grad matches M (torch.allclose): {torch.allclose(X.grad, M)}")
    assert torch.allclose(X.grad, M), f"(a) X.grad should equal M, got {X.grad}"

    # ------------------------------------------------------------------
    # (b) Second backward pass without clearing gradients.
    # ------------------------------------------------------------------
    section("(b) Second backward pass without clearing: gradients accumulate")

    print("The graph of L was freed by the first backward(), so build a new")
    print("forward graph with the same leaf X.  X.grad is NOT cleared here.")
    L2 = (X * M).sum()
    print(f"\nL2 = sum(X * M) = {L2.item():.6f}   (same value as L)")
    L2.backward()
    grad_b = X.grad.detach().clone()
    print("\nX.grad after the second backward (expected 2 * M) =")
    print(X.grad)
    print(f"\nX.grad matches 2 * M (torch.allclose): "
          f"{torch.allclose(X.grad, 2 * M)}")
    assert torch.allclose(X.grad, 2.0 * M), \
        f"(b) X.grad should equal 2 * M, got {X.grad}"

    # ------------------------------------------------------------------
    # (c) Reset and one more backward pass.
    # ------------------------------------------------------------------
    section("(c) Reset X.grad = None, then backward once more")

    X.grad = None
    L3 = (X * M).sum()
    print(f"X.grad was set to None; L3 = sum(X * M) = {L3.item():.6f}")
    L3.backward()
    grad_c = X.grad.detach().clone()
    print("\nX.grad after resetting and one backward (expected M again) =")
    print(X.grad)
    print(f"\nX.grad matches M (torch.allclose): {torch.allclose(X.grad, M)}")

    # ------------------------------------------------------------------
    # Loud checks.
    # ------------------------------------------------------------------
    assert torch.allclose(grad_a, M), f"(a) stored grad should equal M, got {grad_a}"
    assert torch.allclose(grad_b, 2.0 * M), \
        f"(b) stored grad should equal 2 * M, got {grad_b}"
    assert torch.allclose(grad_c, M), f"(c) stored grad should equal M, got {grad_c}"

    print()
    print("Summary of the three gradient results (each should be exact):")
    print("  (a) X.grad = M      =", grad_a.tolist())
    print("  (b) X.grad = 2 * M  =", grad_b.tolist())
    print("  (c) X.grad = M      =", grad_c.tolist())

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
