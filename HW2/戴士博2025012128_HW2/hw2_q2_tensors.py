"""Homework 2 - Question 2: tensor arithmetic, shape and indexing.

2.1 Create float32 random tensors A and B of shape (5, 5), create C with
    torch.zeros_like(A), then compute A + B, A - B, the elementwise product
    (A * B) and the matrix product (A @ B).  Everything is done in PyTorch.
2.2 Create T = torch.arange(24).reshape(6, 4), flatten it to (24,), and extract
    rows 1..4 and columns 1..2 with zero-based indexing, i.e. T[1:5, 1:3].
    Also show T.reshape(4, 6) and T.permute(1, 0) to illustrate the difference
    between reshape and permute.
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
    # 2.1 Tensor arithmetic
    # ------------------------------------------------------------------
    section("Question 2.1: tensor arithmetic (float32, shape (5, 5))")

    A = torch.rand(5, 5, dtype=torch.float32)
    B = torch.rand(5, 5, dtype=torch.float32)
    C = torch.zeros_like(A)

    print(f"A: shape={tuple(A.shape)}, dtype={A.dtype}")
    print(f"B: shape={tuple(B.shape)}, dtype={B.dtype}")
    print(f"C = torch.zeros_like(A): shape={tuple(C.shape)}, dtype={C.dtype}")
    print(f"C contains only zeros: {bool(torch.count_nonzero(C) == 0)}")

    print("\nA =")
    print(A)
    print("\nB =")
    print(B)
    print("\nC = torch.zeros_like(A) =")
    print(C)

    print("\nA + B (elementwise addition) =")
    print(A + B)

    print("\nA - B (elementwise subtraction) =")
    print(A - B)

    print("\nELEMENTWISE product, A * B (NOT matrix multiplication) =")
    print(A * B)

    print("\nMATRIX product, A @ B (matrix multiplication) =")
    print(A @ B)

    assert tuple(A.shape) == (5, 5) and tuple(B.shape) == (5, 5)
    assert tuple(C.shape) == (5, 5) and C.dtype == torch.float32
    assert bool(torch.count_nonzero(C) == 0)
    assert tuple((A * B).shape) == (5, 5) and tuple((A @ B).shape) == (5, 5)
    assert torch.equal(A * B, torch.mul(A, B)), "A * B must be elementwise"
    assert torch.allclose(A @ B, torch.matmul(A, B)), "A @ B must be the matrix product"

    # ------------------------------------------------------------------
    # 2.2 Shape and indexing
    # ------------------------------------------------------------------
    section("Question 2.2: shape and indexing")

    T = torch.arange(24).reshape(6, 4)
    print(f"T = torch.arange(24).reshape(6, 4), shape={tuple(T.shape)}")
    print(T)

    flat = T.reshape(-1)                       # T.flatten() is equivalent here
    print(f"\nFlattened T: T.reshape(-1), shape={tuple(flat.shape)}")
    print(flat)

    central = T[1:5, 1:3]                      # zero-based rows 1..4, columns 1..2
    print(f"\nCentral slice T[1:5, 1:3] (rows 1..4, columns 1..2), "
          f"shape={tuple(central.shape)}")
    print(central)

    print("\nFor comparison (same values, different index order):")
    print(f"  T.reshape(4, 6) reinterprets the row-major memory order, "
          f"shape={tuple(T.reshape(4, 6).shape)}:")
    print(T.reshape(4, 6))
    print(f"\n  T.permute(1, 0) swaps the two axes without changing the values, "
          f"shape={tuple(T.permute(1, 0).shape)}:")
    print(T.permute(1, 0))

    # ------------------------------------------------------------------
    # Loud checks.
    # ------------------------------------------------------------------
    assert tuple(flat.shape) == (24,), f"flattened shape should be (24,), got {flat.shape}"
    assert tuple(central.shape) == (4, 2), \
        f"central slice shape should be (4, 2), got {central.shape}"
    assert torch.equal(central, torch.tensor([[5, 6], [9, 10], [13, 14], [17, 18]])), \
        f"unexpected central slice values: {central}"
    assert torch.equal(T.permute(1, 0), T.T), "permute(1, 0) must equal the transpose"

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
