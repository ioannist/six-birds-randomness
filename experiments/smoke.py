import numpy as np
import randomness_ledger


def main() -> None:
    _ = randomness_ledger
    arr = np.array([1, 2, 3])
    print(f"OK smoke numpy_sum={arr.sum()}")


if __name__ == "__main__":
    main()
