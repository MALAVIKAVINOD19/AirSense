import numpy as np

for split in ["train", "validation", "test"]:

    X = np.load(
        f"data/processed/normalized/{split}_X.npy"
    )

    y = np.load(
        f"data/processed/normalized/{split}_y.npy"
    )

    print(f"\n{split.upper()}")

    print("Input contains NaN:", np.isnan(X).any())
    print("Target contains NaN:", np.isnan(y).any())

    print("Input contains infinity:", np.isinf(X).any())
    print("Target contains infinity:", np.isinf(y).any())

    print("Input shape:", X.shape)
    print("Target shape:", y.shape)

print("\nData verification completed!")