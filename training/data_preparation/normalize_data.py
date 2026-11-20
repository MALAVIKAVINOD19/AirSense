import numpy as np
import os

# Load training data
X_train = np.load("data/processed/sequences/train_X.npy")
y_train = np.load("data/processed/sequences/train_y.npy")

# Calculate statistics using training data only
combined_train = np.concatenate(
    [X_train.reshape(-1, 12), y_train],
    axis=0
)

mean = combined_train.mean(axis=0)
std = combined_train.std(axis=0)

# Avoid division by zero
std[std == 0] = 1.0

os.makedirs("data/processed/normalized", exist_ok=True)

# Save normalization values
np.save("data/processed/normalized/mean.npy", mean)
np.save("data/processed/normalized/std.npy", std)

# Normalize all three datasets
for split in ["train", "validation", "test"]:

    X = np.load(f"data/processed/sequences/{split}_X.npy")
    y = np.load(f"data/processed/sequences/{split}_y.npy")

    X = (X - mean) / std
    y = (y - mean) / std

    np.save(
        f"data/processed/normalized/{split}_X.npy", X
    )

    np.save(
        f"data/processed/normalized/{split}_y.npy", y
    )

    print(f"\n{split.upper()}")
    print("Normalized input:", X.shape)
    print("Normalized target:", y.shape)

print("\nNormalization completed!")