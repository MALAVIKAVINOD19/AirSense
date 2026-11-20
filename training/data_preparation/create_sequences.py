import pandas as pd
import numpy as np
import os

features = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

INPUT_WINDOW = 6

os.makedirs("data/processed/sequences", exist_ok=True)

for split in ["train", "validation", "test"]:

    df = pd.read_csv(f"data/processed/{split}_clean.csv")

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    gap = df["DateTime"].diff() != pd.Timedelta(hours=1)
    df["segment"] = gap.cumsum()

    X = []
    y = []

    for _, group in df.groupby("segment"):

        values = group[features].values

        for i in range(len(values) - INPUT_WINDOW):

            X.append(values[i:i + INPUT_WINDOW])
            y.append(values[i + INPUT_WINDOW])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    np.save(
        f"data/processed/sequences/{split}_X.npy", X
    )

    np.save(
        f"data/processed/sequences/{split}_y.npy", y
    )

    print(f"\n{split.upper()}")
    print("Input shape:", X.shape)
    print("Target shape:", y.shape)
    print("Forecasting examples:", len(X))

print("\nAll forecasting sequences created!")