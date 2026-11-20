from datasets import load_from_disk
import pandas as pd
import os

# Load original dataset
ds = load_from_disk("data/raw/air_quality_dataset")

# Select numerical features
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

os.makedirs("data/processed", exist_ok=True)

for split in ["train", "validation", "test"]:

    df = ds[split].to_pandas()

    # Convert date and time
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S"
    )

    # Select required columns
    df = df[["DateTime"] + features].copy()

    # Keep complete observations
    df = df.dropna(subset=features)

    # Save cleaned dataset
    df.to_csv(
        f"data/processed/{split}_clean.csv",
        index=False
    )

    print(f"\n{split.upper()}")
    print("Clean observations:", len(df))
    print("Saved successfully!")

print("\nAll datasets prepared!")