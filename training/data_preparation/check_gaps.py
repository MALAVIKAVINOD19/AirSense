import pandas as pd

for split in ["train", "validation", "test"]:

    df = pd.read_csv(f"data/processed/{split}_clean.csv")

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    gaps = df["DateTime"].diff()

    consecutive = (gaps == pd.Timedelta(hours=1)).sum()
    missing_hours = (gaps > pd.Timedelta(hours=1)).sum()

    print(f"\n{split.upper()}")
    print("Total observations:", len(df))
    print("Consecutive hourly pairs:", consecutive)
    print("Gaps between observations:", missing_hours)