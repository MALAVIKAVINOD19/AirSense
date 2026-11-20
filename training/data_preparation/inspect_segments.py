import pandas as pd

for split in ["train", "validation", "test"]:

    df = pd.read_csv(f"data/processed/{split}_clean.csv")

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    gap = df["DateTime"].diff() != pd.Timedelta(hours=1)
    df["segment"] = gap.cumsum()

    lengths = df.groupby("segment").size().sort_values(
        ascending=False
    )

    print(f"\n{split.upper()}")
    print("Number of segments:", len(lengths))
    print("Longest segment:", lengths.max())
    print("Top 10 segment lengths:")
    print(lengths.head(10).to_list())