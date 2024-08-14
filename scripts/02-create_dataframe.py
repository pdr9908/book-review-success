import pandas as pd
import glob


def main():
    files = glob.glob("data/processed_data/*.json")
    print(files)
    dfs = []
    for file in files:
        df = pd.read_json(file, orient="records", lines=True)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_json("data/full_dataframe.json", orient="records")


if __name__ == "__main__":
    main()
