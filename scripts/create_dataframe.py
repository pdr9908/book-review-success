import pandas as pd
import glob


def main():
    files = glob.glob("data/processing/output*.json")
    file_nums = [int(file.split("_")[-1].split(".json")[0]) for file in files]
    file_nums.sort()

    dfs = []
    for i, file in enumerate(files):
        print(i)
        df = pd.read_json(file, orient="records", lines=True)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_json(
        f"data/processed/full_dataframe_{file_nums[0]}-{file_nums[-1]}.json",
        orient="records",
    )


if __name__ == "__main__":
    main()
