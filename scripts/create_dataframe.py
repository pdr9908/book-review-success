import os
import argparse
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split


def load_and_concatenate_files(input_dir: str) -> pd.DataFrame:
    """
    Load all JSON files from the specified directory and concatenate
    them into a single DataFrame.

    Args:
        input_dir (str): Directory containing the JSON files to be
        concatenated.

    Returns:
        pd.DataFrame: A single concatenated DataFrame.
    """
    # Find all JSON files in the input directory
    input_files = glob(os.path.join(input_dir, "*.json"))

    # Load each file into a DataFrame and concatenate them
    df_list = [pd.read_json(file, lines=True) for file in input_files]
    concatenated_df = pd.concat(df_list, ignore_index=True)

    return concatenated_df


def split_and_save_dataframe(
    df: pd.DataFrame, output_dir: str, test_size: float = 0.2
) -> None:
    """
    Split the DataFrame into training and testing sets and save them as JSON
    files.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        output_dir (str): Directory where the split DataFrames will be saved.
        test_size (float): Proportion of the dataset to include in the test.

    Returns:
        None
    """
    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the training and testing DataFrames as JSON files
    train_df.to_json(
        os.path.join(output_dir, "train_df.json"), orient="records", lines=True
    )
    test_df.to_json(
        os.path.join(output_dir, "test_df.json"), orient="records", lines=True
    )

    print(f"Training and testing data saved to {output_dir}")


def main():

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Concatenate JSON files and train/test split."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/featured/",
        help="Directory containing the JSON files to be concatenated.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tidy/",
        help="Directory to save the split training and testing sets.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the data to use for the test set.",
    )
    args = parser.parse_args()

    # Load and concatenate the JSON files
    df = load_and_concatenate_files(args.input_dir)

    # Split the DataFrame into training and testing sets and save them
    split_and_save_dataframe(df, args.output_dir, args.test_size)


if __name__ == "__main__":
    main()
