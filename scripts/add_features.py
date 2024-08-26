import os
import argparse
import pandas as pd
from glob import glob
from typing import List
import sys

sys.path.append(".")

from src.data_preparation.feature_engineering_utils import (  # noqa: E402, E501
    get_df_text_features,
)


def process_and_save_features(input_file: str, output_dir: str) -> None:
    """
    Process a single JSON file by adding features and saving the result.

    Args:
        input_file (str): The path to the input JSON file.
        output_dir (str): The directory where the output file should be saved.

    Returns:
        None
    """
    # Determine the output file path
    file_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, file_name)

    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping...")
        return

    # Load the processed data into a DataFrame
    df = pd.read_json(input_file, lines=True)

    # Apply feature engineering
    df = get_df_text_features(df)

    # Save the enhanced DataFrame to the output directory
    df.to_json(output_file, orient="records", lines=True)
    print(f"Features added and saved to {output_file}")


def main() -> None:
    """
    Main function to handle command-line arguments and process all JSON files
    in the specified input directory by adding features.

    Args:
        None

    Returns:
        None
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Add features to processed data files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/",
        help="Directory containing the processed JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/featured/",
        help="Directory to save the JSON files with added features.",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSON files in the input directory
    input_files: List[str] = glob(os.path.join(args.input_dir, "*.json"))

    # Process each file
    for input_file in input_files:
        process_and_save_features(input_file, args.output_dir)


if __name__ == "__main__":
    main()
