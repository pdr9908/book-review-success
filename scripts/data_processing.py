import argparse
import sys

sys.path.append(".")

from src.script_utils.data_processing_utils import process_chunks  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="data/raw/goodreads_reviews_dedup.json.gz",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=100000,
        help="Number of lines per chunk.",
    )
    parser.add_argument(
        "-s",
        "--save_interval",
        type=int,
        default=5,
        help="Number of chunks before saving.",
    )
    parser.add_argument(
        "-o",
        "--output_prefix",
        type=str,
        default="data/processing/output_",
        help="Prefix for the output files.",
    )
    parser.add_argument(
        "-st",
        "--state_file",
        type=str,
        default="data/processing/processing_state.txt",
        help="",
    )

    args = parser.parse_args()
    # Parameters

    desired_columns = [
        "user_id",
        "book_id",
        "rating",
        "review_text",
        "date_updated",
        "n_votes",
    ]
    process_chunks(args, desired_columns)


if __name__ == "__main__":
    main()
