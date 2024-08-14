import pandas as pd
from langdetect import detect
from langdetect.detector import LangDetectException
import argparse
import os


def detect_lang(text: str) -> str:
    """Return the language of `text`."""
    try:
        return detect(text)
    except LangDetectException:
        return "other"


def process_chunk(chunk: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return the processed pandas dataframe `chunk`.

    The processed chunk contains only the columns in `cols`, only texts
    that are written in English, and the date_updated column in formatted to
    be in datetime format.
    """
    chunk = chunk[cols].copy()
    chunk["lang"] = chunk["review_text"].apply(lambda x: detect_lang(x))
    chunk = chunk[chunk.lang == "en"]
    chunk["date_updated"] = pd.to_datetime(
        chunk.date_updated, format="%a %b %d %X %z %Y", utc=True
    )

    return chunk


def get_last_processed_chunk(state_file: str) -> int:
    """Return the content of `state_file` which contains the last processed
    chunk index."""
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return int(f.read().strip())
    return -1  # No chunks processed yet


def update_processed_chunk(state_file: str, chunk_index: int) -> None:
    """Store `chunk_index`, the last processed chunk index, in `state_file`."""
    with open(state_file, "w") as f:
        f.write(str(chunk_index))


def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="data/raw_data/goodreads_reviews_dedup.json.gz",
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
        default="data/processed_data/output_",
        help="Prefix for the output files.",
    )
    args = parser.parse_args()
    # Parameters

    state_file = "data/processing_state.txt"

    # Initialize counters
    save_counter = 0

    # Get the last processed chunk
    last_processed_chunk = get_last_processed_chunk(state_file)
    chunk_counter = last_processed_chunk + 1
    save_counter = chunk_counter // args.save_interval
    desired_columns = [
        "user_id",
        "book_id",
        "rating",
        "review_text",
        "date_updated",
        "n_votes",
    ]

    dfs = []
    for chunk_idx, chunk in enumerate(
        pd.read_json(args.file, lines=True, chunksize=args.chunksize)
    ):
        # skip all chunks already processed
        if chunk_idx <= last_processed_chunk:
            continue

        print(f"Processing Chunk {chunk_idx}")

        # process the current chunk
        chunk = process_chunk(chunk, desired_columns)
        dfs.append(chunk)

        # save dataframe for each interval
        if chunk_idx % args.save_interval == 0:
            save_counter += 1
            df = pd.concat(dfs)
            df.to_json(
                f"{args.output_prefix}{save_counter}.json",
                orient="records",
                lines=True,
            )
            dfs = []  # clear list to free up memory
            update_processed_chunk(state_file, chunk_idx)

    # Save the final processed data
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_json(
            f"{args.output_prefix}{save_counter + 1}.json",
            orient="records",
            lines=True,
        )

    print("Processing and saving complete.")


if __name__ == "__main__":
    main()
