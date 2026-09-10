import pandas as pd
from pathlib import Path


# Input and output files
input_file = "data/raw/twcs/twcs.csv"
output_file = "data/processed/amazonhelp.csv"

# Make sure the output folder exists
Path("data/processed").mkdir(parents=True, exist_ok=True)

# We only need these columns
columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# Store AmazonHelp rows here
amazon_chunks = []

print("Reading dataset...")

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    chunksize=100_000,
):
    # Keep AmazonHelp's messages
    amazon_rows = chunk[chunk["author_id"] == "AmazonHelp"]

    if not amazon_rows.empty:
        amazon_chunks.append(amazon_rows)

    print(f"Processed {len(chunk):,} rows...")

# Combine all AmazonHelp rows
amazon_df = pd.concat(amazon_chunks, ignore_index=True)

# Save the extracted dataset
amazon_df.to_csv(output_file, index=False)

print("\nExtraction complete!")
print(f"AmazonHelp rows: {len(amazon_df):,}")
print(f"Saved to: {output_file}")