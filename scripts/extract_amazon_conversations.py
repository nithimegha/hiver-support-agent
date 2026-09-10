import pandas as pd
from pathlib import Path

input_file = "data/raw/twcs/twcs.csv"
output_file = "data/processed/amazonhelp_conversations.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# --------------------------------------------------
# Step 1: Find all AmazonHelp tweets
# --------------------------------------------------

amazon_rows = []

print("Step 1: Finding AmazonHelp tweets...")

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    chunksize=100_000,
):
    rows = chunk[chunk["author_id"] == "AmazonHelp"]

    if not rows.empty:
        amazon_rows.append(rows)

amazon_df = pd.concat(amazon_rows, ignore_index=True)

print(f"AmazonHelp tweets found: {len(amazon_df):,}")


# --------------------------------------------------
# Step 2: Get the IDs of customer tweets
# that AmazonHelp replied to
# --------------------------------------------------

customer_tweet_ids = set(
    pd.to_numeric(
        amazon_df["in_response_to_tweet_id"],
        errors="coerce"
    )
    .dropna()
    .astype("int64")
)

print(
    f"Customer tweets directly connected to AmazonHelp: "
    f"{len(customer_tweet_ids):,}"
)


# --------------------------------------------------
# Step 3: Find those customer tweets
# in the original dataset
# --------------------------------------------------

customer_rows = []

print("\nStep 3: Finding customer messages...")

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    chunksize=100_000,
):
    rows = chunk[
        chunk["tweet_id"].isin(customer_tweet_ids)
    ]

    if not rows.empty:
        customer_rows.append(rows)

customer_df = pd.concat(customer_rows, ignore_index=True)


# --------------------------------------------------
# Step 4: Combine customer messages + AmazonHelp replies
# --------------------------------------------------

conversation_df = pd.concat(
    [customer_df, amazon_df],
    ignore_index=True
)

conversation_df = conversation_df.drop_duplicates(
    subset=["tweet_id"]
)

conversation_df = conversation_df.sort_values(
    "tweet_id"
)

conversation_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\nExtraction complete!")

print(f"Customer messages: {len(customer_df):,}")
print(f"AmazonHelp messages: {len(amazon_df):,}")
print(f"Total conversation messages: {len(conversation_df):,}")

print(f"\nSaved to:")
print(output_file)