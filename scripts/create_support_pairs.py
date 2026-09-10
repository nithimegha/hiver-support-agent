import pandas as pd
from pathlib import Path


input_file = "data/processed/amazonhelp_conversations.csv"
output_file = "data/processed/amazonhelp_support_pairs.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("Loading AmazonHelp conversation data...")

df = pd.read_csv(input_file)

# Create a lookup using tweet_id as the key.
tweet_lookup = df.set_index("tweet_id").to_dict("index")

pairs = []

print("Creating customer → AmazonHelp pairs...")

for _, row in df.iterrows():

    # We only want responses written by AmazonHelp.
    if row["author_id"] != "AmazonHelp":
        continue

    parent_id = row["in_response_to_tweet_id"]

    # Ignore AmazonHelp tweets without a parent tweet.
    if pd.isna(parent_id):
        continue

    try:
        parent_id = int(float(parent_id))
    except (ValueError, TypeError):
        continue

    # Find the customer tweet that AmazonHelp replied to.
    customer = tweet_lookup.get(parent_id)

    if customer is None:
        continue

    # Make sure the parent tweet is from a customer.
    if customer["inbound"] is not True:
        continue

    pairs.append(
        {
            "customer_tweet_id": parent_id,
            "customer_text": customer["text"],
            "customer_created_at": customer["created_at"],
            "amazon_tweet_id": row["tweet_id"],
            "amazon_response": row["text"],
            "amazon_created_at": row["created_at"],
        }
    )


pairs_df = pd.DataFrame(pairs)

pairs_df.to_csv(output_file, index=False)

print("\nSupport pair creation complete!")
print(f"Total customer → AmazonHelp pairs: {len(pairs_df):,}")
print(f"Saved to: {output_file}")

print("\nFirst 5 support pairs:")
print(pairs_df.head(5).to_string(index=False))