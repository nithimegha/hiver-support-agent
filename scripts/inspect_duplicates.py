import pandas as pd


file_path = "data/processed/amazonhelp_support_pairs.csv"

df = pd.read_csv(file_path)

# Find customer tweets that have multiple AmazonHelp responses.
duplicate_counts = (
    df["customer_tweet_id"]
    .value_counts()
)

duplicate_ids = duplicate_counts[
    duplicate_counts > 1
].head(10).index


print("Number of customer tweets with multiple responses:")
print((duplicate_counts > 1).sum())

print("\nSample duplicate cases:")

for customer_id in duplicate_ids:

    rows = df[df["customer_tweet_id"] == customer_id]

    print("\n" + "=" * 80)

    print("CUSTOMER:")
    print(rows.iloc[0]["customer_text"])

    print("\nAMAZONHELP RESPONSES:")

    for _, row in rows.iterrows():
        print("-", row["amazon_response"])