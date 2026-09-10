import pandas as pd
from pathlib import Path


input_file = "data/processed/amazonhelp_support_pairs.csv"
output_file = "data/processed/amazonhelp_resolution_cases.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

print("Loading support pairs...")

df = pd.read_csv(input_file)

# Convert timestamp to datetime so responses can be ordered.
df["amazon_created_at"] = pd.to_datetime(
    df["amazon_created_at"],
    errors="coerce"
)

# Sort responses for each customer message by time.
df = df.sort_values(
    ["customer_tweet_id", "amazon_created_at"]
)

cases = []

print("Grouping AmazonHelp responses...")

for customer_id, group in df.groupby("customer_tweet_id"):

    customer_text = group.iloc[0]["customer_text"]

    responses = group["amazon_response"].dropna().tolist()

    if not responses:
        continue

    combined_response = "\n".join(
        f"{i}. {response}"
        for i, response in enumerate(responses, start=1)
    )

    cases.append(
        {
            "customer_tweet_id": customer_id,
            "customer_text": customer_text,
            "amazon_response_count": len(responses),
            "historical_resolution": combined_response,
        }
    )


cases_df = pd.DataFrame(cases)

cases_df.to_csv(
    output_file,
    index=False
)

print("\nResolution case creation complete!")

print(f"Original support pairs: {len(df):,}")
print(f"Unique customer cases: {len(cases_df):,}")

print(
    "\nCases with multiple AmazonHelp responses:"
)
print(
    (cases_df["amazon_response_count"] > 1).sum()
)

print("\nSaved to:")
print(output_file)

print("\nFirst 5 resolution cases:")

for _, row in cases_df.head(5).iterrows():

    print("\n" + "=" * 80)

    print("CUSTOMER:")
    print(row["customer_text"])

    print(
        f"\nAMAZONHELP RESPONSES "
        f"({row['amazon_response_count']}):"
    )

    print(row["historical_resolution"])