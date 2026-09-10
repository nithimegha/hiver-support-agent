import pandas as pd


file_path = "data/processed/amazonhelp_support_pairs.csv"

df = pd.read_csv(file_path)

print("Total support pairs:", len(df))

print("\nUnique customer tweets:", df["customer_tweet_id"].nunique())

print("Unique AmazonHelp replies:", df["amazon_tweet_id"].nunique())

print("\nDuplicate customer tweet IDs:")
print(
    df["customer_tweet_id"].duplicated().sum()
)

print("\nMissing customer messages:")
print(
    df["customer_text"].isna().sum()
)

print("\nMissing AmazonHelp responses:")
print(
    df["amazon_response"].isna().sum()
)


# Calculate text lengths
df["customer_length"] = df["customer_text"].fillna("").str.len()
df["response_length"] = df["amazon_response"].fillna("").str.len()

print("\nCustomer message length:")
print(df["customer_length"].describe())

print("\nAmazonHelp response length:")
print(df["response_length"].describe())


print("\nFirst 10 customer → AmazonHelp examples:")

for _, row in df.head(10).iterrows():

    print("\n" + "=" * 80)

    print("CUSTOMER:")
    print(row["customer_text"])

    print("\nAMAZONHELP:")
    print(row["amazon_response"])