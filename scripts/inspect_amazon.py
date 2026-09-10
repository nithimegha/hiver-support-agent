import pandas as pd

file_path = "data/processed/amazonhelp.csv"

# Read only 2,000 rows for inspection
df = pd.read_csv(file_path, nrows=2000)

print("Shape of sample:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nInbound vs outbound:")
print(df["inbound"].value_counts())

print("\nSample customer messages:")
customer_messages = df[df["inbound"] == True]

for _, row in customer_messages.head(20).iterrows():
    print("-" * 80)
    print("Tweet ID:", row["tweet_id"])
    print("Customer:", row["author_id"])
    print("Text:", row["text"])
    print("Response tweet ID:", row["response_tweet_id"])
    print("In response to:", row["in_response_to_tweet_id"])

print("\nSample AmazonHelp responses:")
amazon_messages = df[df["inbound"] == False]

for _, row in amazon_messages.head(20).iterrows():
    print("-" * 80)
    print("Tweet ID:", row["tweet_id"])
    print("Text:", row["text"])
    print("In response to:", row["in_response_to_tweet_id"])